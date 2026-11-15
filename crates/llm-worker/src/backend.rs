//! llama.cpp-backed [`LlmPipeline`](super::LlmPipeline) (GGUF via llama-cpp-4).

use super::{ChatRequest, ChatResponse, LlmError, LlmOptions, LlmPipeline, Role};
use llama_cpp_4::{
    AddBos, LlamaBackend, LlamaBatch, LlamaContextParams, LlamaModel, LlamaModelParams,
    LlamaSampler, Special,
};
use llama_cpp_4::model::LlamaChatMessage;
use std::num::NonZeroU32;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// GGUF model session. Owns the llama backend guard, so it is `Send + Sync`.
/// A fresh context is built per request (cheap next to decode; avoids
/// self-referential struct). Sync only, tokio-free.
pub struct LlamaCppLlm {
    backend: LlamaBackend,
    model: LlamaModel,
    opts: LlmOptions,
}

impl std::fmt::Debug for LlamaCppLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppLlm").finish_non_exhaustive()
    }
}

impl LlamaCppLlm {
    /// Load a GGUF file. Checks the path first so a missing file never
    /// initializes the process-global backend.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::Load`] when the file is missing or llama.cpp
    /// refuses it.
    pub fn load(model_path: &Path, opts: &LlmOptions) -> Result<Self, LlmError> {
        if !model_path.is_file() {
            return Err(LlmError::Load(format!(
                "model file not found: {}",
                model_path.display()
            )));
        }
        let mut backend = LlamaBackend::init()
            .map_err(|e| LlmError::Load(format!("backend init failed: {e:?}")))?;
        backend.void_logs();
        let params = LlamaModelParams::default().with_n_gpu_layers(opts.n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, model_path, &params).map_err(|e| {
            LlmError::Load(format!("load {} failed: {e:?}", model_path.display()))
        })?;
        Ok(Self {
            backend,
            model,
            opts: opts.clone(),
        })
    }

    fn render(&self, req: &ChatRequest) -> Result<String, LlmError> {
        let msgs = req
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                };
                LlamaChatMessage::new(role.to_string(), m.content.clone())
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| LlmError::InvalidRequest(format!("bad chat message: {e:?}")))?;
        self.model
            .apply_chat_template(None, &msgs, true)
            .map_err(|e| LlmError::ChatTemplate(format!("{e:?}")))
    }

    fn run(
        &self,
        req: &ChatRequest,
        mut sink: Option<&mut dyn FnMut(&str)>,
    ) -> Result<ChatResponse, LlmError> {
        if req.messages.is_empty() {
            return Err(LlmError::InvalidRequest("no messages".to_string()));
        }
        let max_tokens = req.max_tokens.unwrap_or(512);
        let temperature = req.temperature.unwrap_or(0.7);
        let start = Instant::now();

        let prompt = self.render(req)?;
        let prompt_tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| LlmError::Tokenizer(format!("{e:?}")))?;
        if prompt_tokens.len() as u64 >= self.opts.n_ctx as u64 {
            return Err(LlmError::Context(format!(
                "prompt ({} tokens) fills n_ctx ({})",
                prompt_tokens.len(),
                self.opts.n_ctx
            )));
        }

        let mut ctx = self
            .model
            .new_context(
                &self.backend,
                LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(self.opts.n_ctx))
                    .with_n_threads(self.opts.threads as i32)
                    .with_n_threads_batch(self.opts.threads as i32),
            )
            .map_err(|e| LlmError::Context(format!("{e:?}")))?;

        let mut batch = LlamaBatch::new(prompt_tokens.len().max(1), 1);
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            batch
                .add(tok, i as i32, &[0], i + 1 == prompt_tokens.len())
                .map_err(|e| LlmError::Decode(format!("{e:?}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| LlmError::Decode(format!("{e:?}")))?;

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs() as u32 ^ d.subsec_nanos());
        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(seed),
        ]);

        let mut text = String::new();
        let mut generated = 0u32;
        let mut pos = prompt_tokens.len() as i32;
        let mut step = LlamaBatch::new(1, 1);
        for _ in 0..max_tokens {
            if pos >= self.opts.n_ctx as i32 {
                break;
            }
            let tok = sampler.sample(&ctx, 0);
            if self.model.is_eog_token(tok) {
                break;
            }
            let bytes = self
                .model
                .token_to_bytes(tok, Special::Plaintext)
                .map_err(|e| LlmError::Decode(format!("{e:?}")))?;
            let piece = String::from_utf8_lossy(&bytes);
            text.push_str(&piece);
            if let Some(s) = sink.as_deref_mut() {
                s(&piece);
            }
            generated += 1;
            step.clear();
            step.add(tok, pos, &[0], true)
                .map_err(|e| LlmError::Decode(format!("{e:?}")))?;
            pos += 1;
            ctx.decode(&mut step)
                .map_err(|e| LlmError::Decode(format!("{e:?}")))?;
        }

        Ok(ChatResponse {
            backend_id: self.backend_id().to_string(),
            text,
            tokens: generated,
            millis: start.elapsed().as_millis() as u64,
        })
    }
}

impl LlmPipeline for LlamaCppLlm {
    fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        self.run(req, None)
    }

    fn complete_stream(
        &self,
        req: &ChatRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResponse, LlmError> {
        self.run(req, Some(sink))
    }

    fn backend_id(&self) -> &'static str {
        if cfg!(feature = "llm-cuda") {
            "llama-cpp-cuda"
        } else {
            "llama-cpp-vulkan"
        }
    }
}
