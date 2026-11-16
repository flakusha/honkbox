//! llama.cpp-backed [`EmbedPipeline`](super::EmbedPipeline) (GGUF via
//! llama-cpp-4, REQ-4).
//!
//! Per input: tokenize (BOS prepended), decode one whole-input batch into a
//! fresh embeddings context (mean pooling, non-causal attention — what
//! llama.cpp's embedding models expect; the binding documents
//! `LlamaAttentionType::NonCausal` as the extraction-time default for
//! embedding models), read the sequence embedding, L2-normalize. Inputs are
//! batched **separately** — one decode per input — so batch order and
//! per-input isolation are trivially preserved at the cost of one decode
//! round-trip per input (embeddings are cheap next to chat decode).

use llama_cpp_4::context::params::{LlamaAttentionType, LlamaContextParams, LlamaPoolingType};
use llama_cpp_4::model::AddBos;
use llama_cpp_4::{LlamaBackend, LlamaBatch, LlamaModel, LlamaModelParams};
use std::num::NonZeroU32;
use std::path::Path;

use super::{EmbedPipeline, LlmError, LlmOptions};

/// GGUF embedding model session. Owns the llama backend guard, so it is
/// `Send + Sync`; a fresh context is built per input (same pattern as the
/// chat backend — cheap next to the forward pass).
pub struct LlamaCppEmbed {
    backend: LlamaBackend,
    model: LlamaModel,
    opts: LlmOptions,
}

impl std::fmt::Debug for LlamaCppEmbed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppEmbed").finish_non_exhaustive()
    }
}

impl LlamaCppEmbed {
    /// Load a GGUF embedding model. Checks the path first so a missing file
    /// never initializes the process-global backend.
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
            .map_err(|err| LlmError::Load(format!("backend init failed: {err:?}")))?;
        backend.void_logs();
        let params = LlamaModelParams::default().with_n_gpu_layers(opts.n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, model_path, &params).map_err(|err| {
            LlmError::Load(format!("load {} failed: {err:?}", model_path.display()))
        })?;
        Ok(Self {
            backend,
            model,
            opts: opts.clone(),
        })
    }

    /// Embedding-context params: mean pooling (`n_embd`-wide sequence
    /// embedding — the common retrieval shape), non-causal attention
    /// (embedding models are bidirectional encoders; the binding documents
    /// `NonCausal` as the extraction-time default), embeddings enabled.
    fn embedding_context_params(&self) -> LlamaContextParams {
        LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.opts.n_ctx))
            .with_n_threads(i32::try_from(self.opts.threads).unwrap_or(i32::MAX))
            .with_n_threads_batch(i32::try_from(self.opts.threads).unwrap_or(i32::MAX))
            .with_pooling_type(LlamaPoolingType::Mean)
            .with_attention_type(LlamaAttentionType::NonCausal)
            .with_embeddings(true)
    }

    /// Embed one input: tokenize, decode, read the pooled sequence
    /// embedding, L2-normalize.
    fn embed_one(&self, input: &str) -> Result<Vec<f32>, LlmError> {
        let tokens = self
            .model
            .str_to_token(input, AddBos::Always)
            .map_err(|err| LlmError::Tokenizer(format!("{err:?}")))?;
        if tokens.is_empty() {
            return Err(LlmError::InvalidRequest(
                "input tokenized to zero tokens".to_owned(),
            ));
        }
        if u64::try_from(tokens.len()).unwrap_or(u64::MAX) >= u64::from(self.opts.n_ctx) {
            return Err(LlmError::Context(format!(
                "input ({} tokens) fills n_ctx ({})",
                tokens.len(),
                self.opts.n_ctx
            )));
        }
        let mut ctx = self
            .model
            .new_context(&self.backend, self.embedding_context_params())
            .map_err(|err| LlmError::Context(format!("{err:?}")))?;
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (position, &token) in tokens.iter().enumerate() {
            batch
                .add(
                    token,
                    i32::try_from(position).unwrap_or(i32::MAX),
                    &[0],
                    false,
                )
                .map_err(|err| LlmError::Decode(format!("{err:?}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|err| LlmError::Decode(format!("{err:?}")))?;
        let embedding = ctx
            .embeddings_seq_ith(0)
            .map_err(|err| LlmError::Decode(format!("{err:?}")))?;
        Ok(l2_normalize(embedding))
    }
}

/// L2-normalize: unit length, or the first basis vector for the degenerate
/// all-zero embedding (a real model never produces one; the guard keeps
/// the contract total).
fn l2_normalize(embedding: &[f32]) -> Vec<f32> {
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter().map(|x| x / norm).collect()
    } else {
        let mut unit = vec![0.0_f32; embedding.len()];
        if !unit.is_empty() {
            unit[0] = 1.0;
        }
        unit
    }
}

impl EmbedPipeline for LlamaCppEmbed {
    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        inputs.iter().map(|input| self.embed_one(input)).collect()
    }

    fn prompt_tokens(&self, inputs: &[String]) -> u32 {
        inputs
            .iter()
            .map(|input| {
                u32::try_from(
                    self.model
                        .str_to_token(input, AddBos::Always)
                        .map_or(0, |tokens| tokens.len()),
                )
                .unwrap_or(u32::MAX)
            })
            .sum()
    }

    fn backend_id(&self) -> &'static str {
        if cfg!(feature = "llm-cuda") {
            "llama-cpp-embed-cuda"
        } else {
            "llama-cpp-embed-vulkan"
        }
    }
}
