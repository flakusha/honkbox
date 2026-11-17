//! llama.cpp-backed [`LlmPipeline`](super::LlmPipeline) (GGUF via llama-cpp-4).

use super::{
    AbortState, ChatRequest, ChatResponse, LlmError, LlmOptions, LlmPipeline, Role, ToolCallOut,
};
use llama_cpp_4::chat::{
    ChatApplyParams, ChatParams, ChatTemplates, ToolChoice, json_schema_to_grammar,
    parse_messages_oaicompat,
};
use llama_cpp_4::context::LlamaContext;
use llama_cpp_4::model::LlamaChatMessage;
#[cfg(feature = "vision")]
use llama_cpp_4::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText,
};
use llama_cpp_4::{
    AddBos, LlamaBackend, LlamaBatch, LlamaContextParams, LlamaModel, LlamaModelParams,
    LlamaSampler, Special,
};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
#[cfg(feature = "vision")]
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Minimum candidate mass kept by `typical`/`min_p`/`xtc` filters.
const SAMPLER_MIN_KEEP: usize = 1;

/// `xtc` runner-up proximity threshold (llama.cpp default).
const XTC_THRESHOLD: f32 = 0.1;

/// DRY sequence breakers (llama.cpp defaults: newline, colon, quote, asterisk).
const DRY_BREAKERS: [&str; 4] = ["\n", ":", "\"", "*"];

/// Generic any-JSON GBNF grammar, enforcing `response_format:
/// {"type":"json_object"}`. Copied verbatim from llama-cpp-4's bundled
/// `src/grammar/json.gbnf` (upstream llama.cpp `grammars/json.gbnf`).
const JSON_GBNF: &str = r#"root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null")

object ::=
  "{" (
            string ":" value
    ("," string ":" value)*
  )? "}"

array  ::=
  "[" (
            value
    ("," value)*
  )? "]"

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\""

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
"#;

use llama_cpp_4::token::LlamaToken;

/// Everything [`LlamaCppLlm::run`] needs to start sampling one request: the
/// rendered prompt (with media markers in place, vision chat), its token
/// form (unused on the mtmd path — the projector re-tokenizes), the
/// tool-template context (for output parsing) and the grammar sampler
/// head, if any.
struct PromptPlan {
    prompt: String,
    tokens: Vec<LlamaToken>,
    chat_params: Option<ChatParams>,
    grammar_head: Option<LlamaSampler>,
}

/// GGUF model session. Owns the llama backend guard, so it is `Send + Sync`.
/// A fresh context is built per request (cheap next to decode; avoids
/// self-referential struct). Sync only, tokio-free.
pub struct LlamaCppLlm {
    backend: LlamaBackend,
    model: LlamaModel,
    opts: LlmOptions,
    /// Multimodal projector when one was configured ([`LlmOptions::
    /// mmproj_path`]); behind a `Mutex` because `eval_chunks` mutates the
    /// projector's internal encode state (`tokenize` alone would only need
    /// a shared handle). Compiled only with the `vision` feature (see the
    /// feature docs in `Cargo.toml` for the vendor link caveat).
    #[cfg(feature = "vision")]
    mtmd: Option<Mutex<MtmdContext>>,
    /// Cancel signal: shared with `abort`; checked at each token step.
    abort: Arc<AbortState>,
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
        // Fail fast on a misconfigured projector before any backend/model
        // work: the existence half of the check is cheap and the error
        // names the mmproj, not the model.
        if let Some(mmproj_path) = &opts.mmproj_path
            && !mmproj_path.is_file()
        {
            return Err(LlmError::Load(format!(
                "mmproj file not found: {}",
                mmproj_path.display()
            )));
        }
        let mut backend = LlamaBackend::init()
            .map_err(|e| LlmError::Load(format!("backend init failed: {e:?}")))?;
        backend.void_logs();
        let params = LlamaModelParams::default().with_n_gpu_layers(opts.n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, model_path, &params)
            .map_err(|e| LlmError::Load(format!("load {} failed: {e:?}", model_path.display())))?;
        // Vision chat: load the configured mmproj projector against the text
        // model. A configured-but-failed projector is a loud load error —
        // never a silent text-only start (the request gate would answer 400
        // for image parts on a model the operator believes is multimodal).
        #[cfg(feature = "vision")]
        let mtmd = match opts.mmproj_path.as_deref() {
            Some(mmproj_path) => {
                MtmdContext::void_logs();
                let mtmd =
                    MtmdContext::init_from_file(mmproj_path, &model, MtmdContextParams::default())
                        .map_err(|e| {
                            LlmError::Load(format!(
                                "mmproj load {} failed: {e:?}",
                                mmproj_path.display()
                            ))
                        })?;
                if !mtmd.supports_vision() {
                    return Err(LlmError::Load(format!(
                        "mmproj {} carries no vision encoder",
                        mmproj_path.display()
                    )));
                }
                Some(Mutex::new(mtmd))
            }
            None => None,
        };
        #[cfg(not(feature = "vision"))]
        if opts.mmproj_path.is_some() {
            // Honest refusal, not silent ignoring: the operator asked for a
            // multimodal pipeline and this build cannot serve one.
            return Err(LlmError::Load(
                "mmproj configured but this build lacks the llm-worker `vision` feature (vendor link caveat; rebuild with --features llm-vision)".to_owned(),
            ));
        }
        Ok(Self {
            backend,
            model,
            opts: opts.clone(),
            #[cfg(feature = "vision")]
            mtmd,
            abort: Arc::default(),
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
                    Role::Tool => "tool",
                };
                LlamaChatMessage::new(role.to_string(), m.content.clone())
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| LlmError::InvalidRequest(format!("bad chat message: {e:?}")))?;
        self.model
            .apply_chat_template(None, &msgs, true)
            .map_err(|e| LlmError::ChatTemplate(format!("{e:?}")))
    }

    /// Worker messages as the OpenAI-shaped JSON array the chat-tools
    /// template layer consumes (`REQ-3`): tool roles, assistant `tool_calls`
    /// and `tool_call_id` round-trip here; empty content on an
    /// assistant-with-tool_calls serializes as `null`.
    fn openai_messages(req: &ChatRequest) -> String {
        let messages = req
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                    Role::Tool => "tool",
                };
                let mut obj = serde_json::json!({ "role": role });
                if m.content.is_empty() && !m.tool_calls.is_empty() {
                    obj["content"] = serde_json::Value::Null;
                } else {
                    obj["content"] = m.content.clone().into();
                }
                if let Some(id) = &m.tool_call_id {
                    obj["tool_call_id"] = id.clone().into();
                }
                if !m.tool_calls.is_empty() {
                    obj["tool_calls"] = serde_json::Value::Array(
                        m.tool_calls
                            .iter()
                            .map(|tc| {
                                serde_json::json!({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.arguments,
                                    },
                                })
                            })
                            .collect(),
                    );
                }
                obj
            })
            .collect::<Vec<_>>();
        serde_json::to_string(&messages).unwrap_or_else(|_| "[]".to_owned())
    }

    fn plan_prompt(&self, req: &ChatRequest) -> Result<PromptPlan, LlmError> {
        if let Some(tools_json) = req.tools_json.as_deref() {
            let messages_json = parse_messages_oaicompat(&Self::openai_messages(req))
                .map_err(|e| LlmError::InvalidRequest(format!("bad tool messages: {e:?}")))?;
            let templates = ChatTemplates::from_model(&self.model, None)
                .map_err(|e| LlmError::ChatTemplate(format!("{e:?}")))?;
            let mut apply = ChatApplyParams::new(messages_json).with_tools(tools_json);
            apply = if req.tool_choice_none {
                apply.with_tool_choice(ToolChoice::None)
            } else {
                apply.with_tool_choice(ToolChoice::Auto)
            };
            let params = templates
                .apply(&apply)
                .map_err(|e| LlmError::ChatTemplate(format!("{e:?}")))?;
            let tokens = params
                .tokenize_prompt(&self.model, AddBos::Always)
                .map_err(|e| LlmError::Tokenizer(format!("{e:?}")))?;
            let grammar_head = params.grammar_sampler(&self.model);
            return Ok(PromptPlan {
                prompt: params.prompt.clone(),
                tokens,
                chat_params: Some(params),
                grammar_head,
            });
        }
        // Grammar-only constraint: raw GBNF wins, then `json_schema`
        // (converted here so a non-convertible schema surfaces as an
        // `InvalidRequest` the route maps to an honest 400), then the
        // generic any-JSON grammar for `json_object`.
        let grammar = if let Some(grammar) = req.grammar.as_deref() {
            Some(grammar.to_owned())
        } else if let Some(schema) = req.json_schema.as_deref() {
            Some(json_schema_to_grammar(schema, false).map_err(|e| {
                LlmError::InvalidRequest(format!("json_schema is not convertible: {e:?}"))
            })?)
        } else if req.json_object {
            Some(JSON_GBNF.to_owned())
        } else {
            None
        };
        let grammar_head = grammar
            .map(|grammar| {
                LlamaSampler::try_grammar(&self.model, &grammar, "root")
                    .map_err(|e| LlmError::InvalidRequest(format!("invalid grammar: {e:?}")))
            })
            .transpose()?;
        let prompt = self.render(req)?;
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| LlmError::Tokenizer(format!("{e:?}")))?;
        Ok(PromptPlan {
            prompt,
            tokens,
            chat_params: None,
            grammar_head,
        })
    }

    /// Vision prefill (mtmd): decode the request's image attachments into
    /// bitmaps, tokenize the marker-bearing prompt into text+image chunks,
    /// and evaluate them into `ctx` (logits left on the final token, so
    /// sampling starts immediately). Returns the KV position after the
    /// prompt. Requires an mmproj projector; image content on a text-only
    /// model is the explicit [`LlmError::Vision`] refusal — never a
    /// flattened or silently dropped image (`REQ-8`).
    #[cfg(feature = "vision")]
    fn eval_images(
        &self,
        ctx: &mut LlamaContext<'_>,
        prompt: &str,
        req: &ChatRequest,
    ) -> Result<(i32, usize), LlmError> {
        let Some(mtmd) = self.mtmd.as_ref() else {
            return Err(LlmError::Vision(
                "image content requires an mmproj projector; this model is text-only".to_owned(),
            ));
        };
        let mut mtmd = mtmd
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut bitmaps = Vec::new();
        for message in &req.messages {
            for image in &message.images {
                if image.is_empty() {
                    return Err(LlmError::InvalidRequest(
                        "empty image attachment".to_owned(),
                    ));
                }
                // `from_buf` sniffs the container format (stb_image);
                // a client-sent payload the projector cannot decode is a
                // request problem, not a server one.
                bitmaps.push(MtmdBitmap::from_buf(&mtmd, &image.bytes).map_err(|e| {
                    LlmError::InvalidRequest(format!("image attachment not decodable: {e}"))
                })?);
            }
        }
        let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        // `add_special = true` matches the text path's `AddBos::Always`;
        // `parse_special = true` so chat-template specials in the rendered
        // prompt tokenize as specials, exactly as `str_to_token` did.
        let text = MtmdInputText::new(prompt, true, true);
        let mut chunks = MtmdInputChunks::new();
        mtmd.tokenize(&text, &refs, &mut chunks)
            .map_err(|e| LlmError::Tokenizer(format!("vision tokenize failed: {e:?}")))?;
        let mut n_past = 0i32;
        let n_batch = i32::try_from(ctx.n_batch()).unwrap_or(i32::MAX);
        mtmd.eval_chunks(ctx, &chunks, 0, 0, n_batch, true, &mut n_past)
            .map_err(|e| LlmError::Context(format!("vision eval failed: {e:?}")))?;
        Ok((n_past, usize::try_from(n_past).unwrap_or(usize::MAX)))
    }

    /// Non-`vision` build: every image request is the explicit refusal
    /// (feature documented in `Cargo.toml`; no image is ever flattened or
    /// dropped silently).
    #[cfg(not(feature = "vision"))]
    fn eval_images(
        &self,
        _ctx: &mut LlamaContext<'_>,
        _prompt: &str,
        _req: &ChatRequest,
    ) -> Result<(i32, usize), LlmError> {
        // Signature parity with the `vision` variant keeps the call site
        // cfg-free; `self` is deliberately unused here.
        let _ = self;
        Err(LlmError::Vision(
            "this build has no vision support (llm-worker `vision` feature off; vendor link caveat, see Cargo.toml)"
                .to_owned(),
        ))
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
        let generation = self.abort.begin();
        // `REQ-3`: constrained generation — tools take the model's
        // tool-capable chat template (grammar + per-family output parser);
        // grammar-only constraints (raw grammar, json_schema, json_object)
        // ride the simple render plus a GBNF grammar sampler as chain head.
        let plan = self.plan_prompt(req)?;
        let chat_params = plan.chat_params;
        let grammar_head = plan.grammar_head;
        let n_images: usize = req.messages.iter().map(|m| m.images.len()).sum();

        let mut ctx = self
            .model
            .new_context(
                &self.backend,
                LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(self.opts.n_ctx))
                    .with_n_threads(i32::try_from(self.opts.threads).unwrap_or(i32::MAX))
                    .with_n_threads_batch(i32::try_from(self.opts.threads).unwrap_or(i32::MAX)),
            )
            .map_err(|e| LlmError::Context(format!("{e:?}")))?;

        // Prefill: plain text decodes the tokenized prompt; image content
        // goes through the projector (tokenize + eval chunks in one call).
        // `pos` continues from wherever the prefill left the KV cache.
        let (mut pos, prompt_len) = if n_images == 0 {
            let prompt_tokens = plan.tokens;
            if u64::try_from(prompt_tokens.len()).unwrap_or(u64::MAX) >= u64::from(self.opts.n_ctx)
            {
                return Err(LlmError::Context(format!(
                    "prompt ({} tokens) fills n_ctx ({})",
                    prompt_tokens.len(),
                    self.opts.n_ctx
                )));
            }
            let mut batch = LlamaBatch::new(prompt_tokens.len().max(1), 1);
            for (i, &tok) in prompt_tokens.iter().enumerate() {
                batch
                    .add(
                        tok,
                        i32::try_from(i).unwrap_or(i32::MAX),
                        &[0],
                        i + 1 == prompt_tokens.len(),
                    )
                    .map_err(|e| LlmError::Decode(format!("{e:?}")))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| LlmError::Decode(format!("{e:?}")))?;
            (
                i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX),
                prompt_tokens.len(),
            )
        } else {
            self.eval_images(&mut ctx, &plan.prompt, req)?
        };

        let seed = req.seed.map_or_else(wallclock_seed, |seed| {
            u32::try_from(seed).unwrap_or(u32::MAX)
        });
        let sampler = self.sampler_chain(req, grammar_head, temperature, seed);

        let mut text = String::new();
        let mut generated = 0u32;
        let mut step = LlamaBatch::new(1, 1);
        while generated < max_tokens {
            // Abort checkpoint: the per-token decode returns to this loop, so
            // checking here stops generation one token after `abort` fires —
            // no FFI-level abort needed.
            if self.abort.aborted_since(generation) {
                break;
            }
            if pos >= i32::try_from(self.opts.n_ctx).unwrap_or(i32::MAX) {
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

        // `REQ-3`: parse tool calls out of a tool-constrained reply with
        // the template's own per-family parser; the parsed content also
        // replaces the raw text (reasoning wrappers stripped).
        let (tool_calls, text) = match chat_params.as_ref().filter(|_| !req.tool_choice_none) {
            Some(params) => parse_tool_output(params, &text),
            None => (Vec::new(), text),
        };

        Ok(ChatResponse {
            backend_id: self.backend_id().to_string(),
            text,
            tokens: generated,
            prompt_tokens: u32::try_from(prompt_len).unwrap_or(u32::MAX),
            tool_calls,
            millis: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
    }

    /// Sampler chain for one request: the optional grammar head first
    /// (already trigger/prefill-correct when it came from a tool template;
    /// a raw-grammar head constrains generated output only), then the
    /// wired `REQ-3` knobs in llama.cpp `common` order (dry → penalties →
    /// `top_k` → typical → `min_p` → xtc → temperature → dist).
    fn sampler_chain(
        &self,
        req: &ChatRequest,
        grammar_head: Option<LlamaSampler>,
        temperature: f32,
        seed: u32,
    ) -> LlamaSampler {
        let knobs = &req.sampler;
        let mut chain: Vec<LlamaSampler> = Vec::new();
        if let Some(head) = grammar_head {
            chain.push(head);
        }
        if knobs
            .dry_multiplier
            .is_some_and(|multiplier| multiplier > 0.0)
        {
            // The breakers are compile-time literals: `dry`'s interior-NUL
            // panic path is unreachable here.
            chain.push(LlamaSampler::default().dry(
                &self.model,
                knobs.dry_multiplier.unwrap_or(0.0),
                knobs.dry_base.unwrap_or(1.75),
                i32::try_from(knobs.dry_allowed_length.unwrap_or(2)).unwrap_or(i32::MAX),
                -1,
                DRY_BREAKERS,
            ));
        }
        if let Some(penalty) = knobs.repeat_penalty {
            chain.push(LlamaSampler::penalties_simple(
                self.model.n_vocab(),
                -1,
                penalty,
            ));
        }
        if let Some(k) = knobs.top_k {
            chain.push(LlamaSampler::top_k(k));
        }
        if let Some(p) = knobs.typical_p {
            chain.push(LlamaSampler::typical(p, SAMPLER_MIN_KEEP));
        }
        if let Some(p) = knobs.min_p {
            chain.push(LlamaSampler::min_p(p, SAMPLER_MIN_KEEP));
        }
        if let Some(probability) = knobs
            .xtc_probability
            .filter(|probability| *probability > 0.0)
        {
            chain.push(LlamaSampler::xtc(
                probability,
                XTC_THRESHOLD,
                SAMPLER_MIN_KEEP,
                seed,
            ));
        }
        match knobs.dynatemp_range.filter(|range| *range > 0.0) {
            Some(range) => chain.push(LlamaSampler::temp_ext(
                temperature,
                range,
                knobs.dynatemp_exponent.unwrap_or(1.0),
            )),
            None => chain.push(LlamaSampler::temp(temperature)),
        }
        chain.push(LlamaSampler::dist(seed));
        LlamaSampler::chain_simple(chain)
    }
}

/// Fallback seed for unseeded requests: wall-clock seconds xor nanoseconds
/// (same derivation as before request-controlled seeds existed).
fn wallclock_seed() -> u32 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        u32::try_from(d.as_secs()).unwrap_or(u32::MAX) ^ d.subsec_nanos()
    })
}

/// Parse a tool-constrained reply with the template's per-family parser
/// (`REQ-3`). Returns the parsed calls plus the reply's plain content
/// (reasoning-stripped). A parse failure is not an error: the model simply
/// did not produce a recognizable tool-call envelope, so the raw text
/// stands and no calls are reported.
fn parse_tool_output(params: &ChatParams, text: &str) -> (Vec<ToolCallOut>, String) {
    let Ok(message_json) = params.parse(text, false) else {
        return (Vec::new(), text.to_owned());
    };
    let Ok(message) = serde_json::from_str::<serde_json::Value>(&message_json) else {
        return (Vec::new(), text.to_owned());
    };
    let tool_calls = message
        .get("tool_calls")
        .and_then(serde_json::Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(|call| {
                    let name = call
                        .pointer("/function/name")
                        .and_then(serde_json::Value::as_str)?;
                    Some(ToolCallOut {
                        name: name.to_owned(),
                        arguments: call
                            .pointer("/function/arguments")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("{}")
                            .to_owned(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let content = message
        .get("content")
        .and_then(serde_json::Value::as_str)
        .map_or_else(|| text.to_owned(), str::to_owned);
    (tool_calls, content)
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

    fn abort(&self) {
        self.abort.abort();
    }

    fn backend_id(&self) -> &'static str {
        if cfg!(feature = "llm-cuda") {
            "llama-cpp-cuda"
        } else {
            "llama-cpp-vulkan"
        }
    }
}
