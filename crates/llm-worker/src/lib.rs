//! LLM sessions (llama-cpp flavors, token streaming); sync only, no HTTP/DB.

mod backend;
mod embed;

pub use backend::LlamaCppLlm;
pub use embed::LlamaCppEmbed;

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Role of a single chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Human / caller turn.
    User,
    /// Model turn.
    Assistant,
    /// System instructions.
    System,
    /// Tool result turn (`REQ-3`); pairs with `tool_call_id`.
    Tool,
}

/// One chat message: a role plus its text (empty for assistant turns that
/// consist only of tool calls).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// Who said it.
    pub role: Role,
    /// Message text (already normalized to empty when the caller sent
    /// `content: null`). For vision requests the server has already spliced
    /// the projector's media marker into this text, once per
    /// [`ChatMessage::images`] entry, at the offsets the client's content
    /// parts occupied.
    pub content: String,
    /// `role: Tool`: the assistant `tool_call` id this message answers.
    pub tool_call_id: Option<String>,
    /// `role: Assistant`: tool calls already made (`REQ-3`).
    pub tool_calls: Vec<ToolCallSpec>,
    /// Image attachments for this message, in marker order (vision chat).
    /// Decoded bytes (the server strips the `data:` URI envelope); formats
    /// are whatever `stb_image` decodes — JPEG/PNG/BMP/GIF in practice.
    pub images: Vec<ChatImage>,
}

impl Default for ChatMessage {
    fn default() -> Self {
        Self {
            role: Role::User,
            content: String::new(),
            tool_call_id: None,
            tool_calls: Vec::new(),
            images: Vec::new(),
        }
    }
}

/// One decoded image attachment (vision chat). A newtype over the bytes
/// only so `Debug` prints a length, never half a megabyte of pixels.
#[derive(Clone, PartialEq, Eq)]
pub struct ChatImage {
    /// Encoded image file bytes (JPEG/PNG/…; `MtmdBitmap::from_buf`
    /// sniffs the format).
    pub bytes: Vec<u8>,
}

impl ChatImage {
    /// Wrap encoded image bytes.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Encoded image byte count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// `true` when the attachment is empty (nothing to embed).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl std::fmt::Debug for ChatImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ChatImage({} bytes)", self.bytes.len())
    }
}

/// A tool call crossing the worker boundary (`REQ-3`). `arguments` is a
/// JSON-encoded string; `id` may be empty (the server assigns response
/// ids).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallSpec {
    /// Client-side call id (echoed assistant turns); may be empty.
    pub id: String,
    /// Function name.
    pub name: String,
    /// JSON-encoded arguments string.
    pub arguments: String,
}

/// Chat completion request (the server converts its own DTOs into this).
///
/// The constraint fields (`REQ-3`) are honored on the llama.cpp backend:
/// `grammar` / `json_schema` / `json_object` become GBNF grammar samplers,
/// `tools_json` switches onto the model's tool-capable chat template and
/// enables tool-call parsing. The stub emulates all of them
/// deterministically.
///
/// Vision (`REQ-8`): image attachments ride per-message
/// ([`ChatMessage::images`]), with the projector's media marker already
/// spliced into the message text at the right offsets. [`LlamaCppLlm`]
/// evaluates them through the loaded mmproj; a text-only model (no
/// `LlmOptions::mmproj_path`) refuses with [`LlmError::Vision`].
///
/// `Default` exists for partial construction (`..Default::default()` on
/// call sites like the CLI); an empty `model`/`messages` request is an
/// [`LlmError::InvalidRequest`] downstream, never a silent default model.
#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    /// Model id (informational; [`LlamaCppLlm`] already holds its loaded file).
    pub model: String,
    /// Conversation turns.
    pub messages: Vec<ChatMessage>,
    /// Max tokens to generate (default 512).
    pub max_tokens: Option<u32>,
    /// Sampling temperature (default 0.7).
    pub temperature: Option<f32>,
    /// Deterministic sampling seed. `Some(seed)` makes the sampler chain
    /// reproducible (same prompt + seed ⇒ same tokens); `None` keeps the
    /// wall-clock-derived seed.
    pub seed: Option<u64>,
    /// Raw GBNF grammar; generation is constrained to it verbatim.
    pub grammar: Option<String>,
    /// JSON Schema text; converted to GBNF (unsupported schema =
    /// [`LlmError::InvalidRequest`]).
    pub json_schema: Option<String>,
    /// Constrain output to any valid JSON object (`response_format:
    /// json_object`).
    pub json_object: bool,
    /// `OpenAI` `tools` array JSON. Present ⇒ tool-capable chat template +
    /// tool-call parsing.
    pub tools_json: Option<String>,
    /// `true` = `tool_choice: "none"`: tools are rendered for context but
    /// no call is requested and none is parsed out.
    pub tool_choice_none: bool,
    /// Extended sampler knobs (`REQ-3`); `None` per knob = llama.cpp
    /// default / off.
    pub sampler: SamplerParams,
}

/// Extended sampler parameters crossing the worker boundary (`REQ-3`).
/// Every knob is wired into the llama.cpp sampler chain when present.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplerParams {
    /// Distinct tokens considered per step (`<= 0` disables).
    pub top_k: Option<i32>,
    /// Minimum probability for a token to be kept.
    pub min_p: Option<f32>,
    /// Locally-typical sampling mass.
    pub typical_p: Option<f32>,
    /// Repetition penalty (1.0 = off).
    pub repeat_penalty: Option<f32>,
    /// Probability of excising the top token on a close runner-up (0 = off).
    pub xtc_probability: Option<f32>,
    /// DRY repetition penalty strength (0 = off).
    pub dry_multiplier: Option<f32>,
    /// DRY penalty exponent base.
    pub dry_base: Option<f32>,
    /// DRY sequence length below which no penalty applies.
    pub dry_allowed_length: Option<u32>,
    /// Dynamic temperature range (0 = off).
    pub dynatemp_range: Option<f32>,
    /// Dynamic temperature exponent.
    pub dynatemp_exponent: Option<f32>,
}

/// Chat completion response.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Which backend produced this (`stub-llm` | `llama-cpp-vulkan` | `llama-cpp-cuda`).
    pub backend_id: String,
    /// Generated text (assistant `content`; empty when the reply is only
    /// tool calls).
    pub text: String,
    /// Number of generated tokens.
    pub tokens: u32,
    /// Number of prompt tokens after tokenization (llama.cpp path reports
    /// the real count; the stub reports 0).
    pub prompt_tokens: u32,
    /// Tool calls parsed from the constrained output (`REQ-3`); empty =
    /// none (the reply's `finish_reason` is then stop/length as usual).
    pub tool_calls: Vec<ToolCallOut>,
    /// Wall-clock time in milliseconds.
    pub millis: u64,
}

/// One parsed tool call on a worker reply (`REQ-3`). The server assigns
/// the `OpenAI` response `id`; the parser does not surface one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallOut {
    /// Function name.
    pub name: String,
    /// JSON-encoded arguments string ("{}" when the model passed none).
    pub arguments: String,
}

/// Whether this build compiles the mtmd vision path (feature `vision`): the
/// catalog derives `capabilities.vision` from sibling-mmproj discovery
/// AND this flag, so `GET /v1/models` never advertises vision a build
/// cannot serve.
pub const VISION_COMPILED: bool = cfg!(feature = "vision");

/// Options for [`LlamaCppLlm::load`].
#[derive(Debug, Clone)]
pub struct LlmOptions {
    /// Context size in tokens. This is the KV cache's size: the cache holds
    /// `n_ctx` tokens of K and V, so the context size IS the cache budget.
    /// [`LlmOptions::vram_budget_bytes`] clamps it further.
    pub n_ctx: u32,
    /// The element type the KV cache is stored in. `F16` is llama.cpp's
    /// default; `Q8_0` halves the cache bytes at a small quality cost.
    pub kv_quant: KvQuant,
    /// Whether the KV cache is offloaded to the GPU. `false` keeps the cache
    /// in RAM (the cache-in-RAM placement) and leaves the VRAM to the
    /// weights.
    pub offload_kqv: bool,
    /// The VRAM ceiling for the KV cache, in bytes. `Some` clamps `n_ctx`
    /// so the cache stays inside the ceiling (see [`clamp_n_ctx`]); `None`
    /// means the operator declared no ceiling.
    pub vram_budget_bytes: Option<u64>,
    /// GPU layers to offload (`u32::MAX` = all).
    pub n_gpu_layers: u32,
    /// CPU threads.
    pub threads: u32,
    /// Multimodal projector (mmproj) GGUF to load alongside the model
    /// (vision chat). `None` = text-only: image-bearing requests are
    /// rejected with [`LlmError::Vision`]. A configured-but-unloadable
    /// projector is a [`LlmError::Load`] — never a silent text-only start.
    pub mmproj_path: Option<std::path::PathBuf>,
}

impl Default for LlmOptions {
    fn default() -> Self {
        Self {
            n_ctx: 8192,
            kv_quant: KvQuant::F16,
            offload_kqv: true,
            vram_budget_bytes: None,
            n_gpu_layers: u32::MAX,
            threads: std::thread::available_parallelism()
                .map_or(4, |n| u32::try_from(n.get()).unwrap_or(u32::MAX)),
            mmproj_path: None,
        }
    }
}

/// The element type the attention (KV) cache is stored in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvQuant {
    /// 16-bit floats — llama.cpp's default.
    #[default]
    F16,
    /// 8-bit quantized: half the cache bytes, a little perplexity.
    Q8_0,
}

impl KvQuant {
    /// Bytes per cached element: 2 for [`KvQuant::F16`], 1 for
    /// [`KvQuant::Q8_0`].
    #[must_use]
    pub fn bytes_per_element(self) -> u64 {
        match self {
            Self::F16 => 2,
            Self::Q8_0 => 1,
        }
    }
}

/// The KV cache's bytes per cached token: the K and the V cache each hold
/// `n_embd` elements per layer, so one token costs
/// `n_layer * n_embd * 2 * bytes_per_element` bytes.
///
/// The estimate ignores grouped-query attention (where the V cache is
/// narrower than the K cache), so it OVER-estimates the real cache — the
/// clamp below is therefore conservative, never optimistic.
#[must_use]
pub fn kv_cache_bytes_per_token(n_embd: u32, n_layer: u32, quant: KvQuant) -> u64 {
    u64::from(n_layer) * u64::from(n_embd) * 2 * quant.bytes_per_element()
}

/// The KV cache's total bytes for `n_ctx` cached tokens.
#[must_use]
pub fn kv_cache_bytes(n_ctx: u32, n_embd: u32, n_layer: u32, quant: KvQuant) -> u64 {
    u64::from(n_ctx) * kv_cache_bytes_per_token(n_embd, n_layer, quant)
}

/// Clamp the context size so the KV cache stays inside `budget_bytes`.
///
/// `None` budget means the operator declared no ceiling: the context size is
/// returned unchanged. The result is never 0 — a budget too small for a
/// single token keeps one token, and the request then fails with the usual
/// "prompt fills `n_ctx`" error instead of a zero-sized context.
#[must_use]
pub fn clamp_n_ctx(
    n_ctx: u32,
    budget_bytes: Option<u64>,
    n_embd: u32,
    n_layer: u32,
    quant: KvQuant,
) -> u32 {
    let Some(budget) = budget_bytes else {
        return n_ctx;
    };
    let per_token = kv_cache_bytes_per_token(n_embd, n_layer, quant);
    if per_token == 0 {
        return n_ctx;
    }
    let affordable = u32::try_from(budget / per_token).unwrap_or(u32::MAX);
    n_ctx.min(affordable.max(1))
}

/// Errors from [`LlmPipeline`] backends (hand-rolled, no thiserror).
#[derive(Debug)]
pub enum LlmError {
    /// Model file missing, backend init failed, or load failed.
    Load(String),
    /// Chat template rendering failed.
    ChatTemplate(String),
    /// Tokenization failed.
    Tokenizer(String),
    /// Context creation failed, or the prompt does not fit `n_ctx`.
    Context(String),
    /// Prompt decode or token generation failed.
    Decode(String),
    /// Vision request the loaded pipeline cannot serve: image content on
    /// a model without an mmproj projector (`REQ-8` refusal; the server
    /// gates this before the call — this is defense in depth).
    Vision(String),
    /// Request is unusable (e.g. no messages).
    InvalidRequest(String),
    /// Attached HTTP backend failed (upstream status + bounded reason).
    Upstream {
        /// Upstream HTTP status (400/502/503 classes).
        status: u16,
        /// Bounded upstream reason; never a key or credential-bearing URL.
        message: String,
    },
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(e) => write!(f, "failed to load model: {e}"),
            Self::ChatTemplate(e) => write!(f, "chat template failed: {e}"),
            Self::Tokenizer(e) => write!(f, "tokenizer failed: {e}"),
            Self::Context(e) => write!(f, "context failed: {e}"),
            Self::Decode(e) => write!(f, "decode failed: {e}"),
            Self::Vision(e) => write!(f, "vision unsupported: {e}"),
            Self::Upstream { status, message } => {
                write!(f, "upstream llama-server {status}: {message}")
            }
            Self::InvalidRequest(e) => write!(f, "invalid request: {e}"),
        }
    }
}

impl std::error::Error for LlmError {}

/// Sync LLM pipeline: full completion plus token-streaming completion.
pub trait LlmPipeline: Send + Sync {
    /// Run one chat completion.
    fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;
    /// Run one chat completion, pushing each decoded piece to `sink` as it arrives.
    ///
    /// Token pieces stream through `sink`; the returned [`ChatResponse`] is
    /// the final event and carries the final accounting, including
    /// `prompt_tokens` (the OpenAI-style usage surface reads it from there).
    fn complete_stream(
        &self,
        req: &ChatRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResponse, LlmError>;
    /// Best-effort cancel of any in-flight generation on this pipeline.
    ///
    /// Contract:
    /// - returns promptly (atomics only; no locks the worker holds, no join);
    /// - must not panic when no generation is running (a no-op is valid);
    /// - ordering relative to `complete`/`complete_stream` is relaxed: the
    ///   abort is observed at the generation's next internal checkpoint, and
    ///   an abort arriving before the run captures its generation may be
    ///   missed (the caller still owns shutdown semantics).
    ///
    /// Backends return the partial result as `Ok` rather than an error: the
    /// caller requested the cancellation, so a mid-stream error event would
    /// be noise for an already-disconnected client.
    fn abort(&self) {}
    /// Backend id (`stub-llm` | `llama-cpp-vulkan` | `llama-cpp-cuda`).
    fn backend_id(&self) -> &'static str;
}

/// Shared abort state for backends: a generation counter. [`AbortState::abort`]
/// bumps it; a run that captured the counter at entry is cancelled once it
/// changed. Atomics only, so `abort` can never deadlock with the worker.
#[derive(Debug, Default)]
pub(crate) struct AbortState {
    generation: AtomicU64,
}

impl AbortState {
    /// Capture the generation at run start.
    fn begin(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// True once an `abort` happened after `generation` was captured.
    fn aborted_since(&self, generation: u64) -> bool {
        self.generation.load(Ordering::Relaxed) != generation
    }

    /// Cancel the current (or next) generation.
    fn abort(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }
}

/// Deterministic stub backend: canned text, no model, no GPU. Emulates the
/// `REQ-3` request surface deterministically: `tools` (when not suppressed
/// by `tool_choice_none`) make it call the first offered tool with
/// schema-shaped arguments; `grammar`/`json_schema`/`json_object` make it
/// answer with a parseable JSON object. Image attachments (`REQ-8` vision
/// chat) are accepted and — like every other request feature here —
/// emulated: the reply is the same canned text, no vision inference
/// happens (the real gate lives in [`LlamaCppLlm`]).
#[derive(Debug, Default, Clone)]
pub struct StubLlm {
    abort: Arc<AbortState>,
}

impl LlmPipeline for StubLlm {
    fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let start = Instant::now();
        let base_text = match req.messages.last() {
            Some(m) => format!("stub reply to: {}", m.content),
            None => "stub reply".to_string(),
        };
        let tool = req
            .tools_json
            .as_deref()
            .and_then(stub_first_tool)
            .filter(|_| !req.tool_choice_none);
        let constrained = req.json_object || req.json_schema.is_some() || req.grammar.is_some();
        let (text, tool_calls) = if let Some((name, arguments)) = tool {
            // A tool is on the table and not suppressed: the stub always
            // calls it and puts nothing next to the call.
            (String::new(), vec![ToolCallOut { name, arguments }])
        } else if constrained {
            (
                serde_json::json!({ "reply": base_text }).to_string(),
                Vec::new(),
            )
        } else {
            (base_text, Vec::new())
        };
        Ok(ChatResponse {
            backend_id: self.backend_id().to_string(),
            tokens: u32::try_from(text.split_whitespace().count()).unwrap_or(u32::MAX),
            prompt_tokens: 0,
            tool_calls,
            millis: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
            text,
        })
    }

    fn complete_stream(
        &self,
        req: &ChatRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResponse, LlmError> {
        let generation = self.abort.begin();
        let resp = self.complete(req)?;
        // Token-by-word stream so an abort is observable between pieces.
        // Tool-call replies stream nothing (the route frames the parsed
        // call); the response still carries the parsed calls.
        let mut text = String::new();
        let mut tokens = 0u32;
        for word in resp.text.split_inclusive(' ') {
            if self.abort.aborted_since(generation) {
                break;
            }
            sink(word);
            text.push_str(word);
            tokens += 1;
        }
        Ok(ChatResponse {
            backend_id: resp.backend_id,
            tokens,
            prompt_tokens: resp.prompt_tokens,
            tool_calls: resp.tool_calls,
            millis: resp.millis,
            text,
        })
    }

    fn abort(&self) {
        self.abort.abort();
    }

    fn backend_id(&self) -> &'static str {
        "stub-llm"
    }
}

/// First tool of an `OpenAI` `tools` array JSON, as a deterministic
/// (name, arguments) pair: the arguments object fills each declared
/// property with `"stub"` in declaration order, so a request schema is
/// honored shape-wise without running inference.
fn stub_first_tool(tools_json: &str) -> Option<(String, String)> {
    let tools = serde_json::from_str::<serde_json::Value>(tools_json).ok()?;
    let function = tools.get(0)?.get("function")?;
    let name = function.get("name")?.as_str()?.to_owned();
    let mut arguments = serde_json::Map::new();
    if let Some(props) = function
        .get("parameters")
        .and_then(|p| p.get("properties"))
        .and_then(serde_json::Value::as_object)
    {
        for (key, value) in props {
            arguments.insert(key.clone(), stub_property_value(value));
        }
    }
    let arguments = serde_json::Value::Object(arguments).to_string();
    Some((name, arguments))
}

/// Deterministic filler for one schema property: `"stub"` for most types,
/// `true`/`0`/`null` where the JSON type demands it.
fn stub_property_value(schema: &serde_json::Value) -> serde_json::Value {
    match schema.get("type").and_then(serde_json::Value::as_str) {
        Some("boolean") => serde_json::Value::Bool(true),
        Some("number" | "integer") => serde_json::Value::from(0),
        Some("null") => serde_json::Value::Null,
        _ => serde_json::Value::String("stub".to_owned()),
    }
}

/// Sync embedding pipeline: batched inputs → L2-normalized vectors
/// (REQ-4). Deliberately separate from [`LlmPipeline`]: chat and embedding
/// models are different artifacts (different GGUFs, different context
/// flags), never interchangeable.
pub trait EmbedPipeline: Send + Sync {
    /// Embed each input. `result[i]` corresponds to `inputs[i]` (batch
    /// order preserved) and is L2-normalized (unit length; the all-zero
    /// degenerate case maps onto the first basis vector).
    ///
    /// # Errors
    ///
    /// [`LlmError::Load`] for a broken model, [`LlmError::Tokenizer`] for
    /// a non-UTF8-safe input, [`LlmError::Context`] when an input does not
    /// fit the embedding context.
    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, LlmError>;
    /// Sum of the inputs' token counts (the `usage.prompt_tokens`
    /// surface): the real tokenizer on the llama.cpp path, a deterministic
    /// word count on the stub.
    fn prompt_tokens(&self, inputs: &[String]) -> u32;
    /// Backend id (`stub-embed` | `llama-cpp-embed-vulkan` | `llama-cpp-embed-cuda`).
    fn backend_id(&self) -> &'static str;
}

/// Embedding dimension of [`StubEmbed`]: small so tests stay fast, big
/// enough that per-input vectors are distinguishable.
const STUB_EMBED_DIM: usize = 16;

/// Deterministic stub embedding backend: a content-hashed, L2-normalized
/// 16-dim vector per input. No model, no GPU; identical inputs always
/// produce identical vectors, distinct inputs (almost always) distinct
/// ones.
#[derive(Debug, Default, Clone)]
pub struct StubEmbed;

/// Deterministic unit vector for `input`: byte contributions accumulated
/// per dimension, then L2-normalized (the all-zero input maps onto the
/// first basis vector).
fn stub_vector(input: &str) -> Vec<f32> {
    let mut vector = [0.0_f32; STUB_EMBED_DIM];
    for (position, byte) in input.bytes().enumerate() {
        vector[position % STUB_EMBED_DIM] += f32::from(byte).mul_add(0.01, -1.28);
    }
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.iter().map(|x| x / norm).collect()
    } else {
        let mut unit = vec![0.0_f32; STUB_EMBED_DIM];
        unit[0] = 1.0;
        unit
    }
}

impl EmbedPipeline for StubEmbed {
    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        Ok(inputs.iter().map(|input| stub_vector(input)).collect())
    }

    fn prompt_tokens(&self, inputs: &[String]) -> u32 {
        inputs
            .iter()
            .map(|input| u32::try_from(input.split_whitespace().count()).unwrap_or(u32::MAX))
            .sum()
    }

    fn backend_id(&self) -> &'static str {
        "stub-embed"
    }
}

const _: fn() = || {
    fn assert_pipeline<T: LlmPipeline>() {}
    fn assert_embed<T: EmbedPipeline>() {}
    assert_pipeline::<StubLlm>();
    assert_pipeline::<LlamaCppLlm>();
    assert_embed::<StubEmbed>();
    assert_embed::<LlamaCppEmbed>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn req() -> ChatRequest {
        ChatRequest {
            model: "stub".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: "hello".to_string(),
                ..ChatMessage::default()
            }],
            max_tokens: None,
            temperature: None,
            seed: None,
            grammar: None,
            json_schema: None,
            json_object: false,
            tools_json: None,
            tool_choice_none: false,
            sampler: SamplerParams::default(),
        }
    }

    const TOOLS_JSON: &str = r#"[{"type":"function","function":{"name":"lookup_weather","description":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;

    /// The KV cache's byte math: one token costs the K and the V cache per
    /// layer, and `Q8_0` halves it.
    #[test]
    fn kv_cache_bytes_scale_with_quant() {
        // 8-wide embedding, 1 layer: F16 -> 8 * 2 (K+V) * 2 bytes = 32.
        assert_eq!(kv_cache_bytes_per_token(8, 1, KvQuant::F16), 32);
        assert_eq!(kv_cache_bytes_per_token(8, 1, KvQuant::Q8_0), 16);
        assert_eq!(kv_cache_bytes(100, 8, 1, KvQuant::F16), 3200);
    }

    /// The context size IS the cache budget: the clamp keeps the cache
    /// inside the operator's ceiling and never returns 0.
    #[test]
    fn clamp_n_ctx_respects_the_budget() {
        // No ceiling -> the context size is unchanged.
        assert_eq!(clamp_n_ctx(8192, None, 4096, 32, KvQuant::F16), 8192);
        // 4096 * 32 layers * 2 (K+V) * 2 bytes = 524288 bytes per token, so
        // a 1 MiB ceiling buys exactly two tokens.
        assert_eq!(clamp_n_ctx(8192, Some(1 << 20), 4096, 32, KvQuant::F16), 2);
        // `Q8_0` halves the bytes per token: twice the tokens fit.
        assert_eq!(clamp_n_ctx(8192, Some(1 << 20), 4096, 32, KvQuant::Q8_0), 4);
        // A ceiling too small for one token still keeps one token (a
        // zero-sized context would fail every request instead).
        assert_eq!(clamp_n_ctx(8192, Some(1), 4096, 32, KvQuant::F16), 1);
        // The clamp never grows the context.
        assert_eq!(clamp_n_ctx(1, Some(1 << 30), 4096, 32, KvQuant::F16), 1);
    }

    #[test]
    fn stub_tools_call_first_tool_with_schema_arguments() {
        let llm = StubLlm::default();
        let mut req = req();
        req.tools_json = Some(TOOLS_JSON.to_owned());
        let resp = llm.complete(&req).unwrap();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "lookup_weather");
        assert_eq!(resp.tool_calls[0].arguments, r#"{"city":"stub"}"#);
        assert_eq!(resp.text, "");
        // Streaming mirrors the parsed calls (route frames the fragments).
        let streamed = llm.complete_stream(&req, &mut |_| {}).unwrap();
        assert_eq!(streamed.tool_calls, resp.tool_calls);
    }

    #[test]
    fn stub_tool_choice_none_suppresses_calls() {
        let llm = StubLlm::default();
        let mut req = req();
        req.tools_json = Some(TOOLS_JSON.to_owned());
        req.tool_choice_none = true;
        let resp = llm.complete(&req).unwrap();
        assert_eq!(resp.tool_calls, [] as [ToolCallOut; 0]);
        assert!(resp.text.contains("hello"));
    }

    #[test]
    fn stub_json_object_yields_parseable_content() {
        let llm = StubLlm::default();
        let mut object_req = req();
        object_req.json_object = true;
        let resp = llm.complete(&object_req).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&resp.text).unwrap();
        assert_eq!(parsed["reply"], "stub reply to: hello");
        // Raw grammar behaves the same on the stub (deterministic JSON).
        let mut grammar_req = req();
        grammar_req.grammar = Some("root ::= \"{}\"".to_owned());
        let grammar_resp = llm.complete(&grammar_req).unwrap();
        serde_json::from_str::<serde_json::Value>(&grammar_resp.text).unwrap();
    }

    #[test]
    fn stub_tool_role_round_trip_is_plain_text() {
        let llm = StubLlm::default();
        let mut req = req();
        req.messages = vec![
            ChatMessage {
                role: Role::User,
                content: "weather?".to_string(),
                ..ChatMessage::default()
            },
            ChatMessage {
                role: Role::Assistant,
                content: String::new(),
                tool_call_id: None,
                tool_calls: vec![ToolCallSpec {
                    id: "call_1".to_owned(),
                    name: "lookup_weather".to_owned(),
                    arguments: r#"{"city":"Reykjavik"}"#.to_owned(),
                }],
                ..ChatMessage::default()
            },
            ChatMessage {
                role: Role::Tool,
                content: "4C, raining".to_string(),
                tool_call_id: Some("call_1".to_owned()),
                ..ChatMessage::default()
            },
        ];
        let resp = llm.complete(&req).unwrap();
        assert_eq!(resp.tool_calls, [] as [ToolCallOut; 0]);
        assert!(resp.text.contains("4C, raining"));
    }

    #[test]
    fn stub_round_trip() {
        let llm = StubLlm::default();
        assert_eq!(llm.backend_id(), "stub-llm");
        let resp = llm.complete(&req()).unwrap();
        assert_eq!(resp.backend_id, "stub-llm");
        assert!(resp.text.contains("hello"));

        let mut pieces = Vec::new();
        let streamed = llm
            .complete_stream(&req(), &mut |s: &str| pieces.push(s.to_string()))
            .unwrap();
        assert_eq!(streamed.text, resp.text);
        assert_eq!(streamed.text, pieces.concat());
    }

    #[test]
    fn abort_stops_stub_stream_and_returns_partial() {
        let llm = StubLlm::default();
        let mut pieces = Vec::new();
        // Abort before the run captures its generation: contract says this
        // may be missed, so abort mid-stream via the sink callback instead.
        let llm2 = std::sync::Arc::new(StubLlm::default());
        let llm3 = std::sync::Arc::clone(&llm2);
        let streamed = llm2
            .complete_stream(&req(), &mut |s: &str| {
                pieces.push(s.to_string());
                // Cancel after the first piece: the loop must stop at the
                // next checkpoint and return the partial text as `Ok`.
                llm3.abort();
            })
            .unwrap();
        assert_eq!(pieces.len(), 1, "stream stopped right after the abort");
        assert_eq!(streamed.text, pieces.concat());
        assert_eq!(streamed.tokens, 1);
        // `complete` is unaffected by a stale abort from a finished run.
        let fresh = llm.complete(&req()).unwrap();
        assert!(fresh.text.contains("hello"));
    }

    #[test]
    fn abort_with_no_generation_is_noop() {
        let llm = StubLlm::default();
        llm.abort();
        llm.abort();
        let resp = llm.complete_stream(&req(), &mut |_| {}).unwrap();
        assert!(resp.text.contains("hello"));
    }

    #[test]
    fn load_missing_file_errors() {
        let err = LlamaCppLlm::load(Path::new("/nonexistent/model.gguf"), &LlmOptions::default())
            .unwrap_err();
        assert!(matches!(err, LlmError::Load(_)));
    }

    #[test]
    fn load_missing_mmproj_errors_before_backend_init() {
        // The projector existence check is fail-fast (before backend init and
        // model load), so the error names the mmproj even though the "model"
        // file is also garbage — the misconfiguration is the operator's to
        // fix, never a silent text-only start.
        let base = std::env::temp_dir().join(format!("honkbox-llm-mmproj-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let model = base.join("model.gguf");
        std::fs::write(&model, b"not a gguf").unwrap();
        let opts = LlmOptions {
            mmproj_path: Some(base.join("missing-mmproj.gguf")),
            ..LlmOptions::default()
        };
        let err = LlamaCppLlm::load(&model, &opts).unwrap_err();
        std::fs::remove_dir_all(&base).unwrap();
        let LlmError::Load(message) = err else {
            panic!("expected Load error, got {err:?}");
        };
        assert!(message.contains("mmproj file not found"), "{message}");
    }

    #[test]
    fn stub_accepts_image_attachments_as_emulated_text() {
        // The stub emulates vision requests: image attachments ride along,
        // the reply stays the canned text (no vision inference happens).
        let llm = StubLlm::default();
        let mut req = req();
        req.messages[0].images = vec![ChatImage::new(vec![1, 2, 3])];
        let resp = llm.complete(&req).unwrap();
        assert!(resp.text.contains("hello"));
        // Debug never prints pixel bytes.
        assert_eq!(
            format!("{:?}", ChatImage::new(vec![9; 700_000])),
            "ChatImage(700000 bytes)"
        );
        assert!(ChatImage::new(Vec::new()).is_empty());
        assert_eq!(ChatImage::new(vec![1, 2]).len(), 2);
    }

    #[test]
    fn stub_seed_is_deterministic() {
        // The stub has no sampler, but the seed-honored contract holds at the
        // worker API level: two same-seed runs are byte-identical in text and
        // token counts, with or without a seed supplied.
        let llm = StubLlm::default();
        let mut seeded = req();
        seeded.seed = Some(42);
        let a = llm.complete(&seeded).unwrap();
        let b = llm.complete(&seeded).unwrap();
        assert_eq!(a.text, b.text);
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.prompt_tokens, b.prompt_tokens);
        let unseeded = llm.complete(&req()).unwrap();
        assert_eq!(unseeded.text, a.text);
        assert_eq!(unseeded.tokens, a.tokens);
    }

    #[test]
    fn stub_stream_reports_prompt_tokens_in_final_event() {
        // Stream contract: pieces go through the sink, and the final
        // `ChatResponse` carries `prompt_tokens` (0 for the stub).
        let llm = StubLlm::default();
        let resp = llm.complete_stream(&req(), &mut |_| {}).unwrap();
        assert_eq!(resp.prompt_tokens, 0);
        assert!(resp.tokens > 0);
    }
}
