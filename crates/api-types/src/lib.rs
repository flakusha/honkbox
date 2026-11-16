//! Shared DTOs: the only types that cross the Rust/TS boundary.
//!
//! No routes, DB, tokio, or backend logic — serde + `ts-rs` + `utoipa`
//! derives only, so codegen never pulls in the server dependency tree.

use serde::{Deserialize, Serialize};
use ts_rs::TS;
use utoipa::ToSchema;

/// Liveness probe response.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct Health {
    /// Always `"ok"` while the server is alive.
    pub status: String,
}

/// Binary name + version response.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct Version {
    /// Binary name, always `"honkbox"`.
    pub name: String,
    /// Crate version, from `CARGO_PKG_VERSION`.
    pub version: String,
}

/// Chat role.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    /// Human / caller turn.
    User,
    /// Model turn.
    Assistant,
    /// System instructions.
    System,
}

/// One chat message.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct ChatMessage {
    /// Who said it.
    pub role: ChatRole,
    /// Message text.
    pub content: String,
}

/// Chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct ChatRequest {
    /// Model id (informational; the backend is chosen by the server).
    pub model: String,
    /// Conversation turns.
    pub messages: Vec<ChatMessage>,
    /// Max tokens to generate; backend default when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature; backend default when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Per-request residency budget for this model's swap entry, in seconds
    /// (NEW-8; the Ollama `keep_alive` / LM Studio `ttl` analog). The entry's
    /// effective window is `max(remaining, this)`, capped at
    /// `[swap] max_ttl_secs`; `0` = use-then-unload, still shielded by the
    /// remaining window so it cannot thrash. Values over the cap are clamped
    /// at the route (`warn!`, never a 400); `[swap] pin` entries ignore it
    /// entirely. Ignored when the swap manager is off (stub backends,
    /// `[swap] enabled = false`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
}

/// Chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct ChatResponse {
    /// Which backend produced this.
    pub backend_id: String,
    /// Generated text.
    pub text: String,
    /// Number of generated tokens.
    pub tokens: u32,
    /// Wall-clock time in milliseconds.
    pub millis: u64,
}

/// One transcript segment.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct TranscriptSegment {
    /// Segment start, milliseconds from the audio start.
    pub start_ms: u32,
    /// Segment end, milliseconds from the audio start.
    pub end_ms: u32,
    /// Segment text.
    pub text: String,
}

/// Transcription result.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct Transcript {
    /// Which backend produced this.
    pub backend_id: String,
    /// Full transcription text.
    pub text: String,
    /// Timestamped segments, in order.
    pub segments: Vec<TranscriptSegment>,
    /// Wall-clock time in milliseconds.
    pub millis: u64,
}

/// Speech synthesis request.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct SynthesizeRequest {
    /// Model id (informational; the backend is chosen by the server).
    pub model: String,
    /// Text to synthesize.
    pub text: String,
    /// Voice name; backend default when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// Output sample rate in Hz; backend default when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    /// Per-request residency budget for this model's swap entry, in seconds
    /// (NEW-8; same contract as [`ChatRequest::ttl_seconds`]). Over-cap
    /// values are clamped at the route; audio pipelines are startup-loaded
    /// today, so the advice takes effect once per-kind swap lands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
}

/// Model modality as discovered on disk (`GET /v1/models` entries). Mirrors
/// the registry's on-disk kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    /// Language-model weights (LLM root).
    Llm,
    /// Stable-Diffusion checkpoint / diffusion-model weights.
    Sd,
    /// `LoRA` adapter.
    Lora,
    /// Embedding / textual-inversion weights.
    Embedding,
    /// Standalone VAE weights.
    Vae,
    /// Speech-to-text weights (STT root).
    Stt,
    /// Text-to-speech weights (TTS root).
    Tts,
}

/// GGUF header summary (`GET /v1/models` entry `metadata`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct GgufMetadata {
    /// GGUF spec version.
    pub version: u32,
    /// Tensor count declared in the header.
    pub tensor_count: u64,
    /// `general.name` KV string, when present.
    pub name: Option<String>,
    /// `general.architecture` KV string, when present.
    pub architecture: Option<String>,
}

/// Safetensors header summary (`GET /v1/models` entry `metadata`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct SafetensorsMetadata {
    /// Tensor entries in the JSON header (`__metadata__` excluded).
    pub tensor_count: u64,
    /// Unique dtype names, sorted (`F32`, `F16`, `BF16`, …).
    pub dtypes: Vec<String>,
}

/// Optional on-disk header metadata, tagged by container format.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
#[serde(tag = "format", rename_all = "snake_case")]
pub enum ModelMetadata {
    /// `.gguf`: llama.cpp container header.
    Gguf(GgufMetadata),
    /// `.safetensors`: `HuggingFace` container header.
    Safetensors(SafetensorsMetadata),
}

/// Feature flags for one model (`GET /v1/models` entry `capabilities`).
///
/// Defaults derive from the entry's `kind` (see [`ModelCapabilities::for_kind`]);
/// an operator `[[models.entry]]` capabilities declaration overrides
/// individual flags. Emitted as a nested object for OpenAI/llama-swap-style
/// consumers — loop-lore reads the same values off the entry root as flat
/// fields, which are emitted alongside (`tool_calling`, `modalities`, …).
// The five-bool shape is the wire contract (llama-swap's nested capabilities
// map); a state machine would change what clients parse, not improve it.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ModelCapabilities {
    /// Chat-template conversation turns (an LLM pipeline can serve it).
    pub chat: bool,
    /// Raw completion; mirrors `chat` (honkbox loads whole turns either way).
    pub completion: bool,
    /// OpenAI-style tool/function calling.
    pub tool_calling: bool,
    /// Vision (`image` content parts).
    pub vision: bool,
    /// Embedding vectors (e.g. an operator-declared embedding GGUF).
    pub embeddings: bool,
}

impl ModelCapabilities {
    /// Inferred defaults for `kind`: LLM entries chat/completion-capable;
    /// on-disk `embedding` weights embedding-capable; everything else
    /// featureless until declared.
    #[must_use]
    pub fn for_kind(kind: ModelKind) -> Self {
        let (chat, embeddings) = match kind {
            ModelKind::Llm => (true, false),
            ModelKind::Embedding => (false, true),
            _ => (false, false),
        };
        Self {
            chat,
            completion: chat,
            tool_calling: false,
            vision: false,
            embeddings,
        }
    }
}

/// One discovered model in the catalog (`GET /v1/models`).
///
/// `id` is the operator alias when one is configured for the file, else the
/// registry slug; requests accept either (the raw slug always works, via
/// `slug`). The loop-lore flat fields (`owned_by`, `context_length`,
/// `max_output`, `thinking`, `tool_calling`, `modalities`) mirror the nested
/// `capabilities` object so both consumer shapes work.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct ModelCatalogEntry {
    /// Operator alias when configured, else the registry slug.
    pub id: String,
    /// Registry slug (`llm:…` / `comfy:…`), always present regardless of `id`.
    pub slug: String,
    /// Modality (capability overrides may reclassify a root, e.g. an
    /// operator-declared embedding GGUF under an LLM root).
    pub kind: ModelKind,
    /// Ownership label; local files are always `"local"`.
    pub owned_by: String,
    /// Training context length from GGUF metadata; `null` when unprobed.
    pub context_length: Option<u64>,
    /// Generation cap; `null` (honkbox has no per-model output ceiling).
    pub max_output: Option<u64>,
    /// Reasoning/thinking support (no honkbox model declares it today).
    pub thinking: bool,
    /// OpenAI-style tool calling; `capabilities.tool_calling` mirrored flat.
    pub tool_calling: bool,
    /// Content modalities (`["text"]`, plus `"vision"` when declared).
    pub modalities: Vec<String>,
    /// Feature flags (also mirrored into the flat fields above).
    pub capabilities: ModelCapabilities,
    /// Directory category (`checkpoints`, `loras`, LLM parent dir, or `root`).
    pub category: String,
    /// Absolute path to the weight file.
    pub path: String,
    /// File size in bytes.
    pub bytes: u64,
    /// Header-derived metadata (GGUF/safetensors); absent when the file has
    /// no probeable header or the header is invalid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ModelMetadata>,
}

/// Model catalog: OpenAI-style list envelope (`GET /v1/models`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct ModelCatalog {
    /// Always `"list"` (OpenAI-compatible envelope).
    pub object: String,
    /// Discovered models, sorted by id.
    pub data: Vec<ModelCatalogEntry>,
}

/// Download modality target (`POST /v1/downloads`): picks the managed
/// `<data_dir>/models` subdir the completed file lands in (and labels the
/// download metrics). The on-disk `ModelKind` set is wider (`lora`,
/// `embedding`, `vae` come from `ComfyUI` category dirs); downloads only
/// target the four root modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum DownloadKind {
    /// Language-model weights (`<data_dir>/models/llm`).
    Llm,
    /// Stable-Diffusion weights (`<data_dir>/models/sd/checkpoints` - the
    /// registry classifies SD `.gguf` only under a checkpoints dir).
    Sd,
    /// Speech-to-text weights (`<data_dir>/models/stt`).
    Stt,
    /// Text-to-speech weights (`<data_dir>/models/tts`).
    Tts,
}

/// Download lifecycle state (`GET /v1/downloads` polling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum DownloadStatus {
    /// Accepted, waiting for a concurrency slot (max 2).
    Queued,
    /// Fetching bytes from scratch.
    Running,
    /// Appending to an existing sibling `.part` via a ranged `GET`.
    Resuming,
    /// File committed to its final path; a catalog rescan was triggered.
    Done,
    /// Terminal failure; the `.part` + meta sidecar are kept for a later
    /// resume (except checksum mismatch, which deletes them).
    Failed,
    /// Cancelled via `DELETE /v1/downloads/{id}`; `.part` + meta deleted.
    Cancelled,
}

/// Start one managed download (`POST /v1/downloads`).
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct DownloadRequest {
    /// `HuggingFace` repo id (`owner/name`); together with `file` it derives
    /// the source URL `https://huggingface.co/{repo}/resolve/main/{file}`
    /// and enables the LFS sha256 checksum check when the API reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// File path inside `repo` (its last segment becomes the local file
    /// name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Direct download URL; the alternative to `repo` + `file`. Exactly one
    /// source form must be present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Target modality (see `DownloadKind`).
    pub kind: DownloadKind,
}

/// 202 acceptance body (`POST /v1/downloads`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct DownloadAccepted {
    /// Download id (`dl_<ulid>`); poll/cancel with it.
    pub id: String,
    /// State at acceptance: `queued` normally, `done` when the target file
    /// already existed (`already_present`), or the in-flight state when an
    /// identical download is already running (single-flight join).
    pub state: DownloadStatus,
}

/// One managed download's progress (`GET /v1/downloads[/id]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct DownloadState {
    /// Download id (`dl_<ulid>`).
    pub id: String,
    /// Lifecycle state.
    pub state: DownloadStatus,
    /// Bytes committed to the `.part` file so far.
    pub bytes_done: u64,
    /// Expected total size; absent until the server reports a length
    /// (and on terminal states where none was ever learned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_total: Option<u64>,
    /// Source URL the bytes are fetched from.
    pub source: String,
    /// Final file path under the managed models root.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_path: Option<String>,
    /// Failure reason (only when `state` is `failed`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Ranged resume is possible for this source: the server supplied a
    /// strong `ETag` (usable as `If-Range` validator) and advertised
    /// `Accept-Ranges: bytes`. `false` means the next pull cold-restarts.
    pub resumable: bool,
    /// `done` because the target file already existed (no bytes fetched).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub already_present: bool,
}

/// Download list envelope (`GET /v1/downloads`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct DownloadList {
    /// Known downloads, sorted by id.
    pub downloads: Vec<DownloadState>,
}

/// Fit-pill verdict for a curated catalog entry against the configured
/// `[models] vram_budget_bytes`. Estimates are advisory heuristics (per-kind
/// overhead factors over weight bytes), never a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Pill {
    /// Estimated VRAM fits the budget (`est <= budget`).
    Ok,
    /// Weights fit but the estimate does not (`bytes <= budget < est`):
    /// loads, likely offloads — slow.
    Slow,
    /// Weights alone exceed the budget (`bytes > budget`).
    Fail,
}

/// Local scan presence for one curated catalog entry (`GET /v1/catalog`);
/// absent when no scanned root holds a matching file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct CatalogLocal {
    /// Matched catalog slug (`llm:…` / `comfy:…` / `stt:…` / `tts:…`).
    pub slug: String,
    /// On-disk weight size in bytes (the pill's byte input).
    pub bytes: u64,
    /// Estimated VRAM need (advisory heuristic, not a measurement); absent
    /// without a configured `[models] vram_budget_bytes`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub est_vram_bytes: Option<u64>,
    /// Fit pill vs the configured budget; absent without one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pill: Option<Pill>,
}

/// One curated catalog offering (`GET /v1/catalog`). The manifest is a
/// checked-in JSON list — review-by-PR is the signature; entries carry a
/// `verified_date` and rot without it (12 months ⇒ drop at release).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct CatalogEntry {
    /// Stable manifest id (e.g. `llama-3.2-1b-q4`); NOT a catalog slug.
    pub id: String,
    /// `HuggingFace` repo id the file downloads from (`owner/name`).
    pub repo: String,
    /// File path inside `repo` (last segment is the local file name).
    pub file: String,
    /// Modality (drives the download target dir and the pill factor).
    pub kind: ModelKind,
    /// Quantization/format tag (e.g. `Q4_K_M`, `fp16`).
    pub quant: String,
    /// Model license tag (e.g. `apache-2.0`, `llama3.2`).
    pub license: String,
    /// Manual verification month (`YYYY-MM`); entries unverified for 12
    /// months are dropped at release.
    pub verified_date: String,
    /// Short human note (why this entry is curated).
    pub notes: String,
    /// Local scan presence + fit pill; absent when not found under the
    /// scanned roots.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local: Option<CatalogLocal>,
}

/// Curated catalog envelope (`GET /v1/catalog`), OpenAI-style list shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct CatalogList {
    /// Always `"list"` (OpenAI-compatible envelope).
    pub object: String,
    /// Curated entries, sorted by id.
    pub data: Vec<CatalogEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regenerate the checked-in TS bindings (drift is gated by `git diff` in CI).
    #[test]
    fn export_ts() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../packages/web-ui/src/gen");
        std::fs::create_dir_all(&dir).expect("create gen dir");
        let config = ts_rs::Config::new();
        let mut out =
            String::from("// AUTO-GENERATED by `cargo test -p honkbox-api-types` — do not edit.\n");
        for decl in [
            Health::decl(&config),
            Version::decl(&config),
            ChatRole::decl(&config),
            ChatMessage::decl(&config),
            ChatRequest::decl(&config),
            ChatResponse::decl(&config),
            TranscriptSegment::decl(&config),
            Transcript::decl(&config),
            SynthesizeRequest::decl(&config),
            ModelKind::decl(&config),
            ModelCapabilities::decl(&config),
            GgufMetadata::decl(&config),
            SafetensorsMetadata::decl(&config),
            ModelMetadata::decl(&config),
            ModelCatalogEntry::decl(&config),
            ModelCatalog::decl(&config),
            DownloadKind::decl(&config),
            DownloadStatus::decl(&config),
            DownloadRequest::decl(&config),
            DownloadAccepted::decl(&config),
            DownloadState::decl(&config),
            DownloadList::decl(&config),
            Pill::decl(&config),
            CatalogLocal::decl(&config),
            CatalogEntry::decl(&config),
            CatalogList::decl(&config),
        ] {
            out.push_str(&decl);
            out.push('\n');
        }
        std::fs::write(dir.join("api-types.ts"), out).expect("write api-types.ts");
    }

    /// Inferred defaults: LLM chats, on-disk embeddings embed, the rest are
    /// featureless until an operator declaration overrides them.
    #[test]
    fn capabilities_default_from_kind() {
        let llm = ModelCapabilities::for_kind(ModelKind::Llm);
        assert!(llm.chat && llm.completion);
        assert!(!llm.tool_calling && !llm.vision && !llm.embeddings);

        let embedding = ModelCapabilities::for_kind(ModelKind::Embedding);
        assert!(embedding.embeddings);
        assert!(!embedding.chat && !embedding.completion);

        for kind in [
            ModelKind::Sd,
            ModelKind::Lora,
            ModelKind::Vae,
            ModelKind::Stt,
            ModelKind::Tts,
        ] {
            let caps = ModelCapabilities::for_kind(kind);
            assert!(!caps.chat && !caps.completion && !caps.embeddings);
        }
    }
}
