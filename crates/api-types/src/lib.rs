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
}

/// Model modality as discovered on disk (`GET /v1/models` entries). Mirrors
/// the registry's on-disk kinds.
#[derive(Debug, Clone, Copy, Serialize, TS, ToSchema)]
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

/// One discovered model in the catalog (`GET /v1/models`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct ModelCatalogEntry {
    /// Registry slug, the stable request-facing id (`llm:…` / `comfy:…`).
    pub id: String,
    /// Modality derived from the scanned root.
    pub kind: ModelKind,
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
            GgufMetadata::decl(&config),
            SafetensorsMetadata::decl(&config),
            ModelMetadata::decl(&config),
            ModelCatalogEntry::decl(&config),
            ModelCatalog::decl(&config),
        ] {
            out.push_str(&decl);
            out.push('\n');
        }
        std::fs::write(dir.join("api-types.ts"), out).expect("write api-types.ts");
    }
}
