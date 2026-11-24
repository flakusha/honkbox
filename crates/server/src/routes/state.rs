//! Shared runtime state: [`AppConfig`] and the router-wide limits that
//! live next to it.
//!
//! Extracted from `routes.rs`; the router registrations stay there.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::http::{Extensions, HeaderMap, StatusCode, Version as HttpVersion, header};
use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{SttPipeline, TtsPipeline};
use honkbox_llm_worker::LlmPipeline;
use honkbox_recognition_worker::RecognitionPipeline;

/// Runtime switch for the UI fallback plus the shared inference pipelines.
///
/// Pipelines are `Clone`-safe (`Arc`): stub unit structs cost nothing, file
/// backends load once at startup and are shared across requests. With the
/// LLM swap manager active (`swap`), chat routes resolve the request slug
/// through it instead of the static `llm` pipeline.
#[derive(Clone)]
pub struct AppConfig {
    /// Serve `web_dist_dir` from disk instead of the embedded bundle.
    pub dev: bool,
    /// Directory served in dev mode (Vite output).
    pub web_dist_dir: PathBuf,
    /// Model catalog from the registry scan: served by `GET /v1/models`,
    /// swapped by `POST /v1/models/refresh`, and used to validate per-request
    /// `model` fields. `RwLock<Arc<_>>` cheap-swap: readers clone the `Arc`
    /// under a short read lock, refresh swaps it wholesale. P1 replaces the
    /// scan snapshot with live load state (swap manager).
    pub catalog: Arc<RwLock<Arc<ModelCatalog>>>,
    /// Registry roots re-scanned by `POST /v1/models/refresh` (startup
    /// configuration; file-level changes are picked up, root config changes
    /// need a restart).
    pub roots: honkbox_registry::ModelRoots,
    /// Startup-configured chat slug (`--llm-model`); accepted as a request
    /// `model` alongside the catalog.
    pub llm_model: Option<String>,
    /// Startup-configured transcription slug (`--stt-model`).
    pub stt_model: Option<String>,
    /// Startup-configured synthesis slug (`--tts-model`).
    pub tts_model: Option<String>,
    /// Shared chat pipeline (`stub-llm` or `llama-cpp-*`).
    pub llm: Arc<dyn LlmPipeline>,
    /// Shared transcription pipeline (`stub-stt` or `whisper-*`).
    pub stt: Arc<dyn SttPipeline>,
    /// Shared synthesis pipeline (`stub-tts` or `kokoro-*`).
    pub tts: Arc<dyn TtsPipeline>,
    /// Startup-configured image pipeline (REQ-10): `stub` (no `--sd-model`)
    /// or `sd-cpp` on the `--sd-model` checkpoint. Shared across requests;
    /// generations acquire the GPU lease per request like every other
    /// GPU route.
    pub sd: Arc<dyn honkbox_sd_core::SdPipeline>,
    /// Startup-configured image slug (`--sd-model`); accepted as a request
    /// `model` on `POST /v1/images/generations` alongside the catalog.
    pub sd_model: Option<String>,
    /// LLM swap manager (P1 of the llama-swap parity plan). When `Some`,
    /// `/v1/chat`(+`/stream`) resolve the request `model` slug through it:
    /// load-on-demand construction, per-entry idle TTL, LRU capacity. `None`
    /// keeps the startup-loaded `llm` pipeline (stub backends, or
    /// `[swap] enabled = false`). 429 semantics are unchanged either way —
    /// the residency slot stays the only inference gate.
    pub swap: Option<Arc<crate::swap::SwapManager>>,
    /// POST /v1/embeddings pipeline cache (REQ-4): load-once per slug with
    /// idle-TTL reuse, capacity 1 (see [`crate::swap::EmbedCache`] for why
    /// this is not the chat swap manager). Deliberately absent from
    /// `/running` + `/v1/memory`: embedding pipelines are cheap and
    /// single-slot; the residency readout stays the chat warm set.
    pub embed: Arc<crate::swap::EmbedCache>,
    /// Ceiling on per-request `ttl_seconds` advice (`[swap] max_ttl_secs`):
    /// over-cap values are clamped here with a `warn!` — never rejected.
    /// Known even when the swap manager is off, so every route applies the
    /// same policy to the advice field.
    pub swap_max_ttl: Duration,
    /// Resolved data dir; supplies the registry-cache DB default
    /// (`<data_dir>/registry.sqlite3`) on refresh. `None` disables the
    /// cache default (explicit `HONKBOX_REGISTRY_DB` still wins).
    pub data_dir: Option<PathBuf>,
    /// Configured VRAM budget (`[models] vram_budget_bytes` / env): drives
    /// the curated catalog's advisory fit pills; `None` omits them.
    pub vram_budget: Option<u64>,
    /// Operator model bindings (`[[models.entry]]`): rebound against every
    /// fresh scan (startup, `POST /v1/models/refresh`, post-download rescan).
    pub model_entries: Vec<crate::config::ModelEntryConfig>,
    /// Managed download supervisor (`/v1/downloads`): fetches model files
    /// into `<data_dir>/models/{llm,sd,stt,tts}` and fires the rescan
    /// signal on every commit.
    pub downloads: Arc<crate::downloads::DownloadManager>,
    /// Binary artifact store (`/v1/artifacts`, TASK-binary-artifact-api):
    /// content-addressed blobs under `<data_dir>/artifacts` with the
    /// resumable-chunk upload registry. Degrades honestly when there is no
    /// data dir (uploads 500, downloads 404).
    pub artifacts: Arc<crate::artifacts::Registry>,
    /// Auth settings (REQ-7): `[server] auth` mode + `[auth] api_key_env`
    /// NAME. The middleware reads the key VALUE per request via `std::env`
    /// (rotation without restart); the value is never stored or logged.
    pub auth: crate::config::AuthConfig,
    /// Shared recognition pipeline (recognition tickets): `stub-recognition`
    /// or the rxing barcode backend (`barcode` feature). Stage 1 is
    /// synchronous `/v1/recognize` for barcode-class latency; the OCR
    /// document pipeline becomes job-based when FEAT-ocr-ingest lands.
    pub recognition: Arc<dyn RecognitionPipeline>,
}

impl AppConfig {
    /// Current catalog snapshot: a cheap `Arc` clone under a short read lock.
    pub(super) fn catalog_snapshot(&self) -> Arc<ModelCatalog> {
        self.catalog
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

/// Deliberate request-body ceiling for the whole router (H1, ReviewedAudio):
/// axum's default 2 MiB caps audio uploads at ~65 s of 16 kHz mono s16
/// (48000 Hz mono s16 would be ~22 s), which breaks any real recording.
/// 64 MiB = 10 minutes of 48 kHz mono s16le (`48_000 * 2 * 600 = 57.6 MB`)
/// with headroom for multipart framing and the octet-stream RIFF sniff —
/// sized for the largest transcription input the whisper path plausibly
/// serves, still far below any memory concern for a local server. Applies
/// to every route; JSON DTOs are tiny, so the limit is only load-bearing
/// on `/v1/audio/transcriptions` and `/v1/transcribe`. `pub(crate)`: the
/// capability advertisement reports it (`formats.limits.body_bytes`) and
/// the contract test cross-checks the two against each other.
pub(crate) const AUDIO_BODY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

/// Response-compression size floor (FEAT-response-compression; the ticket
/// names no number, this is the pick): bodies SMALLER than this are passed
/// through uncompressed — a sub-1 KiB JSON costs more codec time than the
/// bytes it saves and gzip of tiny bodies can even grow them. 1 KiB is the
/// common proxy default. Payloads at/above it compress when the client asks
/// (`Accept-Encoding: gzip` / `br`) and decompress byte-identical. (`u16`:
/// `SizeAbove`'s field width.)
pub(crate) const COMPRESSION_MIN_BYTES: u16 = 1024;

/// `POST /v1/audio/speech` `input` cap (TASK-speech-input-cap,
/// blocking-grade finding): CHARACTERS, not bytes — 4096, the `OpenAI` TTS
/// precedent. Larger `input` is a named 400 (`input_too_long`), never a
/// silent truncation: reject-not-clamp, same policy as `cfg_scale` and
/// `strength`. Under-cap requests are byte-identical to the pre-ticket
/// behavior.
pub(crate) const SPEECH_INPUT_MAX_CHARS: usize = 4096;

/// TASK-binary-artifact-api: a partial-content response (any response
/// carrying `Content-Range`) must NEVER travel through the compression
/// layer — a compressed 206 corrupts `Content-Range` math and breaks resume
/// offsets, and the artifacts bytes route serves exactly those. This is the
/// same predicate-composition mechanism that excludes SSE by content type
/// (`DefaultPredicate`'s `NotForContentType` arm); it keys on the
/// `Content-Range` header instead so arbitrary artifact content types are
/// covered, and plain 200 downloads still compress when negotiated.
#[allow(dead_code)] // axum compression predicate fixture; salvo compression is in salvo_middleware
pub(crate) fn partial_content_stays_plain(
    _: StatusCode,
    _: HttpVersion,
    headers: &HeaderMap,
    _: &Extensions,
) -> bool {
    !headers.contains_key(header::CONTENT_RANGE)
}
