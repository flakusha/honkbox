//! HTTP surface: `/health`, `/api/version`, `/v1/models`(+`/refresh`),
//! `/metrics`, `/v1/openapi.json`, chat/transcribe/synthesize, image
//! generations (REQ-10, incl. the `ControlNet` conditioned render),
//! `/v1/jobs` (+`DELETE /v1/jobs/{id}`, `POST …/cancel`) and the
//! `/v1/events` SSE lifecycle feed, UI fallback.
//!
//! The `OpenAPI` document is served as plain JSON via `Json` — no
//! `utoipa-axum` router glue.

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Query, Request, State},
    http::{Extensions, HeaderMap, HeaderValue, StatusCode, Version as HttpVersion, header},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{delete, get, post},
};
use base64::Engine as _;
use honkbox_api_types::{
    ArtifactChunk, ArtifactInfo, ArtifactManifest, AuthCapabilities, Capabilities, CatalogEntry,
    CatalogList, CatalogLocal, ChatContent, ChatMessage, ChatRequest, ChatResponse, ChatRole,
    ChatTool, CompressionRange, ControlNetFeature, DeltaFunction, DeltaToolCall, DownloadAccepted,
    DownloadKind, DownloadList, DownloadRequest, DownloadState, DownloadStatus, EmbeddingsData,
    EmbeddingsInput, EmbeddingsRequest, EmbeddingsResponse, EmbeddingsUsage, ExternalTool,
    ExternalToolsCapability, FeatureFlags, FinishReason, FormatLimits, Formats, FunctionCall,
    FunctionDefinition, GgufMetadata, Health, ImageB64, ImageOpArtifact, ImageOpCommand,
    ImageOpCrop, ImageOpInput, ImageOpKind, ImageOpOutput, ImageOpRequest, ImageOpResize,
    ImageOpResponse, ImageOpRotate, ImageOpsCapability, ImagesGenerationRequest,
    ImagesGenerationResponse, ImagesHiresFix, ImagesLora, JobCancelResult, JobEvent, JobEventType,
    JobInfo, JobList, JsonSchemaSpec, MemoryCapacity, MemoryGpu, MemoryInfo, ModelCapabilities,
    ModelCatalog, ModelCatalogEntry, ModelDeleteResult, ModelKind, ModelMetadata, NamedFunction,
    OpenAiChatCompletion, OpenAiChatRequest, OpenAiChoice, OpenAiChunk, OpenAiChunkChoice,
    OpenAiDelta, OpenAiError, OpenAiUsage, OutputFormatCapability, Pill, QueueCapabilities,
    QueueKindCapacity, RecKind, RecognitionEngineMeta, RecognitionFeature, RecognitionPage,
    RecognitionRegion, RecognizeRequest, RecognizeResult, ResidentModel, ResponseFormat,
    ResponseFormatType, RunningList, RunningModel, SafetensorsMetadata, ServerInfo, SpeechRequest,
    SynthesizeRequest, ToolCall, ToolCallType, ToolCallingCapability, ToolChoiceSpec, ToolType,
    Transcript, TranscriptSegment, TranscriptionText, TranscriptionVerbose, UnloadAllResult,
    UnloadResult, Version, VisionFeature, Voice, VoiceList,
};
use sha2::{Digest as _, Sha256};

/// Re-exported auth settings (REQ-7): `app` consumers (integration tests,
/// `main`) build `AppConfig.auth` from these.
#[allow(unused_imports)] // which name is used varies per #[path]-included test binary
pub use crate::config::{AuthConfig, AuthMode};
use honkbox_audio_worker::{
    SttPipeline, SynthesizeRequest as WorkerSynthRequest, SynthesizedAudio,
    TranscribeRequest as WorkerTranscribeRequest, Transcript as WorkerTranscript,
    TranscriptSegment as WorkerTranscriptSegment, TtsPipeline, decode_wav, encode_wav,
};
use honkbox_llm_worker::{
    ChatImage as WorkerChatImage, ChatMessage as WorkerChatMessage,
    ChatRequest as WorkerChatRequest, LlmError, LlmPipeline, Role, SamplerParams, ToolCallOut,
    ToolCallSpec,
};
use honkbox_recognition_worker::{
    RecError, RecKind as WorkerRecKind, RecognitionPipeline,
    RecognizeRequest as WorkerRecognizeRequest,
};
use rust_embed::Embed;
use serde::Deserialize;
use tokio::sync::{OnceCell, mpsc};
use tokio_stream::{
    Stream, StreamExt as _,
    wrappers::{BroadcastStream, UnboundedReceiverStream, errors::BroadcastStreamRecvError},
};
use tower_http::compression::predicate::SizeAbove;
use tower_http::compression::{CompressionLayer, DefaultPredicate, Predicate as _};
use utoipa::OpenApi;

#[path = "routes/external_tools.rs"]
mod external_tools;

#[allow(unused_imports)] // main.rs is the only caller; #[path] test harnesses omit it
pub(crate) use self::external_tools::init_external_tools;
#[allow(unused_imports)] // referenced by the OpenApi derive's generated types
use self::external_tools::{
    __path_get_external_tool, __path_get_external_tools, __path_post_external_tools_reload,
    get_external_tool, get_external_tools, post_external_tools_reload,
};

#[path = "routes/health.rs"]
mod health;

#[path = "routes/capabilities.rs"]
mod capabilities;

#[path = "routes/models.rs"]
mod models;

#[path = "routes/runtime.rs"]
mod runtime;

#[allow(unused_imports)]
use self::capabilities::{__path_get_capabilities, get_capabilities};
#[allow(unused_imports)] // used by capability contract tests via routes::
pub use self::capabilities::{ADMITTING_ROUTES, admitting_route};
#[allow(unused_imports)]
// handler names are used by .route() and the OpenApi derive's generated types
use self::health::{
    __path_get_version, __path_health, __path_metrics_endpoint, __path_openapi_json, get_version,
    health, metrics_endpoint, openapi_json,
};
#[allow(unused_imports)]
use self::models::{
    __path_delete_model, __path_get_models, __path_post_model_unload, __path_refresh_models,
    delete_model, get_models, post_model_unload, refresh_models,
};
#[allow(unused_imports)]
use self::runtime::{
    __path_get_memory, __path_get_running, __path_post_unload_all, __path_post_unload_one,
    get_memory, get_running, post_unload_all, post_unload_one,
};

#[path = "routes/catalog.rs"]
mod catalog;

#[path = "routes/bindings.rs"]
mod bindings;

#[path = "routes/downloads.rs"]
mod downloads;

#[path = "routes/ui.rs"]
mod ui;

#[allow(unused_imports)] // main.rs and in-file tests use these via routes::
pub use self::bindings::{ModelBindings, bind_entries, build_catalog};
#[allow(unused_imports)] // main.rs and in-file tests use these via routes::
pub use self::catalog::{refresh_catalog, rescan_catalog, rescan_on_downloads};
#[allow(unused_imports)]
use self::downloads::{
    __path_delete_download, __path_get_download, __path_list_downloads, __path_post_download,
    delete_download, get_download, list_downloads, post_download,
};
#[allow(unused_imports)] // shared by later route slices and the UI fallback registration
pub(super) use self::ui::{fallback, resolve_model_slug, validate_model};

/// Compile-time UI bundle (release always embeds; dev serves from disk instead).
#[derive(Embed)]
#[folder = "../../web-dist/"]
struct EmbeddedUi;

/// `OpenAPI` document for the whole HTTP surface.
#[derive(OpenApi)]
#[openapi(
    paths(
        health,
        get_version,
        get_capabilities,
        get_models,
        refresh_models,
        post_model_unload,
        get_memory,
        openapi_json,
        metrics_endpoint,
        post_chat,
        post_chat_stream,
        post_chat_completions,
        post_images_generations,
        post_images_edits,
        post_image_ops,
        post_transcribe,
        post_synthesize,
        post_audio_transcriptions,
        post_audio_translations,
        post_audio_speech,
        get_audio_voices,
        get_jobs,
        post_job_cancel,
        delete_job,
        get_events,
        delete_model,
        post_download,
        list_downloads,
        get_download,
        delete_download,
        get_catalog,
        get_running,
        post_unload_all,
        post_unload_one,
        post_embeddings,
        post_recognize,
        get_external_tools,
        get_external_tool,
        post_external_tools_reload,
        crate::artifacts::post_artifact,
        crate::artifacts::get_artifact_bytes
    ),
    components(schemas(
        Health,
        Version,
        Capabilities,
        ServerInfo,
        FeatureFlags,
        VisionFeature,
        ControlNetFeature,
        Formats,
        OutputFormatCapability,
        FormatLimits,
        CompressionRange,
        AuthCapabilities,
        QueueCapabilities,
        QueueKindCapacity,
        ChatMessage,
        ChatRequest,
        ChatResponse,
        ChatRole,
        Transcript,
        TranscriptSegment,
        SynthesizeRequest,
        SpeechRequest,
        TranscriptionText,
        TranscriptionVerbose,
        ModelKind,
        ModelCapabilities,
        GgufMetadata,
        SafetensorsMetadata,
        ModelMetadata,
        ModelCatalogEntry,
        ModelCatalog,
        UnloadResult,
        UnloadAllResult,
        ModelDeleteResult,
        Voice,
        VoiceList,
        ResidentModel,
        MemoryCapacity,
        MemoryGpu,
        MemoryInfo,
        RunningModel,
        RunningList,
        OpenAiChatRequest,
        ToolCall,
        ToolCallType,
        FunctionCall,
        ChatTool,
        ToolType,
        FunctionDefinition,
        ToolChoiceSpec,
        ToolCallingCapability,
        ExternalTool,
        ExternalToolsCapability,
        NamedFunction,
        ResponseFormat,
        ResponseFormatType,
        JsonSchemaSpec,
        FinishReason,
        OpenAiUsage,
        OpenAiChoice,
        OpenAiChatCompletion,
        OpenAiDelta,
        DeltaToolCall,
        DeltaFunction,
        OpenAiChunkChoice,
        OpenAiChunk,
        OpenAiError,
        EmbeddingsRequest,
        EmbeddingsInput,
        EmbeddingsData,
        EmbeddingsUsage,
        EmbeddingsResponse,
        ImagesGenerationRequest,
        ImagesLora,
        ImagesHiresFix,
        ImageB64,
        ImagesGenerationResponse,
        ImageOpKind,
        ImageOpCommand,
        ImageOpResize,
        ImageOpCrop,
        ImageOpRotate,
        ImageOpInput,
        ImageOpArtifact,
        ImageOpRequest,
        ImageOpOutput,
        ImageOpResponse,
        ImageOpsCapability,
        RecKind,
        RecognizeRequest,
        RecognitionRegion,
        RecognitionPage,
        RecognitionEngineMeta,
        RecognizeResult,
        RecognitionFeature,
        JobInfo,
        JobList,
        JobCancelResult,
        JobEventType,
        JobEvent,
        DownloadKind,
        DownloadStatus,
        DownloadRequest,
        DownloadAccepted,
        DownloadState,
        DownloadList,
        Pill,
        CatalogLocal,
        CatalogEntry,
        CatalogList,
        ArtifactChunk,
        ArtifactInfo,
        ArtifactManifest
    ))
)]
pub struct ApiDoc;

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
    fn catalog_snapshot(&self) -> Arc<ModelCatalog> {
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
const COMPRESSION_MIN_BYTES: u16 = 1024;

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
fn partial_content_stays_plain(
    _: StatusCode,
    _: HttpVersion,
    headers: &HeaderMap,
    _: &Extensions,
) -> bool {
    !headers.contains_key(header::CONTENT_RANGE)
}

/// Build the full router.
pub fn app(config: AppConfig) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/version", get(get_version))
        .route("/v1/capabilities", get(get_capabilities))
        .route("/v1/models", get(get_models))
        .route("/v1/models/refresh", post(refresh_models))
        .route("/v1/models/{slug}/unload", post(post_model_unload))
        .route("/v1/models/{slug}", delete(delete_model))
        .route("/v1/catalog", get(get_catalog))
        .route("/v1/memory", get(get_memory))
        .route("/v1/openapi.json", get(openapi_json))
        .route(
            "/v1/external-tools",
            get(external_tools::get_external_tools),
        )
        .route(
            "/v1/external-tools/reload",
            post(external_tools::post_external_tools_reload),
        )
        .route(
            "/v1/external-tools/{id}",
            get(external_tools::get_external_tool),
        )
        .route("/v1/chat", post(post_chat))
        .route("/v1/chat/stream", post(post_chat_stream))
        .route("/v1/chat/completions", post(post_chat_completions))
        .route("/v1/embeddings", post(post_embeddings))
        .route("/v1/images/generations", post(post_images_generations))
        .route("/v1/images/edits", post(post_images_edits))
        .route("/v1/image-ops", post(post_image_ops))
        .route("/v1/recognize", post(post_recognize))
        .route("/v1/transcribe", post(post_transcribe))
        .route("/v1/synthesize", post(post_synthesize))
        .route("/v1/audio/voices", get(get_audio_voices))
        .route("/v1/audio/transcriptions", post(post_audio_transcriptions))
        .route("/v1/audio/translations", post(post_audio_translations))
        .route("/v1/audio/speech", post(post_audio_speech))
        .route("/v1/jobs", get(get_jobs))
        .route("/v1/jobs/{id}/cancel", post(post_job_cancel))
        .route("/v1/jobs/{id}", delete(delete_job))
        .route("/v1/events", get(get_events))
        .route("/running", get(get_running))
        .route("/api/models/unload", post(post_unload_all))
        .route("/api/models/unload/{model}", post(post_unload_one))
        .route("/v1/downloads", post(post_download).get(list_downloads))
        .route(
            "/v1/downloads/{id}",
            get(get_download).delete(delete_download),
        )
        .route(
            "/v1/artifacts",
            // TASK-binary-artifact-api: bodies stream straight to disk under
            // the chunk/artifact caps, so the POST route raises the router's
            // 64 MiB audio ceiling to the configured total cap (plus per-
            // chunk framing headroom) instead of rejecting large uploads
            // before the streaming caps can name their 413s.
            post(crate::artifacts::post_artifact).layer(DefaultBodyLimit::max(
                usize::try_from(
                    config
                        .artifacts
                        .max_artifact_bytes()
                        .saturating_add(crate::artifacts::CHUNK_MAX_BYTES),
                )
                .unwrap_or(usize::MAX),
            )),
        )
        .route(
            "/v1/artifacts/{id}/bytes",
            get(crate::artifacts::get_artifact_bytes),
        )
        .route("/metrics", get(metrics_endpoint))
        .route_layer(middleware::from_fn_with_state(
            config.clone(),
            require_bearer,
        ))
        .fallback(fallback)
        // H1 (ReviewedAudio): lift the silent 2 MiB default; see
        // [`AUDIO_BODY_LIMIT_BYTES`] for the byte math.
        .layer(DefaultBodyLimit::max(AUDIO_BODY_LIMIT_BYTES))
        // FEAT-response-compression: gzip/brotli on ordinary JSON/binary
        // routes; the predicate is DefaultPredicate's exclusion set (gRPC,
        // `image/*`, `text/event-stream` — BOTH SSE routes stay unbuffered)
        // with a 1 KiB size floor. Mounted INSIDE `track_metrics` so
        // status/`Retry-After` bookkeeping still sees the final response.
        .layer(
            CompressionLayer::new().compress_when(
                DefaultPredicate::new()
                    .and(SizeAbove::new(COMPRESSION_MIN_BYTES))
                    .and(partial_content_stays_plain),
            ),
        )
        .layer(middleware::from_fn(track_metrics))
        .with_state(config)
}

/// Bearer auth middleware (REQ-7). `route_layer` runs ONLY for matched
/// routes, so the UI/static fallback (and its assets) is structurally
/// exempt — the web UI keeps working without a key. `GET /health` and
/// `GET /metrics` are exempt too (probes/scrapers). When
/// `[server] auth = "bearer"`, every other route requires
/// `Authorization: Bearer <key>`; the key VALUE is read from the
/// `[auth] api_key_env` env var AT REQUEST TIME (`std::env` is
/// process-global, safe in async; rotation works without a restart) and is
/// never logged, never in metrics, never echoed in responses. Missing/
/// unset env is fail-closed: every bearer request gets 401.
async fn require_bearer(State(cfg): State<AppConfig>, req: Request, next: Next) -> Response {
    if cfg.auth.mode != crate::config::AuthMode::Bearer
        || is_probe_exempt(req.method(), req.uri().path())
    {
        return next.run(req).await;
    }
    let expected = env_var(&cfg.auth.api_key_env).filter(|key| !key.is_empty());
    let provided = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            v.strip_prefix("Bearer ")
                .or_else(|| v.strip_prefix("bearer "))
        })
        .map(str::trim)
        .filter(|key| !key.is_empty());
    let ok = matches!((expected, provided), (Some(e), Some(p)) if constant_time_eq(e.as_bytes(), p.as_bytes()));
    if ok {
        return next.run(req).await;
    }
    unauthorized_response()
}

/// Keyless routes: the liveness/metrics probes, GET only (so a scraper
/// cannot write anything and probes stay cheap).
fn is_probe_exempt(method: &axum::http::Method, path: &str) -> bool {
    method == axum::http::Method::GET && (path == "/health" || path == "/metrics")
}

/// Env read for the bearer key VALUE (REQ-7). Production: `std::env`, per
/// request, so rotation needs no restart. Test builds (this module is
/// `#[path]`-included into integration tests that run with
/// `forbid(unsafe_code)`): [`set_env_var`] injects overrides instead of
/// mutating the process-global env; the real env is still consulted first
/// when the name is not overridden.
#[cfg(not(test))]
fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

#[cfg(test)]
static ENV_OVERRIDES: std::sync::Mutex<Option<std::collections::HashMap<String, String>>> =
    std::sync::Mutex::new(None);

/// Test-only env override (`Some` sets, `None` clears); see [`env_var`].
/// Only the `#[path]`-included integration tests call it, so the unit-test
/// build of this module sees it as dead code.
#[cfg(test)]
#[allow(dead_code)]
pub fn set_env_var(name: &str, value: Option<&str>) {
    let mut slot = ENV_OVERRIDES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let map = slot.get_or_insert_with(Default::default);
    match value {
        Some(v) => {
            map.insert(name.to_owned(), v.to_owned());
        }
        None => {
            map.remove(name);
        }
    }
}

#[cfg(test)]
fn env_var(name: &str) -> Option<String> {
    let slot = ENV_OVERRIDES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    slot.as_ref()
        .and_then(|map| map.get(name).cloned())
        .or_else(|| std::env::var(name).ok())
}

/// Hand-rolled constant-time byte comparison (REQ-7: no new dependency).
/// Double-pass over the longer slice with zero-fill for the shorter one, so
/// the work done does not depend on where (or whether) the inputs differ;
/// the length XOR folds any length mismatch into the same accumulator.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut acc = (a.len() ^ b.len()) as u64;
    for i in 0..a.len().max(b.len()) {
        let x = u64::from(a.get(i).copied().unwrap_or(0));
        let y = u64::from(b.get(i).copied().unwrap_or(0));
        acc |= x ^ y;
    }
    acc == 0
}

/// 401 with the OpenAI-shaped error envelope. Built inline via `json!` —
/// `OpenAiError::new` maps non-400/404/429/5xx classes to `server_error`,
/// and auth needs `authentication_error` + `invalid_api_key`; api-types is
/// shared DTO territory we do not extend here.
fn unauthorized_response() -> Response {
    let body = serde_json::json!({
        "error": {
            "message": "Invalid API key provided.",
            "type": "authentication_error",
            "param": null,
            "code": "invalid_api_key"
        }
    });
    (StatusCode::UNAUTHORIZED, Json(body)).into_response()
}

/// Per-request metrics: one route/code counter + one duration observation
/// (see `metrics`); no per-request allocation. Every 429 without an
/// explicit `Retry-After` (the GPU-slot busy contract, shared by the native
/// and `OpenAI` chat routes) carries a 1s hint — additive on the native
/// route, whose body/status do not change. The admission queue's own
/// rejections rely on this default (the queue has no better estimate than
/// the operator's retry cadence), so an explicit header is never clobbered.
async fn track_metrics(req: Request, next: Next) -> Response {
    let route = crate::metrics::route_label(req.uri().path());
    let started = Instant::now();
    let mut response = next.run(req).await;
    if response.status() == StatusCode::TOO_MANY_REQUESTS
        && !response.headers().contains_key(header::RETRY_AFTER)
    {
        response.headers_mut().insert(
            header::RETRY_AFTER,
            crate::admission::retry_after_header_value(),
        );
    }
    crate::metrics::record_request(route, response.status().as_u16(), started.elapsed());
    response
}

/// `GET /v1/catalog`: the curated manifest merged with the local scan
/// snapshot. Match by (kind, file name); fit pills (`ok`/`slow`/`fail`)
/// compute only from the matched file's bytes against
/// `[models] vram_budget_bytes` — without a budget the advisory fields are
/// omitted and entries show manifest metadata + local bytes only. Pills
/// are advice, never enforcement.
#[utoipa::path(
    get,
    path = "/v1/catalog",
    responses(
        (status = 200, description = "Curated catalog with local presence + fit pills", body = CatalogList),
    )
)]
async fn get_catalog(State(config): State<AppConfig>) -> Response {
    let snapshot = config.catalog_snapshot();
    let list = crate::catalog::build(&snapshot, config.vram_budget);
    match serde_json::to_string(&list) {
        Ok(body) => ([(header::CONTENT_TYPE, "application/json")], body).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("catalog serialize failed: {err}"),
        )
            .into_response(),
    }
}

/// `POST /v1/transcribe` query: model slug + language for the worker request;
/// `sample_rate` is required only for raw `audio/pcm-s16le` bodies.
/// `ttl_seconds` is per-request residency advice (NEW-8): clamped at this
/// route, consumed once per-kind swap actors land (audio pipelines are
/// startup-loaded today).
#[derive(Debug, Deserialize)]
struct TranscribeQuery {
    model: Option<String>,
    language: Option<String>,
    sample_rate: Option<u32>,
    ttl_seconds: Option<u64>,
}

/// `api-types` message → worker message (`REQ-3`): the tool role,
/// `tool_call_id` and assistant `tool_calls` round-trip; missing/null
/// content becomes the empty string (assistant-with-tool_calls shape).
/// Media marker the projector's tokenizer splices embeddings at (the
/// llama.cpp mtmd default, `MtmdContext::default_marker()`). The server
/// splices it into message text at each `image_url` part's flattened-text
/// offset before the worker call, so image embeddings land where the
/// client put the parts.
const MEDIA_MARKER: &str = "<__media__>";

/// Decode one `image_url` payload: a `data:<mime>;base64,<payload>` URI.
/// The bytes stay encoded (JPEG/PNG/…) — the projector's bitmap loader
/// sniffs the container format. A non-base64 `data:` URI cannot carry
/// image bytes; anything else is the [`image_data_rejection`] case.
fn decode_image_url(url: &str) -> Result<Vec<u8>, String> {
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(image_data_rejection(url).unwrap_or_else(|| "invalid image_url".to_owned()));
    };
    let Some((meta, payload)) = rest.split_once(',') else {
        return Err(format!(
            "malformed data: URI {url:?}: missing `,payload` section"
        ));
    };
    if meta.rsplit(';').next().is_none_or(|flag| flag != "base64") {
        return Err(format!(
            "unsupported data: URI {url:?}: image payloads must be base64 (`data:<mime>;base64,...`)"
        ));
    }
    // Standard with padding is the norm; tolerate the no-pad and URL-safe
    // variants rather than teaching clients the alphabet rules.
    for engine in [
        &base64::engine::general_purpose::STANDARD,
        &base64::engine::general_purpose::STANDARD_NO_PAD,
        &base64::engine::general_purpose::URL_SAFE,
        &base64::engine::general_purpose::URL_SAFE_NO_PAD,
    ] {
        if let Ok(bytes) = base64::engine::Engine::decode(engine, payload.trim()) {
            if bytes.is_empty() {
                break;
            }
            return Ok(bytes);
        }
    }
    Err("invalid image data URL: base64 payload does not decode (or is empty)".to_owned())
}

/// One API message → worker message. Vision messages get the media
/// marker spliced at their part offsets and the `data:` payloads decoded
/// to bytes; a bad payload is a client error the routes map to 400.
fn worker_message(message: &ChatMessage) -> Result<WorkerChatMessage, String> {
    let role = match message.role {
        ChatRole::User => Role::User,
        ChatRole::Assistant => Role::Assistant,
        ChatRole::System => Role::System,
        ChatRole::Tool => Role::Tool,
    };
    let content = if message.content.images.is_empty() {
        message.content.text.clone().unwrap_or_default()
    } else {
        message.content.text_with_media_markers(MEDIA_MARKER)
    };
    let mut images = Vec::with_capacity(message.content.images.len());
    for image in &message.content.images {
        images.push(WorkerChatImage::new(decode_image_url(&image.url)?));
    }
    Ok(WorkerChatMessage {
        role,
        content,
        tool_call_id: message.tool_call_id.clone(),
        tool_calls: message
            .tool_calls
            .iter()
            .flatten()
            .map(|tc| ToolCallSpec {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                arguments: tc.function.arguments.clone(),
            })
            .collect(),
        images,
    })
}

/// All API messages → worker messages (vision decoding can fail per
/// message).
fn worker_messages(messages: &[ChatMessage]) -> Result<Vec<WorkerChatMessage>, String> {
    messages.iter().map(worker_message).collect()
}

/// Chat-capability refusal message (shared by the completions envelope and
/// the native plain-text surface).
fn chat_unsupported_message(model: &str) -> String {
    format!(
        "model {model:?} is not chat-capable: its catalog entry declares capabilities.chat = false (a mmproj projector file or non-chat weights cannot serve conversation turns)"
    )
}

/// Worker request skeleton shared by both routes: constraint fields absent
/// (`REQ-3` fields ride only on `/v1/chat/completions`), default sampler.
/// Message conversion (vision decoding) can fail per message — the caller
/// maps that to its surface's 400.
fn worker_request_base(
    slug: String,
    messages: Result<Vec<WorkerChatMessage>, String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<WorkerChatRequest, String> {
    Ok(WorkerChatRequest {
        model: slug,
        messages: messages?,
        max_tokens,
        temperature,
        seed: None,
        grammar: None,
        json_schema: None,
        json_object: false,
        tools_json: None,
        tool_choice_none: false,
        sampler: SamplerParams::default(),
    })
}

/// `X-Job-Id` response-header name (NEW-3): every GPU route stamps its
/// in-flight registry id. On SSE routes it precedes the stream body (a
/// client can cancel the stream it is reading); on sync routes it arrives
/// with the completed response, so cancel-discovery goes through
/// `GET /v1/jobs`.
const JOB_ID_HEADER: header::HeaderName = header::HeaderName::from_static("x-job-id");

/// Attach the job id header to a built response.
fn with_job_id(mut response: Response, job_id: &str) -> Response {
    if let Ok(value) = HeaderValue::from_str(job_id) {
        response.headers_mut().insert(JOB_ID_HEADER, value);
    }
    response
}

/// Admission first, registration second: a rejected GPU request must not
/// enter the registry (no `job_registered`/`job_finished` events and no
/// `X-Job-Id` on a request that never ran). Every admitted route goes
/// through this one seam so the ordering cannot drift between them.
fn admit_then_register(
    registry: &crate::jobs_registry::Registry,
    acquired: Result<crate::admission::GpuLease, (StatusCode, String)>,
    kind: crate::jobs_registry::JobKind,
    model: Option<String>,
    interruptible: bool,
) -> Result<(crate::admission::GpuLease, crate::jobs_registry::JobGuard), (StatusCode, String)> {
    let lease = acquired?;
    let job = registry.register(kind, model, interruptible);
    Ok((lease, job))
}

/// REQ-8: image content on a model without a projector. Vision capability
/// is catalog-derived: a sibling mmproj projector next to the weights, or
/// an explicit operator `capabilities.vision = true`.
fn vision_unsupported_message(model: &str) -> String {
    format!(
        "model {model:?} is not vision-capable: image_url parts need an mmproj projector next to the model weights (or an operator capabilities.vision override); see GET /v1/models modalities"
    )
}

/// Only `data:` URI image payloads are accepted: the server decodes the
/// base64 inline and hands the bytes to the projector's own image loader.
/// Remote URLs would turn the server into an outbound fetcher (SSRF
/// surface) — an explicit rejection, never a silent skip or a fetch.
fn image_data_rejection(url: &str) -> Option<String> {
    (!url.starts_with("data:")).then(|| {
        format!(
            "unsupported image_url {url:?}: only data: URIs are accepted (remote URLs are not fetched)"
        )
    })
}

/// Vision gate over chat requests (all three chat routes): images in the
/// conversation need (a) a vision-capable model and (b) `data:` URI
/// payloads. `entry` is the resolved catalog entry (`None` — e.g. the raw
/// startup-configured model — is treated as not vision-capable). `None`
/// passes; `Some((message, code))` is the 400 payload (`code` names the
/// error on the `OpenAI` surface). The worker re-gates (defense in depth).
fn vision_rejection(
    entry: Option<&ModelCatalogEntry>,
    messages: &[ChatMessage],
) -> Option<(String, &'static str)> {
    if !messages.iter().any(|message| message.content.has_image) {
        return None;
    }
    let model = entry.map_or_else(|| "?".to_owned(), |entry| entry.id.clone());
    if !entry.is_some_and(|entry| entry.capabilities.vision) {
        return Some((vision_unsupported_message(&model), "vision_unsupported"));
    }
    for message in messages {
        for image in &message.content.images {
            if let Some(rejection) = image_data_rejection(&image.url) {
                return Some((rejection, "image_url_unsupported"));
            }
        }
    }
    None
}

/// Chat-capability gate (completions surface): a catalog entry advertising
/// `capabilities.chat = false` — a projector `mmproj*.gguf` file, an
/// operator-declared embedding, or any non-chat weights — is never silently
/// run through the chat pipeline (garbage tokens dressed up as a
/// completion). Mirrors the REQ-4 embeddings gate; `None` entry (the raw
/// startup-configured model) passes, the operator declared it by hand.
fn chat_capable_gate(entry: &ModelCatalogEntry) -> Option<Response> {
    (!entry.capabilities.chat).then(|| {
        openai_error_response(
            StatusCode::BAD_REQUEST,
            chat_unsupported_message(&entry.id),
            Some("chat_unsupported"),
        )
    })
}

/// Request gates before any pipeline work (`/v1/chat/completions`): REQ-4
/// — an embeddings-capable model is never silently run through the chat
/// pipeline (the reply would be garbage tokens dressed up as a
/// completion); chat-capability (a projector file or operator-declared
/// non-chat weights) is refused by name for the same reason; REQ-8 — an
/// `image_url` content part is gated on model vision capability and
/// `data:` URI payloads, an explicit 400 `OpenAI` envelope either way.
/// `Some(response)` is the ready-made rejection.
fn completions_request_gates(
    entry: &ModelCatalogEntry,
    model: &str,
    messages: &[ChatMessage],
) -> Option<Response> {
    embeddings_model_gate(entry, model)
        .or_else(|| chat_capable_gate(entry))
        .or_else(|| {
            vision_rejection(Some(entry), messages).map(|(message, code)| {
                openai_error_response(StatusCode::BAD_REQUEST, message, Some(code))
            })
        })
}

/// Clamp over-cap per-request `ttl_seconds` advice down to the configured
/// cap (`[swap] max_ttl_secs`) with a `warn!` — never a 400: third-party
/// clients send arbitrary values and the clamp is exactly as safe as a
/// rejection would be. `None` passes through (no advice); the swap manager
/// re-caps defensively when it applies the budget.
fn clamp_ttl_seconds(max: Duration, value: Option<u64>) -> Option<u64> {
    let secs = value?;
    let cap = max.as_secs();
    if secs > cap {
        tracing::warn!(
            requested = secs,
            cap,
            "ttl_seconds above [swap] max_ttl_secs; clamping (never rejected)"
        );
        return Some(cap);
    }
    Some(secs)
}

/// Map a swap error onto the HTTP contract: an unknown / non-LLM slug is a
/// 400 (same semantics as `validate_model`), a failed construction is a 500
/// naming the cause (the cache stays consistent — nothing was inserted).
fn swap_rejection(err: crate::swap::SwapError) -> (StatusCode, String) {
    match err {
        crate::swap::SwapError::UnknownSlug(message) => (StatusCode::BAD_REQUEST, message),
        crate::swap::SwapError::Load(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
    }
}

/// Resolve the chat pipeline for a request: through the swap manager when
/// active (the clamped per-request `ttl_seconds` rides along as the entry's
/// window budget), otherwise the startup-configured static pipeline.
async fn chat_pipeline(
    config: &AppConfig,
    model: &str,
    ttl_seconds: Option<u64>,
) -> Result<Arc<dyn LlmPipeline>, (StatusCode, String)> {
    match &config.swap {
        Some(swap) => swap.get(model, ttl_seconds).await.map_err(swap_rejection),
        None => Ok(config.llm.clone()),
    }
}

/// `POST /v1/chat`: sync worker call inside `spawn_blocking`; the residency
/// lease is held across the call. Only the server edge is async.
#[utoipa::path(
    post,
    path = "/v1/chat",
    request_body = ChatRequest,
    responses(
        (status = 200, description = "Chat completion", body = ChatResponse),
        (status = 400, description = "Unknown model slug"),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Inference failed")
    )
)]
async fn post_chat(
    State(config): State<AppConfig>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, (StatusCode, String)> {
    let catalog = config.catalog_snapshot();
    validate_model(Some(&req.model), config.llm_model.as_deref(), &catalog)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    // Aliases resolve to their owning registry slug; the swap manager and
    // the worker both key on the slug (an unresolved value is the
    // startup-configured model, same as before aliases existed).
    let slug =
        resolve_model_slug(&req.model, &catalog).map_or_else(|| req.model.clone(), str::to_owned);
    // Chat-capability parity with the completions gate: a catalog entry
    // declaring `chat = false` (a projector `mmproj*.gguf`, a declared
    // embedding) is refused by name; the raw startup-configured model (no
    // catalog entry) passes — the operator declared it by hand.
    if let Some(entry) = resolve_model_entry(&slug, &catalog)
        && !entry.capabilities.chat
    {
        return Err((StatusCode::BAD_REQUEST, chat_unsupported_message(&slug)));
    }
    // REQ-8 parity with `/v1/chat/completions`: image content is gated on
    // model vision capability (sibling mmproj or operator override) and
    // `data:` URI payloads — a plain-text 400 on the native route, never a
    // silently flattened image.
    if let Some((message, _)) =
        vision_rejection(resolve_model_entry(&slug, &catalog), &req.messages)
    {
        return Err((StatusCode::BAD_REQUEST, message));
    }
    let pipe = chat_pipeline(
        &config,
        &slug,
        clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds),
    )
    .await?;
    // NEW-3: take the GPU slot first, then register the in-flight job. A
    // rejected request never enters the registry (no lifecycle events, no
    // X-Job-Id), and the admission error is returned explicitly rather
    // than with `?`. The blocking FFI call itself is not interruptible —
    // the cancel token is marked but never checked here.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await,
        crate::jobs_registry::JobKind::Chat,
        Some(slug.clone()),
        false,
    ) {
        Ok(pair) => pair,
        Err(error) => return Err(error),
    };
    let worker_req = worker_request_base(
        slug,
        worker_messages(&req.messages),
        req.max_tokens,
        req.temperature,
    )
    .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
    let reply = tokio::task::spawn_blocking(move || {
        let _lease = lease;
        pipe.complete(&worker_req).map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("chat failed: {err}"),
            )
        })
    })
    .await
    .map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("chat task failed: {err}"),
        )
    })??;
    Ok(with_job_id(
        Json(ChatResponse {
            backend_id: reply.backend_id,
            text: reply.text,
            tokens: reply.tokens,
            millis: reply.millis,
        })
        .into_response(),
        job.id(),
    ))
}

/// Drop the GPU residency lease immediately (the disconnect path): the slot
/// frees at the token boundary while `abort()` stops the FFI loop one token
/// later. Taking the [`crate::admission::GpuLease`] out of the slot drops
/// it, which releases residency AND pokes the admission queue so a queued
/// waiter wakes now instead of at its deadline.
fn free_gpu_lease(slot: &Mutex<Option<crate::admission::GpuLease>>) {
    slot.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
}

/// SSE body wrapper: dropping the body (client disconnect, or the stream
/// ending normally) marks the request cancelled and calls the pipeline's
/// `abort` hook so the synchronous FFI generation stops at its next token
/// step; the worker `JoinHandle` lives exactly as long as the body it feeds.
struct CancelOnDrop<S> {
    /// Wrapped token stream.
    inner: S,
    /// Set on drop; checked by the worker at each token boundary.
    cancelled: Arc<AtomicBool>,
    /// Pipeline `abort` hook (atomics-only state: cannot deadlock with the
    /// blocking worker, which holds no lock across `complete_stream`).
    pipe: Arc<dyn LlmPipeline>,
    /// Blocking worker task feeding `inner`.
    _worker: tokio::task::JoinHandle<()>,
}

impl<S: Stream + Unpin> Stream for CancelOnDrop<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> Drop for CancelOnDrop<S> {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.pipe.abort();
    }
}

/// `POST /v1/chat/stream`: same blocking call, tokens forwarded as SSE
/// (`data: {"token":"..."}` + `data: [DONE]`) over an unbounded channel so
/// the sink never blocks the worker thread.
///
/// Disconnect handling: dropping the SSE body sets a cancel flag and calls
/// the pipeline `abort` hook; the worker notices at its **next token
/// boundary** — token delivery stops, the GPU residency lease is freed, and
/// the synchronous FFI loop exits at its next internal checkpoint (per-token
/// decode returns to our loop, so early return needs no FFI-level abort).
#[utoipa::path(
    post,
    path = "/v1/chat/stream",
    request_body = ChatRequest,
    responses(
        (status = 200, description = "SSE token stream"),
        (status = 400, description = "Unknown model slug"),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Inference failed")
    )
)]
async fn post_chat_stream(
    State(config): State<AppConfig>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, (StatusCode, String)> {
    let catalog = config.catalog_snapshot();
    validate_model(Some(&req.model), config.llm_model.as_deref(), &catalog)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    // Alias → owning slug (same contract as `post_chat`).
    let slug =
        resolve_model_slug(&req.model, &catalog).map_or_else(|| req.model.clone(), str::to_owned);
    // Chat-capability parity with the completions gate (see `post_chat`).
    if let Some(entry) = resolve_model_entry(&slug, &catalog)
        && !entry.capabilities.chat
    {
        return Err((StatusCode::BAD_REQUEST, chat_unsupported_message(&slug)));
    }
    // REQ-8 parity with `/v1/chat/completions`: image content is gated on
    // model vision capability and `data:` URI payloads (plain-text 400 on
    // the native route).
    if let Some((message, _)) =
        vision_rejection(resolve_model_entry(&slug, &catalog), &req.messages)
    {
        return Err((StatusCode::BAD_REQUEST, message));
    }
    let pipe = chat_pipeline(
        &config,
        &slug,
        clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds),
    )
    .await?;
    let worker_req = worker_request_base(
        slug.clone(),
        worker_messages(&req.messages),
        req.max_tokens,
        req.temperature,
    )
    .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let cancelled = Arc::new(AtomicBool::new(false));
    // REQ-6 (Option B): the busy condition must be an HTTP 429, never a 200
    // with an error frame - so the slot is taken HERE through the admission
    // queue (REQ-6 Option A on top: a busy slot grants one bounded wait),
    // before the handler returns the SSE body. A rejection short-circuits
    // as a plain 429 with no registered job and no X-Job-Id.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await,
        crate::jobs_registry::JobKind::Chat,
        Some(slug),
        true,
    ) {
        Ok(pair) => pair,
        Err(error) => return Err(error),
    };
    // NEW-3: only an admitted stream is registry-visible and
    // interruptible — its cancel token is checked at every token boundary
    // below (the same abort-at-next-boundary path an SSE disconnect takes).
    // The lease sits in a slot the token sink can drain: once the client is
    // gone, the next token boundary frees the slot right away. The handler
    // returns the SSE body without awaiting the worker, so tokens stream as
    // they are produced.
    let lease_slot = Arc::new(Mutex::new(None::<crate::admission::GpuLease>));
    // Separate handle for the SSE body: the worker closure consumes its own
    // clone, the drop path calls `abort` through this one.
    let abort_pipe = Arc::clone(&pipe);
    let worker_cancelled = cancelled.clone();
    let job_token = job.token().clone();
    let job_id = job.id().to_owned();
    let worker = tokio::task::spawn_blocking(move || {
        // The job guard rides with the worker: the registry entry lives
        // exactly as long as the stream (the JoinHandle is dropped/detached
        // when the SSE body goes away, the task still unwinds and
        // unregisters).
        let _job = job;
        *lease_slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(lease);
        let result = pipe.complete_stream(&worker_req, &mut |token: &str| {
            if worker_cancelled.load(Ordering::Relaxed) || job_token.is_cancelled() {
                // Also abort the FFI loop itself, not just delivery
                // (disconnect, or `POST /v1/jobs/{id}/cancel`).
                pipe.abort();
                free_gpu_lease(&lease_slot);
                return;
            }
            let data = serde_json::json!({ "token": token }).to_string();
            if tx.send(Ok(Event::default().data(data))).is_err() {
                // Receiver gone without the drop guard (drop raced the send).
                worker_cancelled.store(true, Ordering::Relaxed);
                pipe.abort();
                free_gpu_lease(&lease_slot);
            }
        });
        // A disconnect between the last token and completion frees here.
        free_gpu_lease(&lease_slot);
        if let Err(err) = result {
            let _ = tx.send(Ok(Event::default()
                .event("error")
                .data(format!("chat failed: {err}"))));
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
    });
    Ok(with_job_id(
        Sse::new(CancelOnDrop {
            inner: UnboundedReceiverStream::new(rx),
            cancelled,
            pipe: abort_pipe,
            _worker: worker,
        })
        .into_response(),
        &job_id,
    ))
}

/// OpenAI-shaped error body → JSON response (`/v1/chat/completions` only;
/// the native route keeps its plain-text errors).
pub(crate) fn openai_error_response(
    status: StatusCode,
    message: String,
    code: Option<&str>,
) -> Response {
    let body = OpenAiError::new(status.as_u16(), message, code.map(str::to_owned));
    (status, Json(body)).into_response()
}

/// Resolved `REQ-3` constraint fields for one completions request.
struct ChatConstraints {
    /// Raw GBNF grammar, honored verbatim.
    grammar: Option<String>,
    /// JSON Schema text (`response_format: json_schema`); the worker
    /// converts it to GBNF (unconvertible schema = worker `InvalidRequest`
    /// = honest 400).
    json_schema: Option<String>,
    /// `response_format: json_object` → generic any-JSON grammar.
    json_object: bool,
    /// `OpenAI` tools array JSON; `None` = no tools offered.
    tools_json: Option<String>,
    /// `tool_choice` clamped: only `"none"` survives (everything else
    /// behaves as `auto`). The clamp is disclosed machine-readably on
    /// `GET /v1/capabilities` (`tools`) — never a silent downgrade.
    tool_choice_none: bool,
    /// Extended sampler knobs, wired when present.
    sampler: SamplerParams,
}

/// Resolve the `REQ-3` request fields into worker-protocol constraints.
/// Every field is enforced, clamped, or rejected with an `OpenAI`
/// envelope — never silently ignored: output constraints (`grammar`,
/// `response_format`) and `tools` are mutually exclusive (llama.cpp
/// itself rejects the combination; dropping one side would be fake
/// support), and so are `grammar` and `response_format` with each other.
/// An empty conversation is rejected here too: the llama backend refuses
/// it (`InvalidRequest`), so the route answers 400 up front rather than
/// surfacing a backend-specific 500 later.
///
/// Accepted `tool_choice` vocabulary (`ToolChoiceSpec::HONORED_MODES` /
/// `CLAMPED_MODES`): `"none"` and `"auto"` are honored verbatim;
/// `"required"`, `"any"`, and the named-function form are accepted and
/// clamped to `"auto"` — disclosed on `GET /v1/capabilities` (`tools`),
/// never a 400. The vendored `llama-cpp-4` binding also exposes a
/// grammar-enforced `ToolChoice::Required`; it stays clamped here pending
/// an exact-choice consumer (TASK-llm-tool-choice-contract follow-up).
fn resolve_constraints(req: &OpenAiChatRequest) -> Result<ChatConstraints, (&'static str, String)> {
    if req.messages.is_empty() {
        return Err((
            "messages_required",
            "messages is required: pass at least one conversation turn".to_owned(),
        ));
    }
    let tools_json = match req.tools.as_deref() {
        None | Some([]) => None,
        Some(tools) => Some(
            serde_json::to_string(tools)
                .map_err(|err| ("unsupported_constraint", format!("tools: {err}")))?,
        ),
    };
    let mut json_schema = None;
    let mut json_object = false;
    match &req.response_format {
        Some(ResponseFormat {
            format_type: ResponseFormatType::JsonSchema,
            json_schema: Some(spec),
        }) => {
            if spec.schema.is_null() {
                return Err((
                    "unsupported_constraint",
                    "response_format.json_schema.schema is required".to_owned(),
                ));
            }
            json_schema = Some(spec.schema.to_string());
        }
        Some(ResponseFormat {
            format_type: ResponseFormatType::JsonSchema,
            json_schema: None,
        }) => {
            return Err((
                "unsupported_constraint",
                "response_format.json_schema is required for type json_schema".to_owned(),
            ));
        }
        Some(ResponseFormat {
            format_type: ResponseFormatType::JsonObject,
            ..
        }) => json_object = true,
        _ => {}
    }
    if tools_json.is_some() && (req.grammar.is_some() || json_schema.is_some() || json_object) {
        return Err((
            "unsupported_constraint",
            "tools combined with grammar or response_format is not supported: the tool-call constraint and the output constraint are mutually exclusive"
                .to_owned(),
        ));
    }
    if req.grammar.is_some() && (json_schema.is_some() || json_object) {
        return Err((
            "unsupported_constraint",
            "grammar and response_format are mutually exclusive: pass one output constraint"
                .to_owned(),
        ));
    }
    Ok(ChatConstraints {
        grammar: req.grammar.clone(),
        json_schema,
        json_object,
        tools_json,
        tool_choice_none: req
            .tool_choice
            .as_ref()
            .is_some_and(ToolChoiceSpec::suppresses_tools),
        sampler: SamplerParams {
            top_k: req.top_k,
            min_p: req.min_p,
            typical_p: req.typical_p,
            repeat_penalty: req.repeat_penalty,
            xtc_probability: req.xtc_probability,
            dry_multiplier: req.dry_multiplier,
            dry_base: req.dry_base,
            dry_allowed_length: req.dry_allowed_length,
            dynatemp_range: req.dynatemp_range,
            dynatemp_exponent: req.dynatemp_exponent,
        },
    })
}

/// Small error payload from [`validate_embeddings_request`]; the route
/// turns it into the `OpenAI` error envelope.
struct EmbeddingsRejection {
    status: StatusCode,
    message: String,
    code: &'static str,
}

impl EmbeddingsRejection {
    fn response(self) -> Response {
        openai_error_response(self.status, self.message, Some(self.code))
    }
}

/// Parse and validate an embeddings request: JSON shape, known model,
/// embeddings capability, and non-empty inputs.
fn validate_embeddings_request(
    config: &AppConfig,
    body: &Bytes,
) -> Result<(ModelCatalogEntry, Vec<String>), EmbeddingsRejection> {
    let req: EmbeddingsRequest =
        serde_json::from_slice(body).map_err(|err| EmbeddingsRejection {
            status: StatusCode::BAD_REQUEST,
            message: format!("invalid JSON body: {err}"),
            code: "invalid_json_body",
        })?;
    let model = req.model.trim();
    let catalog = config.catalog_snapshot();
    let entry = resolve_model_entry(model, &catalog).ok_or_else(|| EmbeddingsRejection {
        status: StatusCode::NOT_FOUND,
        message: format!(
            "unknown model {model:?}: not an alias or registry slug (see GET /v1/models)"
        ),
        code: "model_not_found",
    })?;
    // Capability gate (REQ-4 acceptance): only models the operator declared
    // embeddings-capable load at all — a chat GGUF handed to the embedding
    // path would produce garbage vectors, so it is refused by name.
    if !entry.capabilities.embeddings {
        return Err(EmbeddingsRejection {
            status: StatusCode::BAD_REQUEST,
            message: format!(
                "model {:?} is not embeddings-capable: declare capabilities.embeddings = true for it ([[models.entry]]) and use an embedding GGUF",
                entry.id
            ),
            code: "not_embeddings_capable",
        });
    }
    let inputs = req.input.into_vec();
    if inputs.is_empty() || inputs.iter().any(String::is_empty) {
        return Err(EmbeddingsRejection {
            status: StatusCode::BAD_REQUEST,
            message: "input is required: at least one non-empty string".to_owned(),
            code: "input_required",
        });
    }
    Ok((entry.clone(), inputs))
}

/// `POST /v1/embeddings` (REQ-4): OpenAI-shaped embeddings over the same
/// residency model as chat — one GPU slot, `spawn_blocking` inference.
/// Resolution: alias/slug → 404 envelope when unknown; a model without
/// `capabilities.embeddings` is refused with an explicit 400 (never
/// loaded, never garbage tokens). The pipeline is cached per slug
/// (`EmbedCache`: load-once, idle-TTL reuse, capacity 1) outside the
/// slot; the embed call itself runs inside the lease.
///
/// Errors: 400 bad body / not-embeddings-capable, 404 unknown model, 429
/// busy slot (+`Retry-After` via the shared middleware), 500 embed failure.
#[utoipa::path(
    post,
    path = "/v1/embeddings",
    request_body = EmbeddingsRequest,
    responses(
        (status = 200, description = "Embeddings", body = EmbeddingsResponse),
        (status = 400, description = "Bad request or not an embeddings-capable model", body = OpenAiError),
        (status = 404, description = "Unknown model", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Embedding failed", body = OpenAiError)
    )
)]
async fn post_embeddings(State(config): State<AppConfig>, body: Bytes) -> Response {
    let (entry, inputs) = match validate_embeddings_request(&config, &body) {
        Ok(pair) => pair,
        Err(rejection) => return rejection.response(),
    };
    let served_id = entry.id.clone();
    // Pipeline resolution (cache hit = cheap; miss = blocking GGUF load)
    // runs on the blocking pool, outside the GPU slot — same order as the
    // chat routes (load first, then reserve).
    let cache = Arc::clone(&config.embed);
    let load_entry = entry.clone();
    let pipe = match tokio::task::spawn_blocking(move || cache.get(&load_entry)).await {
        Ok(Ok(pipe)) => pipe,
        Ok(Err(message)) => {
            return openai_error_response(StatusCode::INTERNAL_SERVER_ERROR, message, None);
        }
        Err(err) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("embed task failed: {err}"),
                None,
            );
        }
    };
    // NEW-3: the slot comes through the admission queue first (REQ-6
    // Option A); only an admitted request gets a registry-visible job
    // (never interruptible: one blocking FFI call). A 429 therefore has no
    // lifecycle events and no X-Job-Id, keeping the shared envelope.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Embed).await,
        crate::jobs_registry::JobKind::Embeddings,
        Some(entry.slug.clone()),
        false,
    ) {
        Ok(pair) => pair,
        Err((status, message)) => {
            return openai_error_response(status, message, Some("gpu_slot_busy"));
        }
    };
    let reply = tokio::task::spawn_blocking(move || {
        let _lease = lease;
        pipe.embed(&inputs)
            .map(|embeddings| (embeddings, pipe.prompt_tokens(&inputs)))
            .map_err(|err| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("embed failed: {err}"),
                )
            })
    })
    .await
    .map_err(|err| format!("embed task failed: {err}"));
    let (embeddings, prompt_tokens) = match reply {
        Ok(Ok(pair)) => pair,
        Ok(Err((StatusCode::TOO_MANY_REQUESTS, message))) => {
            return openai_error_response(
                StatusCode::TOO_MANY_REQUESTS,
                message,
                Some("gpu_slot_busy"),
            );
        }
        Ok(Err((_, message))) | Err(message) => {
            return openai_error_response(StatusCode::INTERNAL_SERVER_ERROR, message, None);
        }
    };
    with_job_id(
        Json(EmbeddingsResponse {
            object: "list".to_owned(),
            data: embeddings
                .into_iter()
                .enumerate()
                .map(|(index, embedding)| EmbeddingsData {
                    object: "embedding".to_owned(),
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    embedding,
                })
                .collect(),
            model: served_id,
            usage: EmbeddingsUsage {
                prompt_tokens,
                total_tokens: u64::from(prompt_tokens),
            },
        })
        .into_response(),
        job.id(),
    )
}

/// REQ-4 reverse gate: an embeddings-capable model never runs through the
/// chat pipeline (garbage tokens dressed up as a completion). `Some`
/// carries the explicit 400 response.
fn embeddings_model_gate(entry: &ModelCatalogEntry, model: &str) -> Option<Response> {
    entry.capabilities.embeddings.then(|| {
        openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "model {model:?} is an embeddings model (capabilities.embeddings = true) and cannot serve chat completions"
            ),
            Some("embeddings_model_for_chat"),
        )
    })
}

/// SD-side model resolution for `POST /v1/images/generations` (REQ-10):
/// alias-or-slug over the catalog, restricted to `ModelKind::Sd` entries
/// (the comfy/SD category). Returns the owning registry slug.
fn resolve_sd_model<'a>(model: &str, catalog: &'a ModelCatalog) -> Option<&'a str> {
    resolve_model_entry(model, catalog)
        .filter(|e| e.kind == ModelKind::Sd)
        .map(|e| e.slug.as_str())
}

/// Validated image request: parsed size, clamped count, and the resolved
/// generation recipe (sampler/scheduler/CFG/steps/negative/seed base; the
/// fields a named recipe supplied or a request overrode — omitting every
/// one is byte-identical to the pre-passthrough behavior). The only
/// rejection paths for `POST /v1/images/generations` before the GPU slot;
/// `Err` is (message, error code) for a 400 `OpenAI` envelope. Unlike `n`
/// (clamped with a warn), unknown recipe/sampler/scheduler names,
/// out-of-range `cfg_scale`/`steps`, and invalid `hires_fix` values are
/// hard 400s — silently substituting a recipe the caller did not ask for
/// would be dishonest about what was rendered.
struct ImageRecipe {
    sampler: honkbox_sd_core::Sampler,
    scheduler: honkbox_sd_core::Scheduler,
    cfg_scale: f32,
    negative_prompt: Option<String>,
    seed_base: u64,
    /// Diffusion steps actually used (server default, recipe default, or
    /// explicit override — see `validate_recipe`'s precedence docs).
    steps: u32,
    /// Named curated recipe applied, if any (`None` = field-by-field or
    /// pure defaults). Recorded verbatim in the job bundle.
    name: Option<&'static str>,
}

/// CFG guidance bounds for the HTTP surface (documented on the DTO).
const IMAGE_CFG_SCALE_RANGE: std::ops::RangeInclusive<f32> = 1.0..=30.0;
/// Accepted diffusion-steps range for the HTTP surface (documented on the
/// DTO); outside it is a 400, never a clamp.
const IMAGE_STEPS_RANGE: std::ops::RangeInclusive<u32> = 1..=100;
/// Hires-fix scale bounds for the HTTP surface (documented on the DTO).
const IMAGE_HIRES_SCALE_RANGE: std::ops::RangeInclusive<f32> = 1.0..=4.0;

/// One curated generation recipe: a sampler/scheduler/cfg/steps bundle for
/// a checkpoint family. Values mirror the vendored stable-diffusion.cpp
/// presets/community standards, constrained to what sd-core's
/// `Sampler`/`Scheduler` enums support — nothing here can fail to parse at
/// generation time.
struct ImageRecipeSpec {
    name: &'static str,
    sampler: honkbox_sd_core::Sampler,
    scheduler: honkbox_sd_core::Scheduler,
    cfg_scale: f32,
    steps: u32,
}

/// The curated recipe set (`POST /v1/images/generations` + `/edits`):
/// - `sd15`: SD 1.x/2.x — the vendor SD1.5/SD2.1 presets leave everything
///   at the sd.cpp defaults (`euler_a`, discrete schedule, CFG 7, 20 steps).
/// - `sdxl`: SDXL base + finetunes — the community-standard
///   dpm++2m/karras combo at 25 steps.
/// - `sd35`: SD3.5 — the vendor `stable_diffusion_3_5_large` preset
///   (euler, CFG 4.5, 28 steps).
/// - `turbo`: turbo-distilled (SD/SDXL turbo) — the vendor turbo presets
///   set CFG 1.0 at 4 steps, sampler/scheduler left at defaults.
/// - `lcm`: LCM-distilled / LCM-LoRA — the vendor `lcm_lora_sdxl` modifier
///   (lcm sampler, CFG 2.0, 8 steps).
/// - `flux-dev` / `flux-schnell`: FLUX.1 — the vendor flux presets (euler,
///   CFG 1.0 — distilled guidance handles steering, 28 / 4 steps).
const IMAGE_RECIPES: &[ImageRecipeSpec] = &[
    ImageRecipeSpec {
        name: "sd15",
        sampler: honkbox_sd_core::Sampler::EulerA,
        scheduler: honkbox_sd_core::Scheduler::Discrete,
        cfg_scale: 7.0,
        steps: 20,
    },
    ImageRecipeSpec {
        name: "sdxl",
        sampler: honkbox_sd_core::Sampler::Dpmpp2m,
        scheduler: honkbox_sd_core::Scheduler::Karras,
        cfg_scale: 7.0,
        steps: 25,
    },
    ImageRecipeSpec {
        name: "sd35",
        sampler: honkbox_sd_core::Sampler::Euler,
        scheduler: honkbox_sd_core::Scheduler::Discrete,
        cfg_scale: 4.5,
        steps: 28,
    },
    ImageRecipeSpec {
        name: "turbo",
        sampler: honkbox_sd_core::Sampler::EulerA,
        scheduler: honkbox_sd_core::Scheduler::Discrete,
        cfg_scale: 1.0,
        steps: 4,
    },
    ImageRecipeSpec {
        name: "lcm",
        sampler: honkbox_sd_core::Sampler::Lcm,
        scheduler: honkbox_sd_core::Scheduler::Discrete,
        cfg_scale: 2.0,
        steps: 8,
    },
    ImageRecipeSpec {
        name: "flux-dev",
        sampler: honkbox_sd_core::Sampler::Euler,
        scheduler: honkbox_sd_core::Scheduler::Discrete,
        cfg_scale: 1.0,
        steps: 28,
    },
    ImageRecipeSpec {
        name: "flux-schnell",
        sampler: honkbox_sd_core::Sampler::Euler,
        scheduler: honkbox_sd_core::Scheduler::Discrete,
        cfg_scale: 1.0,
        steps: 4,
    },
];

/// Every accepted recipe wire name, in table order (for 400 messages).
fn image_recipe_names() -> Vec<&'static str> {
    IMAGE_RECIPES.iter().map(|r| r.name).collect()
}

/// Look up one curated recipe by exact wire name.
fn resolve_recipe(name: &str) -> Option<&'static ImageRecipeSpec> {
    IMAGE_RECIPES.iter().find(|r| r.name == name)
}

/// Shared recipe validation for both images routes: `generations` passes
/// JSON DTO fields straight through; `edits` passes multipart-parsed
/// strings. `Err` is (message, error code) for a 400 `OpenAI` envelope.
/// Unknown recipe/sampler/scheduler names and out-of-range `cfg_scale`/
/// `steps` are hard 400s — silently substituting a recipe the caller did
/// not ask for would be dishonest about what was rendered.
///
/// Precedence per field: an explicit request field wins over the named
/// recipe's default, which wins over the server default — so `recipe:
/// "sdxl"` plus `sampler: "euler"` renders euler with the recipe's
/// remaining (scheduler/cfg/steps) defaults.
fn validate_recipe(
    recipe: Option<&str>,
    sampler: Option<&str>,
    scheduler: Option<&str>,
    cfg_scale: Option<f32>,
    negative_prompt: Option<&str>,
    seed: Option<u64>,
    steps: Option<u32>,
) -> Result<ImageRecipe, (String, &'static str)> {
    let recipe_name = recipe.map(str::trim).filter(|s| !s.is_empty());
    let spec = match recipe_name {
        Some(name) => Some(resolve_recipe(name).ok_or_else(|| {
            (
                format!(
                    "unknown recipe {name:?}: accepted values are {}",
                    image_recipe_names().join(", ")
                ),
                "unknown_recipe",
            )
        })?),
        None => None,
    };
    let sampler_name = sampler.map(str::trim).filter(|s| !s.is_empty());
    let sampler = match sampler_name {
        Some(name) => honkbox_sd_core::Sampler::parse(name).ok_or_else(|| {
            (
                format!(
                    "unknown sampler {name:?}: accepted values are {}",
                    honkbox_sd_core::SAMPLER_NAMES.join(", ")
                ),
                "unknown_sampler",
            )
        })?,
        None => spec.map_or(honkbox_sd_core::Sampler::Euler, |s| s.sampler),
    };
    let scheduler_name = scheduler.map(str::trim).filter(|s| !s.is_empty());
    let scheduler = match scheduler_name {
        Some(name) => honkbox_sd_core::Scheduler::parse(name).ok_or_else(|| {
            (
                format!(
                    "unknown scheduler {name:?}: accepted values are {}",
                    honkbox_sd_core::SCHEDULER_NAMES.join(", ")
                ),
                "unknown_scheduler",
            )
        })?,
        // The fixed recipe's recorded `normal` name IS the vendor discrete
        // schedule; wire spellings follow sd.cpp (`discrete`).
        None => spec.map_or(honkbox_sd_core::Scheduler::Discrete, |s| s.scheduler),
    };
    let cfg_scale = match cfg_scale {
        Some(v) if IMAGE_CFG_SCALE_RANGE.contains(&v) && v.is_finite() => v,
        Some(v) => {
            return Err((
                format!("cfg_scale {v} outside accepted range [1.0, 30.0]: rejected, not clamped"),
                "invalid_cfg_scale",
            ));
        }
        None => spec.map_or(7.0, |s| s.cfg_scale),
    };
    let steps = match steps {
        Some(v) if IMAGE_STEPS_RANGE.contains(&v) => v,
        Some(v) => {
            return Err((
                format!("steps {v} outside accepted range [1, 100]: rejected, not clamped"),
                "invalid_steps",
            ));
        }
        None => spec.map_or(IMAGE_STEPS, |s| s.steps),
    };
    let negative_prompt = negative_prompt
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    Ok(ImageRecipe {
        sampler,
        scheduler,
        cfg_scale,
        negative_prompt,
        seed_base: seed.unwrap_or(IMAGE_SEED_BASE),
        steps,
        name: spec.map(|s| s.name),
    })
}

/// Validate the optional hires-fix pass against the parsed base size:
/// per-field range checks (`scale`, `denoise`), `target` size parsing, and
/// the shared [`honkbox_sd_core::MAX_EDGE`] output cap (the final PNG lands
/// at the hires target, so it must stay under the cap too).
fn validate_hires_fix(
    hires: Option<&ImagesHiresFix>,
    base: (u32, u32),
) -> Result<Option<honkbox_sd_core::HiresFix>, (String, &'static str)> {
    let Some(hires) = hires else {
        return Ok(None);
    };
    let scale = match hires.scale {
        Some(v) if IMAGE_HIRES_SCALE_RANGE.contains(&v) && v.is_finite() => v,
        Some(v) => {
            return Err((
                format!(
                    "hires_fix.scale {v} outside accepted range [1.0, 4.0]: rejected, not clamped"
                ),
                "invalid_hires_scale",
            ));
        }
        None => 2.0,
    };
    let target = match hires
        .target
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        Some(raw) => Some(
            crate::jobs::parse_size(raw)
                .map_err(|err| (format!("hires_fix.target: {err}"), "invalid_hires_target"))?,
        ),
        None => None,
    };
    let denoising_strength = match hires.denoise {
        Some(v) if v.is_finite() && v > 0.0 && v <= 1.0 => v,
        Some(v) => {
            return Err((
                format!(
                    "hires_fix.denoise {v} outside accepted range (0, 1]: rejected, not clamped"
                ),
                "invalid_hires_denoise",
            ));
        }
        None => 0.7,
    };
    let fix = honkbox_sd_core::HiresFix {
        scale,
        target,
        denoising_strength,
    };
    let (target_width, target_height) = honkbox_sd_core::hires_target(base, &fix);
    let edge = base.0.max(base.1).max(target_width).max(target_height);
    if edge > honkbox_sd_core::MAX_EDGE {
        return Err((
            format!(
                "hires_fix output {target_width}x{target_height} exceeds the {}px output cap",
                honkbox_sd_core::MAX_EDGE
            ),
            "invalid_hires_target",
        ));
    }
    Ok(Some(fix))
}

/// Validated `POST /v1/images/generations` request: parsed base size,
/// clamped count, the resolved recipe, and the optional hires-fix pass
/// (tuple alias keeps `validate_images_request`'s signature readable — the
/// same pattern as `ResolvedLoras`).
type ValidatedImages = (
    u32,
    u32,
    u32,
    ImageRecipe,
    Option<honkbox_sd_core::HiresFix>,
    honkbox_sd_core::OutputFormat,
    Option<u8>,
);

/// Parse the requested output container against the registry's compiled
/// encoder rows (omitted = PNG). Unknown names are a named 400 listing the
/// registry-derived accepted values (same reject-don't-guess policy as
/// `response_format`). The wire name → enum mapping stays in THIS module
/// (the pre-registry call surface — the `#[path]`-included registry module
/// must not link the native backends); alignment with the registry rows is
/// locked by `format_registry_matches_the_compiled_encoders` below.
fn parse_output_format(raw: &str) -> Result<honkbox_sd_core::OutputFormat, (String, &'static str)> {
    let accepted = crate::formats::output_rows()
        .map(|row| format!("{:?}", row.name))
        .collect::<Vec<_>>()
        .join(", ");
    honkbox_sd_core::OutputFormat::from_name(raw).ok_or_else(|| {
        (
            format!("unsupported output_format {raw:?}: accepted values are {accepted}"),
            "unsupported_output_format",
        )
    })
}

/// Parse `output_compression`: an integer JPEG quality in `[0, 100]`.
/// Non-numeric, fractional, negative, or out-of-range values are a named
/// 400 — rejected, never clamped. The value is only applied to JPEG; PNG is
/// lossless and WebP is lossless-only today, so there it is ignored (DTO
/// docs say so). `None` = the sd-core default (quality 90).
fn parse_output_compression(
    raw: Option<&serde_json::Value>,
) -> Result<Option<u8>, (String, &'static str)> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    match raw.as_u64() {
        Some(q) if q <= 100 => Ok(Some(u8::try_from(q).expect("q <= 100"))),
        _ => Err((
            format!(
                "output_compression must be an integer in [0, 100]; got {raw}: rejected, not clamped"
            ),
            "invalid_output_compression",
        )),
    }
}

fn validate_images_request(
    req: &ImagesGenerationRequest,
) -> Result<ValidatedImages, (String, &'static str)> {
    let (width, height) = crate::jobs::parse_size(req.size.as_deref().unwrap_or("1024x1024"))
        .map_err(|err| (err, "invalid_size"))?;
    if let Some(format) = req
        .response_format
        .as_deref()
        .map(str::trim)
        .filter(|f| *f != "b64_json")
    {
        return Err((
            format!("unsupported response_format {format:?}: only \"b64_json\" is supported"),
            "unsupported_response_format",
        ));
    }
    let n = req.n.unwrap_or(1);
    if n == 0 || n > 4 {
        tracing::warn!(
            requested = n,
            "images n outside [1, 4]; clamping (never rejected)"
        );
    }
    let recipe = validate_recipe(
        req.recipe.as_deref(),
        req.sampler.as_deref(),
        req.scheduler.as_deref(),
        req.cfg_scale,
        req.negative_prompt.as_deref(),
        req.seed,
        req.steps,
    )?;
    let hires = validate_hires_fix(req.hires_fix.as_ref(), (width, height))?;
    // Output container: the registry is the source of truth (omitted =
    // PNG); unknown names are a named 400 listing the registry-derived
    // accepted values (same reject-don't-guess policy as `response_format`).
    let format = parse_output_format(req.output_format.as_deref().unwrap_or("png").trim())?;
    let compression = parse_output_compression(req.output_compression.as_ref())?;
    Ok((
        width,
        height,
        n.clamp(1, 4),
        recipe,
        hires,
        format,
        compression,
    ))
}

/// Accepted `LoRA` multiplier range for the HTTP surface (documented on the
/// DTO); outside it is a 400, never a silent clamp.
const IMAGE_LORA_STRENGTH_RANGE: std::ops::RangeInclusive<f32> = -2.0..=2.0;

/// Max `LoRA` adapters per images request (documented on the DTO); more is a
/// 400 — the vendor applies them sequentially, and cost scales linearly.
const IMAGE_LORA_MAX: usize = 2;

/// Resolve request `LoRA` references against the catalog (`ModelKind::Lora`
/// entries, alias-or-slug exactly like SD models). Returns the
/// pipeline-ready refs, sidecar records, and advisory warnings. Rejections:
/// more than [`IMAGE_LORA_MAX`] adapters, empty/unknown slug or alias,
/// strength outside [`IMAGE_LORA_STRENGTH_RANGE`], and adapters spanning
/// more than one directory (the vendor applies a single `LoRA` dir per model
/// config).
///
/// License rule on record (ReviewedLicense-AGPL §2b/§2c): honkbox never
/// bundles `LoRA` weights. The curated catalog `license` is surfaced as a
/// response warning when a manifest row matches the local file name; a
/// missing license means all-rights-reserved, i.e. local-only use. A
/// base-model mismatch cannot be detected today (no base tags are recorded
/// on scan entries or `LoRA` files), so it is neither warned about nor
/// rejected — an honest gap, not a silent pass.
/// Resolved `LoRA` request: pipeline-ready refs, sidecar records, and the
/// advisory license warnings (a tuple alias keeps `resolve_loras`'s
/// signature readable).
type ResolvedLoras = (
    Vec<honkbox_sd_core::LoraRef>,
    Vec<crate::jobs::LoraRecord>,
    Vec<String>,
);

fn resolve_loras(
    loras: &[ImagesLora],
    catalog: &ModelCatalog,
) -> Result<ResolvedLoras, (String, &'static str)> {
    if loras.len() > IMAGE_LORA_MAX {
        return Err((
            format!(
                "loras: at most {IMAGE_LORA_MAX} adapters per request (got {})",
                loras.len()
            ),
            "too_many_loras",
        ));
    }
    let mut refs = Vec::with_capacity(loras.len());
    let mut records = Vec::with_capacity(loras.len());
    let mut warnings = Vec::with_capacity(loras.len());
    for lora in loras {
        let slug = lora.slug.trim();
        if slug.is_empty() {
            return Err(("loras: slug must not be empty".to_owned(), "unknown_lora"));
        }
        let Some(entry) =
            resolve_model_entry(slug, catalog).filter(|entry| entry.kind == ModelKind::Lora)
        else {
            return Err((
                format!(
                    "unknown lora {slug:?}: not a LoRA catalog slug or alias (see GET /v1/models)"
                ),
                "unknown_lora",
            ));
        };
        let strength = match lora.strength {
            Some(v) if IMAGE_LORA_STRENGTH_RANGE.contains(&v) && v.is_finite() => v,
            Some(v) => {
                return Err((
                    format!(
                        "lora {slug:?}: strength {v} outside accepted range [-2.0, 2.0]: rejected, not clamped"
                    ),
                    "invalid_lora_strength",
                ));
            }
            None => 1.0,
        };
        // License surfacing: the curated manifest carries a per-file
        // `license`; scan-only entries have none. Join is by file name (the
        // same stable key the catalog build uses).
        let file_name = entry.path.rsplit('/').next().unwrap_or(&entry.path);
        let license = crate::catalog::manifest()
            .iter()
            .find(|m| m.kind == ModelKind::Lora && m.file.rsplit('/').next() == Some(file_name))
            .map(|m| m.license.as_str())
            .filter(|license| !license.is_empty());
        warnings.push(match license {
            Some(license) => format!("lora {slug}: license {license}"),
            None => format!(
                "lora {slug}: no license recorded — treat as all-rights-reserved (local-only)"
            ),
        });
        refs.push(honkbox_sd_core::LoraRef {
            path: PathBuf::from(&entry.path),
            strength,
        });
        records.push(crate::jobs::LoraRecord {
            slug: entry.slug.clone(),
            strength,
        });
    }
    if let Some(first) = refs.first() {
        let dir = first.path.parent();
        if refs.iter().any(|r| r.path.parent() != dir) {
            return Err((
                "loras: all adapters must live in one directory (the vendor applies a single LoRA dir per generation)".to_owned(),
                "mixed_lora_dirs",
            ));
        }
    }
    Ok((refs, records, warnings))
}

/// Resolved `ControlNet` inputs for a validated generation request: the
/// catalog-resolved weights file, the decoded PNG conditioning image, the
/// optional guidance-strength override, and the optional named conditioning
/// type (what the image depicts — `None` = the vendor default `none`, the
/// preprocessed-map behavior).
struct ResolvedControlNet {
    weights: PathBuf,
    image: Vec<u8>,
    strength: Option<f32>,
    control_type: Option<honkbox_sd_core::sd_control_type_t>,
}

/// Outcome of [`controlnet_pair`]: the `(slug, image)` pair (a placeholder
/// when refusing) plus an optional rejection response.
type ControlNetPair = ((String, Vec<u8>), Option<Response>);

/// Validate the honkbox `ControlNet` extension fields: the slug/conditioning
/// pair (both halves or a named 400), strict standard base64, and the PNG
/// magic (the sd path consumes the control image exactly like every other
/// PNG input; anything else cannot be honestly accepted).
///
/// `None` = nothing actually requested (whitespace-only fields are not a
/// request). `Some((pair, None))` = a valid `(slug, decoded image)`;
/// `Some((_, Some(response)))` = a rejection: `controlnet` without
/// `control_image` or vice versa (400 `controlnet_pair`), undecodable
/// base64 or non-PNG bytes (400 `invalid_control_image`).
fn controlnet_pair(req: &ImagesGenerationRequest) -> Option<ControlNetPair> {
    let slug = req
        .controlnet
        .as_deref()
        .map(str::trim)
        .filter(|slug| !slug.is_empty());
    let image = req
        .control_image
        .as_deref()
        .map(str::trim)
        .filter(|image| !image.is_empty());
    let (slug, image) = match (slug, image) {
        (None, None) => return None,
        (Some(slug), Some(image)) => (slug, image),
        (Some(_), None) => {
            return Some((
                (String::new(), Vec::new()),
                Some(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    "controlnet and control_image must be supplied together: control_image is missing (the base64 PNG conditioning image for the controlnet slug)".to_owned(),
                    Some("controlnet_pair"),
                )),
            ));
        }
        (None, Some(_)) => {
            return Some((
                (String::new(), Vec::new()),
                Some(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    "controlnet and control_image must be supplied together: controlnet is missing (the catalog slug the conditioning image pairs with)".to_owned(),
                    Some("controlnet_pair"),
                )),
            ));
        }
    };
    let decoded = match base64::engine::general_purpose::STANDARD.decode(image) {
        Ok(bytes) => bytes,
        Err(err) => {
            return Some((
                (String::new(), Vec::new()),
                Some(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("control_image is not usable: invalid base64 ({err})"),
                    Some("invalid_control_image"),
                )),
            ));
        }
    };
    if let Err(reject) = crate::formats::identify_png(&decoded) {
        return Some((
            (String::new(), Vec::new()),
            Some(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "control_image is not usable: decoded bytes are not a PNG ({})",
                    reject.detail()
                ),
                Some("invalid_control_image"),
            )),
        ));
    }
    Some(((slug.to_owned(), decoded), None))
}

/// `ControlNet` request handling for `POST /v1/images/generations` (honkbox
/// extension, not part of the `OpenAI` shape): full validation, then either
/// resolved inputs threaded into the generation or an explicit unsupported
/// verdict for a pipeline that cannot apply them. `(None, None)` = nothing
/// actually requested (whitespace-only fields are not a request) and plain
/// generation proceeds.
///
/// Rejections, in order: the pair/image checks of [`controlnet_pair`], a
/// `controlnet` slug/alias that is not a `controlnet`-kind catalog entry
/// (400 `unknown_controlnet` — the same alias-or-slug rule as SD models and
/// `LoRA`s), a non-finite or out-of-range `controlnet_strength` (400
/// `invalid_controlnet_strength`, the same reject-not-clamp rule as img2img
/// `strength`), a `control_type` that is not one of the engine's names
/// (400 `unknown_control_type` — the name table lives in the vendored
/// engine, so validation and execution can never disagree), and a
/// configured pipeline that cannot carry control inputs
/// (501 `controlnet_unsupported`) — never a silent ignore: rendering
/// without the requested conditioning image would misrepresent the output.
fn resolve_controlnet(
    req: &ImagesGenerationRequest,
    catalog: &ModelCatalog,
    pipeline: &dyn honkbox_sd_core::SdPipeline,
) -> (Option<ResolvedControlNet>, Option<Response>) {
    // The named conditioning type (`control_type`): trimmed and treated like
    // every other control field — whitespace-only is not a request.
    let named_type = req
        .control_type
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty());
    let pair = match controlnet_pair(req) {
        // `control_type` modifies a `controlnet`/`control_image` pair; a
        // stray modifier is refused like a lone pair half — a plain render
        // would never use it, so accepting it would silently mislead.
        None if named_type.is_some() => {
            return (
                None,
                Some(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    "control_type without a controlnet/control_image pair: the conditioning type modifies a ControlNet request, so it is refused on its own (a plain render would not use it)".to_owned(),
                    Some("controlnet_pair"),
                )),
            );
        }
        None => return (None, None),
        Some((pair, None)) => pair,
        Some((_, Some(response))) => return (None, Some(response)),
    };
    let (slug, decoded) = pair;
    // Slug: alias-or-slug against the catalog, `controlnet` kind only.
    let entry = match resolve_model_entry(&slug, catalog) {
        Some(entry) if entry.kind == ModelKind::ControlNet => entry,
        _ => {
            return (
                None,
                Some(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "unknown controlnet {slug:?}: not a ControlNet catalog slug or alias (see GET /v1/models)"
                    ),
                    Some("unknown_controlnet"),
                )),
            );
        }
    };
    let strength = match req.controlnet_strength {
        Some(v) if v.is_finite() && (0.0..=1.0).contains(&v) && v > 0.0 => Some(v),
        Some(v) => {
            return (
                None,
                Some(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "controlnet_strength {v} outside accepted range (0, 1]: rejected, not clamped"
                    ),
                    Some("invalid_controlnet_strength"),
                )),
            );
        }
        None => None,
    };
    // The conditioning type is validated against the engine's own name
    // table (`control_type_from_name` → `str_to_sd_control_type`): an
    // unknown name is a named 400, not an ignored field — the render would
    // either drop the conditioning (classic nets) or fail mid-generation
    // (union nets).
    let control_type = match named_type {
        None => None,
        Some(name) => match honkbox_sd_core::control_type_from_name(name) {
            Some(parsed) => Some(parsed),
            None => {
                return (
                    None,
                    Some(openai_error_response(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "unknown control_type {name:?}: not a ControlNet type the engine knows (none, openpose, depth, scribble, canny, normal, segment, tile, repaint)"
                        ),
                        Some("unknown_control_type"),
                    )),
                );
            }
        },
    };
    // Honest unsupported verdict: a pipeline without a `ControlNet` branch
    // (the stub backend) would render the prompt unconditioned.
    if pipeline.backend_id() != honkbox_sd_core::SD_CPP_BACKEND_ID {
        return (
            None,
            Some(openai_error_response(
                StatusCode::NOT_IMPLEMENTED,
                format!(
                    "controlnet {slug:?} is accepted and validated but not applied: the configured {:?} pipeline does not apply control images — the request is refused rather than rendered without its conditioning image",
                    pipeline.backend_id()
                ),
                Some("controlnet_unsupported"),
            )),
        );
    }
    (
        Some(ResolvedControlNet {
            weights: PathBuf::from(&entry.path),
            image: decoded,
            strength,
            control_type,
        }),
        None,
    )
}

/// Write one HTTP generation's job bundle (best-effort: a bundle-write
/// failure never fails an otherwise successful generation).
fn write_generation_bundle(
    config: &AppConfig,
    inputs: &crate::jobs::BundleInputs,
    bytes: &[u8],
    elapsed_ms: u64,
) {
    let Some(data_dir) = &config.data_dir else {
        return;
    };
    if let Err(err) = crate::jobs::write_bundle(data_dir, inputs, bytes, elapsed_ms) {
        tracing::warn!(error = %err, "job bundle write failed for HTTP generation");
    }
}

/// Default generation recipe for the HTTP surface: the same defaults as
/// `honkbox generate` (seed 42 base, 20 steps, euler/normal, CFG 7.0),
/// recorded in the bundle sidecars. Per-image seed = base + index; a
/// request may override the recipe (`sampler`/`scheduler`/`cfg_scale`/
/// `negative_prompt`/`seed`) or name a curated `recipe` — explicit fields
/// win over recipe defaults, and omitted fields fall back to these server
/// defaults.
const IMAGE_SEED_BASE: u64 = 42;
/// Server-default diffusion steps for the HTTP surface (CLI default): used
/// when the request names no recipe and passes no `steps` override.
const IMAGE_STEPS: u32 = 20;

/// Why an image generation stopped: a backend failure (500), or the job's
/// cancel token/flag interrupted the in-flight render — kept a named
/// variant so the route can answer the honest 409 `job_cancelled` instead
/// of a generic 500.
enum GenerateFailure {
    Cancelled,
    Failed(String),
}

/// One fresh engine-stop flag per image. The job token is the cross-image
/// cancel state; a flag left set by image 1 must never be inherited by
/// image 2 before it starts.
fn new_image_cancel_flag() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

/// 409 body for a cancel that landed before image `index` started. The
/// wording differs from the mid-generation arm so a client can tell that
/// no render was in flight.
fn image_cancel_before_start(job_id: &str, index: u64, total: u64, completed: u64) -> String {
    format!(
        "job {job_id} cancelled before image {index} of {total} started ({completed} of {total} image(s) generated)"
    )
}

/// 409 body for a cancel that aborted image `index` while it rendered.
fn image_cancel_mid_generation(job_id: &str, index: u64, total: u64, completed: u64) -> String {
    format!(
        "job {job_id} cancelled while generating image {index} of {total} ({completed} of {total} image(s) generated)"
    )
}

/// One PNG generation under `spawn_blocking` (GPU work off the async
/// runtime, like every other GPU route). Returns the encoded container
/// bytes, or why it stopped. `cancel` is THIS image's engine-stop flag and
/// `token` the job's cross-image registry token: the cancel endpoint can
/// only reach the token, so this future bridges the two — the moment the
/// token is marked the flag is set, and a pipeline polling the flag stops
/// the running sd.cpp generation through `sd_cancel_generation`. The bridge
/// waits for the worker to return, then re-checks the token so a backend
/// that ignores the flag cannot answer 200 after a cancel.
async fn generate_image_bytes(
    pipeline: std::sync::Arc<dyn honkbox_sd_core::SdPipeline>,
    txt2img: honkbox_sd_core::Txt2ImgRequest,
    format: honkbox_sd_core::OutputFormat,
    compression: Option<u8>,
    cancel: Arc<AtomicBool>,
    token: tokio_util::sync::CancellationToken,
) -> Result<Vec<u8>, GenerateFailure> {
    let worker_cancel = Arc::clone(&cancel);
    let mut worker = tokio::task::spawn_blocking(move || {
        let img = pipeline
            .generate(&txt2img, &worker_cancel)
            .map_err(|err| match err {
                honkbox_sd_core::SdError::Cancelled => GenerateFailure::Cancelled,
                other => GenerateFailure::Failed(other.to_string()),
            })?;
        match compression {
            Some(q) => honkbox_sd_core::encode_image_with_quality(img, format, q)
                .map_err(|err| GenerateFailure::Failed(err.to_string())),
            None => honkbox_sd_core::encode_image(img, format)
                .map_err(|err| GenerateFailure::Failed(err.to_string())),
        }
    });
    let join_failure = |err: tokio::task::JoinError| {
        GenerateFailure::Failed(format!("generation task failed: {err}"))
    };
    // Token → flag bridge: whichever lands first wins; a cancel makes the
    // worker return shortly (engine stop), and its answer — normally
    // `Cancelled` — decides the response.
    let outcome = tokio::select! {
        outcome = &mut worker => outcome.map_err(join_failure)?,
        () = token.cancelled() => {
            cancel.store(true, Ordering::Release);
            (&mut worker).await.map_err(join_failure)?
        }
    };
    // A backend that ignores the flag can still return Ok after the token
    // was marked. The token is the authority: never hand back a full image
    // for a cancelled request.
    if token.is_cancelled() {
        return Err(GenerateFailure::Cancelled);
    }
    outcome
}

/// Sidecar `scheduler` name for a resolved recipe: the fixed recipe's
/// discrete schedule keeps the historical `normal` recording name;
/// explicit requests record the sd.cpp wire spelling.
fn image_scheduler_name(scheduler: honkbox_sd_core::Scheduler) -> String {
    if scheduler == honkbox_sd_core::Scheduler::Discrete {
        honkbox_sd_core::SCHEDULER.to_string()
    } else {
        scheduler.name().to_string()
    }
}

/// `POST /v1/images/generations` (REQ-10): the `OpenAI` image-generation
/// shape over the startup-configured [`SdPipeline`]. SD pipelines are
/// startup-static (no swap manager), so a request `model` is validated and
/// recorded (bundle sidecar) but the pipeline stays the one built at
/// startup — with the stub backend any slug is honest; with `sd-cpp` the
/// checkpoint is the `--sd-model` one.
///
/// `output_format` selects the container for the returned bytes (`png`
/// default, `jpeg`, `webp` lossless; unknown names are a 400 naming it) and
/// the bundle sidecar/extension agree with it. `output_compression` is the
/// JPEG quality `[0, 100]`; non-numeric/out-of-range values are a named 400,
/// and the knob is ignored (as documented) for lossless PNG/WebP.
///
/// Each image in the request is generated under the GPU lease
/// (`spawn_blocking`, like every other GPU route) and written to the
/// existing job bundle store (`<data_dir>/jobs/`), so CLI GC and
/// recordkeeping cover HTTP generations too. Errors are `OpenAI`
/// envelopes: 400 bad `size`/`response_format`/`output_format`/
/// `output_compression`/recipe params (unknown
/// recipe/sampler/scheduler name, `cfg_scale` outside [1.0, 30.0], `steps`
/// outside [1, 100], invalid `hires_fix` scale/target/denoise, `LoRA` slug
/// / strength / count violations), 404 unknown model, 429 busy, 500
/// backend failure. `n` clamps to `[1, 4]` (warn, never reject).
/// Per-request `loras` resolve against `ModelKind::Lora` catalog entries;
/// their license notes surface as response `warnings` (weights are never
/// bundled; ReviewedLicense-AGPL §2b/§2c). `recipe` selects a curated
/// sampler/scheduler/cfg/steps bundle (per-field overrides win, then
/// server defaults); `hires_fix` runs the vendor-native second-pass
/// upscale and is recorded in the bundle sidecar. A valid `controlnet` +
/// `control_image` pair (plus optional `controlnet_strength`) is applied by
/// the sd-cpp pipeline — the weights load with the model context and the
/// conditioning image steers the render — while backends without a
/// `ControlNet` branch get the named 501 instead.
// Route handlers read long but stay linear: parse → validate → resolve →
// generate → bundle. `completions_stream` carries the same allow.
#[allow(clippy::too_many_lines)]
#[utoipa::path(
    post,
    path = "/v1/images/generations",
    request_body = ImagesGenerationRequest,
    responses(
        (status = 200, description = "Generated images (base64-encoded bytes in the requested output_format)", body = ImagesGenerationResponse),
        (status = 400, description = "Bad size, response_format, output_format, output_compression, recipe, sampler, scheduler, cfg_scale, steps, hires_fix, loras, or controlnet params (pair, image, or slug)", body = OpenAiError),
        (status = 404, description = "Unknown model", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Generation failed", body = OpenAiError),
        (status = 501, description = "Valid controlnet params accepted but not applicable by the sd pipeline (controlnet_unsupported)", body = OpenAiError)
    )
)]
async fn post_images_generations(
    State(config): State<AppConfig>,
    Json(req): Json<ImagesGenerationRequest>,
) -> Response {
    let (width, height, n, recipe, hires, format, compression) = match validate_images_request(&req)
    {
        Ok(valid) => valid,
        Err((message, code)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
    };
    let catalog = config.catalog_snapshot();
    let (pipeline_loras, lora_records, lora_warnings) =
        match resolve_loras(req.loras.as_deref().unwrap_or_default(), &catalog) {
            Ok(resolved) => resolved,
            Err((message, code)) => {
                return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
            }
        };
    // ControlNet (honkbox extension): slug + conditioning image are
    // validated here; a fully valid pair rides the generation request when
    // the configured pipeline can apply it, and pipelines that cannot are
    // answered with the honest 501 ([`resolve_controlnet`]) — never a
    // silent ignore that would render something else than what was asked
    // for.
    let (controlnet, controlnet_refusal) = resolve_controlnet(&req, &catalog, config.sd.as_ref());
    if let Some(response) = controlnet_refusal {
        return response;
    }
    let slug: String = match req
        .model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
    {
        Some(model) => match resolve_sd_model(model, &catalog) {
            Some(slug) => slug.to_owned(),
            None => {
                return openai_error_response(
                    StatusCode::NOT_FOUND,
                    format!(
                        "unknown model {model:?}: not an SD catalog slug or alias (see GET /v1/models)"
                    ),
                    Some("model_not_found"),
                );
            }
        },
        // Server default: the startup-configured SD pipeline.
        None => config.sd_model.clone().unwrap_or_default(),
    };
    let pipeline = Arc::clone(&config.sd);
    let prompt = req.prompt;
    let started = Instant::now();
    // NEW-3, engine level: the images job is registry-visible and
    // interruptible — the loop checks the cancel token BETWEEN images, and
    // the running sd.cpp generation itself is aborted mid-flight through
    // `sd_cancel_generation` (a per-image flag bridges the token into the
    // pipeline seam). Admission comes FIRST: a 429 has no registered job,
    // no lifecycle events and no X-Job-Id.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Sd).await,
        crate::jobs_registry::JobKind::Images,
        Some(slug.clone()),
        true,
    ) {
        Ok(pair) => pair,
        Err((status, message)) => return openai_error_response(status, message, None),
    };
    let mut images = Vec::with_capacity(n as usize);
    let backend_id = pipeline.backend_id();
    // Provenance (TASK-determinism-contract): the verified digest sidecar
    // exists only when the model arrived as a checksum-verified download;
    // locally-placed or unverified weights stay null — never a fabricated
    // hash. The vendored engine identity maps from the backend id (None
    // for the stub, which has no separate engine).
    let model_sha256 = resolve_model_entry(&slug, &catalog)
        .and_then(|entry| crate::downloads::artifact_sha256(std::path::Path::new(&entry.path)));
    let engine_version = honkbox_sd_core::engine_version_for(backend_id);
    // The token is the cross-image cancel state; the engine-stop flag is
    // fresh per image (created inside the loop). Reusing one flag would
    // let a cancel of image 1 be inherited by image 2 before it starts.
    let job_token = job.token().clone();
    for i in 0..u64::from(n) {
        // Cooperative cancellation (NEW-3): the token is checked BETWEEN
        // images here — a cancel marked while an earlier image rendered is
        // refused before this one starts — and, through this image's fresh
        // flag, DURING an image: the engine aborts its sampling loop and
        // the worker below answers the same named 409.
        if job.token().is_cancelled() {
            return with_job_id(
                openai_error_response(
                    StatusCode::CONFLICT,
                    image_cancel_before_start(job.id(), i + 1, u64::from(n), i),
                    Some("job_cancelled"),
                ),
                job.id(),
            );
        }
        // Fresh engine-stop flag for THIS image only; the token above is
        // the state shared across the batch.
        let cancel = new_image_cancel_flag();
        // Per-image seeds keep n>1 outputs distinct while staying
        // deterministic per request (explicit seed base when given).
        let seed = recipe.seed_base.wrapping_add(i);
        let bytes = match generate_image_bytes(
            Arc::clone(&pipeline),
            honkbox_sd_core::Txt2ImgRequest {
                seed,
                steps: recipe.steps,
                prompt: prompt.clone(),
                width,
                height,
                model_slug: slug.clone(),
                clip: None,
                vae: None,
                sampler: Some(recipe.sampler),
                scheduler: Some(recipe.scheduler),
                cfg_scale: Some(recipe.cfg_scale),
                negative_prompt: recipe.negative_prompt.clone(),
                loras: (!pipeline_loras.is_empty()).then(|| pipeline_loras.clone()),
                init_image: None,
                mask_image: None,
                strength: None,
                hires: hires.clone(),
                control_net: controlnet.as_ref().map(|c| c.weights.clone()),
                control_image: controlnet.as_ref().map(|c| c.image.clone()),
                controlnet_strength: controlnet.as_ref().and_then(|c| c.strength),
                control_type: controlnet.as_ref().and_then(|c| c.control_type),
            },
            format,
            compression,
            Arc::clone(&cancel),
            job_token.clone(),
        )
        .await
        {
            Ok(bytes) => bytes,
            Err(GenerateFailure::Cancelled) => {
                // The in-flight render was aborted mid-sampling: answer the
                // named 409, honestly counting the interrupted image as NOT
                // generated. A cancel that landed before this image started
                // takes the pre-start branch above instead.
                return with_job_id(
                    openai_error_response(
                        StatusCode::CONFLICT,
                        image_cancel_mid_generation(job.id(), i + 1, u64::from(n), i),
                        Some("job_cancelled"),
                    ),
                    job.id(),
                );
            }
            Err(GenerateFailure::Failed(message)) => {
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    message,
                    Some("generation_failed"),
                );
            }
        };
        write_generation_bundle(
            &config,
            &crate::jobs::BundleInputs {
                model_slug: slug.clone(),
                prompt: prompt.clone(),
                seed,
                steps: recipe.steps,
                width,
                height,
                format,
                backend_id,
                model_sha256: model_sha256.clone(),
                engine_version,
                sampler: recipe.sampler.name().to_string(),
                scheduler: image_scheduler_name(recipe.scheduler),
                cfg_scale: recipe.cfg_scale,
                negative_prompt: recipe.negative_prompt.clone(),
                loras: lora_records.clone(),
                strength: None,
                recipe: recipe.name.map(str::to_owned),
                hires: hires.as_ref().map(|fix| crate::jobs::HiresRecord {
                    upscaler: "latent",
                    target: {
                        let (tw, th) = honkbox_sd_core::hires_target((width, height), fix);
                        format!("{tw}x{th}")
                    },
                    scale: fix.scale,
                    denoising_strength: fix.denoising_strength,
                }),
            },
            &bytes,
            u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        );
        images.push(ImageB64 {
            b64_json: base64::engine::general_purpose::STANDARD.encode(&bytes),
        });
    }
    drop(lease);
    with_job_id(
        Json(ImagesGenerationResponse {
            created: epoch_secs(),
            data: images,
            warnings: (!lora_warnings.is_empty()).then_some(lora_warnings),
        })
        .into_response(),
        job.id(),
    )
}

/// Output dimensions from a PNG's IHDR (big-endian `u32` at offsets 16/20).
/// Only called on registry-verified bytes (`require_png_bytes` /
/// `crate::formats::identify_png` own the magic check); `None` for
/// truncated data.
fn png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 24 {
        return None;
    }
    Some((
        u32::from_be_bytes(bytes[16..20].try_into().ok()?),
        u32::from_be_bytes(bytes[20..24].try_into().ok()?),
    ))
}

/// Validate PNG-required bytes (edits `mask`; `ControlNet` `control_image`):
/// the registry identifies by magic, and `need_ihdr` additionally demands a
/// parseable IHDR. `Err` is the message reason — it names the sniffed
/// format, never an extension. (`image` runs the same sniff inline because
/// it also consumes the IHDR dimensions.)
fn require_png_bytes(bytes: &[u8], need_ihdr: bool) -> Result<(), String> {
    let sniffed = crate::formats::identify_png(bytes);
    if sniffed.is_err() || (need_ihdr && png_dimensions(bytes).is_none()) {
        return Err(match sniffed {
            Err(reject) => reject.detail(),
            Ok(_) => "IHDR dimensions missing".to_owned(),
        });
    }
    Ok(())
}

/// Collected multipart fields for `/v1/images/edits`.
struct ImageEditUpload {
    /// `image` part: the source PNG (declared content type + bytes).
    image: Option<(String, Bytes)>,
    /// `mask` part: optional inpaint-mask PNG.
    mask: Option<(String, Bytes)>,
    /// Render prompt (required).
    prompt: Option<String>,
    /// Optional SD model (alias or catalog slug).
    model: Option<String>,
    /// img2img denoise strength in `(0, 1]`; default
    /// [`honkbox_sd_core::IMG2IMG_STRENGTH`] (the vendor default).
    strength: Option<f32>,
    /// Output size `WxH`; default = the source image's dimensions.
    size: Option<String>,
    /// Recipe overrides, validated exactly like `generations` (a named
    /// recipe plus per-field overrides; explicit fields win).
    recipe: Option<String>,
    sampler: Option<String>,
    scheduler: Option<String>,
    cfg_scale: Option<f32>,
    negative_prompt: Option<String>,
    seed: Option<u64>,
    /// Diffusion steps override; validated like `generations`.
    steps: Option<u32>,
    /// Output container: `png` (default) | `jpeg` | `webp`; validated in
    /// the handler (named 400) like `generations`.
    output_format: Option<String>,
    /// JPEG quality `[0, 100]`; kept raw here so the handler can reject
    /// non-numeric / out-of-range values with the named 400, never a clamp.
    output_compression: Option<String>,
    /// Per-request `LoRA` adapters, the same validated `ImagesLora` list as
    /// txt2img (max 2, strength `[-2, 2]`); parsed as a JSON array from the
    /// form field and resolved by [`resolve_loras`] — no separate edits-side
    /// validation.
    loras: Option<Vec<ImagesLora>>,
}

/// Collect the multipart fields. Unknown fields are ignored (client SDKs
/// add extras). The boxed error is the ready-made `OpenAI` envelope.
async fn parse_image_edit_upload(
    mut multipart: Multipart,
) -> Result<ImageEditUpload, Box<Response>> {
    let mut edit = ImageEditUpload {
        image: None,
        mask: None,
        prompt: None,
        model: None,
        strength: None,
        size: None,
        recipe: None,
        sampler: None,
        scheduler: None,
        cfg_scale: None,
        negative_prompt: None,
        seed: None,
        steps: None,
        output_format: None,
        output_compression: None,
        loras: None,
    };
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| bad_multipart(format!("invalid multipart body: {err}")))?
    {
        let name = field.name().unwrap_or("").to_owned();
        match name.as_str() {
            "image" | "mask" => {
                let content_type = field.content_type().unwrap_or("").to_owned();
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|err| bad_multipart(format!("unreadable {name} field: {err}")))?;
                if name == "image" {
                    edit.image = Some((content_type, bytes));
                } else {
                    edit.mask = Some((content_type, bytes));
                }
            }
            "prompt" | "model" | "size" | "recipe" | "sampler" | "scheduler"
            | "negative_prompt" | "strength" | "cfg_scale" | "seed" | "steps" | "output_format"
            | "output_compression" | "loras" => {
                let text = field
                    .text()
                    .await
                    .map_err(|err| bad_multipart(format!("unreadable {name} field: {err}")))?;
                set_edit_field(&mut edit, &name, &text)?;
            }
            _ => {}
        }
    }
    Ok(edit)
}

/// One non-file multipart field, with per-field parse validation (the
/// semantic range checks live in the handler, next to their 400 codes).
fn set_edit_field(edit: &mut ImageEditUpload, name: &str, text: &str) -> Result<(), Box<Response>> {
    let value = text.trim().to_owned();
    match name {
        "prompt" | "model" | "size" | "recipe" | "sampler" | "scheduler" | "negative_prompt"
        | "output_format" => {
            let value = (!value.is_empty()).then_some(value);
            match name {
                "prompt" => edit.prompt = value,
                "model" => edit.model = value,
                "size" => edit.size = value,
                "recipe" => edit.recipe = value,
                "sampler" => edit.sampler = value,
                "scheduler" => edit.scheduler = value,
                "output_format" => edit.output_format = value,
                _ => edit.negative_prompt = value,
            }
        }
        "strength" => {
            edit.strength =
                Some(value.parse().map_err(|_| {
                    bad_multipart("strength must be a number in (0, 1]".to_owned())
                })?);
        }
        "cfg_scale" => {
            edit.cfg_scale = Some(value.parse().map_err(|_| {
                bad_multipart("cfg_scale must be a number in [1.0, 30.0]".to_owned())
            })?);
        }
        "seed" => {
            edit.seed =
                Some(value.parse().map_err(|_| {
                    bad_multipart("seed must be a non-negative integer".to_owned())
                })?);
        }
        "steps" => {
            edit.steps =
                Some(value.parse().map_err(|_| {
                    bad_multipart("steps must be an integer in [1, 100]".to_owned())
                })?);
        }
        "output_compression" => {
            edit.output_compression = (!value.is_empty()).then_some(value);
        }
        "loras" => {
            // Same shape as txt2img (`[{slug, strength}, ...]`); the semantic
            // checks (count, range, catalog, single dir) all live in
            // [`resolve_loras`], next to their 400 codes.
            let parsed: Vec<ImagesLora> = serde_json::from_str(&value).map_err(|err| {
                bad_multipart(format!(
                    "loras must be a JSON array of {{slug, strength}} objects: {err}"
                ))
            })?;
            edit.loras = Some(parsed);
        }
        _ => {}
    }
    Ok(())
}

/// `POST /v1/images/edits`: pragmatic honkbox img2img/inpaint shape over
/// the same SD pipeline seam as `/v1/images/generations` (single image per
/// request, `n` fixed at 1). `image` is a required PNG (magic-sniffed; the
/// declared content type is not trusted), `mask` an optional PNG that
/// turns the run into inpainting; `prompt` is required; `strength` (the
/// denoise amount, `(0, 1]`, default 0.75) applies to the init image;
/// `size` defaults to the source image's dimensions; the remaining recipe
/// fields (`recipe`/`sampler`/`scheduler`/`cfg_scale`/`steps`/
/// `negative_prompt`/`seed`) validate exactly like `generations` — a named
/// recipe plus per-field overrides, explicit fields winning. `output_format`
/// (`png` default | `jpeg` | `webp` lossless) and `output_compression`
/// (JPEG quality `[0, 100]`; ignored for lossless PNG/WebP) mirror the
/// `generations` knobs — unknown format names and non-numeric/out-of-range
/// compression values are named 400s, never clamps. No
/// `hires_fix` on this route: the output stays at the source image's
/// dimensions. Inputs are staged as temp
/// PNG paths (the vendor reads images from the filesystem) and removed
/// after the generation, including on failure. `loras` (a JSON array of
/// `{slug, strength}` objects) resolves through the same
/// [`resolve_loras`] as `generations` — max 2, strength `[-2, 2]`
/// reject-not-clamp, catalog lookup, single directory — with license notes
/// surfaced as response `warnings` and the resolved slugs/strengths recorded
/// in the bundle sidecar. The stub
/// backend honestly ignores image inputs, so its pixels never depend on
/// the upload; tests cover the wire/validation paths. Errors are `OpenAI`
/// envelopes: 400 (missing image/prompt, non-PNG bytes, bad
/// `strength`/`size`/`recipe`/`output_format`/`output_compression`/`loras`
/// fields), 404
/// unknown model, 429 busy, 500 staging or generation failure.
#[allow(clippy::too_many_lines)] // linear multipart handler; see `post_images_generations`
#[utoipa::path(
    post,
    path = "/v1/images/edits",
    request_body(content = String, description = "multipart/form-data: image (PNG, required), mask? (PNG), prompt (required), model?, strength?, size?, recipe?, sampler?, scheduler?, cfg_scale?, steps?, negative_prompt?, seed?, output_format? (png|jpeg|webp), output_compression? (JPEG quality 0-100), loras? (JSON array of {slug, strength})"),
    responses(
        (status = 200, description = "Edited image (base64-encoded bytes in the requested output_format)", body = ImagesGenerationResponse),
        (status = 400, description = "Missing image/prompt, non-PNG bytes, bad strength, size, recipe, output_format, output_compression, or loras fields", body = OpenAiError),
        (status = 404, description = "Unknown model", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Generation failed", body = OpenAiError)
    )
)]
async fn post_images_edits(State(config): State<AppConfig>, multipart: Multipart) -> Response {
    let edit = match parse_image_edit_upload(multipart).await {
        Ok(edit) => edit,
        Err(response) => return *response,
    };
    let Some((_, image_bytes)) = edit.image else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "image is required: attach the source PNG as the `image` form field".to_owned(),
            Some("image_required"),
        );
    };
    // Registry authority: identify by magic — never the declared content
    // type or the multipart filename — then require a real canvas (IHDR
    // parses to non-zero dimensions; it doubles as the `size` default).
    let sniffed = crate::formats::identify_png(&image_bytes);
    let dims = sniffed
        .as_ref()
        .ok()
        .and_then(|_| png_dimensions(&image_bytes))
        .filter(|(w, h)| *w > 0 && *h > 0);
    if dims.is_none() {
        let reason = sniffed
            .err()
            .map_or("IHDR dimensions missing or zero".to_owned(), |reject| {
                reject.detail()
            });
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("image must be PNG bytes ({reason})"),
            Some("invalid_image"),
        );
    }
    if let Some((_, mask_bytes)) = edit.mask.as_ref()
        && let Err(reason) = require_png_bytes(mask_bytes, true)
    {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("mask must be PNG bytes ({reason})"),
            Some("invalid_mask"),
        );
    }
    let Some(prompt) = edit.prompt else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "prompt is required".to_owned(),
            Some("prompt_required"),
        );
    };
    let strength = match edit.strength {
        Some(v) if v.is_finite() && (0.0..=1.0).contains(&v) && v > 0.0 => v,
        Some(v) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("strength {v} outside accepted range (0, 1]: rejected, not clamped"),
                Some("invalid_strength"),
            );
        }
        None => honkbox_sd_core::IMG2IMG_STRENGTH,
    };
    let (width, height) = match edit.size.as_deref() {
        Some(raw) => match crate::jobs::parse_size(raw) {
            Ok(size) => size,
            Err(err) => {
                return openai_error_response(StatusCode::BAD_REQUEST, err, Some("invalid_size"));
            }
        },
        // Default: the source image's own dimensions (PNG IHDR, verified
        // by the sniff above).
        None => dims.expect("PNG dimensions verified above"),
    };
    let recipe = match validate_recipe(
        edit.recipe.as_deref(),
        edit.sampler.as_deref(),
        edit.scheduler.as_deref(),
        edit.cfg_scale,
        edit.negative_prompt.as_deref(),
        edit.seed,
        edit.steps,
    ) {
        Ok(recipe) => recipe,
        Err((message, code)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
    };
    let format = match parse_output_format(edit.output_format.as_deref().unwrap_or("png").trim()) {
        Ok(format) => format,
        Err((message, code)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
    };
    let compression = match edit.output_compression.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(raw) => match raw.parse::<u8>() {
            Ok(q) if q <= 100 => Some(q),
            _ => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "output_compression must be an integer in [0, 100]; got {raw:?}: rejected, not clamped"
                    ),
                    Some("invalid_output_compression"),
                );
            }
        },
    };
    let catalog = config.catalog_snapshot();
    // Per-request `loras`: validated and resolved by the shared
    // [`resolve_loras`] (max 2, strength `[-2, 2]` reject-not-clamp, catalog
    // lookup, single directory) — identical to `/v1/images/generations`.
    let (pipeline_loras, lora_records, lora_warnings) =
        match resolve_loras(edit.loras.as_deref().unwrap_or_default(), &catalog) {
            Ok(resolved) => resolved,
            Err((message, code)) => {
                return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
            }
        };
    let slug: String = match edit
        .model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
    {
        Some(model) => match resolve_sd_model(model, &catalog) {
            Some(slug) => slug.to_owned(),
            None => {
                return openai_error_response(
                    StatusCode::NOT_FOUND,
                    format!(
                        "unknown model {model:?}: not an SD catalog slug or alias (see GET /v1/models)"
                    ),
                    Some("model_not_found"),
                );
            }
        },
        // Server default: the startup-configured SD pipeline.
        None => config.sd_model.clone().unwrap_or_default(),
    };
    // Shared admission gate (REQ-6 Option A): same slot acquisition as the
    // generations route (lease dropped at the route ends below).
    let lease = match crate::admission::acquire(honkbox_residency::ResourceKind::Sd).await {
        Ok(lease) => lease,
        Err((status, message)) => return openai_error_response(status, message, None),
    };
    // Stage the inputs as scratch PNGs (the vendor reads images from
    // paths); both are removed after the generation, on every path below.
    let token = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let init_path = std::env::temp_dir().join(format!(
        "honkbox-edit-init-{}-{token}.png",
        std::process::id()
    ));
    let mask_path = std::env::temp_dir().join(format!(
        "honkbox-edit-mask-{}-{token}.png",
        std::process::id()
    ));
    let staged_mask = edit.mask.is_some();
    let staging =
        std::fs::write(&init_path, &image_bytes).and_then(|()| match edit.mask.as_ref() {
            Some((_, mask_bytes)) => std::fs::write(&mask_path, mask_bytes),
            None => Ok(()),
        });
    if let Err(err) = staging {
        let _ = std::fs::remove_file(&init_path);
        let _ = std::fs::remove_file(&mask_path);
        drop(lease);
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to stage input image: {err}"),
            Some("staging_failed"),
        );
    }
    let pipeline = Arc::clone(&config.sd);
    let backend_id = pipeline.backend_id();
    // Provenance (TASK-determinism-contract): verified-digest sidecar when
    // the model was a checksum-verified download; null otherwise (never a
    // fabricated hash). Vendored engine identity from the backend id
    // (None for the stub — no separate engine).
    let model_sha256 = resolve_model_entry(&slug, &catalog)
        .and_then(|entry| crate::downloads::artifact_sha256(std::path::Path::new(&entry.path)));
    let engine_version = honkbox_sd_core::engine_version_for(backend_id);
    let started = Instant::now();
    let generated = generate_image_bytes(
        pipeline,
        honkbox_sd_core::Txt2ImgRequest {
            prompt: prompt.clone(),
            seed: recipe.seed_base,
            steps: recipe.steps,
            width,
            height,
            model_slug: slug.clone(),
            clip: None,
            vae: None,
            sampler: Some(recipe.sampler),
            scheduler: Some(recipe.scheduler),
            cfg_scale: Some(recipe.cfg_scale),
            negative_prompt: recipe.negative_prompt.clone(),
            loras: (!pipeline_loras.is_empty()).then(|| pipeline_loras.clone()),
            init_image: Some(init_path.clone()),
            mask_image: staged_mask.then(|| mask_path.clone()),
            strength: Some(strength),
            // No hires pass on img2img: the vendor applies hires to the
            // txt2img branch only, and the route fixes the output at the
            // source image's dimensions.
            hires: None,
            // ControlNet is a generations-route extension; img2img keeps
            // the plain OpenAI edits shape.
            control_net: None,
            control_image: None,
            controlnet_strength: None,
            control_type: None,
        },
        format,
        compression,
        // The edits route registers no job, so nothing can cancel it: a
        // fresh flag and a never-cancelled token (the pipeline never
        // observes a cancel here).
        Arc::new(AtomicBool::new(false)),
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let _ = std::fs::remove_file(&init_path);
    if staged_mask {
        let _ = std::fs::remove_file(&mask_path);
    }
    let bytes = match generated {
        Ok(bytes) => bytes,
        Err(failure) => {
            // `Cancelled` cannot happen with a never-set flag; the arm
            // keeps the match exhaustive without inventing a success.
            let message = match failure {
                GenerateFailure::Failed(message) => message,
                GenerateFailure::Cancelled => String::from("generation cancelled"),
            };
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                message,
                Some("generation_failed"),
            );
        }
    };
    write_generation_bundle(
        &config,
        &crate::jobs::BundleInputs {
            model_slug: slug,
            prompt,
            seed: recipe.seed_base,
            steps: recipe.steps,
            width,
            height,
            format,
            backend_id,
            model_sha256,
            engine_version,
            sampler: recipe.sampler.name().to_string(),
            scheduler: image_scheduler_name(recipe.scheduler),
            cfg_scale: recipe.cfg_scale,
            negative_prompt: recipe.negative_prompt,
            loras: lora_records,
            strength: Some(strength),
            recipe: recipe.name.map(str::to_owned),
            hires: None,
        },
        &bytes,
        u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    );
    drop(lease);
    Json(ImagesGenerationResponse {
        created: epoch_secs(),
        data: vec![ImageB64 {
            b64_json: base64::engine::general_purpose::STANDARD.encode(&bytes),
        }],
        warnings: (!lora_warnings.is_empty()).then_some(lora_warnings),
    })
    .into_response()
}

/// Owning catalog entry for a request `model` value (the alias `id` or the
/// registry `slug`); [`resolve_model_slug`] reads the slug off it.
fn resolve_model_entry<'a>(
    model: &str,
    catalog: &'a ModelCatalog,
) -> Option<&'a ModelCatalogEntry> {
    catalog
        .data
        .iter()
        .find(|e| e.id == model || e.slug == model)
}

/// Unix seconds (the `OpenAI` `created` field).
fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `finish_reason` for a completed reply. The worker does not surface a
/// native stop cause, so `length` is the documented heuristic: reported
/// generated tokens reached the requested `max_tokens` cap.
fn finish_reason(tokens: u32, max_tokens: Option<u32>) -> FinishReason {
    if max_tokens.is_some_and(|max| tokens >= max) {
        FinishReason::Length
    } else {
        FinishReason::Stop
    }
}

/// Token accounting from a worker reply.
fn usage_of(reply: &honkbox_llm_worker::ChatResponse) -> OpenAiUsage {
    OpenAiUsage {
        prompt_tokens: reply.prompt_tokens,
        completion_tokens: reply.tokens,
        total_tokens: u64::from(reply.prompt_tokens) + u64::from(reply.tokens),
    }
}

/// `POST /v1/chat/completions`: the `OpenAI`-compatible translation surface
/// (REQ-1) over the SAME pipeline, GPU lease, and swap decision as
/// `/v1/chat` — never an HTTP proxy. JSON body or `OpenAI` SSE
/// (`stream: true`). Errors are `OpenAI` envelopes; statuses: 400 bad
/// request, 404 unknown model, 429 busy (+`Retry-After` via the shared
/// middleware), 500 inference failure. Undeclared request keys (`top_p`,
/// `stop`, …) are accepted and ignored by serde.
///
/// An omitted/empty `model` is a 400 naming the requirement — the native
/// route's empty-means-server-default fallback stays native-only.
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    request_body = OpenAiChatRequest,
    responses(
        (status = 200, description = "Chat completion (JSON or SSE)", body = OpenAiChatCompletion),
        (status = 400, description = "Bad request", body = OpenAiError),
        (status = 404, description = "Unknown model", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Inference failed", body = OpenAiError)
    )
)]
// ~105 lines: validation gates, constraint resolution, the swap decision,
// and the two wire shapes (JSON + SSE) share one handler; splitting the
// dispatch would break the REQ-6/NEW-3 ordering story (register, queue,
// branch on the stream flag) across two functions.
#[allow(clippy::too_many_lines)]
async fn post_chat_completions(State(config): State<AppConfig>, body: Bytes) -> Response {
    let req: OpenAiChatRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(err) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid JSON body: {err}"),
                None,
            );
        }
    };
    let Some(model) = req
        .model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
    else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "model is required: pass an operator alias or registry slug from GET /v1/models"
                .to_owned(),
            Some("model_required"),
        );
    };
    let catalog = config.catalog_snapshot();
    let Some(entry) = resolve_model_entry(model, &catalog) else {
        return openai_error_response(
            StatusCode::NOT_FOUND,
            format!("unknown model {model:?}: not an alias or registry slug (see GET /v1/models)"),
            Some("model_not_found"),
        );
    };
    let served_id = entry.id.clone();
    let slug = entry.slug.clone();
    // REQ-4 + REQ-8 request gates (see [`completions_request_gates`]).
    if let Some(response) = completions_request_gates(entry, model, &req.messages) {
        return response;
    }
    // `REQ-3` request surface: resolve tools + output constraints + sampler
    // knobs into the worker protocol (clamps and 400s are inside; nothing
    // is silently dropped).
    let constraints = match resolve_constraints(&req) {
        Ok(constraints) => constraints,
        Err((code, message)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
    };
    let worker_messages = match worker_messages(&req.messages) {
        Ok(messages) => messages,
        Err(message) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                message,
                Some("invalid_image_data"),
            );
        }
    };
    let worker_req = WorkerChatRequest {
        model: slug.clone(),
        messages: worker_messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        seed: req.seed,
        grammar: constraints.grammar,
        json_schema: constraints.json_schema,
        json_object: constraints.json_object,
        tools_json: constraints.tools_json,
        tool_choice_none: constraints.tool_choice_none,
        sampler: constraints.sampler,
    };
    // Same swap path as `/v1/chat`: load-on-demand through the swap manager
    // when active, else the startup pipeline. Unknown/non-LLM slugs are the
    // route's 404; construction failures 500.
    let pipe = match &config.swap {
        Some(swap) => {
            match swap
                .get(
                    &slug,
                    clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds),
                )
                .await
            {
                Ok(pipe) => pipe,
                Err(crate::swap::SwapError::UnknownSlug(message)) => {
                    return openai_error_response(StatusCode::NOT_FOUND, message, None);
                }
                Err(crate::swap::SwapError::Load(message)) => {
                    return openai_error_response(StatusCode::INTERNAL_SERVER_ERROR, message, None);
                }
            }
        }
        None => config.llm.clone(),
    };
    // NEW-3 + REQ-6: take the GPU slot ONCE through the admission queue for
    // either shape, then register the job (interruptible only on the
    // streaming path — the sync completion is one blocking FFI call). A
    // rejection keeps the OpenAI envelope + shared `Retry-After` hint but
    // creates no job, events or X-Job-Id.
    let streaming = req.stream.is_some_and(|stream| stream);
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await,
        crate::jobs_registry::JobKind::Chat,
        Some(slug.clone()),
        streaming,
    ) {
        Ok(pair) => pair,
        Err((status, message)) => {
            return openai_error_response(status, message, Some("gpu_slot_busy"));
        }
    };
    if streaming {
        // REQ-6 (Option B): the slot is held BEFORE the SSE body is
        // returned, so a busy slot is a real 429 — never a 200 whose only
        // content is an error frame (loop-lore's `data:`-only parser would
        // read that as a clean empty completion).
        completions_stream(
            lease,
            job,
            pipe,
            worker_req,
            &served_id,
            id_frame(),
            req.max_tokens,
        )
    } else {
        completions_json(lease, job, pipe, worker_req, &served_id, req.max_tokens).await
    }
}

/// Non-streaming completion: one blocking worker call inside the GPU lease,
/// mapped onto the exact `OpenAI` JSON shape.
async fn completions_json(
    lease: crate::admission::GpuLease,
    job: crate::jobs_registry::JobGuard,
    pipe: Arc<dyn LlmPipeline>,
    worker_req: WorkerChatRequest,
    served_id: &str,
    max_tokens: Option<u32>,
) -> Response {
    let reply = tokio::task::spawn_blocking(move || {
        let _lease = lease;
        pipe.complete(&worker_req).map_err(|err| {
            // `REQ-3`: unusable constraints (unparseable grammar, non-
            // convertible schema) and `REQ-8`: image content the loaded
            // pipeline cannot serve (worker-side vision gate, defense in
            // depth behind the route gate) are client errors, not server
            // ones; the envelope code names the class.
            match &err {
                LlmError::InvalidRequest(_) => (
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    format!("chat failed: {err}"),
                ),
                LlmError::Vision(_) => (
                    StatusCode::BAD_REQUEST,
                    "vision_unsupported",
                    format!("chat failed: {err}"),
                ),
                _ => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal",
                    format!("chat failed: {err}"),
                ),
            }
        })
    })
    .await
    .map_err(|err| format!("chat task failed: {err}"));
    let reply = match reply {
        Ok(Ok(reply)) => reply,
        Ok(Err((StatusCode::BAD_REQUEST, code, message))) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
        Ok(Err((_, _, message))) | Err(message) => {
            return openai_error_response(StatusCode::INTERNAL_SERVER_ERROR, message, None);
        }
    };
    let usage = usage_of(&reply);
    let tool_calls = reply
        .tool_calls
        .iter()
        .map(|call| ToolCall {
            // Unique per response (REQ-3 acceptance); echoed back via
            // `tool_call_id` on the follow-up `role: "tool"` message.
            id: format!("call_{}", ulid::Ulid::new()),
            call_type: ToolCallType::Function,
            function: FunctionCall {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            },
        })
        .collect::<Vec<_>>();
    let finish = if tool_calls.is_empty() {
        finish_reason(reply.tokens, max_tokens)
    } else {
        FinishReason::ToolCalls
    };
    let content = if tool_calls.is_empty() || !reply.text.is_empty() {
        ChatContent::text(reply.text)
    } else {
        // Tool-call-only reply: `OpenAI` sends `content: null` there.
        ChatContent::default()
    };
    with_job_id(
        Json(OpenAiChatCompletion {
            id: format!("chatcmpl-{}", ulid::Ulid::new()),
            object: "chat.completion".to_owned(),
            created: epoch_secs(),
            model: served_id.to_owned(),
            choices: vec![OpenAiChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content,
                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                    tool_call_id: None,
                },
                finish_reason: finish,
            }],
            usage,
        })
        .into_response(),
        job.id(),
    )
}

/// Fresh completion id + creation timestamp (shared by every SSE frame).
fn id_frame() -> (String, u64) {
    (format!("chatcmpl-{}", ulid::Ulid::new()), epoch_secs())
}

/// One SSE chunk frame (serialization never fails on these DTOs).
fn chunk_frame(frame: &OpenAiChunk) -> Event {
    let data = serde_json::to_string(frame).unwrap_or_else(|_| "{}".to_owned());
    Event::default().data(data)
}

/// `OpenAI` SSE translation of `complete_stream` (REQ-1): role+first-token
/// frame, one content frame per token, a final frame with `finish_reason`
/// and `usage`, then a literal `data: [DONE]`. No `event:` frames on the
/// success path — they are reserved for inference failures, which arrive
/// as an `event: error` data frame before `[DONE]` (the SSE body is
/// already 200 by then). The GPU-slot busy condition is NOT one of them:
/// REQ-6 (Option B) reserves the slot in the handler, so a busy slot
/// becomes a real HTTP 429 before this function runs — the `lease` here is
/// already held and handed to the worker.
// 105 lines: the SSE translation (chunks, tool-call deltas, finish reasons,
// [DONE]) is one linear pipeline; splitting it hurts readability.
#[allow(clippy::too_many_lines)]
fn completions_stream(
    lease: crate::admission::GpuLease,
    job: crate::jobs_registry::JobGuard,
    pipe: Arc<dyn LlmPipeline>,
    worker_req: WorkerChatRequest,
    served_id: &str,
    (id, created): (String, u64),
    max_tokens: Option<u32>,
) -> Response {
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let cancelled = Arc::new(AtomicBool::new(false));
    let lease_slot = Arc::new(Mutex::new(None::<crate::admission::GpuLease>));
    let abort_pipe = Arc::clone(&pipe);
    let worker_cancelled = cancelled.clone();
    let job_token = job.token().clone();
    let job_id = job.id().to_owned();
    let frame_model = served_id.to_owned();
    // `REQ-3`: with tools on the table, pieces are buffered until the
    // parsed tool calls are known, then framed as `delta.tool_calls`
    // fragments (arguments split across frames; per-index concatenation
    // yields the JSON string). Without tools, content streams
    // token-by-token as before — including `json_object` replies.
    let tool_streaming = worker_req.tools_json.is_some();
    let worker = tokio::task::spawn_blocking(move || {
        // The job guard rides with the worker (the registry entry lives as
        // long as the stream). The slot was taken in the handler (REQ-6);
        // park the lease where the token sink can drain it at the first
        // cancel/disconnect.
        let _job = job;
        *lease_slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(lease);
        let mut first = true;
        let mut buffered: Vec<String> = Vec::new();
        let result = pipe.complete_stream(&worker_req, &mut |token: &str| {
            if worker_cancelled.load(Ordering::Relaxed) || job_token.is_cancelled() {
                // Disconnect, or `POST /v1/jobs/{id}/cancel` (NEW-3): abort
                // the FFI loop itself, not just delivery.
                pipe.abort();
                free_gpu_lease(&lease_slot);
                return;
            }
            if tool_streaming {
                buffered.push(token.to_owned());
                return;
            }
            let delivered = send_token_delta(&tx, &id, created, &frame_model, &mut first, token);
            if !delivered {
                // Receiver gone without the drop guard (drop raced the send).
                worker_cancelled.store(true, Ordering::Relaxed);
                pipe.abort();
                free_gpu_lease(&lease_slot);
            }
        });
        // A disconnect between the last token and completion frees here.
        free_gpu_lease(&lease_slot);
        match result {
            Ok(reply) => {
                let usage = usage_of(&reply);
                let finish = if reply.tool_calls.is_empty() {
                    finish_reason(reply.tokens, max_tokens)
                } else {
                    FinishReason::ToolCalls
                };
                if tool_streaming {
                    if reply.tool_calls.is_empty() {
                        // No call was made (e.g. `tool_choice: "none"`, or
                        // the model answered in prose): the buffered pieces
                        // stream as content now.
                        emit_content_stream(&tx, &id, created, &frame_model, &buffered);
                    } else {
                        // Tool-call reply: a role marker frame, then the
                        // parsed calls as fragments. The buffered raw
                        // pieces are superseded by the parser output.
                        emit_tool_stream(&tx, &id, created, &frame_model, &reply.tool_calls);
                    }
                }
                let final_frame = OpenAiChunk {
                    id,
                    object: "chat.completion.chunk".to_owned(),
                    created,
                    model: frame_model,
                    choices: vec![OpenAiChunkChoice {
                        index: 0,
                        delta: OpenAiDelta {
                            role: None,
                            content: None,
                            tool_calls: None,
                        },
                        finish_reason: Some(finish),
                    }],
                    usage: Some(usage),
                };
                let _ = tx.send(Ok(chunk_frame(&final_frame)));
            }
            Err(err) => {
                let body = OpenAiError::new(500, format!("chat failed: {err}"), None);
                let _ = tx.send(Ok(Event::default()
                    .event("error")
                    .data(serde_json::to_string(&body).unwrap_or_default())));
            }
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
    });
    with_job_id(
        Sse::new(CancelOnDrop {
            inner: UnboundedReceiverStream::new(rx),
            cancelled,
            pipe: abort_pipe,
            _worker: worker,
        })
        .into_response(),
        &job_id,
    )
}

/// Emit a buffered tools stream: a role marker frame, then the parsed
/// calls as `delta.tool_calls` fragments with fresh per-response ids.
fn emit_tool_stream(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    id: &str,
    created: u64,
    model: &str,
    tool_calls: &[ToolCallOut],
) {
    send_delta(
        tx,
        id,
        created,
        model,
        OpenAiDelta {
            role: Some(ChatRole::Assistant),
            content: None,
            tool_calls: None,
        },
    );
    let ids: Vec<String> = tool_calls
        .iter()
        .map(|_| format!("call_{}", ulid::Ulid::new()))
        .collect();
    for delta in tool_delta_frames(tool_calls, &ids) {
        send_delta(tx, id, created, model, delta);
    }
}

/// Emit buffered pieces as a plain content stream (the first piece carries
/// the role marker).
fn emit_content_stream(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    id: &str,
    created: u64,
    model: &str,
    pieces: &[String],
) {
    for (piece_index, piece) in pieces.iter().enumerate() {
        let delta = if piece_index == 0 {
            OpenAiDelta {
                role: Some(ChatRole::Assistant),
                content: Some(piece.clone()),
                tool_calls: None,
            }
        } else {
            OpenAiDelta {
                role: None,
                content: Some(piece.clone()),
                tool_calls: None,
            }
        };
        send_delta(tx, id, created, model, delta);
    }
}

/// Send one streamed content token (the role marker rides the first
/// piece). Returns `false` when the receiver is gone — the caller then
/// cancels the generation.
#[allow(clippy::needless_pass_by_ref_mut)] // `first` is mutated across calls
fn send_token_delta(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    id: &str,
    created: u64,
    model: &str,
    first: &mut bool,
    token: &str,
) -> bool {
    let delta = if *first {
        *first = false;
        OpenAiDelta {
            role: Some(ChatRole::Assistant),
            content: Some(token.to_owned()),
            tool_calls: None,
        }
    } else {
        OpenAiDelta {
            role: None,
            content: Some(token.to_owned()),
            tool_calls: None,
        }
    };
    let frame = OpenAiChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created,
        model: model.to_owned(),
        choices: vec![OpenAiChunkChoice {
            index: 0,
            delta,
            finish_reason: None,
        }],
        usage: None,
    };
    tx.send(Ok(chunk_frame(&frame))).is_ok()
}

/// Send one content/tool delta chunk (serialization never fails on these
/// DTOs; a gone receiver just drops the frame post-completion).
fn send_delta(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    id: &str,
    created: u64,
    model: &str,
    delta: OpenAiDelta,
) {
    let frame = OpenAiChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created,
        model: model.to_owned(),
        choices: vec![OpenAiChunkChoice {
            index: 0,
            delta,
            finish_reason: None,
        }],
        usage: None,
    };
    let _ = tx.send(Ok(chunk_frame(&frame)));
}

/// Split an arguments JSON string into frame-sized pieces (char-boundary
/// safe). Short payloads still split in half, so the streamed form always
/// exercises the fragment protocol (per-index concatenation reproduces the
/// original string).
fn argument_frames(arguments: &str) -> Vec<String> {
    let chars: Vec<char> = arguments.chars().collect();
    if chars.len() < 2 {
        return vec![arguments.to_owned()];
    }
    let per_frame = chars.len().div_ceil(4).max(2);
    chars
        .chunks(per_frame)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

/// `delta.tool_calls` frames for parsed calls (`REQ-3`): per call a first
/// fragment carrying `id`/`type`/`function.name` + empty `arguments`, then
/// continuation fragments with `function.arguments` pieces at the same
/// `index`. `ids` runs parallel to `tool_calls`.
fn tool_delta_frames(tool_calls: &[ToolCallOut], ids: &[String]) -> Vec<OpenAiDelta> {
    let mut deltas = Vec::new();
    for (index, (call, call_id)) in tool_calls.iter().zip(ids).enumerate() {
        let index = u32::try_from(index).unwrap_or(u32::MAX);
        deltas.push(OpenAiDelta {
            role: None,
            content: None,
            tool_calls: Some(vec![DeltaToolCall {
                index,
                id: Some(call_id.clone()),
                call_type: Some(ToolCallType::Function),
                function: Some(DeltaFunction {
                    name: Some(call.name.clone()),
                    arguments: Some(String::new()),
                }),
            }]),
        });
        for piece in argument_frames(&call.arguments) {
            deltas.push(OpenAiDelta {
                role: None,
                content: None,
                tool_calls: Some(vec![DeltaToolCall {
                    index,
                    id: None,
                    call_type: None,
                    function: Some(DeltaFunction {
                        name: None,
                        arguments: Some(piece),
                    }),
                }]),
            });
        }
    }
    deltas
}

/// `POST /v1/transcribe`: content-type dispatch (`audio/wav` → `decode_wav`,
/// `audio/pcm-s16le` → raw LE bytes + required `?sample_rate=`), then the
/// sync worker call inside `spawn_blocking`.
#[utoipa::path(
    post,
    path = "/v1/transcribe",
    responses(
        (status = 200, description = "Transcription", body = Transcript),
        (status = 400, description = "Unknown model, bad audio, or missing sample_rate"),
        (status = 415, description = "Use audio/wav or audio/pcm-s16le"),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Transcription failed")
    )
)]
async fn post_transcribe(
    State(config): State<AppConfig>,
    Query(query): Query<TranscribeQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    let catalog = config.catalog_snapshot();
    validate_model(
        query.model.as_deref(),
        config.stt_model.as_deref(),
        &catalog,
    )
    .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    let _ttl = clamp_ttl_seconds(config.swap_max_ttl, query.ttl_seconds);
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let (sample_rate, pcm): (u32, Vec<i16>) = match content_type.as_str() {
        "audio/wav" | "audio/wave" | "audio/x-wav" => decode_wav(&body)
            .map_err(|err| (StatusCode::BAD_REQUEST, format!("invalid wav: {err}")))?,
        "audio/pcm-s16le" => {
            let sample_rate = query.sample_rate.ok_or((
                StatusCode::BAD_REQUEST,
                "audio/pcm-s16le requires ?sample_rate=".to_owned(),
            ))?;
            if !body.len().is_multiple_of(2) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "odd audio/pcm-s16le length".to_owned(),
                ));
            }
            let (pairs, _) = body.as_chunks::<2>();
            let pcm = pairs.iter().map(|pair| i16::from_le_bytes(*pair)).collect();
            (sample_rate, pcm)
        }
        other => {
            return Err((
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                format!("unsupported content-type {other:?}; use audio/wav or audio/pcm-s16le"),
            ));
        }
    };
    let worker_req = WorkerTranscribeRequest {
        model: query.model.unwrap_or_default(),
        language: query.language,
        translate: false,
    };
    // NEW-3: registry-visible while the request lives (never interruptible:
    // one blocking FFI call; the token is only marked by a cancel).
    let job = crate::jobs_registry::global().register(
        crate::jobs_registry::JobKind::Transcribe,
        (!worker_req.model.is_empty()).then_some(worker_req.model.clone()),
        false,
    );
    let out = run_transcription(config.stt.clone(), worker_req, sample_rate, pcm).await?;
    Ok(with_job_id(
        Json(Transcript {
            backend_id: out.backend_id,
            text: out.text,
            segments: out
                .segments
                .into_iter()
                .map(|segment| TranscriptSegment {
                    start_ms: segment.start_ms,
                    end_ms: segment.end_ms,
                    text: segment.text,
                })
                .collect(),
            millis: out.millis,
        })
        .into_response(),
        job.id(),
    ))
}

/// `POST /v1/synthesize`: JSON in, WAV bytes out (`audio/wav`).
#[utoipa::path(
    post,
    path = "/v1/synthesize",
    request_body = SynthesizeRequest,
    responses(
        (status = 200, description = "WAV audio"),
        (status = 400, description = "Unknown model slug; blank `text`; `text` over the 4096-character cap (`text_too_long`)"),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Synthesis failed")
    )
)]
async fn post_synthesize(
    State(config): State<AppConfig>,
    Json(req): Json<SynthesizeRequest>,
) -> Result<Response, (StatusCode, String)> {
    // Same boundary as `/v1/audio/speech` (TASK-speech-input-cap): this is
    // the native twin of the OpenAI-shaped route — same worker, same
    // unbounded-PCM risk; reject-not-clamp (no silent truncation).
    if req.text.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "text is required: the text to synthesize".to_owned(),
        ));
    }
    let text_chars = req.text.chars().count();
    if text_chars > SPEECH_INPUT_MAX_CHARS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "text is {text_chars} characters; the cap is {SPEECH_INPUT_MAX_CHARS} — rejected, not truncated"
            ),
        ));
    }
    let catalog = config.catalog_snapshot();
    validate_model(Some(&req.model), config.tts_model.as_deref(), &catalog)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    let _ttl = clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds);
    let worker_req = WorkerSynthRequest {
        model: req.model.clone(),
        text: req.text,
        voice: req.voice,
        sample_rate: req.sample_rate,
    };
    // NEW-3: registry-visible while the request lives (one blocking FFI
    // call — the cancel token is marked but never checked).
    let job = crate::jobs_registry::global().register(
        crate::jobs_registry::JobKind::Synthesize,
        (!worker_req.model.is_empty()).then_some(worker_req.model.clone()),
        false,
    );
    let audio = run_synthesis(config.tts.clone(), worker_req).await?;
    let wav = encode_wav(audio.sample_rate, &audio.pcm16);
    Ok(with_job_id(
        ([(header::CONTENT_TYPE, "audio/wav")], wav).into_response(),
        job.id(),
    ))
}

/// Shared STT execution (native `/v1/transcribe` and REQ-9
/// `/v1/audio/transcriptions` are the same worker call under two wire
/// shapes): run the pipeline inside the GPU slot on the blocking pool. The
/// slot comes through the admission queue (REQ-6 Option A); the caller
/// registered the job (NEW-3) — one blocking FFI call, never interruptible.
/// Audio pipelines share the `llm` admission bucket: the fixed ledger kind
/// vocabulary has no audio kind, and `llm` is the warm-model-pipeline
/// bucket the audio workers already load alongside (the label is admission
/// topology, not model taxonomy).
async fn run_transcription(
    pipe: Arc<dyn SttPipeline>,
    worker_req: WorkerTranscribeRequest,
    sample_rate: u32,
    pcm: Vec<i16>,
) -> Result<WorkerTranscript, (StatusCode, String)> {
    let lease = crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await?;
    tokio::task::spawn_blocking(move || {
        let _lease = lease;
        pipe.transcribe(&pcm, sample_rate, &worker_req)
            .map_err(|err| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("transcribe failed: {err}"),
                )
            })
    })
    .await
    .map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("transcribe task failed: {err}"),
        )
    })?
}

/// Shared TTS execution (native `/v1/synthesize` and REQ-9
/// `/v1/audio/speech` are the same worker call under two wire shapes).
/// Audio pipelines share the `llm` admission bucket (see
/// [`run_transcription`] for why).
async fn run_synthesis(
    pipe: Arc<dyn TtsPipeline>,
    worker_req: WorkerSynthRequest,
) -> Result<SynthesizedAudio, (StatusCode, String)> {
    let lease = crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await?;
    tokio::task::spawn_blocking(move || {
        let _lease = lease;
        pipe.synthesize(&worker_req).map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("synthesize failed: {err}"),
            )
        })
    })
    .await
    .map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("synthesize task failed: {err}"),
        )
    })?
}

/// Decode an uploaded audio payload for the REQ-9 transcriptions route:
/// the same formats the native `/v1/transcribe` accepts (`audio/wav`,
/// `audio/pcm-s16le` + `sample_rate`), plus a RIFF sniff for
/// `application/octet-stream` parts (the `openai` SDK's default file
/// content type).
fn decode_upload_audio(
    content_type: &str,
    bytes: &[u8],
    sample_rate: Option<u32>,
) -> Result<(u32, Vec<i16>), (StatusCode, String)> {
    let mime = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match mime.as_str() {
        "audio/wav" | "audio/wave" | "audio/x-wav" => decode_wav(bytes)
            .map_err(|err| (StatusCode::BAD_REQUEST, format!("invalid wav: {err}"))),
        "audio/pcm-s16le" => {
            let sample_rate = sample_rate.ok_or((
                StatusCode::BAD_REQUEST,
                "audio/pcm-s16le requires a `sample_rate` form field or `?sample_rate=` query"
                    .to_owned(),
            ))?;
            if !bytes.len().is_multiple_of(2) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "odd audio/pcm-s16le length".to_owned(),
                ));
            }
            let (pairs, _) = bytes.as_chunks::<2>();
            let pcm = pairs.iter().map(|pair| i16::from_le_bytes(*pair)).collect();
            Ok((sample_rate, pcm))
        }
        "" | "application/octet-stream" | "binary/octet-stream" => {
            if bytes.starts_with(b"RIFF") {
                decode_wav(bytes)
                    .map_err(|err| (StatusCode::BAD_REQUEST, format!("invalid wav: {err}")))
            } else {
                Err((
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "unrecognized audio payload: send audio/wav (or a RIFF wav under application/octet-stream), or audio/pcm-s16le with a sample_rate".to_owned(),
                ))
            }
        }
        other => Err((
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            format!("unsupported file content-type {other:?}; use audio/wav or audio/pcm-s16le"),
        )),
    }
}

/// Collected multipart fields for `/v1/audio/transcriptions`.
struct TranscriptionUpload {
    /// `file` part: content type + bytes.
    file: Option<(String, Bytes)>,
    /// Optional server-default-overridable STT model.
    model: Option<String>,
    /// Optional language hint.
    language: Option<String>,
    /// `json` (default) | `verbose_json`.
    response_format: Option<String>,
    /// pcm-s16le input rate (form field; query supplies the fallback).
    sample_rate: Option<u32>,
}

/// Collect the multipart fields. Unknown fields are ignored (`OpenAI`
/// SDKs add `purpose` and friends). The boxed error is the ready-made
/// `OpenAI` envelope response.
async fn parse_transcription_upload(
    mut multipart: Multipart,
    sample_rate_query: Option<u32>,
) -> Result<TranscriptionUpload, Box<Response>> {
    let mut upload = TranscriptionUpload {
        file: None,
        model: None,
        language: None,
        response_format: None,
        sample_rate: sample_rate_query,
    };
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| bad_multipart(format!("invalid multipart body: {err}")))?
    {
        let name = field.name().unwrap_or("").to_owned();
        match name.as_str() {
            "file" => {
                let content_type = field.content_type().unwrap_or("").to_owned();
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|err| bad_multipart(format!("unreadable file field: {err}")))?;
                upload.file = Some((content_type, bytes));
            }
            "model" | "language" | "response_format" | "sample_rate" => {
                let text = field
                    .text()
                    .await
                    .map_err(|err| bad_multipart(format!("unreadable {name} field: {err}")))?;
                set_upload_field(&mut upload, name.as_str(), text)?;
            }
            _ => {}
        }
    }
    Ok(upload)
}

/// One non-file multipart field, with per-field validation.
fn set_upload_field(
    upload: &mut TranscriptionUpload,
    name: &str,
    text: String,
) -> Result<(), Box<Response>> {
    let value = (!text.trim().is_empty()).then_some(text);
    match name {
        "model" => upload.model = value,
        "language" => upload.language = value,
        "response_format" => upload.response_format = value,
        "sample_rate" => {
            match value
                .as_deref()
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .and_then(|text| text.parse::<u32>().ok())
            {
                Some(rate) => upload.sample_rate = Some(rate),
                None if value.is_some() => {
                    return Err(bad_multipart(
                        "sample_rate must be a positive integer (Hz)".to_owned(),
                    ));
                }
                None => {}
            }
        }
        _ => {}
    }
    Ok(())
}

/// Multipart parse failure → `OpenAI` 400 envelope. Deliberately boxed:
/// the `Result` error variant must stay small (`result_large_err`),
/// trading the `unnecessary_box_returns` nag.
#[allow(clippy::unnecessary_box_returns)]
fn bad_multipart(message: String) -> Box<Response> {
    Box::new(openai_error_response(
        StatusCode::BAD_REQUEST,
        message,
        None,
    ))
}

/// `POST /v1/audio/transcriptions` (REQ-9): `OpenAI`-shaped multipart
/// wrapper over the same STT worker as `/v1/transcribe`. `file` is
/// required (`audio/wav`, RIFF-sniffed `application/octet-stream`, or
/// `audio/pcm-s16le` + `sample_rate` form field / query); `model` is
/// optional (server default STT); `language` optional; `response_format`
/// `json` (default), `verbose_json` (adds the worker's timestamped
/// segments), `srt` or `vtt` (`SubRip` / `WebVTT` subtitles rendered from the
/// same segments; `application/x-subrip` / `text/vtt`). Errors are
/// `OpenAI` envelopes; only a request that is not multipart at all gets
/// axum's transport-level 400 before the handler runs. The native route is
/// unchanged.
#[utoipa::path(
    post,
    path = "/v1/audio/transcriptions",
    request_body(content = String, description = "multipart/form-data: file, model?, language?, response_format? (json|verbose_json|srt|vtt), sample_rate?"),
    responses(
        (status = 200, description = "Transcription (json / verbose_json / srt / vtt)", body = TranscriptionText),
        (status = 400, description = "Missing file, bad audio, unknown model, unsupported response_format, or missing sample_rate", body = OpenAiError),
        (status = 415, description = "Unsupported audio content type", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Transcription failed", body = OpenAiError)
    )
)]
async fn post_audio_transcriptions(
    State(config): State<AppConfig>,
    Query(query): Query<AudioSampleRateQuery>,
    multipart: Multipart,
) -> Response {
    let upload = match parse_transcription_upload(multipart, query.sample_rate).await {
        Ok(upload) => upload,
        Err(response) => return *response,
    };
    let Some((content_type, bytes)) = upload.file else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "file is required: attach the audio payload as a `file` form field".to_owned(),
            None,
        );
    };
    match upload.response_format.as_deref() {
        None | Some("json" | "verbose_json" | "srt" | "vtt") => {}
        Some(other) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "unsupported response_format {other:?}: only \"json\", \"verbose_json\", \"srt\", \"vtt\""
                ),
                None,
            );
        }
    }
    let catalog = config.catalog_snapshot();
    if let Err(message) = validate_model(
        upload.model.as_deref(),
        config.stt_model.as_deref(),
        &catalog,
    ) {
        return openai_error_response(StatusCode::BAD_REQUEST, message, None);
    }
    let (sample_rate, pcm) = match decode_upload_audio(&content_type, &bytes, upload.sample_rate) {
        Ok(pair) => pair,
        Err((code, message)) => return openai_error_response(code, message, None),
    };
    let worker_req = WorkerTranscribeRequest {
        model: upload.model.unwrap_or_default(),
        language: upload.language,
        translate: false,
    };
    // NEW-3: registry-visible while the request lives (one blocking FFI
    // call — the cancel token is marked but never checked).
    let job = crate::jobs_registry::global().register(
        crate::jobs_registry::JobKind::Transcribe,
        (!worker_req.model.is_empty()).then_some(worker_req.model.clone()),
        false,
    );
    let out = match run_transcription(config.stt.clone(), worker_req, sample_rate, pcm).await {
        Ok(out) => out,
        Err((StatusCode::TOO_MANY_REQUESTS, message)) => {
            return with_job_id(
                openai_error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    message,
                    Some("gpu_slot_busy"),
                ),
                job.id(),
            );
        }
        Err((code, message)) => return openai_error_response(code, message, None),
    };
    let response = match upload.response_format.as_deref() {
        Some("verbose_json") => Json(TranscriptionVerbose {
            text: out.text,
            segments: out
                .segments
                .into_iter()
                .map(|segment| TranscriptSegment {
                    start_ms: segment.start_ms,
                    end_ms: segment.end_ms,
                    text: segment.text,
                })
                .collect(),
        })
        .into_response(),
        Some("srt") => (
            [(header::CONTENT_TYPE, "application/x-subrip")],
            render_srt(&out.segments),
        )
            .into_response(),
        Some("vtt") => (
            [(header::CONTENT_TYPE, "text/vtt; charset=utf-8")],
            render_vtt(&out.segments),
        )
            .into_response(),
        _ => Json(TranscriptionText { text: out.text }).into_response(),
    };
    with_job_id(response, job.id())
}

/// SRT timestamp: `HH:MM:SS,mmm` (comma decimal separator, per the
/// `SubRip` spec).
fn srt_timestamp(ms: u32) -> String {
    subtitle_timestamp(ms, ',')
}

/// `WebVTT` timestamp: `HH:MM:SS.mmm`.
fn vtt_timestamp(ms: u32) -> String {
    subtitle_timestamp(ms, '.')
}

/// `HH:MM:SS{sep}mmm` from milliseconds. Hours grow past two digits
/// unclamped (both formats allow it; whisper segments never get there).
fn subtitle_timestamp(ms: u32, sep: char) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}{sep}{millis:03}")
}

/// Render worker segments as `SubRip` (`response_format = "srt"`):
/// `index\nstart --> end\ntext\n\n` per cue, 1-based. No segments = an
/// empty body (a no-speech input yields an empty subtitle track).
fn render_srt(segments: &[WorkerTranscriptSegment]) -> String {
    let mut out = String::new();
    for (index, segment) in segments.iter().enumerate() {
        out.push_str(&(index + 1).to_string());
        out.push('\n');
        out.push_str(&srt_timestamp(segment.start_ms));
        out.push_str(" --> ");
        out.push_str(&srt_timestamp(segment.end_ms));
        out.push('\n');
        out.push_str(&segment.text);
        out.push_str("\n\n");
    }
    out
}

/// Render worker segments as `WebVTT` (`response_format = "vtt"`): the
/// `WEBVTT` header, then the same cue shape with dot-decimal timestamps.
fn render_vtt(segments: &[WorkerTranscriptSegment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for (index, segment) in segments.iter().enumerate() {
        out.push_str(&(index + 1).to_string());
        out.push('\n');
        out.push_str(&vtt_timestamp(segment.start_ms));
        out.push_str(" --> ");
        out.push_str(&vtt_timestamp(segment.end_ms));
        out.push('\n');
        out.push_str(&segment.text);
        out.push_str("\n\n");
    }
    out
}

/// `POST /v1/audio/translations` (REQ-9 sibling; `ReviewedAudio` P1):
/// `OpenAI` translation shape — the multipart `file` is decoded exactly
/// like `/v1/audio/transcriptions` and transcribed with whisper's
/// **translate** task (source language auto-detected, output always
/// English). `model` optional (server default STT); `response_format`
/// `json` (default) / `verbose_json` / `srt` / `vtt`. No `language` field:
/// the endpoint contract fixes the OUTPUT language, and whisper's translate
/// task takes none. Errors are `OpenAI` envelopes, same as transcriptions.
#[utoipa::path(
    post,
    path = "/v1/audio/translations",
    request_body(content = String, description = "multipart/form-data: file, model?, response_format? (json|verbose_json|srt|vtt), sample_rate?"),
    responses(
        (status = 200, description = "English translation (json / verbose_json / srt / vtt)", body = TranscriptionText),
        (status = 400, description = "Missing file, bad audio, unknown model, unsupported response_format, unexpected language field, or missing sample_rate", body = OpenAiError),
        (status = 415, description = "Unsupported audio content type", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Translation failed", body = OpenAiError)
    )
)]
async fn post_audio_translations(
    State(config): State<AppConfig>,
    Query(query): Query<AudioSampleRateQuery>,
    multipart: Multipart,
) -> Response {
    let upload = match parse_transcription_upload(multipart, query.sample_rate).await {
        Ok(upload) => upload,
        Err(response) => return *response,
    };
    let Some((content_type, bytes)) = upload.file else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "file is required: attach the audio payload as a `file` form field".to_owned(),
            None,
        );
    };
    // The shared parser accepts `language` (the transcriptions field); on
    // the translations contract it is a client error, never silently
    // ignored — the output language is fixed to English.
    if upload.language.is_some() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "language is not a /v1/audio/translations field: the output is always English"
                .to_owned(),
            None,
        );
    }
    match upload.response_format.as_deref() {
        None | Some("json" | "verbose_json" | "srt" | "vtt") => {}
        Some(other) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "unsupported response_format {other:?}: only \"json\", \"verbose_json\", \"srt\", \"vtt\""
                ),
                None,
            );
        }
    }
    let catalog = config.catalog_snapshot();
    if let Err(message) = validate_model(
        upload.model.as_deref(),
        config.stt_model.as_deref(),
        &catalog,
    ) {
        return openai_error_response(StatusCode::BAD_REQUEST, message, None);
    }
    let (sample_rate, pcm) = match decode_upload_audio(&content_type, &bytes, upload.sample_rate) {
        Ok(pair) => pair,
        Err((code, message)) => return openai_error_response(code, message, None),
    };
    let worker_req = WorkerTranscribeRequest {
        model: upload.model.unwrap_or_default(),
        language: None,
        translate: true,
    };
    // NEW-3: registry-visible while the request lives (one blocking FFI
    // call — the cancel token is marked but never checked).
    let job = crate::jobs_registry::global().register(
        crate::jobs_registry::JobKind::Translations,
        (!worker_req.model.is_empty()).then_some(worker_req.model.clone()),
        false,
    );
    let out = match run_transcription(config.stt.clone(), worker_req, sample_rate, pcm).await {
        Ok(out) => out,
        Err((StatusCode::TOO_MANY_REQUESTS, message)) => {
            return with_job_id(
                openai_error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    message,
                    Some("gpu_slot_busy"),
                ),
                job.id(),
            );
        }
        Err((code, message)) => return openai_error_response(code, message, None),
    };
    let response = match upload.response_format.as_deref() {
        Some("verbose_json") => Json(TranscriptionVerbose {
            text: out.text,
            segments: out
                .segments
                .into_iter()
                .map(|segment| TranscriptSegment {
                    start_ms: segment.start_ms,
                    end_ms: segment.end_ms,
                    text: segment.text,
                })
                .collect(),
        })
        .into_response(),
        Some("srt") => (
            [(header::CONTENT_TYPE, "application/x-subrip")],
            render_srt(&out.segments),
        )
            .into_response(),
        Some("vtt") => (
            [(header::CONTENT_TYPE, "text/vtt; charset=utf-8")],
            render_vtt(&out.segments),
        )
            .into_response(),
        _ => Json(TranscriptionText { text: out.text }).into_response(),
    };
    with_job_id(response, job.id())
}

/// `GET /v1/jobs` (NEW-3): the in-flight registry — admitted GPU requests
/// (running, or waiting in the admission queue), oldest first. Entries
/// appear when a route admits a request and vanish when it ends; this is a
/// liveness view, not history (see `jobs_registry`).
#[utoipa::path(
    get,
    path = "/v1/jobs",
    responses(
        (status = 200, description = "In-flight jobs", body = JobList),
    )
)]
async fn get_jobs() -> Json<JobList> {
    Json(JobList {
        object: "list".to_owned(),
        data: crate::jobs_registry::global().list(),
    })
}

/// Shared cancel machinery for `POST /v1/jobs/{id}/cancel` and
/// `DELETE /v1/jobs/{id}` (same semantics, two spellings — the REST-shaped
/// verb and the explicit action): mark the job's cooperative-stop token
/// and answer honestly. 200 `{cancelled: bool}` — `true` for EVERY found
/// id, because the token was marked; whether the work actually stops early
/// is the separate `interruptible` contract reported by `GET /v1/jobs`
/// and the job lifecycle events (image generations check between images
/// AND mid-sampling via `sd_cancel_generation`; streaming chat checks at
/// the next token boundary; a single blocking FFI call runs to completion).
/// 404: unknown id (never registered, or the request already ended — the
/// registry holds live work only).
fn cancel_job_response(id: &str) -> Response {
    match crate::jobs_registry::global().cancel(id) {
        Some(outcome) => Json(JobCancelResult {
            cancelled: outcome.cancelled,
        })
        .into_response(),
        None => (StatusCode::NOT_FOUND, format!("unknown job id {id:?}")).into_response(),
    }
}

/// `POST /v1/jobs/{id}/cancel` (NEW-3): mark the job's cooperative-stop
/// token. See [`cancel_job_response`] for the honest response contract.
#[utoipa::path(
    post,
    path = "/v1/jobs/{id}/cancel",
    responses(
        (status = 200, description = "Token marked", body = JobCancelResult),
        (status = 404, description = "Unknown job id"),
    )
)]
async fn post_job_cancel(AxumPath(id): AxumPath<String>) -> Response {
    cancel_job_response(&id)
}

/// `DELETE /v1/jobs/{id}`: the REST-shaped spelling of
/// `POST /v1/jobs/{id}/cancel` — deleting an in-flight job means requesting
/// its cooperative cancel, nothing more. Same token machinery, same honest
/// 200 `{cancelled}` / 404 contract (see [`cancel_job_response`]); the
/// DELETE does not wait for the job to end and never removes finished-work
/// records (job bundles under `<data_dir>/jobs/` are a separate concern).
#[utoipa::path(
    delete,
    path = "/v1/jobs/{id}",
    responses(
        (status = 200, description = "Cancel requested (same contract as POST /v1/jobs/{id}/cancel)", body = JobCancelResult),
        (status = 404, description = "Unknown job id"),
    )
)]
async fn delete_job(AxumPath(id): AxumPath<String>) -> Response {
    cancel_job_response(&id)
}

/// `GET /v1/events`: the registry-wide job lifecycle feed as a
/// Server-Sent-Events stream (observability only — it registers no job and
/// touches no GPU slot). Every in-flight GPU job of every kind emits:
///
/// - `event: job_registered` — the job entered the registry (route
///   admitted the request; it may still be waiting in the admission
///   queue);
/// - `event: job_cancelled` — a cancel was REQUESTED for the job (POST
///   `.../cancel` or DELETE `/v1/jobs/{id}`); not a completion event;
/// - `event: job_finished` — the job left the registry (its request ended,
///   any outcome).
///
/// Each frame's JSON `data` mirrors the SSE `event` line in its `event`
/// field (`JobEvent`): `id`, `kind`, `model` (when known),
/// `started_at_ms`, `interruptible`, and `cancelled` (= a cancel had been
/// requested at that point; on `job_finished` it is `true` for a
/// token-marked job, which for a non-interruptible job still ran to
/// completion). Frames are transition-only: a subscriber that falls more
/// than 256 events behind gets an `events_dropped` frame naming the skipped
/// count (honest gap marker, never silent skipping), and (re)connecting
/// replays nothing — `GET /v1/jobs` is the current-state snapshot.
#[utoipa::path(
    get,
    path = "/v1/events",
    responses(
        (status = 200, description = "SSE stream of job lifecycle events (job_registered / job_cancelled / job_finished; catalogue above)", content_type = "text/event-stream"),
    )
)]
async fn get_events() -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let receiver = crate::jobs_registry::global().subscribe();
    let stream = BroadcastStream::new(receiver).filter_map(|item| match item {
        Ok(event) => Some(Ok::<_, Infallible>(job_event_frame(&event))),
        // This consumer fell behind: surface the gap instead of quietly
        // skipping the missed events. (A closed channel has no error
        // variant here — the stream simply ends, and the registry's sender
        // never closes anyway.)
        Err(BroadcastStreamRecvError::Lagged(skipped)) => {
            let frame = Event::default().event("events_dropped").data(format!(
                "{{\"event\":\"events_dropped\",\"skipped\":{skipped}}}"
            ));
            Some(Ok(frame))
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// One SSE frame for a registry lifecycle event: the `event:` line mirrors
/// the JSON `event` field, so JSON-only consumers can ignore the envelope.
fn job_event_frame(event: &JobEvent) -> Event {
    Event::default()
        .event(event.event.label())
        .data(serde_json::to_string(&event).unwrap_or_default())
}

/// Query carried over from the native route: `audio/pcm-s16le` uploads
/// without a `sample_rate` form field.
#[derive(Debug, Deserialize)]
struct AudioSampleRateQuery {
    /// Input sample rate in Hz (pcm-s16le uploads only).
    sample_rate: Option<u32>,
}

/// Map a Kokoro voice-id prefix to `(language, gender)` metadata.
/// Kokoro ids are `<lang><gender>_<name>`: `a`/`b` US/GB English, `e` es,
/// `f` fr, `h`/`z` zh, `i` it, `j` ja, `p` pt-BR; `f` female, `m` male.
fn kokoro_voice_meta(id: &str) -> (&'static str, &'static str) {
    let mut chars = id.chars();
    let lang = chars.next();
    let gender = chars.next();
    let language = match lang {
        Some('a') => "en-US",
        Some('b') => "en-GB",
        Some('e') => "es-ES",
        Some('f') => "fr-FR",
        Some('h' | 'z') => "zh-CN",
        Some('i') => "it-IT",
        Some('j') => "ja-JP",
        Some('p') => "pt-BR",
        _ => "unknown",
    };
    let gender = match gender {
        Some('f') => "female",
        Some('m') => "male",
        _ => "unknown",
    };
    (language, gender)
}

/// `GET /v1/audio/voices`: the supported TTS voice set as an OpenAI-style
/// list (`{object: "list", data: [{object: "voice", …}]}`). The Kokoro
/// backend cannot enumerate voices at runtime, so this serves the
/// statically documented `KOKORO_VOICES` set — honest, no model loaded,
/// no state. `data[].id` values are exactly what `/v1/audio/speech`
/// accepts as `voice`.
#[utoipa::path(
    get,
    path = "/v1/audio/voices",
    responses((status = 200, description = "Supported TTS voices", body = VoiceList))
)]
async fn get_audio_voices() -> Response {
    let data = honkbox_audio_worker::KOKORO_VOICES
        .iter()
        .map(|id| {
            let (language, gender) = kokoro_voice_meta(id);
            Voice {
                id: (*id).to_owned(),
                object: "voice".to_owned(),
                language: language.to_owned(),
                gender: gender.to_owned(),
            }
        })
        .collect();
    Json(VoiceList {
        object: "list".to_owned(),
        data,
    })
    .into_response()
}

/// The image-ops batch cap: inputs (and thus outputs) per request.
const IMAGE_OP_MAX_INPUTS: usize = 8;
/// `resize` edge cap (px). With the registry pixel budget
/// ([`crate::formats::MAX_DECODED_PIXELS`]) this bounds both allocation
/// and encode time on the blocking pool.
const IMAGE_OP_MAX_EDGE: u32 = 8192;
/// Decoder-level strict caps (`image::Limits`): no decoded edge above this,
/// and no more than this much decoder allocation at once — a small PNG can
/// decode to gigabytes without these. Both caps come from the format
/// registry (the same budget table the pixel check below enforces).
fn image_decode_limits() -> image::Limits {
    // `Limits` is `#[non_exhaustive]`: start from `default()` (512 MiB
    // allocation allowance) and tighten the strict caps.
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(crate::formats::MAX_DECODED_EDGE);
    limits.max_image_height = Some(crate::formats::MAX_DECODED_EDGE);
    // RGBA of the pixel cap is ~67 MB; intermediates stay well under
    // this allowance.
    limits.max_alloc = Some(crate::formats::MAX_DECODED_ALLOC_BYTES);
    limits
}
/// Fallback ceiling for the SVG source size, used only when the sniffed
/// registry row carries no byte budget (named, never `u64::MAX`).
const SVG_FALLBACK_MAX_SOURCE_BYTES: u64 = 16 * 1024 * 1024;
/// Fallback ceiling for the SVG element count, used only when the sniffed
/// registry row carries no element budget.
const SVG_FALLBACK_MAX_ELEMENTS: u64 = 100_000;
/// Fallback ceiling for the ORA archive entry count, used only when the
/// sniffed registry row carries no entry budget.
const ORA_FALLBACK_MAX_ENTRIES: u64 = 4096;
/// Fallback ceiling for ONE ORA archive entry, used only when the sniffed
/// registry row carries no per-entry budget.
const ORA_FALLBACK_MAX_ENTRY_BYTES: u64 = 64 * 1024 * 1024;
/// Fallback ceiling for the ORA archive's total uncompressed size, used
/// only when the sniffed registry row carries no size budget.
const ORA_FALLBACK_MAX_TOTAL_BYTES: u64 = 512 * 1024 * 1024;
/// Fallback pixel ceiling for a rasterized vector or a flattened layered
/// document, used only when the sniffed registry row carries no pixel
/// budget.
const RASTER_FALLBACK_MAX_PIXELS: u64 = crate::formats::MAX_DECODED_PIXELS;

/// The SVG read limits, built from the sniffed registry row: the registry
/// stays the only source of truth for limits, and the fallbacks apply only
/// to a row that carries no budget for that field.
fn svg_read_limits(row: &crate::formats::FormatInfo) -> crate::svg::SvgLimits {
    crate::svg::SvgLimits {
        max_source_bytes: row
            .budget
            .uncompressed_bytes
            .unwrap_or(SVG_FALLBACK_MAX_SOURCE_BYTES),
        max_elements: row.budget.elements.unwrap_or(SVG_FALLBACK_MAX_ELEMENTS),
        max_pixels: row
            .budget
            .decoded_pixels
            .unwrap_or(RASTER_FALLBACK_MAX_PIXELS),
    }
}

/// The ORA read limits, built from the sniffed registry row. The
/// registry's container-entry ceiling is also the layer ceiling: a stack's
/// layers are archive entries.
fn ora_read_limits(row: &crate::formats::FormatInfo) -> crate::ora::OraLimits {
    crate::ora::OraLimits {
        max_entries: row.budget.entries.unwrap_or(ORA_FALLBACK_MAX_ENTRIES),
        max_entry_bytes: row
            .budget
            .entry_bytes
            .unwrap_or(ORA_FALLBACK_MAX_ENTRY_BYTES),
        max_total_bytes: row
            .budget
            .uncompressed_bytes
            .unwrap_or(ORA_FALLBACK_MAX_TOTAL_BYTES),
        max_layers: row.budget.entries.unwrap_or(ORA_FALLBACK_MAX_ENTRIES),
        max_pixels: row
            .budget
            .decoded_pixels
            .unwrap_or(RASTER_FALLBACK_MAX_PIXELS),
    }
}
/// `request_id` / `client` string cap, characters.
const IMAGE_OP_ID_MAX_CHARS: usize = 128;
/// Idempotency memory bounds (TASK-image-ops-command-dto): at most this
/// many `request_id`s remembered and this many retained response bytes —
/// oldest-first eviction, so retries replay exactly but memory stays
/// capped. Failures are never remembered (a retry re-executes).
const IMAGE_OP_DEDUP_MAX_ENTRIES: usize = 128;
const IMAGE_OP_DEDUP_MAX_BYTES: usize = 64 * 1024 * 1024;

/// A settled image-op attempt, replayable byte-identical: the completed
/// response, or the exact error the first attempt answered (status +
/// message + named code). Errors carry the registry job id the execute
/// path registered, so the replay still stamps `X-Job-Id`.
#[derive(Clone)]
struct ImageOpSettled {
    response: Result<ImageOpResponse, (StatusCode, String, Option<String>)>,
    job_id: Option<String>,
}

/// One remembered image-op submission: the canonical command hash it
/// answered and the replayable outcome. Same-hash losers of a same-key
/// race await the cell (single execution, identical replay); a different
/// hash under the same `request_id` is a conflict, never silent execution.
struct ImageOpDedupEntry {
    command_sha256: [u8; 32],
    /// Approximate retained size of the settled response (set only for
    /// successes; failures are forgotten and their size never counts).
    retained_bytes: std::sync::atomic::AtomicUsize,
    /// The single-flight cell: concurrent same-key retries all await the
    /// one execution (tokio `get_or_init` guarantees exactly one closure
    /// runs even under concurrency).
    outcome: OnceCell<ImageOpSettled>,
}

/// The image-op idempotency ledger: `request_id` → entry, bounded on
/// entries AND retained bytes (oldest-first eviction).
#[derive(Default)]
struct ImageOpDedup {
    map: HashMap<String, Arc<ImageOpDedupEntry>>,
    /// Insertion order for oldest-first eviction (stale ids tolerated).
    order: VecDeque<String>,
}

/// Outcome of [`ImageOpDedup::reserve`].
enum ImageOpReservation {
    /// Fresh key: this caller participates in the single-flight cell.
    Execute(Arc<ImageOpDedupEntry>),
    /// Same key + same canonical command: replay the first outcome.
    Replay(Arc<ImageOpDedupEntry>),
    /// Same key with a DIFFERENT command: a named 400, never execution.
    Conflict,
}

impl ImageOpDedup {
    /// Reserve `request_id` for `command_sha256`.
    fn reserve(&mut self, request_id: &str, command_sha256: [u8; 32]) -> ImageOpReservation {
        if let Some(entry) = self.map.get(request_id) {
            return if entry.command_sha256 == command_sha256 {
                ImageOpReservation::Replay(Arc::clone(entry))
            } else {
                ImageOpReservation::Conflict
            };
        }
        let entry = Arc::new(ImageOpDedupEntry {
            command_sha256,
            retained_bytes: const { std::sync::atomic::AtomicUsize::new(0) },
            outcome: OnceCell::new(),
        });
        self.map.insert(request_id.to_owned(), Arc::clone(&entry));
        self.order.push_back(request_id.to_owned());
        self.trim();
        ImageOpReservation::Execute(entry)
    }

    /// Record the retained size of a settled SUCCESS (bounded cache).
    fn remember(&mut self, request_id: &str, retained_bytes: usize) {
        if let Some(entry) = self.map.get(request_id) {
            entry
                .retained_bytes
                .store(retained_bytes, std::sync::atomic::Ordering::Relaxed);
        }
        self.trim();
    }

    /// Forget an entry (a FAILED attempt): a genuine retry re-executes.
    fn forget(&mut self, request_id: &str) {
        self.map.remove(request_id);
    }

    /// Evict oldest-first while over either budget.
    fn trim(&mut self) {
        while self.map.len() > IMAGE_OP_DEDUP_MAX_ENTRIES
            || self.retained_bytes() > IMAGE_OP_DEDUP_MAX_BYTES
        {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.map.remove(&oldest);
        }
    }

    /// Sum of retained response bytes.
    fn retained_bytes(&self) -> usize {
        self.map
            .values()
            .map(|entry| {
                entry
                    .retained_bytes
                    .load(std::sync::atomic::Ordering::Relaxed)
            })
            .sum()
    }
}

/// The image-op idempotency ledger (process-global, like the job registry).
static IMAGE_OP_DEDUP: LazyLock<Mutex<ImageOpDedup>> =
    LazyLock::new(|| Mutex::new(ImageOpDedup::default()));

/// The op wire name for sidecars/labels (the tag of the command).
fn op_label(command: &ImageOpCommand) -> &'static str {
    match command {
        ImageOpCommand::Resize(_) => "resize",
        ImageOpCommand::Crop(_) => "crop",
        ImageOpCommand::Rotate(_) => "rotate",
        ImageOpCommand::Grayscale => "grayscale",
        ImageOpCommand::Format => "format",
    }
}

/// Lowercase-hex SHA-256 of `bytes` (sidecar + response digests).
fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Validate the bounded image-op params (reject-not-clamp on every number):
/// resize edges in `[1, 8192]` with the pixel cap, crop rects at least 1px
/// (the image-bounds check happens per-input against the real pixels), and
/// rotation restricted to the lossless quarter turns. `Err((message, code))`
/// is a named 400.
fn validate_image_op_params(command: &ImageOpCommand) -> Result<(), (String, &'static str)> {
    match command {
        ImageOpCommand::Resize(resize) => {
            if resize.width == 0
                || resize.width > IMAGE_OP_MAX_EDGE
                || resize.height == 0
                || resize.height > IMAGE_OP_MAX_EDGE
            {
                return Err((
                    format!(
                        "resize width/height must be in [1, {IMAGE_OP_MAX_EDGE}] pixels; got {}x{} — rejected, not clamped",
                        resize.width, resize.height
                    ),
                    "invalid_image_op_params",
                ));
            }
            if u64::from(resize.width) * u64::from(resize.height)
                > crate::formats::MAX_DECODED_PIXELS
            {
                return Err((
                    format!(
                        "resize target {}x{} exceeds the {}-pixel cap — rejected, not clamped",
                        resize.width,
                        resize.height,
                        crate::formats::MAX_DECODED_PIXELS
                    ),
                    "invalid_image_op_params",
                ));
            }
        }
        ImageOpCommand::Crop(crop) => {
            if crop.width == 0 || crop.height == 0 {
                return Err((
                    "crop width/height must be at least 1 pixel — rejected, not clamped".to_owned(),
                    "invalid_image_op_params",
                ));
            }
        }
        ImageOpCommand::Rotate(rotate) => {
            if !matches!(rotate.degrees, 90 | 180 | 270) {
                return Err((
                    format!(
                        "rotate degrees must be one of 90, 180, 270; got {} — rejected, not approximated",
                        rotate.degrees
                    ),
                    "invalid_image_op_params",
                ));
            }
        }
        ImageOpCommand::Grayscale | ImageOpCommand::Format => {}
    }
    Ok(())
}

/// Decode a `data:image/…;base64,…` URI. The declared MIME must be one of
/// the registry's accepted input containers (the caller additionally sniffs
/// the real bytes, so declared/actual mismatches are caught there); the
/// declared row is returned alongside the bytes for that comparison.
fn decode_data_uri(
    uri: &str,
) -> Result<(&'static crate::formats::FormatInfo, Vec<u8>), ImageOpWorkError> {
    let bad = |message: String| ImageOpWorkError::Bad {
        message,
        code: "invalid_image_op_inputs",
    };
    let trimmed = uri.trim();
    let Some(rest) = trimmed.strip_prefix("data:") else {
        return Err(bad(
            "input data URI must start with `data:image/…;base64,`".to_owned()
        ));
    };
    let Some((header, payload)) = rest.split_once(',') else {
        return Err(bad(
            "input data URI is missing the `,<base64 payload>` part".to_owned(),
        ));
    };
    let Some(mime) = header.strip_suffix(";base64") else {
        return Err(bad("input data URI must declare `;base64`".to_owned()));
    };
    let Some(declared) = crate::formats::image_ops_by_mime(mime) else {
        return Err(bad(format!(
            "unsupported input container {mime:?}: accepted data URIs are {}",
            crate::formats::image_ops_inputs()
                .map(|row| row.mime)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        // Tolerate padding-stripped payloads (some encoders trim `=`).
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(payload.trim()))
        .map_err(|e| bad(format!("input data URI is not valid base64: {e}")))?;
    Ok((declared, bytes))
}

/// Resolve an artifact input against the bundle store:
/// `<data_dir>/jobs/{bundle}/op_XX.{ext}` (ops bundles) or
/// `final_XX.{ext}` (generation bundles). The bundle id must parse as a
/// ULID, so no client-supplied bytes ever reach the filesystem join.
fn read_artifact(data_dir: &Path, artifact: &ImageOpArtifact) -> Result<Vec<u8>, ImageOpWorkError> {
    let bad = |message: String| ImageOpWorkError::Bad {
        message,
        code: "invalid_image_op_inputs",
    };
    let id = ulid::Ulid::from_string(artifact.bundle.trim()).map_err(|_| {
        bad(format!(
            "artifact bundle {:?} is not a bundle id",
            artifact.bundle
        ))
    })?;
    if artifact.index > 99 {
        return Err(bad(format!(
            "artifact index {} is out of range (0-99)",
            artifact.index
        )));
    }
    let dir = data_dir.join("jobs").join(id.to_string());
    // Candidate artifact names derive from the registry's accepted input
    // extensions (the same rows the magic sniff admits below).
    let names: Vec<String> = crate::formats::image_ops_extensions()
        .flat_map(|ext| {
            [
                format!("op_{:02}.{ext}", artifact.index),
                format!("final_{:02}.{ext}", artifact.index),
            ]
        })
        .collect();
    let entries = std::fs::read_dir(&dir).map_err(|_| {
        bad(format!(
            "artifact bundle {:?} not found (expired by GC, or never written by this server)",
            artifact.bundle
        ))
    })?;
    let mut found: Vec<std::ffi::OsString> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if names.iter().any(|wanted| *wanted == name_str) {
            found.push(name);
        }
    }
    found.sort();
    match found.first() {
        Some(name) => std::fs::read(dir.join(name)).map_err(|e| {
            ImageOpWorkError::Failed(format!(
                "cannot read artifact {:?}: {e}",
                name.to_string_lossy()
            ))
        }),
        None => Err(bad(format!(
            "artifact {} of bundle {:?} not found",
            artifact.index, artifact.bundle
        ))),
    }
}

/// One processed unit: the encoded output bytes and their digest.
struct ImageOpUnitOutput {
    bytes: Vec<u8>,
    sha256: String,
}

/// A failed image-op batch. The whole batch fails fast on the first bad
/// input — partial results would mislead an editor about what landed.
enum ImageOpWorkError {
    /// Cancel marked between units (`completed` units are done).
    Cancelled { completed: usize },
    /// Named 400-class rejection.
    Bad { message: String, code: &'static str },
    /// 500-class failure (decode/encode/IO).
    Failed(String),
}

/// Apply the op to one decoded image. Deterministic by construction: the
/// single fixed resample is Lanczos3, crops are pixel-exact, rotations are
/// the lossless quarter turns, grayscale uses the `image` crate's Luma
/// conversion, and `format` passes pixels through untouched.
fn apply_image_op(
    command: &ImageOpCommand,
    decoded: image::DynamicImage,
) -> Result<image::DynamicImage, ImageOpWorkError> {
    Ok(match command {
        ImageOpCommand::Resize(resize) => decoded.resize_exact(
            resize.width,
            resize.height,
            image::imageops::FilterType::Lanczos3,
        ),
        ImageOpCommand::Crop(crop) => {
            // Bounds checked against the REAL pixels here: the DTO-level
            // validation knows the rect shape, not the image.
            if crop.x.saturating_add(crop.width) > decoded.width()
                || crop.y.saturating_add(crop.height) > decoded.height()
            {
                return Err(ImageOpWorkError::Bad {
                    message: format!(
                        "crop rect {}+{}x{}+{} exceeds the {}x{} input — rejected, not clamped",
                        crop.x,
                        crop.width,
                        crop.y,
                        crop.height,
                        decoded.width(),
                        decoded.height()
                    ),
                    code: "invalid_crop_rect",
                });
            }
            decoded.crop_imm(crop.x, crop.y, crop.width, crop.height)
        }
        ImageOpCommand::Rotate(rotate) => match rotate.degrees {
            90 => decoded.rotate90(),
            180 => decoded.rotate180(),
            270 => decoded.rotate270(),
            // DTO-level validation rejects every other value first.
            _ => decoded,
        },
        ImageOpCommand::Grayscale => decoded.grayscale(),
        ImageOpCommand::Format => decoded,
    })
}

/// Admit one image-op input: the registry identifies the container by
/// magic for BOTH input forms — the data URI's declared MIME was resolved
/// at parse and must agree with the bytes (a disagreement is the named
/// `format_mismatch`), artifacts declare nothing, and bytes matching a
/// non-accepted row (e.g. a video-frame or 3D format) name that row in the
/// rejection. Returns the admitted row for its decode budget.
fn admit_op_input(
    raw: &[u8],
    declared: Option<&'static crate::formats::FormatInfo>,
) -> Result<&'static crate::formats::FormatInfo, ImageOpWorkError> {
    let accepted = crate::formats::image_ops_inputs()
        .map(|row| row.name)
        .collect::<Vec<_>>()
        .join(", ");
    let Some(row) = crate::formats::sniff(raw) else {
        return Err(ImageOpWorkError::Bad {
            message: format!("input is not a recognizable image (accepted containers: {accepted})"),
            code: "invalid_image_op_inputs",
        });
    };
    if let Some(expected) = declared
        && expected.name != row.name
    {
        return Err(ImageOpWorkError::Bad {
            message: format!(
                "input declares {} but the bytes are {} (magic sniff)",
                expected.mime, row.name
            ),
            code: "format_mismatch",
        });
    }
    if !row.image_ops_input {
        return Err(ImageOpWorkError::Bad {
            message: format!(
                "unsupported input container {} ({}): accepted containers are {accepted}",
                row.name,
                row.kind.label()
            ),
            code: "invalid_image_op_inputs",
        });
    }
    Ok(row)
}

/// Run the op units (blocking pool): per input — cancel check, decode,
/// apply, encode. Fails fast on the first bad input. The inputs are
/// resolved and safety-checked BEFORE the job is admitted.
/// One admitted image-op input: the resolved bytes, the registry row that
/// identified them, and the bookkeeping the sidecar records.
struct AdmittedOpInput {
    /// Where the input came from (`data_uri` / `artifact:<bundle>/<index>`).
    source: String,
    /// The raw container bytes.
    raw: Vec<u8>,
    /// The registry row that identified the container; its budget drives
    /// the read limits.
    row: &'static crate::formats::FormatInfo,
    /// The SHA-256 of the raw bytes (the sidecar input record).
    sha256: String,
}

/// Resolve and safety-check every image-op input BEFORE a job is admitted.
///
/// Resolution (data-URI decode / artifact read) and the registry sniff
/// happen here, and so does the format-specific safety check with the
/// sniffed row's budgets: a hostile vector or layered document is a named
/// 400 and NO job is registered at all.
fn admit_op_inputs(
    inputs: &[ImageOpInput],
    data_dir: Option<&Path>,
) -> Result<Vec<AdmittedOpInput>, ImageOpWorkError> {
    let mut admitted = Vec::with_capacity(inputs.len());
    for input in inputs {
        let (source, declared, raw) = match input {
            ImageOpInput::DataUri(uri) => {
                let (declared, bytes) = decode_data_uri(uri)?;
                (String::from("data_uri"), Some(declared), bytes)
            }
            ImageOpInput::Artifact(artifact) => {
                let Some(dir) = data_dir else {
                    return Err(ImageOpWorkError::Bad {
                        message: "artifact inputs need a configured data dir (this server keeps no job bundles)"
                            .to_owned(),
                        code: "invalid_image_op_inputs",
                    });
                };
                (
                    format!("artifact:{}/{}", artifact.bundle, artifact.index),
                    None,
                    read_artifact(dir, artifact)?,
                )
            }
        };
        // Container authority is the registry magic (see `admit_op_input`):
        // the sniffed format wins, declarations must agree with it.
        let row = admit_op_input(&raw, declared)?;
        let sha256 = sha256_hex(&raw);
        check_admitted_container(row, &raw)?;
        admitted.push(AdmittedOpInput {
            source,
            raw,
            row,
            sha256,
        });
    }
    Ok(admitted)
}

/// The format-specific safety check of an admitted input: the SVG
/// validator and the ORA reader run with the sniffed row's budgets, so a
/// hostile container is rejected before any job is admitted.
fn check_admitted_container(
    row: &crate::formats::FormatInfo,
    raw: &[u8],
) -> Result<(), ImageOpWorkError> {
    let bad = |message: String| ImageOpWorkError::Bad {
        message,
        code: "invalid_image_op_inputs",
    };
    match row.kind {
        crate::formats::FormatKind::Svg => crate::svg::validate_svg(raw, &svg_read_limits(row))
            .map(|_| ())
            .map_err(|e| bad(format!("cannot read svg container: {e}")))?,
        crate::formats::FormatKind::Ora => crate::ora::read_ora(raw, &ora_read_limits(row))
            .map(|_| ())
            .map_err(|e| bad(format!("cannot read ora container: {e}")))?,
        _ => {}
    }
    Ok(())
}

/// Decode one admitted input into pixels: the SVG rasterizer and the ORA
/// flattener for the vector and layered rows, the `image` crate reader for
/// the flat raster rows. The limits come from the sniffed registry row.
fn decode_admitted_input(
    admitted: &AdmittedOpInput,
) -> Result<image::DynamicImage, ImageOpWorkError> {
    let bad = |message: String| ImageOpWorkError::Bad {
        message,
        code: "invalid_image_op_inputs",
    };
    match admitted.row.kind {
        crate::formats::FormatKind::Svg => {
            crate::svg::rasterize_svg(&admitted.raw, &svg_read_limits(admitted.row))
                .map(crate::raster::RasterImage::into_dynamic_image)
                .map_err(|e| bad(format!("cannot rasterize svg: {e}")))
        }
        crate::formats::FormatKind::Ora => {
            crate::ora::flatten_ora(&admitted.raw, &ora_read_limits(admitted.row))
                .map(crate::raster::RasterImage::into_dynamic_image)
                .map_err(|e| bad(format!("cannot flatten ora: {e}")))
        }
        _ => {
            // Bounded decode: the reader-level strict caps stop
            // decompression-bomb allocations before they happen; the pixel
            // cap (the sniffed row's registry budget) is the honest reject
            // line for oversized-but-bounded inputs.
            let mut reader = image::ImageReader::new(std::io::Cursor::new(&admitted.raw));
            reader.limits(image_decode_limits());
            reader
                .with_guessed_format()
                .map_err(|e| bad(format!("cannot read image container: {e}")))?
                .decode()
                .map_err(|e| bad(format!("cannot decode image: {e}")))
        }
    }
}

fn run_image_op_units(
    command: &ImageOpCommand,
    inputs: &[AdmittedOpInput],
    format: honkbox_sd_core::OutputFormat,
    jpeg_quality: Option<u8>,
    token: &tokio_util::sync::CancellationToken,
) -> Result<(Vec<ImageOpUnitOutput>, Vec<crate::jobs::OpsInputRecord>), ImageOpWorkError> {
    let mut outputs = Vec::with_capacity(inputs.len());
    let mut records = Vec::with_capacity(inputs.len());
    for (i, admitted) in inputs.iter().enumerate() {
        // Cooperative cancellation (NEW-3 contract): honored BETWEEN units;
        // the unit in flight always completes.
        if token.is_cancelled() {
            return Err(ImageOpWorkError::Cancelled { completed: i });
        }
        let decoded = decode_admitted_input(admitted)?;
        let pixels = u64::from(decoded.width()) * u64::from(decoded.height());
        let pixel_cap = admitted
            .row
            .budget
            .decoded_pixels
            .unwrap_or(RASTER_FALLBACK_MAX_PIXELS);
        if pixels > pixel_cap {
            return Err(ImageOpWorkError::Bad {
                message: format!(
                    "input is {}x{} ({pixels} pixels); the cap is {pixel_cap} pixels — rejected, not downscaled",
                    decoded.width(),
                    decoded.height()
                ),
                code: "invalid_image_op_inputs",
            });
        }
        let worked = apply_image_op(command, decoded)?;
        let rgba = worked.to_rgba8();
        let generated = honkbox_sd_core::GeneratedImage {
            width: rgba.width(),
            height: rgba.height(),
            rgba: rgba.into_raw(),
        };
        let bytes = honkbox_sd_core::encode_image_with_quality(
            generated,
            format,
            jpeg_quality.unwrap_or(honkbox_sd_core::DEFAULT_JPEG_QUALITY),
        )
        .map_err(|e| ImageOpWorkError::Failed(format!("output encode failed: {e}")))?;
        records.push(crate::jobs::OpsInputRecord {
            source: admitted.source.clone(),
            sha256: admitted.sha256.clone(),
        });
        outputs.push(ImageOpUnitOutput {
            sha256: sha256_hex(&bytes),
            bytes,
        });
    }
    Ok((outputs, records))
}

/// The blocking-pool work result (units + sidecar input records).
type ImageOpWork = (Vec<ImageOpUnitOutput>, Vec<crate::jobs::OpsInputRecord>);

/// A pre-admission rejection: the named 400 with NO job (the job is
/// registered only after every input passed the safety checks).
fn settled_without_job(error: ImageOpWorkError) -> ImageOpSettled {
    match error {
        ImageOpWorkError::Bad { message, code } => ImageOpSettled {
            response: Err((StatusCode::BAD_REQUEST, message, Some(code.to_owned()))),
            job_id: None,
        },
        ImageOpWorkError::Failed(message) => ImageOpSettled {
            response: Err((StatusCode::INTERNAL_SERVER_ERROR, message, None)),
            job_id: None,
        },
        ImageOpWorkError::Cancelled { completed } => ImageOpSettled {
            response: Err((
                StatusCode::CONFLICT,
                format!("image-op cancelled before completion ({completed} input(s) processed)"),
                Some("job_cancelled".to_owned()),
            )),
            job_id: None,
        },
    }
}

/// A worker crash during pre-admission is an internal 500 (not a hostile
/// input). No job exists yet, but the replay still gets an id like every
/// other failure path; that id is deliberately NOT in the registry, so a
/// cancel against it is a 404.
fn image_op_pre_admission_join_failure(message: String) -> ImageOpSettled {
    ImageOpSettled {
        response: Err((StatusCode::INTERNAL_SERVER_ERROR, message, None)),
        job_id: Some(format!("job_{}", ulid::Ulid::new())),
    }
}

/// Map the blocking-pool work outcome onto units or a settleable failure:
/// cancel → 409 between units, bad input → named 400, work/join failure →
/// 500. Every failure stamps the job id so the replay keeps `X-Job-Id`
/// (boxed: `ImageOpSettled` carries the full response and is big).
fn settle_image_op_work(
    job_id: &str,
    total_inputs: usize,
    work: Result<Result<ImageOpWork, ImageOpWorkError>, tokio::task::JoinError>,
) -> Result<ImageOpWork, Box<ImageOpSettled>> {
    match work {
        Ok(Ok(pair)) => Ok(pair),
        Ok(Err(ImageOpWorkError::Cancelled { completed })) => Err(Box::new(ImageOpSettled {
            response: Err((
                StatusCode::CONFLICT,
                format!(
                    "job {job_id} cancelled before completion ({completed} of {total_inputs} input(s) processed)"
                ),
                Some("job_cancelled".to_owned()),
            )),
            job_id: Some(job_id.to_owned()),
        })),
        Ok(Err(ImageOpWorkError::Bad { message, code })) => Err(Box::new(ImageOpSettled {
            response: Err((StatusCode::BAD_REQUEST, message, Some(code.to_owned()))),
            job_id: Some(job_id.to_owned()),
        })),
        Ok(Err(ImageOpWorkError::Failed(message))) => Err(Box::new(ImageOpSettled {
            response: Err((StatusCode::INTERNAL_SERVER_ERROR, message, None)),
            job_id: Some(job_id.to_owned()),
        })),
        Err(join) => Err(Box::new(ImageOpSettled {
            response: Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("image-op worker failed: {join}"),
                None,
            )),
            job_id: Some(job_id.to_owned()),
        })),
    }
}

/// Execute one image-op command: register the job, acquire CPU admission,
/// run the units on the blocking pool (cancel checked between inputs),
/// write the bundle (best-effort, like generation), and build the response.
/// NEVER returns without a settleable outcome — every failure path returns
/// the exact error the caller must replay.
#[allow(clippy::too_many_lines)] // one linear execution: admit, register, run, settle
async fn execute_image_op(
    config: &AppConfig,
    req: &ImageOpRequest,
    command_sha256: [u8; 32],
    format: honkbox_sd_core::OutputFormat,
    jpeg_quality: Option<u8>,
) -> ImageOpSettled {
    let started = Instant::now();
    // Pre-admission: every input is resolved and safety-checked BEFORE a
    // job is registered, so a hostile container is a named 400 and the job
    // registry never sees it.
    let inputs = req.inputs.clone();
    let data_dir = config.data_dir.clone();
    let admitted =
        match tokio::task::spawn_blocking(move || admit_op_inputs(&inputs, data_dir.as_deref()))
            .await
        {
            Ok(Ok(admitted)) => admitted,
            Ok(Err(err)) => return settled_without_job(err),
            Err(join) => {
                return image_op_pre_admission_join_failure(format!(
                    "image-op worker failed: {join}"
                ));
            }
        };
    let job = crate::jobs_registry::global().register(
        crate::jobs_registry::JobKind::ImageOps,
        None,
        req.inputs.len() > 1,
    );
    // Admission (TASK-image-ops-job-route): CPU-only work takes the
    // ledger's CPU-only bucket (`rag`, 0 GPU slots) — the fixed 8-kind
    // vocabulary has no image-ops kind, the same nearest-bucket mapping the
    // audio routes use with `llm`. Image ops therefore contend with rag
    // pipelines on CPU threads instead of bypassing the ledger entirely.
    let lease = match crate::admission::acquire(honkbox_residency::ResourceKind::Rag).await {
        Ok(lease) => lease,
        Err((status, message)) => {
            return ImageOpSettled {
                response: Err((status, message, None)),
                job_id: Some(job.id().to_owned()),
            };
        }
    };
    let command = req.command.clone();
    let token = job.token().clone();
    let work = tokio::task::spawn_blocking(move || {
        run_image_op_units(&command, &admitted, format, jpeg_quality, &token)
    })
    .await;
    drop(lease);
    let (units, input_records) = match settle_image_op_work(job.id(), req.inputs.len(), work) {
        Ok(pair) => pair,
        Err(settled) => return *settled,
    };
    // Bundle (best-effort, exactly like generation: a bundle write failure
    // never discards completed work — the response gains a warning).
    let mut warnings: Vec<String> = Vec::new();
    let bundle = config.data_dir.as_deref().and_then(|data_dir| {
        let output_records: Vec<crate::jobs::OpsOutputRecord> = units
            .iter()
            .enumerate()
            .map(|(i, unit)| crate::jobs::OpsOutputRecord {
                file: format!("op_{i:02}.{}", format.extension()),
                sha256: unit.sha256.clone(),
            })
            .collect();
        let byte_refs: Vec<&[u8]> = units.iter().map(|unit| unit.bytes.as_slice()).collect();
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let ops_inputs = crate::jobs::OpsBundleInputs {
            op: op_label(&req.command).to_owned(),
            command_sha256: sha256_hex(&command_sha256),
            request_id: req.request_id.clone(),
            client: req.client.clone(),
            inputs: input_records,
            format,
        };
        match crate::jobs::write_ops_bundle(
            data_dir,
            &ops_inputs,
            &output_records,
            &byte_refs,
            elapsed_ms,
        ) {
            Ok(dir) => Some(
                dir.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            ),
            Err(err) => {
                tracing::warn!(%err, "image-ops bundle write failed; response still complete");
                warnings.push(
                    "job bundle could not be written; artifact inputs will not resolve".to_owned(),
                );
                None
            }
        }
    });
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    ImageOpSettled {
        response: Ok(ImageOpResponse {
            created,
            job_id: job.id().to_owned(),
            request_id: req.request_id.clone(),
            bundle,
            outputs: units
                .iter()
                .map(|unit| ImageOpOutput {
                    b64_json: base64::engine::general_purpose::STANDARD.encode(&unit.bytes),
                    sha256: unit.sha256.clone(),
                    format: format.name().to_owned(),
                })
                .collect(),
            warnings: (!warnings.is_empty()).then_some(warnings),
        }),
        job_id: Some(job.id().to_owned()),
    }
}

/// A named image-op rejection: status + message + error code — the shape
/// the validators return and the handler turns into the envelope response
/// (the same pattern as `validate_images_request`).
type ImageOpRejection = (StatusCode, String, Option<&'static str>);

/// Validate an image-op request (reject-not-clamp, named 400s): id/client
/// string caps, batch cap, output container, JPEG quality, and the per-op
/// param bounds. `Err(ImageOpRejection)` is the named 400.
fn validate_image_op_request(
    req: &ImageOpRequest,
) -> Result<(honkbox_sd_core::OutputFormat, Option<u8>), ImageOpRejection> {
    if req.request_id.is_empty() || req.request_id.chars().count() > IMAGE_OP_ID_MAX_CHARS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "request_id must be 1..={IMAGE_OP_ID_MAX_CHARS} characters; got {}",
                req.request_id.chars().count()
            ),
            Some("invalid_image_op_request_id"),
        ));
    }
    if let Some(client) = req.client.as_deref()
        && (client.is_empty() || client.chars().count() > IMAGE_OP_ID_MAX_CHARS)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "client must be 1..={IMAGE_OP_ID_MAX_CHARS} characters; got {}",
                client.chars().count()
            ),
            Some("invalid_image_op_request"),
        ));
    }
    if req.inputs.is_empty() || req.inputs.len() > IMAGE_OP_MAX_INPUTS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "inputs must contain 1..={IMAGE_OP_MAX_INPUTS} images; got {}",
                req.inputs.len()
            ),
            Some("invalid_image_op_inputs"),
        ));
    }
    // Output container: the registry is the source of truth (omitted =
    // PNG); unknown names are a named 400 listing the registry-derived
    // accepted values.
    let format = match req.output_format.as_deref().map(str::trim) {
        None => honkbox_sd_core::OutputFormat::Png,
        Some(raw) => parse_output_format(raw)
            .map_err(|(message, code)| (StatusCode::BAD_REQUEST, message, Some(code)))?,
    };
    let jpeg_quality = parse_output_compression(req.output_compression.as_ref())
        .map_err(|(message, code)| (StatusCode::BAD_REQUEST, message, Some(code)))?;
    validate_image_op_params(&req.command)
        .map_err(|(message, code)| (StatusCode::BAD_REQUEST, message, Some(code)))?;
    Ok((format, jpeg_quality))
}

/// `POST /v1/image-ops` (epic-editor-integration): one bounded CPU op over
/// `1..=8` inputs — resize/crop/rotate/grayscale/format — deterministic by
/// construction (Lanczos3, fixed encoders), with `request_id`-keyed
/// idempotency (same key + same canonical command replays the first job's
/// response byte-identically, no double execution; same key + different
/// command is a named 400). The job is registry-visible while live (kind
/// `image_ops`, cancel honored BETWEEN inputs), its lifecycle streams on
/// `GET /v1/events`, and the result bundle lands under
/// `<data_dir>/jobs/{bundle}/` via the existing jobs path (sidecar: command
/// hash, input/output hashes, tool version, attempts, outcome). Inputs are
/// data URIs or this server's own bundle artifacts only — no paths, no
/// `file://`, no remote URLs (nothing is fetched, so there is no SSRF
/// surface). Unsupported-in-principle ops do not exist in the vocabulary:
/// the op set is exactly what the in-process `image` crate covers
/// (TASK-cheap-ops-spike Tier 0), so there is no 501 path today — unknown
/// op tags are a 400 from parsing, and the capabilities advertisement lists
/// the supported vocabulary.
#[utoipa::path(
    post,
    path = "/v1/image-ops",
    request_body = ImageOpRequest,
    responses(
        (status = 200, description = "One output per input, in order", body = ImageOpResponse),
        (status = 400, description = "Unknown field, unknown op, out-of-range params, oversized batch, bad input container/data URI/ artifact, unknown output_format or output_compression, or request_id conflict", body = OpenAiError),
        (status = 409, description = "Job cancelled before completion (between inputs)", body = OpenAiError),
        (status = 429, description = "CPU admission busy (ledger)", body = OpenAiError),
        (status = 500, description = "Decode/transform/encode failed", body = OpenAiError)
    )
)]
async fn post_image_ops(
    State(config): State<AppConfig>,
    body: Result<Json<ImageOpRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(req)) => req,
        Err(rejection) => {
            // deny_unknown_fields / malformed JSON / wrong types all land
            // here as one named 400 carrying serde's specific message.
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid image-op request: {}", rejection.body_text()),
                Some("invalid_image_op_request"),
            );
        }
    };
    let (format, jpeg_quality) = match validate_image_op_request(&req) {
        Ok(validated) => validated,
        Err((status, message, code)) => {
            return openai_error_response(status, message, code);
        }
    };
    // Canonical command hash (the idempotency key's command half): hash the
    // RE-SERIALIZED request — field order is fixed by the DTO, so two wire
    // spellings of the same command collapse into one dedup key while any
    // actual difference (params, inputs, format, …) produces a new hash.
    let command_sha256: [u8; 32] =
        Sha256::digest(serde_json::to_vec(&req).unwrap_or_default()).into();
    let reserved = {
        // std mutex, no await inside: reserve-or-conflict is decided here.
        let mut dedup = IMAGE_OP_DEDUP
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match dedup.reserve(&req.request_id, command_sha256) {
            ImageOpReservation::Conflict => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "request_id {:?} was already used with a different command — retries must resend the identical request; use a fresh request_id for different work",
                        req.request_id
                    ),
                    Some("image_op_request_id_conflict"),
                );
            }
            ImageOpReservation::Execute(entry) => (entry, false),
            ImageOpReservation::Replay(entry) => (entry, true),
        }
    };
    let (entry, is_replay) = reserved;
    // Single flight: whether this caller is the executor or a waiting
    // retry, `get_or_init` runs the execution exactly once per entry —
    // concurrent same-key retries all await the same settled outcome.
    let cmd = req.clone();
    let cfg = config.clone();
    let settled = entry
        .outcome
        .get_or_init(|| async move {
            execute_image_op(&cfg, &cmd, command_sha256, format, jpeg_quality).await
        })
        .await
        .clone();
    if !is_replay {
        // Exactly one caller owns the entry: a success stays remembered
        // (bounded), a failure is forgotten so a genuine retry re-executes.
        let retained = match &settled.response {
            Ok(response) => {
                256 + response
                    .outputs
                    .iter()
                    .map(|out| out.b64_json.len())
                    .sum::<usize>()
            }
            Err((_, message, _)) => 128 + message.len(),
        };
        let mut dedup = IMAGE_OP_DEDUP
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if settled.response.is_ok() {
            dedup.remember(&req.request_id, retained);
        } else {
            dedup.forget(&req.request_id);
        }
    }
    let response = match &settled.response {
        Ok(body) => Json(body.clone()).into_response(),
        Err((status, message, code)) => {
            openai_error_response(*status, message.clone(), code.as_deref())
        }
    };
    match settled.job_id.as_deref() {
        Some(job_id) => with_job_id(response, job_id),
        None => response,
    }
}

/// Shared 501 hint for unsupported recognition kinds: `recognition` is a
/// TOP-LEVEL capabilities field, not a member of `features`.
fn recognition_501_message(detail: &str) -> String {
    format!("{detail}; see GET /v1/capabilities recognition")
}

/// `POST /v1/recognize` (recognition tickets): one image in, recognition
/// out. Stage 1 is synchronous — barcode-class latency is sub-millisecond,
/// so the request-response shape needs no job registry entry; the
/// OCR/document pipeline becomes job-based (cancellable, cache-keyed per
/// FEAT-ocr-ingest) when the first OCR backend lands, and this route is
/// that shape's already-valid request prefix. Only `data:` URI image
/// payloads are accepted: nothing is ever fetched, so there is no SSRF
/// surface (same policy as `/v1/image-ops`). Kinds without a compiled
/// backend answer the named 501 `recognition_kind_unsupported` — the same
/// `kinds()` gate the top-level `recognition` capability advertises, so
/// honest absence is observable from both directions.
#[utoipa::path(
    post,
    path = "/v1/recognize",
    request_body = RecognizeRequest,
    responses(
        (status = 200, description = "Recognition result (`pages: []` when a valid image held nothing recognizable)", body = RecognizeResult),
        (status = 400, description = "Unknown field or kind (typed enum parse), non-`data:` URI, undeclared base64, unsupported container, invalid base64, or empty payload", body = OpenAiError),
        (status = 501, description = "Kind (or engine override) not served by this build's compiled backends (`recognition_kind_unsupported`)", body = OpenAiError),
        (status = 500, description = "Recognition engine failure", body = OpenAiError)
    )
)]
async fn post_recognize(
    State(config): State<AppConfig>,
    body: Result<Json<RecognizeRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(req)) => req,
        Err(rejection) => {
            // deny_unknown_fields / malformed JSON / unknown kind all land
            // here as one named 400 carrying serde's specific message.
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid recognize request: {}", rejection.body_text()),
                Some("invalid_recognize_request"),
            );
        }
    };
    let kind = match req.kind {
        RecKind::Barcode => WorkerRecKind::Barcode,
        RecKind::Ocr => WorkerRecKind::Ocr,
    };
    // Honesty gate FIRST: a kind this build cannot serve is a 501 even when
    // the payload is also broken — the kind is the primary selector, and the
    // 501 message is what a capability-driven client is built to handle.
    if !config.recognition.kinds().contains(&kind) {
        return openai_error_response(
            StatusCode::NOT_IMPLEMENTED,
            recognition_501_message(&format!(
                "recognition kind {:?} is not served by this build (backend {:?} serves {:?})",
                kind.as_str(),
                config.recognition.backend_id(),
                config
                    .recognition
                    .kinds()
                    .iter()
                    .map(|k| k.as_str())
                    .collect::<Vec<_>>()
            )),
            Some("recognition_kind_unsupported"),
        );
    }
    // The route owns input resolution: worker backends only ever see
    // decoded container bytes, never paths or URLs.
    let bytes = match decode_recognize_image(&req.image) {
        Ok(bytes) => bytes,
        Err((message, code)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
    };
    let worker_req = WorkerRecognizeRequest {
        kind,
        image: bytes,
        language: req.language.clone(),
        mode: req.mode.clone(),
        engine: req.engine.clone(),
    };
    let out = match config.recognition.recognize(&worker_req) {
        Ok(out) => out,
        Err(RecError::Unsupported { kind, backend }) => {
            return openai_error_response(
                StatusCode::NOT_IMPLEMENTED,
                recognition_501_message(&format!(
                    "recognition kind {:?} is not served by backend {backend:?}",
                    kind.as_str()
                )),
                Some("recognition_kind_unsupported"),
            );
        }
        Err(RecError::Invalid(message)) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                message,
                Some("invalid_recognize_input"),
            );
        }
        Err(RecError::Engine(message)) => {
            // Metric label carries the kind, never the image or its text.
            tracing::warn!(kind = kind.as_str(), "recognition engine failure");
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("recognition engine failed: {message}"),
                Some("recognition_engine_failed"),
            );
        }
    };
    Json(RecognizeResult {
        kind: req.kind,
        engine: RecognitionEngineMeta {
            id: out.engine.id.to_owned(),
            version: out.engine.version.map(str::to_owned),
            bbox_support: out.engine.bbox_support,
        },
        pages: out
            .pages
            .into_iter()
            .map(|page| RecognitionPage {
                text: page.text,
                regions: page
                    .regions
                    .into_iter()
                    .map(|region| RecognitionRegion {
                        bbox: region.bbox,
                        text: region.text,
                        confidence: region.confidence,
                    })
                    .collect(),
            })
            .collect(),
    })
    .into_response()
}

/// Decode the recognize route's `data:image/…;base64,…` input. Same MIME
/// vocabulary + base64 tolerances as the image-ops path (`decode_data_uri`),
/// minus its artifact/FormatInfo plumbing — recognition stage 1 takes
/// exactly one image and no bundle inputs.
fn decode_recognize_image(uri: &str) -> Result<Vec<u8>, (String, &'static str)> {
    let bad = |message: String| (message, "invalid_recognize_input");
    let trimmed = uri.trim();
    let Some(rest) = trimmed.strip_prefix("data:") else {
        return Err(bad(
            "image must be a data URI (`data:image/…;base64,…`); paths and remote URLs are never fetched".to_owned(),
        ));
    };
    let Some((header, payload)) = rest.split_once(',') else {
        return Err(bad(
            "data URI is missing the `,<base64 payload>` part".to_owned()
        ));
    };
    let Some(mime) = header.strip_suffix(";base64") else {
        return Err(bad("data URI must declare `;base64`".to_owned()));
    };
    if crate::formats::image_ops_by_mime(mime).is_none() {
        return Err(bad(format!(
            "unsupported image container {mime:?}: accepted data URIs are {}",
            crate::formats::image_ops_inputs()
                .map(|row| row.mime)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        // Tolerate padding-stripped payloads (some encoders trim `=`).
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(payload.trim()))
        .map_err(|e| bad(format!("data URI is not valid base64: {e}")))?;
    if bytes.is_empty() {
        return Err(bad("image payload is empty".to_owned()));
    }
    Ok(bytes)
}

/// `POST /v1/audio/speech` (REQ-9): `OpenAI`-shaped TTS. Same worker as
/// `/v1/synthesize`. `model` optional (server default TTS), `voice` passed
/// through; `response_format` is `"wav"` (default, RIFF/WAVE container) or
/// `"pcm"` — raw little-endian signed 16-bit mono samples with **no**
/// container: the worker already returns mono PCM16, so the pcm path
/// serializes the exact same samples minus the 44-byte RIFF header (`OpenAI`
/// pcm semantics). `response_format` accepts `wav` (default) or `pcm`.
#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    request_body = SpeechRequest,
    responses(
        (status = 200, description = "WAV audio (RIFF container)", content_type = "audio/wav"),
        (status = 200, description = "Raw s16le mono PCM (pcm response_format; no container)", content_type = "audio/pcm-s16le"),
        (status = 400, description = "Empty input, oversized input (over 4096 characters), unknown model, or unsupported response_format", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Synthesis failed", body = OpenAiError)
    )
)]
async fn post_audio_speech(
    State(config): State<AppConfig>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    let pcm_out = match req.response_format.as_deref() {
        None | Some("wav") => false,
        Some("pcm") => true,
        Some(other) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "unsupported response_format {other:?}: only \"wav\" and \"pcm\" are produced"
                ),
                None,
            );
        }
    };
    if req.input.trim().is_empty() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "input is required: the text to synthesize".to_owned(),
            None,
        );
    }
    // TASK-speech-input-cap: unbounded `input` means unbounded PCM growth;
    // reject-not-clamp (no silent truncation, no derived clamp).
    let input_chars = req.input.chars().count();
    if input_chars > SPEECH_INPUT_MAX_CHARS {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "input is {input_chars} characters; the cap is {SPEECH_INPUT_MAX_CHARS} — rejected, not truncated"
            ),
            Some("input_too_long"),
        );
    }
    let catalog = config.catalog_snapshot();
    if let Err(message) =
        validate_model(req.model.as_deref(), config.tts_model.as_deref(), &catalog)
    {
        return openai_error_response(StatusCode::BAD_REQUEST, message, None);
    }
    let worker_req = WorkerSynthRequest {
        model: req.model.unwrap_or_default(),
        text: req.input,
        voice: req.voice,
        sample_rate: None,
    };
    // NEW-3: registry-visible while the request lives (one blocking FFI
    // call — the cancel token is marked but never checked).
    let job = crate::jobs_registry::global().register(
        crate::jobs_registry::JobKind::Synthesize,
        (!worker_req.model.is_empty()).then_some(worker_req.model.clone()),
        false,
    );
    let audio = match run_synthesis(config.tts.clone(), worker_req).await {
        Ok(audio) => audio,
        Err((StatusCode::TOO_MANY_REQUESTS, message)) => {
            return with_job_id(
                openai_error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    message,
                    Some("gpu_slot_busy"),
                ),
                job.id(),
            );
        }
        Err((code, message)) => return openai_error_response(code, message, None),
    };
    if pcm_out {
        // Raw s16le: the exact samples the wav path wraps, header omitted.
        let mut raw = Vec::with_capacity(audio.pcm16.len() * 2);
        for sample in &audio.pcm16 {
            raw.extend_from_slice(&sample.to_le_bytes());
        }
        return with_job_id(
            ([(header::CONTENT_TYPE, "audio/pcm-s16le")], raw).into_response(),
            job.id(),
        );
    }
    let wav = encode_wav(audio.sample_rate, &audio.pcm16);
    with_job_id(
        ([(header::CONTENT_TYPE, "audio/wav")], wav).into_response(),
        job.id(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelCapabilitiesOverride, ModelEntryConfig};

    /// The registry's write rows are EXACTLY the compiled encoders — set
    /// and order on both directions. This is the lock that makes
    /// `formats.write` (derived through the registry) byte-compat with the
    /// pre-registry `OutputFormat::ALL` derivation, and it lives HERE (the
    /// routes layer owns the wire-name → enum mapping, so a `#[path]`
    /// -included formats module never links the native backends).
    #[test]
    fn format_registry_matches_the_compiled_encoders() {
        for format in honkbox_sd_core::OutputFormat::ALL {
            assert!(
                crate::formats::output_rows().any(|row| row.output && row.name == format.name()),
                "{} encoder missing from the registry write column",
                format.name()
            );
            assert!(
                parse_output_format(format.name()).is_ok(),
                "{} must parse",
                format.name()
            );
        }
        for row in crate::formats::output_rows() {
            assert!(
                honkbox_sd_core::OutputFormat::from_name(row.name).is_some(),
                "registry write row {:?} has no compiled encoder",
                row.name
            );
        }
        // Unknown names (and wrong-case spellings) stay rejected, naming
        // the registry-derived accepted values.
        for raw in ["gif", "JPEG", "Png", "", "svg", "ora"] {
            let (message, code) = parse_output_format(raw).expect_err(raw);
            assert_eq!(code, "unsupported_output_format");
            assert!(message.contains(raw), "{raw}: {message}");
            assert!(message.contains("\"png\""), "{raw}: {message}");
        }
    }

    /// SRT timestamps use the `SubRip` comma separator; hours pad to two
    /// digits and roll over correctly at minute/second/hour boundaries.
    #[test]
    fn srt_timestamps_render_deterministically() {
        assert_eq!(srt_timestamp(0), "00:00:00,000");
        assert_eq!(srt_timestamp(100), "00:00:00,100");
        assert_eq!(srt_timestamp(3_661_001), "01:01:01,001");
        assert_eq!(srt_timestamp(u32::MAX), "1193:02:47,295");
    }

    /// VTT timestamps use the dot separator; everything else matches SRT.
    #[test]
    fn vtt_timestamps_render_deterministically() {
        assert_eq!(vtt_timestamp(0), "00:00:00.000");
        assert_eq!(vtt_timestamp(1_000), "00:00:01.000");
        assert_eq!(vtt_timestamp(3_661_001), "01:01:01.001");
    }

    /// SRT cue rendering: 1-based numbering, arrow line, blank-line tail.
    #[test]
    fn srt_renders_numbered_cues() {
        let segments = vec![
            WorkerTranscriptSegment {
                start_ms: 0,
                end_ms: 1_500,
                text: "hello".to_owned(),
            },
            WorkerTranscriptSegment {
                start_ms: 2_000,
                end_ms: 3_250,
                text: "world".to_owned(),
            },
        ];
        assert_eq!(
            render_srt(&segments),
            "1\n00:00:00,000 --> 00:00:01,500\nhello\n\n\
             2\n00:00:02,000 --> 00:00:03,250\nworld\n\n"
        );
        // No segments = an empty (valid) subtitle track.
        assert_eq!(render_srt(&[]), "");
    }

    /// VTT cue rendering: the WEBVTT header, then the same cue shape.
    #[test]
    fn vtt_renders_header_and_cues() {
        let segments = vec![WorkerTranscriptSegment {
            start_ms: 0,
            end_ms: 1_500,
            text: "hello".to_owned(),
        }];
        assert_eq!(
            render_vtt(&segments),
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.500\nhello\n\n"
        );
        // Header-only track for a no-speech input.
        assert_eq!(render_vtt(&[]), "WEBVTT\n\n");
    }
    use honkbox_registry::RootKind;

    /// One scan hit for a synthetic file path.
    fn scan_hit(slug: &str, path: &str) -> honkbox_registry::FoundModel {
        honkbox_registry::FoundModel {
            root: RootKind::Llm,
            category: "root".to_owned(),
            path: PathBuf::from(path),
            slug: slug.to_owned(),
            bytes: 1,
        }
    }

    /// One `[[models.entry]]` declaration.
    fn model_entry(
        slug: Option<&str>,
        file_match: Option<&str>,
        aliases: &[&str],
        capabilities: ModelCapabilitiesOverride,
    ) -> ModelEntryConfig {
        ModelEntryConfig {
            slug: slug.map(str::to_owned),
            file_match: file_match.map(str::to_owned),
            aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
            capabilities,
        }
    }

    #[test]
    fn bind_binds_slug_and_match_targets() {
        let found = vec![
            scan_hit("llm:a/snap/sha1/m.gguf", "/r/a/snap/sha1/m.gguf"),
            scan_hit("llm:b/w.bin", "/r/b/w.bin"),
        ];
        let entries = vec![
            model_entry(
                Some("llm:a/snap/sha1/m.gguf"),
                None,
                &["alpha"],
                ModelCapabilitiesOverride::default(),
            ),
            model_entry(
                None,
                Some("w.bin"),
                &["whisper"],
                ModelCapabilitiesOverride::default(),
            ),
        ];
        let bindings = bind_entries(&entries, &found).expect("bind");
        let catalog = build_catalog(&found, &bindings);
        assert_eq!(catalog.data.len(), 2);
        assert_eq!(catalog.data[0].id, "alpha");
        assert_eq!(catalog.data[0].slug, "llm:a/snap/sha1/m.gguf");
        assert_eq!(catalog.data[1].id, "whisper");
        // The flat loop-lore fields mirror the capabilities object.
        assert_eq!(catalog.data[0].modalities, vec!["text"]);
        assert!(catalog.data[0].capabilities.chat);
        assert!(!catalog.data[0].tool_calling);
        assert_eq!(catalog.data[0].owned_by, "local");
        assert_eq!(catalog.data[0].context_length, None);
        assert_eq!(catalog.data[0].max_output, None);
        assert!(!catalog.data[0].thinking);
    }

    /// Alias conflicts are hard errors naming both paths (binding owner
    /// decision: never a silent first-wins).
    #[test]
    fn bind_conflicts_are_errors() {
        let found = vec![
            scan_hit("llm:a/m1.gguf", "/r/a/m1.gguf"),
            scan_hit("llm:b/m2.gguf", "/r/b/m2.gguf"),
        ];

        // The same alias for two files.
        let entries = vec![
            model_entry(
                Some("llm:a/m1.gguf"),
                None,
                &["same"],
                ModelCapabilitiesOverride::default(),
            ),
            model_entry(
                Some("llm:b/m2.gguf"),
                None,
                &["same"],
                ModelCapabilitiesOverride::default(),
            ),
        ];
        let err = bind_entries(&entries, &found).expect_err("duplicate alias");
        assert!(err.contains("m1.gguf") && err.contains("m2.gguf"), "{err}");

        // An alias that collides with another model's registry slug.
        let entries = vec![model_entry(
            Some("llm:a/m1.gguf"),
            None,
            &["llm:b/m2.gguf"],
            ModelCapabilitiesOverride::default(),
        )];
        let err = bind_entries(&entries, &found).expect_err("slug collision");
        assert!(err.contains("collides"), "{err}");

        // Two entries binding the same file.
        let entries = vec![
            model_entry(
                Some("llm:a/m1.gguf"),
                None,
                &[],
                ModelCapabilitiesOverride::default(),
            ),
            model_entry(
                None,
                Some("m1.gguf"),
                &[],
                ModelCapabilitiesOverride::default(),
            ),
        ];
        let err = bind_entries(&entries, &found).expect_err("double bind");
        assert!(err.contains("two [[models.entry]] entries bind"), "{err}");

        // An alias-bearing match hitting several files.
        let multi = vec![
            scan_hit("llm:a/m1.gguf", "/r/a/m1.gguf"),
            scan_hit("llm:b/m1.gguf", "/r/b/m1.gguf"),
        ];
        let entries = vec![model_entry(
            None,
            Some("m1.gguf"),
            &["x"],
            ModelCapabilitiesOverride::default(),
        )];
        let err = bind_entries(&entries, &multi).expect_err("ambiguous match");
        assert!(err.contains("binds 2 files"), "{err}");
    }

    /// Operator capability declarations override the kind-inferred
    /// defaults; `chat = false` + `embeddings = true` reclassifies the
    /// emitted `kind` (embedding GGUF discoverability).
    #[test]
    fn capabilities_override_kind_defaults() {
        let found = vec![scan_hit("llm:e/embed.gguf", "/r/e/embed.gguf")];
        let entries = vec![model_entry(
            Some("llm:e/embed.gguf"),
            None,
            &["embed-0.6b"],
            ModelCapabilitiesOverride {
                chat: Some(false),
                embeddings: Some(true),
                ..Default::default()
            },
        )];
        let bindings = bind_entries(&entries, &found).expect("bind");
        let catalog = build_catalog(&found, &bindings);
        let entry = &catalog.data[0];
        assert_eq!(entry.id, "embed-0.6b");
        assert_eq!(entry.kind, ModelKind::Embedding);
        assert!(!entry.capabilities.chat);
        assert!(!entry.capabilities.completion);
        assert!(entry.capabilities.embeddings);
        assert!(!entry.tool_calling);

        // A partial declaration keeps the inferred defaults for the rest.
        let entries = vec![model_entry(
            Some("llm:e/embed.gguf"),
            None,
            &[],
            ModelCapabilitiesOverride {
                tool_calling: Some(true),
                vision: Some(true),
                ..Default::default()
            },
        )];
        let bindings = bind_entries(&entries, &found).expect("bind");
        let catalog = build_catalog(&found, &bindings);
        let entry = &catalog.data[0];
        assert_eq!(entry.kind, ModelKind::Llm);
        assert!(entry.capabilities.chat);
        assert!(entry.capabilities.tool_calling);
        assert!(entry.capabilities.vision);
        assert_eq!(entry.modalities, vec!["text", "vision"]);
        assert!(entry.tool_calling);
    }

    /// Default-on vision chat: a chat model with a sibling `mmproj*.gguf`
    /// projector advertises `capabilities.vision` with no operator
    /// declaration, because the default feature set compiles the mtmd
    /// path. `/v1/models` must never advertise vision the build cannot
    /// serve, so a text-only build (no `llm-vision`) keeps the same
    /// layout text-only — the assertion tracks `VISION_COMPILED` for
    /// both configs.
    #[test]
    fn sibling_mmproj_derives_vision_by_default() {
        let base =
            std::env::temp_dir().join(format!("honkbox-vision-catalog-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("test dir");
        let model_path = base.join("m1.gguf");
        std::fs::write(&model_path, b"m").expect("write model file");
        let mmproj_path = base.join("mmproj-m1-f16.gguf");
        std::fs::write(&mmproj_path, b"p").expect("write mmproj file");

        let found = vec![scan_hit("llm:v/m1.gguf", &model_path.display().to_string())];
        let bindings = bind_entries(&[], &found).expect("bind");
        let catalog = build_catalog(&found, &bindings);
        let entry = &catalog.data[0];
        std::fs::remove_dir_all(&base).ok();
        assert_eq!(
            entry.capabilities.vision,
            honkbox_llm_worker::VISION_COMPILED
        );
        let expected = if entry.capabilities.vision {
            vec!["text".to_owned(), "vision".to_owned()]
        } else {
            vec!["text".to_owned()]
        };
        assert_eq!(entry.modalities, expected);
    }

    /// A backend that ignores the cancel flag must not answer 200 with a
    /// full image after the token was marked.
    struct CancelIgnoringPipeline;

    impl honkbox_sd_core::SdPipeline for CancelIgnoringPipeline {
        fn generate(
            &self,
            _req: &honkbox_sd_core::Txt2ImgRequest,
            _cancel: &AtomicBool,
        ) -> Result<honkbox_sd_core::GeneratedImage, honkbox_sd_core::SdError> {
            Ok(honkbox_sd_core::GeneratedImage {
                width: 1,
                height: 1,
                rgba: vec![1, 2, 3, 255],
            })
        }

        fn backend_id(&self) -> &'static str {
            "cancel-ignoring-test"
        }
    }

    fn one_pixel_request() -> honkbox_sd_core::Txt2ImgRequest {
        honkbox_sd_core::Txt2ImgRequest {
            prompt: String::new(),
            seed: 1,
            steps: 1,
            width: 1,
            height: 1,
            model_slug: String::new(),
            clip: None,
            vae: None,
            sampler: None,
            scheduler: None,
            cfg_scale: None,
            negative_prompt: None,
            loras: None,
            init_image: None,
            mask_image: None,
            strength: None,
            hires: None,
            control_net: None,
            control_image: None,
            controlnet_strength: None,
            control_type: None,
        }
    }

    #[tokio::test]
    async fn cancelled_token_beats_a_backend_that_returns_ok() {
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        let outcome = generate_image_bytes(
            Arc::new(CancelIgnoringPipeline),
            one_pixel_request(),
            honkbox_sd_core::OutputFormat::Png,
            None,
            new_image_cancel_flag(),
            token,
        )
        .await;
        assert!(matches!(outcome, Err(GenerateFailure::Cancelled)));
    }

    #[test]
    fn each_image_gets_a_fresh_cancel_flag() {
        let first = new_image_cancel_flag();
        let second = new_image_cancel_flag();
        first.store(true, Ordering::Release);
        assert!(!second.load(Ordering::Acquire));
        assert!(!Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn image_cancel_messages_name_the_phase() {
        let before = image_cancel_before_start("job_1", 2, 3, 1);
        let during = image_cancel_mid_generation("job_1", 2, 3, 1);
        assert!(before.contains("before image 2 of 3 started"), "{before}");
        assert!(!before.contains("while generating"), "{before}");
        assert!(during.contains("while generating image 2 of 3"), "{during}");
        assert!(!during.contains("before image"), "{during}");
        assert!(before.contains("job_1") && during.contains("job_1"));
    }

    #[test]
    fn admission_rejection_registers_no_job_and_has_no_job_header() {
        let registry = crate::jobs_registry::Registry::default();
        let mut events = registry.subscribe();
        let rejection = (StatusCode::TOO_MANY_REQUESTS, "slot busy".to_owned());
        let result = admit_then_register(
            &registry,
            Err(rejection.clone()),
            crate::jobs_registry::JobKind::Chat,
            Some("m".to_owned()),
            true,
        );
        let error = result.expect_err("admission rejected");
        assert_eq!(error, rejection);
        assert!(registry.list().is_empty());
        assert!(events.try_recv().is_err(), "no job_registered on a 429");
        let response: Response = (error.0, error.1).into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response.headers().get(JOB_ID_HEADER).is_none());
    }

    #[test]
    fn pre_admission_join_failure_stamps_a_job_id() {
        let settled = image_op_pre_admission_join_failure("worker died".to_owned());
        match settled.response {
            Err((status, message, code)) => {
                assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
                assert_eq!(message, "worker died");
                assert_eq!(code, None);
            }
            Ok(_) => panic!("join failure must be a 500"),
        }
        let id = settled.job_id.expect("id stamped");
        assert!(id.starts_with("job_"), "{id}");
    }

    #[test]
    fn recognition_501_points_at_the_top_level_capability() {
        let message = recognition_501_message("unsupported kind");
        assert!(
            message.contains("GET /v1/capabilities recognition"),
            "{message}"
        );
        assert!(!message.contains("features.recognition"), "{message}");
    }
}
