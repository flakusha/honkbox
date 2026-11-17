//! HTTP surface: `/health`, `/api/version`, `/v1/models`(+`/refresh`),
//! `/metrics`, `/v1/openapi.json`, chat/transcribe/synthesize, image
//! generations (REQ-10), UI fallback.
//!
//! The `OpenAPI` document is served as plain JSON via `Json` — no
//! `utoipa-axum` router glue.

use std::collections::HashMap;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{Multipart, Path as AxumPath, Query, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, Uri, header},
    middleware::{self, Next},
    response::{
        Html, IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use base64::Engine as _;
use honkbox_api_types::{
    CatalogEntry, CatalogList, CatalogLocal, ChatContent, ChatMessage, ChatRequest, ChatResponse,
    ChatRole, ChatTool, DeltaFunction, DeltaToolCall, DownloadAccepted, DownloadKind, DownloadList,
    DownloadRequest, DownloadState, DownloadStatus, EmbeddingsData, EmbeddingsInput,
    EmbeddingsRequest, EmbeddingsResponse, EmbeddingsUsage, FinishReason, FunctionCall,
    FunctionDefinition, GgufMetadata, Health, ImageB64, ImagesGenerationRequest,
    ImagesGenerationResponse, JsonSchemaSpec, MemoryCapacity, MemoryGpu, MemoryInfo,
    ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind, ModelMetadata, NamedFunction,
    OpenAiChatCompletion, OpenAiChatRequest, OpenAiChoice, OpenAiChunk, OpenAiChunkChoice,
    OpenAiDelta, OpenAiError, OpenAiUsage, Pill, ResidentModel, ResponseFormat, ResponseFormatType,
    RunningList, RunningModel, SafetensorsMetadata, SpeechRequest, SynthesizeRequest, ToolCall,
    ToolCallType, ToolChoiceSpec, ToolType, Transcript, TranscriptSegment, TranscriptionText,
    TranscriptionVerbose, UnloadAllResult, UnloadResult, Version,
};

/// Re-exported auth settings (REQ-7): `app` consumers (integration tests,
/// `main`) build `AppConfig.auth` from these.
#[allow(unused_imports)] // which name is used varies per #[path]-included test binary
pub use crate::config::{AuthConfig, AuthMode};
use honkbox_audio_worker::{
    SttPipeline, SynthesizeRequest as WorkerSynthRequest, SynthesizedAudio,
    TranscribeRequest as WorkerTranscribeRequest, Transcript as WorkerTranscript, TtsPipeline,
    decode_wav, encode_wav,
};
use honkbox_llm_worker::{
    ChatMessage as WorkerChatMessage, ChatRequest as WorkerChatRequest, LlmError, LlmPipeline,
    Role, SamplerParams, ToolCallOut, ToolCallSpec,
};
use rust_embed::Embed;
use serde::Deserialize;
use tokio::sync::{Notify, mpsc};
use tokio_stream::{Stream, wrappers::UnboundedReceiverStream};
use utoipa::OpenApi;

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
        get_models,
        refresh_models,
        post_model_unload,
        get_memory,
        openapi_json,
        metrics_endpoint,
        post_chat,
        post_chat_stream,
        post_chat_completions,
        post_transcribe,
        post_synthesize,
        post_download,
        list_downloads,
        get_download,
        delete_download,
        get_catalog,
        get_running,
        post_unload_all,
        post_unload_one
    ),
    components(schemas(
        Health,
        Version,
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
        ImageB64,
        ImagesGenerationResponse,
        DownloadKind,
        DownloadStatus,
        DownloadRequest,
        DownloadAccepted,
        DownloadState,
        DownloadList,
        Pill,
        CatalogLocal,
        CatalogEntry,
        CatalogList
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
    /// Auth settings (REQ-7): `[server] auth` mode + `[auth] api_key_env`
    /// NAME. The middleware reads the key VALUE per request via `std::env`
    /// (rotation without restart); the value is never stored or logged.
    pub auth: crate::config::AuthConfig,
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

/// Build the full router.
pub fn app(config: AppConfig) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/version", get(get_version))
        .route("/v1/models", get(get_models))
        .route("/v1/models/refresh", post(refresh_models))
        .route("/v1/models/{slug}/unload", post(post_model_unload))
        .route("/v1/catalog", get(get_catalog))
        .route("/v1/memory", get(get_memory))
        .route("/v1/openapi.json", get(openapi_json))
        .route("/v1/chat", post(post_chat))
        .route("/v1/chat/stream", post(post_chat_stream))
        .route("/v1/chat/completions", post(post_chat_completions))
        .route("/v1/embeddings", post(post_embeddings))
        .route("/v1/images/generations", post(post_images_generations))
        .route("/v1/transcribe", post(post_transcribe))
        .route("/v1/synthesize", post(post_synthesize))
        .route("/v1/audio/transcriptions", post(post_audio_transcriptions))
        .route("/v1/audio/speech", post(post_audio_speech))
        .route("/running", get(get_running))
        .route("/api/models/unload", post(post_unload_all))
        .route("/api/models/unload/{model}", post(post_unload_one))
        .route("/v1/downloads", post(post_download).get(list_downloads))
        .route(
            "/v1/downloads/{id}",
            get(get_download).delete(delete_download),
        )
        .route("/metrics", get(metrics_endpoint))
        .route_layer(middleware::from_fn_with_state(
            config.clone(),
            require_bearer,
        ))
        .fallback(fallback)
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
/// (see `metrics`); no per-request allocation. Every 429 (the GPU-slot busy
/// contract, shared by the native and `OpenAI` chat routes) also carries a
/// `Retry-After` hint — additive on the native route, whose body/status do
/// not change.
async fn track_metrics(req: Request, next: Next) -> Response {
    let route = crate::metrics::route_label(req.uri().path());
    let started = Instant::now();
    let mut response = next.run(req).await;
    if response.status() == StatusCode::TOO_MANY_REQUESTS {
        response
            .headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    }
    crate::metrics::record_request(route, response.status().as_u16(), started.elapsed());
    response
}

/// Liveness probe. Content-negotiated: the JSON `Health` body by default
/// (and for `Accept: application/json`), a literal `OK` text/plain body for
/// probes that prefer `text/plain` (uptime monitors, `docker --health-cmd`
/// equivalents).
#[utoipa::path(get, path = "/health", responses((status = 200, description = "Server is alive", body = Health)))]
async fn health(headers: HeaderMap) -> Response {
    if prefers_text_plain(&headers) {
        return ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], "OK").into_response();
    }
    Json(Health {
        status: "ok".to_owned(),
    })
    .into_response()
}

/// True when the request's `Accept` header ranks `text/plain` strictly above
/// `application/json` (per-entry q-values, highest wins). No header,
/// wildcards (`*/*`, `text/*`), other media types, or a tie all keep the
/// JSON default — only an explicit text/plain preference flips the body.
fn prefers_text_plain(headers: &HeaderMap) -> bool {
    let Some(value) = headers.get(header::ACCEPT).and_then(|v| v.to_str().ok()) else {
        return false;
    };
    let mut plain_q = None::<f32>;
    let mut json_q = None::<f32>;
    for media in value.split(',') {
        let Some((media_type, params)) = media.split_once(';') else {
            note_media_type(media.trim(), &mut plain_q, &mut json_q, 1.0);
            continue;
        };
        let q = params
            .split(';')
            .find_map(|param| param.trim().strip_prefix("q="))
            .and_then(|q| q.trim().parse::<f32>().ok())
            .unwrap_or(1.0);
        note_media_type(media_type.trim(), &mut plain_q, &mut json_q, q);
    }
    plain_q.is_some() && plain_q > json_q
}

/// Track the best q-value seen for the two media types `/health` picks
/// between; everything else is ignored (wildcards keep the JSON default).
fn note_media_type(media_type: &str, plain_q: &mut Option<f32>, json_q: &mut Option<f32>, q: f32) {
    match media_type.to_ascii_lowercase().as_str() {
        "text/plain" => *plain_q = Some(plain_q.map_or(q, |best| best.max(q))),
        "application/json" => *json_q = Some(json_q.map_or(q, |best| best.max(q))),
        _ => {}
    }
}

/// Binary name + version.
#[utoipa::path(get, path = "/api/version", responses((status = 200, description = "Binary name and version", body = Version)))]
async fn get_version() -> Json<Version> {
    Json(Version {
        name: "honkbox".to_owned(),
        version: env!("CARGO_PKG_VERSION").to_owned(),
    })
}

/// `OpenAPI` 3.1 document as JSON.
#[utoipa::path(get, path = "/v1/openapi.json", responses((status = 200, description = "OpenAPI document")))]
async fn openapi_json() -> impl IntoResponse {
    Json(ApiDoc::openapi())
}

/// `GET /metrics`: minimal Prometheus text exposition (vLLM-vocabulary
/// subset; see `metrics`).
#[utoipa::path(
    get,
    path = "/metrics",
    responses((status = 200, description = "Prometheus text exposition", content_type = "text/plain"))
)]
async fn metrics_endpoint() -> Response {
    // The download metric families live in `downloads.rs` (its own
    // AtomicU64 grids); their exposition fragment is appended here.
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        crate::metrics::exposition() + &crate::downloads::exposition(),
    )
        .into_response()
}

/// `GET /v1/models`: the discovered-model catalog as an OpenAI-style list
/// envelope (`{object: "list", data: [...]}`). Served from the shared scan
/// snapshot; `POST /v1/models/refresh` re-scans and swaps it.
#[utoipa::path(get, path = "/v1/models", responses((status = 200, description = "Discovered model catalog", body = ModelCatalog)))]
async fn get_models(State(config): State<AppConfig>) -> Response {
    let catalog = config.catalog_snapshot();
    // Serialized inline from the shared snapshot (`Arc<T>` only serializes
    // with serde's `rc` feature, so `Json(Arc<…>)` is not `IntoResponse`);
    // one `String` allocation beats cloning every entry into an owned `Json`.
    match serde_json::to_string(&*catalog) {
        Ok(body) => ([(header::CONTENT_TYPE, "application/json")], body).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("catalog serialize failed: {err}"),
        )
            .into_response(),
    }
}

/// `POST /v1/models/refresh`: re-scan the configured registry roots and
/// swap the shared catalog snapshot, returning the fresh body. Idempotent;
/// missing or unreadable roots never fail the request (the scan is lenient —
/// they contribute empty results). Root *configuration* still needs a
/// restart; refresh picks up files added under the configured roots.
#[utoipa::path(post, path = "/v1/models/refresh", responses((status = 200, description = "Freshly scanned model catalog", body = ModelCatalog)))]
async fn refresh_models(State(config): State<AppConfig>) -> Response {
    match rescan_catalog(&config).await {
        Ok(catalog) => match serde_json::to_string(&catalog) {
            Ok(body) => ([(header::CONTENT_TYPE, "application/json")], body).into_response(),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("catalog serialize failed: {err}"),
            )
                .into_response(),
        },
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err).into_response(),
    }
}

/// Alias-or-slug resolution for the control routes: the owning registry
/// slug for a request id, `None` when the catalog has no such entry.
fn control_slug<'a>(model: &str, catalog: &'a ModelCatalog) -> Option<&'a str> {
    resolve_model_slug(model, catalog)
}

/// Map an unload outcome to its HTTP response. `Pinned` is an explicit 409
/// refusal (a loud log accompanies it) - pinned residents are never evicted
/// silently; `NotLoaded` is a 200 idempotent no-op, deliberately NOT the
/// llama-swap 404 `no local server found` (loop-lore's client treats any
/// 200 as an unconditional continue-case, so 200 needs no body matching).
fn unload_response(outcome: crate::swap::EvictOutcome, slug: String) -> Response {
    match outcome {
        crate::swap::EvictOutcome::Evicted => {
            tracing::info!(slug, "explicit unload evicted resident model");
            Json(UnloadResult {
                unloaded: true,
                slug,
                reason: "explicit".to_owned(),
            })
            .into_response()
        }
        crate::swap::EvictOutcome::NotLoaded => Json(UnloadResult {
            unloaded: false,
            slug,
            reason: "not_loaded".to_owned(),
        })
        .into_response(),
        crate::swap::EvictOutcome::Pinned => {
            tracing::warn!(slug, "explicit unload refused: model is pinned");
            (
                StatusCode::CONFLICT,
                format!("model {slug:?} is pinned via [swap].pin; remove the pin to unload it"),
            )
                .into_response()
        }
    }
}

/// `POST /v1/models/{slug}/unload`: explicitly evict one resident model
/// (REQ-5). Accepts an operator alias or the registry slug. A resident
/// non-pinned entry is evicted (`{"unloaded":true,"reason":"explicit"}`);
/// a non-resident or unknown id is an idempotent 200
/// (`{"unloaded":false,"reason":"not_loaded"}`); a pinned resident is
/// refused with 409 and a loud log. In-flight requests are never aborted:
/// eviction only drops the cache's own `Arc` handle.
#[utoipa::path(
    post,
    path = "/v1/models/{slug}/unload",
    params(("slug" = String, Path, description = "Operator alias or registry slug")),
    responses(
        (status = 200, description = "Unloaded or not resident", body = UnloadResult),
        (status = 409, description = "Model is pinned; explicit refusal")
    )
)]
async fn post_model_unload(
    State(config): State<AppConfig>,
    AxumPath(model): AxumPath<String>,
) -> Response {
    let catalog = config.catalog_snapshot();
    // Unknown ids resolve to themselves: the answer is the same idempotent
    // not-loaded no-op either way.
    let slug = control_slug(&model, &catalog).map_or_else(|| model.clone(), str::to_owned);
    let outcome = match &config.swap {
        Some(swap) => swap.evict_by_slug(&slug, crate::metrics::UnloadReason::Explicit),
        None => crate::swap::EvictOutcome::NotLoaded,
    };
    unload_response(outcome, slug)
}

/// `GET /v1/memory`: residency readout (REQ-5) - exactly the models the
/// swap manager considers resident, with `loaded_at`, `expires_at`,
/// `pinned`, and the best-effort `in_use` flag, plus the configured
/// capacity and the GPU slot state. Without a swap manager (stub backends)
/// the resident list is empty and capacity reads zeros - nothing is
/// managed.
#[utoipa::path(
    get,
    path = "/v1/memory",
    responses((status = 200, description = "Residency readout", body = MemoryInfo))
)]
async fn get_memory(State(config): State<AppConfig>) -> Response {
    let catalog = config.catalog_snapshot();
    let (resident, capacity, bytes_reserved) = match &config.swap {
        Some(swap) => {
            let entries = swap.residency();
            // REQ-12: per-entry estimated bytes from the catalog snapshot's
            // registry scan (`bytes` = on-disk weight file size — an
            // estimate, never a measurement); unknown slugs (catalog no
            // longer holds them) contribute nothing.
            let resident: Vec<ResidentModel> = entries
                .iter()
                .map(|entry| ResidentModel {
                    alias: catalog
                        .data
                        .iter()
                        .find(|e| e.slug == entry.slug && e.id != e.slug)
                        .map(|e| e.id.clone()),
                    kind: "llm".to_owned(),
                    slug: entry.slug.clone(),
                    loaded_at: entry.loaded_at,
                    expires_at: entry.expires_at,
                    pinned: entry.pinned,
                    in_use: entry.in_use,
                    bytes: catalog
                        .data
                        .iter()
                        .find(|e| e.slug == entry.slug)
                        .map(|e| e.bytes),
                })
                .collect();
            let bytes_reserved = resident
                .iter()
                .filter_map(|r| r.bytes)
                .reduce(u64::wrapping_add);
            let (llm_max_loaded, llm_ttl_secs) = swap.capacity();
            let capacity = MemoryCapacity {
                llm_max_loaded: u32::try_from(llm_max_loaded).unwrap_or(u32::MAX),
                llm_ttl_secs,
            };
            (resident, capacity, bytes_reserved)
        }
        None => (
            Vec::new(),
            MemoryCapacity {
                llm_max_loaded: 0,
                llm_ttl_secs: 0,
            },
            None,
        ),
    };
    Json(MemoryInfo {
        resident,
        capacity,
        gpu: MemoryGpu {
            slot_busy: honkbox_residency::global().is_busy(),
        },
        bytes_reserved,
    })
    .into_response()
}

/// `GET /running`: llama-swap-compatible warm-set envelope (REQ-5). The
/// envelope key `running` and the entry key `model` are load-bearing -
/// loop-lore's `readRunningModels` reads only `running[].model` and
/// classifies any other envelope as malformed (a probe failure that leaves
/// the LLM resident through a `ComfyUI` render). Extra fields are additive.
#[utoipa::path(
    get,
    path = "/running",
    responses((status = 200, description = "Running (resident) models", body = RunningList))
)]
async fn get_running(State(config): State<AppConfig>) -> Response {
    let now_epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let running: Vec<RunningModel> = match &config.swap {
        Some(swap) => swap
            .residency()
            .into_iter()
            .map(|entry| RunningModel {
                model: entry.slug,
                state: "running".to_owned(),
                ttl_secs: entry
                    .expires_at
                    .map(|expires| expires.saturating_sub(now_epoch)),
                pinned: entry.pinned,
            })
            .collect(),
        None => Vec::new(),
    };
    Json(RunningList { running }).into_response()
}

/// `POST /api/models/unload`: llama-swap-compatible bulk unload (REQ-5) -
/// evict every resident non-pinned entry. Pinned residents are skipped,
/// never refused (the bulk form is the caller's "make room" move).
#[utoipa::path(
    post,
    path = "/api/models/unload",
    responses((status = 200, description = "Non-pinned residents evicted", body = UnloadAllResult))
)]
async fn post_unload_all(State(config): State<AppConfig>) -> Response {
    let evicted = match &config.swap {
        Some(swap) => swap.evict_all_unpinned(crate::metrics::UnloadReason::Explicit),
        None => 0,
    };
    if evicted > 0 {
        tracing::info!(count = evicted, "bulk explicit unload");
    }
    Json(UnloadAllResult {
        unloaded: u32::try_from(evicted).unwrap_or(u32::MAX),
        reason: "explicit".to_owned(),
    })
    .into_response()
}

/// `POST /api/models/unload/{model}`: llama-swap-compatible single unload
/// (REQ-5). Accepts an operator alias or registry slug. Divergence from
/// llama-swap, deliberate: an unknown id is a 404 whose body contains the
/// literal `model not found` (loop-lore's classifier substring-matches it),
/// but a known id with nothing resident is a 200 `{"unloaded":false}` -
/// any 200 is an unconditional continue-case on the client, so the common
/// already-gone race needs no string matching at all. Pinned → 409.
#[utoipa::path(
    post,
    path = "/api/models/unload/{model}",
    params(("model" = String, Path, description = "Operator alias or registry slug")),
    responses(
        (status = 200, description = "Unloaded or not resident", body = UnloadResult),
        (status = 404, description = "Unknown model (body contains `model not found`)"),
        (status = 409, description = "Model is pinned; explicit refusal")
    )
)]
async fn post_unload_one(
    State(config): State<AppConfig>,
    AxumPath(model): AxumPath<String>,
) -> Response {
    let catalog = config.catalog_snapshot();
    let Some(slug) = control_slug(&model, &catalog) else {
        // Literal substring is load-bearing for loop-lore's classifier.
        return (StatusCode::NOT_FOUND, format!("model not found: {model}")).into_response();
    };
    let slug = slug.to_owned();
    let outcome = match &config.swap {
        Some(swap) => swap.evict_by_slug(&slug, crate::metrics::UnloadReason::Explicit),
        None => crate::swap::EvictOutcome::NotLoaded,
    };
    unload_response(outcome, slug)
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

/// Registry re-scan + shared catalog snapshot swap: the body of
/// `POST /v1/models/refresh`, also driven by the download manager's
/// completion signal (`rescan_on_downloads`). An operator-binding conflict
/// in the fresh scan leaves the previous snapshot in place and fails the
/// request — never a silently wrong alias.
async fn rescan_catalog(config: &AppConfig) -> Result<ModelCatalog, String> {
    let roots = config.roots.clone();
    let found = tokio::task::spawn_blocking(move || honkbox_registry::scan_roots(&roots))
        .await
        .map_err(|err| format!("refresh scan failed: {err}"))?;
    let catalog = refresh_catalog(&found, config.data_dir.as_deref(), &config.model_entries)?;
    *config
        .catalog
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::new(catalog.clone());
    Ok(catalog)
}

/// Rescan the registry whenever the download manager commits a file, so a
/// completed download appears in `GET /v1/models` under its normal slug
/// without a manual refresh. Detached loop; failures log and keep waiting.
/// (`allow(dead_code)` only silences the route-test binaries that include
/// `routes.rs` without driving this loop.)
#[cfg_attr(test, allow(dead_code))]
pub(crate) async fn rescan_on_downloads(config: AppConfig, signal: Arc<Notify>) {
    loop {
        signal.notified().await;
        if let Err(err) = rescan_catalog(&config).await {
            tracing::warn!(%err, "post-download rescan failed");
        }
    }
}

/// Registry scan hits → `/v1/models` body, with SQLite cache sync.
///
/// The file walk stays the **source of truth**; `HONKBOX_REGISTRY_DB` (off
/// by default via `"off"`/empty, otherwise a cache path) only adds an
/// incremental cache: a successful non-empty scan upserts its entries plus
/// the operator bindings and prunes rows whose files vanished; when a scan
/// comes back **empty** (roots unavailable) the last-known cached catalog
/// is served as an offline fallback — with the cached aliases, so a restart
/// against empty roots still resolves operator aliases. Every cache failure
/// is logged and non-fatal — the response is always the plain scan result
/// when caching is disabled or the cache is unusable. Operator-binding
/// conflicts (duplicate alias across files, `match` hitting several files)
/// are `Err` — a hard failure at startup, a 500 on refresh.
pub fn refresh_catalog(
    found: &[honkbox_registry::FoundModel],
    data_dir: Option<&Path>,
    model_entries: &[crate::config::ModelEntryConfig],
) -> Result<ModelCatalog, String> {
    let db_default = data_dir.map(|base| base.join("registry.sqlite3"));
    if found.is_empty() {
        // Offline fallback: roots unavailable → serve the cached catalog.
        match honkbox_registry::Db::open_from_env(db_default.as_deref()) {
            Ok(Some(db)) => match db.load_entries() {
                Ok(cached) if !cached.is_empty() => {
                    let bindings = match db.load_bindings() {
                        Ok((aliases, overrides)) => bindings_from_cache(&aliases, &overrides),
                        Err(err) => {
                            tracing::warn!(%err, "registry cache bindings read failed");
                            ModelBindings::default()
                        }
                    };
                    tracing::warn!(
                        count = cached.len(),
                        "scan found no models; serving cached registry catalog"
                    );
                    return Ok(catalog_from_entries(&cached, &bindings));
                }
                Ok(_) => {}
                Err(err) => tracing::warn!(%err, "registry cache read failed"),
            },
            Ok(None) => {}
            Err(err) => tracing::warn!(%err, "registry cache open failed"),
        }
        return Ok(build_catalog(found, &ModelBindings::default()));
    }
    // Bind the operator entries against the fresh hits; a conflict fails
    // loudly (startup: exit; refresh: 500) instead of binding a wrong file.
    let bindings = bind_entries(model_entries, found)?;
    // Cache sync: upsert the fresh scan + bindings, prune entries whose
    // files vanished. Best-effort; a slow/broken cache never fails the
    // request. Entries are built once and reused for the response (header
    // probes are file IO).
    let entries: Vec<honkbox_registry::CatalogEntry> = found
        .iter()
        .map(honkbox_registry::CatalogEntry::from)
        .collect();
    match honkbox_registry::Db::open_from_env(db_default.as_deref()) {
        Ok(Some(db)) => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            if let Err(err) = db.upsert_entries(&entries, now) {
                tracing::warn!(%err, "registry cache upsert failed");
            }
            if let Err(err) = db.sync_bindings(&bindings.alias_rows(), &bindings.override_rows()) {
                tracing::warn!(%err, "registry cache binding sync failed");
            }
            if let Err(err) = db.prune_missing_roots() {
                tracing::warn!(%err, "registry cache prune failed");
            }
        }
        Ok(None) => {}
        Err(err) => tracing::warn!(%err, "registry cache open failed"),
    }
    Ok(catalog_from_entries(&entries, &bindings))
}

/// Operator bindings bound to scan hits: per-slug declared aliases (in
/// declaration order; the first is the served `id`) and capability
/// overrides. Built by [`bind_entries`]; cached rows rebuild it for the
/// offline fallback.
#[derive(Debug, Default)]
pub struct ModelBindings {
    /// slug → declared aliases (declaration order preserved).
    aliases: HashMap<String, Vec<String>>,
    /// slug → operator capability overrides.
    capabilities: HashMap<String, crate::config::ModelCapabilitiesOverride>,
}

impl ModelBindings {
    /// The alias served as `id` for `slug` (first declared), if any.
    fn display_alias(&self, slug: &str) -> Option<&str> {
        self.aliases.get(slug)?.first().map(String::as_str)
    }

    /// Cache rows: alias→slug pairs, grouped per slug in declaration order
    /// so the loaded catalog's `id` (first alias) is stable across restarts.
    fn alias_rows(&self) -> Vec<honkbox_registry::AliasRow> {
        let mut slugs: Vec<&String> = self.aliases.keys().collect();
        slugs.sort_unstable();
        slugs
            .into_iter()
            .flat_map(|slug| {
                self.aliases[slug]
                    .iter()
                    .map(|alias| honkbox_registry::AliasRow {
                        alias: alias.clone(),
                        slug: slug.clone(),
                    })
            })
            .collect()
    }

    /// Cache rows: per-slug capability overrides (slug-sorted; the JSON
    /// column carries the declaration verbatim).
    fn override_rows(&self) -> Vec<honkbox_registry::EntryOverride> {
        let mut rows: Vec<(String, String)> = self
            .capabilities
            .iter()
            .map(|(slug, caps)| (slug.clone(), caps.cache_json()))
            .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        rows.into_iter()
            .map(|(slug, json)| honkbox_registry::EntryOverride {
                slug,
                capabilities_json: json,
            })
            .collect()
    }
}

/// Rebuild [`ModelBindings`] from the registry cache (offline fallback).
/// Rows arrive slug-grouped in declaration order (`sync_bindings`), so the
/// first alias per slug is again the served `id`. Orphan rows were dropped
/// by `load_bindings`.
fn bindings_from_cache(
    aliases: &[honkbox_registry::AliasRow],
    overrides: &[honkbox_registry::EntryOverride],
) -> ModelBindings {
    let mut bindings = ModelBindings::default();
    for row in aliases {
        bindings
            .aliases
            .entry(row.slug.clone())
            .or_default()
            .push(row.alias.clone());
    }
    for row in overrides {
        bindings.capabilities.insert(
            row.slug.clone(),
            crate::config::ModelCapabilitiesOverride::from_cache_json(&row.capabilities_json),
        );
    }
    bindings
}

/// Bind operator `[[models.entry]]` declarations to scan hits.
///
/// Targeting: `slug` binds exactly that registry slug; `match` binds every
/// hit whose weight file **name** (final path component) equals the value.
/// Rules, all **hard errors** naming the offending files:
///
/// - two entries binding the same file (a config-level duplicate, or a
///   `match` that catches another entry's `slug` target);
/// - an alias-bearing `match` hitting several files (the alias would be
///   ambiguous — capabilities-only entries apply to every hit instead);
/// - the same alias declared for two different files, or an alias colliding
///   with a *different* model's registry slug.
///
/// Declarations whose target is absent from this scan bind nothing (the
/// file may appear later); they resurface on the next refresh.
///
/// # Errors
/// Any of the conflict rules above; the message names both paths.
pub fn bind_entries(
    entries: &[crate::config::ModelEntryConfig],
    found: &[honkbox_registry::FoundModel],
) -> Result<ModelBindings, String> {
    let mut bindings = ModelBindings::default();
    // slug → descriptor of the entry that bound it (for duplicate reports).
    let mut bound_by: HashMap<&str, String> = HashMap::new();
    for entry in entries {
        let descriptor = entry_descriptor(entry);
        let targets = targets_of(entry, found);
        if !entry.aliases.is_empty() && targets.len() > 1 {
            return Err(format!(
                "[[models.entry]] {descriptor}: match binds {} files; aliases must name one: {}",
                targets.len(),
                paths_of(&targets)
            ));
        }
        for target in targets {
            if let Some(previous) = bound_by.get(target.slug.as_str()) {
                return Err(format!(
                    "two [[models.entry]] entries bind {}: {} and {descriptor}",
                    target.path.display(),
                    previous
                ));
            }
            bound_by.insert(target.slug.as_str(), descriptor.clone());
            bind_target(&mut bindings, entry, target, found)?;
        }
    }
    Ok(bindings)
}

/// Scan hits one entry targets: the exact `slug`, plus every hit whose
/// weight file name equals `match`.
fn targets_of<'a>(
    entry: &crate::config::ModelEntryConfig,
    found: &'a [honkbox_registry::FoundModel],
) -> Vec<&'a honkbox_registry::FoundModel> {
    found
        .iter()
        .filter(|m| {
            entry.slug.as_deref().is_some_and(|slug| slug == m.slug)
                || entry
                    .file_match
                    .as_deref()
                    .is_some_and(|file_match| file_name(&m.path) == Some(file_match))
        })
        .collect()
}

/// Bind one scan hit: capability overrides (unless nothing is declared) and
/// the entry's aliases, each checked against aliases of other slugs and
/// against other models' registry slugs.
fn bind_target(
    bindings: &mut ModelBindings,
    entry: &crate::config::ModelEntryConfig,
    target: &honkbox_registry::FoundModel,
    found: &[honkbox_registry::FoundModel],
) -> Result<(), String> {
    if entry.capabilities != crate::config::ModelCapabilitiesOverride::default() {
        bindings
            .capabilities
            .insert(target.slug.clone(), entry.capabilities);
    }
    for alias in &entry.aliases {
        let conflicting = bindings
            .aliases
            .iter()
            .find(|(slug, list)| slug.as_str() != target.slug && list.contains(alias))
            .map(|(slug, _)| slug.clone());
        if let Some(other_slug) = conflicting {
            return Err(format!(
                "alias {alias:?} is declared twice: for {other_slug:?} and {} ({})",
                target.slug,
                target.path.display()
            ));
        }
        if let Some(other) = found
            .iter()
            .find(|m| m.slug == *alias && m.slug != target.slug)
        {
            return Err(format!(
                "alias {alias:?} collides with the registry slug of another model ({})",
                other.path.display()
            ));
        }
        bindings
            .aliases
            .entry(target.slug.clone())
            .or_default()
            .push(alias.clone());
    }
    Ok(())
}

/// Stable descriptor for error messages: what the entry targets.
fn entry_descriptor(entry: &crate::config::ModelEntryConfig) -> String {
    if let Some(slug) = &entry.slug {
        format!("slug = {slug:?}")
    } else {
        format!("match = {:?}", entry.file_match.as_deref().unwrap_or(""))
    }
}

/// Final path component (the weight file name `match` compares).
fn file_name(path: &Path) -> Option<&str> {
    path.file_name().and_then(|name| name.to_str())
}

/// Comma-joined display paths for conflict messages.
fn paths_of(targets: &[&honkbox_registry::FoundModel]) -> String {
    targets
        .iter()
        .map(|m| m.path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Registry scan hits → `/v1/models` body. Also feeds `models show --json`
/// so the CLI output and the endpoint agree (P0 exit of the parity plan).
#[must_use]
pub fn build_catalog(
    found: &[honkbox_registry::FoundModel],
    bindings: &ModelBindings,
) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: found
            .iter()
            .map(|m| catalog_entry(m.into(), bindings))
            .collect(),
    }
}

/// Registry catalog entries (e.g. the SQLite cache) → `/v1/models` body.
fn catalog_from_entries(
    entries: &[honkbox_registry::CatalogEntry],
    bindings: &ModelBindings,
) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: entries
            .iter()
            .map(|e| catalog_entry(e.clone(), bindings))
            .collect(),
    }
}

/// Registry catalog entry → response DTO (the only registry→wire mapping).
/// The operator alias (when bound) becomes `id`; the slug always rides its
/// own field. Capability defaults derive from the (possibly reclassified)
/// kind, then operator declarations win; loop-lore's flat fields mirror the
/// nested `capabilities` object.
fn catalog_entry(
    entry: honkbox_registry::CatalogEntry,
    bindings: &ModelBindings,
) -> ModelCatalogEntry {
    let overrides = bindings.capabilities.get(&entry.slug).copied();
    let kind = if overrides
        .is_some_and(crate::config::ModelCapabilitiesOverride::reclassifies_embedding)
    {
        ModelKind::Embedding
    } else {
        match entry.kind {
            honkbox_registry::ModelKind::Llm => ModelKind::Llm,
            honkbox_registry::ModelKind::Sd => ModelKind::Sd,
            honkbox_registry::ModelKind::Lora => ModelKind::Lora,
            honkbox_registry::ModelKind::Embedding => ModelKind::Embedding,
            honkbox_registry::ModelKind::Vae => ModelKind::Vae,
            honkbox_registry::ModelKind::Stt => ModelKind::Stt,
            honkbox_registry::ModelKind::Tts => ModelKind::Tts,
        }
    };
    let mut capabilities = ModelCapabilities::for_kind(kind);
    if let Some(overrides) = overrides {
        if let Some(chat) = overrides.chat {
            capabilities.chat = chat;
        }
        if let Some(embeddings) = overrides.embeddings {
            capabilities.embeddings = embeddings;
        }
        if let Some(tool_calling) = overrides.tool_calling {
            capabilities.tool_calling = tool_calling;
        }
        if let Some(vision) = overrides.vision {
            capabilities.vision = vision;
        }
        capabilities.completion = capabilities.chat;
    }
    let context_length = match entry.metadata.as_ref() {
        Some(honkbox_registry::ModelFileMetadata::Gguf(meta)) => meta.context_length,
        _ => None,
    };
    ModelCatalogEntry {
        id: bindings
            .display_alias(&entry.slug)
            .map_or_else(|| entry.slug.clone(), str::to_owned),
        slug: entry.slug,
        kind,
        owned_by: "local".to_owned(),
        context_length,
        // honkbox has no per-model output ceiling; the field exists for the
        // loop-lore flat contract and stays `null`.
        max_output: None,
        // No honkbox model declares reasoning/thinking support today.
        thinking: false,
        tool_calling: capabilities.tool_calling,
        modalities: if capabilities.vision {
            vec!["text".to_owned(), "vision".to_owned()]
        } else {
            vec!["text".to_owned()]
        },
        capabilities,
        category: entry.category,
        path: entry.path.display().to_string(),
        bytes: entry.bytes,
        metadata: entry.metadata.map(catalog_metadata),
    }
}

/// Registry header metadata → response DTO (part of the registry→wire mapping).
fn catalog_metadata(metadata: honkbox_registry::ModelFileMetadata) -> ModelMetadata {
    match metadata {
        honkbox_registry::ModelFileMetadata::Gguf(meta) => ModelMetadata::Gguf(GgufMetadata {
            version: meta.version,
            tensor_count: meta.tensor_count,
            name: meta.name,
            architecture: meta.architecture,
        }),
        honkbox_registry::ModelFileMetadata::Safetensors(meta) => {
            ModelMetadata::Safetensors(SafetensorsMetadata {
                tensor_count: meta.tensor_count,
                dtypes: meta.dtypes,
            })
        }
    }
}

/// Map an enqueue error onto the HTTP contract: caller payload problems are
/// 400, server-side root/IO problems are 500 (same split as
/// `swap_rejection`'s unknown-vs-failed).
fn enqueue_rejection(err: &crate::downloads::EnqueueError) -> (StatusCode, String) {
    let message = err.to_string();
    let status = match err {
        crate::downloads::EnqueueError::NoModelsRoot
        | crate::downloads::EnqueueError::CreateDir(_) => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::BAD_REQUEST,
    };
    (status, message)
}

/// `POST /v1/downloads`: start one managed download into the configured
/// models root. `{repo, file, kind}` derives the `HuggingFace` resolve URL
/// (and enforces the LFS sha256 when the repo API publishes one); `url` is
/// the direct-source alternative. A second pull of the same target file
/// joins the in-flight download (same id); a target that already exists
/// resolves immediately as `done` + `already_present`.
#[utoipa::path(
    post,
    path = "/v1/downloads",
    request_body = DownloadRequest,
    responses(
        (status = 202, description = "Download accepted", body = DownloadAccepted),
        (status = 400, description = "Missing/ambiguous source, bad repo id, or unusable file name"),
        (status = 500, description = "No managed models root or target directory create failed")
    )
)]
async fn post_download(
    State(config): State<AppConfig>,
    Json(request): Json<DownloadRequest>,
) -> Response {
    match config.downloads.enqueue(request).await {
        Ok(accepted) => (StatusCode::ACCEPTED, Json(accepted)).into_response(),
        Err(err) => enqueue_rejection(&err).into_response(),
    }
}

/// `GET /v1/downloads`: every known download with its progress (poll
/// contract; no SSE).
#[utoipa::path(
    get,
    path = "/v1/downloads",
    responses(
        (status = 200, description = "Known downloads", body = DownloadList),
    )
)]
async fn list_downloads(State(config): State<AppConfig>) -> Json<DownloadList> {
    Json(DownloadList {
        downloads: config.downloads.states(),
    })
}

/// `GET /v1/downloads/{id}`: one download's progress; 404 for an unknown id.
#[utoipa::path(
    get,
    path = "/v1/downloads/{id}",
    params(("id" = String, Path, description = "Download id from the POST acceptance body")),
    responses(
        (status = 200, description = "Download state", body = DownloadState),
        (status = 404, description = "Unknown download id")
    )
)]
async fn get_download(
    State(config): State<AppConfig>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match config.downloads.state(&id) {
        Some(state) => Json(state).into_response(),
        None => (StatusCode::NOT_FOUND, format!("unknown download id {id:?}")).into_response(),
    }
}

/// `DELETE /v1/downloads/{id}`: cancel a download (204). The task stops at
/// the next chunk boundary, deletes the `.part` + meta sidecar, and lands
/// in `cancelled`; a queued download never starts. 409 once terminal.
#[utoipa::path(
    delete,
    path = "/v1/downloads/{id}",
    params(("id" = String, Path, description = "Download id from the POST acceptance body")),
    responses(
        (status = 204, description = "Cancellation triggered"),
        (status = 404, description = "Unknown download id"),
        (status = 409, description = "Download already finished")
    )
)]
async fn delete_download(
    State(config): State<AppConfig>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match config.downloads.cancel(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(crate::downloads::CancelError::Unknown(_)) => {
            (StatusCode::NOT_FOUND, format!("unknown download id {id:?}")).into_response()
        }
        Err(crate::downloads::CancelError::Terminal) => {
            (StatusCode::CONFLICT, "download already finished".to_owned()).into_response()
        }
    }
}

/// Per-request model targeting (P0 of the llama-swap parity plan, extended
/// for REQ-2 aliases).
///
/// The request `model` field is validated and **resolved to its owning
/// registry slug**: a non-empty value may be an operator alias (a served
/// `id`), any catalog slug, or the startup-configured slug for the
/// modality. Nothing is loaded or re-routed here — [`resolve_model_slug`]'s
/// slug then drives the swap manager (or the static startup pipeline).
/// Empty/missing keeps the startup-configured model, exactly as before this
/// check existed. With no catalog entries and no configured slug (stub
/// backends, empty scan) any value passes, so zero-config stub servers keep
/// working. Unknown values are the route's 400/404, unchanged.
fn validate_model(
    model: Option<&str>,
    configured: Option<&str>,
    catalog: &ModelCatalog,
) -> Result<(), String> {
    let Some(model) = model.map(str::trim).filter(|m| !m.is_empty()) else {
        return Ok(());
    };
    if configured.is_some_and(|c| c == model) || resolve_model_slug(model, catalog).is_some() {
        return Ok(());
    }
    Err(format!(
        "unknown model {model:?}: not a catalog slug or alias (see GET /v1/models) and not the configured model"
    ))
}

/// Owning registry slug for a request `model` value: the entry whose alias
/// (served `id`) or registry `slug` matches. Aliases are bound by operator
/// config ([`bind_entries`]); the raw slug always keeps working.
fn resolve_model_slug<'a>(model: &str, catalog: &'a ModelCatalog) -> Option<&'a str> {
    resolve_model_entry(model, catalog).map(|e| e.slug.as_str())
}

/// Embedded `web-dist/` asset when present, dev-disk file in `--dev`,
/// otherwise the static fallback page.
async fn fallback(State(config): State<AppConfig>, uri: Uri) -> Response {
    let raw = uri.path().trim_start_matches('/');
    let file = if raw.is_empty() { "index.html" } else { raw };
    if config.dev {
        if let Some(response) = serve_from_disk(&config.web_dist_dir, file).await {
            return response;
        }
    } else if let Some(asset) = EmbeddedUi::get(file) {
        return asset_response(file, asset.data.into_owned());
    }
    Html(include_str!("../fallback.html")).into_response()
}

/// Serve one file from the dev web-dist directory; `None` = miss or escape attempt.
async fn serve_from_disk(dir: &Path, file: &str) -> Option<Response> {
    if file.contains("..") {
        return None;
    }
    let bytes = tokio::fs::read(dir.join(file)).await.ok()?;
    Some(asset_response(file, bytes))
}

/// Raw bytes with a content type guessed from the extension (no extra deps).
fn asset_response(file: &str, bytes: Vec<u8>) -> Response {
    let mime = match file.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        _ => "application/octet-stream",
    };
    ([(header::CONTENT_TYPE, mime)], bytes).into_response()
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
fn worker_message(message: &ChatMessage) -> WorkerChatMessage {
    let role = match message.role {
        ChatRole::User => Role::User,
        ChatRole::Assistant => Role::Assistant,
        ChatRole::System => Role::System,
        ChatRole::Tool => Role::Tool,
    };
    WorkerChatMessage {
        role,
        content: message.content.text.clone().unwrap_or_default(),
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
    }
}

/// Worker request skeleton shared by both routes: constraint fields absent
/// (`REQ-3` fields ride only on `/v1/chat/completions`), default sampler.
fn worker_request_base(
    slug: String,
    messages: Vec<WorkerChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> WorkerChatRequest {
    WorkerChatRequest {
        model: slug,
        messages,
        max_tokens,
        temperature,
        seed: None,
        grammar: None,
        json_schema: None,
        json_object: false,
        tools_json: None,
        tool_choice_none: false,
        sampler: SamplerParams::default(),
    }
}

/// Try to reserve the process-wide GPU slot; a `Busy` rejection counts the
/// `honkbox_gpu_slot_busy_rejections_total` metric and maps to a 429.
fn reserve_gpu_slot() -> Result<honkbox_residency::Lease<'static>, (StatusCode, String)> {
    honkbox_residency::global().try_reserve().map_err(|err| {
        crate::metrics::note_gpu_slot_busy(crate::metrics::BusyReason::NoWait);
        (StatusCode::TOO_MANY_REQUESTS, err.to_string())
    })
}

/// REQ-8: the named vision refusal, shared by `/v1/chat/completions` and
/// the native chat routes (honkbox ships no mmproj/vision backend).
const VISION_UNSUPPORTED: &str = "vision not supported by backend: image_url content parts require an mmproj vision backend honkbox does not ship";

/// Request gates before any pipeline work (`/v1/chat/completions`): REQ-4
/// — an embeddings-capable model is never silently run through the chat
/// pipeline (the reply would be garbage tokens dressed up as a
/// completion); REQ-8 — an `image_url` content part is an explicit
/// vision refusal, never a silently dropped (or silently flattened)
/// image. `Some(response)` is the ready-made rejection.
fn completions_request_gates(
    entry: &ModelCatalogEntry,
    model: &str,
    messages: &[ChatMessage],
) -> Option<Response> {
    embeddings_model_gate(entry, model).or_else(|| {
        messages
            .iter()
            .any(|message| message.content.has_image)
            .then(|| {
                openai_error_response(
                    StatusCode::BAD_REQUEST,
                    VISION_UNSUPPORTED.to_owned(),
                    Some("vision_unsupported"),
                )
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
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    let catalog = config.catalog_snapshot();
    validate_model(Some(&req.model), config.llm_model.as_deref(), &catalog)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    // Aliases resolve to their owning registry slug; the swap manager and
    // the worker both key on the slug (an unresolved value is the
    // startup-configured model, same as before aliases existed).
    let slug =
        resolve_model_slug(&req.model, &catalog).map_or_else(|| req.model.clone(), str::to_owned);
    // REQ-8 parity with `/v1/chat/completions`: an `image_url` content
    // part is an explicit refusal (plain-text 400 on the native route),
    // never a silently flattened image.
    if req.messages.iter().any(|message| message.content.has_image) {
        return Err((StatusCode::BAD_REQUEST, VISION_UNSUPPORTED.to_owned()));
    }
    let pipe = chat_pipeline(
        &config,
        &slug,
        clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds),
    )
    .await?;
    let worker_req = worker_request_base(
        slug,
        req.messages.iter().map(worker_message).collect(),
        req.max_tokens,
        req.temperature,
    );
    let reply = tokio::task::spawn_blocking(move || match reserve_gpu_slot() {
        Ok(lease) => {
            let _lease = lease;
            pipe.complete(&worker_req).map_err(|err| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("chat failed: {err}"),
                )
            })
        }
        Err(rejection) => Err(rejection),
    })
    .await
    .map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("chat task failed: {err}"),
        )
    })??;
    Ok(Json(ChatResponse {
        backend_id: reply.backend_id,
        text: reply.text,
        tokens: reply.tokens,
        millis: reply.millis,
    }))
}

/// Drop the GPU residency lease immediately (the disconnect path): the slot
/// frees at the token boundary while `abort()` stops the FFI loop one token
/// later.
fn free_gpu_lease(slot: &Mutex<Option<honkbox_residency::Lease<'static>>>) {
    *slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
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
) -> Result<
    Sse<CancelOnDrop<UnboundedReceiverStream<Result<Event, Infallible>>>>,
    (StatusCode, String),
> {
    let catalog = config.catalog_snapshot();
    validate_model(Some(&req.model), config.llm_model.as_deref(), &catalog)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    // Alias → owning slug (same contract as `post_chat`).
    let slug =
        resolve_model_slug(&req.model, &catalog).map_or_else(|| req.model.clone(), str::to_owned);
    // REQ-8 parity with `/v1/chat/completions`: an `image_url` content
    // part is an explicit refusal (plain-text 400 on the native route),
    // never a silently flattened image.
    if req.messages.iter().any(|message| message.content.has_image) {
        return Err((StatusCode::BAD_REQUEST, VISION_UNSUPPORTED.to_owned()));
    }
    let pipe = chat_pipeline(
        &config,
        &slug,
        clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds),
    )
    .await?;
    let worker_req = worker_request_base(
        slug,
        req.messages.iter().map(worker_message).collect(),
        req.max_tokens,
        req.temperature,
    );
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let cancelled = Arc::new(AtomicBool::new(false));
    // REQ-6 (Option B): the busy condition must be an HTTP 429, never a 200
    // with an error frame - so the slot is reserved HERE, before the handler
    // returns the SSE body. A busy slot short-circuits above as a real 429.
    let lease = reserve_gpu_slot()?;
    // The lease sits in a slot the token sink can drain: once the client is
    // gone, the next token boundary frees the slot right away. The handler
    // returns the SSE body without awaiting the worker, so tokens stream as
    // they are produced.
    let lease_slot = Arc::new(Mutex::new(None::<honkbox_residency::Lease<'static>>));
    // Separate handle for the SSE body: the worker closure consumes its own
    // clone, the drop path calls `abort` through this one.
    let abort_pipe = Arc::clone(&pipe);
    let worker_cancelled = cancelled.clone();
    let worker = tokio::task::spawn_blocking(move || {
        *lease_slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(lease);
        let result = pipe.complete_stream(&worker_req, &mut |token: &str| {
            if worker_cancelled.load(Ordering::Relaxed) {
                // Also abort the FFI loop itself, not just delivery.
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
    Ok(Sse::new(CancelOnDrop {
        inner: UnboundedReceiverStream::new(rx),
        cancelled,
        pipe: abort_pipe,
        _worker: worker,
    }))
}

/// OpenAI-shaped error body → JSON response (`/v1/chat/completions` only;
/// the native route keeps its plain-text errors).
fn openai_error_response(status: StatusCode, message: String, code: Option<&str>) -> Response {
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
    /// behaves as `auto`).
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
    let reply = tokio::task::spawn_blocking(move || match reserve_gpu_slot() {
        Ok(lease) => {
            let _lease = lease;
            pipe.embed(&inputs)
                .map(|embeddings| (embeddings, pipe.prompt_tokens(&inputs)))
                .map_err(|err| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("embed failed: {err}"),
                    )
                })
        }
        Err((_, message)) => Err((StatusCode::TOO_MANY_REQUESTS, message)),
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
    .into_response()
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

/// Validated image request: parsed size, clamped count. The only rejection
/// paths for `POST /v1/images/generations` before the GPU slot; `Err` is
/// (message, error code) for a 400 `OpenAI` envelope.
fn validate_images_request(
    req: &ImagesGenerationRequest,
) -> Result<(u32, u32, u32), (String, &'static str)> {
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
    Ok((width, height, n.clamp(1, 4)))
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

/// Fixed generation recipe for the HTTP surface: the same defaults as
/// `honkbox generate` (seed 42 base, 20 steps, euler/normal), recorded in
/// the bundle sidecars. Per-image seed = base + index.
const IMAGE_SEED_BASE: u64 = 42;
/// Fixed diffusion steps for the HTTP surface (CLI default).
const IMAGE_STEPS: u32 = 20;

/// `POST /v1/images/generations` (REQ-10): the `OpenAI` image-generation
/// shape over the startup-configured [`SdPipeline`]. SD pipelines are
/// startup-static (no swap manager), so a request `model` is validated and
/// recorded (bundle sidecar) but the pipeline stays the one built at
/// startup — with the stub backend any slug is honest; with `sd-cpp` the
/// checkpoint is the `--sd-model` one.
///
/// Each image in the request is generated under the GPU lease
/// (`spawn_blocking`, like every other GPU route) and written to the
/// existing job bundle store (`<data_dir>/jobs/`), so CLI GC and
/// recordkeeping cover HTTP generations too. Errors are `OpenAI`
/// envelopes: 400 bad `size`/`response_format`, 404 unknown model, 429
/// busy, 500 backend failure. `n` clamps to `[1, 4]` (warn, never reject).
#[utoipa::path(
    post,
    path = "/v1/images/generations",
    request_body = ImagesGenerationRequest,
    responses(
        (status = 200, description = "Generated images (base64 PNG)", body = ImagesGenerationResponse),
        (status = 400, description = "Bad size or response_format", body = OpenAiError),
        (status = 404, description = "Unknown model", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Generation failed", body = OpenAiError)
    )
)]
async fn post_images_generations(
    State(config): State<AppConfig>,
    Json(req): Json<ImagesGenerationRequest>,
) -> Response {
    let (width, height, n) = match validate_images_request(&req) {
        Ok(valid) => valid,
        Err((message, code)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
    };
    let catalog = config.catalog_snapshot();
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
    let lease = match reserve_gpu_slot() {
        Ok(lease) => lease,
        Err((status, message)) => return openai_error_response(status, message, None),
    };
    let mut images = Vec::with_capacity(n as usize);
    let backend_id = pipeline.backend_id();
    for i in 0..u64::from(n) {
        let pipeline = Arc::clone(&pipeline);
        let gen_prompt = prompt.clone();
        let gen_slug = slug.clone();
        let outcome =
            tokio::task::spawn_blocking(move || -> Result<Vec<u8>, honkbox_sd_core::SdError> {
                let img = pipeline.generate(&honkbox_sd_core::Txt2ImgRequest {
                    // Per-image seeds keep n>1 outputs distinct while staying
                    // deterministic per request.
                    seed: IMAGE_SEED_BASE.wrapping_add(i),
                    steps: IMAGE_STEPS,
                    prompt: gen_prompt,
                    width,
                    height,
                    model_slug: gen_slug,
                    clip: None,
                    vae: None,
                })?;
                honkbox_sd_core::encode_image(img, honkbox_sd_core::OutputFormat::Png)
            })
            .await;
        let bytes = match outcome {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(err)) => {
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    err.to_string(),
                    Some("generation_failed"),
                );
            }
            Err(err) => {
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("generation task failed: {err}"),
                    Some("generation_failed"),
                );
            }
        };
        write_generation_bundle(
            &config,
            &crate::jobs::BundleInputs {
                model_slug: slug.clone(),
                prompt: prompt.clone(),
                seed: IMAGE_SEED_BASE.wrapping_add(i),
                steps: IMAGE_STEPS,
                width,
                height,
                format: honkbox_sd_core::OutputFormat::Png,
                backend_id,
            },
            &bytes,
            u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        );
        images.push(ImageB64 {
            b64_json: base64::engine::general_purpose::STANDARD.encode(&bytes),
        });
    }
    drop(lease);
    Json(ImagesGenerationResponse {
        created: epoch_secs(),
        data: images,
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
    let worker_req = WorkerChatRequest {
        model: slug.clone(),
        messages: req.messages.iter().map(worker_message).collect(),
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
    if req.stream.is_some_and(|stream| stream) {
        // REQ-6 (Option B): reserve the slot BEFORE returning the SSE body,
        // so a busy slot is a real 429 with the OpenAI envelope + the
        // shared `Retry-After` middleware hint - never a 200 whose only
        // content is an error frame (loop-lore's `data:`-only parser would
        // read that as a clean empty completion).
        let lease = match reserve_gpu_slot() {
            Ok(lease) => lease,
            Err((status, message)) => {
                return openai_error_response(status, message, Some("gpu_slot_busy"));
            }
        };
        completions_stream(
            lease,
            pipe,
            worker_req,
            &served_id,
            id_frame(),
            req.max_tokens,
        )
    } else {
        completions_json(pipe, worker_req, &served_id, req.max_tokens).await
    }
}

/// Non-streaming completion: one blocking worker call inside the GPU lease,
/// mapped onto the exact `OpenAI` JSON shape.
async fn completions_json(
    pipe: Arc<dyn LlmPipeline>,
    worker_req: WorkerChatRequest,
    served_id: &str,
    max_tokens: Option<u32>,
) -> Response {
    let reply = tokio::task::spawn_blocking(move || match reserve_gpu_slot() {
        Ok(lease) => {
            let _lease = lease;
            pipe.complete(&worker_req).map_err(|err| {
                // `REQ-3`: unusable constraints (unparseable grammar, non-
                // convertible schema) are client errors, not server ones.
                let status = match &err {
                    LlmError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                };
                (status, format!("chat failed: {err}"))
            })
        }
        Err((_, message)) => Err((StatusCode::TOO_MANY_REQUESTS, message)),
    })
    .await
    .map_err(|err| format!("chat task failed: {err}"));
    let reply = match reply {
        Ok(Ok(reply)) => reply,
        Ok(Err((StatusCode::TOO_MANY_REQUESTS, message))) => {
            return openai_error_response(
                StatusCode::TOO_MANY_REQUESTS,
                message,
                Some("gpu_slot_busy"),
            );
        }
        Ok(Err((StatusCode::BAD_REQUEST, message))) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                message,
                Some("invalid_request"),
            );
        }
        Ok(Err((_, message))) | Err(message) => {
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
    .into_response()
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
    lease: honkbox_residency::Lease<'static>,
    pipe: Arc<dyn LlmPipeline>,
    worker_req: WorkerChatRequest,
    served_id: &str,
    (id, created): (String, u64),
    max_tokens: Option<u32>,
) -> Response {
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let cancelled = Arc::new(AtomicBool::new(false));
    let lease_slot = Arc::new(Mutex::new(None::<honkbox_residency::Lease<'static>>));
    let abort_pipe = Arc::clone(&pipe);
    let worker_cancelled = cancelled.clone();
    let frame_model = served_id.to_owned();
    // `REQ-3`: with tools on the table, pieces are buffered until the
    // parsed tool calls are known, then framed as `delta.tool_calls`
    // fragments (arguments split across frames; per-index concatenation
    // yields the JSON string). Without tools, content streams
    // token-by-token as before — including `json_object` replies.
    let tool_streaming = worker_req.tools_json.is_some();
    let worker = tokio::task::spawn_blocking(move || {
        // The slot was reserved in the handler (REQ-6); park the lease where
        // the token sink can drain it at the first cancel/disconnect.
        *lease_slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(lease);
        let mut first = true;
        let mut buffered: Vec<String> = Vec::new();
        let result = pipe.complete_stream(&worker_req, &mut |token: &str| {
            if worker_cancelled.load(Ordering::Relaxed) {
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
    Sse::new(CancelOnDrop {
        inner: UnboundedReceiverStream::new(rx),
        cancelled,
        pipe: abort_pipe,
        _worker: worker,
    })
    .into_response()
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
) -> Result<Json<Transcript>, (StatusCode, String)> {
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
    };
    let out = run_transcription(config.stt.clone(), worker_req, sample_rate, pcm).await?;
    Ok(Json(Transcript {
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
    }))
}

/// `POST /v1/synthesize`: JSON in, WAV bytes out (`audio/wav`).
#[utoipa::path(
    post,
    path = "/v1/synthesize",
    request_body = SynthesizeRequest,
    responses(
        (status = 200, description = "WAV audio"),
        (status = 400, description = "Unknown model slug"),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Synthesis failed")
    )
)]
async fn post_synthesize(
    State(config): State<AppConfig>,
    Json(req): Json<SynthesizeRequest>,
) -> Result<Response, (StatusCode, String)> {
    let catalog = config.catalog_snapshot();
    validate_model(Some(&req.model), config.tts_model.as_deref(), &catalog)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    let _ttl = clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds);
    let worker_req = WorkerSynthRequest {
        model: req.model,
        text: req.text,
        voice: req.voice,
        sample_rate: req.sample_rate,
    };
    let audio = run_synthesis(config.tts.clone(), worker_req).await?;
    let wav = encode_wav(audio.sample_rate, &audio.pcm16);
    Ok(([(header::CONTENT_TYPE, "audio/wav")], wav).into_response())
}

/// Shared STT execution (native `/v1/transcribe` and REQ-9
/// `/v1/audio/transcriptions` are the same worker call under two wire
/// shapes): run the pipeline inside the GPU slot on the blocking pool.
async fn run_transcription(
    pipe: Arc<dyn SttPipeline>,
    worker_req: WorkerTranscribeRequest,
    sample_rate: u32,
    pcm: Vec<i16>,
) -> Result<WorkerTranscript, (StatusCode, String)> {
    tokio::task::spawn_blocking(move || match reserve_gpu_slot() {
        Ok(lease) => {
            let _lease = lease;
            pipe.transcribe(&pcm, sample_rate, &worker_req)
                .map_err(|err| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("transcribe failed: {err}"),
                    )
                })
        }
        Err(rejection) => Err(rejection),
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
async fn run_synthesis(
    pipe: Arc<dyn TtsPipeline>,
    worker_req: WorkerSynthRequest,
) -> Result<SynthesizedAudio, (StatusCode, String)> {
    tokio::task::spawn_blocking(move || match reserve_gpu_slot() {
        Ok(lease) => {
            let _lease = lease;
            pipe.synthesize(&worker_req).map_err(|err| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("synthesize failed: {err}"),
                )
            })
        }
        Err(rejection) => Err(rejection),
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
/// `json` (default) or `verbose_json` (adds the worker's timestamped
/// segments). Errors are `OpenAI` envelopes; only a request that is not
/// multipart at all gets axum's transport-level 400 before the handler
/// runs. The native route is unchanged.
#[utoipa::path(
    post,
    path = "/v1/audio/transcriptions",
    request_body(content = String, description = "multipart/form-data: file, model?, language?, response_format?, sample_rate?"),
    responses(
        (status = 200, description = "Transcription", body = TranscriptionText),
        (status = 400, description = "Missing file, bad audio, unknown model, or missing sample_rate", body = OpenAiError),
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
        None | Some("json" | "verbose_json") => {}
        Some(other) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "unsupported response_format {other:?}: only \"json\" and \"verbose_json\""
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
    };
    let out = match run_transcription(config.stt.clone(), worker_req, sample_rate, pcm).await {
        Ok(out) => out,
        Err((StatusCode::TOO_MANY_REQUESTS, message)) => {
            return openai_error_response(
                StatusCode::TOO_MANY_REQUESTS,
                message,
                Some("gpu_slot_busy"),
            );
        }
        Err((code, message)) => return openai_error_response(code, message, None),
    };
    if upload.response_format.as_deref() == Some("verbose_json") {
        Json(TranscriptionVerbose {
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
        .into_response()
    } else {
        Json(TranscriptionText { text: out.text }).into_response()
    }
}

/// Query carried over from the native route: `audio/pcm-s16le` uploads
/// without a `sample_rate` form field.
#[derive(Debug, Deserialize)]
struct AudioSampleRateQuery {
    /// Input sample rate in Hz (pcm-s16le uploads only).
    sample_rate: Option<u32>,
}

/// `POST /v1/audio/speech` (REQ-9): `OpenAI`-shaped TTS. Same worker as
/// `/v1/synthesize`; returns `audio/wav` bytes. `model` optional (server
/// default TTS), `voice` passed through, `response_format` only `"wav"`.
#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    request_body = SpeechRequest,
    responses(
        (status = 200, description = "WAV audio", content_type = "audio/wav"),
        (status = 400, description = "Empty input, unknown model, or unsupported response_format", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Synthesis failed", body = OpenAiError)
    )
)]
async fn post_audio_speech(
    State(config): State<AppConfig>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    match req.response_format.as_deref() {
        None | Some("wav") => {}
        Some(other) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("unsupported response_format {other:?}: only \"wav\" is produced"),
                None,
            );
        }
    }
    if req.input.trim().is_empty() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "input is required: the text to synthesize".to_owned(),
            None,
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
    let audio = match run_synthesis(config.tts.clone(), worker_req).await {
        Ok(audio) => audio,
        Err((StatusCode::TOO_MANY_REQUESTS, message)) => {
            return openai_error_response(
                StatusCode::TOO_MANY_REQUESTS,
                message,
                Some("gpu_slot_busy"),
            );
        }
        Err((code, message)) => return openai_error_response(code, message, None),
    };
    let wav = encode_wav(audio.sample_rate, &audio.pcm16);
    ([(header::CONTENT_TYPE, "audio/wav")], wav).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelCapabilitiesOverride, ModelEntryConfig};
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
}
