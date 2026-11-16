//! HTTP surface: `/health`, `/api/version`, `/v1/models`(+`/refresh`),
//! `/metrics`, `/v1/openapi.json`, chat/transcribe/synthesize, UI fallback.
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
    extract::{Query, Request, State},
    http::{HeaderMap, StatusCode, Uri, header},
    middleware::{self, Next},
    response::{
        Html, IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use honkbox_api_types::{
    CatalogEntry, CatalogList, CatalogLocal, ChatMessage, ChatRequest, ChatResponse, ChatRole,
    DownloadAccepted, DownloadKind, DownloadList, DownloadRequest, DownloadState, DownloadStatus,
    GgufMetadata, Health, ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind,
    ModelMetadata, Pill, SafetensorsMetadata, SynthesizeRequest, Transcript, TranscriptSegment,
    Version,
};
use honkbox_audio_worker::{
    SttPipeline, SynthesizeRequest as WorkerSynthRequest,
    TranscribeRequest as WorkerTranscribeRequest, Transcript as WorkerTranscript, TtsPipeline,
    decode_wav, encode_wav,
};
use honkbox_llm_worker::{
    ChatMessage as WorkerChatMessage, ChatRequest as WorkerChatRequest, LlmPipeline, Role,
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
        openapi_json,
        metrics_endpoint,
        post_chat,
        post_chat_stream,
        post_transcribe,
        post_synthesize,
        post_download,
        list_downloads,
        get_download,
        delete_download,
        get_catalog
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
        ModelKind,
        ModelCapabilities,
        GgufMetadata,
        SafetensorsMetadata,
        ModelMetadata,
        ModelCatalogEntry,
        ModelCatalog,
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
    /// LLM swap manager (P1 of the llama-swap parity plan). When `Some`,
    /// `/v1/chat`(+`/stream`) resolve the request `model` slug through it:
    /// load-on-demand construction, per-entry idle TTL, LRU capacity. `None`
    /// keeps the startup-loaded `llm` pipeline (stub backends, or
    /// `[swap] enabled = false`). 429 semantics are unchanged either way —
    /// the residency slot stays the only inference gate.
    pub swap: Option<Arc<crate::swap::SwapManager>>,
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
        .route("/v1/catalog", get(get_catalog))
        .route("/v1/openapi.json", get(openapi_json))
        .route("/v1/chat", post(post_chat))
        .route("/v1/chat/stream", post(post_chat_stream))
        .route("/v1/transcribe", post(post_transcribe))
        .route("/v1/synthesize", post(post_synthesize))
        .route("/v1/downloads", post(post_download).get(list_downloads))
        .route(
            "/v1/downloads/{id}",
            get(get_download).delete(delete_download),
        )
        .route("/metrics", get(metrics_endpoint))
        .fallback(fallback)
        .layer(middleware::from_fn(track_metrics))
        .with_state(config)
}

/// Per-request metrics: one route/code counter + one duration observation
/// (see `metrics`); no per-request allocation.
async fn track_metrics(req: Request, next: Next) -> Response {
    let route = crate::metrics::route_label(req.uri().path());
    let started = Instant::now();
    let response = next.run(req).await;
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

/// `api-types` message → worker message (only the role enum differs).
fn worker_message(message: &ChatMessage) -> WorkerChatMessage {
    let role = match message.role {
        ChatRole::User => Role::User,
        ChatRole::Assistant => Role::Assistant,
        ChatRole::System => Role::System,
    };
    WorkerChatMessage {
        role,
        content: message.content.clone(),
    }
}

/// Try to reserve the process-wide GPU slot; a `Busy` rejection counts the
/// `honkbox_gpu_slot_busy_rejections_total` metric and maps to a 429.
fn reserve_gpu_slot() -> Result<honkbox_residency::Lease<'static>, (StatusCode, String)> {
    honkbox_residency::global().try_reserve().map_err(|err| {
        crate::metrics::note_gpu_slot_busy();
        (StatusCode::TOO_MANY_REQUESTS, err.to_string())
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
    let pipe = chat_pipeline(
        &config,
        &slug,
        clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds),
    )
    .await?;
    let worker_req = WorkerChatRequest {
        model: slug,
        messages: req.messages.iter().map(worker_message).collect(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
    };
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
    let pipe = chat_pipeline(
        &config,
        &slug,
        clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds),
    )
    .await?;
    let worker_req = WorkerChatRequest {
        model: slug,
        messages: req.messages.iter().map(worker_message).collect(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
    };
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let cancelled = Arc::new(AtomicBool::new(false));
    // The lease sits in a slot the token sink can drain: once the client is
    // gone, the next token boundary frees the slot right away. The handler
    // returns the SSE body without awaiting the worker, so tokens stream as
    // they are produced.
    let lease_slot = Arc::new(Mutex::new(None::<honkbox_residency::Lease<'static>>));
    // Separate handle for the SSE body: the worker closure consumes its own
    // clone, the drop path calls `abort` through this one.
    let abort_pipe = Arc::clone(&pipe);
    let worker_cancelled = cancelled.clone();
    let worker = tokio::task::spawn_blocking(move || match reserve_gpu_slot() {
        Ok(lease) => {
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
        }
        Err((_, message)) => {
            let _ = tx.send(Ok(Event::default().event("error").data(message)));
            let _ = tx.send(Ok(Event::default().data("[DONE]")));
        }
    });
    Ok(Sse::new(CancelOnDrop {
        inner: UnboundedReceiverStream::new(rx),
        cancelled,
        pipe: abort_pipe,
        _worker: worker,
    }))
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
    let pipe = config.stt.clone();
    let out: WorkerTranscript = tokio::task::spawn_blocking(move || match reserve_gpu_slot() {
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
    })??;
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
    let pipe = config.tts.clone();
    let audio = tokio::task::spawn_blocking(move || match reserve_gpu_slot() {
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
    })??;
    let wav = encode_wav(audio.sample_rate, &audio.pcm16);
    Ok(([(header::CONTENT_TYPE, "audio/wav")], wav).into_response())
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
