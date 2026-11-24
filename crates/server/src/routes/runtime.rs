//! Runtime residency and unload routes, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only handlers live here.

use std::time::{Duration, Instant};

use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::{HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use honkbox_api_types::{
    MemoryCapacity, MemoryGpu, MemoryInfo, ResidentModel, RunningList, RunningModel,
    UnloadAllResult, UnloadResult,
};

use crate::swap::{DrainProbe, SwapManager};

use super::AppConfig;

/// Fixed courtesy grace for an explicit unload: a real model can take much
/// longer to free, so control routes report the truth at the bound instead
/// of blocking indefinitely. No new config knob.
const DRAIN_BOUND: Duration = Duration::from_millis(200);
const DRAIN_POLL: Duration = Duration::from_millis(5);
/// Additive header; existing JSON bodies stay byte-compatible for loop-lore.
const DRAIN_HEADER: &str = "x-honkbox-drain";

/// Post-unload state of the evicted pipelines, observed without a lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrainState {
    /// No holder left and no new resident under the evicted slug(s).
    Drained,
    /// An in-flight holder still owns the pipeline; it is never killed.
    InFlight,
    /// A queued admission re-loaded the slug inside the wait window.
    Reloaded,
}

impl DrainState {
    fn header_value(self) -> &'static str {
        match self {
            Self::Drained => "drained",
            Self::InFlight => "in-flight",
            Self::Reloaded => "reloaded",
        }
    }
}

/// Wait at most [`DRAIN_BOUND`] for the evicted pipelines' in-flight `Arc`
/// holders to drop. A queued admission can re-load the same slug in this
/// window; that is reported as `reloaded`, never hidden. No GPU slot is taken
/// here and no request is aborted — the readouts stay honest instead.
async fn drain_state(swap: &SwapManager, probes: &[(String, DrainProbe)]) -> DrainState {
    if probes.is_empty() {
        return DrainState::Drained;
    }
    let deadline = Instant::now() + DRAIN_BOUND;
    loop {
        let in_flight = probes.iter().any(|(_, probe)| !probe.is_empty());
        let resident = swap.residency();
        let reloaded = probes
            .iter()
            .any(|(slug, _)| resident.iter().any(|entry| entry.slug == *slug));
        if !in_flight && !reloaded {
            return DrainState::Drained;
        }
        if Instant::now() >= deadline {
            return if in_flight {
                DrainState::InFlight
            } else {
                DrainState::Reloaded
            };
        }
        tokio::time::sleep(DRAIN_POLL).await;
    }
}

fn with_drain(mut response: Response, state: DrainState) -> Response {
    response.headers_mut().insert(
        HeaderName::from_static(DRAIN_HEADER),
        HeaderValue::from_static(state.header_value()),
    );
    response
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
pub(crate) async fn get_memory(State(config): State<AppConfig>) -> Response {
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
pub(crate) async fn get_running(State(config): State<AppConfig>) -> Response {
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
    responses((status = 200, description = "Non-pinned residents evicted; X-Honkbox-Drain: drained|in-flight|reloaded (evicted entries only)", body = UnloadAllResult))
)]
pub(crate) async fn post_unload_all(State(config): State<AppConfig>) -> Response {
    // Capture weak handles BEFORE eviction; pinned survivors are not probed.
    let probes = config
        .swap
        .as_ref()
        .map_or_else(Vec::new, |swap| swap.drain_probes());
    let evicted = match &config.swap {
        Some(swap) => swap.evict_all_unpinned(crate::metrics::UnloadReason::Explicit),
        None => 0,
    };
    if evicted > 0 {
        tracing::info!(count = evicted, "bulk explicit unload");
    }
    if probes.is_empty() {
        return Json(UnloadAllResult {
            unloaded: u32::try_from(evicted).unwrap_or(u32::MAX),
            reason: "explicit".to_owned(),
        })
        .into_response();
    }
    let state = match &config.swap {
        Some(swap) => drain_state(swap, &probes).await,
        None => DrainState::Drained,
    };
    with_drain(
        Json(UnloadAllResult {
            unloaded: u32::try_from(evicted).unwrap_or(u32::MAX),
            reason: "explicit".to_owned(),
        })
        .into_response(),
        state,
    )
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
        (status = 200, description = "Unloaded or not resident; X-Honkbox-Drain: drained|in-flight|reloaded", body = UnloadResult),
        (status = 404, description = "Unknown model (body contains `model not found`)"),
        (status = 409, description = "Model is pinned; explicit refusal")
    )
)]
pub(crate) async fn post_unload_one(
    State(config): State<AppConfig>,
    AxumPath(model): AxumPath<String>,
) -> Response {
    let catalog = config.catalog_snapshot();
    let Some(slug) = super::models::control_slug(&model, &catalog) else {
        // Literal substring is load-bearing for loop-lore's classifier.
        return (StatusCode::NOT_FOUND, format!("model not found: {model}")).into_response();
    };
    let slug = slug.to_owned();
    // Weak handle captured while the entry is still resident: after eviction
    // it reports the in-flight holders that are deliberately left running.
    let probes: Vec<(String, DrainProbe)> = config
        .swap
        .as_ref()
        .and_then(|swap| swap.drain_probe(&slug))
        .map(|probe| vec![(slug.clone(), probe)])
        .unwrap_or_default();
    let outcome = match &config.swap {
        Some(swap) => swap.evict_by_slug(&slug, crate::metrics::UnloadReason::Explicit),
        None => crate::swap::EvictOutcome::NotLoaded,
    };
    let response = super::models::unload_response(outcome, slug);
    if !response.status().is_success() || probes.is_empty() {
        return response;
    }
    let state = match &config.swap {
        Some(swap) => drain_state(swap, &probes).await,
        None => DrainState::Drained,
    };
    with_drain(response, state)
}
