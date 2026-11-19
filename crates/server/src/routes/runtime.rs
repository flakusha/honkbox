//! Runtime residency and unload routes, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only handlers live here.

use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::{
    MemoryCapacity, MemoryGpu, MemoryInfo, ResidentModel, RunningList, RunningModel,
    UnloadAllResult, UnloadResult,
};

use super::AppConfig;

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
pub(super) async fn get_memory(State(config): State<AppConfig>) -> Response {
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
pub(super) async fn get_running(State(config): State<AppConfig>) -> Response {
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
pub(super) async fn post_unload_all(State(config): State<AppConfig>) -> Response {
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
pub(super) async fn post_unload_one(
    State(config): State<AppConfig>,
    AxumPath(model): AxumPath<String>,
) -> Response {
    let catalog = config.catalog_snapshot();
    let Some(slug) = super::models::control_slug(&model, &catalog) else {
        // Literal substring is load-bearing for loop-lore's classifier.
        return (StatusCode::NOT_FOUND, format!("model not found: {model}")).into_response();
    };
    let slug = slug.to_owned();
    let outcome = match &config.swap {
        Some(swap) => swap.evict_by_slug(&slug, crate::metrics::UnloadReason::Explicit),
        None => crate::swap::EvictOutcome::NotLoaded,
    };
    super::models::unload_response(outcome, slug)
}
