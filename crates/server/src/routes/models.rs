//! Model catalog and model-lifecycle routes, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only handlers and their helpers live
//! here.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use honkbox_api_types::{CatalogList, ModelCatalog, ModelDeleteResult, OpenAiError, UnloadResult};

use super::AppConfig;

/// `GET /v1/catalog`: the raw catalog list, serialized from the shared
/// snapshot so the response is byte-identical across refreshes.
#[utoipa::path(
    get,
    path = "/v1/catalog",
    responses(
        (status = 200, description = "Curated catalog with local presence + fit pills", body = CatalogList),
    )
)]
pub(super) async fn get_catalog(State(config): State<AppConfig>) -> Response {
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

/// `GET /v1/models`: the discovered-model catalog as an OpenAI-style list
/// envelope (`{object: "list", data: [...]}`). Served from the shared scan
/// snapshot; `POST /v1/models/refresh` re-scans and swaps it.
#[utoipa::path(get, path = "/v1/models", responses((status = 200, description = "Discovered model catalog", body = ModelCatalog)))]
pub(super) async fn get_models(State(config): State<AppConfig>) -> Response {
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
pub(super) async fn refresh_models(State(config): State<AppConfig>) -> Response {
    match super::rescan_catalog(&config).await {
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
pub(super) fn control_slug<'a>(model: &str, catalog: &'a ModelCatalog) -> Option<&'a str> {
    super::resolve_model_slug(model, catalog)
}

/// Map an unload outcome to its HTTP response. `Pinned` is an explicit 409
/// refusal (a loud log accompanies it) - pinned residents are never evicted
/// silently; `NotLoaded` is a 200 idempotent no-op, deliberately NOT the
/// llama-swap 404 `no local server found` (loop-lore's client treats any
/// 200 as an unconditional continue-case, so 200 needs no body matching).
pub(super) fn unload_response(outcome: crate::swap::EvictOutcome, slug: String) -> Response {
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
pub(super) async fn post_model_unload(
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

/// `DELETE /v1/models/{slug}` (`NewItems` §1): explicit, **permanent** model
/// deletion — the weight file (or directory) is removed from disk and the
/// catalog entry is dropped; there is no trash and no undo, recovery means
/// downloading the model again. Guardrails: an unknown slug (alias or
/// registry slug) is a 404; a resident or busy model (chat warm-set
/// residency: resident, pinned, or in-flight) is refused with 409 — unload
/// it first (`POST /v1/models/{slug}/unload`). Success removes the file(s),
/// re-scans the registry roots (best-effort SQLite cache sync) and drops
/// the entry from the shared catalog snapshot, then returns 200 naming
/// exactly what was removed.
#[utoipa::path(
    delete,
    path = "/v1/models/{slug}",
    params(("slug" = String, Path, description = "Operator alias or registry slug")),
    responses(
        (status = 200, description = "Model deleted permanently", body = ModelDeleteResult),
        (status = 404, description = "Unknown slug or alias", body = OpenAiError),
        (status = 409, description = "Model is resident, pinned, or busy; unload it first", body = OpenAiError),
        (status = 500, description = "File removal failed", body = OpenAiError)
    )
)]
pub(super) async fn delete_model(
    State(config): State<AppConfig>,
    AxumPath(model): AxumPath<String>,
) -> Response {
    let catalog = config.catalog_snapshot();
    let Some(entry) = catalog
        .data
        .iter()
        .find(|e| e.slug == model || e.id == model)
    else {
        return super::openai_error_response(
            StatusCode::NOT_FOUND,
            format!("unknown model {model:?}: not a catalog slug or alias (see GET /v1/models)"),
            None,
        );
    };
    let slug = entry.slug.clone();
    let path = PathBuf::from(&entry.path);
    // Resident/busy guard: the swap manager is the chat warm set; any entry
    // it holds is either pinned, idle-resident, or in flight — all three
    // refuse (audio pipelines are startup-loaded and own no residency slot,
    // so they are never "resident" here).
    if let Some(swap) = &config.swap
        && let Some(resident) = swap.residency().iter().find(|r| r.slug == slug)
    {
        let why = if resident.pinned {
            "pinned via [swap].pin"
        } else if resident.in_use {
            "busy (in-flight request holds the pipeline)"
        } else {
            "resident"
        };
        tracing::warn!(slug, "model deletion refused: {why}");
        return super::openai_error_response(
            StatusCode::CONFLICT,
            format!("model {slug:?} is {why}; unload it before deleting"),
            None,
        );
    }
    let removed = match delete_model_path(&path).await {
        Ok(removed) => removed,
        Err(err) => {
            return super::openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("delete of {} failed: {err}", path.display()),
                None,
            );
        }
    };
    // Catalog rows: the file walk is the source of truth, so a rescan is
    // the honest row drop (it also syncs/prunes the registry SQLite cache).
    // Best-effort: if the rescan falls back to the cache (empty roots), the
    // snapshot filter below still drops the deleted entry deterministically.
    if let Err(err) = super::rescan_catalog(&config).await {
        tracing::warn!(%err, slug, "post-delete rescan failed; dropping the entry from the snapshot anyway");
    }
    let mut fresh = (*config.catalog_snapshot()).clone();
    fresh.data.retain(|e| e.slug != slug);
    *config
        .catalog
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::new(fresh);
    tracing::info!(slug, removed = removed.len(), "model deleted permanently");
    Json(ModelDeleteResult {
        deleted: true,
        slug,
        removed,
    })
    .into_response()
}

/// Remove one catalog weight file (or directory) from disk. A path that is
/// already gone is fine — the catalog is a snapshot and may be stale.
async fn delete_model_path(path: &Path) -> std::io::Result<Vec<String>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    if path.is_dir() {
        tokio::fs::remove_dir_all(path).await?;
    } else {
        tokio::fs::remove_file(path).await?;
        // The provenance sidecar (downloads.rs) must not outlive its file.
        crate::downloads::remove_artifact_meta(path).await;
    }
    Ok(vec![path.display().to_string()])
}
