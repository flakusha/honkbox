//! Admin routes for declared CLI apps plus the authenticated, allowlisted
//! `ComfyUI` read surface (`/comfyui/*`).
//!
//! No raw `ComfyUI` wire proxy exists here: `POST /prompt`, node-graph bodies,
//! `/ws`, `/interrupt` and `/queue` writes are deliberately absent. The typed
//! routes below are bearer-gated by the router's `route_layer`, and their
//! upstream is validated loopback-only. A ComfyUI-shaped compatibility subset
//! cannot satisfy honkbox's auth and validation rules (see the serve-compat
//! spike), so the wire surface is not exposed.
// size-allow: 400

use std::sync::OnceLock;

use axum::{
    Json,
    body::Body,
    extract::{Path as AxumPath, Query},
    http::{HeaderMap, header},
    response::{IntoResponse, Response},
};
use honkbox_api_types::{
    ComfyUiHistory, ComfyUiProgress, ComfyUiQueue, ComfyUiStatus, ManagedApp, ManagedAppList,
};
use serde::Deserialize;

include!("../apps.rs");

/// The installed manager. `None` until `main` (or a test harness) installs it.
static MANAGER: OnceLock<Mutex<Option<Arc<Manager>>>> = OnceLock::new();

fn slot() -> &'static Mutex<Option<Arc<Manager>>> {
    MANAGER.get_or_init(|| Mutex::new(None))
}

fn lock(slot: &'static Mutex<Option<Arc<Manager>>>) -> MutexGuard<'static, Option<Arc<Manager>>> {
    slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Install the process-wide manager (called once at serve startup).
#[allow(dead_code)] // `#[path]` test harnesses may omit this
pub(crate) fn install_manager(manager: Arc<Manager>) {
    *lock(slot()) = Some(manager);
}

/// Test-only swap: restore `None` in teardown so parallel suites stay isolated.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn set_manager_for_tests(manager: Option<Arc<Manager>>) {
    *lock(slot()) = manager;
}

fn manager() -> Option<Arc<Manager>> {
    lock(slot()).clone()
}

/// Read-route upstream. `None` unless a manager is installed with a validated
/// loopback URL and bounded client — `Manager::new` rejects anything else.
fn comfy_upstream() -> Option<Arc<Manager>> {
    let manager = manager()?;
    (!manager.comfyui_url().is_empty()).then_some(manager)
}

/// Upstream only when the probe also proves the typed surface can serve.
fn comfy_ready_upstream() -> Option<Arc<Manager>> {
    comfy_probe()?;
    comfy_upstream()
}

/// Map a manager snapshot to the additive wire DTO (no paths in the failure
/// string beyond the bounded stderr tail the manager already caps).
fn app_dto(status: AppStatus) -> ManagedApp {
    ManagedApp {
        id: status.id,
        state: status.state.as_str().to_owned(),
        launch: status.launch.as_str().to_owned(),
        resource_kind: status.resource_kind.as_str().to_owned(),
        endpoint: status.endpoint,
        version: status.version,
        error: status.error,
    }
}

fn named_501(what: &str, reason: &str) -> Response {
    (StatusCode::NOT_IMPLEMENTED, format!("{what} unavailable: {reason}")).into_response()
}

/// Admin scope check, layered on top of the router's bearer gate. Scoped keys
/// need `admin` for every app verb; the legacy single-key mode is unchanged.
fn app_error(error: AppError) -> Response {
    (error.status, error.message).into_response()
}

fn require_admin(headers: &HeaderMap) -> Option<Response> {
    let (enabled, scopes) = super::runtime_config::scope_facts();
    if !enabled || scopes.is_empty() {
        return None;
    }
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer ").or_else(|| value.strip_prefix("bearer ")))
        .map(str::trim);
    let allowed = scopes.iter().any(|(name, granted)| {
        granted.iter().any(|scope| scope == "admin")
            && std::env::var(name).ok().filter(|key| !key.is_empty()).is_some_and(|key| {
                presented.is_some_and(|value| super::middleware::constant_time_eq(
                    value.as_bytes(),
                    key.as_bytes(),
                ))
            })
    });
    (!allowed).then(|| {
        (StatusCode::FORBIDDEN, "Missing required scope: admin".to_owned()).into_response()
    })
}

/// List declared apps and their live state.
#[utoipa::path(
    get,
    path = "/v1/apps",
    responses((status = 200, description = "Declared managed apps", body = ManagedAppList))
)]
pub(super) async fn get_apps() -> Json<ManagedAppList> {
    let apps = manager().map_or_else(Vec::new, |manager| manager.list());
    Json(ManagedAppList { apps: apps.into_iter().map(app_dto).collect() })
}

/// One declared app's live state.
#[utoipa::path(
    get,
    path = "/v1/apps/{id}",
    params(("id" = String, Path, description = "Stable app id")),
    responses(
        (status = 200, description = "One managed app", body = ManagedApp),
        (status = 404, description = "Unknown app id")
    )
)]
pub(super) async fn get_app(AxumPath(id): AxumPath<String>) -> Response {
    let Some(status) = manager().and_then(|manager| manager.status(&id)) else {
        return (StatusCode::NOT_FOUND, format!("unknown app {id}")).into_response();
    };
    Json(app_dto(status)).into_response()
}

/// Explicitly start one declared app (never automatic).
#[utoipa::path(
    post,
    path = "/v1/apps/{id}/start",
    params(("id" = String, Path, description = "Stable app id")),
    responses(
        (status = 200, description = "Start accepted", body = ManagedApp),
        (status = 404, description = "Unknown app id"),
        (status = 409, description = "App already running"),
        (status = 501, description = "App not enabled or unsupported platform")
    )
)]
pub(super) async fn post_app_start(headers: HeaderMap, AxumPath(id): AxumPath<String>) -> Response {
    if let Some(forbidden) = require_admin(&headers) {
        return forbidden;
    }
    let Some(manager) = manager() else {
        return named_501("managed apps", "no manager installed");
    };
    if let Err(error) = manager.start(&id).await {
        return app_error(error);
    }
    manager.status(&id).map_or_else(
        || named_501("managed apps", "status lost after start"),
        |status| Json(app_dto(status)).into_response(),
    )
}

/// Explicitly stop one app; kills the whole process group.
#[utoipa::path(
    post,
    path = "/v1/apps/{id}/stop",
    params(("id" = String, Path, description = "Stable app id")),
    responses(
        (status = 200, description = "Stop accepted", body = ManagedApp),
        (status = 404, description = "Unknown app id"),
        (status = 500, description = "Teardown timed out")
    )
)]
pub(super) async fn post_app_stop(headers: HeaderMap, AxumPath(id): AxumPath<String>) -> Response {
    if let Some(forbidden) = require_admin(&headers) {
        return forbidden;
    }
    let Some(manager) = manager() else {
        return named_501("managed apps", "no manager installed");
    };
    if let Err(error) = manager.stop(&id).await {
        return app_error(error);
    }
    manager.status(&id).map_or_else(
        || (StatusCode::NOT_FOUND, format!("unknown app {id}")).into_response(),
        |status| Json(app_dto(status)).into_response(),
    )
}

/// Probe truth: managed readiness when a managed app is declared, otherwise
/// the cached external-tools row. No second truth source is invented.
fn comfy_probe() -> Option<(String, String, Vec<String>, Option<String>)> {
    let manager = comfy_upstream()?;
    if manager.configured("comfyui") {
        if !manager.ready("comfyui") {
            return None;
        }
        let status = manager.status("comfyui")?;
        return Some(("managed".to_owned(), "ready".to_owned(), vec!["workflow".to_owned()], status.version));
    }
    let row = super::external_tools::external_tools_capability()
        .tools
        .into_iter()
        .find(|tool| tool.id == "comfyui")?;
    if !row.available {
        return None;
    }
    Some(("external-tools".to_owned(), "detached".to_owned(), row.capabilities, row.version))
}

fn comfy_unavailable() -> Response {
    named_501("comfyui", "no probed ready app")
}

/// Honest `ComfyUI` readiness, derived from the probe that proved it.
#[utoipa::path(
    get,
    path = "/comfyui/status",
    responses((status = 200, description = "ComfyUI readiness", body = ComfyUiStatus))
)]
pub(super) async fn get_comfyui_status() -> Response {
    let Some((source, state, capabilities, version)) = comfy_probe() else {
        return comfy_unavailable();
    };
    Json(ComfyUiStatus {
        available: true,
        state,
        source,
        version,
        capabilities,
        loopback_only: true,
    })
    .into_response()
}

/// Bounded read of the `ComfyUI` queue: counts only, no node graph.
#[utoipa::path(
    get,
    path = "/comfyui/queue",
    responses(
        (status = 200, description = "Queue counts", body = ComfyUiQueue),
        (status = 501, description = "No probed ready ComfyUI")
    )
)]
pub(super) async fn get_comfyui_queue() -> Response {
    let Some(manager) = comfy_ready_upstream() else {
        return comfy_unavailable();
    };
    match read_json(&manager, "/queue").await {
        Ok(value) => Json(ComfyUiQueue {
            running: value.get("queue_running").and_then(|v| v.as_array()).map_or(0, Vec::len),
            pending: value.get("queue_pending").and_then(|v| v.as_array()).map_or(0, Vec::len),
        })
        .into_response(),
        Err(response) => response,
    }
}

/// Bounded read of one typed history summary (no raw workflow JSON).
#[utoipa::path(
    get,
    path = "/comfyui/history/{id}",
    params(("id" = String, Path, description = "Validated prompt id")),
    responses(
        (status = 200, description = "Typed history summary", body = ComfyUiHistory),
        (status = 400, description = "Invalid prompt id"),
        (status = 501, description = "No probed ready ComfyUI")
    )
)]
pub(super) async fn get_comfyui_history(AxumPath(id): AxumPath<String>) -> Response {
    if !safe_component(&id) {
        return (StatusCode::BAD_REQUEST, "invalid prompt id").into_response();
    }
    let Some(manager) = comfy_ready_upstream() else {
        return comfy_unavailable();
    };
    let path = format!("/history/{id}");
    let value = match read_json(&manager, &path).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let Some(entry) = value.get(&id) else {
        return (StatusCode::NOT_FOUND, format!("unknown prompt {id}")).into_response();
    };
    let status_obj = entry.get("status");
    let completed = status_obj
        .and_then(|value| value.get("completed"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let status = status_obj
        .and_then(|value| value.get("status_str"))
        .and_then(serde_json::Value::as_str)
        .filter(|value| matches!(*value, "success" | "error"))
        .map_or_else(
            || if completed { "completed".to_owned() } else { "queued".to_owned() },
            str::to_owned,
        );
    let outputs = entry
        .get("outputs")
        .and_then(|value| value.as_object())
        .map(|object| object.keys().cloned().collect())
        .unwrap_or_default();
    Json(ComfyUiHistory {
        prompt_id: id,
        status,
        outputs,
        progress: ComfyUiProgress { node: None, completed: None, total: None },
    })
    .into_response()
}

#[derive(Debug, Deserialize)]
pub(super) struct ViewQuery {
    filename: String,
    #[serde(default)]
    kind: Option<String>,
}

/// Validated output/input/temp byte view. Path separators and traversal are
/// rejected here; the upstream URL is built with query encoding, never string
/// interpolation of an arbitrary path.
#[utoipa::path(
    get,
    path = "/comfyui/view",
    params(
        ("filename" = String, Query, description = "Single safe filename"),
        ("kind" = Option<String>, Query, description = "output, input or temp")
    ),
    responses(
        (status = 200, description = "Output bytes"),
        (status = 400, description = "Invalid filename or kind"),
        (status = 501, description = "No probed ready ComfyUI")
    )
)]
pub(super) async fn get_comfyui_view(Query(query): Query<ViewQuery>) -> Response {
    if !safe_component(&query.filename)
        || query.kind.as_deref().is_some_and(|kind| !matches!(kind, "output" | "input" | "temp"))
    {
        return (StatusCode::BAD_REQUEST, "invalid view parameters").into_response();
    }
    let Some(manager) = comfy_ready_upstream() else {
        return comfy_unavailable();
    };
    let Ok(mut url) = reqwest::Url::parse(&format!("{}/view", manager.comfyui_url())) else {
        return comfy_unavailable();
    };
    {
        let mut pairs = url.query_pairs_mut();
        pairs.append_pair("filename", &query.filename);
        if let Some(kind) = &query.kind {
            pairs.append_pair("type", kind);
        }
    }
    let Ok(response) = manager.client().get(url).send().await else {
        return comfy_unavailable();
    };
    if response.status() != StatusCode::OK {
        return (StatusCode::NOT_FOUND, "comfyui output not found").into_response();
    }
    match response.bytes().await {
        Ok(bytes) if bytes.len() <= 16 * 1024 * 1024 => Response::builder()
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(Body::from(bytes))
            .unwrap_or_else(|_| comfy_unavailable()),
        _ => (StatusCode::BAD_GATEWAY, "comfyui output too large").into_response(),
    }
}

/// Single path component, no traversal, bounded length (never a raw path).
fn safe_component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && !value.starts_with('.')
        && !value.chars().any(|ch| matches!(ch, '/' | '\\' | '\0'))
        && value.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

/// Read a bounded JSON body from the validated loopback upstream.
#[allow(clippy::result_large_err)] // the error is an axum response forwarded verbatim
async fn read_json(manager: &Arc<Manager>, path: &str) -> Result<serde_json::Value, Response> {
    let url = format!("{}{path}", manager.comfyui_url());
    let response = manager
        .client()
        .get(url)
        .send()
        .await
        .map_err(|_| comfy_unavailable())?;
    if response.status() != StatusCode::OK {
        return Err(comfy_unavailable());
    }
    let bytes = response.bytes().await.map_err(|_| comfy_unavailable())?;
    if bytes.len() > 1024 * 1024 {
        return Err((StatusCode::BAD_GATEWAY, "comfyui reply too large").into_response());
    }
    serde_json::from_slice(&bytes).map_err(|_| (StatusCode::BAD_GATEWAY, "invalid comfyui reply").into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_ids_and_view_names_are_allowlisted() {
        assert!(safe_component("ComfyUI_00001_.png"));
        assert!(safe_component("prompt-1234"));
        assert!(!safe_component("../etc/passwd"));
        assert!(!safe_component(".."));
        assert!(!safe_component("a/b"));
        assert!(!safe_component(""));
    }

    #[test]
    fn legacy_single_key_skips_admin_requirement() {
        let headers = HeaderMap::new();
        assert!(require_admin(&headers).is_none());
    }
}