//! External-tool discovery routes, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only handlers, cache and DTO mapping
//! live here.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use axum::{
    Json,
    extract::{Path as AxumPath, Query},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::{ExternalTool, ExternalToolsCapability};
use honkbox_external_tools::{Registry, Status as ToolStatus, ToolId, Unavailable};

/// Cached external-engine registry slot: `None` until startup discovery
/// runs; read-only test harnesses get a deterministic "not installed"
/// snapshot instead of spawning probes (or inject one via
/// [`set_registry_for_tests`]).
static EXTERNAL_TOOLS: OnceLock<Arc<tokio::sync::RwLock<Option<Registry>>>> = OnceLock::new();

/// The shared registry slot, created on first use.
fn cache() -> &'static Arc<tokio::sync::RwLock<Option<Registry>>> {
    EXTERNAL_TOOLS.get_or_init(|| Arc::new(tokio::sync::RwLock::new(None)))
}

/// Probe every configured tool once at startup; reloads reuse this cache.
#[allow(dead_code)] // main.rs is the only caller; #[path] test harnesses omit it
pub(crate) async fn init_external_tools(config: crate::config::ExternalToolsConfig) {
    let registry = Registry::discover(config.into_probe_config()).await;
    *cache().write().await = Some(registry);
}

/// Test-only cache swap: a harness installs its deterministic snapshot (or
/// restores the unprobed `None`). Callers must serialize against the
/// slot's other readers and undo the swap before the test ends.
#[cfg(test)]
#[allow(dead_code)] // each #[path] test binary compiles this; not all call it
pub(crate) async fn set_registry_for_tests(registry: Option<Registry>) {
    *cache().write().await = registry;
}

fn missing_tool_status(id: ToolId) -> ToolStatus {
    ToolStatus {
        id,
        version: None,
        capabilities: Vec::new(),
        devices: Vec::new(),
        unavailable: Some(Unavailable::NotInstalled),
    }
}

fn external_tool_status(id: ToolId) -> ToolStatus {
    cache()
        .try_read()
        .ok()
        .and_then(|registry| {
            registry
                .as_ref()
                .and_then(|registry| registry.status(id).cloned())
        })
        .unwrap_or_else(|| missing_tool_status(id))
}

fn external_tool_dto(status: &ToolStatus) -> ExternalTool {
    ExternalTool {
        id: status.id.as_str().to_owned(),
        version: status.version.clone(),
        capabilities: status
            .capabilities
            .iter()
            .map(|capability| capability.as_str().to_owned())
            .collect(),
        devices: status.devices.clone(),
        available: status.is_available(),
        unavailable: status.unavailable.as_ref().map(ToString::to_string),
    }
}

/// The Blender engine the job route drives: the probed version and the
/// binary the probe used.
#[derive(Debug, Clone)]
pub(super) struct BlenderEngine {
    /// The probed Blender version (e.g. `4.5.15`).
    pub version: String,
    /// The executable the probe ran.
    pub binary: PathBuf,
}

/// The Blender engine, when one is installed and probed. `None` before the
/// startup probe ran, when Blender is not installed, or when the probe
/// could not prove a version.
pub(super) fn blender_engine() -> Option<BlenderEngine> {
    let registry = cache().try_read().ok()?;
    let registry = registry.as_ref()?;
    let status = registry.status(ToolId::Blender)?;
    if !status.is_available() {
        return None;
    }
    Some(BlenderEngine {
        version: status.version.clone()?,
        binary: registry.blender_binary(),
    })
}

pub(super) fn external_tools_capability() -> ExternalToolsCapability {
    ExternalToolsCapability {
        tools: ToolId::ALL
            .iter()
            .copied()
            .map(|id| external_tool_dto(&external_tool_status(id)))
            .collect(),
    }
}

/// A missing tool or unproven capability is a named 501, never a panic or a
/// leaked probe error chain.
fn external_tool_501(tool: &str, missing: &str) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        format!("external tool {tool} unavailable: {missing}"),
    )
        .into_response()
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct ExternalToolQuery {
    capability: Option<String>,
}

/// List discovered external engines and their capabilities.
#[utoipa::path(
    get,
    path = "/v1/external-tools",
    responses((status = 200, description = "Discovered external engines", body = ExternalToolsCapability))
)]
pub(super) async fn get_external_tools() -> Json<ExternalToolsCapability> {
    Json(external_tools_capability())
}

/// Probe one external tool, optionally requiring a capability.
#[utoipa::path(
    get,
    path = "/v1/external-tools/{id}",
    params(("id" = String, Path, description = "Stable tool id")),
    responses(
        (status = 200, description = "Probed tool status", body = ExternalTool),
        (status = 404, description = "Unknown tool id"),
        (status = 501, description = "Tool or requested capability is unavailable")
    )
)]
pub(super) async fn get_external_tool(
    AxumPath(id): AxumPath<String>,
    Query(query): Query<ExternalToolQuery>,
) -> Response {
    let Some(tool_id) = ToolId::ALL
        .into_iter()
        .find(|tool| tool.as_str() == id.as_str())
    else {
        return (StatusCode::NOT_FOUND, format!("unknown external tool {id}")).into_response();
    };
    let status = external_tool_status(tool_id);
    if let Some(reason) = &status.unavailable {
        return external_tool_501(tool_id.as_str(), &reason.to_string());
    }
    if let Some(capability) = query.capability.as_deref()
        && !status
            .capabilities
            .iter()
            .any(|cap| cap.as_str() == capability)
    {
        return external_tool_501(tool_id.as_str(), &format!("capability {capability}"));
    }
    Json(external_tool_dto(&status)).into_response()
}

/// Re-probe all external engines and refresh the capability cache.
#[utoipa::path(
    post,
    path = "/v1/external-tools/reload",
    responses((status = 200, description = "Re-probed external engines", body = ExternalToolsCapability))
)]
pub(super) async fn post_external_tools_reload() -> Json<ExternalToolsCapability> {
    if let Some(registry) = cache().write().await.as_mut() {
        registry.reload().await;
    }
    Json(external_tools_capability())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_tool_dto_keeps_probe_truth() {
        let status = ToolStatus {
            id: ToolId::Blender,
            version: Some("4.5.15".to_owned()),
            capabilities: vec![
                honkbox_external_tools::Capability::Render,
                honkbox_external_tools::Capability::Cycles,
            ],
            devices: vec!["CPU".to_owned()],
            unavailable: None,
        };
        let dto = external_tool_dto(&status);
        assert_eq!(dto.id, "blender");
        assert_eq!(dto.version.as_deref(), Some("4.5.15"));
        assert_eq!(dto.capabilities, ["render", "cycles"]);
        assert_eq!(dto.devices, ["CPU"]);
        assert!(dto.available);
        assert!(dto.unavailable.is_none());

        let broken = ToolStatus {
            id: ToolId::ComfyUi,
            unavailable: Some(Unavailable::ProbeFailed("http 500".to_owned())),
            ..status
        };
        let dto = external_tool_dto(&broken);
        assert_eq!(dto.unavailable.as_deref(), Some("probe failed: http 500"));
        assert!(!dto.available);
        let response = external_tool_501("comfyui", "capability workflow");
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    }
}
