//! UI asset routes, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only the handlers and their private
//! helpers live here.

use std::path::Path;

use axum::{
    extract::State,
    http::{Uri, header},
    response::{Html, IntoResponse, Response},
};
use honkbox_api_types::ModelCatalog;

use super::{AppConfig, resolve_model_entry};

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
pub fn validate_model(
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
pub fn resolve_model_slug<'a>(model: &str, catalog: &'a ModelCatalog) -> Option<&'a str> {
    resolve_model_entry(model, catalog).map(|e| e.slug.as_str())
}

/// Embedded `web-dist/` asset when present, dev-disk file in `--dev`,
/// otherwise the static fallback page.
pub async fn fallback(State(config): State<AppConfig>, uri: Uri) -> Response {
    let raw = uri.path().trim_start_matches('/');
    let file = if raw.is_empty() { "index.html" } else { raw };
    if config.dev {
        if let Some(response) = serve_from_disk(&config.web_dist_dir, file).await {
            return response;
        }
    } else if let Some(asset) = super::EmbeddedUi::get(file) {
        return asset_response(file, asset.data.into_owned());
    }
    Html(include_str!("../../fallback.html")).into_response()
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
