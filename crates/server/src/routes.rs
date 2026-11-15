//! HTTP surface: `/health`, `/api/version`, `/v1/openapi.json`, UI fallback.
//!
//! The `OpenAPI` document is served as plain JSON via `Json` — no
//! `utoipa-axum` router glue.

use std::path::{Path, PathBuf};

use axum::{
    Json, Router,
    extract::State,
    http::{Uri, header},
    response::{Html, IntoResponse, Response},
    routing::get,
};
use rust_embed::Embed;
use sd_rip_api_types::{Health, Version};
use utoipa::OpenApi;

/// Compile-time UI bundle (release always embeds; dev serves from disk instead).
#[derive(Embed)]
#[folder = "../../web-dist/"]
struct EmbeddedUi;

/// `OpenAPI` document for the whole HTTP surface.
#[derive(OpenApi)]
#[openapi(
    paths(health, get_version, openapi_json),
    components(schemas(Health, Version))
)]
pub struct ApiDoc;

/// Runtime switch for the UI fallback: embedded bundle vs. disk directory.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Serve `web_dist_dir` from disk instead of the embedded bundle.
    pub dev: bool,
    /// Directory served in dev mode (Vite output).
    pub web_dist_dir: PathBuf,
}

/// Build the full router.
pub fn app(config: AppConfig) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/version", get(get_version))
        .route("/v1/openapi.json", get(openapi_json))
        .fallback(fallback)
        .with_state(config)
}

/// Liveness probe.
#[utoipa::path(get, path = "/health", responses((status = 200, description = "Server is alive", body = Health)))]
async fn health() -> Json<Health> {
    Json(Health {
        status: "ok".to_owned(),
    })
}

/// Binary name + version.
#[utoipa::path(get, path = "/api/version", responses((status = 200, description = "Binary name and version", body = Version)))]
async fn get_version() -> Json<Version> {
    Json(Version {
        name: "sd-rip".to_owned(),
        version: env!("CARGO_PKG_VERSION").to_owned(),
    })
}

/// `OpenAPI` 3.1 document as JSON.
#[utoipa::path(get, path = "/v1/openapi.json", responses((status = 200, description = "OpenAPI document")))]
async fn openapi_json() -> impl IntoResponse {
    Json(ApiDoc::openapi())
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
