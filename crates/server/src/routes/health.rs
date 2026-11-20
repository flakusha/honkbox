//! Health, version and observability routes, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only the handlers and their private
//! helpers live here.

use axum::{
    Json,
    http::{HeaderMap, header},
    response::{IntoResponse, Response},
};
use honkbox_api_types::{Health, Version};
use utoipa::OpenApi;

/// Liveness probe. Content-negotiated: the JSON `Health` body by default
/// (and for `Accept: application/json`), a literal `OK` text/plain body for
/// probes that prefer `text/plain` (uptime monitors, `docker --health-cmd`
/// equivalents).
#[utoipa::path(get, path = "/health", responses((status = 200, description = "Server is alive", body = Health)))]
pub(super) async fn health(headers: HeaderMap) -> Response {
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

/// Discovery links from the legacy version probe to the canonical contract.
/// The body stays byte-compatible; clients get one place to follow for both
/// the machine-readable capabilities document and the `OpenAPI` schema.
const DISCOVERY_LINKS: &str = concat!(
    "</v1/openapi.json>; rel=\"service-desc\"; type=\"application/json\", ",
    "</v1/capabilities>; rel=\"service-meta\"; type=\"application/json\"",
);

/// Binary name + version.
#[utoipa::path(get, path = "/api/version", responses((status = 200, description = "Binary name and version", body = Version)))]
pub(super) async fn get_version() -> Response {
    (
        [(header::LINK, DISCOVERY_LINKS)],
        Json(Version {
            name: "honkbox".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
        }),
    )
        .into_response()
}

/// `OpenAPI` 3.1 document as JSON.
#[utoipa::path(get, path = "/v1/openapi.json", responses((status = 200, description = "OpenAPI document")))]
pub(super) async fn openapi_json() -> impl IntoResponse {
    Json(super::ApiDoc::openapi())
}

/// `GET /metrics`: minimal Prometheus text exposition (vLLM-vocabulary
/// subset; see `metrics`).
#[utoipa::path(
    get,
    path = "/metrics",
    responses((status = 200, description = "Prometheus text exposition", content_type = "text/plain"))
)]
pub(super) async fn metrics_endpoint() -> Response {
    // The download metric families live in `downloads.rs` (its own
    // AtomicU64 grids); their exposition fragment is appended here, and the
    // residency ledger's fixed-kind admission grids append after those.
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        crate::metrics::exposition()
            + &crate::downloads::exposition()
            + &honkbox_residency::global().exposition(),
    )
        .into_response()
}
