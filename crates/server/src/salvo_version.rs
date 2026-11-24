//! Spike-local salvo `GET /api/version` route (TASK-salvo-migration-spike).
//!
//! No `crate::` imports: the integration test includes this file via `#[path]`.
//! `Version` is local because shared DTOs derive utoipa/ts-rs only; adding
//! `salvo::oapi::ToSchema` there is TASK-salvo-dto-schema-bridge.

use salvo::http::header::LINK;
use salvo::http::HeaderValue;
use salvo::oapi::{endpoint, ToSchema};
use salvo::prelude::*;
use serde::Serialize;

/// Discovery links from the legacy version probe to the canonical contract.
const DISCOVERY_LINKS: &str = concat!(
    "</v1/openapi.json>; rel=\"service-desc\"; type=\"application/json\", ",
    "</v1/capabilities>; rel=\"service-meta\"; type=\"application/json\", ",
    "</api-docs/index.html>; rel=\"help\"; type=\"text/html\"",
);

/// Binary name + version (spike-local DTO, see module note).
#[derive(Debug, Serialize, ToSchema)]
struct Version {
    /// Binary name, always `"honkbox"`.
    name: String,
    /// Crate version, from `CARGO_PKG_VERSION`.
    version: String,
}

/// Binary name + version.
// `#[endpoint]` expands to an async trait impl with no await; clippy pedantic
// flags the generated code, not this handler.
#[allow(clippy::unused_async_trait_impl)]
#[endpoint(responses((status_code = 200, description = "Binary name and version", body = Version)))]
pub async fn get_version(res: &mut Response) {
    res.headers_mut()
        .insert(LINK, HeaderValue::from_static(DISCOVERY_LINKS));
    res.render(Json(Version {
        name: "honkbox".to_owned(),
        version: env!("CARGO_PKG_VERSION").to_owned(),
    }));
}

pub fn router() -> Router {
    Router::new().push(Router::with_path("api/version").get(get_version))
}
