//! Spike-local salvo `GET /api/version` route (TASK-salvo-migration-spike).
//!
//! No `crate::` imports: the integration test includes this file via `#[path]`.
//! `Version` is local because shared DTOs derive utoipa/ts-rs only; adding
//! `salvo::oapi::ToSchema` there is TASK-salvo-dto-schema-bridge.

use salvo::http::HeaderValue;
use salvo::http::header::LINK;
use salvo::oapi::{ToSchema, endpoint};
use salvo::prelude::*;
use salvo::routing::FilterInfo;
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

/// Paths with a goal in `router()`, from structured filter metadata.
pub(crate) fn served_paths() -> Vec<String> {
    fn collect(router: &Router, prefix: &str, out: &mut Vec<String>) {
        let mut path = prefix.to_owned();
        for filter in router.filters() {
            if let FilterInfo::Path(segment) = filter.info() {
                let segment = segment.trim_matches('/');
                if !segment.is_empty() {
                    if !path.is_empty() {
                        path.push('/');
                    }
                    path.push_str(segment);
                }
            }
        }
        if router.goal.is_some() {
            out.push(format!("/{path}"));
        }
        for child in router.routers() {
            collect(child, &path, out);
        }
    }

    let mut paths = Vec::new();
    collect(&router(), "", &mut paths);
    paths.sort();
    paths.dedup();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn served_paths_come_from_router_filters() {
        assert_eq!(served_paths(), vec!["/api/version"]);
    }
}
