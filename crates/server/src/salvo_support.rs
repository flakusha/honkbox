//! Small bridge used by salvo endpoints that call the existing axum-style
//! handlers during the mixed cutover.
//!
//! Handlers keep their framework-neutral logic and return values that
//! implement axum's `IntoResponse`; the salvo endpoint renders that exact
//! status, headers and bytes. No axum router or middleware is involved.
//! `AppConfig` is injected once per request via `Router::affix_state()`.

use axum::response::IntoResponse;
use salvo::http::{ResBody, StatusCode};
use salvo::prelude::*;
use serde::de::DeserializeOwned;

use crate::routes::AppConfig;

/// The config affixed to the API router. Panics only on a wiring bug.
/// Returns an owned `AppConfig` clone so callers can pass `State(config)`.
pub fn config(depot: &Depot) -> AppConfig {
    depot
        .get_typed::<AppConfig>()
        .expect("AppConfig must be affixed to the salvo router")
        .clone()
}

/// Render an axum-style handler result into salvo's response, byte for byte.
pub async fn respond<R: IntoResponse>(res: &mut Response, result: R) {
    let response = result.into_response();
    let status = response.status();
    let headers = response.headers().clone();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("handler body is in-memory during the cutover");

    res.status_code =
        Some(StatusCode::from_u16(status.as_u16()).expect("axum status maps to salvo"));
    res.headers.clear();
    for (name, value) in headers {
        if let Some(name) = name {
            res.headers.append(name, value);
        }
    }
    res.body = ResBody::from(body.to_vec());
}

/// JSON request body with the same 400-class failure as axum's `Json` extractor.
pub async fn json<T: DeserializeOwned>(req: &mut Request) -> Result<T, StatusError> {
    req.parse_json::<T>()
        .await
        .map_err(|error| StatusError::bad_request().detail(error.to_string()))
}

/// Query struct with the same 400-class failure as axum's `Query` extractor.
pub fn query<T: DeserializeOwned>(req: &mut Request) -> Result<T, StatusError> {
    req.parse_queries::<T>()
        .map_err(|error| StatusError::bad_request().detail(error.to_string()))
}

/// Path parameter with a named 400 when absent or of the wrong type.
pub fn path<T>(req: &Request, name: &str) -> Result<T, StatusError>
where
    for<'de> T: salvo::serde::Deserialize<'de>,
{
    req.param::<T>(name).ok_or_else(|| {
        StatusError::bad_request().detail(format!("missing path parameter `{name}`"))
    })
}
