//! Bounded startup-warning document served at `GET /v1/status/warnings`.
//!
//! Warnings keep the exact text the config loader produced (including any
//! operator paths); the document adds only fixed `code`/`source` labels.

use std::sync::{LazyLock, PoisonError, RwLock};

use axum::{
    Json,
    response::{IntoResponse, Response},
};
use serde::Serialize;

/// Hard cap on stored warnings, so neither the store nor the response grows
/// with config history.
#[cfg_attr(test, allow(dead_code))]
const MAX_WARNINGS: usize = 64;

static WARNINGS: LazyLock<RwLock<Vec<String>>> = LazyLock::new(|| RwLock::new(Vec::new()));

/// Stable-order warning document. Both servers serialize this same shape.
#[derive(Debug, Serialize)]
pub(crate) struct StatusWarnings {
    pub(crate) schema_version: u32,
    pub(crate) warnings: Vec<StatusWarning>,
}

/// One warning: fixed labels plus the original message text.
#[derive(Debug, Serialize)]
pub(crate) struct StatusWarning {
    pub(crate) code: String,
    pub(crate) message: String,
    pub(crate) source: String,
}

#[cfg_attr(test, allow(dead_code))]
fn store(warnings: Vec<String>) {
    let mut warnings = warnings;
    warnings.truncate(MAX_WARNINGS);
    *WARNINGS.write().unwrap_or_else(PoisonError::into_inner) = warnings;
}

/// Publish the non-fatal notices collected at startup (overwrites prior set).
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn set_warnings(warnings: Vec<String>) {
    store(warnings);
}

/// Overwrite the store from an isolated test, with the same clamp as prod.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn set_warnings_for_tests(warnings: Vec<String>) {
    store(warnings);
}

/// Snapshot in insertion order, capped at [`MAX_WARNINGS`].
pub(crate) fn warnings_document() -> StatusWarnings {
    let warnings = WARNINGS.read().unwrap_or_else(PoisonError::into_inner);
    StatusWarnings {
        schema_version: 1,
        warnings: warnings
            .iter()
            .map(|message| StatusWarning {
                code: "config".to_owned(),
                message: message.clone(),
                source: "config".to_owned(),
            })
            .collect(),
    }
}

/// `GET /v1/status/warnings`: startup notices as a bounded JSON document.
#[utoipa::path(
    get,
    path = "/v1/status/warnings",
    responses((status = 200, description = "Bounded startup warnings", body = String))
)]
pub(crate) async fn get_status_warnings_axum() -> Response {
    Json(warnings_document()).into_response()
}
