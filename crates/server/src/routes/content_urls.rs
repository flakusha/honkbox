//! `POST /v1/content-urls`: mint a short-lived signed URL for one job's
//! content (TASK-capability-signed-urls). The route only checks that the job
//! id is live in the registry; it never reads the bundle itself — the content
//! route re-validates the name against the manifest when the URL is used.

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::OpenAiError;

use super::middleware;
use super::signed_urls;
use super::runtime_config;
use crate::jobs_registry;

/// Request body: one job ULID plus a manifest name or `bundle.zip`.
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub(crate) struct ContentUrlRequest {
    /// 26-character Crockford ULID of a job in the in-flight registry.
    pub job_id: String,
    /// A file listed in the job's manifest, or the literal `bundle.zip`.
    pub content_name: String,
}

/// Relative URL to hand to a client; the signature lives in its query string
/// and expires at `expires_unix`. No secret value is ever returned.
#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub(crate) struct ContentUrlResponse {
    pub url: String,
    pub expires_unix: u64,
}

fn error(status: StatusCode, message: &str, code: &str) -> Response {
    (status, Json(OpenAiError::new(status.as_u16(), message.to_owned(), Some(code.to_owned()))))
        .into_response()
}

/// Mint a signed relative URL after checking the job exists. HMAC keying and
/// expiry come from runtime config; the secret VALUE is read from its env NAME
/// at request time (rotation needs no restart, and it is never logged/stored).
#[utoipa::path(
    post,
    path = "/v1/content-urls",
    request_body = ContentUrlRequest,
    responses(
        (status = 200, description = "Short-lived signed relative content URL", body = ContentUrlResponse),
        (status = 400, description = "Malformed job id or content name", body = OpenAiError),
        (status = 404, description = "No such job", body = OpenAiError),
        (status = 503, description = "Signed URLs unavailable", body = OpenAiError)
    )
)]
pub(crate) async fn post_content_urls(Json(req): Json<ContentUrlRequest>) -> Response {
    let id = match super::job_content::validate_id(&req.job_id) {
        Ok(id) => id.to_owned(),
        Err(_) => {
            return error(
                StatusCode::BAD_REQUEST,
                "job id must be a 26-character Crockford ULID",
                "invalid_job_id",
            )
        }
    };
    let content = if req.content_name == signed_urls::BUNDLE_CONTENT {
        signed_urls::BUNDLE_CONTENT
    } else {
        match super::job_content::validate_content_name(&req.content_name) {
            Ok(()) => req.content_name.as_str(),
            Err(_) => {
                return error(
                    StatusCode::BAD_REQUEST,
                    "unknown job content name",
                    "invalid_job_content_name",
                )
            }
        }
    };
    // Registry liveness only: no bundle file read here. A finished job's URL
    // stops working when its bundle is removed, like any other content URL.
    if !jobs_registry::global().list().iter().any(|job| job.id == id) {
        return error(StatusCode::NOT_FOUND, "no such job", "job_not_found");
    }
    let (enabled, ttl, env_name) = runtime_config::signed_url_facts();
    if !enabled {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "signed URLs are disabled",
            "signed_urls_disabled",
        );
    }
    let Some(secret) = middleware::env_var(&env_name).filter(|value| !value.is_empty()) else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "signed URL secret is unavailable",
            "signed_url_unavailable",
        );
    };
    let expires_unix = signed_urls::expires_after(ttl);
    let path = if content == signed_urls::BUNDLE_CONTENT {
        format!("/v1/jobs/{id}/bundle.zip")
    } else {
        format!("/v1/jobs/{id}/content/{content}")
    };
    let query = signed_urls::signed_query(
        secret.as_bytes(),
        &id,
        content,
        expires_unix,
        signed_urls::READ_SCOPE,
    );
    (
        StatusCode::OK,
        Json(ContentUrlResponse {
            url: format!("{path}?{query}"),
            expires_unix,
        }),
    )
        .into_response()
}
