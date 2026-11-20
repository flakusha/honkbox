//! `POST /v1/asset-packs/fetch`: fetch one curated CC0 pack by URL.
//!
//! Registration stays in `routes.rs` (the OpenAPI coverage gate scans that
//! file); this module holds validation and the refusal mapping only. The
//! QAL guard runs here at enqueue AND in the fetch write seam, so a refused
//! pack never produces a request, a directory, or a `.part`.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::OpenAiError;

use super::AppConfig;
use crate::catalog::asset_packs as packs;
use crate::catalog::asset_packs::{FetchJob, FetchRecord, FetchRequest};

/// Ids become file names, so keep them to one safe path segment.
const MAX_ID_BYTES: usize = 128;
/// Bounded source URL (no control characters, exact http/https prefix).
const MAX_URL_BYTES: usize = 2048;

fn error(status: StatusCode, message: &str, code: &str) -> Response {
    (
        status,
        Json(OpenAiError::new(
            status.as_u16(),
            message.to_owned(),
            Some(code.to_owned()),
        )),
    )
        .into_response()
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_ID_BYTES
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

fn valid_url(value: &str) -> bool {
    let Some(rest) = value
        .strip_prefix("http://")
        .or_else(|| value.strip_prefix("https://"))
    else {
        return false;
    };
    value.len() <= MAX_URL_BYTES
        && !rest.is_empty()
        && !rest.starts_with('/')
        && !rest
            .bytes()
            .any(|b| b.is_ascii_whitespace() || b < 0x20 || b == 0x7f)
}

/// Fetch a curated CC0 pack into `<data_dir>/asset-packs/<artifact_id>.bin`
/// and return the recorded sha256 + license row for bundle manifests.
#[utoipa::path(
    post,
    path = "/v1/asset-packs/fetch",
    request_body = FetchRequest,
    responses(
        (status = 200, description = "Fetched pack with sha256 and license row", body = FetchRecord),
        (status = 400, description = "Invalid pack id, URL, or artifact id", body = OpenAiError),
        (status = 412, description = "Pack license forbids redistribution", body = OpenAiError),
        (status = 500, description = "No data dir or local IO failure", body = OpenAiError),
        (status = 502, description = "Source failed or checksum mismatched", body = OpenAiError)
    )
)]
pub(super) async fn post_asset_pack_fetch(
    State(config): State<AppConfig>,
    Json(req): Json<FetchRequest>,
) -> Response {
    if !valid_id(&req.pack_id) {
        return error(
            StatusCode::BAD_REQUEST,
            &format!("pack_id must be 1..={MAX_ID_BYTES} characters of [A-Za-z0-9._-]"),
            "invalid_pack_id",
        );
    }
    if !valid_url(&req.source_url) {
        return error(
            StatusCode::BAD_REQUEST,
            &format!("source_url must be an http(s) URL of at most {MAX_URL_BYTES} bytes"),
            "invalid_source_url",
        );
    }
    if !valid_id(&req.artifact_id) {
        return error(
            StatusCode::BAD_REQUEST,
            &format!("artifact_id must be 1..={MAX_ID_BYTES} characters of [A-Za-z0-9._-]"),
            "invalid_artifact_id",
        );
    }

    // Enqueue-time gate: no HTTP request, no destination directory, no bytes.
    if let Err(refusal) = packs::guard_pack(&req.pack_id) {
        return error(refusal.status(), &refusal.message(), refusal.code());
    }

    let Some(root) = config.data_dir.clone() else {
        return error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "no data dir: asset packs cannot be stored",
            "no_data_dir",
        );
    };
    let job = FetchJob {
        pack_id: req.pack_id,
        source_url: req.source_url,
        artifact_id: req.artifact_id,
        root,
    };
    match packs::fetch(&job).await {
        Ok(record) => (StatusCode::OK, Json(record)).into_response(),
        Err(packs::FetchError::Refused(refusal)) => {
            error(refusal.status(), &refusal.message(), refusal.code())
        }
        Err(packs::FetchError::Status(code)) => error(
            StatusCode::BAD_GATEWAY,
            &format!("asset pack source returned HTTP {code}"),
            "asset_pack_source_status",
        ),
        Err(packs::FetchError::Checksum(message)) => error(
            StatusCode::BAD_GATEWAY,
            &message,
            "asset_pack_checksum_mismatch",
        ),
        Err(packs::FetchError::Transfer(message)) => error(
            StatusCode::BAD_GATEWAY,
            &message,
            "asset_pack_transfer_failed",
        ),
        Err(err) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &err.to_string(),
            "asset_pack_io_failed",
        ),
    }
}
