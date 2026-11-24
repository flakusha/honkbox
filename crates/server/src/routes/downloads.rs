//! Managed-download routes, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only the handlers and their private
//! helpers live here.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::{DownloadAccepted, DownloadList, DownloadRequest, DownloadState};

use super::AppConfig;

/// Map an enqueue error onto the HTTP contract: caller payload problems are
/// 400, server-side root/IO problems are 500 (same split as
/// `swap_rejection`'s unknown-vs-failed).
fn enqueue_rejection(err: &crate::downloads::EnqueueError) -> (StatusCode, String) {
    let message = err.to_string();
    let status = match err {
        crate::downloads::EnqueueError::NoModelsRoot
        | crate::downloads::EnqueueError::CreateDir(_) => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::BAD_REQUEST,
    };
    (status, message)
}

/// `POST /v1/downloads`: start one managed download into the configured
/// models root. `{repo, file, kind}` derives the `HuggingFace` resolve URL
/// (and enforces the LFS sha256 when the repo API publishes one); `url` is
/// the direct-source alternative. A second pull of the same target file
/// joins the in-flight download (same id); a target that already exists
/// resolves immediately as `done` + `already_present`.
#[utoipa::path(
    post,
    path = "/v1/downloads",
    request_body = DownloadRequest,
    responses(
        (status = 202, description = "Download accepted", body = DownloadAccepted),
        (status = 400, description = "Missing/ambiguous source, bad repo id, or unusable file name"),
        (status = 500, description = "No managed models root or target directory create failed")
    )
)]
pub(crate) async fn post_download(
    State(config): State<AppConfig>,
    Json(request): Json<DownloadRequest>,
) -> Response {
    match config.downloads.enqueue(request).await {
        Ok(accepted) => (StatusCode::ACCEPTED, Json(accepted)).into_response(),
        Err(err) => enqueue_rejection(&err).into_response(),
    }
}

/// `GET /v1/downloads`: every known download with its progress (poll
/// contract; no SSE).
#[utoipa::path(
    get,
    path = "/v1/downloads",
    responses(
        (status = 200, description = "Known downloads", body = DownloadList),
    )
)]
pub(crate) async fn list_downloads(State(config): State<AppConfig>) -> Json<DownloadList> {
    Json(DownloadList {
        downloads: config.downloads.states(),
    })
}

/// `GET /v1/downloads/{id}`: one download's progress; 404 for an unknown id.
#[utoipa::path(
    get,
    path = "/v1/downloads/{id}",
    params(("id" = String, Path, description = "Download id from the POST acceptance body")),
    responses(
        (status = 200, description = "Download state", body = DownloadState),
        (status = 404, description = "Unknown download id")
    )
)]
pub(crate) async fn get_download(
    State(config): State<AppConfig>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match config.downloads.state(&id) {
        Some(state) => Json(state).into_response(),
        None => (StatusCode::NOT_FOUND, format!("unknown download id {id:?}")).into_response(),
    }
}

/// `DELETE /v1/downloads/{id}`: cancel a download (204). The task stops at
/// the next chunk boundary, deletes the `.part` + meta sidecar, and lands
/// in `cancelled`; a queued download never starts. 409 once terminal.
#[utoipa::path(
    delete,
    path = "/v1/downloads/{id}",
    params(("id" = String, Path, description = "Download id from the POST acceptance body")),
    responses(
        (status = 204, description = "Cancellation triggered"),
        (status = 404, description = "Unknown download id"),
        (status = 409, description = "Download already finished")
    )
)]
pub(crate) async fn delete_download(
    State(config): State<AppConfig>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match config.downloads.cancel(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(crate::downloads::CancelError::Unknown(_)) => {
            (StatusCode::NOT_FOUND, format!("unknown download id {id:?}")).into_response()
        }
        Err(crate::downloads::CancelError::Terminal) => {
            (StatusCode::CONFLICT, "download already finished".to_owned()).into_response()
        }
    }
}
