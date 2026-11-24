//! Read-only generation history over finished job bundles.
//!
//! Finished bundles live at `<data_dir>/jobs/<ulid>/` with a `final_00.json`
//! sidecar. There is no job-id to bundle index and no `manifest.json` on
//! image bundles, so this module reads the sidecar directly; it never calls
//! the content routes and never mutates job state. File access runs on the
//! blocking pool, skips symlinks, and caps both sidecar size and the number
//! of bundle directories scanned.

use std::fs;
use std::io::Read as _;
use std::path::Path;

use axum::{
    Json,
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::{GenerationRecord, GenerationRecordList};

use super::AppConfig;
use super::job_content;
use super::openai_constraints::openai_error_response;

/// Page size when the client sends no `limit`.
const DEFAULT_LIMIT: usize = 20;
/// Largest admitted page size; over-cap values are rejected, not clamped.
const MAX_LIMIT: usize = 100;
/// Largest sidecar admitted from a bundle directory.
const MAX_SIDECAR_BYTES: u64 = 256 * 1024;
/// Largest number of bundle directories considered by a list request.
const MAX_SCAN_DIRS: usize = 4096;
const SIDECAR: &str = "final_00.json";

fn default_limit() -> usize {
    DEFAULT_LIMIT
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct RecordQuery {
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

/// Extract the acceptance fields from one bundle sidecar. Missing, malformed
/// or oversized sidecars are skipped: GC can leave partial bundles.
fn read_record(dir: &Path) -> Option<GenerationRecord> {
    let path = dir.join(SIDECAR);
    let meta = fs::symlink_metadata(&path).ok()?;
    if !meta.is_file() || meta.len() > MAX_SIDECAR_BYTES {
        return None;
    }
    let mut bytes = Vec::new();
    fs::File::open(&path)
        .ok()?
        .take(MAX_SIDECAR_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_SIDECAR_BYTES {
        return None;
    }
    let sidecar: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    Some(GenerationRecord {
        id: dir.file_name()?.to_str()?.to_owned(),
        prompt: sidecar.get("prompt")?.as_str()?.to_owned(),
        seed: sidecar.get("seed")?.as_u64()?,
        model_slug: sidecar.get("model_slug")?.as_str()?.to_owned(),
        model_sha256: sidecar["model_sha256"].as_str().map(str::to_owned),
        engine_version: sidecar["engine_version"].as_str().map(str::to_owned),
        inputs_sha256: sidecar.get("inputs_sha256")?.as_str()?.to_owned(),
    })
}

fn empty_page() -> GenerationRecordList {
    GenerationRecordList {
        object: "list".to_owned(),
        data: Vec::new(),
        has_more: false,
    }
}

/// Synchronous filesystem walk; called only from `spawn_blocking`.
fn list_records(root: &Path, limit: usize, offset: usize) -> Result<GenerationRecordList, String> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(empty_page()),
        Err(err) => return Err(err.to_string()),
    };
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| err.to_string())?;
        // `DirEntry::file_type` does not follow symlinks.
        if !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if job_content::validate_id(&name).is_err() {
            continue;
        }
        names.push(name);
    }
    // ULIDs sort lexicographically by creation time, so descending order is
    // newest first without consulting file mtimes (which GC can rewrite).
    names.sort_unstable_by(|a, b| b.cmp(a));
    // ponytail: at most 4096 newest bundles are reachable through paging; add
    // an on-disk index if retention grows past that.
    names.truncate(MAX_SCAN_DIRS);
    let has_more = names.len() > offset.saturating_add(limit);
    let data = names
        .into_iter()
        .skip(offset)
        .take(limit)
        .filter_map(|name| read_record(&root.join(name)))
        .collect();
    Ok(GenerationRecordList {
        object: "list".to_owned(),
        data,
        has_more,
    })
}

fn bad_request(code: &'static str, message: &str) -> Response {
    openai_error_response(StatusCode::BAD_REQUEST, message.to_owned(), Some(code))
}

fn not_found() -> Response {
    openai_error_response(
        StatusCode::NOT_FOUND,
        "generation record not found".to_owned(),
        Some("generation_record_not_found"),
    )
}

fn storage_error(detail: &str) -> Response {
    tracing::warn!(detail = %detail, "generation record storage failure");
    openai_error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "generation record storage failure".to_owned(),
        Some("storage_unavailable"),
    )
}

/// `GET /v1/jobs/records`: bounded, read-only page over finished bundles.
#[utoipa::path(
    get,
    path = "/v1/jobs/records",
    params(
        ("limit" = Option<usize>, Query, description = "Page size, 1..100 (default 20)"),
        ("offset" = Option<usize>, Query, description = "Records to skip from newest")
    ),
    responses(
        (status = 200, description = "Page of generation records", body = GenerationRecordList),
        (status = 400, description = "Limit outside 1..100"),
        (status = 500, description = "Storage failure")
    )
)]
pub(super) async fn get_job_records(
    State(config): State<AppConfig>,
    Query(query): Query<RecordQuery>,
) -> Response {
    if query.limit == 0 || query.limit > MAX_LIMIT {
        return bad_request("invalid_limit", "limit must be between 1 and 100");
    }
    let Some(root) = config.data_dir.map(|dir| dir.join("jobs")) else {
        return Json(empty_page()).into_response();
    };
    match tokio::task::spawn_blocking(move || list_records(&root, query.limit, query.offset)).await
    {
        Ok(Ok(page)) => Json(page).into_response(),
        Ok(Err(detail)) => storage_error(&detail),
        Err(_) => storage_error("blocking task failed"),
    }
}

/// `GET /v1/jobs/records/{id}`: one finished generation bundle by ULID.
#[utoipa::path(
    get,
    path = "/v1/jobs/records/{id}",
    params(("id" = String, Path, description = "Bundle ULID directory name")),
    responses(
        (status = 200, description = "One generation record", body = GenerationRecord),
        (status = 400, description = "Invalid ULID"),
        (status = 404, description = "No finished bundle with that id")
    )
)]
pub(super) async fn get_job_record(
    State(config): State<AppConfig>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if job_content::validate_id(&id).is_err() {
        return bad_request(
            "invalid_job_id",
            "job id must be a 26-character Crockford ULID",
        );
    }
    let Some(root) = config.data_dir.map(|dir| dir.join("jobs")) else {
        return not_found();
    };
    let dir = root.join(id);
    match tokio::task::spawn_blocking(move || read_record(&dir)).await {
        Ok(Some(record)) => Json(record).into_response(),
        _ => not_found(),
    }
}
