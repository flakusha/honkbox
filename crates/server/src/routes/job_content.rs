//! Job bundle downloads: `GET /v1/jobs/{id}/bundle.zip` and
//! `GET /v1/jobs/{id}/content/{name}`.
//!
//! The only client-controlled inputs are a 26-character ULID and (for the
//! single-file route) one of three fixed names. Both are validated before a
//! path is built, so no host path can enter a response or escape
//! `<data_dir>/jobs/<id>/`. Bundle files are read at the top level only:
//! nested directories (the recipe's `.staging` tree) and symlinks are
//! ignored, never traversed or followed. The single-file route uses
//! `symlink_metadata` for the same reason: a manifest name that is a link
//! to a file outside the bundle is a 404.
//!
//! The ZIP is rebuilt on each request from sorted names with fixed entry
//! timestamps, permissions and compression settings, so identical bundle
//! contents always produce identical bytes. No wall clock is consulted.
//! File reads and compression run on the blocking pool under the caps below.

use std::fmt;
use std::io::{Cursor, Read as _};
use std::path::{Path, PathBuf};

use axum::{
    body::Body,
    extract::{Path as AxumPath, State},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use tokio::io::AsyncReadExt as _;
use tokio_util::io::ReaderStream;
use zip::write::SimpleFileOptions;

use super::AppConfig;
use super::openai_constraints::openai_error_response;

/// Maximum number of top-level files admitted to a bundle ZIP.
const MAX_ZIP_ENTRIES: usize = 256;
/// Total uncompressed bytes admitted to a bundle ZIP.
const MAX_UNCOMPRESSED_BYTES: u64 = 512 * 1024 * 1024;
/// Total ZIP bytes admitted to a response.
const MAX_COMPRESSED_BYTES: u64 = 512 * 1024 * 1024;
/// Bounded single-file download (same order of magnitude as the bundle cap).
pub const MAX_CONTENT_BYTES: u64 = MAX_UNCOMPRESSED_BYTES;

/// The content route's closed allowlist and each name's content type.
const CONTENT_TYPES: &[(&str, &str)] = &[
    ("manifest.json", "application/json"),
    ("output.glb", "model/gltf-binary"),
    ("output_turntable.webp", "image/webp"),
];

/// Named rejection, rendered as an OpenAI-style envelope. Details that can
/// carry host paths are logged server-side and never echoed to the caller.
#[derive(Debug)]
enum JobContentError {
    /// `{id}` is not a 26-character Crockford ULID: 400.
    InvalidId,
    /// `{name}` is outside the allowlist: 400.
    InvalidName,
    /// No data dir, no job dir, no file, or an unreadable job dir: 404.
    NotFound,
    /// Entry count or byte cap breached: 413.
    TooLarge(u64),
    /// Corrupt or changing bundle, or a failure building the ZIP: 500.
    Storage(String),
}

impl fmt::Display for JobContentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidId => f.write_str("job id must be a 26-character Crockford ULID"),
            Self::InvalidName => f.write_str("unknown job content name"),
            Self::NotFound => f.write_str("job content not found"),
            Self::TooLarge(cap) => {
                write!(
                    f,
                    "job content exceeds the {cap} byte cap: rejected, not clamped"
                )
            }
            Self::Storage(_) => f.write_str("job content storage failure"),
        }
    }
}

impl JobContentError {
    fn code(&self) -> &'static str {
        match self {
            Self::InvalidId => "invalid_job_id",
            Self::InvalidName => "invalid_job_content_name",
            Self::NotFound => "job_content_not_found",
            Self::TooLarge(_) => "job_content_too_large",
            Self::Storage(_) => "storage_unavailable",
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            Self::InvalidId | Self::InvalidName => StatusCode::BAD_REQUEST,
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::TooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::Storage(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn into_response(self) -> Response {
        let status = self.status();
        let code = self.code();
        let message = match &self {
            Self::Storage(detail) => {
                tracing::warn!(detail = %detail, "job content storage failure");
                self.to_string()
            }
            _ => self.to_string(),
        };
        openai_error_response(status, message, Some(code))
    }
}

/// Validate before any filesystem join; a bad id never reaches the disk.
fn validate_id(id: &str) -> Result<&str, JobContentError> {
    if id.len() != 26 || ulid::Ulid::from_string(id).is_err() {
        return Err(JobContentError::InvalidId);
    }
    Ok(id)
}

/// The bundle directory for a validated id (no data dir → no bundle).
fn job_dir(config: &AppConfig, id: &str) -> Option<PathBuf> {
    Some(config.data_dir.as_ref()?.join("jobs").join(id))
}

/// Fixed entry metadata: no wall clock, no host permissions, no ZIP64.
fn zip_options() -> SimpleFileOptions {
    SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .compression_level(Some(6))
        .last_modified_time(zip::DateTime::default())
        .unix_permissions(0o644)
        .large_file(false)
}

/// `GET /v1/jobs/{id}/bundle.zip`: rebuild a deterministic ZIP of the job
/// bundle. Missing/unreadable job directories are 404; count and byte caps
/// are reject-not-clamp 413s.
#[utoipa::path(
    get,
    path = "/v1/jobs/{id}/bundle.zip",
    params(("id" = String, Path, description = "26-character Crockford ULID of the job bundle")),
    responses(
        (status = 200, description = "Deterministic ZIP of the job bundle", content_type = "application/zip"),
        (status = 400, description = "Malformed job id", body = honkbox_api_types::OpenAiError),
        (status = 404, description = "No such job bundle", body = honkbox_api_types::OpenAiError),
        (status = 413, description = "Bundle exceeds the entry or byte caps", body = honkbox_api_types::OpenAiError)
    )
)]
pub(super) async fn get_job_bundle(
    State(config): State<AppConfig>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let id = match validate_id(&id) {
        Ok(id) => id.to_owned(),
        Err(err) => return err.into_response(),
    };
    let Some(dir) = job_dir(&config, &id) else {
        return JobContentError::NotFound.into_response();
    };
    let bytes = match tokio::task::spawn_blocking(move || build_zip(&dir)).await {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(err)) => return err.into_response(),
        Err(err) => {
            return JobContentError::Storage(format!("bundle task failed: {err}")).into_response();
        }
    };
    let mut response = (StatusCode::OK, Body::from(bytes)).into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/zip"),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("attachment; filename=\"job-bundle.zip\""),
    );
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    response
}

/// Collect top-level regular files, apply the caps, then write them in name
/// order with invariant metadata. A file that grew past its recorded size is
/// rejected rather than read past the cap.
fn build_zip(dir: &Path) -> Result<Vec<u8>, JobContentError> {
    let mut files = Vec::new();
    let mut total = 0u64;
    let entries = std::fs::read_dir(dir).map_err(|_| JobContentError::NotFound)?;
    for entry in entries {
        let entry = entry.map_err(|_| JobContentError::NotFound)?;
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        // Directories (including nested ones), symlinks and special files
        // are skipped: the archive is a flat snapshot of this directory.
        if !kind.is_file() {
            continue;
        }
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        let size = entry
            .metadata()
            .map_err(|_| JobContentError::NotFound)?
            .len();
        total = total
            .checked_add(size)
            .ok_or(JobContentError::TooLarge(MAX_UNCOMPRESSED_BYTES))?;
        if total > MAX_UNCOMPRESSED_BYTES || files.len() >= MAX_ZIP_ENTRIES {
            return Err(JobContentError::TooLarge(MAX_UNCOMPRESSED_BYTES));
        }
        files.push((name, entry.path(), size));
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let options = zip_options();
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, path, size) in files {
        writer
            .start_file(name, options)
            .map_err(|err| JobContentError::Storage(format!("zip entry: {err}")))?;
        let file = std::fs::File::open(&path).map_err(|_| JobContentError::NotFound)?;
        let copied = std::io::copy(&mut file.take(size.saturating_add(1)), &mut writer)
            .map_err(|err| JobContentError::Storage(format!("read bundle file: {err}")))?;
        if copied > size {
            return Err(JobContentError::TooLarge(MAX_UNCOMPRESSED_BYTES));
        }
    }
    let out = writer
        .finish()
        .map_err(|err| JobContentError::Storage(format!("finish zip: {err}")))?
        .into_inner();
    if out.len() as u64 > MAX_COMPRESSED_BYTES {
        return Err(JobContentError::TooLarge(MAX_COMPRESSED_BYTES));
    }
    Ok(out)
}

/// `GET /v1/jobs/{id}/content/{name}`: stream one allowlisted bundle file.
/// Missing files are 404, oversized files are 413; nothing is buffered whole.
#[utoipa::path(
    get,
    path = "/v1/jobs/{id}/content/{name}",
    params(
        ("id" = String, Path, description = "26-character Crockford ULID of the job bundle"),
        ("name" = String, Path, description = "One of manifest.json, output.glb, output_turntable.webp")
    ),
    responses(
        (status = 200, description = "The requested bundle file", content_type = "application/octet-stream"),
        (status = 400, description = "Malformed job id or disallowed name", body = honkbox_api_types::OpenAiError),
        (status = 404, description = "No such job bundle or file", body = honkbox_api_types::OpenAiError),
        (status = 413, description = "File exceeds the byte cap", body = honkbox_api_types::OpenAiError)
    )
)]
pub(super) async fn get_job_content(
    State(config): State<AppConfig>,
    AxumPath((id, name)): AxumPath<(String, String)>,
) -> Response {
    let id = match validate_id(&id) {
        Ok(id) => id.to_owned(),
        Err(err) => return err.into_response(),
    };
    let Some(content_type) = CONTENT_TYPES
        .iter()
        .find(|(allowed, _)| *allowed == name)
        .map(|(_, content_type)| *content_type)
    else {
        return JobContentError::InvalidName.into_response();
    };
    let Some(path) = job_dir(&config, &id).map(|dir| dir.join(&name)) else {
        return JobContentError::NotFound.into_response();
    };
    // symlink_metadata: an allowlisted name must be a regular file, never a
    // link to a path outside <data_dir>/jobs/<id>/.
    let meta = match tokio::fs::symlink_metadata(&path).await {
        Ok(meta) if meta.is_file() => meta,
        _ => return JobContentError::NotFound.into_response(),
    };
    if meta.len() > MAX_CONTENT_BYTES {
        return JobContentError::TooLarge(MAX_CONTENT_BYTES).into_response();
    }
    let Ok(file) = tokio::fs::File::open(&path).await else {
        return JobContentError::NotFound.into_response();
    };
    let stream = ReaderStream::new(file.take(meta.len()));
    let mut response = Response::new(Body::from_stream(stream));
    let headers = response.headers_mut();
    if let Ok(value) = HeaderValue::from_str(content_type) {
        headers.insert(header::CONTENT_TYPE, value);
    }
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    response
}
