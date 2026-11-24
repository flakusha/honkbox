//! Job bundle downloads: `GET /v1/jobs/{id}/bundle.zip` and
//! `GET /v1/jobs/{id}/content/{name}`.
//!
//! The only client-controlled inputs are a 26-character ULID and (for the
//! single-file route) a name from the job's `manifest.json` (the manifest
//! itself plus every `artifacts[].name`). Both are validated before a path is
//! built, so no host path can enter a response or escape
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
    extract::{Path as AxumPath, Query, State},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use tokio::io::AsyncReadExt as _;
use tokio_util::io::ReaderStream;
use zip::write::SimpleFileOptions;

use super::AppConfig;
use super::openai_constraints::openai_error_response;
use super::runtime_config;
use super::signed_urls;

/// Maximum number of top-level files admitted to a bundle ZIP.
const MAX_ZIP_ENTRIES: usize = 256;
/// Total uncompressed bytes admitted to a bundle ZIP.
const MAX_UNCOMPRESSED_BYTES: u64 = 512 * 1024 * 1024;
/// Total ZIP bytes admitted to a response.
const MAX_COMPRESSED_BYTES: u64 = 512 * 1024 * 1024;
/// Bounded single-file download (same order of magnitude as the bundle cap).
pub const MAX_CONTENT_BYTES: u64 = MAX_UNCOMPRESSED_BYTES;

/// A manifest-listed name must be a plain file name: no separators, no parent
/// traversal, no absolute path and a bounded length. `manifest.json` itself
/// passes the same check.
pub(super) fn validate_content_name(name: &str) -> Result<(), JobContentError> {
    if name.is_empty()
        || name.len() > 128
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        // The HMAC message is `|`-delimited: a name containing `|` lets a
        // signed token for one field be parsed as another field's value.
        || name.contains('|')
    {
        return Err(JobContentError::InvalidName);
    }
    Ok(())
}

/// Map a bundle file name to the MIME type it is served with. Registry
/// promotion in `blender_exec::commit` uses the same mapping.
pub(super) fn content_type_for_name(name: &str) -> &'static str {
    match name.rsplit_once('.').map_or("", |(_, ext)| ext) {
        "glb" => "model/gltf-binary",
        "webp" => "image/webp",
        "png" => "image/png",
        "json" => "application/json",
        _ => "application/octet-stream",
    }
}

/// Named rejection, rendered as an OpenAI-style envelope. Details that can
/// carry host paths are logged server-side and never echoed to the caller.
#[derive(Debug)]
pub(super) enum JobContentError {
    /// `{id}` is not a 26-character Crockford ULID: 400.
    InvalidId,
    /// Signed query params are missing, tampered with or expired: 401.
    InvalidSignedUrl,
    /// `{name}` is unsafe or not listed in the bundle manifest: 400.
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
            Self::InvalidSignedUrl => f.write_str("invalid or expired signed URL"),
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
            Self::InvalidSignedUrl => "invalid_signed_url",
            Self::InvalidId => "invalid_job_id",
            Self::InvalidName => "invalid_job_content_name",
            Self::NotFound => "job_content_not_found",
            Self::TooLarge(_) => "job_content_too_large",
            Self::Storage(_) => "storage_unavailable",
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            Self::InvalidSignedUrl => StatusCode::UNAUTHORIZED,
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
pub(super) fn validate_id(id: &str) -> Result<&str, JobContentError> {
    if id.len() != 26 || ulid::Ulid::from_string(id).is_err() {
        return Err(JobContentError::InvalidId);
    }
    Ok(id)
}

/// The bundle directory for a validated id (no data dir → no bundle).
fn job_dir(config: &AppConfig, id: &str) -> Option<PathBuf> {
    Some(config.data_dir.as_ref()?.join("jobs").join(id))
}

/// Verify additive signed params when present. Without them the route keeps
/// the exact bearer path enforced by the router middleware; with them the
/// bearer header is irrelevant and the HMAC decides. No secret value is logged.
fn check_signed(
    id: &str,
    content: &str,
    query: &signed_urls::SignedQuery,
) -> Result<(), JobContentError> {
    if !query.is_present() {
        return Ok(());
    }
    let (enabled, _, env_name) = runtime_config::signed_url_facts();
    let secret = super::middleware::env_var(&env_name).unwrap_or_default();
    if !enabled
        || secret.is_empty()
        || !signed_urls::verify_query(secret.as_bytes(), id, content, query)
    {
        return Err(JobContentError::InvalidSignedUrl);
    }
    Ok(())
}

/// Resolve a requested name against the bundle's manifest. `manifest.json`
/// is always allowed; every other name must appear in `artifacts[].name`.
/// Names are validated before any path is built, and symlinks are never
/// followed: only a regular file can be served. Runs on the blocking pool.
fn resolve_content(
    dir: &Path,
    name: &str,
) -> Result<(PathBuf, &'static str, u64), JobContentError> {
    validate_content_name(name)?;
    let manifest_path = dir.join("manifest.json");
    // Never follow a symlink (or read a non-regular file): a manifest that
    // points outside the bundle must not leak its bytes.
    let manifest_meta =
        std::fs::symlink_metadata(&manifest_path).map_err(|_| JobContentError::NotFound)?;
    if !manifest_meta.is_file() {
        return Err(JobContentError::NotFound);
    }
    let bytes = std::fs::read(&manifest_path).map_err(|_| JobContentError::NotFound)?;
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|err| JobContentError::Storage(format!("bad bundle manifest: {err}")))?;
    let mut allowed: Vec<(&str, &'static str)> = vec![("manifest.json", "application/json")];
    if let Some(artifacts) = manifest.get("artifacts").and_then(|value| value.as_array()) {
        for entry in artifacts {
            let Some(candidate) = entry.get("name").and_then(|value| value.as_str()) else {
                continue;
            };
            // One poisoned name must not make every other name unreachable.
            if validate_content_name(candidate).is_err() {
                continue;
            }
            allowed.push((candidate, content_type_for_name(candidate)));
        }
    }
    let Some((_, content_type)) = allowed.iter().find(|(allowed, _)| *allowed == name) else {
        return Err(JobContentError::InvalidName);
    };
    let path = dir.join(name);
    let meta = std::fs::symlink_metadata(&path).map_err(|_| JobContentError::NotFound)?;
    if !meta.is_file() {
        return Err(JobContentError::NotFound);
    }
    if meta.len() > MAX_CONTENT_BYTES {
        return Err(JobContentError::TooLarge(MAX_CONTENT_BYTES));
    }
    Ok((path, *content_type, meta.len()))
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
    params(
        ("id" = String, Path, description = "26-character Crockford ULID of the job bundle"),
        ("expires" = String, Query, description = "Unix expiry of a signed URL"),
        ("sig" = String, Query, description = "HMAC-SHA256 signature of this job/content tuple"),
        ("scope" = String, Query, description = "Scope bound into the signature")
    ),
    responses(
        (status = 200, description = "Deterministic ZIP of the job bundle", content_type = "application/zip"),
        (status = 401, description = "Invalid or expired signed URL", body = honkbox_api_types::OpenAiError),
        (status = 400, description = "Malformed job id", body = honkbox_api_types::OpenAiError),
        (status = 404, description = "No such job bundle", body = honkbox_api_types::OpenAiError),
        (status = 413, description = "Bundle exceeds the entry or byte caps", body = honkbox_api_types::OpenAiError)
    )
)]
pub(crate) async fn get_job_bundle(
    State(config): State<AppConfig>,
    AxumPath(id): AxumPath<String>,
    Query(signed): Query<signed_urls::SignedQuery>,
) -> Response {
    let id = match validate_id(&id) {
        Ok(id) => id.to_owned(),
        Err(err) => return err.into_response(),
    };
    if let Err(err) = check_signed(&id, signed_urls::BUNDLE_CONTENT, &signed) {
        return err.into_response();
    }
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

/// `GET /v1/jobs/{id}/content/{name}`: stream one manifest-listed bundle file.
/// Missing files are 404, oversized files are 413; nothing is buffered whole.
#[utoipa::path(
    get,
    path = "/v1/jobs/{id}/content/{name}",
    params(
        ("id" = String, Path, description = "26-character Crockford ULID of the job bundle"),
        ("name" = String, Path, description = "A file listed in the job's manifest.json: manifest.json or any artifacts[].name"),
        ("expires" = String, Query, description = "Unix expiry of a signed URL"),
        ("sig" = String, Query, description = "HMAC-SHA256 signature of this job/content tuple"),
        ("scope" = String, Query, description = "Scope bound into the signature")
    ),
    responses(
        (status = 200, description = "The requested bundle file", content_type = "application/octet-stream"),
        (status = 400, description = "Malformed job id or disallowed name", body = honkbox_api_types::OpenAiError),
        (status = 401, description = "Invalid or expired signed URL", body = honkbox_api_types::OpenAiError),
        (status = 404, description = "No such job bundle or file", body = honkbox_api_types::OpenAiError),
        (status = 413, description = "File exceeds the byte cap", body = honkbox_api_types::OpenAiError)
    )
)]
pub(crate) async fn get_job_content(
    State(config): State<AppConfig>,
    AxumPath((id, name)): AxumPath<(String, String)>,
    Query(signed): Query<signed_urls::SignedQuery>,
) -> Response {
    let id = match validate_id(&id) {
        Ok(id) => id.to_owned(),
        Err(err) => return err.into_response(),
    };
    if let Err(err) = check_signed(&id, &name, &signed) {
        return err.into_response();
    }
    let Some(dir) = job_dir(&config, &id) else {
        return JobContentError::NotFound.into_response();
    };
    // Name validation, the manifest allowlist and the symlink/size checks all
    // touch the filesystem: keep them off the async worker.
    let (path, content_type, len) =
        match tokio::task::spawn_blocking(move || resolve_content(&dir, &name)).await {
            Ok(Ok(resolved)) => resolved,
            Ok(Err(err)) => return err.into_response(),
            Err(err) => {
                return JobContentError::Storage(format!("content task failed: {err}"))
                    .into_response();
            }
        };
    let Ok(file) = tokio::fs::File::open(&path).await else {
        return JobContentError::NotFound.into_response();
    };
    let stream = ReaderStream::new(file.take(len));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_names_are_plain_files_with_known_types() {
        let too_long = "x".repeat(129);
        for bad in [
            "../outside",
            "a/b",
            "a\\b",
            "/etc/passwd",
            "C:\\x",
            "a|b", // `|` is the HMAC field delimiter
            "",
            too_long.as_str(),
        ] {
            assert!(
                validate_content_name(bad).is_err(),
                "{bad:?} must be rejected"
            );
        }
        for good in ["manifest.json", "custom.glb", "bake_01.webp", "map.png"] {
            validate_content_name(good).expect("a plain name is admitted");
        }
        assert_eq!(content_type_for_name("custom.glb"), "model/gltf-binary");
        assert_eq!(content_type_for_name("bake.webp"), "image/webp");
        assert_eq!(content_type_for_name("map.png"), "image/png");
        assert_eq!(content_type_for_name("manifest.json"), "application/json");
        assert_eq!(
            content_type_for_name("unknown.bin"),
            "application/octet-stream"
        );
    }
}
