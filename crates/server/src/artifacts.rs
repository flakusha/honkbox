//! Binary artifact transport (TASK-binary-artifact-api): content-addressed
//! blob storage with chunked/resumable upload and ranged streaming download.
//!
//! Storage shape (under the resolved data dir's `artifacts/` root):
//! - `blobs/<aa>/<sha256>` — the artifact bytes, addressed by their own
//!   sha256 (`<aa>` = first two hex chars). Content addressing makes
//!   uploads idempotent at the storage layer: the same bytes always land at
//!   the same path, so a completed blob is immutable and deduplicated.
//! - `blobs/<aa>/<sha256>.json` — the served sidecar ([`ArtifactInfo`]):
//!   size, chunk count, declared content type.
//! - `uploads/<ulid>/` — one in-progress upload per directory: received
//!   chunks (`<index>-<digest8>.chunk`, renamed into place after fsync) and
//!   a small `manifest.json` chunk registry (`index → {sha256, bytes}`).
//!   The on-disk manifest is what makes an interrupted upload resumable
//!   across a server restart; the client re-sends only the chunks missing
//!   from the manifest it last saw.
//!
//! Temp discipline mirrors `downloads.rs`: bytes land in a sibling `.part`
//! file in the final directory tree, fsync, then `rename` — a reader never
//! sees a half-written blob, and a crash leaves the `.part` (or the intact
//! `uploads/` manifest) for the next attempt. Both `blobs/` and `uploads/`
//! live under one root, so every rename is same-filesystem.
//!
//! Memory contract: nothing is ever buffered whole. Each chunk (or a raw
//! single-shot body) streams through one fixed-size buffer that hashes and
//! writes as it drains; downloads stream from disk through [`BlobStream`]
//! (64 KiB heap per in-flight response regardless of blob size). Chunk
//! payloads travel as multipart binary parts or a raw body — never base64
//! — so the wire and the disk see the same bytes.
//!
//! Caps: [`CHUNK_MAX_BYTES`] bounds one multipart part (8 MiB),
//! `MAX_CHUNK_COUNT` bounds a session's declared chunks, and
//! `MAX_PARTS_PER_REQUEST` bounds parts per request; `max_artifact_bytes`
//! (config `[artifacts] max_artifact_bytes`, default
//! [`crate::config::DEFAULT_ARTIFACT_MAX_BYTES`]) bounds the total artifact.
//! Byte caps are reject-not-clamp with named 413 codes (`chunk_too_large` /
//! `artifact_too_large`); count/part bounds and active declared content types
//! are named 400s.
//!
//! Path safety: ids and upload ids are the ONLY client-controlled strings,
//! and both are validated to closed formats (64 hex chars / canonical ULID)
//! before they ever touch the filesystem — there is no client-supplied
//! path anywhere.
//!
//! Route admission is deliberately NOT wired into the GPU resource ledger:
//! uploads/downloads are pure disk IO with no model residency, so there is
//! no slot to acquire.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::{FromRequest as _, Multipart, Path as AxumPath, Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use honkbox_api_types::{ArtifactChunk, ArtifactInfo, ArtifactManifest};
use sha2::{Digest as _, Sha256};
use tokio::io::{AsyncRead, ReadBuf};
use tokio::sync::Mutex;
use tokio_stream::{Stream, StreamExt as _};

use crate::routes::AppConfig;

/// Maximum bytes for ONE multipart chunk (8 MiB). The total artifact cap is
/// the configurable bound; this one exists so a single part can neither
/// balloon a request body nor stall the hash/write loop on an unbounded
/// read. Raw single-shot bodies are bounded by the total cap only — a whole
/// artifact in one body is a legitimate upload shape.
pub const CHUNK_MAX_BYTES: u64 = 8 * 1024 * 1024;

/// Maximum chunks one upload may declare. Bounds the index space (and the
/// per-chunk finalize work); the total byte cap applies independently.
const MAX_CHUNK_COUNT: u32 = 10_000;
/// Maximum multipart parts accepted in one request, regardless of the
/// declared chunk count (chunks may arrive across several requests).
const MAX_PARTS_PER_REQUEST: u32 = 1_024;

/// Fixed streaming buffer (64 KiB): several chunk writes per syscall batch,
/// small enough that dozens of concurrent transfers stay memory-trivial.
const STREAM_BUF: usize = 64 * 1024;

/// Age past which an upload directory not live in the registry is reclaimed
/// on the next chunked request. Deliberately generous: a client can idle an
/// hour and still resume.
const UPLOAD_SESSION_TTL: Duration = Duration::from_hours(1);

/// Header carrying the upload id of a chunked upload being resumed.
const UPLOAD_ID_HEADER: &str = "x-upload-id";
/// Header declaring the total number of chunks (required when starting a
/// chunked upload; must match on every resumed request).
const CHUNK_COUNT_HEADER: &str = "x-chunk-count";
/// Header recording the artifact's content type for later GETs (default
/// `application/octet-stream`).
const CONTENT_TYPE_HEADER: &str = "x-artifact-content-type";
/// Per-part header carrying the chunk index (0-based). Parts without it
/// fill the lowest unreceived index, in arrival order.
const CHUNK_INDEX_HEADER: &str = "x-chunk-index";

/// Declared content types that can execute script or markup if a browser
/// ever renders them. Refused at upload; GET also forces attachment +
/// nosniff, so no stored type can execute in the server origin regardless.
const ACTIVE_CONTENT_TYPES: &[&str] = &[
    "text/html",
    "text/xhtml",
    "application/xhtml+xml",
    "image/svg",
    "image/svg+xml",
    "text/javascript",
    "text/ecmascript",
    "text/jscript",
    "text/vbscript",
    "text/x-javascript",
    "application/javascript",
    "application/ecmascript",
    "application/x-javascript",
    "application/x-ecmascript",
    "application/x-vbscript",
    "text/xml",
    "application/xml",
    "application/xslt+xml",
    "application/xaml+xml",
    "application/x-shockwave-flash",
    "application/x-silverlight-app",
    "message/rfc822",
];

/// Why an artifact request was rejected; rendered as the OpenAI-style
/// envelope by [`ArtifactError::into_response`] (named codes per the ticket
/// contract, following the house envelope conventions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ArtifactError {
    /// Storage root unresolved (server has no data dir): 500.
    StorageUnavailable,
    /// Not a real artifact / unknown or malformed id: 404.
    ArtifactNotFound(String),
    /// Unknown or malformed upload id: 404.
    UploadNotFound(String),
    /// Artifact byte total over the configured cap: 413.
    ArtifactTooLarge(u64),
    /// One chunk over the per-chunk cap: 413.
    ChunkTooLarge(u64),
    /// Same chunk index re-sent with different bytes, or a conflicting
    /// mid-upload redeclaration: 409.
    UploadConflict(String),
    /// Chunk index not below the declared count: 400.
    InvalidChunkIndex(u32),
    /// Missing/invalid `X-Chunk-Count`: 400.
    InvalidChunkCount(String),
    /// Unusable `X-Artifact-Content-Type` value: 400.
    InvalidContentType(String),
    /// A multipart part carried zero bytes: 400.
    EmptyChunk(u32),
    /// More parts in one request than the bound allows: 400.
    TooManyParts(u32),
    /// Malformed, multi-range, or unsatisfiable `Range`: 416.
    InvalidRange(String),
    /// `If-Match` does not match the current `ETag`: 412.
    PreconditionFailed,
    /// Disk IO failed mid-stream: 500.
    Storage(String),
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageUnavailable => {
                f.write_str("artifact storage unavailable: server has no data dir")
            }
            Self::ArtifactNotFound(id) => write!(f, "unknown artifact {id:?}"),
            Self::UploadNotFound(id) => write!(f, "unknown upload id {id:?}"),
            Self::ArtifactTooLarge(cap) => {
                write!(
                    f,
                    "artifact exceeds the {cap} byte cap: rejected, not clamped"
                )
            }
            Self::ChunkTooLarge(cap) => write!(
                f,
                "chunk exceeds the {cap} byte per-chunk cap: rejected, not clamped"
            ),
            Self::UploadConflict(msg) => write!(f, "upload conflict: {msg}"),
            Self::InvalidChunkIndex(index) => {
                write!(f, "chunk index {index} is beyond the declared chunk count")
            }
            Self::InvalidChunkCount(msg)
            | Self::InvalidContentType(msg)
            | Self::InvalidRange(msg) => write!(f, "{msg}"),
            Self::EmptyChunk(index) => {
                write!(
                    f,
                    "chunk {index} carried zero bytes; empty parts are rejected"
                )
            }
            Self::TooManyParts(limit) => {
                write!(f, "multipart request carried more than {limit} parts")
            }
            Self::PreconditionFailed => f.write_str("If-Match does not match the current ETag"),
            Self::Storage(msg) => write!(f, "artifact storage failure: {msg}"),
        }
    }
}

impl ArtifactError {
    /// Named envelope code for the `error.code` field.
    fn code(&self) -> &'static str {
        match self {
            Self::StorageUnavailable | Self::Storage(_) => "storage_unavailable",
            Self::ArtifactNotFound(_) => "artifact_not_found",
            Self::UploadNotFound(_) => "upload_not_found",
            Self::ArtifactTooLarge(_) => "artifact_too_large",
            Self::ChunkTooLarge(_) => "chunk_too_large",
            Self::UploadConflict(_) => "upload_conflict",
            Self::InvalidChunkIndex(_) => "invalid_chunk_index",
            Self::InvalidChunkCount(_) => "invalid_chunk_count",
            Self::InvalidContentType(_) => "invalid_content_type",
            Self::EmptyChunk(_) => "empty_chunk",
            Self::TooManyParts(_) => "too_many_parts",
            Self::InvalidRange(_) => "invalid_range",
            Self::PreconditionFailed => "precondition_failed",
        }
    }

    /// HTTP status for the named code.
    fn status(&self) -> StatusCode {
        match self {
            Self::StorageUnavailable | Self::Storage(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::ArtifactNotFound(_) | Self::UploadNotFound(_) => StatusCode::NOT_FOUND,
            Self::ArtifactTooLarge(_) | Self::ChunkTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::UploadConflict(_) => StatusCode::CONFLICT,
            Self::InvalidChunkIndex(_)
            | Self::InvalidChunkCount(_)
            | Self::InvalidContentType(_)
            | Self::EmptyChunk(_)
            | Self::TooManyParts(_) => StatusCode::BAD_REQUEST,
            Self::InvalidRange(_) => StatusCode::RANGE_NOT_SATISFIABLE,
            Self::PreconditionFailed => StatusCode::PRECONDITION_FAILED,
        }
    }

    /// Render as the OpenAI-style error envelope.
    fn into_response(self) -> Response {
        let code = self.code();
        let status = self.status();
        let message = match &self {
            // Storage errors carry absolute <data_dir> paths: log the full
            // detail server-side, never echo it back to the client.
            Self::Storage(detail) => {
                tracing::warn!(detail = %detail, "artifact storage failure");
                "artifact storage failure".to_owned()
            }
            _ => self.to_string(),
        };
        crate::routes::openai_error_response(status, message, Some(code))
    }
}

/// Per-upload mutable state. Lives in [`Registry::uploads`] while the
/// upload is active and persists to `uploads/<id>/manifest.json` after
/// every accepted chunk, so an interrupted upload resumes even after a
/// server restart.
#[derive(Debug)]
struct UploadState {
    id: String,
    /// Total chunk count declared by the client; immutable once created.
    chunk_count: u32,
    /// Declared content type served by later GETs.
    content_type: String,
    /// Received chunk registry: index → digest + byte count.
    chunks: BTreeMap<u32, ChunkRecord>,
    /// Running byte total (the artifact cap checks against this).
    total_bytes: u64,
    /// Lowest unreceived index; untagged parts fill this first (O(1) per part).
    lowest_free: u32,
}

/// One received chunk's identity (content-addressed, size-accounted).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct ChunkRecord {
    sha256: String,
    bytes: u64,
}

/// Upload/blob storage. Cheap to share (`Arc` in `AppConfig`). Every live
/// upload serializes on its own `Arc<Mutex<UploadState>>`: chunk appends and
/// finalize are read-modify-write cycles over the manifest, so the lock IS
/// the correctness story for concurrent chunk posts. Unlike `downloads.rs`
/// it is deliberately held across finalize IO — the finalize `.part` name is
/// derived from the upload id and two racing finalizes would otherwise
/// collide on it.
pub struct Registry {
    /// `<data_dir>/artifacts` root, or `None` when the server has no data
    /// dir (both routes degrade honestly: uploads 500, downloads 404).
    root: Option<PathBuf>,
    /// Total artifact byte cap (`[artifacts] max_artifact_bytes`).
    max_artifact_bytes: u64,
    /// Live uploads by id (mirror of the on-disk manifests for resumption
    /// within this process; disk is the fallback after a restart).
    uploads: Mutex<HashMap<String, Arc<Mutex<UploadState>>>>,
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registry")
            .field("root", &self.root)
            .field("max_artifact_bytes", &self.max_artifact_bytes)
            .finish_non_exhaustive()
    }
}

impl Registry {
    /// Build the registry over `<data_dir>/artifacts`; `None` disables
    /// storage (routes degrade honestly).
    #[must_use]
    pub fn new(data_dir: Option<PathBuf>, max_artifact_bytes: u64) -> Self {
        Self {
            root: data_dir.map(|base| base.join("artifacts")),
            max_artifact_bytes,
            uploads: Mutex::new(HashMap::new()),
        }
    }

    /// Directory of one in-progress upload (`artifacts/uploads/<id>`).
    fn upload_dir(&self, id: &str) -> Option<PathBuf> {
        self.root.as_ref().map(|root| root.join("uploads").join(id))
    }

    /// The configured total artifact cap (the router body-limit sizing for
    /// the POST route).
    #[must_use]
    pub fn max_artifact_bytes(&self) -> u64 {
        self.max_artifact_bytes
    }

    /// Store raw `bytes` as a content-addressed artifact — the synchronous
    /// counterpart to the streaming upload routes, for callers that already
    /// hold the bytes in memory (e.g. generated images).
    ///
    /// The id is the lowercase sha256 of `bytes`. The blob and its sidecar are
    /// published under `blobs/<aa>/<sha256>` exactly as the upload path
    /// publishes them, so identical bytes dedup to a single immutable blob.
    /// Only the [`ArtifactInfo`] descriptor is returned — no filesystem path
    /// ever crosses this API.
    ///
    /// Returns [`ArtifactError::StorageUnavailable`] with no data dir and
    /// [`ArtifactError::ArtifactTooLarge`] past the configured cap.
    pub fn put_bytes(
        &self,
        bytes: &[u8],
        content_type: &str,
    ) -> Result<ArtifactInfo, ArtifactError> {
        let root = self
            .root
            .as_ref()
            .ok_or(ArtifactError::StorageUnavailable)?;
        if bytes.len() as u64 > self.max_artifact_bytes {
            return Err(ArtifactError::ArtifactTooLarge(self.max_artifact_bytes));
        }
        let sha256 = hex_lower(&Sha256::digest(bytes));
        let target = Self::blob_path(root, &sha256);

        let blobs = root.join("blobs");
        std::fs::create_dir_all(&blobs)
            .map_err(|err| ArtifactError::Storage(format!("create blobs dir: {err}")))?;
        // One token for both `.part` paths: parallel callers never share a
        // temp name, so a concurrent rename cannot steal another's part.
        let token = ulid::Ulid::new();
        let part = blobs.join(format!("{token}.part"));
        std::fs::write(&part, bytes)
            .map_err(|err| ArtifactError::Storage(format!("write {}: {err}", part.display())))?;
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|err| {
                ArtifactError::Storage(format!("create {}: {err}", parent.display()))
            })?;
        }
        // Same dedup rule as `publish_blob`: an existing blob is immutable,
        // so discard our `.part` instead of rewriting the target.
        if target.exists() {
            let _ = std::fs::remove_file(&part);
        } else {
            std::fs::rename(&part, &target).map_err(|err| {
                ArtifactError::Storage(format!("rename {}: {err}", part.display()))
            })?;
        }

        let info = ArtifactInfo {
            id: sha256.clone(),
            sha256,
            size_bytes: bytes.len() as u64,
            chunk_count: 1,
            content_type: content_type.to_owned(),
        };
        let sidecar = Self::sidecar_path(root, &info.sha256);
        let sidecar_part = sidecar.with_extension(format!("json.{token}.part"));
        let raw = serde_json::to_vec(&info).map_err(|err| {
            ArtifactError::Storage(format!("serialize {}: {err}", sidecar.display()))
        })?;
        std::fs::write(&sidecar_part, &raw).map_err(|err| {
            ArtifactError::Storage(format!("write {}: {err}", sidecar_part.display()))
        })?;
        std::fs::File::open(&sidecar_part)
            .and_then(|file| file.sync_all())
            .map_err(|err| {
                ArtifactError::Storage(format!("fsync {}: {err}", sidecar_part.display()))
            })?;
        std::fs::rename(&sidecar_part, &sidecar).map_err(|err| {
            ArtifactError::Storage(format!("rename {}: {err}", sidecar_part.display()))
        })?;
        Ok(info)
    }

    /// The stored blob for an artifact id (its sha256) with its size.
    /// `None` when the id is malformed or nothing is stored under it — the
    /// only way a job route can hand a file to an external engine without
    /// ever accepting a filesystem path from the caller.
    #[must_use]
    pub fn stored_blob(&self, id: &str) -> Option<(PathBuf, u64)> {
        let id = normalize_artifact_id(id)?;
        let root = self.root.as_ref()?;
        let path = Self::blob_path(root, &id);
        let size = std::fs::metadata(&path).ok()?.len();
        Some((path, size))
    }

    /// Content-addressed blob path for a digest (`blobs/<aa>/<sha256>`).
    fn blob_path(root: &Path, sha256: &str) -> PathBuf {
        root.join("blobs").join(&sha256[..2]).join(sha256)
    }

    /// Completed-artifact sidecar path (`blobs/<aa>/<sha256>.json`).
    fn sidecar_path(root: &Path, sha256: &str) -> PathBuf {
        let mut path = Self::blob_path(root, sha256);
        path.set_extension("json");
        path
    }

    /// Fetch (or restore from disk) the live upload for `id`; `Ok(None)`
    /// means unknown/malformed id → 404.
    async fn upload(&self, id: &str) -> Result<Option<Arc<Mutex<UploadState>>>, ArtifactError> {
        if ulid::Ulid::from_string(id).is_err() {
            return Ok(None);
        }
        let mut live = self.uploads.lock().await;
        if let Some(found) = live.get(id) {
            return Ok(Some(Arc::clone(found)));
        }
        let Some(dir) = self.upload_dir(id) else {
            return Ok(None);
        };
        let raw = match tokio::fs::read(dir.join("manifest.json")).await {
            Ok(raw) => raw,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(ArtifactError::Storage(format!("read manifest: {err}"))),
        };
        let saved: SavedManifest = serde_json::from_slice(&raw)
            .map_err(|err| ArtifactError::Storage(format!("corrupt upload manifest: {err}")))?;
        // Rebuild the free-index cursor once on resume (BTreeMap lookup per
        // slot); every subsequent append advances it incrementally.
        let lowest_free = (0..saved.chunk_count)
            .find(|index| !saved.chunks.contains_key(index))
            .unwrap_or(saved.chunk_count);
        let state = Arc::new(Mutex::new(UploadState {
            id: saved.upload_id,
            chunk_count: saved.chunk_count,
            content_type: saved.content_type,
            chunks: saved.chunks,
            total_bytes: saved.total_bytes,
            lowest_free,
        }));
        live.insert(id.to_owned(), Arc::clone(&state));
        Ok(Some(state))
    }

    /// Drop a finished upload from the live map (after its directory was
    /// removed).
    async fn forget(&self, id: &str) {
        self.uploads.lock().await.remove(id);
    }

    /// Lazily reclaim stale, non-live upload directories. O(directory
    /// entries), all IO failures ignored — housekeeping must never fail an
    /// upload. `protect` is the id being resumed right now: it may be old
    /// and disk-only after a restart, but the client still owns it.
    // ponytail: lazy TTL sweep ceiling; periodic reaper if upload volume grows.
    async fn sweep_abandoned_uploads(&self, protect: Option<&str>) {
        let Some(root) = self.root.as_ref() else {
            return;
        };
        let uploads = root.join("uploads");
        let Ok(mut entries) = tokio::fs::read_dir(&uploads).await else {
            return;
        };
        let live: HashSet<String> = self.uploads.lock().await.keys().cloned().collect();
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().into_owned();
            if protect == Some(name.as_str()) || live.contains(&name) {
                continue;
            }
            let Ok(kind) = entry.file_type().await else {
                continue;
            };
            if !kind.is_dir() {
                continue;
            }
            let stale = entry
                .metadata()
                .await
                .ok()
                .and_then(|meta| meta.modified().ok())
                .and_then(|mtime| mtime.elapsed().ok())
                .is_some_and(|age| age >= UPLOAD_SESSION_TTL);
            if stale {
                let _ = tokio::fs::remove_dir_all(entry.path()).await;
            }
        }
    }
}

/// On-disk shape of `uploads/<id>/manifest.json` ([`UploadState`] plus its
/// id, so a restored upload can be keyed under the right map entry).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SavedManifest {
    upload_id: String,
    chunk_count: u32,
    content_type: String,
    total_bytes: u64,
    chunks: BTreeMap<u32, ChunkRecord>,
}

impl From<&UploadState> for SavedManifest {
    fn from(state: &UploadState) -> Self {
        Self {
            upload_id: state.id.clone(),
            chunk_count: state.chunk_count,
            content_type: state.content_type.clone(),
            total_bytes: state.total_bytes,
            chunks: state.chunks.clone(),
        }
    }
}

/// Artifact ids are sha256 digests: exactly 64 hex chars (case-normalized).
/// This is the whole path-safety story for the GET route — anything else
/// (including `..`, separators, percent tricks) never reaches the
/// filesystem.
fn normalize_artifact_id(id: &str) -> Option<String> {
    let lowered = id.to_ascii_lowercase();
    (lowered.len() == 64 && lowered.bytes().all(|b| b.is_ascii_hexdigit())).then_some(lowered)
}

/// sha256 digest → 64-char lowercase hex.
pub(crate) fn hex_lower(digest: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Atomic tiny-JSON write (`.part` + fsync + rename), shared by the upload
/// manifest and the completed sidecar.
async fn write_json_atomic(
    path: &Path,
    value: &impl serde::Serialize,
) -> Result<(), ArtifactError> {
    let raw = serde_json::to_vec(value)
        .map_err(|err| ArtifactError::Storage(format!("serialize {}: {err}", path.display())))?;
    let part = path.with_extension("json.part");
    tokio::fs::write(&part, &raw)
        .await
        .map_err(|err| ArtifactError::Storage(format!("write {}: {err}", part.display())))?;
    sync_file(&part).await?;
    tokio::fs::rename(&part, path)
        .await
        .map_err(|err| ArtifactError::Storage(format!("rename {}: {err}", part.display())))?;
    Ok(())
}

/// fsync one file by path (downloads.rs discipline: durable bytes before
/// the rename makes them visible).
async fn sync_file(path: &Path) -> Result<(), ArtifactError> {
    let file = tokio::fs::File::open(path).await.map_err(|err| {
        ArtifactError::Storage(format!("open {} for fsync: {err}", path.display()))
    })?;
    file.sync_all()
        .await
        .map_err(|err| ArtifactError::Storage(format!("fsync {}: {err}", path.display())))
}

/// Identity of one streamed payload.
struct Streamed {
    sha256: String,
    bytes: u64,
}

/// Deletes a streamed `.part` on drop unless [`PartGuard::disarm`] was
/// called. Created at every `.part` creation site so cap breaches, IO errors,
/// and digest conflicts cannot strand bytes — whichever `?` fires.
struct PartGuard {
    path: PathBuf,
    armed: bool,
}

impl PartGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    /// The bytes now live at their final path (or were deliberately
    /// discarded); nothing to clean up.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PartGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Mapped body chunks: both the raw request stream and multipart fields
/// surface as `Result<Bytes, _>`, so errors are pre-stringified and both
/// callers share this type. Lifetime-generic because a multipart `Field`
/// borrows its parser.
type ChunkStream<'a> = Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send + 'a>>;

/// Stream `chunks` into `part_path`, hashing and capping as bytes flow —
/// one pass over caller-provided pieces, nothing buffered whole. Any cap
/// breach (or IO failure) deletes the `.part` before returning the named
/// error, so a rejected chunk leaves no debris.
async fn stream_to_file(
    mut chunks: ChunkStream<'_>,
    part_path: &Path,
    chunk_cap: Option<u64>,
    total_base: u64,
    total_cap: u64,
) -> Result<Streamed, ArtifactError> {
    use tokio::io::AsyncWriteExt as _;
    let mut file = tokio::fs::File::create(part_path)
        .await
        .map_err(|err| ArtifactError::Storage(format!("create {}: {err}", part_path.display())))?;
    let mut hasher = Sha256::new();
    let mut written: u64 = 0;
    let mut failure: Option<ArtifactError> = None;
    while let Some(item) = chunks.as_mut().next().await {
        let chunk = match item {
            Ok(chunk) => chunk,
            Err(err) => {
                failure = Some(ArtifactError::Storage(format!("request body read: {err}")));
                break;
            }
        };
        if let Some(cap) = chunk_cap
            && written.saturating_add(chunk.len() as u64) > cap
        {
            failure = Some(ArtifactError::ChunkTooLarge(cap));
            break;
        }
        if total_base
            .saturating_add(written)
            .saturating_add(chunk.len() as u64)
            > total_cap
        {
            failure = Some(ArtifactError::ArtifactTooLarge(total_cap));
            break;
        }
        hasher.update(&chunk);
        written += chunk.len() as u64;
        if let Err(err) = file.write_all(&chunk).await {
            failure = Some(ArtifactError::Storage(format!(
                "write {}: {err}",
                part_path.display()
            )));
            break;
        }
    }
    let _ = file.shutdown().await;
    if let Some(err) = failure {
        let _ = tokio::fs::remove_file(part_path).await;
        return Err(err);
    }
    let sha256 = hex_lower(&hasher.finalize());
    sync_file(part_path).await?;
    Ok(Streamed {
        sha256,
        bytes: written,
    })
}

/// Register an accepted chunk: dedupe/conflict-check against the manifest,
/// rename the `.part` into its final `<index>-<digest8>.chunk` name, and
/// persist the on-disk manifest (the resume record). `Ok(false)` = an
/// idempotent duplicate (identical bytes, no state change).
async fn accept_chunk(
    state: &mut UploadState,
    manifest_path: &Path,
    index: u32,
    record: ChunkRecord,
    part_path: &Path,
    final_path: &Path,
) -> Result<bool, ArtifactError> {
    if let Some(existing) = state.chunks.get(&index) {
        if *existing == record {
            // Idempotent re-POST: the caller's part guard discards the
            // duplicate `.part`; no state change.
            return Ok(false);
        }
        // The caller's part guard discards the streamed bytes on conflict.
        return Err(ArtifactError::UploadConflict(format!(
            "chunk {index} already received with a different digest"
        )));
    }
    tokio::fs::rename(part_path, final_path)
        .await
        .map_err(|err| ArtifactError::Storage(format!("rename {}: {err}", part_path.display())))?;
    let bytes = record.bytes;
    state.chunks.insert(index, record);
    state.total_bytes = state.total_bytes.saturating_add(bytes);
    while state.lowest_free < state.chunk_count && state.chunks.contains_key(&state.lowest_free) {
        state.lowest_free += 1;
    }
    write_json_atomic(manifest_path, &SavedManifest::from(&*state)).await?;
    Ok(true)
}

/// Concatenate the received chunks (in index order) into the
/// content-addressed blob, hashing as it streams; publish the sidecar; then
/// remove the upload directory. Called with the upload lock held (see
/// [`Registry`]).
async fn finalize_upload(
    reg: &Registry,
    state: &UploadState,
) -> Result<ArtifactInfo, ArtifactError> {
    debug_assert_eq!(
        state.chunks.values().map(|r| r.bytes).sum::<u64>(),
        state.total_bytes,
        "chunk registry byte accounting"
    );
    let Some(root) = reg.root.clone() else {
        return Err(ArtifactError::StorageUnavailable);
    };
    let dir = reg
        .upload_dir(&state.id)
        .ok_or(ArtifactError::StorageUnavailable)?;
    tokio::fs::create_dir_all(root.join("blobs"))
        .await
        .map_err(|err| ArtifactError::Storage(format!("create blobs dir: {err}")))?;
    // The `.part` lives IN the blobs tree: same filesystem as the rename
    // target (the downloads.rs discipline).
    let part = root.join("blobs").join(format!("{}.part", state.id));
    let mut file = tokio::fs::File::create(&part)
        .await
        .map_err(|err| ArtifactError::Storage(format!("create {}: {err}", part.display())))?;
    let sha256 = {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; STREAM_BUF];
        for (&index, record) in &state.chunks {
            let chunk_path = dir.join(format!(
                "{index}-{}.chunk",
                record.sha256.get(..8).unwrap_or("chunk")
            ));
            let mut chunk_file = tokio::fs::File::open(&chunk_path).await.map_err(|err| {
                ArtifactError::Storage(format!("open {}: {err}", chunk_path.display()))
            })?;
            loop {
                let n = chunk_file.read(&mut buf).await.map_err(|err| {
                    ArtifactError::Storage(format!("read {}: {err}", chunk_path.display()))
                })?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
                file.write_all(&buf[..n]).await.map_err(|err| {
                    ArtifactError::Storage(format!("write {}: {err}", part.display()))
                })?;
            }
        }
        file.flush()
            .await
            .map_err(|err| ArtifactError::Storage(format!("flush {}: {err}", part.display())))?;
        file.sync_all()
            .await
            .map_err(|err| ArtifactError::Storage(format!("fsync {}: {err}", part.display())))?;
        hex_lower(&hasher.finalize())
    };
    let target = Registry::blob_path(&root, &sha256);
    publish_blob(&root, &part, &target).await?;
    let info = ArtifactInfo {
        id: sha256.clone(),
        sha256,
        size_bytes: state.total_bytes,
        chunk_count: state.chunk_count,
        content_type: state.content_type.clone(),
    };
    write_json_atomic(&Registry::sidecar_path(&root, &info.sha256), &info).await?;
    let _ = tokio::fs::remove_dir_all(&dir).await;
    Ok(info)
}

/// Land a finalized `.part` at its content-addressed target. Identical
/// bytes always map to the same path, so an existing blob is deduplicated
/// (the new `.part` is discarded) rather than rewritten.
async fn publish_blob(_root: &Path, part: &Path, target: &Path) -> Result<(), ArtifactError> {
    if let Some(parent) = target.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|err| ArtifactError::Storage(format!("create {}: {err}", parent.display())))?;
    }
    if tokio::fs::try_exists(target).await.unwrap_or(false) {
        let _ = tokio::fs::remove_file(part).await;
    } else {
        tokio::fs::rename(part, target)
            .await
            .map_err(|err| ArtifactError::Storage(format!("rename {}: {err}", part.display())))?;
    }
    Ok(())
}

/// Wire the manifest DTO for the current upload state; `artifact` is set
/// exactly when the upload finalized (flipping `status` to `complete`).
fn manifest_dto(state: &UploadState, artifact: Option<ArtifactInfo>) -> ArtifactManifest {
    ArtifactManifest {
        upload_id: state.id.clone(),
        chunk_count: state.chunk_count,
        received: state.chunks.keys().copied().collect(),
        chunks: state
            .chunks
            .iter()
            .map(|(&index, record)| ArtifactChunk {
                index,
                sha256: record.sha256.clone(),
                bytes: record.bytes,
            })
            .collect(),
        status: if artifact.is_some() {
            "complete".to_owned()
        } else {
            "uploading".to_owned()
        },
        artifact,
    }
}

/// Single-shot raw-body upload: the whole body is the artifact. Streams to
/// a `.part` in the blobs tree, hashes along the way, renames into its
/// digest path, publishes the sidecar. Returns the artifact descriptor.
async fn store_single_shot(
    reg: &Registry,
    content_type: String,
    body: Body,
) -> Result<ArtifactInfo, ArtifactError> {
    let Some(root) = reg.root.clone() else {
        return Err(ArtifactError::StorageUnavailable);
    };
    tokio::fs::create_dir_all(root.join("blobs"))
        .await
        .map_err(|err| ArtifactError::Storage(format!("create blobs dir: {err}")))?;
    let token = ulid::Ulid::new().to_string();
    let part = root.join("blobs").join(format!("{token}.part"));
    let mut guard = PartGuard::new(part.clone());
    let chunks: ChunkStream = Box::pin(
        body.into_data_stream()
            .map(|item| item.map_err(|err| err.to_string())),
    );
    let streamed = stream_to_file(chunks, &part, None, 0, reg.max_artifact_bytes).await?;
    let target = Registry::blob_path(&root, &streamed.sha256);
    publish_blob(&root, &part, &target).await?;
    guard.disarm(); // renamed, or dedup-discarded by publish_blob
    let info = ArtifactInfo {
        id: streamed.sha256.clone(),
        sha256: streamed.sha256,
        size_bytes: streamed.bytes,
        chunk_count: 1,
        content_type,
    };
    write_json_atomic(&Registry::sidecar_path(&root, &info.sha256), &info).await?;
    Ok(info)
}

/// Declared content type: a single visible-ASCII `type/subtype` value
/// (commas/parameters rejected — the value is echoed into response headers
/// later); default `application/octet-stream`. Active content types are
/// refused here, and GET forces attachment + nosniff as a second layer.
fn declared_content_type(headers: &HeaderMap) -> Result<String, ArtifactError> {
    let Some(value) = headers.get(CONTENT_TYPE_HEADER) else {
        return Ok("application/octet-stream".to_owned());
    };
    let raw = value
        .to_str()
        .map_err(|_| ArtifactError::InvalidContentType("not visible ASCII".to_owned()))?
        .trim();
    if raw.is_empty()
        || raw.len() > 100
        || !raw.contains('/')
        || raw.contains(',')
        || raw.contains(';')
    {
        return Err(ArtifactError::InvalidContentType(format!(
            "unusable content type {raw:?}: want a single type/subtype value"
        )));
    }
    let lowered = raw.to_ascii_lowercase();
    if ACTIVE_CONTENT_TYPES.contains(&lowered.as_str())
        || (lowered.starts_with("text/") && lowered.ends_with("+xml"))
    {
        return Err(ArtifactError::InvalidContentType(format!(
            "active content type {raw:?} is not allowed for artifacts"
        )));
    }
    Ok(raw.to_owned())
}

/// Parse `X-Chunk-Count`: required when starting a chunked upload, in
/// `1..=MAX_CHUNK_COUNT`. The bound keeps index space and finalize work finite.
fn declared_chunk_count(headers: &HeaderMap) -> Result<u32, ArtifactError> {
    let raw = headers
        .get(CHUNK_COUNT_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|raw| !raw.is_empty())
        .ok_or_else(|| {
            ArtifactError::InvalidChunkCount(
                "chunked uploads require X-Chunk-Count: the total number of chunks".to_owned(),
            )
        })?;
    raw.parse::<u32>()
        .ok()
        .filter(|&count| (1..=MAX_CHUNK_COUNT).contains(&count))
        .ok_or_else(|| {
            ArtifactError::InvalidChunkCount(format!(
                "X-Chunk-Count {raw:?} is not an integer in 1..={MAX_CHUNK_COUNT}"
            ))
        })
}

/// `POST /v1/artifacts`: upload one binary artifact.
///
/// Two modes on one route:
///
/// **Single-shot** — any non-multipart body (usually
/// `application/octet-stream`). The whole body is the artifact; it streams
/// to disk (never buffered) and is rejected with a named 413
/// `artifact_too_large` past the configured cap. Returns `201` with the
/// [`ArtifactInfo`] descriptor — `id` IS the content sha256, and it is the
/// `{id}` of the bytes route.
///
/// **Chunked/resumable** — `multipart/form-data`. Declare `X-Chunk-Count`
/// on the first request; each form part is one binary chunk, optionally
/// tagged with an `X-Chunk-Index` header (parts without it fill the lowest
/// unreceived index, so chunks may arrive in any order). Every response
/// carries the current [`ArtifactManifest`] (`status: "uploading"` with the
/// received-chunk registry). To resume an interrupted upload, re-send ONLY
/// the missing chunks with `X-Upload-Id: <upload_id>` set (the count must
/// match); a duplicate chunk re-POST with identical bytes is idempotent
/// (`200`, no state change), while different bytes for a received index are
/// `409 upload_conflict`. The request delivering the last missing chunk
/// finalizes in-line: its manifest flips to `status: "complete"` with the
/// `artifact` descriptor (content sha256, size, chunk count, id). Chunks
/// are capped at 8 MiB each (named 413 `chunk_too_large`), the artifact
/// total at the configured cap (`artifact_too_large`).
///
/// Optional `X-Artifact-Content-Type` (both modes) records the content type
/// served by `GET /v1/artifacts/{id}/bytes` (default
/// `application/octet-stream`). Active types (HTML, SVG, JavaScript, XML
/// markup) and parameterised values are rejected. GET always forces
/// `Content-Disposition: attachment` and `X-Content-Type-Options: nosniff`.
#[utoipa::path(
    post,
    path = "/v1/artifacts",
    request_body(content = String, description = "raw artifact bytes (single-shot), or multipart/form-data where each part is one binary chunk"),
    responses(
        (status = 201, description = "Single-shot upload stored", body = ArtifactInfo),
        (status = 200, description = "Chunk(s) accepted; current upload manifest (complete when the final chunk landed)", body = ArtifactManifest),
        (status = 400, description = "Bad chunk count/index or unusable content type", body = honkbox_api_types::OpenAiError),
        (status = 404, description = "Unknown X-Upload-Id", body = honkbox_api_types::OpenAiError),
        (status = 409, description = "Chunk index re-sent with different bytes, or chunk count changed", body = honkbox_api_types::OpenAiError),
        (status = 413, description = "Chunk over 8 MiB (chunk_too_large) or artifact over the configured total cap (artifact_too_large)", body = honkbox_api_types::OpenAiError),
        (status = 500, description = "No data dir (storage unavailable) or disk failure", body = honkbox_api_types::OpenAiError)
    )
)]
pub async fn post_artifact(State(config): State<AppConfig>, request: Request) -> Response {
    let reg = &config.artifacts;
    let is_multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|kind| kind.trim().eq_ignore_ascii_case("multipart/form-data"));
    if is_multipart {
        return match store_multipart(reg, request).await {
            Ok(response) => response,
            Err(err) => err.into_response(),
        };
    }
    let content_type = match declared_content_type(request.headers()) {
        Ok(content_type) => content_type,
        Err(err) => return err.into_response(),
    };
    // Content-Length pre-check: reject oversized single-shots BEFORE
    // streaming when the client declared the size (a named 413 instead of
    // the framework's bare one).
    if let Some(length) = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        && length > reg.max_artifact_bytes
    {
        return ArtifactError::ArtifactTooLarge(reg.max_artifact_bytes).into_response();
    }
    match store_single_shot(reg, content_type, request.into_body()).await {
        Ok(info) => (StatusCode::CREATED, Json(info)).into_response(),
        Err(err) => err.into_response(),
    }
}

/// Create a fresh upload session (directory + on-disk manifest + live map
/// entry).
async fn create_upload_session(
    reg: &Registry,
    chunk_count: u32,
    content_type: String,
) -> Result<Arc<Mutex<UploadState>>, ArtifactError> {
    let id = ulid::Ulid::new().to_string();
    let dir = reg
        .upload_dir(&id)
        .ok_or(ArtifactError::StorageUnavailable)?;
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|err| ArtifactError::Storage(format!("create {}: {err}", dir.display())))?;
    let fresh = SavedManifest {
        upload_id: id.clone(),
        chunk_count,
        content_type: content_type.clone(),
        total_bytes: 0,
        chunks: BTreeMap::new(),
    };
    write_json_atomic(&dir.join("manifest.json"), &fresh).await?;
    let state = Arc::new(Mutex::new(UploadState {
        id: id.clone(),
        chunk_count,
        content_type,
        chunks: BTreeMap::new(),
        total_bytes: 0,
        lowest_free: 0,
    }));
    reg.uploads.lock().await.insert(id, Arc::clone(&state));
    Ok(state)
}

/// Resolve (or create) the upload session for a chunked request; everything
/// downstream serializes on the returned state's lock. The flag is true only
/// when THIS request created the session, so only then is cleanup on error
/// required — a resumed session must stay available for a later retry.
async fn resolve_upload_session(
    reg: &Registry,
    headers: &HeaderMap,
    content_type: String,
) -> Result<(Arc<Mutex<UploadState>>, bool), ArtifactError> {
    let resume_id = headers
        .get(UPLOAD_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|raw| !raw.is_empty());
    reg.sweep_abandoned_uploads(resume_id).await;
    if let Some(id) = resume_id {
        let state = reg
            .upload(id)
            .await?
            .ok_or_else(|| ArtifactError::UploadNotFound(id.to_owned()))?;
        Ok((state, false))
    } else {
        let chunk_count = declared_chunk_count(headers)?;
        let state = create_upload_session(reg, chunk_count, content_type).await?;
        Ok((state, true))
    }
}

/// A resumed request re-declaring a different count is a conflict, not a
/// silent redefinition.
fn check_declared_count(state: &UploadState, headers: &HeaderMap) -> Result<(), ArtifactError> {
    if let Some(raw) = headers
        .get(CHUNK_COUNT_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.trim().parse::<u32>().ok())
        && raw != state.chunk_count
    {
        return Err(ArtifactError::UploadConflict(format!(
            "X-Chunk-Count {raw} does not match the session's declared {}",
            state.chunk_count
        )));
    }
    Ok(())
}

/// Chunk index for one incoming part: the explicit `X-Chunk-Index` wins,
/// otherwise the lowest unreceived index — insertion order never matters.
fn chunk_index_for(state: &UploadState, explicit: Option<u32>) -> Result<u32, ArtifactError> {
    match explicit {
        Some(index) => Ok(index),
        None if state.lowest_free < state.chunk_count => Ok(state.lowest_free),
        None => Err(ArtifactError::UploadConflict(
            "all chunks received; no free index for an untagged part".to_owned(),
        )),
    }
}

/// Stream one multipart part to its `.part`, then register it (dedupe /
/// conflict check + rename + manifest persistence).
async fn store_one_chunk(
    reg: &Registry,
    state: &mut UploadState,
    dir: &Path,
    index: u32,
    field: axum::extract::multipart::Field<'_>,
) -> Result<(), ArtifactError> {
    let token = ulid::Ulid::new().to_string();
    let part_path = dir.join(format!("{index}-{token}.chunk.part"));
    let mut part = PartGuard::new(part_path.clone());
    // Per-chunk cap + remaining artifact budget; the stream aborts at
    // whichever trips first, naming the right cap.
    let chunks: ChunkStream<'_> = Box::pin(field.map(|item| item.map_err(|err| err.to_string())));
    let streamed = stream_to_file(
        chunks,
        &part_path,
        Some(CHUNK_MAX_BYTES),
        state.total_bytes,
        reg.max_artifact_bytes,
    )
    .await?;
    if streamed.bytes == 0 {
        return Err(ArtifactError::EmptyChunk(index));
    }
    let chunk_suffix = streamed.sha256.get(..8).unwrap_or("chunk").to_owned();
    let final_path = dir.join(format!("{index}-{chunk_suffix}.chunk"));
    let committed = accept_chunk(
        state,
        &dir.join("manifest.json"),
        index,
        ChunkRecord {
            sha256: streamed.sha256,
            bytes: streamed.bytes,
        },
        &part_path,
        &final_path,
    )
    .await?;
    if committed {
        // Renamed into place; the guard has nothing left to remove.
        part.disarm();
    }
    Ok(())
}

/// Chunked multipart upload; `Ok` is always a 200 manifest response.
async fn store_multipart(reg: &Registry, request: Request) -> Result<Response, ArtifactError> {
    let headers = request.headers().clone();
    let content_type = declared_content_type(&headers)?;
    let mut multipart = Multipart::from_request(request, &())
        .await
        .map_err(|err| ArtifactError::Storage(format!("multipart parse: {err}")))?;
    let (upload, created_here) = resolve_upload_session(reg, &headers, content_type).await?;
    let session_id = upload.lock().await.id.clone();

    let result = receive_chunks(reg, &headers, upload, &mut multipart).await;
    if result.is_err() && created_here {
        // A fresh session whose request failed was never handed to the
        // client, so nobody can resume it: drop its dir and live entry.
        if let Some(dir) = reg.upload_dir(&session_id) {
            let _ = tokio::fs::remove_dir_all(dir).await;
        }
        reg.forget(&session_id).await;
    }
    result
}

/// Consume every part of an already-resolved upload. Errors on a resumed
/// session leave it intact for a later retry; fresh-session cleanup is the
/// caller's job.
async fn receive_chunks(
    reg: &Registry,
    headers: &HeaderMap,
    upload: Arc<Mutex<UploadState>>,
    multipart: &mut Multipart,
) -> Result<Response, ArtifactError> {
    {
        let state = upload.lock().await;
        check_declared_count(&state, headers)?;
    }

    let mut last_manifest: Option<ArtifactManifest> = None;
    let mut parts = 0u32;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| ArtifactError::Storage(format!("multipart read: {err}")))?
    {
        parts += 1;
        if parts > MAX_PARTS_PER_REQUEST {
            return Err(ArtifactError::TooManyParts(MAX_PARTS_PER_REQUEST));
        }
        let explicit = field
            .headers()
            .get(CHUNK_INDEX_HEADER)
            .and_then(|value| value.to_str().ok())
            .and_then(|raw| raw.trim().parse::<u32>().ok());
        let mut state = upload.lock().await;
        let index = chunk_index_for(&state, explicit)?;
        if index >= state.chunk_count {
            return Err(ArtifactError::InvalidChunkIndex(index));
        }
        let dir = reg
            .upload_dir(&state.id)
            .ok_or(ArtifactError::StorageUnavailable)?;
        store_one_chunk(reg, &mut state, &dir, index, field).await?;
        if u32::try_from(state.chunks.len()) == Ok(state.chunk_count) {
            let info = finalize_upload(reg, &state).await?;
            let manifest = manifest_dto(&state, Some(info));
            let id = state.id.clone();
            drop(state);
            reg.forget(&id).await;
            return Ok((StatusCode::OK, Json(manifest)).into_response());
        }
        last_manifest = Some(manifest_dto(&state, None));
    }
    // No parts at all is a client bug, not a manifest to report.
    last_manifest
        .map(|manifest| (StatusCode::OK, Json(manifest)).into_response())
        .ok_or_else(|| {
            ArtifactError::InvalidChunkCount(
                "multipart body carried no chunks (no form parts)".to_owned(),
            )
        })
}

/// Read one stored sidecar (or fall back to metadata-only when the sidecar
/// was lost between the blob rename and its write).
async fn read_sidecar(root: &Path, id: &str, size: u64) -> ArtifactInfo {
    match tokio::fs::read(Registry::sidecar_path(root, id)).await {
        Ok(raw) => {
            serde_json::from_slice::<ArtifactInfo>(&raw).unwrap_or_else(|_| metadata_info(id, size))
        }
        Err(_) => metadata_info(id, size),
    }
}

/// Sidecar-less descriptor: default content type, single-chunk provenance.
fn metadata_info(id: &str, size: u64) -> ArtifactInfo {
    ArtifactInfo {
        id: id.to_owned(),
        sha256: id.to_owned(),
        size_bytes: size,
        chunk_count: 1,
        content_type: "application/octet-stream".to_owned(),
    }
}

/// `Range: bytes=a-b` (single range only) → inclusive `(start, end)` slice;
/// `None` when no range was asked. `Err` is the honest 416: bogus syntax,
/// multiple ranges, or an unsatisfiable range, each naming the artifact
/// size.
fn parse_range(value: &str, size: u64) -> Result<Option<(u64, u64)>, ArtifactError> {
    let invalid = |msg: String| ArtifactError::InvalidRange(msg);
    let raw = value.trim();
    let Some(spec) = raw.strip_prefix("bytes=") else {
        return Err(invalid(format!(
            "only `bytes=` ranges are supported, got {raw:?}"
        )));
    };
    if spec.contains(',') {
        return Err(invalid(
            "multi-range requests are not supported: send a single bytes=a-b range".to_owned(),
        ));
    }
    let spec = spec.trim();
    let unsatisfiable = || {
        ArtifactError::InvalidRange(format!(
            "range {spec:?} is unsatisfiable for a {size} byte artifact"
        ))
    };
    // Suffix form: the last N bytes.
    if let Some(suffix) = spec.strip_prefix('-') {
        let n: u64 = suffix
            .trim()
            .parse()
            .map_err(|_| invalid(format!("malformed range {spec:?}")))?;
        if n == 0 || size == 0 {
            return Err(unsatisfiable());
        }
        return Ok(Some((size.saturating_sub(n), size - 1)));
    }
    let (start_raw, end_raw) = spec
        .split_once('-')
        .ok_or_else(|| invalid(format!("malformed range {spec:?}: want bytes=a-b")))?;
    let start: u64 = start_raw
        .trim()
        .parse()
        .map_err(|_| invalid(format!("malformed range {spec:?}: bad start")))?;
    let end = match end_raw.trim() {
        "" => size.checked_sub(1).ok_or_else(unsatisfiable)?,
        raw => raw
            .trim()
            .parse::<u64>()
            .map_err(|_| invalid(format!("malformed range {spec:?}: bad end")))?,
    };
    if start > end {
        return Err(invalid(format!(
            "malformed range {spec:?}: start exceeds end"
        )));
    }
    if start >= size {
        return Err(unsatisfiable());
    }
    Ok(Some((start, end.min(size - 1))))
}

/// `If-Match` against the current strong `ETag`: `*` matches any existing
/// artifact, otherwise each listed validator compares by digest (weak
/// `W/` validators never match a strong `ETag`).
fn if_match_matches(headers: &HeaderMap, id: &str) -> bool {
    let Some(raw) = headers
        .get(header::IF_MATCH)
        .and_then(|value| value.to_str().ok())
    else {
        return true;
    };
    raw.split(',').any(|candidate| {
        let candidate = candidate.trim();
        // A weak validator never matches a strong ETag (RFC 9110 §13.1.2).
        candidate == "*" || candidate.trim_matches('"') == id
    })
}

/// `If-Range` with the `ETag`: only a match on the current digest applies the
/// `Range`; any other value (stale `ETag` or a date form) serves the full
/// representation (RFC 7233 §3.2).
fn if_range_matches(headers: &HeaderMap, id: &str) -> bool {
    headers
        .get(header::IF_RANGE)
        .and_then(|value| value.to_str().ok())
        .is_none_or(|raw| raw.trim().trim_matches('"') == id)
}

/// `GET /v1/artifacts/{id}/bytes` (HEAD behaves identically minus the
/// body): stream the stored artifact from disk.
///
/// Always answers `Accept-Ranges: bytes` and `ETag: "<sha256>"`. A single
/// `Range: bytes=a-b` request gets `206` + `Content-Range` + exactly the
/// slice; bogus/multi/unsatisfiable ranges are a named `416 invalid_range`
/// carrying `Content-Range: bytes */<size>`. `If-Match` with a stale `ETag`
/// is `412 precondition_failed`; `If-Range` with a stale `ETag` serves the
/// full `200` (RFC 7233). `{id}` is the content sha256 hex digest from the
/// upload manifest (`artifact.id`). Range responses leave this route
/// uncompressed — the router's compression predicate skips every response
/// carrying `Content-Range` — so range math and resume offsets stay honest.
#[utoipa::path(
    get,
    path = "/v1/artifacts/{id}/bytes",
    params(("id" = String, Path, description = "Artifact id (the content sha256 hex digest) from the upload manifest")),
    responses(
        (status = 200, description = "Full artifact bytes (binary body)", content_type = "application/octet-stream"),
        (status = 206, description = "Single byte-range slice (Content-Range included)", content_type = "application/octet-stream"),
        (status = 404, description = "Unknown artifact id", body = honkbox_api_types::OpenAiError),
        (status = 412, description = "If-Match does not match the current ETag", body = honkbox_api_types::OpenAiError),
        (status = 416, description = "Malformed, multi-, or unsatisfiable Range", body = honkbox_api_types::OpenAiError)
    )
)]
pub async fn get_artifact_bytes(
    State(config): State<AppConfig>,
    method: Method,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let Some(id) = normalize_artifact_id(&id) else {
        return ArtifactError::ArtifactNotFound(id).into_response();
    };
    let Some(root) = config.artifacts.root.clone() else {
        return ArtifactError::ArtifactNotFound(id).into_response();
    };
    let target = Registry::blob_path(&root, &id);
    let size = match tokio::fs::metadata(&target).await {
        Ok(meta) => meta.len(),
        Err(_) => return ArtifactError::ArtifactNotFound(id).into_response(),
    };
    let content_type = read_sidecar(&root, &id, size).await.content_type;
    let etag = format!("\"{id}\"");

    if !if_match_matches(&headers, &id) {
        return ArtifactError::PreconditionFailed.into_response();
    }

    let range = if headers.contains_key(header::RANGE) {
        let raw = headers
            .get(header::RANGE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        if if_range_matches(&headers, &id) {
            match parse_range(raw, size) {
                Ok(slice) => slice,
                Err(err) => {
                    let mut response = err.into_response();
                    if let Ok(value) = HeaderValue::from_str(&format!("bytes */{size}")) {
                        response.headers_mut().insert(header::CONTENT_RANGE, value);
                    }
                    return response;
                }
            }
        } else {
            None // stale If-Range: serve the full representation
        }
    } else {
        None
    };

    let is_head = method == Method::HEAD;
    let (file, range_headers) = if let Some((start, end)) = range {
        use tokio::io::AsyncSeekExt as _;
        let mut file = match tokio::fs::File::open(&target).await {
            Ok(file) => file,
            Err(err) => return ArtifactError::Storage(format!("open blob: {err}")).into_response(),
        };
        if let Err(err) = file.seek(std::io::SeekFrom::Start(start)).await {
            return ArtifactError::Storage(format!("seek blob: {err}")).into_response();
        }
        let length = end - start + 1;
        (file, Some((format!("bytes {start}-{end}/{size}"), length)))
    } else {
        let file = match tokio::fs::File::open(&target).await {
            Ok(file) => file,
            Err(err) => return ArtifactError::Storage(format!("open blob: {err}")).into_response(),
        };
        (file, None)
    };

    let length = range_headers.as_ref().map_or(size, |&(_, length)| length);
    let body = Body::from_stream(BlobStream::new(file, length));
    let mut response = Response::new(if is_head { Body::empty() } else { body });
    if range_headers.is_some() {
        *response.status_mut() = StatusCode::PARTIAL_CONTENT;
    }
    {
        let out = response.headers_mut();
        if let Ok(value) = HeaderValue::from_str(&content_type) {
            out.insert(header::CONTENT_TYPE, value);
        }
        out.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
        // Stored bytes must never execute in the server origin: force a
        // download and block sniffing even if a bad type slipped in.
        out.insert(
            header::CONTENT_DISPOSITION,
            HeaderValue::from_static("attachment"),
        );
        out.insert(
            "x-content-type-options",
            HeaderValue::from_static("nosniff"),
        );
        if let Ok(value) = HeaderValue::from_str(&etag) {
            out.insert(header::ETAG, value);
        }
        if let Ok(value) = length.to_string().parse() {
            out.insert(header::CONTENT_LENGTH, value);
        }
        if let Some((content_range, _)) = range_headers
            && let Ok(value) = HeaderValue::from_str(&content_range)
        {
            out.insert(header::CONTENT_RANGE, value);
        }
    }
    response
}

/// Disk → HTTP body stream: reads the open file through a fixed reusable
/// buffer with a byte budget (the range length or the full size). One
/// allocation up front; nothing accumulates, so a multi-GiB artifact costs
/// [`STREAM_BUF`] bytes of heap while streaming.
struct BlobStream {
    file: tokio::fs::File,
    buf: Vec<u8>,
    remaining: u64,
}

impl BlobStream {
    fn new(file: tokio::fs::File, length: u64) -> Self {
        Self {
            file,
            buf: vec![0u8; STREAM_BUF],
            remaining: length,
        }
    }
}

impl Stream for BlobStream {
    type Item = std::io::Result<Bytes>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.remaining == 0 {
            return std::task::Poll::Ready(None);
        }
        let mut read_buf = ReadBuf::new(&mut this.buf);
        match <tokio::fs::File as AsyncRead>::poll_read(Pin::new(&mut this.file), cx, &mut read_buf)
        {
            std::task::Poll::Pending => std::task::Poll::Pending,
            std::task::Poll::Ready(Ok(())) => {
                let filled = read_buf.filled().len();
                if filled == 0 {
                    this.remaining = 0;
                    return std::task::Poll::Ready(Some(Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "artifact blob ended before the promised length",
                    ))));
                }
                let take = usize::try_from(this.remaining)
                    .unwrap_or(usize::MAX)
                    .min(filled);
                this.remaining -= take as u64;
                std::task::Poll::Ready(Some(Ok(Bytes::copy_from_slice(&read_buf.filled()[..take]))))
            }
            std::task::Poll::Ready(Err(err)) => {
                this.remaining = 0;
                std::task::Poll::Ready(Some(Err(err)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_bytes_publishes_blob_and_dedups() {
        let base =
            std::env::temp_dir().join(format!("honkbox-artifacts-putbytes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let reg = Registry::new(Some(base.clone()), 1024 * 1024);

        let bytes = b"the quick brown fox jumps over the lazy dog";
        let info = reg.put_bytes(bytes, "text/plain").expect("store bytes");
        assert_eq!(info.id, info.sha256);
        assert_eq!(info.size_bytes, bytes.len() as u64);
        assert_eq!(info.chunk_count, 1);
        assert_eq!(info.content_type, "text/plain");

        // Read back through the existing read path.
        let (path, size) = reg.stored_blob(&info.id).expect("stored blob");
        assert_eq!(size, bytes.len() as u64);
        assert_eq!(std::fs::read(&path).expect("read blob"), bytes.as_slice());

        // Sidecar published alongside the blob.
        let sidecar = Registry::sidecar_path(&base.join("artifacts"), &info.sha256);
        assert!(sidecar.exists());

        // Identical bytes are idempotent: still exactly one blob on disk.
        let again = reg.put_bytes(bytes, "text/plain").expect("store again");
        assert_eq!(again.id, info.id);
        let shard = path.parent().expect("shard dir");
        let blob_files: Vec<_> = std::fs::read_dir(shard)
            .expect("read shard")
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy() == info.sha256.as_str())
            .collect();
        assert_eq!(blob_files.len(), 1, "identical bytes must leave one blob");

        let _ = std::fs::remove_dir_all(&base);
    }
}
