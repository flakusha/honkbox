//! Managed model downloads (NEW-4 phase A): `POST /v1/downloads` supervisor
//! with sibling `.part` temp files, ranged resume, HF LFS sha256 checksums,
//! out-of-band cancel, and a 2-slot concurrency cap.
//!
//! Shape: one supervisor struct (`DownloadManager`: `Mutex<HashMap>` of
//! entries + `Semaphore(2)`) plus one spawned task per download — downloads
//! are independent IO with no shared state but the target path
//! (ReviewedNewItems-Approach §1), so no actor loop is needed. A second pull
//! of the same target file joins the in-flight id (single-flight).
//!
//! Temp discipline mirrors `jobs::write_atomic`: bytes land in
//! `<final>.part` inside the final directory (same-filesystem rename, and
//! the registry never classifies `.part` — its extension tables list real
//! weight extensions only), fsync, `rename`. A sidecar
//! `<final>.part.meta.json` `{url, etag, bytes_total, sha256?}` makes a
//! crash mid-download resumable only under a saved strong `ETag`: the next pull
//! resumes with `Range: bytes=<done>-` + `If-Range: <etag>`; a 206 must echo
//! that `ETag` and start at the saved offset, else the bad range is discarded
//! (sidecars deleted) and the pull retries cold. A 200 truncates + restarts;
//! without a resumable source the DTO advertises `resumable: false`.
//!
//! Checksum policy: enforced when the source provides one (HF LFS sha256,
//! fetched from the repo API for `repo`+`file` requests and persisted in
//! the meta sidecar across resumes), never invented otherwise — no local
//! hashing ceremony for plain files. A mismatch deletes the `.part` and
//! fails the download. Cancel deletes `.part` + meta; transient failures
//! keep both so the next pull resumes. ENOSPC surfaces as `failed` with the
//! `.part` kept (no free-space preflight; std has no statvfs).
//!
//! Metrics: `honkbox_download_bytes_total{kind}` +
//! `honkbox_download_failures_total{kind}` are owned here as `AtomicU64`
//! grids and appended to `/metrics` by `exposition` (the `metrics.rs`
//! module stays untouched — a peer worker owns it this round).
//!
//! Startup: stale `*.part` files are left in place (resume picks them up on
//! the next pull); only their count is logged. No automatic model eviction.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use honkbox_api_types::{
    DownloadAccepted, DownloadKind, DownloadRequest, DownloadState, DownloadStatus,
};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt as _;
use tokio::sync::{Notify, Semaphore};
use tokio_stream::StreamExt as _;
use tokio_util::sync::CancellationToken;

/// Concurrent download cap: two parallel fetches are generous for one disk.
const MAX_CONCURRENT: usize = 2;
/// Sibling temp extension of the in-progress file.
const PART_SUFFIX: &str = ".part";
/// Sibling sidecar recording `{url, etag, bytes_total, sha256?}` for resume.
const META_SUFFIX: &str = ".part.meta.json";
/// Sibling sidecar holding the VERIFIED digest of a completed download
/// (`<final>.sha256.json` `{"sha256": "…"}`). Written only when the source
/// published a checksum AND the streamed bytes matched it — the file never
/// records an unverified value. Survives completion (unlike
/// `.part.meta.json`) so generations can attribute provenance
/// (TASK-determinism-contract).
const ARTIFACT_META_SUFFIX: &str = ".sha256.json";
/// `HuggingFace` repo-metadata endpoint (`siblings[].lfs.sha256`).
const HF_API_BASE: &str = "https://huggingface.co/api/models";
/// `HuggingFace` raw-file resolve endpoint.
const HF_RESOLVE_BASE: &str = "https://huggingface.co";
/// Per-request timeout for the (small) HF metadata lookup only; body
/// transfers are unbounded — a 40 GiB pull must not die on a timer.
const METADATA_TIMEOUT: Duration = Duration::from_secs(30);
/// Download kinds, in metrics-slot order.
const KIND_LABELS: [&str; 4] = ["llm", "sd", "stt", "tts"];

/// `honkbox_download_bytes_total{kind}` grid (`llm`, `sd`, `stt`, `tts`).
static DOWNLOAD_BYTES: LazyLock<[AtomicU64; 4]> =
    LazyLock::new(|| [const { AtomicU64::new(0) }; 4]);
/// `honkbox_download_failures_total{kind}` grid, same slot order.
static DOWNLOAD_FAILURES: LazyLock<[AtomicU64; 4]> =
    LazyLock::new(|| [const { AtomicU64::new(0) }; 4]);

/// Metrics slot for a download kind (index into `KIND_LABELS`/grids).
fn kind_slot(kind: DownloadKind) -> usize {
    match kind {
        DownloadKind::Llm => 0,
        DownloadKind::Sd => 1,
        DownloadKind::Stt => 2,
        DownloadKind::Tts => 3,
    }
}

/// Prometheus exposition fragment for the download metric families; the
/// routes layer appends it to the `metrics::exposition` body.
pub(crate) fn exposition() -> String {
    let mut out = String::new();
    push_family(
        &mut out,
        "honkbox_download_bytes_total",
        "Bytes fetched into the managed models root, by kind.",
        "counter",
    );
    for (slot, label) in KIND_LABELS.iter().enumerate() {
        let _ = writeln!(
            out,
            "honkbox_download_bytes_total{{kind=\"{label}\"}} {}",
            DOWNLOAD_BYTES[slot].load(Ordering::Relaxed)
        );
    }
    push_family(
        &mut out,
        "honkbox_download_failures_total",
        "Failed downloads (transfer errors and checksum mismatches), by kind.",
        "counter",
    );
    for (slot, label) in KIND_LABELS.iter().enumerate() {
        let _ = writeln!(
            out,
            "honkbox_download_failures_total{{kind=\"{label}\"}} {}",
            DOWNLOAD_FAILURES[slot].load(Ordering::Relaxed)
        );
    }
    out
}

/// HELP + TYPE header lines for one metric family (mirrors `metrics.rs`).
fn push_family(out: &mut String, name: &str, help: &str, kind: &str) {
    let _ = write!(out, "# HELP {name} {help}\n# TYPE {name} {kind}\n");
}

/// Why a `POST /v1/downloads` was not accepted. `NoModelsRoot`/`CreateDir`
/// map to 500 (server config/IO), the rest to 400 (caller's payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnqueueError {
    /// No managed models root: the data dir was unresolved at startup.
    NoModelsRoot,
    /// Neither or both of `repo`+`file` and `url` were supplied.
    InvalidSource,
    /// `repo` is not a plausible `owner/name` `HuggingFace` id.
    BadRepo(String),
    /// The derived local file name is unusable (empty, separators, `..`).
    BadFileName(String),
    /// The target directory could not be created.
    CreateDir(String),
}

impl fmt::Display for EnqueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoModelsRoot => {
                write!(f, "no managed models root: data dir unresolved at startup")
            }
            Self::InvalidSource => {
                write!(f, "supply exactly one source: `repo`+`file`, or `url`")
            }
            Self::BadRepo(repo) => write!(f, "bad HuggingFace repo id {repo:?}: want owner/name"),
            Self::BadFileName(name) => write!(f, "unusable file name {name:?}"),
            Self::CreateDir(err) => write!(f, "cannot create target dir: {err}"),
        }
    }
}

impl Error for EnqueueError {}

/// Why `DELETE /v1/downloads/{id}` failed; `Unknown` maps to 404, `Terminal`
/// to 409.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CancelError {
    /// No download with this id (never started, or pre-restart entry).
    Unknown(String),
    /// The download already reached a terminal state.
    Terminal,
}

impl fmt::Display for CancelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "unknown download id {id:?}"),
            Self::Terminal => write!(f, "download already finished"),
        }
    }
}

impl Error for CancelError {}

/// Sidecar of an in-progress/aborted download (`<final>.part.meta.json`).
/// `sha256` extends the documented `{url, etag, bytes_total}` shape: the
/// checksum must survive restarts or resume would silently drop enforcement.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PartMeta {
    /// Source the `.part` bytes were fetched from (resume checks equality).
    url: String,
    /// Strong validator for `If-Range` on the resumed request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    etag: Option<String>,
    /// Total size the server reported (drives the DTO before any response).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bytes_total: Option<u64>,
    /// Expected sha256 when the source published one (HF LFS).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sha256: Option<String>,
}

/// Supervisor-side mutable record of one download (shared with its task via
/// `Arc<Mutex<_>>`; locks are held for field writes only, never across IO).
struct Entry {
    /// Lifecycle state; the handler may only enter `cancelled`, the task
    /// owns every other write (its terminal outcome always wins).
    state: DownloadStatus,
    /// Bytes committed to the `.part` so far.
    bytes_done: u64,
    /// Expected total, once known.
    bytes_total: Option<u64>,
    /// Source URL.
    source: String,
    /// Final target path.
    target: PathBuf,
    /// Failure reason (`failed` only).
    error: Option<String>,
    /// Ranged resume possible (strong `ETag` + `Accept-Ranges: bytes` seen).
    resumable: bool,
    /// `done` without fetching: the target file already existed.
    already_present: bool,
    /// Cooperative cancel handle; `DELETE` triggers it, the task observes it
    /// at the next chunk boundary.
    token: CancellationToken,
}

impl Entry {
    /// Wire DTO projection.
    fn dto(&self, id: &str) -> DownloadState {
        DownloadState {
            id: id.to_owned(),
            state: self.state,
            bytes_done: self.bytes_done,
            bytes_total: self.bytes_total,
            source: self.source.clone(),
            target_path: Some(self.target.display().to_string()),
            error: self.error.clone(),
            resumable: self.resumable,
            already_present: self.already_present,
        }
    }

    /// Terminal states never transition again (except the task overwriting
    /// a handler-set `cancelled` with its real outcome).
    fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            DownloadStatus::Done | DownloadStatus::Failed | DownloadStatus::Cancelled
        )
    }
}

/// Everything a spawned task needs to fetch one file.
struct Job {
    /// Download id (`dl_<ulid>`).
    id: String,
    /// Metrics label + managed subdir owner.
    kind: DownloadKind,
    /// Source URL.
    url: String,
    /// Final file path (`.part` + meta are its siblings).
    target: PathBuf,
    /// Expected sha256, when the source published one.
    expected_sha: Option<String>,
    /// Shared supervisor record.
    entry: Arc<Mutex<Entry>>,
}

/// Download supervisor: registry of entries, single-flight map, the
/// concurrency semaphore, and the rescan signal fired on commit.
pub struct DownloadManager {
    /// All known downloads (terminal ones stay for `GET` history).
    entries: Mutex<HashMap<String, Arc<Mutex<Entry>>>>,
    /// In-flight target path → id (single-flight per target file).
    in_flight: Mutex<HashMap<PathBuf, String>>,
    /// Global concurrency cap (2 permits).
    sem: Arc<Semaphore>,
    /// Shared HTTP client (rustls; direct connections, no proxy).
    client: reqwest::Client,
    /// Managed models root `<data_dir>/models` (`None`: starts are refused).
    models_root: Option<PathBuf>,
    /// Notified once per committed file; drives the post-download rescan.
    rescan: Arc<Notify>,
}

impl DownloadManager {
    /// Build a supervisor; `models_root` is the managed `<data_dir>/models`
    /// directory (`None` refuses starts). Fails only if the HTTP client
    /// cannot be constructed (TLS backend init).
    pub fn new(models_root: Option<PathBuf>) -> Result<Self, String> {
        // Direct connections only: no ambient proxy env (local-first box;
        // keeps tests hermetic against an ambient `HTTP_PROXY`).
        let client = reqwest::Client::builder()
            .user_agent(concat!("honkbox/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(30))
            .no_proxy()
            .build()
            .map_err(|err| format!("download HTTP client: {err}"))?;
        Ok(Self {
            entries: Mutex::new(HashMap::new()),
            in_flight: Mutex::new(HashMap::new()),
            sem: Arc::new(Semaphore::new(MAX_CONCURRENT)),
            client,
            models_root,
            rescan: Arc::new(Notify::new()),
        })
    }

    /// The rescan signal, fired after every committed file; wire it to the
    /// registry refresh loop so completed downloads appear in
    /// `GET /v1/models` with their normal slug.
    pub fn rescan_signal(&self) -> Arc<Notify> {
        Arc::clone(&self.rescan)
    }

    /// Accept a download request: validate, resolve the target under the
    /// managed root, look up the HF checksum when a repo file is requested,
    /// then enqueue (or join an in-flight twin). Returns the accepted id.
    pub async fn enqueue(
        self: &Arc<Self>,
        request: DownloadRequest,
    ) -> Result<DownloadAccepted, EnqueueError> {
        // Exactly one source form.
        let (repo, file) = match (&request.repo, &request.file, &request.url) {
            (Some(repo), Some(file), None) => (Some(repo), Some(file)),
            (None, None, Some(_)) => (None, None),
            _ => return Err(EnqueueError::InvalidSource),
        };
        let url = match (repo, file) {
            (Some(repo), Some(file)) => {
                Self::validate_repo(repo)?;
                // `?`/`#` would split the resolve URL (HF file names never
                // contain them); reject instead of mangling the source.
                if file.contains(['?', '#']) {
                    return Err(EnqueueError::BadFileName(file.clone()));
                }
                format!("{HF_RESOLVE_BASE}/{repo}/resolve/main/{file}")
            }
            (None, None) => request.url.clone().unwrap_or_default(),
            _ => return Err(EnqueueError::InvalidSource),
        };
        let name = Self::url_file_name(&url)?;
        let expected_sha = match (repo, file) {
            (Some(repo), Some(file)) => hf_expected_sha256(&self.client, repo, file).await,
            (None, None) => None,
            _ => return Err(EnqueueError::InvalidSource),
        };
        self.enqueue_planned(request.kind, url, &name, expected_sha)
    }

    /// Test hook: enqueue with an injected checksum (no HF API in tests;
    /// the integration suite compiles this module under `cfg(test)`).
    /// Unused inside the bin's own test build — only the integration suite
    /// calls it.
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn enqueue_with_sha(
        self: &Arc<Self>,
        kind: DownloadKind,
        url: String,
        expected_sha: Option<String>,
    ) -> Result<DownloadAccepted, EnqueueError> {
        let name = Self::url_file_name(&url)?;
        self.enqueue_planned(kind, url, &name, expected_sha)
    }

    /// Enqueue with a pre-planned source (split out so tests can inject a
    /// checksum without talking to the HF API).
    fn enqueue_planned(
        self: &Arc<Self>,
        kind: DownloadKind,
        url: String,
        name: &str,
        expected_sha: Option<String>,
    ) -> Result<DownloadAccepted, EnqueueError> {
        let dir = self.target_dir(kind)?;
        let target = dir.join(name);
        // Single-flight: join an identical in-flight download.
        {
            let in_flight = self
                .in_flight
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(id) = in_flight.get(&target) {
                let entries = self
                    .entries
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(entry) = entries.get(id) {
                    let guard = entry
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if !guard.is_terminal() {
                        return Ok(DownloadAccepted {
                            id: id.clone(),
                            state: guard.state,
                        });
                    }
                }
            }
        }
        // Target already present: done without fetching anything.
        if target.is_file() {
            let id = self.insert(Entry {
                state: DownloadStatus::Done,
                bytes_done: 0,
                bytes_total: None,
                source: url,
                target,
                error: None,
                resumable: false,
                already_present: true,
                token: CancellationToken::new(),
            });
            return Ok(DownloadAccepted {
                id,
                state: DownloadStatus::Done,
            });
        }
        // The final directory is created here (not by the task) so an
        // unusable root fails the POST instead of the download.
        if let Err(err) = std::fs::create_dir_all(&dir) {
            return Err(EnqueueError::CreateDir(err.to_string()));
        }
        let id = format!("dl_{}", ulid::Ulid::new());
        let entry = Arc::new(Mutex::new(Entry {
            state: DownloadStatus::Queued,
            bytes_done: 0,
            bytes_total: None,
            source: url.clone(),
            target: target.clone(),
            error: None,
            resumable: false,
            already_present: false,
            token: CancellationToken::new(),
        }));
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id.clone(), Arc::clone(&entry));
        self.in_flight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(target.clone(), id.clone());
        let job = Job {
            id: id.clone(),
            kind,
            url,
            target,
            expected_sha,
            entry,
        };
        // Detached for the download's lifetime (it outlives the POST that
        // started it); the supervisor registry keeps it reachable.
        tokio::spawn(run_download(Arc::clone(self), job));
        Ok(DownloadAccepted {
            id,
            state: DownloadStatus::Queued,
        })
    }

    /// Cancel a download: triggers its token; the task stops at the next
    /// chunk boundary, deletes `.part` + meta, and lands in `cancelled`.
    /// A `queued` download never starts. Errors once terminal (409).
    pub fn cancel(&self, id: &str) -> Result<(), CancelError> {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = entries
            .get(id)
            .ok_or_else(|| CancelError::Unknown(id.to_owned()))?;
        let mut guard = entry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.is_terminal() {
            return Err(CancelError::Terminal);
        }
        guard.token.cancel();
        guard.state = DownloadStatus::Cancelled;
        Ok(())
    }

    /// Snapshot of all downloads, sorted by id.
    pub fn states(&self) -> Vec<DownloadState> {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut out: Vec<_> = entries
            .iter()
            .map(|(id, entry)| {
                entry
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .dto(id)
            })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// Snapshot of one download; `None` for an unknown id.
    pub fn state(&self, id: &str) -> Option<DownloadState> {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries.get(id).map(|entry| {
            entry
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .dto(id)
        })
    }

    /// Insert a ready entry and return its fresh id.
    fn insert(&self, entry: Entry) -> String {
        let id = format!("dl_{}", ulid::Ulid::new());
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id.clone(), Arc::new(Mutex::new(entry)));
        id
    }

    /// Managed target directory for a kind. `sd` writes under
    /// `checkpoints/` so the registry classifies both `.safetensors` and
    /// `.gguf` SD weights (root-level `ComfyUI` `.gguf` files would not).
    fn target_dir(&self, kind: DownloadKind) -> Result<PathBuf, EnqueueError> {
        let root = self
            .models_root
            .as_ref()
            .ok_or(EnqueueError::NoModelsRoot)?;
        Ok(match kind {
            DownloadKind::Llm => root.join("llm"),
            DownloadKind::Sd => root.join("sd").join("checkpoints"),
            DownloadKind::Stt => root.join("stt"),
            DownloadKind::Tts => root.join("tts"),
        })
    }

    /// Plausible `HuggingFace` `owner/name` repo id (charset + no dot-leading
    /// components; the resolve URL is built from it unescaped).
    fn validate_repo(repo: &str) -> Result<(), EnqueueError> {
        let plausible = |part: &str| {
            !part.is_empty()
                && part.len() <= 96
                && !part.starts_with('.')
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        };
        match repo.split_once('/') {
            Some((owner, name)) if plausible(owner) && plausible(name) => Ok(()),
            _ => Err(EnqueueError::BadRepo(repo.to_owned())),
        }
    }

    /// Local file name from a URL: last path segment (query/fragment
    /// stripped), sanitized for the filesystem (percent-encoding is kept
    /// as-is — a valid, if ugly, file name).
    fn url_file_name(url: &str) -> Result<String, EnqueueError> {
        let path = url.split(['?', '#']).next().unwrap_or(url);
        let raw = path.rsplit('/').next().unwrap_or("");
        let name = raw.trim();
        let usable = !name.is_empty()
            && name != "."
            && name != ".."
            && name.len() <= 200
            && !name
                .chars()
                .any(|c| c == '/' || c == '\\' || c.is_control());
        if usable {
            Ok(name.to_owned())
        } else {
            Err(EnqueueError::BadFileName(url.to_owned()))
        }
    }
}

/// Fetch the HF repo metadata and return the file's LFS sha256 when the
/// API publishes one. Absent/unavailable metadata never fails the download
/// — the checksum policy is "enforce when the source provides one".
async fn hf_expected_sha256(client: &reqwest::Client, repo: &str, file: &str) -> Option<String> {
    let url = format!("{HF_API_BASE}/{repo}?blobs=true");
    let response = match client.get(url).timeout(METADATA_TIMEOUT).send().await {
        Ok(response) if response.status().is_success() => response,
        Ok(response) => {
            tracing::warn!(
                %repo,
                status = %response.status(),
                "HF metadata unavailable; downloading without checksum"
            );
            return None;
        }
        Err(err) => {
            tracing::warn!(%repo, %err, "HF metadata lookup failed; downloading without checksum");
            return None;
        }
    };
    let body: serde_json::Value = match response.json().await {
        Ok(body) => body,
        Err(err) => {
            tracing::warn!(%repo, %err, "HF metadata decode failed; downloading without checksum");
            return None;
        }
    };
    body.get("siblings")
        .and_then(serde_json::Value::as_array)
        .and_then(|siblings| {
            siblings
                .iter()
                .find(|s| s.get("rfilename").and_then(serde_json::Value::as_str) == Some(file))
        })
        .and_then(|s| s.pointer("/lfs/sha256"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// Remove the single-flight mapping only when it is still ours (a newer
/// pull may have replaced it in the tiny cancelled-but-cleaning window).
fn release_target(manager: &DownloadManager, target: &Path, id: &str) {
    let mut in_flight = manager
        .in_flight
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if in_flight.get(target).map(String::as_str) == Some(id) {
        in_flight.remove(target);
    }
}

/// Field-write helper (locks are never held across IO).
fn patch_entry(entry: &Arc<Mutex<Entry>>, patch: impl FnOnce(&mut Entry)) {
    let mut guard = entry
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    patch(&mut guard);
}

/// Delete `.part` + meta sidecar (best effort; used on cancel and commit).
async fn remove_sidecars(target: &Path) {
    let _ = tokio::fs::remove_file(with_suffix(target, PART_SUFFIX)).await;
    let _ = tokio::fs::remove_file(with_suffix(target, META_SUFFIX)).await;
}

/// `<final><suffix>` sibling path (same directory → same filesystem).
fn with_suffix(target: &Path, suffix: &str) -> PathBuf {
    let mut os = target.as_os_str().to_os_string();
    os.push(suffix);
    PathBuf::from(os)
}

/// Outcome of one fetch attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    /// Bytes committed to the final path.
    Done,
    /// Target file already existed at commit time; nothing overwritten.
    AlreadyPresent,
    /// Cancelled via the token (sidecars deleted by the finisher).
    Cancelled,
    /// Failed; `keep_part` false means the fetcher already deleted the
    /// sidecars (checksum mismatch).
    Failed {
        /// Human-readable cause for the DTO.
        error: String,
        /// False only after a checksum mismatch deleted the sidecars.
        keep_part: bool,
    },
    /// Resume response unusable (416 / wrong offset / wrong `ETag`). Resolved
    /// inside `fetch` by deleting sidecars and retrying cold once; the saved
    /// checksum is carried over so the retry still enforces it.
    RetryCold {
        /// sha256 recorded by the sidecar, if any.
        saved_sha: Option<String>,
    },
}

impl Outcome {
    /// Transient failure: sidecars kept for the next resume.
    fn failed(error: String) -> Self {
        Self::Failed {
            error,
            keep_part: true,
        }
    }
}

/// Task body: wait for a slot, fetch, then land the terminal state. The
/// terminal write wins over a racing handler-side `cancelled` mark — the
/// task owns the truth about the outcome.
async fn run_download(manager: Arc<DownloadManager>, job: Job) {
    let token = entry_token(&job.entry);
    let _permit = match token
        .run_until_cancelled(Arc::clone(&manager.sem).acquire_owned())
        .await
    {
        Some(Ok(permit)) => permit,
        Some(Err(err)) => {
            finish_failed(
                &manager,
                &job,
                &format!("download scheduler failed: {err}"),
                true,
            )
            .await;
            return;
        }
        // Cancelled while queued: never started, just clean up.
        None => return finish_cancelled(&manager, &job).await,
    };
    match fetch(&manager.client, &job, &token).await {
        Outcome::Done => finish_done(&manager, &job, false),
        Outcome::AlreadyPresent => finish_done(&manager, &job, true),
        Outcome::Cancelled => finish_cancelled(&manager, &job).await,
        Outcome::Failed { error, keep_part } => {
            finish_failed(&manager, &job, &error, keep_part).await;
        }
        // Defensive: `fetch` resolves this internally; it never reaches here.
        Outcome::RetryCold { .. } => {
            finish_failed(&manager, &job, "unusable resume response", false).await;
        }
    }
}

/// Clone the entry's cancel handle (short lock).
fn entry_token(entry: &Arc<Mutex<Entry>>) -> CancellationToken {
    entry
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .token
        .clone()
}

/// Terminal write: committed → `done` + rescan signal (the catalog refresh
/// loop picks the file up with its normal slug).
fn finish_done(manager: &DownloadManager, job: &Job, already_present: bool) {
    release_target(manager, &job.target, &job.id);
    patch_entry(&job.entry, |e| {
        e.state = DownloadStatus::Done;
        e.already_present = already_present;
        e.error = None;
    });
    manager.rescan.notify_one();
}

/// Terminal write: failed → `failed` + failure metric. Sidecars survive
/// unless the fetcher already removed them (checksum mismatch).
async fn finish_failed(manager: &DownloadManager, job: &Job, error: &str, keep_part: bool) {
    if !keep_part {
        remove_sidecars(&job.target).await;
    }
    DOWNLOAD_FAILURES[kind_slot(job.kind)].fetch_add(1, Ordering::Relaxed);
    release_target(manager, &job.target, &job.id);
    patch_entry(&job.entry, |e| {
        e.state = DownloadStatus::Failed;
        e.error = Some(error.to_owned());
    });
}

/// Terminal write: cancelled → `cancelled` + sidecar removal.
async fn finish_cancelled(manager: &DownloadManager, job: &Job) {
    remove_sidecars(&job.target).await;
    release_target(manager, &job.target, &job.id);
    patch_entry(&job.entry, |e| {
        e.state = DownloadStatus::Cancelled;
    });
}

/// Post-header negotiation result (what to write, from where, with which
/// validator and checksum expectation).
struct Negotiated {
    /// 206-append mode (false: truncate + restart).
    append: bool,
    /// Byte offset the write starts at.
    start_done: u64,
    /// Total size when known.
    total: Option<u64>,
    /// Fresh validator from THIS response (persisted for resume).
    etag: Option<String>,
    /// Ranged resume possible for future pulls (strong `ETag` seen).
    can_resume: bool,
    /// Expected sha256 (fresh repo lookup wins, else the persisted one).
    expected_sha: Option<String>,
}

/// Resume inputs: the saved sidecar (only for the SAME source url) and
/// the size of the `.part` it describes (0 when there is nothing to
/// resume — a `.part` without a matching meta cannot be validated).
async fn resume_state(part: &Path, meta_path: &Path, url: &str) -> (Option<PartMeta>, u64) {
    let saved = read_meta(meta_path).await.filter(|meta| meta.url == url);
    let size = match &saved {
        Some(_) => tokio::fs::metadata(part).await.map_or(0, |meta| meta.len()),
        None => 0,
    };
    (saved, size)
}

/// Weak validators cannot guarantee the `.part` prefix identity, so only a
/// strong `ETag` may gate a ranged resume.
fn strong_etag(value: Option<&str>) -> Option<&str> {
    value.filter(|etag| !etag.starts_with("W/"))
}

/// What a resume attempt carries into the negotiation: the requested
/// offset and the sidecar values still trusted for this url.
struct ResumeInputs {
    /// Was a resume attempted (matching meta, non-empty `.part`, strong `ETag`)?
    resume: bool,
    /// Byte offset requested (`Range: bytes=<resumed_size>-`).
    resumed_size: u64,
    /// Total the sidecar recorded (fallback when 206 lacks a total).
    saved_total: Option<u64>,
    /// Strong `ETag` the saved prefix was fetched under; a 206 must echo it.
    saved_etag: Option<String>,
    /// sha the sidecar recorded (fallback when the fresh lookup is none).
    saved_sha: Option<String>,
    /// sha from the fresh repo lookup (wins over the persisted one).
    fresh_sha: Option<String>,
}

/// Post-header negotiation: from the response status/validators decide
/// append-vs-restart, the byte offsets, and the checksum expectation.
/// `Err` is a terminal fetch outcome (unexpected status / offset mismatch).
fn negotiate_response(
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    content_length: Option<u64>,
    inputs: &ResumeInputs,
) -> Result<Negotiated, Outcome> {
    let etag = headers
        .get(reqwest::header::ETAG)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let accept_ranges = headers
        .get(reqwest::header::ACCEPT_RANGES)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value == "bytes");
    let can_resume = strong_etag(etag.as_deref()).is_some() && accept_ranges;
    // 416 means the saved prefix no longer describes this source: stop
    // asking for that Range — `fetch` drops the sidecars and retries cold.
    if inputs.resume && status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        return Err(Outcome::RetryCold {
            saved_sha: inputs.saved_sha.clone(),
        });
    }
    // Resume negotiation: 206 appends; a 200 after a ranged request means
    // the range was ignored or the content changed → truncate + restart.
    let ranged_ok = inputs.resume && status == reqwest::StatusCode::PARTIAL_CONTENT;
    let restart = !ranged_ok && status == reqwest::StatusCode::OK;
    if !ranged_ok && !restart {
        // Transient (429/5xx) or permanent (404) — keep sidecars either
        // way; a retry re-decides freshness via `If-Range`.
        return Err(Outcome::failed(format!("unexpected HTTP status {status}")));
    }
    let content_range = headers
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(parse_content_range);
    if ranged_ok {
        // Append only when the exact offset was honored AND the server
        // echoes the validator the prefix was fetched under; otherwise the
        // tail would be spliced onto bytes from a different version.
        let offset_ok = content_range.is_some_and(|(start, _, _)| start == inputs.resumed_size);
        let etag_ok = inputs
            .saved_etag
            .as_deref()
            .is_some_and(|saved| etag.as_deref() == Some(saved));
        if !offset_ok || !etag_ok {
            return Err(Outcome::RetryCold {
                saved_sha: inputs.saved_sha.clone(),
            });
        }
    }
    let total = if ranged_ok {
        content_range
            .and_then(|(_, _, total)| total)
            .or(inputs.saved_total)
    } else {
        content_length
    };
    Ok(Negotiated {
        append: ranged_ok,
        start_done: if ranged_ok { inputs.resumed_size } else { 0 },
        total,
        etag,
        can_resume,
        // The fresh repo lookup wins over the persisted sha.
        expected_sha: inputs.fresh_sha.clone().or(inputs.saved_sha.clone()),
    })
}

/// Send the GET (with `Range`/`If-Range` when resuming); `None` from the
/// cancel race maps to `Outcome::Cancelled`.
async fn send_get(
    client: &reqwest::Client,
    job: &Job,
    token: &CancellationToken,
    saved: Option<&PartMeta>,
    resumed_size: u64,
    resume: bool,
) -> Result<reqwest::Response, Outcome> {
    let mut request = client.get(&job.url);
    if resume {
        request = request.header(reqwest::header::RANGE, format!("bytes={resumed_size}-"));
        if let Some(etag) = saved.and_then(|meta| meta.etag.as_deref()) {
            request = request.header(reqwest::header::IF_RANGE, etag);
        }
    }
    match token.run_until_cancelled(request.send()).await {
        None => Err(Outcome::Cancelled),
        Some(Err(err)) => Err(Outcome::failed(format!("request failed: {err}"))),
        Some(Ok(response)) => Ok(response),
    }
}

/// Commit a fully fetched `.part`: honor an externally landed target,
/// otherwise rename into place and drop the sidecar, persisting the
/// verified digest (if the source published one and it matched) as the
/// artifact provenance sidecar.
async fn commit(job: &Job, part: &Path, verified: Option<String>) -> Outcome {
    // Someone landed the file while we fetched: keep theirs, drop ours.
    if job.target.is_file() {
        remove_sidecars(&job.target).await;
        let _ = tokio::fs::remove_file(with_suffix(&job.target, ARTIFACT_META_SUFFIX)).await;
        return Outcome::AlreadyPresent;
    }
    if let Err(err) = tokio::fs::rename(part, &job.target).await {
        return Outcome::failed(format!("commit rename failed: {err}"));
    }
    let _ = tokio::fs::remove_file(with_suffix(&job.target, META_SUFFIX)).await;
    if let Some(sha256) = verified {
        write_artifact_meta(&job.target, &sha256);
    }
    Outcome::Done
}

/// Terminal tail of a fetched stream: fsync, checksum enforcement (only
/// when the source published one — a mismatch deletes the sidecars), then
/// the commit rename, carrying the verified digest for provenance.
async fn finalize(
    job: &Job,
    part: &Path,
    file: tokio::fs::File,
    hasher: Option<Sha256>,
    expected: Option<String>,
    done: u64,
) -> Outcome {
    if let Err(err) = file.sync_all().await {
        return Outcome::failed(format!("cannot flush {}: {err}", part.display()));
    }
    drop(file);
    let mut verified: Option<String> = None;
    if let (Some(expected), Some(hash)) = (expected, hasher) {
        let got = hex_lower(&hash.finalize());
        if got != expected.to_ascii_lowercase() {
            remove_sidecars(&job.target).await;
            return Outcome::Failed {
                error: format!("checksum mismatch: expected sha256 {expected}, got {got}"),
                keep_part: false,
            };
        }
        verified = Some(got);
    }
    let outcome = commit(job, part, verified).await;
    if outcome == Outcome::Done {
        patch_entry(&job.entry, |e| e.bytes_done = done);
    }
    outcome
}

/// Persist/refresh the meta sidecar after the headers land (so a crash
/// mid-body is resumable against the negotiated validator + total).
fn record_meta(meta_path: &Path, job: &Job, negotiated: &Negotiated) {
    write_meta(
        meta_path,
        &PartMeta {
            url: job.url.clone(),
            etag: negotiated.etag.clone(),
            bytes_total: negotiated.total,
            sha256: negotiated.expected_sha.clone(),
        },
    );
}

/// Sidecar of a COMPLETED download (`<final>.sha256.json`): the digest of
/// the bytes that were streamed AND verified against the source's
/// published checksum. Never written without that verification — an
/// unverified download leaves no digest anywhere.
#[derive(serde::Serialize, serde::Deserialize)]
struct ArtifactMeta {
    sha256: String,
}

/// Persist the verified digest beside the completed artifact (tiny file,
/// sync write; failure only loses provenance and is logged).
fn write_artifact_meta(target: &Path, sha256: &str) {
    let path = with_suffix(target, ARTIFACT_META_SUFFIX);
    match serde_json::to_string(&ArtifactMeta {
        sha256: sha256.to_owned(),
    }) {
        Ok(json) => {
            if let Err(err) = std::fs::write(&path, json) {
                tracing::warn!(path = %path.display(), %err, "artifact meta write failed");
            }
        }
        Err(err) => tracing::warn!(%err, "artifact meta encode failed"),
    }
}

/// The verified digest recorded beside a completed download, if any.
/// `None` = unknown (no sidecar, malformed JSON, or not 64 lowercase hex
/// characters) — callers record null, never a fabricated hash. Sync read
/// of a tiny file, safe from the request path.
#[must_use]
pub fn artifact_sha256(target: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(with_suffix(target, ARTIFACT_META_SUFFIX)).ok()?;
    let meta: ArtifactMeta = serde_json::from_str(&raw).ok()?;
    let looks_like_sha = meta.sha256.len() == 64
        && meta
            .sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase());
    looks_like_sha.then_some(meta.sha256)
}

/// Delete the provenance sidecar of `target` (best effort) — used when the
/// model file itself is deleted so the sidecar never outlives its file.
pub async fn remove_artifact_meta(target: &Path) {
    let _ = tokio::fs::remove_file(with_suffix(target, ARTIFACT_META_SUFFIX)).await;
}

/// Open the `.part` for the negotiated mode and seed the hasher: append
/// continues at the recorded offset (hashing the prefix once — a 206 under
/// `If-Range` guarantees prefix identity), a restart truncates from
/// scratch.
async fn open_part(
    part: &Path,
    append: bool,
    resumed_size: u64,
    expected: Option<&String>,
) -> Result<(tokio::fs::File, Option<Sha256>), Outcome> {
    let opened = if append {
        tokio::fs::OpenOptions::new().append(true).open(part).await
    } else {
        // Cold start / stale resume: truncate whatever is there.
        tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(part)
            .await
    };
    let file = match opened {
        Ok(file) => file,
        Err(err) => {
            return Err(Outcome::failed(format!(
                "cannot open {}: {err}",
                part.display()
            )));
        }
    };
    let mut hasher = expected.map(|_| Sha256::new());
    if append
        && let Some(hash) = hasher.as_mut()
        && let Err(err) = hash_prefix(part, resumed_size, hash).await
    {
        return Err(Outcome::failed(format!(
            "cannot hash existing .part: {err}"
        )));
    }
    Ok((file, hasher))
}

/// The actual ranged fetch: ETag-gated resume negotiation, streamed writes
/// with cancel-at-chunk-boundary, checksum verification, fsync + rename. An
/// unusable resume response (416 / wrong offset / wrong `ETag`) self-heals:
/// sidecars are dropped and the fetch is retried cold once.
async fn fetch(client: &reqwest::Client, job: &Job, token: &CancellationToken) -> Outcome {
    let part = with_suffix(&job.target, PART_SUFFIX);
    let meta_path = with_suffix(&job.target, META_SUFFIX);
    let mut cold_sha = None;
    for cold_retry in [false, true] {
        let outcome = fetch_attempt(
            client,
            job,
            token,
            &part,
            &meta_path,
            cold_retry,
            cold_sha.clone(),
        )
        .await;
        match outcome {
            Outcome::RetryCold { saved_sha } => {
                // Drop the poisoned prefix; carry its checksum into the cold
                // retry so verification cannot silently disappear.
                remove_sidecars(&job.target).await;
                cold_sha = saved_sha;
            }
            terminal => return terminal,
        }
    }
    // Unreachable: without a resume there is no Range to reject.
    Outcome::failed("cold retry after unusable resume failed".to_owned())
}

/// One fetch attempt; `cold_retry` ignores the sidecar entirely.
// Setup → negotiate → stream → finalize stays in one function so cancel
// handling and sidecar state cannot drift apart.
#[allow(clippy::too_many_lines)]
async fn fetch_attempt(
    client: &reqwest::Client,
    job: &Job,
    token: &CancellationToken,
    part: &Path,
    meta_path: &Path,
    cold_retry: bool,
    cold_sha: Option<String>,
) -> Outcome {
    // Another pull (or process) committed the file while we waited.
    if job.target.is_file() {
        remove_sidecars(&job.target).await;
        return Outcome::AlreadyPresent;
    }
    let (saved, resumed_size) = if cold_retry {
        (None, 0)
    } else {
        resume_state(part, meta_path, &job.url).await
    };
    let resume = resumed_size > 0
        && saved
            .as_ref()
            .and_then(|meta| strong_etag(meta.etag.as_deref()))
            .is_some();

    patch_entry(&job.entry, |e| {
        e.state = if resume {
            DownloadStatus::Resuming
        } else {
            DownloadStatus::Running
        };
        e.bytes_done = if resume { resumed_size } else { 0 };
        e.bytes_total = if resume {
            saved.as_ref().and_then(|meta| meta.bytes_total)
        } else {
            None
        };
    });

    let response = match send_get(client, job, token, saved.as_ref(), resumed_size, resume).await {
        Ok(response) => response,
        Err(outcome) => return outcome,
    };
    let negotiated = match negotiate_response(
        response.status(),
        response.headers(),
        response.content_length(),
        &ResumeInputs {
            resume,
            resumed_size,
            saved_total: saved.as_ref().and_then(|meta| meta.bytes_total),
            saved_sha: saved
                .as_ref()
                .and_then(|meta| meta.sha256.clone())
                .or(cold_sha),
            saved_etag: saved.as_ref().and_then(|meta| meta.etag.clone()),
            fresh_sha: job.expected_sha.clone(),
        },
    ) {
        Ok(negotiated) => negotiated,
        Err(outcome) => return outcome,
    };

    patch_entry(&job.entry, |e| {
        e.bytes_done = negotiated.start_done;
        e.bytes_total = negotiated.total;
        e.resumable = negotiated.can_resume;
    });
    record_meta(meta_path, job, &negotiated);

    let (mut file, mut hasher) = match open_part(
        part,
        negotiated.append,
        resumed_size,
        negotiated.expected_sha.as_ref(),
    )
    .await
    {
        Ok(opened) => opened,
        Err(outcome) => return outcome,
    };

    let stream = response.bytes_stream();
    tokio::pin!(stream);
    let mut done = negotiated.start_done;
    loop {
        let chunk = tokio::select! {
            biased;
            () = token.cancelled() => {
                drop(file);
                return Outcome::Cancelled;
            }
            item = stream.next() => match item {
                Some(Ok(chunk)) => chunk,
                Some(Err(err)) => {
                    drop(file);
                    // Connection loss and ENOSPC (write-side lands just
                    // below) both come back here: transient failure,
                    // sidecars kept.
                    return Outcome::failed(format!("transfer failed: {err}"));
                }
                None => break,
            }
        };
        if let Err(err) = file.write_all(&chunk).await {
            return Outcome::failed(format!("write failed: {err}"));
        }
        if let Some(hash) = hasher.as_mut() {
            hash.update(&chunk);
        }
        let chunk_len = u64::try_from(chunk.len()).unwrap_or(u64::MAX);
        done += chunk_len;
        DOWNLOAD_BYTES[kind_slot(job.kind)].fetch_add(chunk_len, Ordering::Relaxed);
        patch_entry(&job.entry, |e| e.bytes_done = done);
    }
    finalize(
        job,
        part,
        file,
        hasher,
        negotiated.expected_sha.clone(),
        done,
    )
    .await
}

/// Read + parse the meta sidecar; any failure means "no usable resume
/// state" (a cold restart overwrites it).
async fn read_meta(path: &Path) -> Option<PartMeta> {
    let raw = tokio::fs::read_to_string(path).await.ok()?;
    serde_json::from_str(&raw).ok()
}

/// Persist the meta sidecar (tiny file, sync write; failure only loses
/// resumability of a *future* crash and is logged).
fn write_meta(path: &Path, meta: &PartMeta) {
    match serde_json::to_string(meta) {
        Ok(json) => {
            if let Err(err) = std::fs::write(path, json) {
                tracing::warn!(path = %path.display(), %err, "download meta write failed");
            }
        }
        Err(err) => tracing::warn!(%err, "download meta encode failed"),
    }
}

/// Feed the first `len` bytes of `path` into `hash` (1 MiB steps).
async fn hash_prefix(path: &Path, len: u64, hash: &mut Sha256) -> std::io::Result<()> {
    use tokio::io::AsyncReadExt as _;
    let mut file = tokio::fs::File::open(path).await?;
    let mut remaining = len;
    let mut buf = vec![0u8; 1024 * 1024];
    while remaining > 0 {
        let want = usize::try_from(remaining.min(buf.len() as u64)).unwrap_or(buf.len());
        let read = file.read(&mut buf[..want]).await?;
        if read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "part shorter than recorded resume offset",
            ));
        }
        hash.update(&buf[..read]);
        remaining -= u64::try_from(read).unwrap_or(u64::MAX);
    }
    Ok(())
}

/// `bytes 4-11/12` → `(4, 11, Some(12))`; a `*` total parses as `None`.
fn parse_content_range(value: &str) -> Option<(u64, u64, Option<u64>)> {
    let rest = value.strip_prefix("bytes ")?;
    let (range, total) = rest.split_once('/')?;
    let (first, last) = range.split_once('-')?;
    let first = first.parse().ok()?;
    let last = last.parse().ok()?;
    // `*` means "total unknown": keep the valid range instead of dropping
    // the whole header, which would trap every later resume attempt.
    let total = if total == "*" {
        None
    } else {
        total.parse().ok()
    };
    Some((first, last, total))
}

/// sha256 digest → 64-char lowercase hex.
fn hex_lower(digest: &[u8]) -> String {
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Count stale `*.part` files under the managed models root (startup log
/// only — they are deliberately kept for resume, never swept).
pub fn count_stale_parts(root: Option<&Path>) -> usize {
    let Some(root) = root else {
        return 0;
    };
    let mut count = 0;
    walk_parts(root, 0, &mut count);
    count
}

/// Recursive `*.part` counter (bounded depth; `models/{kind}/file.part`).
fn walk_parts(dir: &Path, depth: usize, out: &mut usize) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_parts(&path, depth + 1, out);
        } else if path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|name| name.ends_with(PART_SUFFIX))
        {
            *out += 1;
        }
    }
}
