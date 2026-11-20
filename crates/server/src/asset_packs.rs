//! Curated third-party asset packs (TASK-qal-guard + TASK-cc0-pack-fetch).
//!
//! One reviewed list (`packs.json`, parsed in `catalog`) is the whole
//! license authority: QAL rows are no-redistribution and are refused before
//! any bytes are requested; CC0 rows are fetchable, with a locally computed
//! sha256 and the curated license row written beside the bytes so bundle
//! manifests can join on it. No fuzzy filename matching: a filename is only
//! accepted when the curated list names it verbatim.
//!
//! Temp discipline mirrors `downloads.rs`: `<final>.part` in the final
//! directory, flush + fsync, then rename; failures remove the `.part`.

use std::fmt;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt as _;
use tokio_stream::StreamExt as _;

/// One curated pack row. All fields come from the reviewed list; nothing is
/// inferred from a URL or a downloaded file name.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PackEntry {
    /// Stable curated id (the only join key callers should use).
    pub id: String,
    /// Where the pack is published.
    pub source_url: String,
    /// Content license id (`QAL-1.0` or `CC0-1.0`).
    pub license: String,
    /// Whether honkbox may bundle/serve the bytes.
    pub redistributable: bool,
    /// Human-readable restriction term for a refusal.
    #[serde(default)]
    pub restriction: Option<String>,
    /// Published digest, only when the source actually published one.
    #[serde(default)]
    pub sha256: Option<String>,
    /// Published md5 (PolyHaven publishes one per asset); integrity
    /// cross-check metadata, never a substitute for the local sha256.
    #[serde(default)]
    pub source_md5: Option<String>,
    /// Exact file names the curated list recognizes as this pack.
    #[serde(default)]
    pub file_names: Vec<String>,
    /// Attribution text for the manifest row.
    #[serde(default)]
    pub attribution_text: Option<String>,
}

/// Exact curated-id lookup, with an exact curated-filename fallback. An
/// unlisted name never matches — there is no substring/prefix heuristic.
#[must_use]
pub(crate) fn lookup(id_or_file: &str) -> Option<&'static PackEntry> {
    crate::catalog::packs()
        .iter()
        .find(|pack| pack.id == id_or_file || pack.file_names.iter().any(|name| name == id_or_file))
}

/// Named 412 refusal for a pack that may not be redistributed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PackRefusal {
    /// The pack's curated license forbids bundling/serving (or it has no
    /// curated license record at all — fail closed).
    Restricted { pack: String, term: String },
}

impl PackRefusal {
    /// Wire code in the OpenAI-style error envelope.
    pub(crate) fn code(&self) -> &'static str {
        "pack_license_restricted"
    }

    /// The deployment asked for a redistribution the license forbids.
    pub(crate) fn status(&self) -> StatusCode {
        StatusCode::PRECONDITION_FAILED
    }

    /// Names the pack and the exact term — never a silent skip.
    pub(crate) fn message(&self) -> String {
        match self {
            Self::Restricted { pack, term } => {
                format!("asset pack {pack:?}: redistribution refused; {term}")
            }
        }
    }

    fn restricted(pack: impl Into<String>, term: impl Into<String>) -> Self {
        Self::Restricted {
            pack: pack.into(),
            term: term.into(),
        }
    }
}

impl fmt::Display for PackRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.message())
    }
}

/// Gate every pack before any request or write. Unlisted packs fail closed:
/// without a curated license row there is no evidence of redistribution
/// rights.
pub(crate) fn guard_pack(pack_id: &str) -> Result<(), PackRefusal> {
    match lookup(pack_id) {
        Some(pack) if pack.redistributable && pack.license != "QAL-1.0" => Ok(()),
        Some(pack) => Err(PackRefusal::restricted(
            pack.id.clone(),
            pack.restriction
                .as_deref()
                .unwrap_or("no redistribution license recorded"),
        )),
        None => Err(PackRefusal::restricted(
            pack_id,
            "not a curated pack: no redistribution license recorded",
        )),
    }
}

/// Request body for `POST /v1/asset-packs/fetch`.
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub(crate) struct FetchRequest {
    /// Curated pack id.
    pub pack_id: String,
    /// Direct source URL for this fetch.
    pub source_url: String,
    /// Safe artifact id used for the stored bytes and license sidecar.
    pub artifact_id: String,
}

/// What the fetch wrote: the record returned to the caller and the license
/// row read back by the bundle-manifest renderer.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub(crate) struct FetchRecord {
    pub pack_id: String,
    pub artifact_id: String,
    pub license: String,
    /// Curated publisher page from the reviewed pack list.
    pub source_url: String,
    /// Actual URL the bytes were fetched from (may be an operator mirror).
    pub fetch_url: String,
    /// Locally computed sha256 of the bytes actually stored.
    pub sha256: String,
    /// Publisher's md5, when the curated row records one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_md5: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution_text: Option<String>,
    pub bytes: u64,
}

/// Fully validated fetch inputs plus the resolved destination root. No
/// global registry or ambient data-dir lookup: the route owns resolution.
#[derive(Debug, Clone)]
pub(crate) struct FetchJob {
    pub pack_id: String,
    pub source_url: String,
    pub artifact_id: String,
    /// Root under which `asset-packs/` is created.
    pub root: PathBuf,
}

/// Why a fetch failed. A refused pack carries the named 412 refusal.
#[derive(Debug)]
pub(crate) enum FetchError {
    /// License gate refused the pack (no bytes requested).
    Refused(PackRefusal),
    /// Local filesystem or JSON failure.
    Io(String),
    /// HTTP status outside 2xx.
    Status(u16),
    /// Body transfer failed.
    Transfer(String),
    /// Published sha256 did not match the locally computed digest.
    Checksum(String),
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(refusal) => write!(f, "{refusal}"),
            Self::Io(err) => write!(f, "asset pack IO failed: {err}"),
            Self::Status(code) => write!(f, "asset pack source returned HTTP {code}"),
            Self::Transfer(err) => write!(f, "asset pack transfer failed: {err}"),
            Self::Checksum(err) => write!(f, "asset pack checksum mismatch: {err}"),
        }
    }
}

/// Fetch one curated CC0 pack into `root/asset-packs/<artifact_id>.bin` and
/// write `<artifact_id>.license.json` with the sha256 + license row.
pub(crate) async fn fetch(job: &FetchJob) -> Result<FetchRecord, FetchError> {
    // Gate at the write seam too: a refusal must not even create the
    // destination directory, let alone a `.part`.
    guard_pack(&job.pack_id).map_err(FetchError::Refused)?;
    let pack = lookup(&job.pack_id).expect("guard accepted a curated pack");

    let dir = job.root.join("asset-packs");
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|err| FetchError::Io(err.to_string()))?;
    let final_path = dir.join(format!("{}.bin", job.artifact_id));
    let part = dir.join(format!("{}.bin.part", job.artifact_id));

    let client = reqwest::Client::builder()
        .user_agent(concat!("honkbox/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .map_err(|err| FetchError::Io(format!("HTTP client: {err}")))?;
    let response = client
        .get(&job.source_url)
        .send()
        .await
        .map_err(|err| FetchError::Transfer(err.to_string()))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(FetchError::Status(status));
    }

    let mut file = tokio::fs::File::create(&part)
        .await
        .map_err(|err| FetchError::Io(err.to_string()))?;
    let mut hasher = Sha256::new();
    let mut bytes = 0u64;
    let stream = response.bytes_stream();
    tokio::pin!(stream);
    let streamed = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|err| FetchError::Transfer(err.to_string()))?;
            file.write_all(&chunk)
                .await
                .map_err(|err| FetchError::Io(err.to_string()))?;
            hasher.update(&chunk);
            bytes = bytes.saturating_add(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
        }
        file.flush()
            .await
            .map_err(|err| FetchError::Io(err.to_string()))?;
        file.sync_all()
            .await
            .map_err(|err| FetchError::Io(err.to_string()))?;
        Ok::<(), FetchError>(())
    }
    .await;
    drop(file);
    if let Err(err) = streamed {
        discard(&part).await;
        return Err(err);
    }

    let sha256 = sha256_hex(hasher.finalize());
    if let Some(expected) = &pack.sha256 {
        if !sha256.eq_ignore_ascii_case(expected) {
            discard(&part).await;
            return Err(FetchError::Checksum(format!(
                "pack {:?}: expected {expected}, got {sha256}",
                pack.id
            )));
        }
    }
    tokio::fs::rename(&part, &final_path)
        .await
        .map_err(|err| FetchError::Io(err.to_string()))?;

    let record = FetchRecord {
        pack_id: pack.id.clone(),
        artifact_id: job.artifact_id.clone(),
        license: pack.license.clone(),
        source_url: pack.source_url.clone(),
        fetch_url: job.source_url.clone(),
        sha256,
        source_md5: pack.source_md5.clone(),
        attribution_text: pack.attribution_text.clone(),
        bytes,
    };
    let sidecar = dir.join(format!("{}.license.json", job.artifact_id));
    write_json_atomic(&sidecar, &record).await?;
    Ok(record)
}

/// Remove an abandoned `.part`; a failed fetch leaves no bytes behind.
async fn discard(part: &Path) {
    let _ = tokio::fs::remove_file(part).await;
}

/// Same `.part` + fsync + rename discipline as the payload bytes.
async fn write_json_atomic(path: &Path, record: &FetchRecord) -> Result<(), FetchError> {
    let raw = serde_json::to_vec_pretty(record)
        .map_err(|err| FetchError::Io(format!("encode license row: {err}")))?;
    let part = path.with_extension("json.part");
    let mut file = tokio::fs::File::create(&part)
        .await
        .map_err(|err| FetchError::Io(err.to_string()))?;
    file.write_all(&raw)
        .await
        .map_err(|err| FetchError::Io(err.to_string()))?;
    file.flush()
        .await
        .map_err(|err| FetchError::Io(err.to_string()))?;
    file.sync_all()
        .await
        .map_err(|err| FetchError::Io(err.to_string()))?;
    drop(file);
    tokio::fs::rename(&part, path)
        .await
        .map_err(|err| FetchError::Io(err.to_string()))?;
    Ok(())
}

/// Lowercase hex digest (no new hex dependency).
fn sha256_hex(digest: impl AsRef<[u8]>) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest.as_ref() {
        let _ = write!(out, "{byte:02x}");
    }
    out
}
