//! Bounded container reads: the zip and GLB containers behind the layered
//! and binary formats (TASK-container-safety).
//!
//! Every container is read through explicit caps — an entry count, a
//! per-entry byte cap and a total byte cap — and every rejection is a
//! NAMED error ([`ContainerError`]) whose message names the limit it
//! broke. The reader never writes to the filesystem and never extracts an
//! archive: entries are decompressed in memory, counted as they stream
//! out, and hashed with SHA-256 while they stream.
//!
//! This reader is central-directory-only: it lists and reads every entry
//! through `ZipArchive` (the CENTRAL DIRECTORY), never through
//! `ZipStreamReader` or any API that trusts the LOCAL file header, and it
//! never writes an entry to disk by its name. The local-header /
//! central-directory name mismatch advisory (RUSTSEC-2026-0548) therefore
//! does not apply to this path.
//!
//! The caps are enforced in two places, deliberately:
//!
//! - [`Container::open_zip`] validates the CENTRAL DIRECTORY (entry
//!   count, declared per-entry size, declared total) before a single
//!   entry byte is read — a cheap pre-filter on the archive's own claims;
//! - [`Container::read_entry`] caps the DECOMPRESSED bytes as they
//!   arrive, because the declared size is only a claim: a crafted
//!   archive can understate it, and a decompression bomb is exactly that.
//!   The streamed counters are what actually stop a bomb.
//!
//! The same rule applies to the GLB container ([`validate_glb`]): the
//! header and the chunk lengths are validated against the declared
//! length, each chunk is capped, and a JSON chunk that references an
//! EXTERNAL buffer is rejected — a GLB that points at a file on disk is a
//! container that reaches outside itself.

use std::fmt;
use std::io::{Cursor, Read};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};
use zip::result::ZipError;
use zip::{CompressionMethod, ZipArchive};

/// Streaming chunk size for entry reads (64 KiB) — big enough to keep the
/// syscall count down, small enough that the bomb is stopped long before
/// the cap is overshot.
const READ_CHUNK: usize = 64 * 1024;

/// The `glTF` magic of a GLB header (little-endian `0x4654_6C67`).
const GLB_MAGIC: u32 = 0x4654_6C67;

/// GLB chunk type of the JSON chunk (`0x4E4F_534A`).
const GLB_CHUNK_JSON: u32 = 0x4E4F_534A;

/// GLB chunk type of the optional binary chunk (`0x004E_4942`).
const GLB_CHUNK_BIN: u32 = 0x004E_4942;

/// The entry name reported for the GLB JSON chunk in errors.
const GLB_JSON_NAME: &str = "<glb json chunk>";

/// The entry name reported for the GLB binary chunk in errors.
const GLB_BIN_NAME: &str = "<glb bin chunk>";

/// The caps enforced while reading one container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContainerLimits {
    /// Maximum number of entries the container may declare.
    pub max_entries: u64,
    /// Maximum number of bytes a single entry may hold (decompressed).
    pub max_entry_bytes: u64,
    /// Maximum number of bytes the container may hold in total
    /// (decompressed, summed over the entries that were read).
    pub max_total_bytes: u64,
}

/// Every rejection of a container read. The message of each variant names
/// the limit that was broken (or the rule that was violated), never a
/// bare "invalid input".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ContainerError {
    /// The container declares more entries than the entry-count limit.
    TooManyEntries {
        /// The configured entry-count limit.
        limit: u64,
        /// The number of entries the container declared.
        found: u64,
    },
    /// A single entry is larger than the per-entry byte limit.
    EntryTooLarge {
        /// The entry name (or the chunk name for a GLB chunk).
        name: String,
        /// The configured per-entry byte limit.
        limit: u64,
        /// The number of bytes the entry holds.
        found: u64,
    },
    /// The container holds more bytes in total than the total byte limit.
    TotalTooLarge {
        /// The configured total byte limit.
        limit: u64,
        /// The number of bytes the container holds.
        found: u64,
    },
    /// An entry name (or a container reference) violates a path rule.
    UnsafePath {
        /// The offending name.
        name: String,
        /// The rule that was violated (`"parent_traversal"`, `"symlink"`,
        /// `"external_buffer"`, …).
        reason: &'static str,
    },
    /// The container is structurally broken and cannot be read.
    Malformed {
        /// What is wrong with the container.
        detail: String,
    },
    /// The container is well formed but uses something this reader does
    /// not support (an encrypted entry, an unknown compression method).
    Unsupported {
        /// What is unsupported.
        detail: String,
    },
}

impl fmt::Display for ContainerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyEntries { limit, found } => {
                write!(f, "zip entry count {found} exceeds the limit of {limit}")
            }
            Self::EntryTooLarge { name, limit, found } => {
                write!(
                    f,
                    "entry {name} holds {found} bytes, exceeding the limit of {limit} bytes"
                )
            }
            Self::TotalTooLarge { limit, found } => {
                write!(
                    f,
                    "total uncompressed size {found} bytes exceeds the limit of {limit} bytes"
                )
            }
            Self::UnsafePath { name, reason } => {
                write!(f, "unsafe entry path {name}: {reason}")
            }
            Self::Malformed { detail } => write!(f, "malformed container: {detail}"),
            Self::Unsupported { detail } => write!(f, "unsupported container: {detail}"),
        }
    }
}

impl std::error::Error for ContainerError {}

/// One entry of a container, as declared by the container itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ContainerEntry {
    /// The entry name (never a safe path — unsafe names are rejected
    /// before an entry reaches this list).
    pub name: String,
    /// The declared uncompressed size in bytes.
    pub size: u64,
    /// Whether the entry is a directory.
    pub is_dir: bool,
    /// Whether the entry is stored uncompressed (not deflated).
    pub stored: bool,
}

/// The bytes of one container entry, with the SHA-256 of those bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ContainerFile {
    /// The decompressed entry bytes.
    pub bytes: Vec<u8>,
    /// Lowercase-hex SHA-256 of `bytes`.
    pub sha256: String,
}

/// An opened container: its entries, its raw bytes, and the running total
/// of the bytes read out of it.
#[derive(Debug)]
pub(crate) struct Container {
    /// The raw container bytes (kept so a read can re-open the archive
    /// without interior mutability — the container stays `Send + Sync`).
    bytes: Vec<u8>,
    /// The validated entries, in central-directory order.
    entries: Vec<ContainerEntry>,
    /// The number of decompressed bytes read out of this container so far.
    read_total: AtomicU64,
}

impl Container {
    /// Open a zip container and validate every entry it declares.
    ///
    /// The entry count, the declared per-entry sizes, the declared total
    /// size and the entry names are all checked BEFORE any entry byte is
    /// read, so a hostile archive never gets past the central directory.
    pub(crate) fn open_zip(bytes: &[u8], limits: ContainerLimits) -> Result<Self, ContainerError> {
        let mut archive = ZipArchive::new(Cursor::new(bytes))
            .map_err(|err| map_zip_error(err, "reading the zip central directory"))?;
        let count = u64::try_from(archive.len()).unwrap_or(u64::MAX);
        if count > limits.max_entries {
            return Err(ContainerError::TooManyEntries {
                limit: limits.max_entries,
                found: count,
            });
        }
        let mut entries: Vec<ContainerEntry> = Vec::with_capacity(archive.len());
        let mut total = 0u64;
        for index in 0..archive.len() {
            let file = archive
                .by_index(index)
                .map_err(|err| map_zip_error(err, "reading a zip entry header"))?;
            let name = file.name().to_owned();
            if let Some(reason) = path_reason(&name) {
                return Err(ContainerError::UnsafePath { name, reason });
            }
            if is_symlink(&file) {
                return Err(ContainerError::UnsafePath {
                    name,
                    reason: "symlink",
                });
            }
            let size = file.size();
            if size > limits.max_entry_bytes {
                return Err(ContainerError::EntryTooLarge {
                    name,
                    limit: limits.max_entry_bytes,
                    found: size,
                });
            }
            total = total.saturating_add(size);
            if total > limits.max_total_bytes {
                return Err(ContainerError::TotalTooLarge {
                    limit: limits.max_total_bytes,
                    found: total,
                });
            }
            entries.push(ContainerEntry {
                name,
                size,
                is_dir: file.is_dir(),
                stored: matches!(file.compression(), CompressionMethod::Stored),
            });
        }
        Ok(Self {
            bytes: bytes.to_vec(),
            entries,
            read_total: AtomicU64::new(0),
        })
    }

    /// The validated entries of this container, in central-directory order.
    pub(crate) fn entries(&self) -> &[ContainerEntry] {
        &self.entries
    }

    /// The entry with this name, if the container holds one.
    pub(crate) fn entry(&self, name: &str) -> Option<&ContainerEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// Read one entry out of the container.
    ///
    /// The entry is decompressed in [`READ_CHUNK`] chunks and the
    /// DECOMPRESSED byte count is capped as it streams: the declared size
    /// is never trusted, so a crafted archive that understates its entry
    /// (a decompression bomb) is stopped at the cap instead of filling
    /// the heap. The bytes are hashed with SHA-256 while they stream, and
    /// the container's running total is capped across reads.
    pub(crate) fn read_entry(
        &self,
        name: &str,
        limits: ContainerLimits,
    ) -> Result<ContainerFile, ContainerError> {
        let Some(entry) = self.entry(name) else {
            return Err(ContainerError::Malformed {
                detail: format!("the container holds no entry named {name}"),
            });
        };
        let index = self
            .entries
            .iter()
            .position(|candidate| candidate.name == entry.name)
            .unwrap_or(0);
        let mut archive = ZipArchive::new(Cursor::new(self.bytes.as_slice()))
            .map_err(|err| map_zip_error(err, "reading the zip central directory"))?;
        let mut file = archive
            .by_index(index)
            .map_err(|err| map_zip_error(err, name))?;
        let mut bytes: Vec<u8> = Vec::new();
        let mut hasher = Sha256::new();
        let mut chunk = vec![0u8; READ_CHUNK];
        let mut entry_bytes = 0u64;
        loop {
            let read = file
                .read(&mut chunk)
                .map_err(|err| ContainerError::Malformed {
                    detail: format!("reading the entry {name}: {err}"),
                })?;
            if read == 0 {
                break;
            }
            let delta = u64::try_from(read).unwrap_or(u64::MAX);
            entry_bytes = entry_bytes.saturating_add(delta);
            if entry_bytes > limits.max_entry_bytes {
                return Err(ContainerError::EntryTooLarge {
                    name: name.to_owned(),
                    limit: limits.max_entry_bytes,
                    found: entry_bytes,
                });
            }
            self.add_read_total(delta, limits.max_total_bytes)?;
            hasher.update(&chunk[..read]);
            bytes.extend_from_slice(&chunk[..read]);
        }
        Ok(ContainerFile {
            bytes,
            sha256: hex_digest(hasher.finalize()),
        })
    }

    /// Add the bytes of one read to the container's running total, and
    /// reject the read when the total crosses the configured limit.
    fn add_read_total(&self, delta: u64, limit: u64) -> Result<(), ContainerError> {
        let updated =
            self.read_total
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |previous| {
                    let next = previous.saturating_add(delta);
                    if next <= limit { Some(next) } else { None }
                });
        if updated.is_err() {
            let found = self
                .read_total
                .load(Ordering::Relaxed)
                .saturating_add(delta);
            return Err(ContainerError::TotalTooLarge { limit, found });
        }
        Ok(())
    }
}

/// The header facts of a GLB container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GlbInfo {
    /// The GLB version (1 or 2).
    pub version: u32,
    /// The declared total length of the container in bytes.
    pub length: u32,
    /// The length of the JSON chunk in bytes.
    pub json_bytes: u32,
    /// The length of the binary chunk in bytes (0 when there is none).
    pub bin_bytes: u32,
}

/// Validate a GLB container: header, chunk layout, size caps and the
/// external-buffer rule.
///
/// A GLB is self-contained by contract — its JSON chunk may only
/// reference buffers that live INSIDE the container (`data:` URIs). A
/// JSON chunk that points at an external file is rejected as an unsafe
/// path, because following it would read bytes from outside the
/// container that was handed to us.
#[allow(dead_code)] // wired by the GLB read path (integration lane)
pub(crate) fn validate_glb(
    bytes: &[u8],
    limits: ContainerLimits,
) -> Result<GlbInfo, ContainerError> {
    if bytes.len() < 12 {
        return Err(ContainerError::Malformed {
            detail: "the glb header is truncated".to_owned(),
        });
    }
    if read_u32(bytes, 0)? != GLB_MAGIC {
        return Err(ContainerError::Malformed {
            detail: "not a glb container (bad magic)".to_owned(),
        });
    }
    let version = read_u32(bytes, 4)?;
    if version != 1 && version != 2 {
        return Err(ContainerError::Malformed {
            detail: format!("unsupported glb version {version}"),
        });
    }
    let length = read_u32(bytes, 8)?;
    let total = usize::try_from(length).map_err(|_| ContainerError::Malformed {
        detail: format!("the glb length {length} does not fit the address space"),
    })?;
    if total > bytes.len() {
        return Err(ContainerError::Malformed {
            detail: format!(
                "the glb declares {total} bytes but only {} are present",
                bytes.len()
            ),
        });
    }
    let mut offset = 12usize;
    let mut json: Option<&[u8]> = None;
    let mut json_bytes = 0u32;
    let mut bin_bytes = 0u32;
    let mut chunk_index = 0u32;
    while offset < total {
        let chunk_length = usize::try_from(read_u32(bytes, offset)?).unwrap_or(usize::MAX);
        let chunk_type = read_u32(bytes, offset + 4)?;
        let start = offset + 8;
        let end = start.saturating_add(chunk_length);
        if end > total {
            return Err(ContainerError::Malformed {
                detail: format!(
                    "the glb chunk at offset {offset} declares {chunk_length} bytes past the container end"
                ),
            });
        }
        let payload = bytes
            .get(start..end)
            .ok_or_else(|| ContainerError::Malformed {
                detail: format!("the glb chunk at offset {offset} is truncated"),
            })?;
        match chunk_index {
            0 => {
                if chunk_type != GLB_CHUNK_JSON {
                    return Err(ContainerError::Malformed {
                        detail: "the first glb chunk must be the JSON chunk".to_owned(),
                    });
                }
                json = Some(payload);
                json_bytes = u32::try_from(chunk_length).unwrap_or(u32::MAX);
            }
            1 => {
                if chunk_type != GLB_CHUNK_BIN {
                    return Err(ContainerError::Malformed {
                        detail: "the second glb chunk must be the binary chunk".to_owned(),
                    });
                }
                bin_bytes = u32::try_from(chunk_length).unwrap_or(u32::MAX);
            }
            _ => {
                return Err(ContainerError::Malformed {
                    detail: "a glb holds at most a JSON chunk and a binary chunk".to_owned(),
                });
            }
        }
        chunk_index = chunk_index.saturating_add(1);
        // The next chunk starts on a four byte boundary: the chunk length
        // may or may not count the padding, and both spellings are in the
        // wild.
        offset = align_four(end);
    }
    let Some(json_payload) = json else {
        return Err(ContainerError::Malformed {
            detail: "the glb holds no JSON chunk".to_owned(),
        });
    };
    if u64::from(json_bytes) > limits.max_entry_bytes {
        return Err(ContainerError::EntryTooLarge {
            name: GLB_JSON_NAME.to_owned(),
            limit: limits.max_entry_bytes,
            found: u64::from(json_bytes),
        });
    }
    if u64::from(bin_bytes) > limits.max_entry_bytes {
        return Err(ContainerError::EntryTooLarge {
            name: GLB_BIN_NAME.to_owned(),
            limit: limits.max_entry_bytes,
            found: u64::from(bin_bytes),
        });
    }
    let payload_bytes = u64::from(json_bytes).saturating_add(u64::from(bin_bytes));
    if payload_bytes > limits.max_total_bytes {
        return Err(ContainerError::TotalTooLarge {
            limit: limits.max_total_bytes,
            found: payload_bytes,
        });
    }
    let json_text = std::str::from_utf8(json_payload).map_err(|_| ContainerError::Malformed {
        detail: "the glb JSON chunk is not valid UTF-8".to_owned(),
    })?;
    if has_external_buffer(json_text) {
        return Err(ContainerError::UnsafePath {
            name: GLB_JSON_NAME.to_owned(),
            reason: "external_buffer",
        });
    }
    Ok(GlbInfo {
        version,
        length,
        json_bytes,
        bin_bytes,
    })
}

/// Round `value` up to the next four byte boundary (the GLB chunk
/// alignment).
fn align_four(value: usize) -> usize {
    value.saturating_add(3) & !3usize
}

/// Read a little-endian `u32` at `offset`, or fail with a named error.
fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, ContainerError> {
    let end = offset.saturating_add(4);
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| ContainerError::Malformed {
            detail: format!("the container is truncated at offset {offset}"),
        })?;
    let array: [u8; 4] = slice.try_into().map_err(|_| ContainerError::Malformed {
        detail: format!("the container is truncated at offset {offset}"),
    })?;
    Ok(u32::from_le_bytes(array))
}

/// The reason an entry name is unsafe, or `None` when the name is safe.
///
/// The rules are the ones an extractor must honour even though this
/// reader never writes to disk: a name that escapes the container root
/// (`..`), a name that is absolute, a Windows drive letter, a Windows
/// separator (a traversal vector inside a zip), a NUL byte, or an empty
/// name.
fn path_reason(name: &str) -> Option<&'static str> {
    if name.is_empty() {
        return Some("empty");
    }
    if name.contains('\0') {
        return Some("nul_byte");
    }
    if name.contains('\\') {
        return Some("backslash");
    }
    if name.starts_with('/') {
        return Some("absolute");
    }
    if has_drive_letter(name) {
        return Some("drive_letter");
    }
    if name.split('/').any(|component| component == "..") {
        return Some("parent_traversal");
    }
    None
}

/// Whether the name starts with an ASCII letter followed by `:` (a
/// Windows drive letter such as `C:` or `C:/…`).
fn has_drive_letter(name: &str) -> bool {
    let bytes = name.as_bytes();
    matches!(bytes.get(0..2), Some([letter, b':']) if letter.is_ascii_alphabetic())
}

/// Whether the zip entry is a symbolic link (its unix mode says so).
fn is_symlink<R: std::io::Read + std::io::Seek>(file: &zip::read::ZipFile<'_, R>) -> bool {
    file.unix_mode()
        .is_some_and(|mode| mode & 0o170000 == 0o120000)
}

/// Map a zip error onto a named container error: an archive this reader
/// cannot decode (an encrypted entry, an unknown compression method) is
/// `Unsupported`; a structurally broken archive is `Malformed`.
fn map_zip_error(err: ZipError, context: &str) -> ContainerError {
    let detail = format!("{context}: {err}");
    match err {
        ZipError::UnsupportedArchive(_) => ContainerError::Unsupported { detail },
        _ => ContainerError::Malformed { detail },
    }
}

/// Lowercase-hex SHA-256 of the digested bytes.
fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(64);
    for byte in digest.as_ref() {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Whether the JSON chunk references a buffer that lives OUTSIDE the
/// container (a `uri` that is not a `data:` URI).
///
/// This is a JSON-string scan, not a JSON parse: the container contract
/// only asks whether some `uri` value points outside the container, and a
/// scan cannot be fooled by a deeply nested document into unbounded work.
fn has_external_buffer(json: &str) -> bool {
    let bytes = json.as_bytes();
    let mut index = 0usize;
    while index + 5 <= bytes.len() {
        if bytes[index..index + 5] == *b"\"uri\"" {
            if let Some(value) = json_uri_value(bytes, index + 5) {
                return !value.starts_with("data:");
            }
        }
        index = index.saturating_add(1);
    }
    false
}

/// The string value of the `uri` key that starts at `after_key`, if the
/// key is followed by a string value.
fn json_uri_value(bytes: &[u8], after_key: usize) -> Option<&str> {
    let mut index = after_key;
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index = index.saturating_add(1);
    }
    if bytes.get(index) != Some(&b':') {
        return None;
    }
    index = index.saturating_add(1);
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index = index.saturating_add(1);
    }
    if bytes.get(index) != Some(&b'"') {
        return None;
    }
    index = index.saturating_add(1);
    let start = index;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index = index.saturating_add(2),
            b'"' => return std::str::from_utf8(bytes.get(start..index)?).ok(),
            _ => index = index.saturating_add(1),
        }
    }
    None
}
