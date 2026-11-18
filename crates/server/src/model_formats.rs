//! Structural validation for the exportable 3D-model formats
//! (TASK-model-format-validation).
//!
//! The gate sits in front of every model-artifact export: bytes + the
//! claimed file extension go in, a [`ValidatedModel`] provenance record or
//! a named [`ModelReject`] comes out. Header/structure checks ONLY — no
//! in-process rendering, training, or decompression of splat payloads;
//! semantic validation stays with the external `gltf-transform`
//! inspect/validate step of the engine-bundle ticket.
//!
//! # Per-format depth (and its honest limits)
//!
//! - **glb** — the typed header gate below (magic, version 2, declared
//!   length sanity) plus the LANDED bounded container walk
//!   ([`crate::container::validate_glb`]): chunk layout, per-chunk and
//!   total budgets, and the external-buffer rule are that reader's
//!   contract, not duplicated here.
//! - **ply** — a pure-byte header parse: `format` line (ascii and
//!   `binary_little_endian` only; `binary_big_endian` is rejected with the
//!   encoding named), element/property grammar, declared-count budgets,
//!   and a declared-vs-present payload-size crosscheck. The crosscheck is
//!   EXACT only for binary files whose elements carry no list property
//!   (fixed row width); with list properties only a minimum size is
//!   derivable, and ASCII payloads have no derivable length at all — those
//!   are validated as far as the header allows and no further.
//! - **splat** — headerless by spec (the antimatter15/PlayCanvas layout is
//!   a fixed 32-byte record dump: position 3×f32, scale 3×f32, color 4×u8,
//!   quaternion 4×u8): the whole contract is a nonempty size that is a
//!   multiple of the 32-byte stride, within the registry budgets. Record
//!   CONTENT is not verifiable without a full parse; none happens here.
//! - **spz** — v4 plaintext header, verified against the reference
//!   implementation (nianticlabs/spz `load-spz.h`/`load-spz.cc`): NGSP
//!   magic, version 4, and the reference's own point-count bound
//!   (`(fileSize × 1024) / 9`, the compression-ratio ceiling the loader
//!   enforces). Legacy v1–3 files are gzip-wrapped with the header INSIDE
//!   the compressed stream: in-process validation stops at the gzip
//!   envelope (magic + minimum envelope size) and says so — the loader
//!   validates the inner header when it decompresses. `shDegree`,
//!   `flags`, `numStreams`, and `tocByteOffset` are read but only
//!   version/numPoints are enforced: they are advisory fields whose
//!   interpretation is version-negotiated by the loader.
//!
//! # Admission and naming
//!
//! The extension allowlist is [`crate::formats::MODEL_EXPORT_FORMATS`] —
//! the registry rows with `model_export`; this module never restates it.
//! `fbx`/`usd`/`obj` are `external_only` registry rows: a claim for one is
//! answered with [`ModelReject::ExternalOnly`] (code `model_external_only`)
//! and is NEVER converted or silently accepted. Identification is
//! magic-first: a claimed container whose bytes already identify as a
//! DIFFERENT registered format is rejected naming the SNIFFED format,
//! never the claimed extension (the registry's rejection convention).
//!
//! # Rejection codes (named, route-mappable)
//!
//! [`ModelReject::code`] yields the stable wire codes:
//! `unsupported_model_extension`, `model_external_only`,
//! `model_header_invalid`, `model_truncated`, `model_too_large`.
//!
//! # Wiring point (no route consumes the gate yet)
//!
//! The natural consumers — the model artifact ingest path and the
//! engine-bundle export (TASK-model-bundle-validation) — are not landed.
//! When that ticket wires it: call [`validate_model`] with the artifact
//! bytes and the upload's file extension, answer with
//! [`ModelReject::code`]/[`ModelReject::detail`], and write
//! [`validation_sidecar`] as `<artifact>.validation.json` beside the
//! artifact (the `downloads.rs` sidecar-suffix convention, `jobs.rs`
//! sidecar JSON shape). Until then everything here is
//! `#![allow(dead_code)]` in the server binary; the test suite exercises
//! every path via the `#[path]` include.

// No in-binary consumer yet: the gate wires into the model-bundle export
// path (TASK-model-bundle-validation). See the wiring-point section above.
#![allow(dead_code)]

use sha2::{Digest, Sha256};

use crate::container::{validate_glb, ContainerError, ContainerLimits};
use crate::formats::{self, FormatInfo};

/// Version of THIS validator, recorded in every sidecar so a consumer can
/// tell which structural contract produced a verdict.
pub(crate) const VALIDATOR_VERSION: &str = "model-formats/1";

/// Sidecar suffix: the validation record lands as
/// `<artifact>.validation.json` next to the artifact.
pub(crate) const SIDECAR_SUFFIX: &str = ".validation.json";

/// SPZ v4 header: 32 bytes, plaintext (verified against the reference
/// implementation's `static_assert(sizeof(NgspFileHeader) == 32)`).
const SPZ_HEADER_BYTES: usize = 32;
/// SPZ v4 header version (the reference's `LATEST_SPZ_HEADER_VERSION` ==
/// `MIN_ZSTD_SPZ_HEADER_VERSION` == 4).
const SPZ_VERSION: u32 = 4;
/// `numPoints` is consumed as `int32_t` downstream (the reference rejects
/// anything above `INT32_MAX` for exactly this reason); spelled as a const
/// so no cast is involved.
const SPZ_MAX_POINTS: u64 = 2_147_483_647;
/// The reference's generous compression-ratio ceiling
/// (`kMaxCompressionRatio`): a file of `len` bytes cannot honestly carry
/// more than `(len × 1024) / 9` points.
const SPZ_MAX_RATIO: u64 = 1024;
/// Minimum honest bytes per point (the reference's `kMinBytesPerPoint`):
/// the positions stream alone is 3 components × 3 bytes.
const SPZ_MIN_BYTES_PER_POINT: u64 = 9;
/// Minimum legacy gzip envelope: the fixed 10-byte gzip header plus the
/// 8-byte deflate trailer — anything shorter is cut short.
const SPZ_GZIP_MIN_BYTES: usize = 18;
/// The splat record stride (antimatter15/PlayCanvas layout), in bytes.
const SPLAT_RECORD_BYTES: u64 = 32;
/// PLY header bound: a legitimate PLY header is a few hundred bytes; the
/// bound keeps a crafted header from being an unbounded parse.
const PLY_HEADER_MAX: usize = 64 * 1024;
/// PLY element-declaration bound (the 64 KiB header bound already limits
/// the parse; this names the policy in the rejection instead).
const PLY_MAX_ELEMENTS: usize = 1024;

/// Provenance record for a validated artifact: everything the sidecar
/// records and the caller needs to label the artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValidatedModel {
    /// Canonical registry name of the format ("glb", "ply", "splat",
    /// "spz") — the sniffed/validated truth, never the claimed extension.
    pub format: &'static str,
    /// Format version when the header carries one: GLB version (always 2
    /// here), PLY format version (1), SPZ NGSP version (4). `None` for
    /// headerless splat records and gzip-wrapped legacy SPZ (the header is
    /// inside the compressed stream).
    pub format_version: Option<u32>,
    /// Lowercase-hex SHA-256 of the validated bytes (the established
    /// per-module digest form).
    pub sha256: String,
    /// Artifact size in bytes.
    pub size: u64,
}

/// Structured rejection with a stable named code — the gate's whole
/// admission vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModelReject {
    /// The claimed extension is not a model export format (unknown
    /// extension, or a known NON-model container such as `png`).
    UnsupportedExtension,
    /// The claimed extension names an `external_only` interchange format:
    /// conversion happens outside this server, never in-process.
    ExternalOnly {
        /// Canonical name of the external-only format ("fbx", "usd",
        /// "obj").
        format: &'static str,
    },
    /// The bytes are present but the container is structurally invalid
    /// (bad magic, bad version, inconsistent declared sizes with extra
    /// bytes present, broken chunk table, unknown directives).
    HeaderInvalid {
        /// What is wrong with the container.
        detail: String,
    },
    /// The container declares more bytes than are present (or ends
    /// mid-structure).
    Truncated {
        /// Which declared structure is cut short.
        detail: String,
    },
    /// A registry budget is exceeded (artifact bytes, declared element or
    /// point count).
    TooLarge {
        /// The budget that was exceeded.
        limit: u64,
        /// The declared or actual size.
        found: u64,
    },
}

impl ModelReject {
    /// The stable error code for the route layer.
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedExtension => "unsupported_model_extension",
            Self::ExternalOnly { .. } => "model_external_only",
            Self::HeaderInvalid { .. } => "model_header_invalid",
            Self::Truncated { .. } => "model_truncated",
            Self::TooLarge { .. } => "model_too_large",
        }
    }

    /// Human-readable detail for the error message — names what failed
    /// (the sniffed format, the declared structure, the budget), never the
    /// claimed extension.
    pub(crate) fn detail(&self) -> String {
        match self {
            Self::UnsupportedExtension => format!(
                "the exportable model formats are {}; fbx, usd and obj are external-only conversions",
                crate::formats::MODEL_EXPORT_FORMATS.join(", ")
            ),
            Self::ExternalOnly { format } => format!(
                "{format} is external-only: convert it to one of {} outside this server; \
                 it is never read or written in-process",
                crate::formats::MODEL_EXPORT_FORMATS.join(", ")
            ),
            Self::HeaderInvalid { detail } => {
                format!("the model container is invalid: {detail}")
            }
            Self::Truncated { detail } => {
                format!("the model container is truncated: {detail}")
            }
            Self::TooLarge { limit, found } => format!(
                "the model declares or holds {found}, exceeding the limit of {limit}"
            ),
        }
    }
}

/// Validate model-artifact bytes for export. `claimed_extension` is the
/// upload's file extension (leading dot optional, case-insensitive); the
/// BYTES decide identification (magic-first), the extension only selects
/// the expected container.
pub(crate) fn validate_model(
    bytes: &[u8],
    claimed_extension: &str,
) -> Result<ValidatedModel, ModelReject> {
    let extension = claimed_extension.trim_start_matches('.').to_ascii_lowercase();
    let Some(row) = formats::model_row_for_extension(&extension) else {
        return Err(ModelReject::UnsupportedExtension);
    };
    if row.external_only {
        return Err(ModelReject::ExternalOnly { format: row.name });
    }
    // Magic-first: bytes that already identify as a DIFFERENT registered
    // format reject with the sniffed name (the same `.glb` filename can
    // carry jpeg bytes).
    if let Some(other) = formats::sniff(bytes)
        && other.name != row.name
    {
        return Err(ModelReject::HeaderInvalid {
            detail: format!("the bytes are {}", other.name),
        });
    }
    validate_export_format(bytes, row)
}

/// Dispatch to the per-format structural validator and assemble the
/// provenance record. Registry budgets are checked here, before any
/// structure is parsed.
fn validate_export_format(
    bytes: &[u8],
    row: &'static FormatInfo,
) -> Result<ValidatedModel, ModelReject> {
    let size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if let Some(limit) = row.budget.uncompressed_bytes
        && size > limit
    {
        return Err(ModelReject::TooLarge { limit, found: size });
    }
    let format_version = match row.name {
        "glb" => validate_glb_artifact(bytes, row)?,
        "ply" => validate_ply(bytes, row)?,
        "splat" => validate_splat(bytes, row)?,
        "spz" => validate_spz(bytes, row)?,
        // Locked: `model_export` rows are exactly these four (the
        // allowlist invariant test fails first if the registry drifts).
        other => unreachable!("unhandled exportable model format {other}"),
    };
    Ok(ValidatedModel {
        format: row.name,
        format_version,
        sha256: sha256_hex(bytes),
        size,
    })
}

/// The GLB header gate: typed verdicts for the cases the ticket names
/// (truncation, bad magic, wrong version, absurd length), then the landed
/// bounded container walk for chunk layout, budgets, and the
/// external-buffer rule.
fn validate_glb_artifact(
    bytes: &[u8],
    row: &'static FormatInfo,
) -> Result<Option<u32>, ModelReject> {
    if bytes.len() < 12 {
        return Err(ModelReject::Truncated {
            detail: "the 12-byte glb header is truncated".to_owned(),
        });
    }
    if &bytes[0..4] != b"glTF" {
        return Err(ModelReject::HeaderInvalid {
            detail: "not a glb container (bad magic)".to_owned(),
        });
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("4 header bytes"));
    // The export gate validates glTF 2.0 — the exchange version the engine
    // bundles target; legacy glTF 1.0 GLBs name themselves in the reject.
    if version != 2 {
        return Err(ModelReject::HeaderInvalid {
            detail: format!(
                "unsupported glb version {version} (the export gate validates glTF 2.0)"
            ),
        });
    }
    let total = u32::from_le_bytes(bytes[8..12].try_into().expect("4 header bytes"));
    // A container with no room for even one chunk header (12-byte header +
    // 8-byte chunk header) cannot be walked.
    if total < 20 {
        return Err(ModelReject::HeaderInvalid {
            detail: format!(
                "the glb declares a {total}-byte container, too small for a chunk table"
            ),
        });
    }
    if usize::try_from(total).is_ok_and(|t| t > bytes.len()) {
        return Err(ModelReject::Truncated {
            detail: format!(
                "the glb declares {total} bytes but only {} are present",
                bytes.len()
            ),
        });
    }
    // Deep walk (chunk order, chunk types, size caps, external-buffer
    // rule) is the container reader's contract; a chunk whose OWN length
    // lies past the container end is an inconsistency, not a truncated
    // file — so `Malformed` maps to `model_header_invalid`, and the typed
    // truncation verdicts above carry `model_truncated`.
    let info = validate_glb(bytes, container_limits(row)).map_err(map_container_error)?;
    Ok(Some(info.version))
}

/// Map the bounded reader's verdicts onto the gate's named codes. Size
/// caps are budgets; everything else (malformed structure, unsupported
/// feature, an external-buffer reference) is a container the gate will
/// not call valid.
fn map_container_error(err: ContainerError) -> ModelReject {
    match err {
        ContainerError::TooManyEntries { limit, found }
        | ContainerError::EntryTooLarge {
            limit, found, ..
        }
        | ContainerError::TotalTooLarge { limit, found } => {
            ModelReject::TooLarge { limit, found }
        }
        other => ModelReject::HeaderInvalid {
            detail: other.to_string(),
        },
    }
}

/// Container budgets derived from the registry row: the whole-container
/// byte budget caps the total AND each chunk (a GLB's chunks ARE the
/// container), and the entry budget is the glb row's scene-entry ceiling.
fn container_limits(row: &'static FormatInfo) -> ContainerLimits {
    ContainerLimits {
        max_entries: row.budget.entries.unwrap_or(u64::MAX),
        max_entry_bytes: row.budget.uncompressed_bytes.unwrap_or(u64::MAX),
        max_total_bytes: row.budget.uncompressed_bytes.unwrap_or(u64::MAX),
    }
}

/// The PLY encoding declared by the header (big-endian is rejected at
/// parse time, so it never reaches the size checks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyEncoding {
    /// Text payload; no size crosscheck is derivable.
    Ascii,
    /// Binary payload, little-endian rows; size crosscheck applies.
    BinaryLittleEndian,
}

/// One PLY element declaration (header only — no payload is parsed).
#[derive(Debug)]
struct PlyElement {
    /// Declared element count.
    count: u64,
    /// Exact per-record payload width in bytes; `None` when a list
    /// property makes the row variable-width.
    fixed_width: Option<u64>,
    /// Minimum per-record payload width (fixed properties plus each
    /// list's count field).
    min_width: u64,
}

/// The PLY gate: header grammar, declared-count budgets, and (for binary
/// payloads) the declared-vs-present size crosscheck.
fn validate_ply(bytes: &[u8], row: &'static FormatInfo) -> Result<Option<u32>, ModelReject> {
    let header_end = ply_header_end(bytes)?;
    let (encoding, elements) = parse_ply_header(&bytes[..header_end], row)?;

    // Declared sizes vs file size — exact where the header pins the row
    // width, minimum-only when list properties make rows variable, and
    // (honestly) absent for ASCII, whose payload length is not derivable
    // from the header.
    if encoding == PlyEncoding::BinaryLittleEndian {
        let mut exact = Some(0u64);
        let mut min = 0u64;
        for element in &elements {
            let count = element.count;
            if let Some(width) = element.fixed_width {
                exact = exact.map(|e| e.saturating_add(width.saturating_mul(count)));
                min = min.saturating_add(width.saturating_mul(count));
            } else {
                exact = None;
                min = min.saturating_add(element.min_width.saturating_mul(count));
            }
        }
        let present = u64::try_from(bytes.len() - header_end).unwrap_or(u64::MAX);
        if let Some(expected) = exact {
            if present < expected {
                return Err(ModelReject::Truncated {
                    detail: format!(
                        "the ply declares {expected} payload bytes but only {present} follow the header"
                    ),
                });
            }
            if present > expected {
                return Err(ModelReject::HeaderInvalid {
                    detail: format!(
                        "the ply declares {expected} payload bytes but {present} follow the header"
                    ),
                });
            }
        } else if present < min {
            return Err(ModelReject::Truncated {
                detail: format!(
                    "the ply declares elements needing at least {min} payload bytes but only {present} follow the header"
                ),
            });
        }
    }
    Ok(Some(1))
}

/// Find the end of the PLY header (the `end_header` line). A header that
/// runs past [`PLY_HEADER_MAX`] is a bound rejection; a file that ends
/// before the header completes is truncated.
fn ply_header_end(bytes: &[u8]) -> Result<usize, ModelReject> {
    if bytes.len() > PLY_HEADER_MAX {
        // Only the header prefix is examined; a file larger than the bound
        // whose first 64 KiB hold no `end_header` is rejected on the bound.
        if let Some(end) = ply_header_end_within(&bytes[..PLY_HEADER_MAX]) {
            return Ok(end);
        }
        return Err(ModelReject::HeaderInvalid {
            detail: format!(
                "the ply header exceeds the {} KiB bound",
                PLY_HEADER_MAX / 1024
            ),
        });
    }
    ply_header_end_within(bytes).ok_or_else(|| ModelReject::Truncated {
        detail: "the ply header never completes (no end_header within the bytes present)"
            .to_owned(),
    })
}

/// Line-scan for `end_header` within `bytes`; returns the offset just
/// past the terminating newline (CRLF tolerated).
fn ply_header_end_within(bytes: &[u8]) -> Option<usize> {
    let mut pos = 0usize;
    while pos < bytes.len() {
        let end = pos + bytes[pos..].iter().position(|&b| b == b'\n')? + 1;
        let mut line_end = end - 1; // drop the LF
        if line_end > pos && bytes[line_end - 1] == b'\r' {
            line_end -= 1; // drop the CR of a CRLF
        }
        if &bytes[pos..line_end] == b"end_header" {
            return Some(end);
        }
        pos = end;
    }
    None
}

/// Parse the PLY header grammar: `format` line, `element`/`property`
/// declarations, `comment`/`obj_info` skips, `end_header` terminator.
/// Unknown directives are REJECTED (naming the directive): the gate is a
/// structural allowlist, not a best-effort reader.
#[allow(clippy::too_many_lines)] // one linear header pass: grammar, budgets, counts
fn parse_ply_header(
    header: &[u8],
    row: &'static FormatInfo,
) -> Result<(PlyEncoding, Vec<PlyElement>), ModelReject> {
    let mut encoding: Option<PlyEncoding> = None;
    let mut elements: Vec<PlyElement> = Vec::new();
    let mut declared_total: u64 = 0;
    let mut seen_end_header = false;

    for line in header.split(|&b| b == b'\n') {
        let line = strip_cr(line);
        if line.is_empty() {
            continue;
        }
        if line == b"end_header" {
            seen_end_header = true;
            break;
        }
        // Tokens separate on spaces AND tabs (both are legal PLY header
        // whitespace; a tab-separated header must not over-reject).
        let tokens: Vec<&[u8]> = line
            .split(|&b| b == b' ' || b == b'\t')
            .filter(|t| !t.is_empty())
            .collect();
        match tokens[0] {
            b"ply" if elements.is_empty() && encoding.is_none() && tokens.len() == 1 => {}
            b"format" => {
                if encoding.is_some() || tokens.len() != 3 {
                    return Err(ModelReject::HeaderInvalid {
                        detail: "malformed ply format line".to_owned(),
                    });
                }
                if tokens[2] != b"1.0" {
                    return Err(ModelReject::HeaderInvalid {
                        detail: format!(
                            "unsupported ply format version {}",
                            String::from_utf8_lossy(tokens[2])
                        ),
                    });
                }
                encoding = Some(match tokens[1] {
                    b"ascii" => PlyEncoding::Ascii,
                    b"binary_little_endian" => PlyEncoding::BinaryLittleEndian,
                    b"binary_big_endian" => {
                        return Err(ModelReject::HeaderInvalid {
                            detail: "binary_big_endian ply is not accepted (only ascii and \
                                     binary_little_endian)"
                                .to_owned(),
                        })
                    }
                    other => {
                        return Err(ModelReject::HeaderInvalid {
                            detail: format!(
                                "unknown ply encoding {}",
                                String::from_utf8_lossy(other)
                            ),
                        })
                    }
                });
            }
            b"element" => {
                if tokens.len() != 3 {
                    return Err(ModelReject::HeaderInvalid {
                        detail: "malformed ply element line".to_owned(),
                    });
                }
                let count = std::str::from_utf8(tokens[2])
                    .ok()
                    .and_then(|c| c.parse::<u64>().ok())
                    .ok_or_else(|| ModelReject::HeaderInvalid {
                        detail: format!(
                            "element {} declares a malformed count",
                            String::from_utf8_lossy(tokens[1])
                        ),
                    })?;
                declared_total = declared_total.saturating_add(count);
                if let Some(limit) = row.budget.entries
                    && declared_total > limit
                {
                    return Err(ModelReject::TooLarge {
                        limit,
                        found: declared_total,
                    });
                }
                if elements.len() >= PLY_MAX_ELEMENTS {
                    return Err(ModelReject::HeaderInvalid {
                        detail: format!(
                            "the ply declares more than {PLY_MAX_ELEMENTS} elements"
                        ),
                    });
                }
                elements.push(PlyElement {
                    count,
                    fixed_width: Some(0),
                    min_width: 0,
                });
            }
            b"property" => {
                let Some(element) = elements.last_mut() else {
                    return Err(ModelReject::HeaderInvalid {
                        detail: "a ply property precedes any element".to_owned(),
                    });
                };
                match tokens.get(1).copied() {
                    // `property list <count_type> <item_type> <name>`: the
                    // row becomes variable-width; its MINIMUM carries the
                    // count field's width.
                    Some(b"list") => {
                        let [_prop, _list, count_type, _item_type, _name] =
                            tokens.as_slice()
                        else {
                            return Err(ModelReject::HeaderInvalid {
                                detail: "malformed ply list property line".to_owned(),
                            });
                        };
                        let count_width = ply_scalar_width(count_type).ok_or_else(|| {
                            ModelReject::HeaderInvalid {
                                detail: format!(
                                    "unknown ply list count type {}",
                                    String::from_utf8_lossy(count_type)
                                ),
                            }
                        })?;
                        element.fixed_width = None;
                        element.min_width = element.min_width.saturating_add(count_width);
                    }
                    // `property <type> <name>`: a fixed-width scalar.
                    Some(type_name) if tokens.len() == 3 => {
                        let width = ply_scalar_width(type_name).ok_or_else(|| {
                            ModelReject::HeaderInvalid {
                                detail: format!(
                                    "unknown ply property type {}",
                                    String::from_utf8_lossy(type_name)
                                ),
                            }
                        })?;
                        if let Some(fixed) = element.fixed_width {
                            element.fixed_width = Some(fixed.saturating_add(width));
                        }
                        element.min_width = element.min_width.saturating_add(width);
                    }
                    _ => {
                        return Err(ModelReject::HeaderInvalid {
                            detail: "malformed ply property line".to_owned(),
                        })
                    }
                }
            }
            b"comment" | b"obj_info" => {}
            other => {
                return Err(ModelReject::HeaderInvalid {
                    detail: format!(
                        "unrecognized ply header directive {}",
                        String::from_utf8_lossy(other)
                    ),
                })
            }
        }
    }

    if !seen_end_header {
        return Err(ModelReject::HeaderInvalid {
            detail: "the ply header lacks an end_header line".to_owned(),
        });
    }
    let Some(encoding) = encoding else {
        return Err(ModelReject::HeaderInvalid {
            detail: "the ply header lacks a format line".to_owned(),
        });
    };
    if elements.is_empty() {
        return Err(ModelReject::HeaderInvalid {
            detail: "the ply declares no elements".to_owned(),
        });
    }
    Ok((encoding, elements))
}

/// Strip a trailing CR (CRLF headers are in the wild).
fn strip_cr(line: &[u8]) -> &[u8] {
    match line {
        [rest @ .., b'\r'] => rest,
        _ => line,
    }
}

/// PLY scalar type widths (the spec's fixed-width scalar set; list count
/// types come from the same table).
fn ply_scalar_width(name: &[u8]) -> Option<u64> {
    match name {
        b"char" | b"int8" | b"uchar" | b"uint8" => Some(1),
        b"short" | b"int16" | b"ushort" | b"uint16" => Some(2),
        b"int" | b"uint" | b"int32" | b"uint32" | b"float" | b"float32" => Some(4),
        b"double" | b"float64" => Some(8),
        _ => None,
    }
}

/// The splat gate. The format is headerless by spec (verified against the
/// antimatter15/PlayCanvas layout): a fixed 32-byte record dump, so the
/// whole structural contract is a nonempty stride multiple within the
/// registry budgets. Record content is NOT verifiable without a full
/// parse; none happens here (documented honesty, not an oversight).
fn validate_splat(bytes: &[u8], row: &'static FormatInfo) -> Result<Option<u32>, ModelReject> {
    let size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if size == 0 {
        return Err(ModelReject::HeaderInvalid {
            detail: "the splat container holds no records".to_owned(),
        });
    }
    if size % SPLAT_RECORD_BYTES != 0 {
        return Err(ModelReject::HeaderInvalid {
            detail: format!(
                "the splat size {size} is not a multiple of the {SPLAT_RECORD_BYTES}-byte record stride"
            ),
        });
    }
    let count = size / SPLAT_RECORD_BYTES;
    if let Some(limit) = row.budget.entries
        && count > limit
    {
        return Err(ModelReject::TooLarge { limit, found: count });
    }
    Ok(None)
}

/// The SPZ gate. v4 plaintext NGSP header (verified against the reference
/// implementation): magic, version, and the reference's own point-count
/// bound. Legacy v1–3 files are gzip-wrapped with the header INSIDE the
/// compressed stream — validation stops at the envelope, honestly.
fn validate_spz(bytes: &[u8], row: &'static FormatInfo) -> Result<Option<u32>, ModelReject> {
    if bytes.len() >= 4 && &bytes[0..4] == b"NGSP" {
        if bytes.len() < SPZ_HEADER_BYTES {
            return Err(ModelReject::Truncated {
                detail: format!("the {SPZ_HEADER_BYTES}-byte spz header is truncated"),
            });
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().expect("4 header bytes"));
        if version != SPZ_VERSION {
            return Err(ModelReject::HeaderInvalid {
                detail: format!(
                    "unsupported spz version {version} (the export gate validates NGSP v{SPZ_VERSION})"
                ),
            });
        }
        let num_points = u64::from(u32::from_le_bytes(
            bytes[8..12].try_into().expect("4 header bytes"),
        ));
        if num_points == 0 {
            return Err(ModelReject::HeaderInvalid {
                detail: "the spz header declares zero points".to_owned(),
            });
        }
        if num_points > SPZ_MAX_POINTS {
            return Err(ModelReject::HeaderInvalid {
                detail: format!(
                    "the spz header claims {num_points} points, above the {SPZ_MAX_POINTS}-point container bound"
                ),
            });
        }
        // The reference's own bound: a file of `size` bytes cannot
        // honestly claim more points than `(size × 1024) / 9`.
        let size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        let bound = size.saturating_mul(SPZ_MAX_RATIO) / SPZ_MIN_BYTES_PER_POINT;
        if num_points > bound {
            return Err(ModelReject::HeaderInvalid {
                detail: format!(
                    "the spz header claims {num_points} points but {size} bytes admit at most {bound}"
                ),
            });
        }
        if let Some(limit) = row.budget.entries
            && num_points > limit
        {
            return Err(ModelReject::TooLarge {
                limit,
                found: num_points,
            });
        }
        Ok(Some(version))
    } else if bytes.len() >= 3 && bytes[0] == 0x1f && bytes[1] == 0x8b && bytes[2] == 0x08 {
        // Legacy v1–3: the 16-byte header lives inside the gzip stream, so
        // in-process structural validation ends here. The inner header is
        // validated by the loader when it decompresses; this gate does not
        // decompress (no in-process splat handling, per ticket).
        if bytes.len() < SPZ_GZIP_MIN_BYTES {
            return Err(ModelReject::Truncated {
                detail: "the gzip envelope is truncated (a gzip stream carries at least a \
                         10-byte header and an 8-byte trailer)"
                    .to_owned(),
            });
        }
        Ok(None)
    } else {
        Err(ModelReject::HeaderInvalid {
            detail: "the bytes are not an spz container (no NGSP header, no gzip envelope)"
                .to_owned(),
        })
    }
}

/// The sidecar JSON for a validated artifact, written as
/// `<artifact>.validation.json` beside it (downloads.rs suffix
/// convention, jobs.rs JSON shape). Format version is `null` when the
/// header does not carry one.
pub(crate) fn validation_sidecar(validated: &ValidatedModel) -> serde_json::Value {
    serde_json::json!({
        "format": validated.format,
        "format_version": validated.format_version,
        "validator_version": VALIDATOR_VERSION,
        "sha256": validated.sha256,
        "size_bytes": validated.size,
    })
}

/// Lowercase-hex SHA-256 — the established per-module digest form
/// (svg.rs/ora.rs/determinism.rs all carry the same five lines so each
/// module stays standalone under `#[path]` includes).
fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        let _ = write!(out, "{byte:02x}");
    }
    out
}
