//! Binary format registry: the single per-format source of truth for what
//! this build can read and write, derived from the codecs actually compiled
//! in (workspace `image` features + the `honkbox-sd-core` encoders), not a
//! hand-maintained prose list.
//!
//! One row per canonical format name maps it to its [`FormatKind`], IANA
//! MIME type, file extensions (advisory metadata — identification is ALWAYS
//! by magic bytes), leading-byte signatures, read/write capability, and
//! byte/size budgets. Every consumer — the `/v1/capabilities` `formats`
//! section and the route validators — queries this table, so the advertised
//! format set can never drift from the accepted format set:
//!
//! - `formats.read` = rows with `sd_image_ingest` (what the SD ingest
//!   validators accept: edits `image`/`mask`, `ControlNet` `control_image`);
//! - `formats.write` = rows with `output` (the compiled encoder set,
//!   locked to the `honkbox_sd_core::OutputFormat` enum by a routes-level
//!   unit test — the enum mapping itself stays at the routes layer, see
//!   `FormatInfo::output`);
//! - ingest validators identify bytes with [`sniff`]/[`identify_png`] and
//!   reject unknown/mismatched containers with named errors;
//! - output validators accept exactly the `output` rows (the wire name →
//!   `OutputFormat` enum mapping lives at the routes call site);
//! - the model export gate accepts exactly the `model_export` rows — the
//!   [`MODEL_EXPORT_FORMATS`] allowlist shared with the structural
//!   validator ([`crate::model_formats`]); `fbx`/`usd`/`obj` are
//!   `external_only` (external conversion only, never silently accepted);
//!
//! # Compile-time codec derivation
//!
//! The `const _: Option<…>` probes below name one decoder and one encoder
//! type per raster format. Those types only exist when the workspace
//! `Cargo.toml` enables the same-named `image` feature, so flipping a
//! feature off breaks THIS file's compilation and forces the capability
//! columns to be revisited — the table cannot silently drift from the
//! build. The `output` column additionally holds the concrete
//! same codecs sd-core's encode path wraps. The write column is pure data
//! (`output`/`output_compression` bools) and the routes layer owns the
//! wire-name → enum mapping, locked to `OutputFormat::ALL` by a unit test —
//! this module references no sd-core symbols at all, so including it in a
//! test binary never links the native backends.
//!
//! # Enforcement (where each budget is applied)
//!
//! - The image-ops pixel cap (`decoded_pixels` on the accepted rows, the
//!   same `16_777_216` line the route enforced inline before this
//!   registry) is checked after decode, so it also covers a rasterized
//!   vector or a flattened layered document.
//! - **TASK-container-safety** landed the bounded archive reader
//!   ([`crate::container`]): the `entries` / `uncompressed_bytes` /
//!   `entry_bytes` budgets are the limits its reader enforces, and the
//!   `ora` row is the one that feeds them.
//! - **TASK-svg-ora-support** landed the vector reader ([`crate::svg`],
//!   [`crate::ora`]): the `svg` row's `elements` / `uncompressed_bytes`
//!   budgets are the limits the SVG validator enforces, and the `ora`
//!   row's budgets are the limits the ORA reader enforces over the
//!   bounded container.
//! - The `image_ops_input` column IS the admission rule for
//!   `/v1/image-ops`: flipping it on for `svg`/`ora` is what makes the
//!   route accept those containers (the magic sniff still has the last
//!   word, so a mismatched declaration stays `format_mismatch`).
//!   Rows with no capability (`false`/`None` everywhere) are inert: they
//!   appear in no advertisement and no validator accepts them.
//!
//! # Rejection codes
//!
//! [`FormatReject`] carries the structured verdict; call sites attach the
//! stable error code. Canonical pairing: an unrecognized container is
//! `unsupported_format`, a signature that identifies a DIFFERENT format is
//! `format_mismatch`. Established route codes are kept for wire
//! compatibility (`invalid_image`/`invalid_mask`/`invalid_control_image`
//! on the SD ingest path, `invalid_image_op_inputs` on image-ops);
//! `format_mismatch` is used wherever a declaration can actually disagree
//! with the bytes (image-ops data URIs). Every message names the sniffed
//! format, never the claimed extension.

/// Compile-time decoder probes (see the module docs). Each `const` names a
/// codec type gated behind the same-named `image` feature; removing the
/// feature from the workspace is a compile error here, not a silent
/// capability lie.
const _: Option<image::codecs::png::PngDecoder<std::io::Empty>> = None;
const _: Option<image::codecs::jpeg::JpegDecoder<std::io::Empty>> = None;
const _: Option<image::codecs::webp::WebPDecoder<std::io::Empty>> = None;
const _: Option<image::codecs::png::PngEncoder<std::io::Sink>> = None;
const _: Option<image::codecs::jpeg::JpegEncoder<std::io::Sink>> = None;
const _: Option<image::codecs::webp::WebPEncoder<std::io::Sink>> = None;

/// What family a container belongs to — the registry's grouping for the
/// capabilities advertisement, route admission, and the budget semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FormatKind {
    /// Flat raster images (the compiled `image` crate containers).
    Image,
    /// Per-frame interchange containers for the (future) video pipeline.
    VideoFrame,
    /// Vector drawing (XML).
    Svg,
    /// Layered raster document (zip-based, e.g. `OpenRaster`).
    Ora,
    /// 3D scene containers (binary glTF here; the external-only
    /// interchange rows `fbx`/`usd`/`obj` share the family).
    Glb,
    /// Gaussian-splat point cloud.
    Splat,
    /// Opaque passthrough bytes (no identification contract).
    Blob,
}

impl FormatKind {
    /// Human-readable family label for rejection messages.
    pub(crate) fn label(self) -> &'static str {
        match self {
            FormatKind::Image => "raster image",
            FormatKind::VideoFrame => "video frame",
            FormatKind::Svg => "vector drawing",
            FormatKind::Ora => "layered raster document",
            FormatKind::Glb => "3D scene",
            FormatKind::Splat => "gaussian splat",
            FormatKind::Blob => "opaque blob",
        }
    }
}

/// Byte/size budget for one format (decoded pixels, container entries,
/// uncompressed bytes, per-entry bytes, vector elements). Every budget
/// here is ENFORCED by the read path that owns the format: the raster
/// rows by the image-ops pixel cap and the decoder limits, the `svg` row
/// by [`crate::svg`], the `ora` row by [`crate::ora`] over the bounded
/// [`crate::container`] reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FormatBudget {
    /// Ceiling on decoded pixels (`width * height`) once the container
    /// decompresses. `None` = not applicable (a format with no raster
    /// canvas).
    pub decoded_pixels: Option<u64>,
    /// Ceiling on container elements (zip entries for `ora`, scene nodes
    /// for `glb`, vertices for splats).
    pub entries: Option<u64>,
    /// Ceiling on uncompressed/decompressed bytes (the whole source for a
    /// flat container, the summed entry bytes for a zip).
    pub uncompressed_bytes: Option<u64>,
    /// Ceiling on ONE entry's uncompressed bytes (container reads). `None` = not applicable.
    pub entry_bytes: Option<u64>,
    /// Ceiling on the element count of a vector document (SVG). `None` = not applicable.
    pub elements: Option<u64>,
}

/// One leading-byte signature: a list of `(offset, bytes)` pairs that must
/// ALL match. Multiple signatures per row are alternatives (e.g. SVG with
/// and without a BOM).
type Signature = &'static [(usize, &'static [u8])];

/// One registry row: canonical identity + capability + budget.
// The six capability columns are independent route-surface flags — the
// registry IS a capability table, not a state machine — so the bool count
// is the domain, not a design smell.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FormatInfo {
    /// Family (grouping only — capabilities are the columns below).
    pub kind: FormatKind,
    /// Canonical wire name (`formats.read` entries, error messages).
    pub name: &'static str,
    /// IANA-ish media type (data-URI declarations, HTTP responses).
    pub mime: &'static str,
    /// File extensions (advisory only; identification is magic-first).
    pub extensions: &'static [&'static str],
    /// Leading-byte signatures; empty = no reliable magic (never matches
    /// [`sniff`] — admission for such rows is by explicit policy, not
    /// sniffing).
    pub magic: &'static [Signature],
    /// Byte/size budget (see [`FormatBudget`]).
    pub budget: FormatBudget,
    /// `true` when the SD image ingest validators accept these bytes
    /// (edits `image`/`mask`, `ControlNet` `control_image`) — exactly what
    /// `Capabilities::formats.read` advertises.
    pub sd_image_ingest: bool,
    /// `true` when `/v1/image-ops` accepts these bytes as inputs (data URIs
    /// and bundle artifacts).
    pub image_ops_input: bool,
    /// `true` when this build can produce the container — the compiled
    /// `image` codecs behind `honkbox_sd_core`'s encode path, locked to
    /// `OutputFormat::ALL` by a routes-level unit test. The enum handle
    /// itself lives at the routes layer: a `#[path]`-included registry
    /// module must not link the native backends (test binaries include it
    /// beside the llama.cpp modules, and the two vendored ggml copies
    /// collide on duplicate symbols).
    pub output: bool,
    /// `true` when `output_compression` affects this container (JPEG
    /// quality only; PNG is deterministic and WebP is lossless-only).
    pub output_compression: bool,
    /// `true` when this 3D-model container is in the export allowlist
    /// ([`MODEL_EXPORT_FORMATS`], locked by test): the bytes are
    /// structurally validatable in-process by
    /// [`crate::model_formats`] and may be exported to engine bundles.
    pub model_export: bool,
    /// `true` when this row is an interchange format served by EXTERNAL
    /// conversion only: it is never read, never written, never silently
    /// accepted in-process — admission answers with a named rejection
    /// naming the conversion policy (`model_external_only`).
    pub external_only: bool,
}

/// Decoded-pixel ceiling for the accepted raster containers (4096²): the
/// image-ops decode path rejects every input above it (reject, never
/// downscale). Formerly `routes::IMAGE_OP_MAX_PIXELS`.
pub(crate) const MAX_DECODED_PIXELS: u64 = 16_777_216;
/// Decoder-level edge cap (`image::Limits` strict caps): no decoded edge
/// above this, so a small compressed container cannot decode to an absurd
/// canvas before the pixel budget is checked.
pub(crate) const MAX_DECODED_EDGE: u32 = 16_384;
/// Decoder-level allocation allowance (`image::Limits::max_alloc`): room
/// for the RGBA of the pixel cap plus intermediates.
pub(crate) const MAX_DECODED_ALLOC_BYTES: u64 = 128 * 1024 * 1024;

/// Raster budget shared by every accepted container (flat images and
/// frames): the enforced pixel cap + the decoder allocation allowance.
const RASTER_BUDGET: FormatBudget = FormatBudget {
    decoded_pixels: Some(MAX_DECODED_PIXELS),
    entries: None,
    uncompressed_bytes: Some(MAX_DECODED_ALLOC_BYTES),
    entry_bytes: None,
    elements: None,
};

/// PNG: every stream opens with the 8-byte signature; the IHDR chunk
/// carrying dimensions sits at a fixed offset right behind it.
const PNG_SIG: Signature = &[(0, b"\x89PNG\r\n\x1a\n")];
/// JPEG: all streams open with the SOI marker.
const JPEG_SIG: Signature = &[(0, b"\xFF\xD8\xFF")];
/// WebP: a RIFF container whose form type at offset 8 is `WEBP`.
const WEBP_SIG: Signature = &[(0, b"RIFF"), (8, b"WEBP")];
/// Netpbm frames: `P6` (pixmap) / `P5` (graymap) followed by whitespace.
const PPM_SIG: Signature = &[(0, b"P6")];
const PGM_SIG: Signature = &[(0, b"P5")];
/// SVG: XML — with or without a UTF-8 BOM, root `<svg` or an XML prolog.
const SVG_SIGS: &[Signature] = &[
    &[(0, b"<svg")],
    &[(0, b"<?xml")],
    &[(0, "\u{FEFF}".as_bytes()), (3, b"<svg")],
    &[(0, "\u{FEFF}".as_bytes()), (3, b"<?xml")],
];
/// `OpenRaster`: a zip container (local-file header); the `mimetype` entry
/// requirement is checked by the ora ticket, not by this signature.
const ORA_SIG: Signature = &[(0, b"PK\x03\x04")];
/// Binary glTF: the 4-byte `glTF` magic (version follows at offset 4).
const GLB_SIG: Signature = &[(0, b"glTF")];
/// Stanford PLY: an ASCII header line `ply\n` (binary payloads too).
const PLY_SIG: Signature = &[(0, b"ply\n")];
/// SPZ (Niantic), v4 plaintext header: the file OPENS with the 4-byte
/// NGSP magic (`0x5053_474e` little-endian). Legacy v1–3 files are
/// gzip-wrapped (`1f 8b 08`) with the header inside the compressed
/// stream — gzip is a generic envelope, not an SPZ identity, so those
/// files never match [`sniff`] and are admitted by explicit policy
/// (the claimed `.spz` extension) in [`crate::model_formats`].
const SPZ_SIG: Signature = &[(0, b"NGSP")];
/// Autodesk FBX binary: the Kaydara signature. ASCII FBX has no
/// reliable magic and never matches [`sniff`].
const FBX_SIG: Signature = &[(0, b"Kaydara FBX Binary  \x00\x1a\x00")];

/// The registry. Order is the advertisement order (`formats.write` derives
/// from the `output` column in this order, locked to `OutputFormat::ALL`).
///
/// Accepted today: `png` (SD ingest + image-ops) and `jpeg`/`webp`
/// (image-ops inputs + encoders). The 3D-model rows are pure structural
/// rows — no codec probe applies to them: `glb`/`ply`/`splat`/`spz` carry
/// `model_export` (the [`MODEL_EXPORT_FORMATS`] allowlist, structurally
/// validated by [`crate::model_formats`]), while `fbx`/`usd`/`obj` are
/// `external_only` interchange rows — never read, never written, never
/// silently accepted in-process. Every other row is a designed extension
/// point: capabilities stay `false`/`None` until its owning ticket wires
/// the codec and flips the columns — flipping is the whole integration.
static FORMATS: [FormatInfo; 15] = [
    FormatInfo {
        kind: FormatKind::Image,
        name: "png",
        mime: "image/png",
        extensions: &["png"],
        magic: &[PNG_SIG],
        budget: RASTER_BUDGET,
        sd_image_ingest: true,
        image_ops_input: true,
        output: true,
        output_compression: false,
        model_export: false,
        external_only: false,
    },
    FormatInfo {
        kind: FormatKind::Image,
        name: "jpeg",
        mime: "image/jpeg",
        extensions: &["jpeg", "jpg"],
        magic: &[JPEG_SIG],
        budget: RASTER_BUDGET,
        sd_image_ingest: false,
        image_ops_input: true,
        output: true,
        output_compression: true,
        model_export: false,
        external_only: false,
    },
    FormatInfo {
        kind: FormatKind::Image,
        name: "webp",
        mime: "image/webp",
        extensions: &["webp"],
        magic: &[WEBP_SIG],
        budget: RASTER_BUDGET,
        sd_image_ingest: false,
        image_ops_input: true,
        output: true,
        output_compression: false,
        model_export: false,
        external_only: false,
    },
    FormatInfo {
        kind: FormatKind::VideoFrame,
        name: "ppm",
        mime: "image/x-portable-pixmap",
        extensions: &["ppm", "pnm"],
        magic: &[PPM_SIG],
        budget: RASTER_BUDGET,
        sd_image_ingest: false,
        image_ops_input: false,
        output: false,
        output_compression: false,
        model_export: false,
        external_only: false,
    },
    FormatInfo {
        kind: FormatKind::VideoFrame,
        name: "pgm",
        mime: "image/x-portable-graymap",
        extensions: &["pgm", "pnm"],
        magic: &[PGM_SIG],
        budget: RASTER_BUDGET,
        sd_image_ingest: false,
        image_ops_input: false,
        output: false,
        output_compression: false,
        model_export: false,
        external_only: false,
    },
    FormatInfo {
        kind: FormatKind::Svg,
        name: "svg",
        mime: "image/svg+xml",
        extensions: &["svg"],
        magic: SVG_SIGS,
        budget: FormatBudget {
            // A vector document has a raster budget too: the rasterizer
            // produces pixels, so the same pixel cap binds.
            decoded_pixels: Some(MAX_DECODED_PIXELS),
            entries: None,
            // Source-text ceiling (the SVG source, not a decoded canvas).
            uncompressed_bytes: Some(16 * 1024 * 1024),
            entry_bytes: None,
            // Element-count ceiling for the XML tree.
            elements: Some(100_000),
        },
        sd_image_ingest: false,
        image_ops_input: true,
        output: false,
        output_compression: false,
        model_export: false,
        external_only: false,
    },
    FormatInfo {
        kind: FormatKind::Ora,
        name: "ora",
        mime: "image/openraster",
        extensions: &["ora"],
        magic: &[ORA_SIG],
        budget: FormatBudget {
            // Layered canvas budget: the flattened canvas is a raster.
            decoded_pixels: Some(MAX_DECODED_PIXELS),
            // The archive's entry ceiling (also the layer ceiling: the
            // stack's layers are archive entries).
            entries: Some(4096),
            // Total uncompressed bytes across the archive.
            uncompressed_bytes: Some(512 * 1024 * 1024),
            // Ceiling on ONE archive entry (a layer PNG).
            entry_bytes: Some(64 * 1024 * 1024),
            elements: None,
        },
        sd_image_ingest: false,
        image_ops_input: true,
        output: false,
        output_compression: false,
        model_export: false,
        external_only: false,
    },
    FormatInfo {
        kind: FormatKind::Glb,
        name: "glb",
        mime: "model/gltf-binary",
        extensions: &["glb"],
        magic: &[GLB_SIG],
        budget: FormatBudget {
            decoded_pixels: None,
            entries: Some(65_536),
            uncompressed_bytes: Some(512 * 1024 * 1024),
            entry_bytes: None,
            elements: None,
        },
        sd_image_ingest: false,
        image_ops_input: false,
        output: false,
        output_compression: false,
        model_export: true,
        external_only: false,
    },
    FormatInfo {
        kind: FormatKind::Splat,
        name: "ply",
        mime: "application/octet-stream",
        extensions: &["ply"],
        magic: &[PLY_SIG],
        budget: FormatBudget {
            decoded_pixels: None,
            // Vertex ceiling, reserved for the splats epic.
            entries: Some(16_777_216),
            uncompressed_bytes: Some(512 * 1024 * 1024),
            entry_bytes: None,
            elements: None,
        },
        sd_image_ingest: false,
        image_ops_input: false,
        output: false,
        output_compression: false,
        model_export: true,
        external_only: false,
    },
    FormatInfo {
        kind: FormatKind::Splat,
        name: "splat",
        mime: "application/octet-stream",
        extensions: &["splat"],
        // Headerless fixed-stride record dump: no reliable magic, so it
        // never matches `sniff` — admission is by explicit policy only.
        magic: &[],
        budget: FormatBudget {
            decoded_pixels: None,
            entries: Some(16_777_216),
            uncompressed_bytes: Some(512 * 1024 * 1024),
            entry_bytes: None,
            elements: None,
        },
        sd_image_ingest: false,
        image_ops_input: false,
        output: false,
        output_compression: false,
        model_export: true,
        external_only: false,
    },
    FormatInfo {
        kind: FormatKind::Splat,
        name: "spz",
        mime: "application/octet-stream",
        extensions: &["spz"],
        // v4 NGSP plaintext header only (see [`SPZ_SIG`]): legacy v1–3
        // files travel in a generic gzip envelope, which is deliberately
        // NOT an identity here — those are admitted by explicit policy
        // (the claimed `.spz` extension) in [`crate::model_formats`].
        magic: &[SPZ_SIG],
        budget: FormatBudget {
            decoded_pixels: None,
            // Point ceiling, shared with the ply/splat point clouds.
            entries: Some(16_777_216),
            uncompressed_bytes: Some(512 * 1024 * 1024),
            entry_bytes: None,
            elements: None,
        },
        sd_image_ingest: false,
        image_ops_input: false,
        output: false,
        output_compression: false,
        model_export: true,
        external_only: false,
    },
    FormatInfo {
        kind: FormatKind::Glb,
        name: "fbx",
        mime: "application/octet-stream",
        extensions: &["fbx"],
        magic: &[FBX_SIG],
        budget: FormatBudget {
            decoded_pixels: None,
            entries: Some(65_536),
            uncompressed_bytes: Some(512 * 1024 * 1024),
            entry_bytes: None,
            elements: None,
        },
        sd_image_ingest: false,
        image_ops_input: false,
        output: false,
        output_compression: false,
        model_export: false,
        external_only: true,
    },
    FormatInfo {
        kind: FormatKind::Glb,
        name: "usd",
        mime: "application/octet-stream",
        extensions: &["usd", "usda", "usdc", "usdz"],
        // No reliable family magic: `usdz` is a plain zip (the `PK` local
        // file header is already the `ora` row's signature), `usda` and
        // `usdc` are text/custom — admission is by explicit policy only.
        magic: &[],
        budget: FormatBudget {
            decoded_pixels: None,
            entries: Some(65_536),
            uncompressed_bytes: Some(512 * 1024 * 1024),
            entry_bytes: None,
            elements: None,
        },
        sd_image_ingest: false,
        image_ops_input: false,
        output: false,
        output_compression: false,
        model_export: false,
        external_only: true,
    },
    FormatInfo {
        kind: FormatKind::Glb,
        name: "obj",
        mime: "application/octet-stream",
        extensions: &["obj"],
        // Plain text, no reliable magic — admission is by explicit
        // policy only.
        magic: &[],
        budget: FormatBudget {
            decoded_pixels: None,
            entries: Some(65_536),
            uncompressed_bytes: Some(512 * 1024 * 1024),
            entry_bytes: None,
            elements: None,
        },
        sd_image_ingest: false,
        image_ops_input: false,
        output: false,
        output_compression: false,
        model_export: false,
        external_only: true,
    },
    FormatInfo {
        kind: FormatKind::Blob,
        name: "blob",
        mime: "application/octet-stream",
        // No extension contract, no magic: passthrough bytes (vision chat
        // payloads, future opaque uploads) bounded only by the router body
        // limit (`formats.limits.body_bytes`).
        extensions: &[],
        magic: &[],
        budget: FormatBudget {
            decoded_pixels: None,
            entries: None,
            uncompressed_bytes: None,
            entry_bytes: None,
            elements: None,
        },
        sd_image_ingest: false,
        image_ops_input: false,
        output: false,
        output_compression: false,
        model_export: false,
        external_only: false,
    },
];

/// `true` when `sig` matches `bytes` at its recorded offsets.
fn matches_sig(sig: &Signature, bytes: &[u8]) -> bool {
    sig.iter().all(|(offset, part)| {
        bytes.len() >= offset + part.len() && &bytes[*offset..offset + part.len()] == *part
    })
}

/// Identify bytes by magic signature — first matching registry row, in
/// table order. Extensions and declared MIME types are never consulted:
/// two formats sharing an extension are distinguished by their bytes alone.
pub(crate) fn sniff(bytes: &[u8]) -> Option<&'static FormatInfo> {
    FORMATS
        .iter()
        .find(|row| row.magic.iter().any(|sig| matches_sig(sig, bytes)))
}

/// Structured ingest verdict for bytes that must be a specific container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FormatReject {
    /// No registry signature matched — unrecognized container.
    Unknown,
    /// The bytes ARE identified, but as a different format than required.
    Mismatch {
        /// Canonical name of the sniffed format.
        actual: &'static str,
        /// Canonical name the caller required.
        required: &'static str,
    },
}

impl FormatReject {
    /// Parenthetical clause for rejection messages — names the sniffed
    /// format (or the lack of identification), never an extension.
    pub(crate) fn detail(&self) -> String {
        match self {
            FormatReject::Unknown => "unrecognized container".to_owned(),
            FormatReject::Mismatch { actual, .. } => format!("the bytes are {actual}"),
        }
    }
}

/// Identify bytes that must be PNG (the SD image ingest contract: edits
/// `image`/`mask`, `ControlNet` `control_image`). `Err` carries the
/// structured verdict for the named rejection.
pub(crate) fn identify_png(bytes: &[u8]) -> Result<&'static FormatInfo, FormatReject> {
    match sniff(bytes) {
        Some(row) if row.name == "png" => Ok(row),
        Some(row) => Err(FormatReject::Mismatch {
            actual: row.name,
            required: "png",
        }),
        None => Err(FormatReject::Unknown),
    }
}

/// Registry rows that this build can write (the `output` column), in table
/// order — the authoritative source for `formats.write` and the
/// `output_format` accepted-values list.
pub(crate) fn output_rows() -> impl Iterator<Item = &'static FormatInfo> {
    FORMATS.iter().filter(|row| row.output)
}

/// Registry rows the SD image ingest validators accept — exactly what
/// `Capabilities::formats.read` advertises.
pub(crate) fn sd_ingest_names() -> impl Iterator<Item = &'static str> {
    FORMATS
        .iter()
        .filter(|row| row.sd_image_ingest)
        .map(|row| row.name)
}

/// Registry rows accepted as `/v1/image-ops` inputs.
pub(crate) fn image_ops_inputs() -> impl Iterator<Item = &'static FormatInfo> {
    FORMATS.iter().filter(|row| row.image_ops_input)
}

/// The image-ops input row declaring `mime` (unique among the accepted
/// rows). `None` = the declaration is not an accepted container.
pub(crate) fn image_ops_by_mime(mime: &str) -> Option<&'static FormatInfo> {
    image_ops_inputs().find(|row| row.mime == mime)
}

/// Every file extension an image-ops input may use (bundle artifact
/// candidates), derived from the accepted rows' extension lists.
pub(crate) fn image_ops_extensions() -> impl Iterator<Item = &'static str> {
    image_ops_inputs().flat_map(|row| row.extensions.iter().copied())
}

/// The export allowlist for 3D-model artifacts: exactly the registry rows
/// with `model_export`, in table order, pinned as a constant so the
/// structural model gate ([`crate::model_formats`]) and the engine-bundle
/// validator (TASK-model-bundle-validation) share ONE list — no duplicate
/// rules. Locked to the rows by the `model_allowlist_matches_the_registry`
/// tests here and in the `model_formats` suite.
pub(crate) const MODEL_EXPORT_FORMATS: [&str; 4] = ["glb", "ply", "splat", "spz"];

/// Registry rows in the model export allowlist, in table order.
#[cfg(test)]
pub(crate) fn exportable_model_rows() -> impl Iterator<Item = &'static FormatInfo> {
    FORMATS.iter().filter(|row| row.model_export)
}

/// The allowlist names (see [`MODEL_EXPORT_FORMATS`]), derived from the
/// rows — never a second hand-maintained list.
#[cfg(test)]
pub(crate) fn exportable_model_names() -> impl Iterator<Item = &'static str> {
    exportable_model_rows().map(|row| row.name)
}

/// The model-family row claiming `extension` (exportable or
/// `external_only`). `None` = no model row claims the extension, which is
/// itself the admission answer for non-model containers.
pub(crate) fn model_row_for_extension(extension: &str) -> Option<&'static FormatInfo> {
    FORMATS
        .iter()
        .find(|row| {
            (row.model_export || row.external_only)
                && row.extensions.contains(&extension)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1×1 opaque image, encoded per container by the compiled `image`
    /// codecs — the same encoders sd-core's encode path wraps, so the
    /// registry's magic table must identify actual encoder output. (This
    /// module deliberately calls NO sd-core functions: test binaries
    /// `#[path]`-include it next to the swap/llm modules, and linking
    /// sd-core's native-dependent objects beside llama.cpp's vendored ggml
    /// collides on duplicate symbols. The enum type/const below are
    /// metadata-only and link nothing; sd-core's own suite exercises its
    /// encode path.)
    fn encoded(kind: image::ImageFormat) -> Vec<u8> {
        let img = image::DynamicImage::new(1, 1, image::ColorType::Rgba8);
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, kind).expect("1x1 encode");
        out.into_inner()
    }

    /// The compiled encoders' bytes sniff back to their own registry rows:
    /// the magic table agrees with what the build actually produces. The
    /// encoders here are the compiled `image` codecs — the exact ones the
    /// sd-core encode path wraps. (This module deliberately calls NO
    /// sd-core functions: test binaries `#[path]`-include it next to the
    /// swap/llm modules, and a native-symbol pull from sd-core beside
    /// llama.cpp's vendored ggml collides on duplicate symbols. The
    /// name/mime agreement with sd-core is enforced by the capabilities
    /// contract test against the live route.)
    #[test]
    fn compiled_encoders_sniff_to_their_rows() {
        for (name, kind) in [
            ("png", image::ImageFormat::Png),
            ("jpeg", image::ImageFormat::Jpeg),
            ("webp", image::ImageFormat::WebP),
        ] {
            let bytes = encoded(kind);
            let row = sniff(&bytes).expect("encoded output must identify");
            assert_eq!(row.name, name, "sniffed name for {name}");
            assert!(row.output, "output column for {name}");
        }
    }

    /// ACCEPTANCE: two formats sharing an extension (`ppm` and `pgm` both
    /// carry the umbrella `pnm`) are distinguished by magic bytes alone —
    /// identification never consults extensions.
    #[test]
    fn same_extension_different_magic_distinguished() {
        let ppm = sniff(b"P6\n8 8\n255\nrest").expect("P6 is ppm");
        let pgm = sniff(b"P5\n8 8\n255\nrest").expect("P5 is pgm");
        assert_ne!(ppm.name, pgm.name);
        assert_eq!(ppm.kind, FormatKind::VideoFrame);
        assert!(ppm.extensions.contains(&"pnm") && pgm.extensions.contains(&"pnm"));
        // A `.pnm`-named JPEG is a JPEG, not "whatever the extension says":
        // the sniff of JPEG bytes lands on the jpeg row regardless of any
        // extension the bytes might travel under.
        let jpeg = encoded(image::ImageFormat::Jpeg);
        assert_eq!(sniff(&jpeg).expect("jpeg identifies").name, "jpeg");
    }

    /// ACCEPTANCE: unknown formats identify as nothing and every ingest
    /// path rejects them (`Unknown` verdict) — before any admission work.
    #[test]
    fn unknown_bytes_are_rejected_before_admission() {
        for bytes in [
            &b"definitely not a png"[..],
            b"",
            b"GIF89a\xff\xff\xff",
            b"\x00\x00\x01\x00\x01\x00", // ico-flavored header, no row
        ] {
            assert!(sniff(bytes).is_none(), "{bytes:?} must not identify");
            assert_eq!(
                identify_png(bytes),
                Err(FormatReject::Unknown),
                "{bytes:?} must reject as Unknown"
            );
        }
    }

    /// Mismatches name the ACTUAL format: bytes that identify as something
    /// other than the required container get `Mismatch` with the sniffed
    /// canonical name (the same `.png` filename can carry jpeg bytes).
    #[test]
    fn mismatch_names_the_sniffed_format() {
        let jpeg = encoded(image::ImageFormat::Jpeg);
        assert_eq!(
            identify_png(&jpeg),
            Err(FormatReject::Mismatch {
                actual: "jpeg",
                required: "png"
            })
        );
        let err = identify_png(&jpeg).expect_err("jpeg is not png").detail();
        assert_eq!(err, "the bytes are jpeg");
        assert_eq!(FormatReject::Unknown.detail(), "unrecognized container");
    }

    /// The write column is exactly the compiled encoder set, in table
    /// order (the enum-level lock against `OutputFormat::ALL` lives in the
    /// routes module, next to the mapping it guards; this test pins the
    /// registry data itself).
    #[test]
    fn write_column_is_the_compiled_encoder_set() {
        let advertised: Vec<&str> = output_rows().map(|row| row.name).collect();
        assert_eq!(advertised, ["png", "jpeg", "webp"]);
        // The compression knob exists on exactly the lossy container.
        for row in output_rows() {
            assert_eq!(
                row.output_compression,
                row.name == "jpeg",
                "{} compression flag",
                row.name
            );
        }
    }

    /// The SD ingest column stays exactly the sniffable PNG contract the
    /// capabilities section advertises; every accepted ingest row has a
    /// real signature (identification must be possible) and the enforced
    /// pixel budget.
    #[test]
    fn sd_ingest_contract_is_png_only() {
        assert_eq!(sd_ingest_names().collect::<Vec<_>>(), ["png"]);
        for row in FORMATS.iter().filter(|row| row.sd_image_ingest) {
            assert!(!row.magic.is_empty(), "{} must be snappable", row.name);
            assert_eq!(
                row.budget.decoded_pixels,
                Some(MAX_DECODED_PIXELS),
                "{} budget",
                row.name
            );
        }
    }

    /// Image-ops input contract: the accepted rows declare unique MIME
    /// types (data-URI declarations resolve unambiguously), cover every
    /// extension the artifact lookup needs, and all carry the enforced
    /// pixel budget.
    #[test]
    fn image_ops_contract_is_the_accepted_rows() {
        let rows: Vec<&FormatInfo> = image_ops_inputs().collect();
        assert_eq!(
            rows.iter().map(|row| row.name).collect::<Vec<_>>(),
            ["png", "jpeg", "webp", "svg", "ora"]
        );
        let mut mimes: Vec<&str> = rows.iter().map(|row| row.mime).collect();
        mimes.sort_unstable();
        mimes.dedup();
        assert_eq!(mimes.len(), rows.len(), "accepted MIMEs must be unique");
        for row in &rows {
            assert_eq!(
                row.budget.decoded_pixels,
                Some(MAX_DECODED_PIXELS),
                "{} budget",
                row.name
            );
            assert!(!row.magic.is_empty(), "{} must be snappable", row.name);
            assert!(
                image_ops_by_mime(row.mime) == Some(*row),
                "{} by mime",
                row.name
            );
        }
        // `application/octet-stream` (splat/ply/blob rows) is NOT an
        // accepted data-URI declaration.
        assert_eq!(image_ops_by_mime("application/octet-stream"), None);
        let exts: Vec<&str> = image_ops_extensions().collect();
        assert_eq!(exts, ["png", "jpeg", "jpg", "webp", "svg", "ora"]);
    }

    /// Signature matching honors offsets and lengths: partial signatures
    /// (a truncated WebP header) never identify.
    #[test]
    fn partial_signatures_do_not_identify() {
        assert!(sniff(b"RIFF").is_none(), "RIFF alone is not webp");
        assert!(
            sniff(b"RIFF\x00\x00\x00\x00WAVE").is_none(),
            "WAVE form type is not webp"
        );
        let webp = encoded(image::ImageFormat::WebP);
        assert_eq!(sniff(&webp).expect("real webp").name, "webp");
    }

    /// The kind labels keep rejection messages meaningful across families.
    #[test]
    fn kind_labels_describe_the_families() {
        assert_eq!(FormatKind::Image.label(), "raster image");
        assert_eq!(FormatKind::VideoFrame.label(), "video frame");
        assert_eq!(FormatKind::Svg.label(), "vector drawing");
        assert_eq!(FormatKind::Ora.label(), "layered raster document");
        assert_eq!(FormatKind::Glb.label(), "3D scene");
        assert_eq!(FormatKind::Splat.label(), "gaussian splat");
        assert_eq!(FormatKind::Blob.label(), "opaque blob");
    }

    /// ACCEPTANCE: the model export allowlist is exactly the `model_export`
    /// rows, the external-only rows carry NO in-process capability at all,
    /// and the two new magic identities sniff while the policy-admitted
    /// containers (usd/obj, legacy gzip-wrapped spz) never do.
    #[test]
    fn model_allowlist_matches_the_registry() {
        assert_eq!(
            exportable_model_names().collect::<Vec<_>>(),
            MODEL_EXPORT_FORMATS
        );
        for row in exportable_model_rows() {
            assert!(!row.external_only, "{} must not be external-only", row.name);
        }
        for row in FORMATS.iter().filter(|row| row.external_only) {
            assert!(!row.model_export, "{} is not exportable", row.name);
            assert!(!row.sd_image_ingest, "{} ingest", row.name);
            assert!(!row.image_ops_input, "{} image-ops", row.name);
            assert!(!row.output, "{} output", row.name);
        }
        let external: Vec<&str> = FORMATS
            .iter()
            .filter(|row| row.external_only)
            .map(|row| row.name)
            .collect();
        assert_eq!(external, ["fbx", "usd", "obj"]);
        // Magic identities for the two new rows that have one.
        assert_eq!(
            sniff(b"NGSP\x04\x00\x00\x00").map(|row| row.name),
            Some("spz")
        );
        assert_eq!(
            sniff(b"Kaydara FBX Binary  \x00\x1a\x00\x00").map(|row| row.name),
            Some("fbx")
        );
        // Policy-admitted containers have no identity: a usda text file and
        // a generic gzip envelope never sniff to anything.
        assert_eq!(sniff(b"#usda 1.0\n"), None);
        assert_eq!(sniff(b"\x1f\x8b\x08not really gzip"), None);
    }
}
