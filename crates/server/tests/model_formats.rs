//! The structural model-format gate corpus (TASK-model-format-validation).
//!
//! Every fixture is hand-built bytes in a `Vec` — no fixture files, no
//! temp dirs, no shared state — and every rejection asserts the NAMED
//! code (`unsupported_model_extension`, `model_external_only`,
//! `model_header_invalid`, `model_truncated`, `model_too_large`), not
//! just "something went wrong": a gate that reports a truncated GLB the
//! same way it reports an FBX claim has a drifted contract.
//!
//! The GLB fixtures reuse the container lane's builder shape (12-byte
//! header + JSON chunk + optional BIN chunk); the PLY/SPZ fixtures are
//! built from the real format specs (Stanford PLY header grammar;
//! nianticlabs/spz NGSP v4 header layout).

// No lib target on this binary crate, so include the modules directly.
// (model_formats carries its own module-level `#![allow(dead_code)]` for
// the unwired-binary state — no outer attr here, or clippy sees the
// duplication.)
#[allow(dead_code)]
#[path = "../src/container.rs"]
mod container;
#[allow(dead_code)]
#[path = "../src/formats.rs"]
mod formats;
#[path = "../src/model_formats.rs"]
mod model_formats;

use sha2::{Digest, Sha256};

use crate::formats::{exportable_model_names, model_row_for_extension, sniff};
use crate::model_formats::{
    validate_model, ModelReject, SIDECAR_SUFFIX, VALIDATOR_VERSION, ValidatedModel,
};

/// Lowercase-hex SHA-256 — the digest form the sidecar must record.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Build a valid minimal GLB: 12-byte header (magic, version 2, total
/// length) + JSON chunk, plus an optional BIN chunk. Chunk payloads are
/// padded to the four-byte boundary, exactly as the spec requires.
fn glb(json: &[u8], bin: Option<&[u8]>) -> Vec<u8> {
    let pad = |len: usize| (4 - (len % 4)) % 4;
    let json_len = json.len() + pad(json.len());
    let bin_len = bin.map_or(0, |b| b.len() + pad(b.len()));
    let total = 12 + 8 + json_len + usize::from(bin.is_some()) * (8 + bin_len);
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&u32::try_from(total).expect("fixture fits u32").to_le_bytes());
    out.extend_from_slice(&u32::try_from(json_len).expect("fixture fits u32").to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(json);
    out.extend(vec![0u8; pad(json.len())]);
    if let Some(bin) = bin {
        out.extend_from_slice(&u32::try_from(bin_len).expect("fixture fits u32").to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(bin);
        out.extend(vec![0u8; pad(bin.len())]);
    }
    out
}

const GLB_JSON: &[u8] = br#"{"asset":{"version":"2.0"}}"#;

/// A binary-PLY header (format line + element/property lines) plus raw
/// payload bytes.
fn ply(header_elements: &str, payload: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "ply\nformat binary_little_endian 1.0\n{header_elements}end_header\n"
    )
    .into_bytes();
    out.extend_from_slice(payload);
    out
}

/// Three float properties per vertex: a 12-byte fixed row.
const THREE_FLOATS: &str = "property float x\nproperty float y\nproperty float z\n";

/// An SPZ v4 plaintext NGSP header (32 bytes).
fn spz_ngsp(num_points: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    out.extend_from_slice(b"NGSP");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&num_points.to_le_bytes());
    out.extend_from_slice(&[0u8; 4]); // shDegree, fractionalBits, flags, numStreams
    out.extend_from_slice(&0u32.to_le_bytes()); // tocByteOffset
    out.extend_from_slice(&[0u8; 12]); // reserved
    out
}

/// A legacy gzip envelope (v1–3 SPZ travel this way).
fn spz_gzip(total_len: usize) -> Vec<u8> {
    let mut out = vec![0x1f, 0x8b, 0x08];
    out.resize(total_len, 0u8);
    out
}

#[test]
fn minimal_glb_passes() {
    let bytes = glb(GLB_JSON, None);
    let validated = validate_model(&bytes, "glb").expect("the minimal GLB is valid");
    assert_eq!(validated.format, "glb");
    assert_eq!(validated.format_version, Some(2));
    assert_eq!(validated.sha256, sha256_hex(&bytes));
    assert_eq!(validated.size, u64::try_from(bytes.len()).expect("usize fits"));

    // The same bytes pass under an uppercase or dotted extension claim.
    let validated = validate_model(&bytes, ".GLB").expect("extension normalization");
    assert_eq!(validated.format, "glb");

    // A GLB with a binary chunk passes too.
    let bytes = glb(GLB_JSON, Some(b"\x00\x00\x80?"));
    validate_model(&bytes, "glb").expect("the GLB with a BIN chunk is valid");
}

#[test]
fn truncated_glb_fails_model_truncated() {
    // Declared length beyond the bytes present.
    let mut bytes = glb(GLB_JSON, None);
    bytes.truncate(bytes.len() - 4);
    let err = validate_model(&bytes, "glb").expect_err("the GLB is truncated");
    assert_eq!(err.code(), "model_truncated");
    assert!(err.detail().contains("declares"), "got {:?}", err.detail());

    // Header itself cut short.
    let err = validate_model(&b"glTF\x02\x00"[..], "glb")
        .expect_err("the header is cut short");
    assert_eq!(err.code(), "model_truncated");

    // Empty bytes with a glb claim: no header at all is truncation.
    let err = validate_model(&[], "glb").expect_err("no bytes");
    assert_eq!(err.code(), "model_truncated");
}

#[test]
fn glb_bad_magic_fails() {
    let mut bytes = glb(GLB_JSON, None);
    bytes[0] = b'X';
    let err = validate_model(&bytes, "glb").expect_err("the magic is corrupt");
    assert_eq!(err.code(), "model_header_invalid");
    assert!(err.detail().contains("bad magic"), "got {:?}", err.detail());

    // Bytes that identify as a DIFFERENT registered format reject with the
    // sniffed name, never the claimed extension.
    let png_claimed_glb = b"\x89PNG\r\n\x1a\nnot really a png";
    let err = validate_model(png_claimed_glb, "glb").expect_err("the bytes are png");
    assert_eq!(err.code(), "model_header_invalid");
    assert_eq!(
        err.detail(),
        "the model container is invalid: the bytes are png"
    );
}

#[test]
fn glb_legacy_or_absurd_headers_fail() {
    let mut bytes = glb(GLB_JSON, None);
    bytes[4] = 1; // glTF 1.0 legacy version
    let err = validate_model(&bytes, "glb").expect_err("glTF 1.0 is not the export version");
    assert_eq!(err.code(), "model_header_invalid");
    assert!(err.detail().contains("version 1"), "got {:?}", err.detail());

    let mut bytes = glb(GLB_JSON, None);
    bytes[8..12].copy_from_slice(&3u32.to_le_bytes()); // absurd declared length
    let err = validate_model(&bytes, "glb").expect_err("no room for a chunk table");
    assert_eq!(err.code(), "model_header_invalid");

    // A chunk whose own length lies past the container end is an
    // inconsistency (all bytes are present): header_invalid, not truncated.
    let mut bytes = glb(GLB_JSON, None);
    bytes[12..16].copy_from_slice(&4096u32.to_le_bytes()); // JSON chunk lies
    let err = validate_model(&bytes, "glb").expect_err("the chunk length lies");
    assert_eq!(err.code(), "model_header_invalid");
}

#[test]
fn ply_ascii_passes() {
    let mut bytes = b"ply\nformat ascii 1.0\nelement vertex 3\n".to_vec();
    bytes.extend_from_slice(THREE_FLOATS.as_bytes());
    bytes.extend_from_slice(b"end_header\n0 0 0\n1 0 0\n0 1 0\n");
    let validated = validate_model(&bytes, "ply").expect("ascii PLY is valid");
    assert_eq!(validated.format, "ply");
    assert_eq!(validated.format_version, Some(1));
    assert_eq!(validated.sha256, sha256_hex(&bytes));
}

#[test]
fn ply_binary_declared_sizes_are_enforced() {
    let payload = [0u8; 36]; // 3 vertices × 12 bytes
    let bytes = ply(&format!("element vertex 3\n{THREE_FLOATS}"), &payload);
    let validated = validate_model(&bytes, "ply").expect("exact binary PLY is valid");
    assert_eq!(validated.format_version, Some(1));

    // Three bytes short: truncated.
    let bytes = ply(&format!("element vertex 3\n{THREE_FLOATS}"), &[0u8; 33]);
    let err = validate_model(&bytes, "ply").expect_err("payload is short");
    assert_eq!(err.code(), "model_truncated");
    assert!(
        err.detail().contains("36 payload bytes"),
        "got {:?}",
        err.detail()
    );

    // Trailing garbage beyond the declared payload: invalid, not accepted.
    let bytes = ply(&format!("element vertex 3\n{THREE_FLOATS}"), &[0u8; 40]);
    let err = validate_model(&bytes, "ply").expect_err("payload has extra bytes");
    assert_eq!(err.code(), "model_header_invalid");

    // List properties make rows variable-width: only the MINIMUM (the list
    // count field) is derivable, and it is still enforced.
    let elements = format!(
        "element vertex 3\n{THREE_FLOATS}element face 1\nproperty list uchar int vertex_indices\n"
    );
    let min_payload = [0u8; 37]; // 36 vertex bytes + 1 count byte
    let bytes = ply(&elements, &min_payload);
    validate_model(&bytes, "ply").expect("the list PLY meets its minimum");
    let bytes = ply(&elements, &[0u8; 36]);
    let err = validate_model(&bytes, "ply").expect_err("below the list minimum");
    assert_eq!(err.code(), "model_truncated");
}

#[test]
fn ply_header_grammar_is_strict() {
    // Big-endian is a named rejection.
    let mut bytes = b"ply\nformat binary_big_endian 1.0\nelement vertex 1\n".to_vec();
    bytes.extend_from_slice(THREE_FLOATS.as_bytes());
    bytes.extend_from_slice(b"end_header\n");
    let err = validate_model(&bytes, "ply").expect_err("big-endian is rejected");
    assert_eq!(err.code(), "model_header_invalid");
    assert!(
        err.detail().contains("binary_big_endian"),
        "got {:?}",
        err.detail()
    );

    // Unknown directive names itself.
    let bytes = b"ply\nformat ascii 1.0\nsurprise_element 5\nend_header\n".to_vec();
    let err = validate_model(&bytes, "ply").expect_err("unknown directive");
    assert_eq!(err.code(), "model_header_invalid");
    assert!(
        err.detail().contains("surprise_element"),
        "got {:?}",
        err.detail()
    );

    // Tab-separated tokens are legal header whitespace: they parse.
    let bytes = b"ply\nformat ascii 1.0\nelement\tvertex\t3\n".to_vec();
    let mut bytes = bytes;
    bytes.extend_from_slice(b"property\tfloat\tx\nend_header\n0 0\n1 0\n0 1\n");
    let validated = validate_model(&bytes, "ply").expect("tab-separated header parses");
    assert_eq!(validated.format, "ply");

    // Property before any element.
    let mut bytes = b"ply\nformat ascii 1.0\n".to_vec();
    bytes.extend_from_slice(THREE_FLOATS.as_bytes());
    bytes.extend_from_slice(b"end_header\n");
    let err = validate_model(&bytes, "ply").expect_err("property precedes element");
    assert_eq!(err.code(), "model_header_invalid");

    // A header that never completes is a truncated file.
    let bytes = b"ply\nformat ascii 1.0\nelement vertex 3\n".to_vec();
    let err = validate_model(&bytes, "ply").expect_err("no end_header");
    assert_eq!(err.code(), "model_truncated");

    // Declared counts beyond the registry budget: named too_large.
    let mut bytes = b"ply\nformat ascii 1.0\nelement vertex 99999999\n".to_vec();
    bytes.extend_from_slice(THREE_FLOATS.as_bytes());
    bytes.extend_from_slice(b"end_header\n");
    let err = validate_model(&bytes, "ply").expect_err("declared count is over budget");
    assert_eq!(err.code(), "model_too_large");
}

#[test]
fn fbx_usd_obj_are_external_only() {
    for (extension, format) in [
        ("fbx", "fbx"),
        ("usd", "usd"),
        ("usda", "usd"),
        ("usdc", "usd"),
        ("usdz", "usd"),
        ("obj", "obj"),
    ] {
        let err = validate_model(b"anything at all", extension)
            .expect_err("external-only formats never pass");
        match &err {
            ModelReject::ExternalOnly { format: named } => assert_eq!(*named, format),
            other => panic!("expected ExternalOnly for {extension}, got {other:?}"),
        }
        assert_eq!(err.code(), "model_external_only");
        assert!(err.detail().contains("external-only"), "got {:?}", err.detail());
    }

    // Even bytes carrying the REAL Kaydara FBX binary magic are refused on
    // policy: the claim names an external-only format, so the bytes never
    // reach structural validation.
    let kaydara = b"Kaydara FBX Binary  \x00\x1a\x00\x00\x00";
    assert_eq!(sniff(kaydara).map(|row| row.name), Some("fbx"));
    let err = validate_model(kaydara, "fbx").expect_err("magic does not bypass policy");
    assert!(matches!(err, ModelReject::ExternalOnly { format: "fbx" }));
}

#[test]
fn unknown_and_non_model_claims_are_named() {
    let err = validate_model(b"whatever", "xyz").expect_err("unknown extension");
    assert!(matches!(err, ModelReject::UnsupportedExtension));
    assert_eq!(err.code(), "unsupported_model_extension");
    assert!(
        err.detail().contains("glb, ply, splat, spz"),
        "got {:?}",
        err.detail()
    );

    // A known NON-model container is unsupported here, not misvalidated.
    let err = validate_model(b"whatever", "png").expect_err("png is not a model format");
    assert_eq!(err.code(), "unsupported_model_extension");
}

#[test]
fn splat_stride_contract_is_enforced() {
    // Two records: 64 bytes, no header, no version.
    let bytes = vec![0u8; 64];
    let validated = validate_model(&bytes, "splat").expect("aligned splat records pass");
    assert_eq!(validated.format, "splat");
    assert_eq!(validated.format_version, None);

    // One byte off the 32-byte stride is a named rejection.
    let bytes = vec![0u8; 63];
    let err = validate_model(&bytes, "splat").expect_err("not stride-aligned");
    assert_eq!(err.code(), "model_header_invalid");
    assert!(err.detail().contains("32-byte"), "got {:?}", err.detail());

    // An empty container holds no records.
    let err = validate_model(&[], "splat").expect_err("empty splat");
    assert_eq!(err.code(), "model_header_invalid");

    // Bytes that identify as another format reject with the sniffed name.
    let err = validate_model(b"ply\nformat ascii 1.0\nend_header\n", "splat")
        .expect_err("the bytes are ply");
    assert!(
        err.detail().contains("the bytes are ply"),
        "got {:?}",
        err.detail()
    );
}

#[test]
fn spz_ngsp_headers_are_validated() {
    let bytes = spz_ngsp(2);
    let validated = validate_model(&bytes, "spz").expect("a sane NGSP v4 header passes");
    assert_eq!(validated.format, "spz");
    assert_eq!(validated.format_version, Some(4));

    // Legacy version on the plaintext path: named.
    let mut bytes = spz_ngsp(2);
    bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
    let err = validate_model(&bytes, "spz").expect_err("v3 is not the plaintext version");
    assert_eq!(err.code(), "model_header_invalid");
    assert!(err.detail().contains("version 3"), "got {:?}", err.detail());

    // Header cut short.
    let bytes = spz_ngsp(2)[..20].to_vec();
    let err = validate_model(&bytes, "spz").expect_err("header truncated");
    assert_eq!(err.code(), "model_truncated");

    // Zero points and over-bound claims are named.
    let err = validate_model(&spz_ngsp(0), "spz").expect_err("zero points");
    assert_eq!(err.code(), "model_header_invalid");
    let mut bytes = spz_ngsp(2);
    bytes[8..12].copy_from_slice(&4_000_000_000u32.to_le_bytes());
    let err = validate_model(&bytes, "spz").expect_err("above the int32 bound");
    assert_eq!(err.code(), "model_header_invalid");
    let mut bytes = spz_ngsp(2);
    bytes[8..12].copy_from_slice(&1_000_000u32.to_le_bytes());
    let err = validate_model(&bytes, "spz").expect_err("above the file-size bound");
    assert_eq!(err.code(), "model_header_invalid");
    assert!(
        err.detail().contains("admit at most"),
        "got {:?}",
        err.detail()
    );

    // Over the REGISTRY point budget (big enough to pass the ratio bound):
    // named too_large. 16_777_217 points need ~147 KB to pass the
    // compression-ratio bound, so the fixture pads to 150 KB.
    let mut bytes = spz_ngsp(16_777_217);
    bytes.resize(150_000, 0);
    let err = validate_model(&bytes, "spz").expect_err("above the point budget");
    assert_eq!(err.code(), "model_too_large");
    assert_eq!(err.detail(), "the model declares or holds 16777217, exceeding the limit of 16777216");
}

#[test]
fn spz_legacy_gzip_envelope_is_minimal_and_honest() {
    // Legacy v1–3: validation stops at the gzip envelope; no format
    // version is claimed because the header is inside the stream.
    let bytes = spz_gzip(64);
    let validated = validate_model(&bytes, "spz").expect("a gzip envelope passes minimally");
    assert_eq!(validated.format, "spz");
    assert_eq!(validated.format_version, None);

    // Envelope cut short.
    let err = validate_model(&spz_gzip(12), "spz").expect_err("envelope truncated");
    assert_eq!(err.code(), "model_truncated");

    // Neither NGSP nor gzip: named invalid.
    let err = validate_model(b"not an spz container at all", "spz")
        .expect_err("no recognizable envelope");
    assert_eq!(err.code(), "model_header_invalid");
    assert!(err.detail().contains("gzip"), "got {:?}", err.detail());
}

#[test]
fn allowlist_matches_the_registry_rows() {
    // The published constant, the derived names, and the registry rows
    // are ONE set (same lock style as the formats registry tests).
    assert_eq!(
        exportable_model_names().collect::<Vec<_>>(),
        ["glb", "ply", "splat", "spz"]
    );
    for name in exportable_model_names() {
        let row = model_row_for_extension(name).expect("allowlist row resolves");
        assert_eq!(row.name, name);
        assert!(row.model_export, "{name} is exportable");
        assert!(!row.external_only, "{name} is in-process");
    }
    // Every external-only extension resolves to its row, and none of them
    // is exportable or carries any other capability.
    for (extension, name) in [
        ("fbx", "fbx"),
        ("usd", "usd"),
        ("usda", "usd"),
        ("usdc", "usd"),
        ("usdz", "usd"),
        ("obj", "obj"),
    ] {
        let row = model_row_for_extension(extension).expect("external row resolves");
        assert_eq!(row.name, name);
        assert!(row.external_only, "{extension} is external-only");
        assert!(!row.model_export, "{extension} is not exportable");
        assert!(!row.sd_image_ingest && !row.image_ops_input && !row.output);
    }
}

#[test]
fn sidecar_records_provenance() {
    let bytes = glb(GLB_JSON, None);
    let validated: ValidatedModel = validate_model(&bytes, "glb").expect("valid");
    let sidecar = crate::model_formats::validation_sidecar(&validated);
    assert_eq!(sidecar["format"], "glb");
    assert_eq!(sidecar["format_version"], 2);
    assert_eq!(sidecar["validator_version"], VALIDATOR_VERSION);
    assert_eq!(sidecar["sha256"], sha256_hex(&bytes));
    assert_eq!(sidecar["size_bytes"], u64::try_from(bytes.len()).expect("usize fits"));
    assert_eq!(SIDECAR_SUFFIX, ".validation.json");

    // A headerless format records a null version, honestly.
    let splat = vec![0u8; 32];
    let validated = validate_model(&splat, "splat").expect("valid");
    let sidecar = crate::model_formats::validation_sidecar(&validated);
    assert!(sidecar["format_version"].is_null());
}
