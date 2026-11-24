//! Bounded image-op parameter validation and input decoding helpers.

use std::path::Path;

use base64::Engine as _;
use honkbox_api_types::{ImageOpArtifact, ImageOpCommand};

use super::image_ops_exec::ImageOpWorkError;
use super::image_ops_limits::IMAGE_OP_MAX_EDGE;

/// Validate the bounded image-op params (reject-not-clamp on every number):
/// resize edges in `[1, 8192]` with the pixel cap, crop rects at least 1px
/// (the image-bounds check happens per-input against the real pixels), and
/// rotation restricted to the lossless quarter turns. `Err((message, code))`
/// is a named 400.
pub(super) fn validate_image_op_params(
    command: &ImageOpCommand,
) -> Result<(), (String, &'static str)> {
    match command {
        ImageOpCommand::Resize(resize) => {
            if resize.width == 0
                || resize.width > IMAGE_OP_MAX_EDGE
                || resize.height == 0
                || resize.height > IMAGE_OP_MAX_EDGE
            {
                return Err((
                    format!(
                        "resize width/height must be in [1, {IMAGE_OP_MAX_EDGE}] pixels; got {}x{} — rejected, not clamped",
                        resize.width, resize.height
                    ),
                    "invalid_image_op_params",
                ));
            }
            if u64::from(resize.width) * u64::from(resize.height)
                > crate::formats::MAX_DECODED_PIXELS
            {
                return Err((
                    format!(
                        "resize target {}x{} exceeds the {}-pixel cap — rejected, not clamped",
                        resize.width,
                        resize.height,
                        crate::formats::MAX_DECODED_PIXELS
                    ),
                    "invalid_image_op_params",
                ));
            }
        }
        ImageOpCommand::Crop(crop) => {
            if crop.width == 0 || crop.height == 0 {
                return Err((
                    "crop width/height must be at least 1 pixel — rejected, not clamped".to_owned(),
                    "invalid_image_op_params",
                ));
            }
        }
        ImageOpCommand::Rotate(rotate) => {
            if !matches!(rotate.degrees, 90 | 180 | 270) {
                return Err((
                    format!(
                        "rotate degrees must be one of 90, 180, 270; got {} — rejected, not approximated",
                        rotate.degrees
                    ),
                    "invalid_image_op_params",
                ));
            }
        }
        ImageOpCommand::Grayscale | ImageOpCommand::Format | ImageOpCommand::Matting => {}
    }
    Ok(())
}

/// Decode a `data:image/…;base64,…` URI. The declared MIME must be one of
/// the registry's accepted input containers (the caller additionally sniffs
/// the real bytes, so declared/actual mismatches are caught there); the
/// declared row is returned alongside the bytes for that comparison.
pub(super) fn decode_data_uri(
    uri: &str,
) -> Result<(&'static crate::formats::FormatInfo, Vec<u8>), ImageOpWorkError> {
    let bad = |message: String| ImageOpWorkError::Bad {
        message,
        code: "invalid_image_op_inputs",
    };
    let trimmed = uri.trim();
    let Some(rest) = trimmed.strip_prefix("data:") else {
        return Err(bad(
            "input data URI must start with `data:image/…;base64,`".to_owned()
        ));
    };
    let Some((header, payload)) = rest.split_once(',') else {
        return Err(bad(
            "input data URI is missing the `,<base64 payload>` part".to_owned(),
        ));
    };
    let Some(mime) = header.strip_suffix(";base64") else {
        return Err(bad("input data URI must declare `;base64`".to_owned()));
    };
    let Some(declared) = crate::formats::image_ops_by_mime(mime) else {
        return Err(bad(format!(
            "unsupported input container {mime:?}: accepted data URIs are {}",
            crate::formats::image_ops_inputs()
                .map(|row| row.mime)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        // Tolerate padding-stripped payloads (some encoders trim `=`).
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(payload.trim()))
        .map_err(|e| bad(format!("input data URI is not valid base64: {e}")))?;
    Ok((declared, bytes))
}

/// Resolve an artifact input against the bundle store:
/// `<data_dir>/jobs/{bundle}/op_XX.{ext}` (ops bundles) or
/// `final_XX.{ext}` (generation bundles). The bundle id must parse as a
/// ULID, so no client-supplied bytes ever reach the filesystem join.
pub(super) fn read_artifact(
    data_dir: &Path,
    artifact: &ImageOpArtifact,
) -> Result<Vec<u8>, ImageOpWorkError> {
    let bad = |message: String| ImageOpWorkError::Bad {
        message,
        code: "invalid_image_op_inputs",
    };
    let id = ulid::Ulid::from_string(artifact.bundle.trim()).map_err(|_| {
        bad(format!(
            "artifact bundle {:?} is not a bundle id",
            artifact.bundle
        ))
    })?;
    if artifact.index > 99 {
        return Err(bad(format!(
            "artifact index {} is out of range (0-99)",
            artifact.index
        )));
    }
    let dir = data_dir.join("jobs").join(id.to_string());
    // Candidate artifact names derive from the registry's accepted input
    // extensions (the same rows the magic sniff admits below).
    let names: Vec<String> = crate::formats::image_ops_extensions()
        .flat_map(|ext| {
            [
                format!("op_{:02}.{ext}", artifact.index),
                format!("final_{:02}.{ext}", artifact.index),
            ]
        })
        .collect();
    let entries = std::fs::read_dir(&dir).map_err(|_| {
        bad(format!(
            "artifact bundle {:?} not found (expired by GC, or never written by this server)",
            artifact.bundle
        ))
    })?;
    let mut found: Vec<std::ffi::OsString> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if names.iter().any(|wanted| *wanted == name_str) {
            found.push(name);
        }
    }
    found.sort();
    match found.first() {
        Some(name) => std::fs::read(dir.join(name)).map_err(|e| {
            ImageOpWorkError::Failed(format!(
                "cannot read artifact {:?}: {e}",
                name.to_string_lossy()
            ))
        }),
        None => Err(bad(format!(
            "artifact {} of bundle {:?} not found",
            artifact.index, artifact.bundle
        ))),
    }
}
