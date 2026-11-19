//! Image-op execution: admit inputs, run the op units on the blocking pool,
//! and settle every outcome into a replayable response.

use std::path::Path;

use axum::http::StatusCode;
use honkbox_api_types::{ImageOpCommand, ImageOpInput};

use super::image_ops_dedup::{ImageOpSettled, sha256_hex};
use super::image_ops_limits::{
    RASTER_FALLBACK_MAX_PIXELS, image_decode_limits, ora_read_limits, svg_read_limits,
};
use super::image_ops_params::{decode_data_uri, read_artifact};

/// One processed unit: the encoded output bytes and their digest.
pub(super) struct ImageOpUnitOutput {
    pub(super) bytes: Vec<u8>,
    pub(super) sha256: String,
}

/// A failed image-op batch. The whole batch fails fast on the first bad
/// input — partial results would mislead an editor about what landed.
pub(super) enum ImageOpWorkError {
    /// Cancel marked between units (`completed` units are done).
    Cancelled { completed: usize },
    /// Named 400-class rejection.
    Bad { message: String, code: &'static str },
    /// 500-class failure (decode/encode/IO).
    Failed(String),
}

/// Apply the op to one decoded image. Deterministic by construction: the
/// single fixed resample is Lanczos3, crops are pixel-exact, rotations are
/// the lossless quarter turns, grayscale uses the `image` crate's Luma
/// conversion, and `format` passes pixels through untouched.
pub(super) fn apply_image_op(
    command: &ImageOpCommand,
    decoded: image::DynamicImage,
) -> Result<image::DynamicImage, ImageOpWorkError> {
    Ok(match command {
        ImageOpCommand::Resize(resize) => decoded.resize_exact(
            resize.width,
            resize.height,
            image::imageops::FilterType::Lanczos3,
        ),
        ImageOpCommand::Crop(crop) => {
            // Bounds checked against the REAL pixels here: the DTO-level
            // validation knows the rect shape, not the image.
            if crop.x.saturating_add(crop.width) > decoded.width()
                || crop.y.saturating_add(crop.height) > decoded.height()
            {
                return Err(ImageOpWorkError::Bad {
                    message: format!(
                        "crop rect {}+{}x{}+{} exceeds the {}x{} input — rejected, not clamped",
                        crop.x,
                        crop.width,
                        crop.y,
                        crop.height,
                        decoded.width(),
                        decoded.height()
                    ),
                    code: "invalid_crop_rect",
                });
            }
            decoded.crop_imm(crop.x, crop.y, crop.width, crop.height)
        }
        ImageOpCommand::Rotate(rotate) => match rotate.degrees {
            90 => decoded.rotate90(),
            180 => decoded.rotate180(),
            270 => decoded.rotate270(),
            // DTO-level validation rejects every other value first.
            _ => decoded,
        },
        ImageOpCommand::Grayscale => decoded.grayscale(),
        ImageOpCommand::Format => decoded,
    })
}

/// Admit one image-op input: the registry identifies the container by
/// magic for BOTH input forms — the data URI's declared MIME was resolved
/// at parse and must agree with the bytes (a disagreement is the named
/// `format_mismatch`), artifacts declare nothing, and bytes matching a
/// non-accepted row (e.g. a video-frame or 3D format) name that row in the
/// rejection. Returns the admitted row for its decode budget.
pub(super) fn admit_op_input(
    raw: &[u8],
    declared: Option<&'static crate::formats::FormatInfo>,
) -> Result<&'static crate::formats::FormatInfo, ImageOpWorkError> {
    let accepted = crate::formats::image_ops_inputs()
        .map(|row| row.name)
        .collect::<Vec<_>>()
        .join(", ");
    let Some(row) = crate::formats::sniff(raw) else {
        return Err(ImageOpWorkError::Bad {
            message: format!("input is not a recognizable image (accepted containers: {accepted})"),
            code: "invalid_image_op_inputs",
        });
    };
    if let Some(expected) = declared
        && expected.name != row.name
    {
        return Err(ImageOpWorkError::Bad {
            message: format!(
                "input declares {} but the bytes are {} (magic sniff)",
                expected.mime, row.name
            ),
            code: "format_mismatch",
        });
    }
    if !row.image_ops_input {
        return Err(ImageOpWorkError::Bad {
            message: format!(
                "unsupported input container {} ({}): accepted containers are {accepted}",
                row.name,
                row.kind.label()
            ),
            code: "invalid_image_op_inputs",
        });
    }
    Ok(row)
}

/// Run the op units (blocking pool): per input — cancel check, decode,
/// apply, encode. Fails fast on the first bad input. The inputs are
/// resolved and safety-checked BEFORE the job is admitted.
/// One admitted image-op input: the resolved bytes, the registry row that
/// identified them, and the bookkeeping the sidecar records.
pub(super) struct AdmittedOpInput {
    /// Where the input came from (`data_uri` / `artifact:<bundle>/<index>`).
    pub(super) source: String,
    /// The raw container bytes.
    pub(super) raw: Vec<u8>,
    /// The registry row that identified the container; its budget drives
    /// the read limits.
    pub(super) row: &'static crate::formats::FormatInfo,
    /// The SHA-256 of the raw bytes (the sidecar input record).
    pub(super) sha256: String,
}

/// Resolve and safety-check every image-op input BEFORE a job is admitted.
///
/// Resolution (data-URI decode / artifact read) and the registry sniff
/// happen here, and so does the format-specific safety check with the
/// sniffed row's budgets: a hostile vector or layered document is a named
/// 400 and NO job is registered at all.
pub(super) fn admit_op_inputs(
    inputs: &[ImageOpInput],
    data_dir: Option<&Path>,
) -> Result<Vec<AdmittedOpInput>, ImageOpWorkError> {
    let mut admitted = Vec::with_capacity(inputs.len());
    for input in inputs {
        let (source, declared, raw) = match input {
            ImageOpInput::DataUri(uri) => {
                let (declared, bytes) = decode_data_uri(uri)?;
                (String::from("data_uri"), Some(declared), bytes)
            }
            ImageOpInput::Artifact(artifact) => {
                let Some(dir) = data_dir else {
                    return Err(ImageOpWorkError::Bad {
                        message: "artifact inputs need a configured data dir (this server keeps no job bundles)"
                            .to_owned(),
                        code: "invalid_image_op_inputs",
                    });
                };
                (
                    format!("artifact:{}/{}", artifact.bundle, artifact.index),
                    None,
                    read_artifact(dir, artifact)?,
                )
            }
        };
        // Container authority is the registry magic (see `admit_op_input`):
        // the sniffed format wins, declarations must agree with it.
        let row = admit_op_input(&raw, declared)?;
        let sha256 = sha256_hex(&raw);
        check_admitted_container(row, &raw)?;
        admitted.push(AdmittedOpInput {
            source,
            raw,
            row,
            sha256,
        });
    }
    Ok(admitted)
}

/// The format-specific safety check of an admitted input: the SVG
/// validator and the ORA reader run with the sniffed row's budgets, so a
/// hostile container is rejected before any job is admitted.
pub(super) fn check_admitted_container(
    row: &crate::formats::FormatInfo,
    raw: &[u8],
) -> Result<(), ImageOpWorkError> {
    let bad = |message: String| ImageOpWorkError::Bad {
        message,
        code: "invalid_image_op_inputs",
    };
    match row.kind {
        crate::formats::FormatKind::Svg => crate::svg::validate_svg(raw, &svg_read_limits(row))
            .map(|_| ())
            .map_err(|e| bad(format!("cannot read svg container: {e}")))?,
        crate::formats::FormatKind::Ora => crate::ora::read_ora(raw, &ora_read_limits(row))
            .map(|_| ())
            .map_err(|e| bad(format!("cannot read ora container: {e}")))?,
        _ => {}
    }
    Ok(())
}

/// Decode one admitted input into pixels: the SVG rasterizer and the ORA
/// flattener for the vector and layered rows, the `image` crate reader for
/// the flat raster rows. The limits come from the sniffed registry row.
pub(super) fn decode_admitted_input(
    admitted: &AdmittedOpInput,
) -> Result<image::DynamicImage, ImageOpWorkError> {
    let bad = |message: String| ImageOpWorkError::Bad {
        message,
        code: "invalid_image_op_inputs",
    };
    match admitted.row.kind {
        crate::formats::FormatKind::Svg => {
            crate::svg::rasterize_svg(&admitted.raw, &svg_read_limits(admitted.row))
                .map(crate::raster::RasterImage::into_dynamic_image)
                .map_err(|e| bad(format!("cannot rasterize svg: {e}")))
        }
        crate::formats::FormatKind::Ora => {
            crate::ora::flatten_ora(&admitted.raw, &ora_read_limits(admitted.row))
                .map(crate::raster::RasterImage::into_dynamic_image)
                .map_err(|e| bad(format!("cannot flatten ora: {e}")))
        }
        _ => {
            // Bounded decode: the reader-level strict caps stop
            // decompression-bomb allocations before they happen; the pixel
            // cap (the sniffed row's registry budget) is the honest reject
            // line for oversized-but-bounded inputs.
            let mut reader = image::ImageReader::new(std::io::Cursor::new(&admitted.raw));
            reader.limits(image_decode_limits());
            reader
                .with_guessed_format()
                .map_err(|e| bad(format!("cannot read image container: {e}")))?
                .decode()
                .map_err(|e| bad(format!("cannot decode image: {e}")))
        }
    }
}

pub(super) fn run_image_op_units(
    command: &ImageOpCommand,
    inputs: &[AdmittedOpInput],
    format: honkbox_sd_core::OutputFormat,
    jpeg_quality: Option<u8>,
    token: &tokio_util::sync::CancellationToken,
) -> Result<(Vec<ImageOpUnitOutput>, Vec<crate::jobs::OpsInputRecord>), ImageOpWorkError> {
    let mut outputs = Vec::with_capacity(inputs.len());
    let mut records = Vec::with_capacity(inputs.len());
    for (i, admitted) in inputs.iter().enumerate() {
        // Cooperative cancellation (NEW-3 contract): honored BETWEEN units;
        // the unit in flight always completes.
        if token.is_cancelled() {
            return Err(ImageOpWorkError::Cancelled { completed: i });
        }
        let decoded = decode_admitted_input(admitted)?;
        let pixels = u64::from(decoded.width()) * u64::from(decoded.height());
        let pixel_cap = admitted
            .row
            .budget
            .decoded_pixels
            .unwrap_or(RASTER_FALLBACK_MAX_PIXELS);
        if pixels > pixel_cap {
            return Err(ImageOpWorkError::Bad {
                message: format!(
                    "input is {}x{} ({pixels} pixels); the cap is {pixel_cap} pixels — rejected, not downscaled",
                    decoded.width(),
                    decoded.height()
                ),
                code: "invalid_image_op_inputs",
            });
        }
        let worked = apply_image_op(command, decoded)?;
        let rgba = worked.to_rgba8();
        let generated = honkbox_sd_core::GeneratedImage {
            width: rgba.width(),
            height: rgba.height(),
            rgba: rgba.into_raw(),
        };
        let bytes = honkbox_sd_core::encode_image_with_quality(
            generated,
            format,
            jpeg_quality.unwrap_or(honkbox_sd_core::DEFAULT_JPEG_QUALITY),
        )
        .map_err(|e| ImageOpWorkError::Failed(format!("output encode failed: {e}")))?;
        records.push(crate::jobs::OpsInputRecord {
            source: admitted.source.clone(),
            sha256: admitted.sha256.clone(),
        });
        outputs.push(ImageOpUnitOutput {
            sha256: sha256_hex(&bytes),
            bytes,
        });
    }
    Ok((outputs, records))
}

/// The blocking-pool work result (units + sidecar input records).
pub(super) type ImageOpWork = (Vec<ImageOpUnitOutput>, Vec<crate::jobs::OpsInputRecord>);

/// A pre-admission rejection: the named 400 with NO job (the job is
/// registered only after every input passed the safety checks).
pub(super) fn settled_without_job(error: ImageOpWorkError) -> ImageOpSettled {
    match error {
        ImageOpWorkError::Bad { message, code } => ImageOpSettled {
            response: Err((StatusCode::BAD_REQUEST, message, Some(code.to_owned()))),
            job_id: None,
        },
        ImageOpWorkError::Failed(message) => ImageOpSettled {
            response: Err((StatusCode::INTERNAL_SERVER_ERROR, message, None)),
            job_id: None,
        },
        ImageOpWorkError::Cancelled { completed } => ImageOpSettled {
            response: Err((
                StatusCode::CONFLICT,
                format!("image-op cancelled before completion ({completed} input(s) processed)"),
                Some("job_cancelled".to_owned()),
            )),
            job_id: None,
        },
    }
}

/// A worker crash during pre-admission is an internal 500 (not a hostile
/// input). No job exists yet, but the replay still gets an id like every
/// other failure path; that id is deliberately NOT in the registry, so a
/// cancel against it is a 404.
pub(super) fn image_op_pre_admission_join_failure(message: String) -> ImageOpSettled {
    ImageOpSettled {
        response: Err((StatusCode::INTERNAL_SERVER_ERROR, message, None)),
        job_id: Some(format!("job_{}", ulid::Ulid::new())),
    }
}
