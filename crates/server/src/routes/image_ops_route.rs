//! `POST /v1/image-ops` — the bounded CPU image-op route, extracted from
//! `routes.rs`.
//!
//! The router registration stays in `routes.rs`: the `OpenAPI` coverage
//! gate scans that file for `.route()` calls. Only the handler and its
//! request validation live here.

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use honkbox_api_types::{ImageOpCommand, ImageOpRequest, ImageOpResponse, OpenAiError};
use sha2::{Digest as _, Sha256};

use super::image_ops_dedup::{IMAGE_OP_DEDUP, ImageOpReservation};
use super::image_ops_limits::{IMAGE_OP_ID_MAX_CHARS, IMAGE_OP_MAX_INPUTS};
use super::image_ops_params::validate_image_op_params;
use super::image_ops_settle::execute_image_op;
use super::images_inputs::{parse_output_compression, parse_output_format};
use super::middleware::consumer_id;
use super::openai_constraints::openai_error_response;
use super::{AppConfig, chat_prep::with_job_id};

/// A named image-op rejection: status + message + error code — the shape
/// the validators return and the handler turns into the envelope response
/// (the same pattern as `validate_images_request`).
pub(super) type ImageOpRejection = (StatusCode, String, Option<&'static str>);

/// Validate an image-op request (reject-not-clamp, named 400s): id/client
/// string caps, batch cap, output container, JPEG quality, and the per-op
/// param bounds. `Err(ImageOpRejection)` is the named 400.
pub(super) fn validate_image_op_request(
    req: &ImageOpRequest,
) -> Result<(honkbox_sd_core::OutputFormat, Option<u8>), ImageOpRejection> {
    if req.request_id.is_empty() || req.request_id.chars().count() > IMAGE_OP_ID_MAX_CHARS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "request_id must be 1..={IMAGE_OP_ID_MAX_CHARS} characters; got {}",
                req.request_id.chars().count()
            ),
            Some("invalid_image_op_request_id"),
        ));
    }
    if let Some(client) = req.client.as_deref()
        && (client.is_empty() || client.chars().count() > IMAGE_OP_ID_MAX_CHARS)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "client must be 1..={IMAGE_OP_ID_MAX_CHARS} characters; got {}",
                client.chars().count()
            ),
            Some("invalid_image_op_request"),
        ));
    }
    if req.inputs.is_empty() || req.inputs.len() > IMAGE_OP_MAX_INPUTS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "inputs must contain 1..={IMAGE_OP_MAX_INPUTS} images; got {}",
                req.inputs.len()
            ),
            Some("invalid_image_op_inputs"),
        ));
    }
    // Output container: the registry is the source of truth (omitted =
    // PNG); unknown names are a named 400 listing the registry-derived
    // accepted values.
    let format = match req.output_format.as_deref().map(str::trim) {
        None => honkbox_sd_core::OutputFormat::Png,
        Some(raw) => parse_output_format(raw)
            .map_err(|(message, code)| (StatusCode::BAD_REQUEST, message, Some(code)))?,
    };
    let jpeg_quality = parse_output_compression(req.output_compression.as_ref())
        .map_err(|(message, code)| (StatusCode::BAD_REQUEST, message, Some(code)))?;
    // Matting is PNG-in/PNG-out (RGBA cutout): any other output container
    // is a named 400, never a silent re-encode of the cutout.
    if matches!(req.command, ImageOpCommand::Matting)
        && !matches!(format, honkbox_sd_core::OutputFormat::Png)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "matting output must be png (the RGBA cutout); got {} — rejected, not converted",
                format.name()
            ),
            Some("invalid_image_op_matting_output"),
        ));
    }
    validate_image_op_params(&req.command)
        .map_err(|(message, code)| (StatusCode::BAD_REQUEST, message, Some(code)))?;
    Ok((format, jpeg_quality))
}

/// `POST /v1/image-ops` (epic-editor-integration): one bounded CPU op over
/// `1..=8` inputs — resize/crop/rotate/grayscale/format/matting — deterministic by
/// construction (Lanczos3, fixed encoders), with `request_id`-keyed
/// idempotency (same key + same canonical command replays the first job's
/// response byte-identically, no double execution; same key + different
/// command is a named 400). The job is registry-visible while live (kind
/// `image_ops`, cancel honored BETWEEN inputs), its lifecycle streams on
/// `GET /v1/events`, and the result bundle lands under
/// `<data_dir>/jobs/{bundle}/` via the existing jobs path (sidecar: command
/// hash, input/output hashes, tool version, attempts, outcome). Inputs are
/// data URIs or this server's own bundle artifacts only — no paths, no
/// `file://`, no remote URLs (nothing is fetched, so there is no SSRF
/// surface). The Tier 0 ops are exactly what the in-process image crate
/// covers (TASK-cheap-ops-spike), so they have no 501 path — unknown
/// op tags are a 400 from parsing. Matting is the one sidecar op:
/// PNG-in/PNG-out through `HONKBOX_MATTING_URL`, an honest 501
/// (`matting_unavailable`) while unconfigured. The capabilities
/// advertisement lists the vocabulary plus the sidecar liveness flag.
#[utoipa::path(
    post,
    path = "/v1/image-ops",
    request_body = ImageOpRequest,
    responses(
        (status = 200, description = "One output per input, in order", body = ImageOpResponse),
        (status = 400, description = "Unknown field, unknown op, out-of-range params, oversized batch, bad input container/data URI/ artifact, unknown output_format or output_compression, or request_id conflict", body = OpenAiError),
        (status = 409, description = "Job cancelled before completion (between inputs)", body = OpenAiError),
        (status = 429, description = "CPU admission busy (ledger)", body = OpenAiError),
        (status = 501, description = "Matting sidecar unconfigured (matting_unavailable)", body = OpenAiError),
        (status = 500, description = "Decode/transform/encode failed", body = OpenAiError)
    )
)]
pub(crate) async fn post_image_ops(
    State(config): State<AppConfig>,
    headers: HeaderMap,
    body: Result<Json<ImageOpRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(req)) => req,
        Err(rejection) => {
            // deny_unknown_fields / malformed JSON / wrong types all land
            // here as one named 400 carrying serde's specific message.
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid image-op request: {}", rejection.body_text()),
                Some("invalid_image_op_request"),
            );
        }
    };
    let (format, jpeg_quality) = match validate_image_op_request(&req) {
        Ok(validated) => validated,
        Err((status, message, code)) => {
            return openai_error_response(status, message, code);
        }
    };
    // Canonical command hash: command + inputs + output container/quality
    // only. Self-declared `client` and `request_id` are deliberately excluded
    // so changing a label on a retry still replays the first execution.
    let canonical = serde_json::json!({
        "command": &req.command,
        "inputs": &req.inputs,
        "output_format": &req.output_format,
        "output_compression": &req.output_compression,
    });
    let command_sha256: [u8; 32] =
        Sha256::digest(serde_json::to_vec(&canonical).unwrap_or_default()).into();
    // Identity comes from the shared scope layer: env name or bearer digest.
    let consumer = consumer_id(&headers);
    let reserved = {
        // std mutex, no await inside: reserve-or-conflict is decided here.
        let mut dedup = IMAGE_OP_DEDUP
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match dedup.reserve(&consumer, &req.request_id, command_sha256) {
            ImageOpReservation::Conflict => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "request_id {:?} was already used with a different command — retries must resend the identical request; use a fresh request_id for different work",
                        req.request_id
                    ),
                    Some("image_op_request_id_conflict"),
                );
            }
            ImageOpReservation::Execute(entry) => (entry, false),
            ImageOpReservation::Replay(entry) => (entry, true),
        }
    };
    let (entry, is_replay) = reserved;
    // Single flight: whether this caller is the executor or a waiting
    // retry, `get_or_init` runs the execution exactly once per entry —
    // concurrent same-key retries all await the same settled outcome.
    let cmd = req.clone();
    let cfg = config.clone();
    let settled = entry
        .outcome
        .get_or_init(|| async move {
            execute_image_op(&cfg, &cmd, command_sha256, format, jpeg_quality).await
        })
        .await
        .clone();
    if !is_replay {
        // Exactly one caller owns the entry: a success stays remembered
        // (bounded), a failure is forgotten so a genuine retry re-executes.
        let retained = match &settled.response {
            Ok(response) => {
                256 + response
                    .outputs
                    .iter()
                    .map(|out| out.b64_json.len())
                    .sum::<usize>()
            }
            Err((_, message, _)) => 128 + message.len(),
        };
        let mut dedup = IMAGE_OP_DEDUP
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if settled.response.is_ok() {
            dedup.remember(&consumer, &req.request_id, retained);
        } else {
            dedup.forget(&consumer, &req.request_id);
        }
    }
    let response = match &settled.response {
        Ok(body) => Json(body.clone()).into_response(),
        Err((status, message, code)) => {
            openai_error_response(*status, message.clone(), code.as_deref())
        }
    };
    match settled.job_id.as_deref() {
        Some(job_id) => with_job_id(response, job_id),
        None => response,
    }
}
