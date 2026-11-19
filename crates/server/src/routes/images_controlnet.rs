//! `ControlNet` request validation for the image-generation routes,
//! extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only the validation helpers live
//! here.

use std::path::PathBuf;

use axum::{http::StatusCode, response::Response};
use base64::Engine as _;
use honkbox_api_types::{ImagesGenerationRequest, ModelCatalog, ModelKind};

use super::openai_constraints::openai_error_response;

/// Resolved `ControlNet` inputs for a validated generation request: the
/// catalog-resolved weights file, the decoded PNG conditioning image, the
/// optional guidance-strength override, and the optional named conditioning
/// type (what the image depicts — `None` = the vendor default `none`, the
/// preprocessed-map behavior).
pub(super) struct ResolvedControlNet {
    pub(super) weights: PathBuf,
    pub(super) image: Vec<u8>,
    pub(super) strength: Option<f32>,
    pub(super) control_type: Option<honkbox_sd_core::sd_control_type_t>,
}

/// Outcome of [`controlnet_pair`]: the `(slug, image)` pair (a placeholder
/// when refusing) plus an optional rejection response.
pub(super) type ControlNetPair = ((String, Vec<u8>), Option<Response>);

/// Validate the honkbox `ControlNet` extension fields: the slug/conditioning
/// pair (both halves or a named 400), strict standard base64, and the PNG
/// magic (the sd path consumes the control image exactly like every other
/// PNG input; anything else cannot be honestly accepted).
///
/// `None` = nothing actually requested (whitespace-only fields are not a
/// request). `Some((pair, None))` = a valid `(slug, decoded image)`;
/// `Some((_, Some(response)))` = a rejection: `controlnet` without
/// `control_image` or vice versa (400 `controlnet_pair`), undecodable
/// base64 or non-PNG bytes (400 `invalid_control_image`).
pub(super) fn controlnet_pair(req: &ImagesGenerationRequest) -> Option<ControlNetPair> {
    let slug = req
        .controlnet
        .as_deref()
        .map(str::trim)
        .filter(|slug| !slug.is_empty());
    let image = req
        .control_image
        .as_deref()
        .map(str::trim)
        .filter(|image| !image.is_empty());
    let (slug, image) = match (slug, image) {
        (None, None) => return None,
        (Some(slug), Some(image)) => (slug, image),
        (Some(_), None) => {
            return Some((
                (String::new(), Vec::new()),
                Some(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    "controlnet and control_image must be supplied together: control_image is missing (the base64 PNG conditioning image for the controlnet slug)".to_owned(),
                    Some("controlnet_pair"),
                )),
            ));
        }
        (None, Some(_)) => {
            return Some((
                (String::new(), Vec::new()),
                Some(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    "controlnet and control_image must be supplied together: controlnet is missing (the catalog slug the conditioning image pairs with)".to_owned(),
                    Some("controlnet_pair"),
                )),
            ));
        }
    };
    let decoded = match base64::engine::general_purpose::STANDARD.decode(image) {
        Ok(bytes) => bytes,
        Err(err) => {
            return Some((
                (String::new(), Vec::new()),
                Some(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("control_image is not usable: invalid base64 ({err})"),
                    Some("invalid_control_image"),
                )),
            ));
        }
    };
    if let Err(reject) = crate::formats::identify_png(&decoded) {
        return Some((
            (String::new(), Vec::new()),
            Some(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "control_image is not usable: decoded bytes are not a PNG ({})",
                    reject.detail()
                ),
                Some("invalid_control_image"),
            )),
        ));
    }
    Some(((slug.to_owned(), decoded), None))
}

/// `ControlNet` request handling for `POST /v1/images/generations` (honkbox
/// extension, not part of the `OpenAI` shape): full validation, then either
/// resolved inputs threaded into the generation or an explicit unsupported
/// verdict for a pipeline that cannot apply them. `(None, None)` = nothing
/// actually requested (whitespace-only fields are not a request) and plain
/// generation proceeds.
///
/// Rejections, in order: the pair/image checks of [`controlnet_pair`], a
/// `controlnet` slug/alias that is not a `controlnet`-kind catalog entry
/// (400 `unknown_controlnet` — the same alias-or-slug rule as SD models and
/// `LoRA`s), a non-finite or out-of-range `controlnet_strength` (400
/// `invalid_controlnet_strength`, the same reject-not-clamp rule as img2img
/// `strength`), a `control_type` that is not one of the engine's names
/// (400 `unknown_control_type` — the name table lives in the vendored
/// engine, so validation and execution can never disagree), and a
/// configured pipeline that cannot carry control inputs
/// (501 `controlnet_unsupported`) — never a silent ignore: rendering
/// without the requested conditioning image would misrepresent the output.
pub(super) fn resolve_controlnet(
    req: &ImagesGenerationRequest,
    catalog: &ModelCatalog,
    pipeline: &dyn honkbox_sd_core::SdPipeline,
) -> (Option<ResolvedControlNet>, Option<Response>) {
    // The named conditioning type (`control_type`): trimmed and treated like
    // every other control field — whitespace-only is not a request.
    let named_type = req
        .control_type
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty());
    let pair = match controlnet_pair(req) {
        // `control_type` modifies a `controlnet`/`control_image` pair; a
        // stray modifier is refused like a lone pair half — a plain render
        // would never use it, so accepting it would silently mislead.
        None if named_type.is_some() => {
            return (
                None,
                Some(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    "control_type without a controlnet/control_image pair: the conditioning type modifies a ControlNet request, so it is refused on its own (a plain render would not use it)".to_owned(),
                    Some("controlnet_pair"),
                )),
            );
        }
        None => return (None, None),
        Some((pair, None)) => pair,
        Some((_, Some(response))) => return (None, Some(response)),
    };
    let (slug, decoded) = pair;
    // Slug: alias-or-slug against the catalog, `controlnet` kind only.
    let entry = match super::resolve_model_entry(&slug, catalog) {
        Some(entry) if entry.kind == ModelKind::ControlNet => entry,
        _ => {
            return (
                None,
                Some(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "unknown controlnet {slug:?}: not a ControlNet catalog slug or alias (see GET /v1/models)"
                    ),
                    Some("unknown_controlnet"),
                )),
            );
        }
    };
    let strength = match req.controlnet_strength {
        Some(v) if v.is_finite() && (0.0..=1.0).contains(&v) && v > 0.0 => Some(v),
        Some(v) => {
            return (
                None,
                Some(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "controlnet_strength {v} outside accepted range (0, 1]: rejected, not clamped"
                    ),
                    Some("invalid_controlnet_strength"),
                )),
            );
        }
        None => None,
    };
    // The conditioning type is validated against the engine's own name
    // table (`control_type_from_name` → `str_to_sd_control_type`): an
    // unknown name is a named 400, not an ignored field — the render would
    // either drop the conditioning (classic nets) or fail mid-generation
    // (union nets).
    let control_type = match named_type {
        None => None,
        Some(name) => match honkbox_sd_core::control_type_from_name(name) {
            Some(parsed) => Some(parsed),
            None => {
                return (
                    None,
                    Some(openai_error_response(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "unknown control_type {name:?}: not a ControlNet type the engine knows (none, openpose, depth, scribble, canny, normal, segment, tile, repaint)"
                        ),
                        Some("unknown_control_type"),
                    )),
                );
            }
        },
    };
    // Honest unsupported verdict: a pipeline without a `ControlNet` branch
    // (the stub backend) would render the prompt unconditioned.
    if pipeline.backend_id() != honkbox_sd_core::SD_CPP_BACKEND_ID {
        return (
            None,
            Some(openai_error_response(
                StatusCode::NOT_IMPLEMENTED,
                format!(
                    "controlnet {slug:?} is accepted and validated but not applied: the configured {:?} pipeline does not apply control images — the request is refused rather than rendered without its conditioning image",
                    pipeline.backend_id()
                ),
                Some("controlnet_unsupported"),
            )),
        );
    }
    (
        Some(ResolvedControlNet {
            weights: PathBuf::from(&entry.path),
            image: decoded,
            strength,
            control_type,
        }),
        None,
    )
}
