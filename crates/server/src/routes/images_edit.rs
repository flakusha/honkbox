//! `POST /v1/images/edits` — the img2img/inpaint route, extracted from
//! `routes.rs`.
//!
//! The router registration stays in `routes.rs`: the `OpenAPI` coverage
//! gate scans that file for `.route()` calls. Only the handler and its
//! helpers live here.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use axum::{
    Json,
    body::Bytes,
    extract::{Multipart, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use base64::Engine as _;
use honkbox_api_types::{
    ImageArtifact, ImageB64, ImagesGenerationResponse, ImagesLora, OpenAiError,
};

use super::audio_multipart::bad_multipart;
use super::chat_prep::{admit_then_register, with_job_id};
use super::images_generate::{
    GenerateFailure, generate_image_bytes, image_scheduler_name, write_generation_bundle,
    write_generation_bundle_strict,
};
use super::images_inputs::{
    DeliveryMode, parse_output_format, resolve_loras, validate_prompt_safety,
};
use super::images_recipe::{base_model_of, resolve_sd_model, validate_recipe};
use super::openai_constraints::openai_error_response;
use super::{AppConfig, epoch_secs, resolve_model_entry};

use crate::catalog::license_gate::LicenseRefusal;
use sha2::{Digest as _, Sha256};

/// Output dimensions from a PNG's IHDR (big-endian `u32` at offsets 16/20).
/// Only called on registry-verified bytes (`require_png_bytes` /
/// `crate::formats::identify_png` own the magic check); `None` for
/// truncated data.
fn png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 24 {
        return None;
    }
    Some((
        u32::from_be_bytes(bytes[16..20].try_into().ok()?),
        u32::from_be_bytes(bytes[20..24].try_into().ok()?),
    ))
}

/// Validate PNG-required bytes (edits `mask`; `ControlNet` `control_image`):
/// the registry identifies by magic, and `need_ihdr` additionally demands a
/// parseable IHDR. `Err` is the message reason — it names the sniffed
/// format, never an extension. (`image` runs the same sniff inline because
/// it also consumes the IHDR dimensions.)
fn require_png_bytes(bytes: &[u8], need_ihdr: bool) -> Result<(), String> {
    let sniffed = crate::formats::identify_png(bytes);
    if sniffed.is_err() || (need_ihdr && png_dimensions(bytes).is_none()) {
        return Err(match sniffed {
            Err(reject) => reject.detail(),
            Ok(_) => "IHDR dimensions missing".to_owned(),
        });
    }
    Ok(())
}

/// Collected multipart fields for `/v1/images/edits`.
pub(super) struct ImageEditUpload {
    /// `image` part: the source PNG (declared content type + bytes).
    pub(super) image: Option<(String, Bytes)>,
    /// `mask` part: optional inpaint-mask PNG.
    pub(super) mask: Option<(String, Bytes)>,
    /// Render prompt (required).
    pub(super) prompt: Option<String>,
    /// Optional SD model (alias or catalog slug).
    pub(super) model: Option<String>,
    /// img2img denoise strength in `(0, 1]`; default
    /// [`honkbox_sd_core::IMG2IMG_STRENGTH`] (the vendor default).
    pub(super) strength: Option<f32>,
    /// Output size `WxH`; default = the source image's dimensions.
    pub(super) size: Option<String>,
    /// Recipe overrides, validated exactly like `generations` (a named
    /// recipe plus per-field overrides; explicit fields win).
    pub(super) recipe: Option<String>,
    pub(super) sampler: Option<String>,
    pub(super) scheduler: Option<String>,
    pub(super) cfg_scale: Option<f32>,
    pub(super) negative_prompt: Option<String>,
    pub(super) seed: Option<u64>,
    /// Diffusion steps override; validated like `generations`.
    pub(super) steps: Option<u32>,
    /// Output container: `png` (default) | `jpeg` | `webp`; validated in
    /// the handler (named 400) like `generations`.
    pub(super) output_format: Option<String>,
    /// JPEG quality `[0, 100]`; kept raw here so the handler can reject
    /// non-numeric / out-of-range values with the named 400, never a clamp.
    pub(super) output_compression: Option<String>,
    /// Per-request `LoRA` adapters, the same validated `ImagesLora` list as
    /// txt2img (max 2, strength `[-2, 2]`); parsed as a JSON array from the
    /// form field and resolved by [`resolve_loras`] — no separate edits-side
    /// validation.
    pub(super) loras: Option<Vec<ImagesLora>>,
}

/// Collect the multipart fields. Unknown fields are ignored (client SDKs
/// add extras). The boxed error is the ready-made `OpenAI` envelope.
async fn parse_image_edit_upload(
    mut multipart: Multipart,
) -> Result<ImageEditUpload, Box<Response>> {
    let mut edit = ImageEditUpload {
        image: None,
        mask: None,
        prompt: None,
        model: None,
        strength: None,
        size: None,
        recipe: None,
        sampler: None,
        scheduler: None,
        cfg_scale: None,
        negative_prompt: None,
        seed: None,
        steps: None,
        output_format: None,
        output_compression: None,
        loras: None,
    };
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| bad_multipart(format!("invalid multipart body: {err}")))?
    {
        let name = field.name().unwrap_or("").to_owned();
        match name.as_str() {
            "image" | "mask" => {
                let content_type = field.content_type().unwrap_or("").to_owned();
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|err| bad_multipart(format!("unreadable {name} field: {err}")))?;
                if name == "image" {
                    edit.image = Some((content_type, bytes));
                } else {
                    edit.mask = Some((content_type, bytes));
                }
            }
            "prompt" | "model" | "size" | "recipe" | "sampler" | "scheduler"
            | "negative_prompt" | "strength" | "cfg_scale" | "seed" | "steps" | "output_format"
            | "output_compression" | "loras" => {
                let text = field
                    .text()
                    .await
                    .map_err(|err| bad_multipart(format!("unreadable {name} field: {err}")))?;
                set_edit_field(&mut edit, &name, &text)?;
            }
            _ => {}
        }
    }
    Ok(edit)
}

/// One non-file multipart field, with per-field parse validation (the
/// semantic range checks live in the handler, next to their 400 codes).
fn set_edit_field(edit: &mut ImageEditUpload, name: &str, text: &str) -> Result<(), Box<Response>> {
    let value = text.trim().to_owned();
    match name {
        "prompt" | "model" | "size" | "recipe" | "sampler" | "scheduler" | "negative_prompt"
        | "output_format" => {
            let value = (!value.is_empty()).then_some(value);
            match name {
                "prompt" => edit.prompt = value,
                "model" => edit.model = value,
                "size" => edit.size = value,
                "recipe" => edit.recipe = value,
                "sampler" => edit.sampler = value,
                "scheduler" => edit.scheduler = value,
                "output_format" => edit.output_format = value,
                _ => edit.negative_prompt = value,
            }
        }
        "strength" => {
            edit.strength =
                Some(value.parse().map_err(|_| {
                    bad_multipart("strength must be a number in (0, 1]".to_owned())
                })?);
        }
        "cfg_scale" => {
            edit.cfg_scale = Some(value.parse().map_err(|_| {
                bad_multipart("cfg_scale must be a number in [1.0, 30.0]".to_owned())
            })?);
        }
        "seed" => {
            edit.seed =
                Some(value.parse().map_err(|_| {
                    bad_multipart("seed must be a non-negative integer".to_owned())
                })?);
        }
        "steps" => {
            edit.steps =
                Some(value.parse().map_err(|_| {
                    bad_multipart("steps must be an integer in [1, 100]".to_owned())
                })?);
        }
        "output_compression" => {
            edit.output_compression = (!value.is_empty()).then_some(value);
        }
        "loras" => {
            // Same shape as txt2img (`[{slug, strength}, ...]`); the semantic
            // checks (count, range, catalog, single dir) all live in
            // [`resolve_loras`], next to their 400 codes.
            let parsed: Vec<ImagesLora> = serde_json::from_str(&value).map_err(|err| {
                bad_multipart(format!(
                    "loras must be a JSON array of {{slug, strength}} objects: {err}"
                ))
            })?;
            edit.loras = Some(parsed);
        }
        _ => {}
    }
    Ok(())
}

/// `POST /v1/images/edits`: pragmatic honkbox img2img/inpaint shape over
/// the same SD pipeline seam as `/v1/images/generations` (single image per
/// request, `n` fixed at 1). `image` is a required PNG (magic-sniffed; the
/// declared content type is not trusted), `mask` an optional PNG that
/// turns the run into inpainting; `prompt` is required; `strength` (the
/// denoise amount, `(0, 1]`, default 0.75) applies to the init image;
/// `size` defaults to the source image's dimensions; the remaining recipe
/// fields (`recipe`/`sampler`/`scheduler`/`cfg_scale`/`steps`/
/// `negative_prompt`/`seed`) validate exactly like `generations` — a named
/// recipe plus per-field overrides, explicit fields winning. `output_format`
/// (`png` default | `jpeg` | `webp` lossless) and `output_compression`
/// (JPEG quality `[0, 100]`; ignored for lossless PNG/WebP) mirror the
/// `generations` knobs — unknown format names and non-numeric/out-of-range
/// compression values are named 400s, never clamps. No
/// `hires_fix` on this route: the output stays at the source image's
/// dimensions. Inputs are staged as temp
/// PNG paths (the vendor reads images from the filesystem) and removed
/// after the generation, including on failure. `loras` (a JSON array of
/// `{slug, strength}` objects) resolves through the same
/// [`resolve_loras`] as `generations` — max 2, strength `[-2, 2]`
/// reject-not-clamp, catalog lookup, single directory — with license notes
/// surfaced as response `warnings` and the resolved slugs/strengths recorded
/// in the bundle sidecar. The stub
/// backend honestly ignores image inputs, so its pixels never depend on
/// the upload; tests cover the wire/validation paths. Errors are `OpenAI`
/// envelopes: 400 (missing image/prompt, non-PNG bytes, bad
/// `strength`/`size`/`recipe`/`output_format`/`output_compression`/`loras`
/// fields), 404
/// unknown model, 429 busy, 500 staging or generation failure. `X-Honkbox-Delivery:
/// artifact` stores the image in the artifact registry and answers descriptors,
/// `bundle` and `job_id` instead of base64; `inline` (default) keeps the
/// historical base64 `data` response.
#[allow(clippy::too_many_lines)] // linear multipart handler; see `post_images_generations`
#[utoipa::path(
    post,
    path = "/v1/images/edits",
    request_body(content = String, description = "multipart/form-data: image (PNG, required), mask? (PNG), prompt (required), model?, strength?, size?, recipe?, sampler?, scheduler?, cfg_scale?, steps?, negative_prompt?, seed?, output_format? (png|jpeg|webp), output_compression? (JPEG quality 0-100), loras? (JSON array of {slug, strength})"),
    responses(
        (status = 200, description = "Edited image (base64 bytes by default; X-Honkbox-Delivery: artifact returns artifacts[] with bundle and job_id instead)", body = ImagesGenerationResponse),
        (status = 400, description = "Missing image/prompt, non-PNG bytes, bad strength, size, recipe, output_format, output_compression, loras, or X-Honkbox-Delivery fields", body = OpenAiError),
        (status = 404, description = "Unknown model", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Generation failed", body = OpenAiError)
    )
)]
pub(super) async fn post_images_edits(
    State(config): State<AppConfig>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Response {
    let delivery = match DeliveryMode::parse(
        headers
            .get("X-Honkbox-Delivery")
            .map(|value| value.to_str().unwrap_or_default()),
    ) {
        Ok(delivery) => delivery,
        Err((message, code)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
    };
    let edit = match parse_image_edit_upload(multipart).await {
        Ok(edit) => edit,
        Err(response) => return *response,
    };
    // Same shared prompt gate as `/v1/images/generations`: refuse before
    // image checks, admission or any pipeline work.
    if let Some(prompt) = edit.prompt.as_deref()
        && let Err((message, code)) = validate_prompt_safety(prompt)
    {
        return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
    }
    let Some((_, image_bytes)) = edit.image else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "image is required: attach the source PNG as the `image` form field".to_owned(),
            Some("image_required"),
        );
    };
    // Registry authority: identify by magic — never the declared content
    // type or the multipart filename — then require a real canvas (IHDR
    // parses to non-zero dimensions; it doubles as the `size` default).
    let sniffed = crate::formats::identify_png(&image_bytes);
    let dims = sniffed
        .as_ref()
        .ok()
        .and_then(|_| png_dimensions(&image_bytes))
        .filter(|(w, h)| *w > 0 && *h > 0);
    if dims.is_none() {
        let reason = sniffed
            .err()
            .map_or("IHDR dimensions missing or zero".to_owned(), |reject| {
                reject.detail()
            });
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("image must be PNG bytes ({reason})"),
            Some("invalid_image"),
        );
    }
    if let Some((_, mask_bytes)) = edit.mask.as_ref()
        && let Err(reason) = require_png_bytes(mask_bytes, true)
    {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("mask must be PNG bytes ({reason})"),
            Some("invalid_mask"),
        );
    }
    let Some(prompt) = edit.prompt else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "prompt is required".to_owned(),
            Some("prompt_required"),
        );
    };
    let strength = match edit.strength {
        Some(v) if v.is_finite() && (0.0..=1.0).contains(&v) && v > 0.0 => v,
        Some(v) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("strength {v} outside accepted range (0, 1]: rejected, not clamped"),
                Some("invalid_strength"),
            );
        }
        None => honkbox_sd_core::IMG2IMG_STRENGTH,
    };
    let (width, height) = match edit.size.as_deref() {
        Some(raw) => match crate::jobs::parse_size(raw) {
            Ok(size) => size,
            Err(err) => {
                return openai_error_response(StatusCode::BAD_REQUEST, err, Some("invalid_size"));
            }
        },
        // Default: the source image's own dimensions (PNG IHDR, verified
        // by the sniff above).
        None => dims.expect("PNG dimensions verified above"),
    };
    // Mask scope is explicit: a supplied mask selects `inpaint`; no mask means
    // full-image `replace` (the vendor's flat white mask). Reject a mismatched
    // mask before admission instead of relying on the engine's implicit resize.
    let (mask_mode, mask_sha256) = match edit.mask.as_ref() {
        Some((_, mask_bytes)) => {
            if png_dimensions(mask_bytes) != Some((width, height)) {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("mask dimensions must match {width}x{height}"),
                    Some("mask_dimensions_mismatch"),
                );
            }
            (
                Some("inpaint"),
                Some(crate::artifacts::hex_lower(
                    &Sha256::digest(mask_bytes.as_ref())[..],
                )),
            )
        }
        None => (Some("replace"), None),
    };
    let recipe = match validate_recipe(
        edit.recipe.as_deref(),
        edit.sampler.as_deref(),
        edit.scheduler.as_deref(),
        edit.cfg_scale,
        edit.negative_prompt.as_deref(),
        edit.seed,
        edit.steps,
    ) {
        Ok(recipe) => recipe,
        Err((message, code)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
    };
    let format = match parse_output_format(edit.output_format.as_deref().unwrap_or("png").trim()) {
        Ok(format) => format,
        Err((message, code)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
    };
    let compression = match edit.output_compression.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(raw) => match raw.parse::<u8>() {
            Ok(q) if q <= 100 => Some(q),
            _ => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "output_compression must be an integer in [0, 100]; got {raw:?}: rejected, not clamped"
                    ),
                    Some("invalid_output_compression"),
                );
            }
        },
    };
    let catalog = config.catalog_snapshot();
    // Per-request `loras`: validated and resolved by the shared
    // [`resolve_loras`] (max 2, strength `[-2, 2]` reject-not-clamp, catalog
    // lookup, single directory) — identical to `/v1/images/generations`.
    let checkpoint_base = edit
        .model
        .as_deref()
        .and_then(|model| resolve_sd_model(model, &catalog))
        .or(config.sd_model.as_deref())
        .and_then(|slug| base_model_of(slug, &catalog));
    let (pipeline_loras, lora_records, lora_warnings) = match resolve_loras(
        edit.loras.as_deref().unwrap_or_default(),
        &catalog,
        checkpoint_base,
    ) {
        Ok(resolved) => resolved,
        Err((message, code)) => {
            let status = if code.starts_with("model_license_") {
                StatusCode::PRECONDITION_FAILED
            } else {
                StatusCode::BAD_REQUEST
            };
            return openai_error_response(status, message, Some(code));
        }
    };
    let slug: String = match edit
        .model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
    {
        Some(model) => match resolve_sd_model(model, &catalog) {
            Some(slug) => slug.to_owned(),
            None => {
                return openai_error_response(
                    StatusCode::NOT_FOUND,
                    format!(
                        "unknown model {model:?}: not an SD catalog slug or alias (see GET /v1/models)"
                    ),
                    Some("model_not_found"),
                );
            }
        },
        // Server default: the startup-configured SD pipeline.
        None => config.sd_model.clone().unwrap_or_default(),
    };
    // License gate before GPU admission / bundle writes.
    if let Err(refusal) =
        crate::catalog::license_gate::gate(&slug, &catalog, crate::config::licenses())
    {
        return openai_error_response(
            LicenseRefusal::status(),
            refusal.message(),
            Some(refusal.code()),
        );
    }
    // Admission first, registration second: a 429 has no job or `X-Job-Id`.
    // Edits stay non-interruptible (`false`), like the never-cancelled
    // token passed to the pipeline below.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Sd).await,
        crate::jobs_registry::JobKind::Images,
        Some(slug.clone()),
        false,
    ) {
        Ok(pair) => pair,
        Err((status, message)) => return openai_error_response(status, message, None),
    };
    // Stage the inputs as scratch PNGs (the vendor reads images from
    // paths); both are removed after the generation, on every path below.
    let token = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let init_path = std::env::temp_dir().join(format!(
        "honkbox-edit-init-{}-{token}.png",
        std::process::id()
    ));
    let mask_path = std::env::temp_dir().join(format!(
        "honkbox-edit-mask-{}-{token}.png",
        std::process::id()
    ));
    let staged_mask = edit.mask.is_some();
    let staging =
        std::fs::write(&init_path, &image_bytes).and_then(|()| match edit.mask.as_ref() {
            Some((_, mask_bytes)) => std::fs::write(&mask_path, mask_bytes),
            None => Ok(()),
        });
    if let Err(err) = staging {
        let _ = std::fs::remove_file(&init_path);
        let _ = std::fs::remove_file(&mask_path);
        drop(lease);
        return with_job_id(
            openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to stage input image: {err}"),
                Some("staging_failed"),
            ),
            job.id(),
        );
    }
    let pipeline = Arc::clone(&config.sd);
    let backend_id = pipeline.backend_id();
    // Provenance (TASK-determinism-contract): verified-digest sidecar when
    // the model was a checksum-verified download; null otherwise (never a
    // fabricated hash). Vendored engine identity from the backend id
    // (None for the stub — no separate engine).
    let model_sha256 = resolve_model_entry(&slug, &catalog)
        .and_then(|entry| crate::downloads::artifact_sha256(std::path::Path::new(&entry.path)));
    let engine_version = honkbox_sd_core::engine_version_for(backend_id);
    let started = Instant::now();
    let generated = generate_image_bytes(
        pipeline,
        honkbox_sd_core::Txt2ImgRequest {
            prompt: prompt.clone(),
            seed: recipe.seed_base,
            steps: recipe.steps,
            width,
            height,
            model_slug: slug.clone(),
            clip: None,
            vae: None,
            sampler: Some(recipe.sampler),
            scheduler: Some(recipe.scheduler),
            cfg_scale: Some(recipe.cfg_scale),
            negative_prompt: recipe.negative_prompt.clone(),
            loras: (!pipeline_loras.is_empty()).then(|| pipeline_loras.clone()),
            init_image: Some(init_path.clone()),
            mask_image: staged_mask.then(|| mask_path.clone()),
            strength: Some(strength),
            // No hires pass on img2img: the vendor applies hires to the
            // txt2img branch only, and the route fixes the output at the
            // source image's dimensions.
            hires: None,
            // ControlNet is a generations-route extension; img2img keeps
            // the plain OpenAI edits shape.
            control_net: None,
            control_image: None,
            controlnet_strength: None,
            control_type: None,
        },
        format,
        compression,
        // The edits job is non-interruptible, so nothing can cancel this
        // render: a fresh flag and a never-cancelled token (the pipeline
        // never observes a cancel here).
        Arc::new(AtomicBool::new(false)),
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let _ = std::fs::remove_file(&init_path);
    if staged_mask {
        let _ = std::fs::remove_file(&mask_path);
    }
    let bytes = match generated {
        Ok(bytes) => bytes,
        Err(failure) => {
            // `Cancelled` cannot happen with a never-set flag; the arm
            // keeps the match exhaustive without inventing a success.
            let message = match failure {
                GenerateFailure::Failed(message) => message,
                GenerateFailure::Cancelled => String::from("generation cancelled"),
            };
            return with_job_id(
                openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    message,
                    Some("generation_failed"),
                ),
                job.id(),
            );
        }
    };
    let annotations = match super::nsfw::classify_for_image(&bytes, format.extension()) {
        Ok(annotations) => annotations,
        Err((status, message, code)) => {
            return with_job_id(openai_error_response(status, message, Some(code)), job.id());
        }
    };
    let bundle_annotations = annotations
        .iter()
        .map(super::nsfw::bundle_annotation)
        .collect();
    let bundle_inputs = crate::jobs::BundleInputs {
        model_slug: slug,
        prompt,
        seed: recipe.seed_base,
        steps: recipe.steps,
        width,
        height,
        format,
        backend_id,
        model_sha256,
        engine_version,
        sampler: recipe.sampler.name().to_string(),
        scheduler: image_scheduler_name(recipe.scheduler),
        cfg_scale: recipe.cfg_scale,
        negative_prompt: recipe.negative_prompt,
        loras: lora_records,
        strength: Some(strength),
        mask_mode,
        mask_sha256,
        checkpoint_base_model: checkpoint_base.map(str::to_owned),
        recipe: recipe.name.map(str::to_owned),
        hires: None,
        annotations: bundle_annotations,
    };
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let mut data = Vec::new();
    let mut artifacts = Vec::new();
    let mut bundle = None;
    match delivery {
        DeliveryMode::Inline => {
            write_generation_bundle(&config, &bundle_inputs, &bytes, elapsed_ms);
            data.push(ImageB64 {
                b64_json: base64::engine::general_purpose::STANDARD.encode(&bytes),
            });
        }
        DeliveryMode::Artifact => {
            let dir =
                match write_generation_bundle_strict(&config, &bundle_inputs, &bytes, elapsed_ms) {
                    Ok(dir) => dir,
                    Err(message) => {
                        return with_job_id(
                            openai_error_response(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                message,
                                Some("bundle_write_failed"),
                            ),
                            job.id(),
                        );
                    }
                };
            let info = match config.artifacts.put_bytes(&bytes, format.mime()) {
                Ok(info) => info,
                Err(err) => {
                    return with_job_id(
                        openai_error_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("artifact store failed: {err}"),
                            Some("artifact_store_failed"),
                        ),
                        job.id(),
                    );
                }
            };
            let name = format!("final_00.{}", format.extension());
            crate::jobs_registry::global().emit_result(
                job.id(),
                &info.id,
                &info.sha256,
                info.size_bytes,
                crate::jobs_registry::ArtifactFrameMeta {
                    name: &name,
                    kind: "image",
                    index: 0,
                    primary: true,
                },
            );
            if !annotations.is_empty() {
                crate::jobs_registry::global().emit_annotations(job.id(), annotations.clone());
            }
            artifacts.push(ImageArtifact {
                artifact_id: info.id,
                content_type: info.content_type,
                size_bytes: info.size_bytes,
                job_id: job.id().to_owned(),
            });
            bundle = Some(dir);
        }
    }
    drop(lease);
    with_job_id(
        Json(ImagesGenerationResponse {
            created: epoch_secs(),
            data,
            warnings: (!lora_warnings.is_empty()).then_some(lora_warnings),
            artifacts: (!artifacts.is_empty()).then_some(artifacts),
            bundle,
            job_id: (delivery == DeliveryMode::Artifact).then(|| job.id().to_owned()),
        })
        .into_response(),
        job.id(),
    )
}
