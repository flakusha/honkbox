//! `POST /v1/images/generations` handler, extracted from `routes.rs`.
//!
//! The router registration and the `OpenAPI` `paths(...)` entry stay in
//! `routes.rs`; only the handler and its private helpers live here.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use base64::Engine as _;
use honkbox_api_types::{
    ImageArtifact, ImageB64, ImagesGenerationRequest, ImagesGenerationResponse, OpenAiError,
};

use super::chat_prep::{admit_then_register, with_job_id};
use super::images_controlnet::resolve_controlnet;
use super::images_inputs::{DeliveryMode, resolve_loras, validate_images_request};
use super::images_recipe::resolve_sd_model;
use super::request_cache::RequestCache;
use super::{AppConfig, epoch_secs, resolve_model_entry};
use crate::catalog::license_gate::LicenseRefusal;
use crate::jobs::determinism::canonical_hash;

/// Inline-only canonical-request cache: identical requests replay the first
/// response without ever acquiring a GPU lease. Artifact delivery is excluded
/// because each run mints fresh job/artifact ids.
///
/// Caches are scoped to the serving pipeline/catalog pair, never
/// process-global: distinct `AppConfig` setups in one process must not replay
/// each other's responses. The map keeps strong `Arc`s to both while the
/// cache holds live entries, so those addresses cannot be recycled into a
/// different config and collide with the same key.
struct ScopedGenerationCache {
    /// Pin both addresses while responses may still be replayed.
    _pipeline: Arc<dyn honkbox_sd_core::SdPipeline>,
    _catalog: Arc<std::sync::RwLock<Arc<honkbox_api_types::ModelCatalog>>>,
    cache: RequestCache<serde_json::Value>,
}

type GenerationCaches = HashMap<(usize, usize), Arc<ScopedGenerationCache>>;

static GENERATION_CACHES: LazyLock<Mutex<GenerationCaches>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Cache for this config's pipeline/catalog pair. Empty (expired) caches are
/// reclaimed, and with them the address pins: recycled addresses start fresh.
fn scoped_generation_cache(config: &AppConfig) -> Arc<ScopedGenerationCache> {
    let key = (
        Arc::as_ptr(&config.sd).cast::<()>() as usize,
        Arc::as_ptr(&config.catalog).cast::<()>() as usize,
    );
    let mut caches = GENERATION_CACHES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    caches.retain(|_, cache| !cache.cache.is_empty());
    Arc::clone(caches.entry(key).or_insert_with(|| {
        Arc::new(ScopedGenerationCache {
            _pipeline: Arc::clone(&config.sd),
            _catalog: Arc::clone(&config.catalog),
            cache: RequestCache::new(64, Duration::from_secs(300)),
        })
    }))
}

/// Write one HTTP generation's job bundle (best-effort: a bundle-write
/// failure never fails an otherwise successful generation).
pub(super) fn write_generation_bundle(
    config: &AppConfig,
    inputs: &crate::jobs::BundleInputs,
    bytes: &[u8],
    elapsed_ms: u64,
) {
    let Some(data_dir) = &config.data_dir else {
        return;
    };
    if let Err(err) = crate::jobs::write_bundle(data_dir, inputs, bytes, elapsed_ms) {
        tracing::warn!(error = %err, "job bundle write failed for HTTP generation");
    }
}

/// Artifact-mode bundle write: a missing data dir or a failed write fails
/// the request loudly (the inline path above stays best-effort warn-only).
/// Returns the bundle directory for the `bundle` response field.
pub(super) fn write_generation_bundle_strict(
    config: &AppConfig,
    inputs: &crate::jobs::BundleInputs,
    bytes: &[u8],
    elapsed_ms: u64,
) -> Result<String, String> {
    let data_dir = config
        .data_dir
        .as_ref()
        .ok_or_else(|| "artifact delivery requires a configured data dir".to_owned())?;
    let dir = crate::jobs::write_bundle(data_dir, inputs, bytes, elapsed_ms)
        .map_err(|err| format!("job bundle write failed: {err}"))?;
    Ok(dir.to_string_lossy().into_owned())
}

/// Default generation recipe for the HTTP surface: the same defaults as
/// `honkbox generate` (seed 42 base, 20 steps, euler/normal, CFG 7.0),
/// recorded in the bundle sidecars. Per-image seed = base + index; a
/// request may override the recipe (`sampler`/`scheduler`/`cfg_scale`/
/// `negative_prompt`/`seed`) or name a curated `recipe` — explicit fields
/// win over recipe defaults, and omitted fields fall back to these server
/// defaults.
pub(super) const IMAGE_SEED_BASE: u64 = 42;
/// Server-default diffusion steps for the HTTP surface (CLI default): used
/// when the request names no recipe and passes no `steps` override.
pub(super) const IMAGE_STEPS: u32 = 20;

/// Why an image generation stopped: a backend failure (500), or the job's
/// cancel token/flag interrupted the in-flight render — kept a named
/// variant so the route can answer the honest 409 `job_cancelled` instead
/// of a generic 500.
pub(super) enum GenerateFailure {
    Cancelled,
    Failed(String),
}

/// One fresh engine-stop flag per image. The job token is the cross-image
/// cancel state; a flag left set by image 1 must never be inherited by
/// image 2 before it starts.
pub(super) fn new_image_cancel_flag() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

/// 409 body for a cancel that landed before image `index` started. The
/// wording differs from the mid-generation arm so a client can tell that
/// no render was in flight.
pub(super) fn image_cancel_before_start(
    job_id: &str,
    index: u64,
    total: u64,
    completed: u64,
) -> String {
    format!(
        "job {job_id} cancelled before image {index} of {total} started ({completed} of {total} image(s) generated)"
    )
}

/// 409 body for a cancel that aborted image `index` while it rendered.
pub(super) fn image_cancel_mid_generation(
    job_id: &str,
    index: u64,
    total: u64,
    completed: u64,
) -> String {
    format!(
        "job {job_id} cancelled while generating image {index} of {total} ({completed} of {total} image(s) generated)"
    )
}

/// One PNG generation under `spawn_blocking` (GPU work off the async
/// runtime, like every other GPU route). Returns the encoded container
/// bytes, or why it stopped. `cancel` is THIS image's engine-stop flag and
/// `token` the job's cross-image registry token: the cancel endpoint can
/// only reach the token, so this future bridges the two — the moment the
/// token is marked the flag is set, and a pipeline polling the flag stops
/// the running sd.cpp generation through `sd_cancel_generation`. The bridge
/// waits for the worker to return, then re-checks the token so a backend
/// that ignores the flag cannot answer 200 after a cancel.
pub(super) async fn generate_image_bytes(
    pipeline: std::sync::Arc<dyn honkbox_sd_core::SdPipeline>,
    txt2img: honkbox_sd_core::Txt2ImgRequest,
    format: honkbox_sd_core::OutputFormat,
    compression: Option<u8>,
    cancel: Arc<AtomicBool>,
    token: tokio_util::sync::CancellationToken,
) -> Result<Vec<u8>, GenerateFailure> {
    generate_image_bytes_inner(pipeline, txt2img, format, compression, cancel, token, None).await
}

/// Artifact-mode variant: pipeline steps feed the job's `job_progress` SSE
/// frames. `generate_with_progress` defaults to `generate`, so the encoded
/// bytes are unchanged; only the callback is new.
pub(super) async fn generate_image_bytes_with_progress(
    pipeline: std::sync::Arc<dyn honkbox_sd_core::SdPipeline>,
    txt2img: honkbox_sd_core::Txt2ImgRequest,
    format: honkbox_sd_core::OutputFormat,
    compression: Option<u8>,
    cancel: Arc<AtomicBool>,
    token: tokio_util::sync::CancellationToken,
    job_id: &str,
) -> Result<Vec<u8>, GenerateFailure> {
    generate_image_bytes_inner(
        pipeline,
        txt2img,
        format,
        compression,
        cancel,
        token,
        Some(job_id.to_owned()),
    )
    .await
}

async fn generate_image_bytes_inner(
    pipeline: std::sync::Arc<dyn honkbox_sd_core::SdPipeline>,
    txt2img: honkbox_sd_core::Txt2ImgRequest,
    format: honkbox_sd_core::OutputFormat,
    compression: Option<u8>,
    cancel: Arc<AtomicBool>,
    token: tokio_util::sync::CancellationToken,
    progress_job_id: Option<String>,
) -> Result<Vec<u8>, GenerateFailure> {
    let worker_cancel = Arc::clone(&cancel);
    let mut worker = tokio::task::spawn_blocking(move || {
        let to_failure = |err: honkbox_sd_core::SdError| match err {
            honkbox_sd_core::SdError::Cancelled => GenerateFailure::Cancelled,
            other => GenerateFailure::Failed(other.to_string()),
        };
        let img = match &progress_job_id {
            Some(job_id) => pipeline
                .generate_with_progress(&txt2img, &worker_cancel, &|step, total| {
                    crate::jobs_registry::global().emit_progress(job_id, step, total);
                })
                .map_err(to_failure)?,
            None => pipeline
                .generate(&txt2img, &worker_cancel)
                .map_err(to_failure)?,
        };
        match compression {
            Some(q) => honkbox_sd_core::encode_image_with_quality(img, format, q)
                .map_err(|err| GenerateFailure::Failed(err.to_string())),
            None => honkbox_sd_core::encode_image(img, format)
                .map_err(|err| GenerateFailure::Failed(err.to_string())),
        }
    });
    let join_failure = |err: tokio::task::JoinError| {
        GenerateFailure::Failed(format!("generation task failed: {err}"))
    };
    // Token → flag bridge: whichever lands first wins; a cancel makes the
    // worker return shortly (engine stop), and its answer — normally
    // `Cancelled` — decides the response.
    let outcome = tokio::select! {
        outcome = &mut worker => outcome.map_err(join_failure)?,
        () = token.cancelled() => {
            cancel.store(true, Ordering::Release);
            (&mut worker).await.map_err(join_failure)?
        }
    };
    // A backend that ignores the flag can still return Ok after the token
    // was marked. The token is the authority: never hand back a full image
    // for a cancelled request.
    if token.is_cancelled() {
        return Err(GenerateFailure::Cancelled);
    }
    outcome
}

/// Sidecar `scheduler` name for a resolved recipe: the fixed recipe's
/// discrete schedule keeps the historical `normal` recording name;
/// explicit requests record the sd.cpp wire spelling.
pub(super) fn image_scheduler_name(scheduler: honkbox_sd_core::Scheduler) -> String {
    if scheduler == honkbox_sd_core::Scheduler::Discrete {
        honkbox_sd_core::SCHEDULER.to_string()
    } else {
        scheduler.name().to_string()
    }
}

/// `POST /v1/images/generations` (REQ-10): the `OpenAI` image-generation
/// shape over the startup-configured [`SdPipeline`]. SD pipelines are
/// startup-static (no swap manager), so a request `model` is validated and
/// recorded (bundle sidecar) but the pipeline stays the one built at
/// startup — with the stub backend any slug is honest; with `sd-cpp` the
/// checkpoint is the `--sd-model` one.
///
/// `output_format` selects the container for the returned bytes (`png`
/// default, `jpeg`, `webp` lossless; unknown names are a 400 naming it) and
/// the bundle sidecar/extension agree with it. `output_compression` is the
/// JPEG quality `[0, 100]`; non-numeric/out-of-range values are a named 400,
/// and the knob is ignored (as documented) for lossless PNG/WebP.
///
/// Each image in the request is generated under the GPU lease
/// (`spawn_blocking`, like every other GPU route) and written to the
/// existing job bundle store (`<data_dir>/jobs/`), so CLI GC and
/// recordkeeping cover HTTP generations too. Errors are `OpenAI`
/// envelopes: 400 bad `size`/`response_format`/`output_format`/
/// `output_compression`/recipe params (unknown
/// recipe/sampler/scheduler name, `cfg_scale` outside [1.0, 30.0], `steps`
/// outside [1, 100], invalid `hires_fix` scale/target/denoise, `LoRA` slug
/// / strength / count violations), 404 unknown model, 429 busy, 500
/// backend failure. `n` clamps to `[1, 4]` (warn, never reject).
/// Per-request `loras` resolve against `ModelKind::Lora` catalog entries;
/// their license notes surface as response `warnings` (weights are never
/// bundled; ReviewedLicense-AGPL §2b/§2c). `recipe` selects a curated
/// sampler/scheduler/cfg/steps bundle (per-field overrides win, then
/// server defaults); `hires_fix` runs the vendor-native second-pass
/// upscale and is recorded in the bundle sidecar. A valid `controlnet` +
/// `control_image` pair (plus optional `controlnet_strength`) is applied by
/// the sd-cpp pipeline — the weights load with the model context and the
/// conditioning image steers the render — while backends without a
/// `ControlNet` branch get the named 501 instead.
// Route handlers read long but stay linear: parse → validate → resolve →
// generate → bundle. `completions_stream` carries the same allow.
#[allow(clippy::too_many_lines)]
#[utoipa::path(
    post,
    path = "/v1/images/generations",
    request_body = ImagesGenerationRequest,
    responses(
        (status = 200, description = "Generated images (base64-encoded bytes in the requested output_format)", body = ImagesGenerationResponse),
        (status = 400, description = "Bad size, response_format, output_format, output_compression, recipe, sampler, scheduler, cfg_scale, steps, hires_fix, loras, or controlnet params (pair, image, or slug)", body = OpenAiError),
        (status = 404, description = "Unknown model", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Generation failed", body = OpenAiError),
        (status = 501, description = "Valid controlnet params accepted but not applicable by the sd pipeline (controlnet_unsupported)", body = OpenAiError)
    )
)]
pub(super) async fn post_images_generations(
    State(config): State<AppConfig>,
    headers: HeaderMap,
    Json(req): Json<ImagesGenerationRequest>,
) -> Response {
    let (width, height, n, recipe, hires, format, compression, delivery) =
        match validate_images_request(
            &req,
            headers
                .get("X-Honkbox-Delivery")
                .map(|value| value.to_str().unwrap_or_default()),
        ) {
            Ok(valid) => valid,
            Err((message, code)) => {
                return super::openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
            }
        };
    let catalog = config.catalog_snapshot();
    // Canonical hash of the validated request under the serving catalog; a
    // cache hit replays the first response before admission, so identical
    // inline work consumes no slot. The catalog hash keeps a refresh (new
    // models/controlnet entries) from replaying a pre-refresh answer; backend
    // and model identity travel with the instance-scoped key as well.
    let cache = matches!(delivery, DeliveryMode::Inline).then(|| scoped_generation_cache(&config));
    let cache_key = cache.as_ref().map(|_| {
        format!(
            "{}:{}:{}:{}",
            config.sd.backend_id(),
            config.sd_model.as_deref().unwrap_or(""),
            canonical_hash(&serde_json::to_value(&*catalog).unwrap_or(serde_json::Value::Null)),
            canonical_hash(&serde_json::to_value(&req).unwrap_or(serde_json::Value::Null)),
        )
    });
    if let (Some(cache), Some(key)) = (cache.as_ref(), cache_key.as_deref())
        && let Some(hit) = cache.cache.get(key)
    {
        return axum::Json((*hit).clone()).into_response();
    }
    let (pipeline_loras, lora_records, lora_warnings) =
        match resolve_loras(req.loras.as_deref().unwrap_or_default(), &catalog) {
            Ok(resolved) => resolved,
            Err((message, code)) => {
                let status = if code.starts_with("model_license_") {
                    StatusCode::PRECONDITION_FAILED
                } else {
                    StatusCode::BAD_REQUEST
                };
                return super::openai_error_response(status, message, Some(code));
            }
        };
    // ControlNet (honkbox extension): slug + conditioning image are
    // validated here; a fully valid pair rides the generation request when
    // the configured pipeline can apply it, and pipelines that cannot are
    // answered with the honest 501 ([`resolve_controlnet`]) — never a
    // silent ignore that would render something else than what was asked
    // for.
    let (controlnet, controlnet_refusal) = resolve_controlnet(&req, &catalog, config.sd.as_ref());
    if let Some(response) = controlnet_refusal {
        return response;
    }
    let slug: String = match req
        .model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
    {
        Some(model) => match resolve_sd_model(model, &catalog) {
            Some(slug) => slug.to_owned(),
            None => {
                return super::openai_error_response(
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
        return super::openai_error_response(
            LicenseRefusal::status(),
            refusal.message(),
            Some(refusal.code()),
        );
    }
    let pipeline = Arc::clone(&config.sd);
    let prompt = req.prompt;
    let started = Instant::now();
    // NEW-3, engine level: the images job is registry-visible and
    // interruptible — the loop checks the cancel token BETWEEN images, and
    // the running sd.cpp generation itself is aborted mid-flight through
    // `sd_cancel_generation` (a per-image flag bridges the token into the
    // pipeline seam). Admission comes FIRST: a 429 has no registered job,
    // no lifecycle events and no X-Job-Id.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Sd).await,
        crate::jobs_registry::JobKind::Images,
        Some(slug.clone()),
        true,
    ) {
        Ok(pair) => pair,
        Err((status, message)) => return super::openai_error_response(status, message, None),
    };
    let mut images = Vec::with_capacity(n as usize);
    let mut artifacts: Vec<ImageArtifact> = Vec::with_capacity(n as usize);
    let mut bundle: Option<String> = None;
    let backend_id = pipeline.backend_id();
    // Provenance (TASK-determinism-contract): the verified digest sidecar
    // exists only when the model arrived as a checksum-verified download;
    // locally-placed or unverified weights stay null — never a fabricated
    // hash. The vendored engine identity maps from the backend id (None
    // for the stub, which has no separate engine).
    let model_sha256 = resolve_model_entry(&slug, &catalog)
        .and_then(|entry| crate::downloads::artifact_sha256(std::path::Path::new(&entry.path)));
    let engine_version = honkbox_sd_core::engine_version_for(backend_id);
    // The token is the cross-image cancel state; the engine-stop flag is
    // fresh per image (created inside the loop). Reusing one flag would
    // let a cancel of image 1 be inherited by image 2 before it starts.
    let job_token = job.token().clone();
    for i in 0..u64::from(n) {
        // Cooperative cancellation (NEW-3): the token is checked BETWEEN
        // images here — a cancel marked while an earlier image rendered is
        // refused before this one starts — and, through this image's fresh
        // flag, DURING an image: the engine aborts its sampling loop and
        // the worker below answers the same named 409.
        if job.token().is_cancelled() {
            return with_job_id(
                super::openai_error_response(
                    StatusCode::CONFLICT,
                    image_cancel_before_start(job.id(), i + 1, u64::from(n), i),
                    Some("job_cancelled"),
                ),
                job.id(),
            );
        }
        // Fresh engine-stop flag for THIS image only; the token above is
        // the state shared across the batch.
        let cancel = new_image_cancel_flag();
        // Per-image seeds keep n>1 outputs distinct while staying
        // deterministic per request (explicit seed base when given).
        let seed = recipe.seed_base.wrapping_add(i);
        let txt2img = honkbox_sd_core::Txt2ImgRequest {
            seed,
            steps: recipe.steps,
            prompt: prompt.clone(),
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
            init_image: None,
            mask_image: None,
            strength: None,
            hires: hires.clone(),
            control_net: controlnet.as_ref().map(|c| c.weights.clone()),
            control_image: controlnet.as_ref().map(|c| c.image.clone()),
            controlnet_strength: controlnet.as_ref().and_then(|c| c.strength),
            control_type: controlnet.as_ref().and_then(|c| c.control_type),
        };
        let generated = match delivery {
            DeliveryMode::Inline => {
                generate_image_bytes(
                    Arc::clone(&pipeline),
                    txt2img,
                    format,
                    compression,
                    Arc::clone(&cancel),
                    job_token.clone(),
                )
                .await
            }
            DeliveryMode::Artifact => {
                generate_image_bytes_with_progress(
                    Arc::clone(&pipeline),
                    txt2img,
                    format,
                    compression,
                    Arc::clone(&cancel),
                    job_token.clone(),
                    job.id(),
                )
                .await
            }
        };
        let bytes = match generated {
            Ok(bytes) => bytes,
            Err(GenerateFailure::Cancelled) => {
                // The in-flight render was aborted mid-sampling: answer the
                // named 409, honestly counting the interrupted image as NOT
                // generated. A cancel that landed before this image started
                // takes the pre-start branch above instead.
                return with_job_id(
                    super::openai_error_response(
                        StatusCode::CONFLICT,
                        image_cancel_mid_generation(job.id(), i + 1, u64::from(n), i),
                        Some("job_cancelled"),
                    ),
                    job.id(),
                );
            }
            Err(GenerateFailure::Failed(message)) => {
                return super::openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    message,
                    Some("generation_failed"),
                );
            }
        };
        let annotations = match super::nsfw::classify_for_image(&bytes, format.extension()) {
            Ok(annotations) => annotations,
            Err((status, message, code)) => {
                return with_job_id(
                    super::openai_error_response(status, message, Some(code)),
                    job.id(),
                );
            }
        };
        let bundle_annotations = annotations
            .iter()
            .map(super::nsfw::bundle_annotation)
            .collect();
        let bundle_inputs = crate::jobs::BundleInputs {
            model_slug: slug.clone(),
            prompt: prompt.clone(),
            seed,
            steps: recipe.steps,
            width,
            height,
            format,
            backend_id,
            model_sha256: model_sha256.clone(),
            engine_version,
            sampler: recipe.sampler.name().to_string(),
            scheduler: image_scheduler_name(recipe.scheduler),
            cfg_scale: recipe.cfg_scale,
            negative_prompt: recipe.negative_prompt.clone(),
            loras: lora_records.clone(),
            strength: None,
            mask_mode: None,
            mask_sha256: None,
            recipe: recipe.name.map(str::to_owned),
            hires: hires.as_ref().map(|fix| crate::jobs::HiresRecord {
                upscaler: "latent",
                target: {
                    let (tw, th) = honkbox_sd_core::hires_target((width, height), fix);
                    format!("{tw}x{th}")
                },
                scale: fix.scale,
                denoising_strength: fix.denoising_strength,
            }),
            annotations: bundle_annotations,
        };
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        match delivery {
            DeliveryMode::Inline => {
                write_generation_bundle(&config, &bundle_inputs, &bytes, elapsed_ms);
            }
            DeliveryMode::Artifact => {
                let dir = match write_generation_bundle_strict(
                    &config,
                    &bundle_inputs,
                    &bytes,
                    elapsed_ms,
                ) {
                    Ok(dir) => dir,
                    Err(message) => {
                        return with_job_id(
                            super::openai_error_response(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                message,
                                Some("bundle_write_failed"),
                            ),
                            job.id(),
                        );
                    }
                };
                // One bundle dir per image; `bundle` names the first so the
                // field is a single stable handle for the request.
                if bundle.is_none() {
                    bundle = Some(dir);
                }
                let info = match config.artifacts.put_bytes(&bytes, format.mime()) {
                    Ok(info) => info,
                    Err(err) => {
                        return with_job_id(
                            super::openai_error_response(
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
                        index: u32::try_from(i).unwrap_or(u32::MAX),
                        primary: i == 0,
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
            }
        }
        if delivery == DeliveryMode::Inline {
            images.push(ImageB64 {
                b64_json: base64::engine::general_purpose::STANDARD.encode(&bytes),
            });
        }
    }
    drop(lease);
    let response = ImagesGenerationResponse {
        created: epoch_secs(),
        data: images,
        warnings: (!lora_warnings.is_empty()).then_some(lora_warnings),
        artifacts: (!artifacts.is_empty()).then_some(artifacts),
        bundle,
        job_id: (delivery == DeliveryMode::Artifact).then(|| job.id().to_owned()),
    };
    if let (Some(cache), Some(key), Ok(value)) = (
        cache.as_ref(),
        cache_key.as_deref(),
        serde_json::to_value(&response),
    ) {
        cache.cache.put(key.to_owned(), value);
    }
    with_job_id(Json(response).into_response(), job.id())
}
