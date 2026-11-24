//! `POST /v1/ocr/ingest`: path-free, cache-keyed, cancellable OCR jobs.
//!
//! Input is one `data:` URI (PDF or image); nothing is ever fetched from a
//! path or URL. The route owns input resolution and admission, then hands
//! the bytes to the shared in-process recognition pipeline. PDF text layers
//! are extracted with `pdf-extract` (MIT), image inputs run `ocrs` CPU OCR.
//! PDF modes that need rasterization answer the named 501
//! `ocr_pdf_raster_unavailable` — this build has no pure-Rust MIT/Apache
//! rasterizer and never substitutes a stub.
//!
//! `ocrs` 0.14 exposes word boxes but no confidence score, so every region
//! carries `confidence: None` and the engine metadata says
//! `confidence_support: false`. Null is honest, never an invented score.

use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use base64::Engine as _;
use honkbox_api_types::{
    OcrEngineMeta, OcrIngestMode, OcrIngestRequest, OcrJobAccepted, OcrPage, OcrRegion, OcrResult,
};
use honkbox_recognition_worker::{
    OCR_CACHE_ENGINE, OcrCacheKey, RecKind as WorkerRecKind, RecognitionPipeline,
    RecognizeRequest as WorkerRecognizeRequest,
};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use super::chat_prep::{admit_then_register, with_job_id};
use super::request_cache::RequestCache;
use super::{AppConfig, openai_error_response};

/// Decoded source-byte ceiling. Larger inputs get a named 413 before any
/// extraction begins (the router-wide 64 MiB body cap is a different,
/// transport-level limit).
pub(crate) const OCR_MAX_INPUT_BYTES: usize = 16 * 1024 * 1024;

/// Accepted DPI range. `ocrs` reads native pixels and the PDF text layer
/// needs no rendering, but a bounded explicit value is part of the cache
/// key so a later rasterizer can honor it without a wire-contract change.
const OCR_DPI_RANGE: std::ops::RangeInclusive<u32> = 72..=600;

/// Bounded result cache: 64 entries, five-minute TTL. A hit replays the
/// stored result with `cache_hit = true` and takes NO admission lease.
const OCR_CACHE_MAX_ENTRIES: usize = 64;
const OCR_CACHE_TTL: Duration = Duration::from_mins(5);
static OCR_CACHE: LazyLock<RequestCache<OcrResult>> =
    LazyLock::new(|| RequestCache::new(OCR_CACHE_MAX_ENTRIES, OCR_CACHE_TTL));

/// What the data URI carried; PDF text layer and image OCR are distinct
/// execution paths, and only the latter can raster-need modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputKind {
    Pdf,
    Image,
}

/// Engine/model version from D1's dependency pins, never a local copy of a
/// version string. Without the feature the engine cannot serve OCR at all
/// (the route's `serves(Ocr)` gate answers 501 first).
#[cfg(feature = "ocr-ocrs")]
fn ocr_model_version() -> String {
    format!(
        "{}+{}",
        honkbox_recognition_worker::OCRS_VERSION,
        honkbox_recognition_worker::RTEN_VERSION
    )
}

#[cfg(not(feature = "ocr-ocrs"))]
fn ocr_model_version() -> String {
    "unavailable".to_owned()
}

/// Dependency version for an engine meta row. Only the real `ocrs-rten`
/// backend gets D1's version pin; an injected/unknown backend reports none.
fn engine_version(backend: &str) -> Option<String> {
    #[cfg(feature = "ocr-ocrs")]
    {
        (backend == honkbox_recognition_worker::OCRS_BACKEND_ID).then(ocr_model_version)
    }
    #[cfg(not(feature = "ocr-ocrs"))]
    {
        let _ = backend;
        None
    }
}

/// `POST /v1/ocr/ingest`: admit one OCR job and answer `202` with its id.
/// The result is committed atomically into the job bundle as
/// `ocr_result.json` and replayable from the bounded cache.
#[allow(clippy::too_many_lines)] // one validation gate per named error code, by design
#[utoipa::path(
    post,
    path = "/v1/ocr/ingest",
    request_body = OcrIngestRequest,
    responses(
        (status = 202, description = "Job accepted; result lands in the job bundle", body = OcrJobAccepted),
        (status = 400, description = "Malformed request, host path/non-data URI, bad base64, unsupported container, or unbounded DPI", body = honkbox_api_types::OpenAiError),
        (status = 413, description = "Decoded input exceeds the OCR byte cap", body = honkbox_api_types::OpenAiError),
        (status = 429, description = "OCR admission slot busy (`ocr_busy`)", body = honkbox_api_types::OpenAiError),
        (status = 501, description = "No OCR weights/backend (`ocr_engine_unavailable`), or PDF raster mode (`ocr_pdf_raster_unavailable`)", body = honkbox_api_types::OpenAiError),
        (status = 503, description = "No data dir for the job bundle (`ocr_storage_unavailable`)", body = honkbox_api_types::OpenAiError)
    )
)]
pub(super) async fn post_ocr_ingest(
    State(config): State<AppConfig>,
    body: Result<Json<OcrIngestRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(req)) => req,
        Err(rejection) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid OCR ingest request: {}", rejection.body_text()),
                Some("invalid_ocr_ingest_request"),
            );
        }
    };
    // One honesty gate for every mode: no compiled backend / no user weights
    // means no ingest, including the text-layer path. The same `serves(Ocr)`
    // drives `capabilities.recognition.ocr` and `/v1/recognize`.
    if !config.recognition.serves(WorkerRecKind::Ocr) {
        return openai_error_response(
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "OCR engine {:?} is unavailable (no backend or weights); see GET /v1/capabilities recognition",
                config.recognition.backend_id_for(WorkerRecKind::Ocr)
            ),
            Some("ocr_engine_unavailable"),
        );
    }
    let (bytes, kind) = match decode_ocr_input(&req.data_uri) {
        Ok(input) => input,
        Err((status, code, message)) => return openai_error_response(status, message, Some(code)),
    };
    if req.dpi.is_some_and(|dpi| !OCR_DPI_RANGE.contains(&dpi)) {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "dpi must be within {}..={}",
                OCR_DPI_RANGE.start(),
                OCR_DPI_RANGE.end()
            ),
            Some("ocr_invalid_dpi"),
        );
    }
    if kind == InputKind::Pdf && req.mode.needs_raster() {
        // Named 501, never a stub: no pure-Rust MIT/Apache rasterizer exists
        // in this build, so pretending to layout/render would be a lie.
        return openai_error_response(
            StatusCode::NOT_IMPLEMENTED,
            "PDF raster modes are unavailable in this build; use text_layer or replace_text_layer (or send an image)".to_owned(),
            Some("ocr_pdf_raster_unavailable"),
        );
    }
    let cache_key = OcrCacheKey {
        source_sha256: hex_sha256(&bytes),
        engine: OCR_CACHE_ENGINE,
        model_version: ocr_model_version(),
        language: req.language.clone(),
        dpi: req.dpi,
        mode: Some(req.mode.as_str().to_owned()),
    }
    .as_key();
    let cached = OCR_CACHE.get(&cache_key);
    let Some(data_dir) = config.data_dir.clone() else {
        return openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "no data dir: OCR job bundles are disabled".to_owned(),
            Some("ocr_storage_unavailable"),
        );
    };
    let pipeline = Arc::clone(&config.recognition);
    // Admission first, registration second. A cache hit is a pure replay:
    // it registers a job for observability but takes NO GPU/CPU lease.
    let (lease, guard) = if cached.is_some() {
        (
            None,
            crate::jobs_registry::global().register(crate::jobs_registry::JobKind::Ocr, None, true),
        )
    } else {
        match admit_then_register(
            crate::jobs_registry::global(),
            crate::admission::acquire(honkbox_residency::ResourceKind::Ocr).await,
            crate::jobs_registry::JobKind::Ocr,
            None,
            true,
        ) {
            Ok((lease, guard)) => (Some(lease), guard),
            Err((status, message)) => {
                return openai_error_response(status, message, Some("ocr_busy"));
            }
        }
    };
    let job_id = guard.id().to_owned();
    let token = guard.token().clone();
    let mode = req.mode;
    let language = req.language;
    tokio::task::spawn_blocking(move || {
        let _lease = lease;
        let _guard = guard;
        let started = Instant::now();
        let (result, successful) = match cached {
            Some(cached) => {
                let mut replay = (*cached).clone();
                replay.cache_hit = true;
                (replay, true)
            }
            None => run_job(&bytes, kind, mode, language.as_deref(), &pipeline, &token),
        };
        if successful && !result.cancelled {
            OCR_CACHE.put(cache_key, result.clone());
        }
        commit(&data_dir, &result, &pipeline, started);
    });
    with_job_id(
        (
            StatusCode::ACCEPTED,
            Json(OcrJobAccepted {
                job_id: job_id.clone(),
                status: "queued".to_owned(),
            }),
        )
            .into_response(),
        &job_id,
    )
}

/// Decode the path-free `data:` URI. Only an inline PDF or an image the
/// format registry knows is accepted; no fetch, no filesystem resolution.
fn decode_ocr_input(uri: &str) -> Result<(Vec<u8>, InputKind), (StatusCode, &'static str, String)> {
    let invalid = |message: String| (StatusCode::BAD_REQUEST, "invalid_ocr_input", message);
    let trimmed = uri.trim();
    let Some(rest) = trimmed.strip_prefix("data:") else {
        return Err(invalid(
            "OCR input must be a data URI (`data:application/pdf;base64,…` or an image data URI); paths and remote URLs are never fetched".to_owned(),
        ));
    };
    let Some((header, payload)) = rest.split_once(',') else {
        return Err(invalid(
            "data URI is missing the `,<base64 payload>` part".to_owned(),
        ));
    };
    let Some(mime) = header.strip_suffix(";base64") else {
        return Err(invalid("data URI must declare `;base64`".to_owned()));
    };
    let kind = if mime.eq_ignore_ascii_case("application/pdf") {
        InputKind::Pdf
    } else if crate::formats::image_ops_by_mime(mime).is_some() {
        InputKind::Image
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            "unsupported_ocr_container",
            format!(
                "unsupported OCR container {mime:?}: expected application/pdf or a supported image type"
            ),
        ));
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(payload.trim()))
        .map_err(|e| invalid(format!("data URI is not valid base64: {e}")))?;
    if bytes.is_empty() {
        return Err(invalid("OCR input payload is empty".to_owned()));
    }
    if bytes.len() > OCR_MAX_INPUT_BYTES {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "ocr_input_too_large",
            format!(
                "OCR input is {} bytes; the cap is {OCR_MAX_INPUT_BYTES}",
                bytes.len()
            ),
        ));
    }
    Ok((bytes, kind))
}

/// Lowercase hex sha256 of the source bytes: the first field of
/// [`OcrCacheKey`]. No path, name, or payload enters the cache key.
fn hex_sha256(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// One job's work. `successful` is false only for an engine/extraction
/// failure — a cancelled or empty result is a valid job outcome and must
/// still be committed to the bundle.
fn run_job(
    bytes: &[u8],
    kind: InputKind,
    mode: OcrIngestMode,
    language: Option<&str>,
    pipeline: &Arc<dyn RecognitionPipeline>,
    token: &CancellationToken,
) -> (OcrResult, bool) {
    let (pages, engine, cancelled, successful) = match kind {
        InputKind::Pdf => {
            let engine = OcrEngineMeta {
                id: pipeline.backend_id().to_owned(),
                version: engine_version(pipeline.backend_id()),
                bbox_support: false,
                confidence_support: false,
            };
            match pdf_extract::extract_text_from_mem_by_pages(bytes) {
                Ok(texts) => {
                    let (pages, cancelled) = text_pages(texts, |_index| token.is_cancelled());
                    (pages, engine, cancelled, true)
                }
                Err(_err) => {
                    tracing::warn!(kind = "ocr", "PDF text layer extraction failed");
                    (Vec::new(), engine, token.is_cancelled(), false)
                }
            }
        }
        InputKind::Image => {
            if token.is_cancelled() {
                return (empty_result(pipeline, mode, true), true);
            }
            let request = WorkerRecognizeRequest {
                kind: WorkerRecKind::Ocr,
                image: bytes.to_vec(),
                language: language.map(str::to_owned),
                mode: Some(mode.as_str().to_owned()),
                engine: None,
            };
            match pipeline.recognize(&request) {
                Ok(out) => {
                    let engine = OcrEngineMeta {
                        id: out.engine.id.to_owned(),
                        version: engine_version(out.engine.id),
                        bbox_support: out.engine.bbox_support,
                        confidence_support: false,
                    };
                    let cancelled = token.is_cancelled();
                    let pages = out
                        .pages
                        .into_iter()
                        .enumerate()
                        .map(|(index, page)| OcrPage {
                            index: u32::try_from(index).unwrap_or(u32::MAX),
                            text: page.text,
                            regions: page
                                .regions
                                .into_iter()
                                .map(|region| OcrRegion {
                                    bbox: Some(region.bbox),
                                    text: region.text,
                                    // ocrs 0.14 has no score: never invent one.
                                    confidence: region.confidence,
                                })
                                .collect(),
                        })
                        .collect();
                    (pages, engine, cancelled, true)
                }
                Err(err) => {
                    tracing::warn!(kind = "ocr", "image OCR engine failed: {err}");
                    (
                        Vec::new(),
                        engine_meta(pipeline),
                        token.is_cancelled(),
                        false,
                    )
                }
            }
        }
    };
    (
        OcrResult {
            engine,
            mode,
            page_count: u32::try_from(pages.len()).unwrap_or(u32::MAX),
            pages,
            cache_hit: false,
            cancelled,
        },
        successful,
    )
}

/// Engine metadata before any result exists (error/cancel shortcut).
fn engine_meta(pipeline: &Arc<dyn RecognitionPipeline>) -> OcrEngineMeta {
    OcrEngineMeta {
        id: pipeline.backend_id().to_owned(),
        version: engine_version(pipeline.backend_id()),
        bbox_support: false,
        confidence_support: false,
    }
}

fn empty_result(
    pipeline: &Arc<dyn RecognitionPipeline>,
    mode: OcrIngestMode,
    cancelled: bool,
) -> OcrResult {
    OcrResult {
        engine: engine_meta(pipeline),
        mode,
        page_count: 0,
        pages: Vec::new(),
        cache_hit: false,
        cancelled,
    }
}

/// Deterministic line grouping for a PDF text layer: one region per
/// non-empty line in source order, no boxes and no confidence. The `stop`
/// predicate is checked BEFORE each page, so cancellation halts mid-document
/// and the result reports exactly how many pages were processed.
pub(crate) fn text_pages<F: Fn(usize) -> bool>(
    texts: Vec<String>,
    stop: F,
) -> (Vec<OcrPage>, bool) {
    let mut pages = Vec::with_capacity(texts.len());
    for (index, text) in texts.into_iter().enumerate() {
        if stop(index) {
            return (pages, true);
        }
        let lines: Vec<String> = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect();
        let page_text = lines.join("\n");
        pages.push(OcrPage {
            index: u32::try_from(index).unwrap_or(u32::MAX),
            text: page_text,
            regions: lines
                .into_iter()
                .map(|line| OcrRegion {
                    bbox: None,
                    text: line,
                    confidence: None,
                })
                .collect(),
        });
    }
    (pages, false)
}

/// Commit the result and the canonical sidecars. `write_bundle` owns all
/// atomic writes (`<name>.part` + fsync + rename); the OCR payload itself
/// is a separate atomic file in the same job directory.
fn commit(
    data_dir: &std::path::Path,
    result: &OcrResult,
    pipeline: &Arc<dyn RecognitionPipeline>,
    started: Instant,
) {
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let inputs = crate::jobs::BundleInputs {
        model_slug: result.engine.id.clone(),
        prompt: String::new(),
        seed: 0,
        steps: 0,
        width: 0,
        height: 0,
        format: honkbox_sd_core::OutputFormat::Png,
        backend_id: pipeline.backend_id(),
        model_sha256: None,
        engine_version: None,
        sampler: String::new(),
        scheduler: String::new(),
        cfg_scale: 0.0,
        negative_prompt: None,
        loras: Vec::new(),
        strength: None,
        mask_mode: None,
        mask_sha256: None,
        checkpoint_base_model: None,
        recipe: Some(result.mode.as_str().to_owned()),
        hires: None,
        annotations: Vec::new(),
    };
    let payload = serde_json::to_vec_pretty(result);
    match crate::jobs::write_bundle(data_dir, &inputs, &[], elapsed_ms) {
        Ok(dir) => match payload {
            Ok(bytes) => {
                if crate::jobs::write_atomic(&dir, "ocr_result.json", &bytes).is_err() {
                    tracing::warn!(kind = "ocr", "cannot write OCR result bundle file");
                }
            }
            Err(err) => tracing::warn!(kind = "ocr", "cannot encode OCR result: {err}"),
        },
        Err(err) => tracing::warn!(kind = "ocr", "cannot write OCR job bundle: {err}"),
    }
}
