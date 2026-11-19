//! Audio upload routes, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only the handlers and their helpers
//! live here.

use std::sync::Arc;

use axum::{
    Json,
    body::Bytes,
    extract::{Query, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use honkbox_api_types::{SynthesizeRequest, Transcript, TranscriptSegment};
use honkbox_audio_worker::{
    SttPipeline, SynthesizeRequest as WorkerSynthRequest, SynthesizedAudio,
    TranscribeRequest as WorkerTranscribeRequest, Transcript as WorkerTranscript, TtsPipeline,
    decode_wav, encode_wav,
};

use super::chat_prep::{TranscribeQuery, clamp_ttl_seconds, with_job_id};
// The multipart helpers now live in `audio_multipart`; re-exported so the
// sibling `audio_transcriptions` routes keep a single import path.

use super::{AppConfig, SPEECH_INPUT_MAX_CHARS, validate_model};

/// `POST /v1/transcribe`: content-type dispatch (`audio/wav` → `decode_wav`,
/// `audio/pcm-s16le` → raw LE bytes + required `?sample_rate=`), then the
/// sync worker call inside `spawn_blocking`.
#[utoipa::path(
    post,
    path = "/v1/transcribe",
    responses(
        (status = 200, description = "Transcription", body = Transcript),
        (status = 400, description = "Unknown model, bad audio, or missing sample_rate"),
        (status = 415, description = "Use audio/wav or audio/pcm-s16le"),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Transcription failed")
    )
)]
pub(super) async fn post_transcribe(
    State(config): State<AppConfig>,
    Query(query): Query<TranscribeQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    let catalog = config.catalog_snapshot();
    validate_model(
        query.model.as_deref(),
        config.stt_model.as_deref(),
        &catalog,
    )
    .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    let _ttl = clamp_ttl_seconds(config.swap_max_ttl, query.ttl_seconds);
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let (sample_rate, pcm): (u32, Vec<i16>) = match content_type.as_str() {
        "audio/wav" | "audio/wave" | "audio/x-wav" => decode_wav(&body)
            .map_err(|err| (StatusCode::BAD_REQUEST, format!("invalid wav: {err}")))?,
        "audio/pcm-s16le" => {
            let sample_rate = query.sample_rate.ok_or((
                StatusCode::BAD_REQUEST,
                "audio/pcm-s16le requires ?sample_rate=".to_owned(),
            ))?;
            if !body.len().is_multiple_of(2) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "odd audio/pcm-s16le length".to_owned(),
                ));
            }
            let (pairs, _) = body.as_chunks::<2>();
            let pcm = pairs.iter().map(|pair| i16::from_le_bytes(*pair)).collect();
            (sample_rate, pcm)
        }
        other => {
            return Err((
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                format!("unsupported content-type {other:?}; use audio/wav or audio/pcm-s16le"),
            ));
        }
    };
    let worker_req = WorkerTranscribeRequest {
        model: query.model.unwrap_or_default(),
        language: query.language,
        translate: false,
    };
    // NEW-3: registry-visible while the request lives (never interruptible:
    // one blocking FFI call; the token is only marked by a cancel).
    let job = crate::jobs_registry::global().register(
        crate::jobs_registry::JobKind::Transcribe,
        (!worker_req.model.is_empty()).then_some(worker_req.model.clone()),
        false,
    );
    let out = run_transcription(config.stt.clone(), worker_req, sample_rate, pcm).await?;
    Ok(with_job_id(
        Json(Transcript {
            backend_id: out.backend_id,
            text: out.text,
            segments: out
                .segments
                .into_iter()
                .map(|segment| TranscriptSegment {
                    start_ms: segment.start_ms,
                    end_ms: segment.end_ms,
                    text: segment.text,
                })
                .collect(),
            millis: out.millis,
        })
        .into_response(),
        job.id(),
    ))
}

/// `POST /v1/synthesize`: JSON in, WAV bytes out (`audio/wav`).
#[utoipa::path(
    post,
    path = "/v1/synthesize",
    request_body = SynthesizeRequest,
    responses(
        (status = 200, description = "WAV audio"),
        (status = 400, description = "Unknown model slug; blank `text`; `text` over the 4096-character cap (`text_too_long`)"),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Synthesis failed")
    )
)]
pub(super) async fn post_synthesize(
    State(config): State<AppConfig>,
    Json(req): Json<SynthesizeRequest>,
) -> Result<Response, (StatusCode, String)> {
    // Same boundary as `/v1/audio/speech` (TASK-speech-input-cap): this is
    // the native twin of the OpenAI-shaped route — same worker, same
    // unbounded-PCM risk; reject-not-clamp (no silent truncation).
    if req.text.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "text is required: the text to synthesize".to_owned(),
        ));
    }
    let text_chars = req.text.chars().count();
    if text_chars > SPEECH_INPUT_MAX_CHARS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "text is {text_chars} characters; the cap is {SPEECH_INPUT_MAX_CHARS} — rejected, not truncated"
            ),
        ));
    }
    let catalog = config.catalog_snapshot();
    validate_model(Some(&req.model), config.tts_model.as_deref(), &catalog)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    let _ttl = clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds);
    let worker_req = WorkerSynthRequest {
        model: req.model.clone(),
        text: req.text,
        voice: req.voice,
        sample_rate: req.sample_rate,
    };
    // NEW-3: registry-visible while the request lives (one blocking FFI
    // call — the cancel token is marked but never checked).
    let job = crate::jobs_registry::global().register(
        crate::jobs_registry::JobKind::Synthesize,
        (!worker_req.model.is_empty()).then_some(worker_req.model.clone()),
        false,
    );
    let audio = run_synthesis(config.tts.clone(), worker_req).await?;
    let wav = encode_wav(audio.sample_rate, &audio.pcm16);
    Ok(with_job_id(
        ([(header::CONTENT_TYPE, "audio/wav")], wav).into_response(),
        job.id(),
    ))
}

/// Shared STT execution (native `/v1/transcribe` and REQ-9
/// `/v1/audio/transcriptions` are the same worker call under two wire
/// shapes): run the pipeline inside the GPU slot on the blocking pool. The
/// slot comes through the admission queue (REQ-6 Option A); the caller
/// registered the job (NEW-3) — one blocking FFI call, never interruptible.
/// Audio pipelines share the `llm` admission bucket: the fixed ledger kind
/// vocabulary has no audio kind, and `llm` is the warm-model-pipeline
/// bucket the audio workers already load alongside (the label is admission
/// topology, not model taxonomy).
pub(super) async fn run_transcription(
    pipe: Arc<dyn SttPipeline>,
    worker_req: WorkerTranscribeRequest,
    sample_rate: u32,
    pcm: Vec<i16>,
) -> Result<WorkerTranscript, (StatusCode, String)> {
    let lease = crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await?;
    tokio::task::spawn_blocking(move || {
        let _lease = lease;
        pipe.transcribe(&pcm, sample_rate, &worker_req)
            .map_err(|err| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("transcribe failed: {err}"),
                )
            })
    })
    .await
    .map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("transcribe task failed: {err}"),
        )
    })?
}

/// Shared TTS execution (native `/v1/synthesize` and REQ-9
/// `/v1/audio/speech` are the same worker call under two wire shapes).
/// Audio pipelines share the `llm` admission bucket (see
/// [`run_transcription`] for why).
pub(super) async fn run_synthesis(
    pipe: Arc<dyn TtsPipeline>,
    worker_req: WorkerSynthRequest,
) -> Result<SynthesizedAudio, (StatusCode, String)> {
    let lease = crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await?;
    tokio::task::spawn_blocking(move || {
        let _lease = lease;
        pipe.synthesize(&worker_req).map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("synthesize failed: {err}"),
            )
        })
    })
    .await
    .map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("synthesize task failed: {err}"),
        )
    })?
}
