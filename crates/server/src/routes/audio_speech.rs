//! `POST /v1/audio/speech` — the `OpenAI`-shaped TTS route, extracted from
//! `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage
//! gate scans that file for `.route()` calls. Only the handler lives here.

use axum::{
    Json,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use honkbox_api_types::{OpenAiError, SpeechRequest};
use honkbox_audio_worker::{SynthesizeRequest as WorkerSynthRequest, encode_wav};

use super::audio_upload::run_synthesis;
use super::chat_prep::with_job_id;
use super::openai_constraints::openai_error_response;
use super::{AppConfig, SPEECH_INPUT_MAX_CHARS, validate_model};

/// `POST /v1/audio/speech` (REQ-9): `OpenAI`-shaped TTS. Same worker as
/// `/v1/synthesize`. `model` optional (server default TTS), `voice` passed
/// through; `response_format` is `"wav"` (default, RIFF/WAVE container) or
/// `"pcm"` — raw little-endian signed 16-bit mono samples with **no**
/// container: the worker already returns mono PCM16, so the pcm path
/// serializes the exact same samples minus the 44-byte RIFF header (`OpenAI`
/// pcm semantics). `response_format` accepts `wav` (default) or `pcm`.
#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    request_body = SpeechRequest,
    responses(
        (status = 200, description = "WAV audio (RIFF container)", content_type = "audio/wav"),
        (status = 200, description = "Raw s16le mono PCM (pcm response_format; no container)", content_type = "audio/pcm-s16le"),
        (status = 400, description = "Empty input, oversized input (over 4096 characters), unknown model, or unsupported response_format", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Synthesis failed", body = OpenAiError)
    )
)]
pub(super) async fn post_audio_speech(
    State(config): State<AppConfig>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    let pcm_out = match req.response_format.as_deref() {
        None | Some("wav") => false,
        Some("pcm") => true,
        Some(other) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "unsupported response_format {other:?}: only \"wav\" and \"pcm\" are produced"
                ),
                None,
            );
        }
    };
    if req.input.trim().is_empty() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "input is required: the text to synthesize".to_owned(),
            None,
        );
    }
    // TASK-speech-input-cap: unbounded `input` means unbounded PCM growth;
    // reject-not-clamp (no silent truncation, no derived clamp).
    let input_chars = req.input.chars().count();
    if input_chars > SPEECH_INPUT_MAX_CHARS {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "input is {input_chars} characters; the cap is {SPEECH_INPUT_MAX_CHARS} — rejected, not truncated"
            ),
            Some("input_too_long"),
        );
    }
    let catalog = config.catalog_snapshot();
    if let Err(message) =
        validate_model(req.model.as_deref(), config.tts_model.as_deref(), &catalog)
    {
        return openai_error_response(StatusCode::BAD_REQUEST, message, None);
    }
    let worker_req = WorkerSynthRequest {
        model: req.model.unwrap_or_default(),
        text: req.input,
        voice: req.voice,
        sample_rate: None,
    };
    // NEW-3: registry-visible while the request lives (one blocking FFI
    // call — the cancel token is marked but never checked).
    let job = crate::jobs_registry::global().register(
        crate::jobs_registry::JobKind::Synthesize,
        (!worker_req.model.is_empty()).then_some(worker_req.model.clone()),
        false,
    );
    let audio = match run_synthesis(config.tts.clone(), worker_req).await {
        Ok(audio) => audio,
        Err((StatusCode::TOO_MANY_REQUESTS, message)) => {
            return with_job_id(
                openai_error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    message,
                    Some("gpu_slot_busy"),
                ),
                job.id(),
            );
        }
        Err((code, message)) => return openai_error_response(code, message, None),
    };
    if pcm_out {
        // Raw s16le: the exact samples the wav path wraps, header omitted.
        let mut raw = Vec::with_capacity(audio.pcm16.len() * 2);
        for sample in &audio.pcm16 {
            raw.extend_from_slice(&sample.to_le_bytes());
        }
        return with_job_id(
            ([(header::CONTENT_TYPE, "audio/pcm-s16le")], raw).into_response(),
            job.id(),
        );
    }
    let wav = encode_wav(audio.sample_rate, &audio.pcm16);
    with_job_id(
        ([(header::CONTENT_TYPE, "audio/wav")], wav).into_response(),
        job.id(),
    )
}
