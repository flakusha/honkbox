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
use honkbox_audio_worker::{
    SynthesizeRequest as WorkerSynthRequest, encode_flac, encode_opus, encode_wav,
};

use super::audio_upload::run_synthesis;
use super::chat_prep::{admit_then_register, with_job_id};
use super::openai_constraints::openai_error_response;
use super::{AppConfig, SPEECH_INPUT_MAX_CHARS, validate_model};

/// Output container for `POST /v1/audio/speech`. Every variant encodes the
/// same worker PCM16 — the choice is container only, never a re-synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpeechFormat {
    /// RIFF/WAVE container (default).
    Wav,
    /// Raw little-endian s16 mono, no container (`OpenAI` pcm semantics).
    Pcm,
    /// Lossless FLAC (pure-Rust encoder, any worker rate passes through).
    Flac,
    /// Ogg-encapsulated Opus, RFC 7845 gapless (Opus-native worker rates).
    Opus,
}

/// Resolves `response_format` and validates `input` caps; `Err` is the
/// `(status, message, code)` triple for an OpenAI-style error response.
type SpeechRequestError = (StatusCode, String, Option<&'static str>);

fn parse_speech_request(req: &SpeechRequest) -> Result<SpeechFormat, SpeechRequestError> {
    let format = match req.response_format.as_deref() {
        None | Some("wav") => SpeechFormat::Wav,
        Some("pcm") => SpeechFormat::Pcm,
        Some("flac") => SpeechFormat::Flac,
        Some("opus") => SpeechFormat::Opus,
        Some(other) => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "unsupported response_format {other:?}: only \"wav\", \"pcm\", \"flac\" and \"opus\" are produced"
                ),
                None,
            ));
        }
    };
    if req.input.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "input is required: the text to synthesize".to_owned(),
            None,
        ));
    }
    // TASK-speech-input-cap: unbounded `input` means unbounded PCM growth;
    // reject-not-clamp (no silent truncation, no derived clamp).
    let input_chars = req.input.chars().count();
    if input_chars > SPEECH_INPUT_MAX_CHARS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "input is {input_chars} characters; the cap is {SPEECH_INPUT_MAX_CHARS} — rejected, not truncated"
            ),
            Some("input_too_long"),
        ));
    }
    Ok(format)
}

/// `POST /v1/audio/speech` (REQ-9): `OpenAI`-shaped TTS. Same worker as
/// `/v1/synthesize`. `model` optional (server default TTS), `voice` passed
/// through; `response_format` is `"wav"` (default, RIFF/WAVE container),
/// `"pcm"` (raw s16le, no container), `"flac"` (lossless), or `"opus"`
/// (Ogg-encapsulated, gapless). Applies to every TTS backend (stub + Kokoro):
/// encoding runs on the worker's PCM16, never in the voice model.
#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    request_body = SpeechRequest,
    responses(
        (status = 200, description = "WAV audio (RIFF container)", content_type = "audio/wav"),
        (status = 200, description = "Raw s16le mono PCM (pcm response_format; no container)", content_type = "audio/pcm-s16le"),
        (status = 200, description = "Lossless FLAC (flac response_format)", content_type = "audio/flac"),
        (status = 200, description = "Ogg-encapsulated Opus, gapless (opus response_format)", content_type = "audio/ogg"),
        (status = 400, description = "Empty input, oversized input (over 4096 characters), unknown model, or unsupported response_format", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Synthesis or audio encoding failed", body = OpenAiError)
    )
)]
pub(crate) async fn post_audio_speech(
    State(config): State<AppConfig>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    let format = match parse_speech_request(&req) {
        Ok(format) => format,
        Err((code, message, err_code)) => {
            return openai_error_response(code, message, err_code);
        }
    };
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
    // Admission first, registration second: a 429/503 carries no job id and
    // never enters the registry. Audio uses the `llm` bucket; one blocking
    // FFI call, never interruptible.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await,
        crate::jobs_registry::JobKind::Synthesize,
        (!worker_req.model.is_empty()).then_some(worker_req.model.clone()),
        false,
    ) {
        Ok(pair) => pair,
        Err((code, message)) => {
            return openai_error_response(
                code,
                message,
                (code == StatusCode::TOO_MANY_REQUESTS).then_some("gpu_slot_busy"),
            );
        }
    };
    let audio = match run_synthesis(config.tts.clone(), worker_req, lease).await {
        Ok(audio) => audio,
        Err((code, message)) => return openai_error_response(code, message, None),
    };
    if format == SpeechFormat::Pcm {
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
    if format == SpeechFormat::Wav {
        let wav = encode_wav(audio.sample_rate, &audio.pcm16);
        return with_job_id(
            ([(header::CONTENT_TYPE, "audio/wav")], wav).into_response(),
            job.id(),
        );
    }
    // FLAC/Opus are CPU-bound encodes: off the async executor, same
    // spawn_blocking discipline as synthesis. The encoder input is the
    // worker's PCM16, so stub and Kokoro (24 kHz native) share the path.
    let (mime, bytes) = match tokio::task::spawn_blocking(move || match format {
        SpeechFormat::Flac => {
            encode_flac(audio.sample_rate, &audio.pcm16).map(|bytes| ("audio/flac", bytes))
        }
        SpeechFormat::Opus => {
            encode_opus(audio.sample_rate, &audio.pcm16).map(|bytes| ("audio/ogg", bytes))
        }
        SpeechFormat::Wav | SpeechFormat::Pcm => unreachable!("handled above"),
    })
    .await
    {
        Err(err) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("encode task failed: {err}"),
                None,
            );
        }
        Ok(Err(err)) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("encode failed: {err}"),
                None,
            );
        }
        Ok(Ok((mime, bytes))) => (mime, bytes),
    };
    with_job_id(
        ([(header::CONTENT_TYPE, mime)], bytes).into_response(),
        job.id(),
    )
}
