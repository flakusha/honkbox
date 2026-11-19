//! `POST /v1/audio/transcriptions` + `POST /v1/audio/translations` —
//! the `OpenAI`-shaped multipart audio routes, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage
//! gate scans that file for `.route()` calls. Only the handlers live here.

use axum::{
    Json,
    extract::{Multipart, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use honkbox_api_types::{OpenAiError, TranscriptSegment, TranscriptionText, TranscriptionVerbose};
use honkbox_audio_worker::TranscribeRequest as WorkerTranscribeRequest;

use super::audio_multipart::{decode_upload_audio, parse_transcription_upload};
use super::audio_subtitles::{render_srt, render_vtt};
use super::audio_upload::run_transcription;
use super::chat_prep::with_job_id;
use super::openai_constraints::openai_error_response;
use super::{AppConfig, validate_model};
use serde::Deserialize;

/// Query carried over from the native route: `audio/pcm-s16le` uploads
/// without a `sample_rate` form field.
#[derive(Debug, Deserialize)]
pub struct AudioSampleRateQuery {
    /// Input sample rate in Hz (pcm-s16le uploads only).
    pub(super) sample_rate: Option<u32>,
}

/// `POST /v1/audio/transcriptions` (REQ-9): `OpenAI`-shaped multipart
/// wrapper over the same STT worker as `/v1/transcribe`. `file` is
/// required (`audio/wav`, RIFF-sniffed `application/octet-stream`, or
/// `audio/pcm-s16le` + `sample_rate` form field / query); `model` is
/// optional (server default STT); `language` optional; `response_format`
/// `json` (default), `verbose_json` (adds the worker's timestamped
/// segments), `srt` or `vtt` (`SubRip` / `WebVTT` subtitles rendered from the
/// same segments; `application/x-subrip` / `text/vtt`). Errors are
/// `OpenAI` envelopes; only a request that is not multipart at all gets
/// axum's transport-level 400 before the handler runs. The native route is
/// unchanged.
#[utoipa::path(
    post,
    path = "/v1/audio/transcriptions",
    request_body(content = String, description = "multipart/form-data: file, model?, language?, response_format? (json|verbose_json|srt|vtt), sample_rate?"),
    responses(
        (status = 200, description = "Transcription (json / verbose_json / srt / vtt)", body = TranscriptionText),
        (status = 400, description = "Missing file, bad audio, unknown model, unsupported response_format, or missing sample_rate", body = OpenAiError),
        (status = 415, description = "Unsupported audio content type", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Transcription failed", body = OpenAiError)
    )
)]
pub(super) async fn post_audio_transcriptions(
    State(config): State<AppConfig>,
    Query(query): Query<AudioSampleRateQuery>,
    multipart: Multipart,
) -> Response {
    let upload = match parse_transcription_upload(multipart, query.sample_rate).await {
        Ok(upload) => upload,
        Err(response) => return *response,
    };
    let Some((content_type, bytes)) = upload.file else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "file is required: attach the audio payload as a `file` form field".to_owned(),
            None,
        );
    };
    match upload.response_format.as_deref() {
        None | Some("json" | "verbose_json" | "srt" | "vtt") => {}
        Some(other) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "unsupported response_format {other:?}: only \"json\", \"verbose_json\", \"srt\", \"vtt\""
                ),
                None,
            );
        }
    }
    let catalog = config.catalog_snapshot();
    if let Err(message) = validate_model(
        upload.model.as_deref(),
        config.stt_model.as_deref(),
        &catalog,
    ) {
        return openai_error_response(StatusCode::BAD_REQUEST, message, None);
    }
    let (sample_rate, pcm) = match decode_upload_audio(&content_type, &bytes, upload.sample_rate) {
        Ok(pair) => pair,
        Err((code, message)) => return openai_error_response(code, message, None),
    };
    let worker_req = WorkerTranscribeRequest {
        model: upload.model.unwrap_or_default(),
        language: upload.language,
        translate: false,
    };
    // NEW-3: registry-visible while the request lives (one blocking FFI
    // call — the cancel token is marked but never checked).
    let job = crate::jobs_registry::global().register(
        crate::jobs_registry::JobKind::Transcribe,
        (!worker_req.model.is_empty()).then_some(worker_req.model.clone()),
        false,
    );
    let out = match run_transcription(config.stt.clone(), worker_req, sample_rate, pcm).await {
        Ok(out) => out,
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
    let response = match upload.response_format.as_deref() {
        Some("verbose_json") => Json(TranscriptionVerbose {
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
        })
        .into_response(),
        Some("srt") => (
            [(header::CONTENT_TYPE, "application/x-subrip")],
            render_srt(&out.segments),
        )
            .into_response(),
        Some("vtt") => (
            [(header::CONTENT_TYPE, "text/vtt; charset=utf-8")],
            render_vtt(&out.segments),
        )
            .into_response(),
        _ => Json(TranscriptionText { text: out.text }).into_response(),
    };
    with_job_id(response, job.id())
}

/// `POST /v1/audio/translations` (REQ-9 sibling; `ReviewedAudio` P1):
/// `OpenAI` translation shape — the multipart `file` is decoded exactly
/// like `/v1/audio/transcriptions` and transcribed with whisper's
/// **translate** task (source language auto-detected, output always
/// English). `model` optional (server default STT); `response_format`
/// `json` (default) / `verbose_json` / `srt` / `vtt`. No `language` field:
/// the endpoint contract fixes the OUTPUT language, and whisper's translate
/// task takes none. Errors are `OpenAI` envelopes, same as transcriptions.
#[utoipa::path(
    post,
    path = "/v1/audio/translations",
    request_body(content = String, description = "multipart/form-data: file, model?, response_format? (json|verbose_json|srt|vtt), sample_rate?"),
    responses(
        (status = 200, description = "English translation (json / verbose_json / srt / vtt)", body = TranscriptionText),
        (status = 400, description = "Missing file, bad audio, unknown model, unsupported response_format, unexpected language field, or missing sample_rate", body = OpenAiError),
        (status = 415, description = "Unsupported audio content type", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Translation failed", body = OpenAiError)
    )
)]
pub(super) async fn post_audio_translations(
    State(config): State<AppConfig>,
    Query(query): Query<AudioSampleRateQuery>,
    multipart: Multipart,
) -> Response {
    let upload = match parse_transcription_upload(multipart, query.sample_rate).await {
        Ok(upload) => upload,
        Err(response) => return *response,
    };
    let Some((content_type, bytes)) = upload.file else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "file is required: attach the audio payload as a `file` form field".to_owned(),
            None,
        );
    };
    // The shared parser accepts `language` (the transcriptions field); on
    // the translations contract it is a client error, never silently
    // ignored — the output language is fixed to English.
    if upload.language.is_some() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "language is not a /v1/audio/translations field: the output is always English"
                .to_owned(),
            None,
        );
    }
    match upload.response_format.as_deref() {
        None | Some("json" | "verbose_json" | "srt" | "vtt") => {}
        Some(other) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "unsupported response_format {other:?}: only \"json\", \"verbose_json\", \"srt\", \"vtt\""
                ),
                None,
            );
        }
    }
    let catalog = config.catalog_snapshot();
    if let Err(message) = validate_model(
        upload.model.as_deref(),
        config.stt_model.as_deref(),
        &catalog,
    ) {
        return openai_error_response(StatusCode::BAD_REQUEST, message, None);
    }
    let (sample_rate, pcm) = match decode_upload_audio(&content_type, &bytes, upload.sample_rate) {
        Ok(pair) => pair,
        Err((code, message)) => return openai_error_response(code, message, None),
    };
    let worker_req = WorkerTranscribeRequest {
        model: upload.model.unwrap_or_default(),
        language: None,
        translate: true,
    };
    // NEW-3: registry-visible while the request lives (one blocking FFI
    // call — the cancel token is marked but never checked).
    let job = crate::jobs_registry::global().register(
        crate::jobs_registry::JobKind::Translations,
        (!worker_req.model.is_empty()).then_some(worker_req.model.clone()),
        false,
    );
    let out = match run_transcription(config.stt.clone(), worker_req, sample_rate, pcm).await {
        Ok(out) => out,
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
    let response = match upload.response_format.as_deref() {
        Some("verbose_json") => Json(TranscriptionVerbose {
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
        })
        .into_response(),
        Some("srt") => (
            [(header::CONTENT_TYPE, "application/x-subrip")],
            render_srt(&out.segments),
        )
            .into_response(),
        Some("vtt") => (
            [(header::CONTENT_TYPE, "text/vtt; charset=utf-8")],
            render_vtt(&out.segments),
        )
            .into_response(),
        _ => Json(TranscriptionText { text: out.text }).into_response(),
    };
    with_job_id(response, job.id())
}
