//! `POST /v1/audio/transcriptions` + `POST /v1/audio/translations` —
//! the `OpenAI`-shaped multipart audio routes, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage
//! gate scans that file for `.route()` calls. Only the handlers live here.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Json,
    extract::{Multipart, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response, sse::Event, sse::KeepAlive, sse::Sse},
};
use honkbox_api_types::{
    OpenAiError, TranscriptSegment, TranscriptionStreamEvent, TranscriptionStreamPhase,
    TranscriptionText, TranscriptionVerbose,
};
#[cfg(feature = "audio-vad")]
use honkbox_audio_worker::SileroVad;
use honkbox_audio_worker::TranscribeRequest as WorkerTranscribeRequest;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::audio_multipart::{decode_upload_audio, parse_transcription_upload};
use super::audio_subtitles::{render_srt, render_vtt};
use super::audio_upload::run_transcription;
use super::chat_prep::{admit_then_register, with_job_id};
use super::openai_constraints::openai_error_response;
use super::{AppConfig, validate_model};
use serde::Deserialize;

/// Query carried over from the native route: `audio/pcm-s16le` uploads
/// without a `sample_rate` form field. `stream` is the additive SSE switch.
#[derive(Debug, Deserialize)]
pub struct AudioSampleRateQuery {
    /// Input sample rate in Hz (pcm-s16le uploads only).
    pub(super) sample_rate: Option<u32>,
    /// `?stream=true` serves ordered partial transcripts over SSE.
    pub(super) stream: Option<bool>,
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
/// unchanged. `?stream=true` switches to an additive SSE feed of ordered
/// partial/final transcript frames over bounded windows; absent/false keeps
/// the file-only response exactly as before.
#[utoipa::path(
    post,
    path = "/v1/audio/transcriptions",
    params(
        ("sample_rate" = Option<u32>, Query, description = "Input sample rate in Hz (audio/pcm-s16le uploads only)"),
        ("stream" = Option<bool>, Query, description = "true → SSE partial/final transcript frames over bounded windows; absent/false → file-only JSON response")
    ),
    request_body(content = String, description = "multipart/form-data: file, model?, language?, response_format? (json|verbose_json|srt|vtt), sample_rate?"),
    responses(
        (status = 200, description = "File-only transcription (json / verbose_json / srt / vtt); with stream=true, an SSE stream of ordered partial/final transcript frames", body = TranscriptionText),
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
    // Additive mode: the file-only path below is byte-identical when the
    // flag is absent. The SSE feed emits bounded partial/final frames and
    // never touches the file response shape.
    if query.stream.unwrap_or(false) {
        return post_audio_transcriptions_stream(State(config), Query(query), multipart).await;
    }
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
    // Admission first, registration second: a 429/503 carries no job id and
    // never enters the registry. Audio uses the `llm` bucket (see
    // `run_transcription`); one blocking FFI call, never interruptible.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await,
        crate::jobs_registry::JobKind::Transcribe,
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
    let out = match run_transcription(config.stt.clone(), worker_req, sample_rate, pcm, lease).await
    {
        Ok(out) => out,
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
    // Admission first, registration second: a 429/503 carries no job id and
    // never enters the registry. Audio uses the `llm` bucket (see
    // `run_transcription`); one blocking FFI call, never interruptible.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await,
        crate::jobs_registry::JobKind::Translations,
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
    let out = match run_transcription(config.stt.clone(), worker_req, sample_rate, pcm, lease).await
    {
        Ok(out) => out,
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

// ── SSE streaming STT (bounded windows) ─────────────────────────────────
// Additive `?stream=true` mode of the file-only route above. Every bound is
// a compile-time constant: at most `STREAM_TOTAL_MS` of audio, at most
// `STREAM_MAX_FRAMES` frames, at most `STREAM_WINDOW_MS` per window, and a
// bounded channel — no input can grow an unbounded buffer or hold the GPU
// slot past `STREAM_DEADLINE`.

/// Longest single window handed to the STT worker (ms).
const STREAM_WINDOW_MS: u32 = 5_000;
/// Longest accepted stream (ms); also caps lease residency.
const STREAM_TOTAL_MS: u32 = 60_000;
/// Defensive frame ceiling. VAD can emit 32 ms spans, so 60 s of speech is
/// at most 1875 frames; this only catches pathological input without
/// truncating a speech-aligned stream.
const STREAM_MAX_FRAMES: usize = 2048;
/// Bounded SSE channel: slow readers apply back-pressure, never buffering.
const STREAM_CHANNEL_CAPACITY: usize = 8;
/// Wall-clock deadline for one stream; checked between window boundaries.
const STREAM_DEADLINE: Duration = Duration::from_secs(90);
/// Optional Silero ONNX model. Absent (or feature off) → fixed windows.
#[cfg(feature = "audio-vad")]
const VAD_MODEL_ENV: &str = "HONKBOX_VAD_MODEL";

/// Half-open sample range `[start, end)` with its millisecond offsets.
#[derive(Debug, Clone, Copy)]
struct StreamWindow {
    start: usize,
    end: usize,
    start_ms: u32,
    end_ms: u32,
}

fn ms_to_samples(ms: u32, rate: u32) -> usize {
    usize::try_from(u64::from(ms) * u64::from(rate) / 1000).unwrap_or(usize::MAX)
}

fn samples_to_ms(samples: usize, rate: u32) -> u32 {
    if rate == 0 {
        return 0;
    }
    u32::try_from(u64::try_from(samples).unwrap_or(u64::MAX) * 1000 / u64::from(rate))
        .unwrap_or(u32::MAX)
}

/// Fixed-size fallback chunker — also the complete path when `audio-vad`
/// is compiled off. Covers every sample exactly once.
fn fixed_windows(pcm_len: usize, rate: u32, window_ms: u32) -> Vec<StreamWindow> {
    if pcm_len == 0 || rate == 0 || window_ms == 0 {
        return Vec::new();
    }
    let per = ms_to_samples(window_ms, rate).max(1);
    (0..pcm_len)
        .step_by(per)
        .map(|start| {
            let end = start.saturating_add(per).min(pcm_len);
            StreamWindow {
                start,
                end,
                start_ms: samples_to_ms(start, rate),
                end_ms: samples_to_ms(end, rate),
            }
        })
        .collect()
}

/// Speech-aligned windows from the worker's bounded VAD chunker. Returns
/// `None` when VAD is unconfigured, cannot load, or hears no speech — the
/// caller then falls back to fixed windows, so silence is still transcribed
/// honestly rather than silently dropped.
#[cfg(feature = "audio-vad")]
fn vad_windows(pcm: &[i16], rate: u32) -> Option<Vec<StreamWindow>> {
    let path = std::env::var_os(VAD_MODEL_ENV)?;
    let vad = SileroVad::load(std::path::Path::new(&path)).ok()?;
    let spans = vad.bounded_spans(pcm, rate, STREAM_WINDOW_MS).ok()?;
    let windows: Vec<StreamWindow> = spans
        .into_iter()
        .map(|span| StreamWindow {
            start: ms_to_samples(span.start_ms, rate),
            end: ms_to_samples(span.end_ms, rate).min(pcm.len()),
            start_ms: span.start_ms,
            end_ms: span.end_ms,
        })
        .filter(|window| window.end > window.start)
        .collect();
    (!windows.is_empty()).then_some(windows)
}

fn stream_windows(pcm: &[i16], rate: u32) -> Vec<StreamWindow> {
    #[cfg(feature = "audio-vad")]
    {
        if let Some(windows) = vad_windows(pcm, rate) {
            return windows;
        }
    }
    fixed_windows(pcm.len(), rate, STREAM_WINDOW_MS)
}

/// `?stream=true` mode of `/v1/audio/transcriptions`: ordered SSE partial
/// and final transcript frames over bounded audio windows. One admission
/// lease and one registry job span the whole stream; the blocking worker
/// owns both so disconnect (send fails), cancel (token marked), and the
/// deadline each drop them at the next window boundary. The file-only
/// handler above is unchanged when the flag is absent.
#[allow(clippy::too_many_lines)] // one linear pass: validate → admit → stream frames
pub(super) async fn post_audio_transcriptions_stream(
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
    // SSE frames carry transcript text; subtitle formats have no meaning.
    if let Some(other) = upload.response_format.as_deref()
        && other != "json"
    {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("unsupported response_format {other:?} for SSE streaming: only \"json\""),
            None,
        );
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
    if sample_rate == 0 || pcm.is_empty() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "empty audio or zero sample_rate".to_owned(),
            None,
        );
    }
    let total_ms = samples_to_ms(pcm.len(), sample_rate);
    if total_ms > STREAM_TOTAL_MS {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("audio is {total_ms} ms; streaming accepts at most {STREAM_TOTAL_MS} ms"),
            None,
        );
    }
    let worker_req = WorkerTranscribeRequest {
        model: upload.model.unwrap_or_default(),
        language: upload.language,
        translate: false,
    };
    let model_label = (!worker_req.model.is_empty()).then(|| worker_req.model.clone());
    // Same ordering as every GPU route: admission first (a 429 registers no
    // job), then an interruptible registry job. Audio shares the `llm`
    // bucket (see `run_transcription` for why).
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await,
        crate::jobs_registry::JobKind::Transcribe,
        model_label,
        true,
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
    let job_id = job.id().to_owned();
    let cancel = job.token().clone();
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(STREAM_CHANNEL_CAPACITY);
    let pipe = Arc::clone(&config.stt);
    tokio::task::spawn_blocking(move || {
        // Both guards live exactly as long as the stream and drop on every
        // exit path, including the early `return`s below.
        let _job = job;
        let _lease = lease;
        let deadline = Instant::now() + STREAM_DEADLINE;
        let windows = stream_windows(&pcm, sample_rate);
        let total = windows.len().min(STREAM_MAX_FRAMES);
        let mut seq = 0u32;
        let mut last_ms = samples_to_ms(pcm.len(), sample_rate);
        let mut sent_final = false;
        for (idx, window) in windows.into_iter().take(total).enumerate() {
            if cancel.is_cancelled() || Instant::now() >= deadline {
                break;
            }
            let is_final = idx + 1 == total;
            let slice = pcm[window.start..window.end].to_vec();
            let text = match pipe.transcribe(&slice, sample_rate, &worker_req) {
                Ok(transcript) => transcript.text,
                Err(err) => {
                    let frame = TranscriptionStreamEvent {
                        seq,
                        phase: TranscriptionStreamPhase::Final,
                        text: format!("transcribe failed: {err}"),
                        start_ms: window.start_ms,
                        end_ms: window.end_ms,
                    };
                    let data = serde_json::to_string(&frame).unwrap_or_default();
                    let _ = tx.blocking_send(Ok(Event::default().data(data)));
                    return;
                }
            };
            let frame = TranscriptionStreamEvent {
                seq,
                phase: if is_final {
                    TranscriptionStreamPhase::Final
                } else {
                    TranscriptionStreamPhase::Partial
                },
                text,
                start_ms: window.start_ms,
                end_ms: window.end_ms,
            };
            let data = serde_json::to_string(&frame).unwrap_or_default();
            if tx.blocking_send(Ok(Event::default().data(data))).is_err() {
                return; // client disconnected: the slot frees here
            }
            seq += 1;
            last_ms = window.end_ms;
            sent_final = is_final;
        }
        if !sent_final {
            let frame = TranscriptionStreamEvent {
                seq,
                phase: TranscriptionStreamPhase::Final,
                text: String::new(),
                start_ms: last_ms,
                end_ms: last_ms,
            };
            let data = serde_json::to_string(&frame).unwrap_or_default();
            let _ = tx.blocking_send(Ok(Event::default().data(data)));
        }
    });
    with_job_id(
        Sse::new(ReceiverStream::new(rx))
            .keep_alive(KeepAlive::default())
            .into_response(),
        &job_id,
    )
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    #[test]
    fn fixed_windows_split_and_cover_all_samples() {
        // 16 kHz, 5 s window = 80_000 samples.
        let windows = fixed_windows(200_000, 16_000, STREAM_WINDOW_MS);
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].start, 0);
        assert_eq!(windows[0].end - windows[0].start, 80_000);
        assert_eq!(windows[1].start, 80_000);
        assert_eq!(windows[2].end, 200_000);
        assert_eq!(windows[2].end_ms, samples_to_ms(200_000, 16_000));
    }

    #[test]
    fn fixed_windows_accept_short_and_reject_degenerate_inputs() {
        let short = fixed_windows(25_000, 16_000, STREAM_WINDOW_MS);
        assert_eq!(short.len(), 1);
        assert_eq!(short[0].end, 25_000);
        assert!(fixed_windows(0, 16_000, STREAM_WINDOW_MS).is_empty());
        assert!(fixed_windows(100, 0, STREAM_WINDOW_MS).is_empty());
        assert!(fixed_windows(100, 16_000, 0).is_empty());
    }

    #[test]
    fn total_duration_cap_yields_bounded_window_count() {
        let samples = ms_to_samples(STREAM_TOTAL_MS, 16_000);
        assert_eq!(samples_to_ms(samples, 16_000), STREAM_TOTAL_MS);
        let windows = fixed_windows(samples, 16_000, STREAM_WINDOW_MS);
        assert_eq!(
            windows.len(),
            usize::try_from(STREAM_TOTAL_MS / STREAM_WINDOW_MS).unwrap()
        );
        assert!(windows.len() <= STREAM_MAX_FRAMES);
    }

    #[test]
    fn stream_frames_serialize_in_order_with_wire_phases() {
        let frame = TranscriptionStreamEvent {
            seq: 0,
            phase: TranscriptionStreamPhase::Partial,
            text: "hi".to_owned(),
            start_ms: 0,
            end_ms: 100,
        };
        let json = serde_json::to_string(&frame).unwrap();
        assert!(json.starts_with("{\"seq\":0,\"phase\":\"partial\""));
        assert!(json.contains("\"end_ms\":100"));
    }

    #[test]
    fn stream_query_flag_defaults_off() {
        let query: AudioSampleRateQuery = serde_json::from_str(r#"{"sample_rate":8000}"#).unwrap();
        assert!(!query.stream.unwrap_or(false));
        let query: AudioSampleRateQuery = serde_json::from_str(r#"{"stream":true}"#).unwrap();
        assert_eq!(query.stream, Some(true));
    }
}
