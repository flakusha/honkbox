//! Multipart audio-upload helpers, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only the multipart parsing helpers
//! live here.

use axum::{body::Bytes, extract::Multipart, http::StatusCode, response::Response};
use honkbox_audio_worker::decode_wav;

use super::openai_constraints::openai_error_response;

/// Decode an uploaded audio payload for the REQ-9 transcriptions route:
/// the same formats the native `/v1/transcribe` accepts (`audio/wav`,
/// `audio/pcm-s16le` + `sample_rate`), plus a RIFF sniff for
/// `application/octet-stream` parts (the `openai` SDK's default file
/// content type).
pub(super) fn decode_upload_audio(
    content_type: &str,
    bytes: &[u8],
    sample_rate: Option<u32>,
) -> Result<(u32, Vec<i16>), (StatusCode, String)> {
    let mime = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match mime.as_str() {
        "audio/wav" | "audio/wave" | "audio/x-wav" => decode_wav(bytes)
            .map_err(|err| (StatusCode::BAD_REQUEST, format!("invalid wav: {err}"))),
        "audio/pcm-s16le" => {
            let sample_rate = sample_rate.ok_or((
                StatusCode::BAD_REQUEST,
                "audio/pcm-s16le requires a `sample_rate` form field or `?sample_rate=` query"
                    .to_owned(),
            ))?;
            if !bytes.len().is_multiple_of(2) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "odd audio/pcm-s16le length".to_owned(),
                ));
            }
            let (pairs, _) = bytes.as_chunks::<2>();
            let pcm = pairs.iter().map(|pair| i16::from_le_bytes(*pair)).collect();
            Ok((sample_rate, pcm))
        }
        "" | "application/octet-stream" | "binary/octet-stream" => {
            if bytes.starts_with(b"RIFF") {
                decode_wav(bytes)
                    .map_err(|err| (StatusCode::BAD_REQUEST, format!("invalid wav: {err}")))
            } else {
                Err((
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "unrecognized audio payload: send audio/wav (or a RIFF wav under application/octet-stream), or audio/pcm-s16le with a sample_rate".to_owned(),
                ))
            }
        }
        other => Err((
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            format!("unsupported file content-type {other:?}; use audio/wav or audio/pcm-s16le"),
        )),
    }
}

/// Collected multipart fields for `/v1/audio/transcriptions`.
pub(super) struct TranscriptionUpload {
    /// `file` part: content type + bytes.
    pub(super) file: Option<(String, Bytes)>,
    /// Optional server-default-overridable STT model.
    pub(super) model: Option<String>,
    /// Optional language hint.
    pub(super) language: Option<String>,
    /// `json` (default) | `verbose_json`.
    pub(super) response_format: Option<String>,
    /// pcm-s16le input rate (form field; query supplies the fallback).
    pub(super) sample_rate: Option<u32>,
}

/// Collect the multipart fields. Unknown fields are ignored (`OpenAI`
/// SDKs add `purpose` and friends). The boxed error is the ready-made
/// `OpenAI` envelope response.
pub(super) async fn parse_transcription_upload(
    mut multipart: Multipart,
    sample_rate_query: Option<u32>,
) -> Result<TranscriptionUpload, Box<Response>> {
    let mut upload = TranscriptionUpload {
        file: None,
        model: None,
        language: None,
        response_format: None,
        sample_rate: sample_rate_query,
    };
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| bad_multipart(format!("invalid multipart body: {err}")))?
    {
        let name = field.name().unwrap_or("").to_owned();
        match name.as_str() {
            "file" => {
                let content_type = field.content_type().unwrap_or("").to_owned();
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|err| bad_multipart(format!("unreadable file field: {err}")))?;
                upload.file = Some((content_type, bytes));
            }
            "model" | "language" | "response_format" | "sample_rate" => {
                let text = field
                    .text()
                    .await
                    .map_err(|err| bad_multipart(format!("unreadable {name} field: {err}")))?;
                set_upload_field(&mut upload, name.as_str(), text)?;
            }
            _ => {}
        }
    }
    Ok(upload)
}

/// One non-file multipart field, with per-field validation.
pub(super) fn set_upload_field(
    upload: &mut TranscriptionUpload,
    name: &str,
    text: String,
) -> Result<(), Box<Response>> {
    let value = (!text.trim().is_empty()).then_some(text);
    match name {
        "model" => upload.model = value,
        "language" => upload.language = value,
        "response_format" => upload.response_format = value,
        "sample_rate" => {
            match value
                .as_deref()
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .and_then(|text| text.parse::<u32>().ok())
            {
                Some(rate) => upload.sample_rate = Some(rate),
                None if value.is_some() => {
                    return Err(bad_multipart(
                        "sample_rate must be a positive integer (Hz)".to_owned(),
                    ));
                }
                None => {}
            }
        }
        _ => {}
    }
    Ok(())
}

/// Multipart parse failure → `OpenAI` 400 envelope. Deliberately boxed:
/// the `Result` error variant must stay small (`result_large_err`),
/// trading the `unnecessary_box_returns` nag.
#[allow(clippy::unnecessary_box_returns)]
pub(super) fn bad_multipart(message: String) -> Box<Response> {
    Box::new(openai_error_response(
        StatusCode::BAD_REQUEST,
        message,
        None,
    ))
}
