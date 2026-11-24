//! Recognition routes, extracted from `routes.rs`.
//!
//! `POST /v1/recognize` takes one `data:` URI image and returns a
//! recognition result synchronously; unsupported kinds answer a named 501.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use base64::Engine as _;
use honkbox_api_types::{
    OpenAiError, RecKind, RecognitionEngineMeta, RecognitionPage, RecognitionRegion,
    RecognizeRequest, RecognizeResult,
};
use honkbox_recognition_worker::{
    RecError, RecKind as WorkerRecKind, RecognizeRequest as WorkerRecognizeRequest,
};

use super::{AppConfig, openai_error_response};

/// Shared 501 hint for unsupported recognition kinds: `recognition` is a
/// TOP-LEVEL capabilities field, not a member of `features`.
pub(super) fn recognition_501_message(detail: &str) -> String {
    format!("{detail}; see GET /v1/capabilities recognition")
}

fn not_served(detail: &str, code: &'static str) -> Response {
    openai_error_response(
        StatusCode::NOT_IMPLEMENTED,
        recognition_501_message(detail),
        Some(code),
    )
}

/// `POST /v1/recognize` (recognition tickets): one image in, recognition
/// out. Stage 1 is synchronous — barcode-class latency is sub-millisecond,
/// so the request-response shape needs no job registry entry; the
/// OCR/document pipeline becomes job-based (cancellable, cache-keyed per
/// FEAT-ocr-ingest) when the first OCR backend lands, and this route is
/// that shape's already-valid request prefix. Only `data:` URI image
/// payloads are accepted: nothing is ever fetched, so there is no SSRF
/// surface (same policy as `/v1/image-ops`). Kinds without a compiled
/// backend answer the named 501 `recognition_kind_unsupported` — the same
/// `kinds()` gate the top-level `recognition` capability advertises, so
/// honest absence is observable from both directions.
// The honesty gate, image decode and four error mappings belong to this one
// route; splitting them would scatter the 501/400 contract across helpers.
#[allow(clippy::too_many_lines)]
#[utoipa::path(
    post,
    path = "/v1/recognize",
    request_body = RecognizeRequest,
    responses(
        (status = 200, description = "Recognition result (`pages: []` when a valid image held nothing recognizable)", body = RecognizeResult),
        (status = 400, description = "Unknown field or kind (typed enum parse), non-`data:` URI, undeclared base64, unsupported container, invalid base64, or empty payload", body = OpenAiError),
        (status = 501, description = "Kind (or engine override) not served by this build's compiled backends (`recognition_kind_unsupported`)", body = OpenAiError),
        (status = 500, description = "Recognition engine failure", body = OpenAiError)
    )
)]
pub(crate) async fn post_recognize(
    State(config): State<AppConfig>,
    body: Result<Json<RecognizeRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(req)) => req,
        Err(rejection) => {
            // deny_unknown_fields / malformed JSON / unknown kind all land
            // here as one named 400 carrying serde's specific message.
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid recognize request: {}", rejection.body_text()),
                Some("invalid_recognize_request"),
            );
        }
    };
    let kind = match req.kind {
        RecKind::Barcode => WorkerRecKind::Barcode,
        RecKind::Ocr => WorkerRecKind::Ocr,
    };
    // Honesty gate FIRST: a kind this build cannot serve is a 501 even when
    // the payload is also broken — the kind is the primary selector, and the
    // 501 message is what a capability-driven client is built to handle.
    let kind_backend = config.recognition.backend_id_for(kind);
    if !config.recognition.serves(kind) {
        return not_served(
            &format!(
                "recognition kind {:?} is not served by this build (backend {kind_backend:?})",
                kind.as_str()
            ),
            "recognition_kind_unsupported",
        );
    }
    if let Some(engine) = &req.engine
        && engine != kind_backend
    {
        return not_served(
            &format!(
                "recognition engine {engine:?} is not served by this build (backend {kind_backend:?})"
            ),
            "recognition_engine_unsupported",
        );
    }
    // The route owns input resolution: worker backends only ever see
    // decoded container bytes, never paths or URLs.
    let bytes = match decode_recognize_image(&req.image) {
        Ok(bytes) => bytes,
        Err((message, code)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
    };
    let worker_req = WorkerRecognizeRequest {
        kind,
        image: bytes,
        language: req.language.clone(),
        mode: req.mode.clone(),
        engine: req.engine.clone(),
    };
    let out = match config.recognition.recognize(&worker_req) {
        Ok(out) => out,
        Err(RecError::Unsupported { kind, backend }) => {
            return not_served(
                &format!(
                    "recognition kind {:?} is not served by backend {backend:?}",
                    kind.as_str()
                ),
                "recognition_kind_unsupported",
            );
        }
        Err(RecError::Invalid(message)) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                message,
                Some("invalid_recognize_input"),
            );
        }
        Err(RecError::Engine(message)) => {
            // Metric label carries the kind, never the image or its text.
            tracing::warn!(kind = kind.as_str(), "recognition engine failure");
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("recognition engine failed: {message}"),
                Some("recognition_engine_failed"),
            );
        }
    };
    Json(RecognizeResult {
        kind: req.kind,
        engine: RecognitionEngineMeta {
            id: out.engine.id.to_owned(),
            version: out.engine.version.map(str::to_owned),
            bbox_support: out.engine.bbox_support,
        },
        pages: out
            .pages
            .into_iter()
            .map(|page| RecognitionPage {
                text: page.text,
                regions: page
                    .regions
                    .into_iter()
                    .map(|region| RecognitionRegion {
                        bbox: region.bbox,
                        text: region.text,
                        confidence: region.confidence,
                    })
                    .collect(),
            })
            .collect(),
    })
    .into_response()
}

/// Decode the recognize route's `data:image/…;base64,…` input. Same MIME
/// vocabulary + base64 tolerances as the image-ops path (`decode_data_uri`),
/// minus its artifact/FormatInfo plumbing — recognition stage 1 takes
/// exactly one image and no bundle inputs.
pub(super) fn decode_recognize_image(uri: &str) -> Result<Vec<u8>, (String, &'static str)> {
    let bad = |message: String| (message, "invalid_recognize_input");
    let trimmed = uri.trim();
    let Some(rest) = trimmed.strip_prefix("data:") else {
        return Err(bad(
            "image must be a data URI (`data:image/…;base64,…`); paths and remote URLs are never fetched".to_owned(),
        ));
    };
    let Some((header, payload)) = rest.split_once(',') else {
        return Err(bad(
            "data URI is missing the `,<base64 payload>` part".to_owned()
        ));
    };
    let Some(mime) = header.strip_suffix(";base64") else {
        return Err(bad("data URI must declare `;base64`".to_owned()));
    };
    if crate::formats::image_ops_by_mime(mime).is_none() {
        return Err(bad(format!(
            "unsupported image container {mime:?}: accepted data URIs are {}",
            crate::formats::image_ops_inputs()
                .map(|row| row.mime)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        // Tolerate padding-stripped payloads (some encoders trim `=`).
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(payload.trim()))
        .map_err(|e| bad(format!("data URI is not valid base64: {e}")))?;
    if bytes.is_empty() {
        return Err(bad("image payload is empty".to_owned()));
    }
    Ok(bytes)
}
