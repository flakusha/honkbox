//! Chat request-preparation helpers, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only the shared request helpers
//! live here.

use std::sync::Arc;
use std::time::Duration;

use axum::{
    http::{HeaderValue, StatusCode, header},
    response::Response,
};
use honkbox_api_types::{ChatMessage, ChatRole, ModelCatalogEntry};
use honkbox_llm_worker::{
    ChatImage as WorkerChatImage, ChatMessage as WorkerChatMessage,
    ChatRequest as WorkerChatRequest, LlmPipeline, Role, SamplerParams, ToolCallSpec,
};
use serde::Deserialize;

use super::AppConfig;
use super::embeddings::embeddings_model_gate;
use super::openai_constraints::openai_error_response;

/// `POST /v1/transcribe` query: model slug + language for the worker request;
/// `sample_rate` is required only for raw `audio/pcm-s16le` bodies.
/// `ttl_seconds` is per-request residency advice (NEW-8): clamped at this
/// route, consumed once per-kind swap actors land (audio pipelines are
/// startup-loaded today).
#[derive(Debug, Deserialize)]
pub(super) struct TranscribeQuery {
    pub(super) model: Option<String>,
    pub(super) language: Option<String>,
    pub(super) sample_rate: Option<u32>,
    pub(super) ttl_seconds: Option<u64>,
}

/// `api-types` message → worker message (`REQ-3`): the tool role,
/// `tool_call_id` and assistant `tool_calls` round-trip; missing/null
/// content becomes the empty string (assistant-with-tool_calls shape).
/// Media marker the projector's tokenizer splices embeddings at (the
/// llama.cpp mtmd default, `MtmdContext::default_marker()`). The server
/// splices it into message text at each `image_url` part's flattened-text
/// offset before the worker call, so image embeddings land where the
/// client put the parts.
pub(super) const MEDIA_MARKER: &str = "<__media__>";

/// Decode one `image_url` payload: a `data:<mime>;base64,<payload>` URI.
/// The bytes stay encoded (JPEG/PNG/…) — the projector's bitmap loader
/// sniffs the container format. A non-base64 `data:` URI cannot carry
/// image bytes; anything else is the [`image_data_rejection`] case.
pub(super) fn decode_image_url(url: &str) -> Result<Vec<u8>, String> {
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(image_data_rejection(url).unwrap_or_else(|| "invalid image_url".to_owned()));
    };
    let Some((meta, payload)) = rest.split_once(',') else {
        return Err(format!(
            "malformed data: URI {url:?}: missing `,payload` section"
        ));
    };
    if meta.rsplit(';').next().is_none_or(|flag| flag != "base64") {
        return Err(format!(
            "unsupported data: URI {url:?}: image payloads must be base64 (`data:<mime>;base64,...`)"
        ));
    }
    // Standard with padding is the norm; tolerate the no-pad and URL-safe
    // variants rather than teaching clients the alphabet rules.
    for engine in [
        &base64::engine::general_purpose::STANDARD,
        &base64::engine::general_purpose::STANDARD_NO_PAD,
        &base64::engine::general_purpose::URL_SAFE,
        &base64::engine::general_purpose::URL_SAFE_NO_PAD,
    ] {
        if let Ok(bytes) = base64::engine::Engine::decode(engine, payload.trim()) {
            if bytes.is_empty() {
                break;
            }
            return Ok(bytes);
        }
    }
    Err("invalid image data URL: base64 payload does not decode (or is empty)".to_owned())
}

/// One API message → worker message. Vision messages get the media
/// marker spliced at their part offsets and the `data:` payloads decoded
/// to bytes; a bad payload is a client error the routes map to 400.
pub(super) fn worker_message(message: &ChatMessage) -> Result<WorkerChatMessage, String> {
    let role = match message.role {
        ChatRole::User => Role::User,
        ChatRole::Assistant => Role::Assistant,
        ChatRole::System => Role::System,
        ChatRole::Tool => Role::Tool,
    };
    let content = if message.content.images.is_empty() {
        message.content.text.clone().unwrap_or_default()
    } else {
        message.content.text_with_media_markers(MEDIA_MARKER)
    };
    let mut images = Vec::with_capacity(message.content.images.len());
    for image in &message.content.images {
        images.push(WorkerChatImage::new(decode_image_url(&image.url)?));
    }
    Ok(WorkerChatMessage {
        role,
        content,
        tool_call_id: message.tool_call_id.clone(),
        tool_calls: message
            .tool_calls
            .iter()
            .flatten()
            .map(|tc| ToolCallSpec {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                arguments: tc.function.arguments.clone(),
            })
            .collect(),
        images,
    })
}

/// All API messages → worker messages (vision decoding can fail per
/// message).
pub(super) fn worker_messages(messages: &[ChatMessage]) -> Result<Vec<WorkerChatMessage>, String> {
    messages.iter().map(worker_message).collect()
}

/// Chat-capability refusal message (shared by the completions envelope and
/// the native plain-text surface).
pub(super) fn chat_unsupported_message(model: &str) -> String {
    format!(
        "model {model:?} is not chat-capable: its catalog entry declares capabilities.chat = false (a mmproj projector file or non-chat weights cannot serve conversation turns)"
    )
}

/// Worker request skeleton shared by both routes: constraint fields absent
/// (`REQ-3` fields ride only on `/v1/chat/completions`), default sampler.
/// Message conversion (vision decoding) can fail per message — the caller
/// maps that to its surface's 400.
pub(super) fn worker_request_base(
    slug: String,
    messages: Result<Vec<WorkerChatMessage>, String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<WorkerChatRequest, String> {
    Ok(WorkerChatRequest {
        model: slug,
        messages: messages?,
        max_tokens,
        temperature,
        seed: None,
        grammar: None,
        json_schema: None,
        json_object: false,
        tools_json: None,
        tool_choice_none: false,
        sampler: SamplerParams::default(),
    })
}

/// `X-Job-Id` response-header name (NEW-3): every GPU route stamps its
/// in-flight registry id. On SSE routes it precedes the stream body (a
/// client can cancel the stream it is reading); on sync routes it arrives
/// with the completed response, so cancel-discovery goes through
/// `GET /v1/jobs`.
pub(super) const JOB_ID_HEADER: header::HeaderName = header::HeaderName::from_static("x-job-id");

/// Attach the job id header to a built response.
pub(super) fn with_job_id(mut response: Response, job_id: &str) -> Response {
    if let Ok(value) = HeaderValue::from_str(job_id) {
        response.headers_mut().insert(JOB_ID_HEADER, value);
    }
    response
}

/// Admission first, registration second: a rejected GPU request must not
/// enter the registry (no `job_registered`/`job_finished` events and no
/// `X-Job-Id` on a request that never ran). Every admitted route goes
/// through this one seam so the ordering cannot drift between them.
pub(super) fn admit_then_register(
    registry: &crate::jobs_registry::Registry,
    acquired: Result<crate::admission::GpuLease, (StatusCode, String)>,
    kind: crate::jobs_registry::JobKind,
    model: Option<String>,
    interruptible: bool,
) -> Result<(crate::admission::GpuLease, crate::jobs_registry::JobGuard), (StatusCode, String)> {
    let lease = acquired?;
    let job = registry.register(kind, model, interruptible);
    Ok((lease, job))
}

/// REQ-8: image content on a model without a projector. Vision capability
/// is catalog-derived: a sibling mmproj projector next to the weights, or
/// an explicit operator `capabilities.vision = true`.
pub(super) fn vision_unsupported_message(model: &str) -> String {
    format!(
        "model {model:?} is not vision-capable: image_url parts need an mmproj projector next to the model weights (or an operator capabilities.vision override); see GET /v1/models modalities"
    )
}

/// Only `data:` URI image payloads are accepted: the server decodes the
/// base64 inline and hands the bytes to the projector's own image loader.
/// Remote URLs would turn the server into an outbound fetcher (SSRF
/// surface) — an explicit rejection, never a silent skip or a fetch.
pub(super) fn image_data_rejection(url: &str) -> Option<String> {
    (!url.starts_with("data:")).then(|| {
        format!(
            "unsupported image_url {url:?}: only data: URIs are accepted (remote URLs are not fetched)"
        )
    })
}

/// Vision gate over chat requests (all three chat routes): images in the
/// conversation need (a) a vision-capable model and (b) `data:` URI
/// payloads. `entry` is the resolved catalog entry (`None` — e.g. the raw
/// startup-configured model — is treated as not vision-capable). `None`
/// passes; `Some((message, code))` is the 400 payload (`code` names the
/// error on the `OpenAI` surface). The worker re-gates (defense in depth).
pub(super) fn vision_rejection(
    entry: Option<&ModelCatalogEntry>,
    messages: &[ChatMessage],
) -> Option<(String, &'static str)> {
    if !messages.iter().any(|message| message.content.has_image) {
        return None;
    }
    let model = entry.map_or_else(|| "?".to_owned(), |entry| entry.id.clone());
    if !entry.is_some_and(|entry| entry.capabilities.vision) {
        return Some((vision_unsupported_message(&model), "vision_unsupported"));
    }
    for message in messages {
        for image in &message.content.images {
            if let Some(rejection) = image_data_rejection(&image.url) {
                return Some((rejection, "image_url_unsupported"));
            }
        }
    }
    None
}

/// Chat-capability gate (completions surface): a catalog entry advertising
/// `capabilities.chat = false` — a projector `mmproj*.gguf` file, an
/// operator-declared embedding, or any non-chat weights — is never silently
/// run through the chat pipeline (garbage tokens dressed up as a
/// completion). Mirrors the REQ-4 embeddings gate; `None` entry (the raw
/// startup-configured model) passes, the operator declared it by hand.
pub(super) fn chat_capable_gate(entry: &ModelCatalogEntry) -> Option<Response> {
    (!entry.capabilities.chat).then(|| {
        openai_error_response(
            StatusCode::BAD_REQUEST,
            chat_unsupported_message(&entry.id),
            Some("chat_unsupported"),
        )
    })
}

/// Request gates before any pipeline work (`/v1/chat/completions`): REQ-4
/// — an embeddings-capable model is never silently run through the chat
/// pipeline (the reply would be garbage tokens dressed up as a
/// completion); chat-capability (a projector file or operator-declared
/// non-chat weights) is refused by name for the same reason; REQ-8 — an
/// `image_url` content part is gated on model vision capability and
/// `data:` URI payloads, an explicit 400 `OpenAI` envelope either way.
/// `Some(response)` is the ready-made rejection.
pub(super) fn completions_request_gates(
    entry: &ModelCatalogEntry,
    model: &str,
    messages: &[ChatMessage],
) -> Option<Response> {
    embeddings_model_gate(entry, model)
        .or_else(|| chat_capable_gate(entry))
        .or_else(|| {
            vision_rejection(Some(entry), messages).map(|(message, code)| {
                openai_error_response(StatusCode::BAD_REQUEST, message, Some(code))
            })
        })
}

/// Clamp over-cap per-request `ttl_seconds` advice down to the configured
/// cap (`[swap] max_ttl_secs`) with a `warn!` — never a 400: third-party
/// clients send arbitrary values and the clamp is exactly as safe as a
/// rejection would be. `None` passes through (no advice); the swap manager
/// re-caps defensively when it applies the budget.
pub(super) fn clamp_ttl_seconds(max: Duration, value: Option<u64>) -> Option<u64> {
    let secs = value?;
    let cap = max.as_secs();
    if secs > cap {
        tracing::warn!(
            requested = secs,
            cap,
            "ttl_seconds above [swap] max_ttl_secs; clamping (never rejected)"
        );
        return Some(cap);
    }
    Some(secs)
}

/// Map a swap error onto the HTTP contract: an unknown / non-LLM slug is a
/// 400 (same semantics as `validate_model`), a failed construction is a 500
/// naming the cause (the cache stays consistent — nothing was inserted).
pub(super) fn swap_rejection(err: crate::swap::SwapError) -> (StatusCode, String) {
    match err {
        crate::swap::SwapError::UnknownSlug(message) => (StatusCode::BAD_REQUEST, message),
        crate::swap::SwapError::Load(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
    }
}

/// Resolve the chat pipeline for a request: through the swap manager when
/// active (the clamped per-request `ttl_seconds` rides along as the entry's
/// window budget), otherwise the startup-configured static pipeline.
pub(super) async fn chat_pipeline(
    config: &AppConfig,
    model: &str,
    ttl_seconds: Option<u64>,
) -> Result<Arc<dyn LlmPipeline>, (StatusCode, String)> {
    match &config.swap {
        Some(swap) => swap.get(model, ttl_seconds).await.map_err(swap_rejection),
        None => Ok(config.llm.clone()),
    }
}
