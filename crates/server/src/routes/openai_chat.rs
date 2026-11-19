//! `OpenAI`-compatible chat completion route, extracted from `routes.rs`.
//!
//! The router registration and the `OpenAPI` path stay in `routes.rs`; only
//! the handler and its helpers live here.

use std::sync::Arc;

use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::{
    ChatContent, ChatMessage, ChatRole, FinishReason, FunctionCall, ModelCatalog,
    ModelCatalogEntry, OpenAiChatCompletion, OpenAiChatRequest, OpenAiChoice, OpenAiError,
    OpenAiUsage, ToolCall, ToolCallType,
};
use honkbox_llm_worker::{ChatRequest as WorkerChatRequest, LlmError, LlmPipeline};

use super::AppConfig;
use super::chat_prep::{
    admit_then_register, clamp_ttl_seconds, completions_request_gates, with_job_id, worker_messages,
};
use super::openai_constraints::{openai_error_response, resolve_constraints};
use super::openai_stream::{completions_stream, id_frame};

/// Owning catalog entry for a request `model` value (the alias `id` or the
/// registry `slug`); [`super::resolve_model_slug`] reads the slug off it.
pub(super) fn resolve_model_entry<'a>(
    model: &str,
    catalog: &'a ModelCatalog,
) -> Option<&'a ModelCatalogEntry> {
    catalog
        .data
        .iter()
        .find(|e| e.id == model || e.slug == model)
}

/// Unix seconds (the `OpenAI` `created` field).
pub(super) fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `finish_reason` for a completed reply. The worker does not surface a
/// native stop cause, so `length` is the documented heuristic: reported
/// generated tokens reached the requested `max_tokens` cap.
pub(super) fn finish_reason(tokens: u32, max_tokens: Option<u32>) -> FinishReason {
    if max_tokens.is_some_and(|max| tokens >= max) {
        FinishReason::Length
    } else {
        FinishReason::Stop
    }
}

/// Token accounting from a worker reply.
pub(super) fn usage_of(reply: &honkbox_llm_worker::ChatResponse) -> OpenAiUsage {
    OpenAiUsage {
        prompt_tokens: reply.prompt_tokens,
        completion_tokens: reply.tokens,
        total_tokens: u64::from(reply.prompt_tokens) + u64::from(reply.tokens),
    }
}

/// `POST /v1/chat/completions`: the `OpenAI`-compatible translation surface
/// (REQ-1) over the SAME pipeline, GPU lease, and swap decision as
/// `/v1/chat` — never an HTTP proxy. JSON body or `OpenAI` SSE
/// (`stream: true`). Errors are `OpenAI` envelopes; statuses: 400 bad
/// request, 404 unknown model, 429 busy (+`Retry-After` via the shared
/// middleware), 500 inference failure. Undeclared request keys (`top_p`,
/// `stop`, …) are accepted and ignored by serde.
///
/// An omitted/empty `model` is a 400 naming the requirement — the native
/// route's empty-means-server-default fallback stays native-only.
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    request_body = OpenAiChatRequest,
    responses(
        (status = 200, description = "Chat completion (JSON or SSE)", body = OpenAiChatCompletion),
        (status = 400, description = "Bad request", body = OpenAiError),
        (status = 404, description = "Unknown model", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Inference failed", body = OpenAiError)
    )
)]
// ~105 lines: validation gates, constraint resolution, the swap decision,
// and the two wire shapes (JSON + SSE) share one handler; splitting the
// dispatch would break the REQ-6/NEW-3 ordering story (register, queue,
// branch on the stream flag) across two functions.
#[allow(clippy::too_many_lines)]
pub(super) async fn post_chat_completions(
    State(config): State<AppConfig>,
    body: Bytes,
) -> Response {
    let req: OpenAiChatRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(err) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid JSON body: {err}"),
                None,
            );
        }
    };
    let Some(model) = req
        .model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
    else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "model is required: pass an operator alias or registry slug from GET /v1/models"
                .to_owned(),
            Some("model_required"),
        );
    };
    let catalog = config.catalog_snapshot();
    let Some(entry) = resolve_model_entry(model, &catalog) else {
        return openai_error_response(
            StatusCode::NOT_FOUND,
            format!("unknown model {model:?}: not an alias or registry slug (see GET /v1/models)"),
            Some("model_not_found"),
        );
    };
    let served_id = entry.id.clone();
    let slug = entry.slug.clone();
    // REQ-4 + REQ-8 request gates (see [`completions_request_gates`]).
    if let Some(response) = completions_request_gates(entry, model, &req.messages) {
        return response;
    }
    // `REQ-3` request surface: resolve tools + output constraints + sampler
    // knobs into the worker protocol (clamps and 400s are inside; nothing
    // is silently dropped).
    let constraints = match resolve_constraints(&req) {
        Ok(constraints) => constraints,
        Err((code, message)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
    };
    let worker_messages = match worker_messages(&req.messages) {
        Ok(messages) => messages,
        Err(message) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                message,
                Some("invalid_image_data"),
            );
        }
    };
    let worker_req = WorkerChatRequest {
        model: slug.clone(),
        messages: worker_messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        seed: req.seed,
        grammar: constraints.grammar,
        json_schema: constraints.json_schema,
        json_object: constraints.json_object,
        tools_json: constraints.tools_json,
        tool_choice_none: constraints.tool_choice_none,
        sampler: constraints.sampler,
    };
    // Same swap path as `/v1/chat`: load-on-demand through the swap manager
    // when active, else the startup pipeline. Unknown/non-LLM slugs are the
    // route's 404; construction failures 500.
    let pipe = match &config.swap {
        Some(swap) => {
            match swap
                .get(
                    &slug,
                    clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds),
                )
                .await
            {
                Ok(pipe) => pipe,
                Err(crate::swap::SwapError::UnknownSlug(message)) => {
                    return openai_error_response(StatusCode::NOT_FOUND, message, None);
                }
                Err(crate::swap::SwapError::Load(message)) => {
                    return openai_error_response(StatusCode::INTERNAL_SERVER_ERROR, message, None);
                }
            }
        }
        None => config.llm.clone(),
    };
    // NEW-3 + REQ-6: take the GPU slot ONCE through the admission queue for
    // either shape, then register the job (interruptible only on the
    // streaming path — the sync completion is one blocking FFI call). A
    // rejection keeps the OpenAI envelope + shared `Retry-After` hint but
    // creates no job, events or X-Job-Id.
    let streaming = req.stream.is_some_and(|stream| stream);
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await,
        crate::jobs_registry::JobKind::Chat,
        Some(slug.clone()),
        streaming,
    ) {
        Ok(pair) => pair,
        Err((status, message)) => {
            return openai_error_response(status, message, Some("gpu_slot_busy"));
        }
    };
    if streaming {
        // REQ-6 (Option B): the slot is held BEFORE the SSE body is
        // returned, so a busy slot is a real 429 — never a 200 whose only
        // content is an error frame (loop-lore's `data:`-only parser would
        // read that as a clean empty completion).
        completions_stream(
            lease,
            job,
            pipe,
            worker_req,
            &served_id,
            id_frame(),
            req.max_tokens,
        )
    } else {
        completions_json(lease, job, pipe, worker_req, &served_id, req.max_tokens).await
    }
}

/// Non-streaming completion: one blocking worker call inside the GPU lease,
/// mapped onto the exact `OpenAI` JSON shape.
pub(super) async fn completions_json(
    lease: crate::admission::GpuLease,
    job: crate::jobs_registry::JobGuard,
    pipe: Arc<dyn LlmPipeline>,
    worker_req: WorkerChatRequest,
    served_id: &str,
    max_tokens: Option<u32>,
) -> Response {
    let reply = tokio::task::spawn_blocking(move || {
        let _lease = lease;
        pipe.complete(&worker_req).map_err(|err| {
            // `REQ-3`: unusable constraints (unparseable grammar, non-
            // convertible schema) and `REQ-8`: image content the loaded
            // pipeline cannot serve (worker-side vision gate, defense in
            // depth behind the route gate) are client errors, not server
            // ones; the envelope code names the class.
            match &err {
                LlmError::InvalidRequest(_) => (
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    format!("chat failed: {err}"),
                ),
                LlmError::Vision(_) => (
                    StatusCode::BAD_REQUEST,
                    "vision_unsupported",
                    format!("chat failed: {err}"),
                ),
                _ => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal",
                    format!("chat failed: {err}"),
                ),
            }
        })
    })
    .await
    .map_err(|err| format!("chat task failed: {err}"));
    let reply = match reply {
        Ok(Ok(reply)) => reply,
        Ok(Err((StatusCode::BAD_REQUEST, code, message))) => {
            return openai_error_response(StatusCode::BAD_REQUEST, message, Some(code));
        }
        Ok(Err((_, _, message))) | Err(message) => {
            return openai_error_response(StatusCode::INTERNAL_SERVER_ERROR, message, None);
        }
    };
    let usage = usage_of(&reply);
    let tool_calls = reply
        .tool_calls
        .iter()
        .map(|call| ToolCall {
            // Unique per response (REQ-3 acceptance); echoed back via
            // `tool_call_id` on the follow-up `role: "tool"` message.
            id: format!("call_{}", ulid::Ulid::new()),
            call_type: ToolCallType::Function,
            function: FunctionCall {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            },
        })
        .collect::<Vec<_>>();
    let finish = if tool_calls.is_empty() {
        finish_reason(reply.tokens, max_tokens)
    } else {
        FinishReason::ToolCalls
    };
    let content = if tool_calls.is_empty() || !reply.text.is_empty() {
        ChatContent::text(reply.text)
    } else {
        // Tool-call-only reply: `OpenAI` sends `content: null` there.
        ChatContent::default()
    };
    with_job_id(
        Json(OpenAiChatCompletion {
            id: format!("chatcmpl-{}", ulid::Ulid::new()),
            object: "chat.completion".to_owned(),
            created: epoch_secs(),
            model: served_id.to_owned(),
            choices: vec![OpenAiChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content,
                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                    tool_call_id: None,
                },
                finish_reason: finish,
            }],
            usage,
        })
        .into_response(),
        job.id(),
    )
}
