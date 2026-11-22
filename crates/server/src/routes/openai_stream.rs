//! `OpenAI` SSE streaming translation: chunk framing, tool-call deltas,
//! finish reasons and the `[DONE]` sentinel.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use axum::response::{
    IntoResponse, Response,
    sse::{Event, Sse},
};
use honkbox_api_types::{
    ChatRole, DeltaFunction, DeltaToolCall, FinishReason, OpenAiChunk, OpenAiChunkChoice,
    OpenAiDelta, OpenAiError, ToolCallType,
};
use honkbox_llm_worker::{
    ChatRequest as WorkerChatRequest, LlmPipeline, RequestCancel, ToolCallOut,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::chat::{CancelOnDrop, free_gpu_lease};
use super::chat_prep::with_job_id;
use super::openai_chat::{epoch_secs, finish_reason, usage_of};
use super::openai_constraints::llm_error_class;

/// Fresh completion id + creation timestamp (shared by every SSE frame).
pub(super) fn id_frame() -> (String, u64) {
    (format!("chatcmpl-{}", ulid::Ulid::new()), epoch_secs())
}

/// One SSE chunk frame (serialization never fails on these DTOs).
pub(super) fn chunk_frame(frame: &OpenAiChunk) -> Event {
    let data = serde_json::to_string(frame).unwrap_or_else(|_| "{}".to_owned());
    Event::default().data(data)
}

/// `OpenAI` SSE translation of `complete_stream` (REQ-1): role+first-token
/// frame, one content frame per token, a final frame with `finish_reason`
/// success path — inference failures are one terminal `data:` frame with
/// the `OpenAI` error envelope and no `[DONE]` after it. loop-lore's SSE
/// parser ignores `event:` lines, so framing an error that way would drop
/// it. The GPU-slot busy condition is NOT one of them:
/// REQ-6 (Option B) reserves the slot in the handler, so a busy slot
/// becomes a real HTTP 429 before this function runs — the `lease` here is
/// already held and handed to the worker.
// 105 lines: the SSE translation (chunks, tool-call deltas, finish reasons,
// [DONE]) is one linear pipeline; splitting it hurts readability.
#[allow(clippy::too_many_lines)]
pub(super) fn completions_stream(
    lease: crate::admission::GpuLease,
    job: crate::jobs_registry::JobGuard,
    pipe: Arc<dyn LlmPipeline>,
    worker_req: WorkerChatRequest,
    served_id: &str,
    (id, created): (String, u64),
    max_tokens: Option<u32>,
) -> Response {
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancel = RequestCancel::new();
    // Normal end must not abort the shared pipeline when the body drops.
    let done = Arc::new(AtomicBool::new(false));
    let lease_slot = Arc::new(Mutex::new(None::<crate::admission::GpuLease>));
    let abort_pipe = Arc::clone(&pipe);
    let worker_cancel = cancel.clone();
    let body_cancel = cancel.clone();
    let worker_cancelled = cancelled.clone();
    let worker_done = done.clone();
    let job_token = job.token().clone();
    let job_id = job.id().to_owned();
    let frame_model = served_id.to_owned();
    // `REQ-3`: with tools on the table, pieces are buffered until the
    // parsed tool calls are known, then framed as `delta.tool_calls`
    // fragments (arguments split across frames; per-index concatenation
    // yields the JSON string). Without tools, content streams
    // token-by-token as before — including `json_object` replies.
    let tool_streaming = worker_req.tools_json.is_some();
    let worker = tokio::task::spawn_blocking(move || {
        // The job guard rides with the worker (the registry entry lives as
        // long as the stream). The slot was taken in the handler (REQ-6);
        // park the lease where the token sink can drain it at the first
        // cancel/disconnect.
        let _job = job;
        *lease_slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(lease);
        let mut first = true;
        let mut buffered: Vec<String> = Vec::new();
        let result = pipe.complete_stream_cancellable(
            &worker_req,
            &mut |token: &str| {
                if worker_cancelled.load(Ordering::Relaxed) || job_token.is_cancelled() {
                    // Disconnect, or `POST /v1/jobs/{id}/cancel` (NEW-3): abort
                    // the FFI loop itself, not just delivery.
                    worker_cancel.cancel();
                    pipe.abort();
                    free_gpu_lease(&lease_slot);
                    return;
                }
                if tool_streaming {
                    buffered.push(token.to_owned());
                    return;
                }
                let delivered =
                    send_token_delta(&tx, &id, created, &frame_model, &mut first, token);
                if !delivered {
                    // Receiver gone without the drop guard (drop raced the send).
                    worker_cancelled.store(true, Ordering::Relaxed);
                    worker_cancel.cancel();
                    pipe.abort();
                    free_gpu_lease(&lease_slot);
                }
            },
            &worker_cancel,
        );
        // Mark the run finished before sending terminal frames: the SSE
        // body can drop the instant `[DONE]` is consumed.
        worker_done.store(true, Ordering::Relaxed);
        // A disconnect between the last token and completion frees here.
        free_gpu_lease(&lease_slot);
        let mut failed = false;
        match result {
            Ok(reply) => {
                let usage = usage_of(&reply);
                let finish = if reply.tool_calls.is_empty() {
                    finish_reason(reply.tokens, max_tokens)
                } else {
                    FinishReason::ToolCalls
                };
                if tool_streaming {
                    if reply.tool_calls.is_empty() {
                        // No call was made (e.g. `tool_choice: "none"`, or
                        // the model answered in prose): the buffered pieces
                        // stream as content now.
                        emit_content_stream(&tx, &id, created, &frame_model, &buffered);
                    } else {
                        // Tool-call reply: a role marker frame, then the
                        // parsed calls as fragments. The buffered raw
                        // pieces are superseded by the parser output.
                        emit_tool_stream(&tx, &id, created, &frame_model, &reply.tool_calls);
                    }
                }
                let final_frame = OpenAiChunk {
                    id,
                    object: "chat.completion.chunk".to_owned(),
                    created,
                    model: frame_model,
                    choices: vec![OpenAiChunkChoice {
                        index: 0,
                        delta: OpenAiDelta {
                            role: None,
                            content: None,
                            tool_calls: None,
                        },
                        finish_reason: Some(finish),
                    }],
                    usage: Some(usage),
                };
                let _ = tx.send(Ok(chunk_frame(&final_frame)));
            }
            Err(err) => {
                failed = true;
                let (status, code, message) = llm_error_class(&err);
                let body = OpenAiError::new(status.as_u16(), message, Some(code.to_owned()));
                let _ = tx.send(Ok(
                    Event::default().data(serde_json::to_string(&body).unwrap_or_default())
                ));
            }
        }
        if !failed {
            let _ = tx.send(Ok(Event::default().data("[DONE]")));
        }
    });
    with_job_id(
        Sse::new(CancelOnDrop {
            inner: UnboundedReceiverStream::new(rx),
            cancelled,
            done,
            pipe: abort_pipe,
            cancel: body_cancel,
            _worker: worker,
        })
        .into_response(),
        &job_id,
    )
}

/// Emit a buffered tools stream: a role marker frame, then the parsed
/// calls as `delta.tool_calls` fragments with fresh per-response ids.
pub(super) fn emit_tool_stream(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    id: &str,
    created: u64,
    model: &str,
    tool_calls: &[ToolCallOut],
) {
    send_delta(
        tx,
        id,
        created,
        model,
        OpenAiDelta {
            role: Some(ChatRole::Assistant),
            content: None,
            tool_calls: None,
        },
    );
    let ids: Vec<String> = tool_calls
        .iter()
        .map(|_| format!("call_{}", ulid::Ulid::new()))
        .collect();
    for delta in tool_delta_frames(tool_calls, &ids) {
        send_delta(tx, id, created, model, delta);
    }
}

/// Emit buffered pieces as a plain content stream (the first piece carries
/// the role marker).
pub(super) fn emit_content_stream(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    id: &str,
    created: u64,
    model: &str,
    pieces: &[String],
) {
    for (piece_index, piece) in pieces.iter().enumerate() {
        let delta = if piece_index == 0 {
            OpenAiDelta {
                role: Some(ChatRole::Assistant),
                content: Some(piece.clone()),
                tool_calls: None,
            }
        } else {
            OpenAiDelta {
                role: None,
                content: Some(piece.clone()),
                tool_calls: None,
            }
        };
        send_delta(tx, id, created, model, delta);
    }
}

/// Send one streamed content token (the role marker rides the first
/// piece). Returns `false` when the receiver is gone — the caller then
/// cancels the generation.
#[allow(clippy::needless_pass_by_ref_mut)] // `first` is mutated across calls
pub(super) fn send_token_delta(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    id: &str,
    created: u64,
    model: &str,
    first: &mut bool,
    token: &str,
) -> bool {
    let delta = if *first {
        *first = false;
        OpenAiDelta {
            role: Some(ChatRole::Assistant),
            content: Some(token.to_owned()),
            tool_calls: None,
        }
    } else {
        OpenAiDelta {
            role: None,
            content: Some(token.to_owned()),
            tool_calls: None,
        }
    };
    let frame = OpenAiChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created,
        model: model.to_owned(),
        choices: vec![OpenAiChunkChoice {
            index: 0,
            delta,
            finish_reason: None,
        }],
        usage: None,
    };
    tx.send(Ok(chunk_frame(&frame))).is_ok()
}

/// Send one content/tool delta chunk (serialization never fails on these
/// DTOs; a gone receiver just drops the frame post-completion).
pub(super) fn send_delta(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    id: &str,
    created: u64,
    model: &str,
    delta: OpenAiDelta,
) {
    let frame = OpenAiChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created,
        model: model.to_owned(),
        choices: vec![OpenAiChunkChoice {
            index: 0,
            delta,
            finish_reason: None,
        }],
        usage: None,
    };
    let _ = tx.send(Ok(chunk_frame(&frame)));
}

/// Split an arguments JSON string into frame-sized pieces (char-boundary
/// safe). Short payloads still split in half, so the streamed form always
/// exercises the fragment protocol (per-index concatenation reproduces the
/// original string).
pub(super) fn argument_frames(arguments: &str) -> Vec<String> {
    let chars: Vec<char> = arguments.chars().collect();
    if chars.len() < 2 {
        return vec![arguments.to_owned()];
    }
    let per_frame = chars.len().div_ceil(4).max(2);
    chars
        .chunks(per_frame)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

/// `delta.tool_calls` frames for parsed calls (`REQ-3`): per call a first
/// fragment carrying `id`/`type`/`function.name` + empty `arguments`, then
/// continuation fragments with `function.arguments` pieces at the same
/// `index`. `ids` runs parallel to `tool_calls`.
pub(super) fn tool_delta_frames(tool_calls: &[ToolCallOut], ids: &[String]) -> Vec<OpenAiDelta> {
    let mut deltas = Vec::new();
    for (index, (call, call_id)) in tool_calls.iter().zip(ids).enumerate() {
        let index = u32::try_from(index).unwrap_or(u32::MAX);
        deltas.push(OpenAiDelta {
            role: None,
            content: None,
            tool_calls: Some(vec![DeltaToolCall {
                index,
                id: Some(call_id.clone()),
                call_type: Some(ToolCallType::Function),
                function: Some(DeltaFunction {
                    name: Some(call.name.clone()),
                    arguments: Some(String::new()),
                }),
            }]),
        });
        for piece in argument_frames(&call.arguments) {
            deltas.push(OpenAiDelta {
                role: None,
                content: None,
                tool_calls: Some(vec![DeltaToolCall {
                    index,
                    id: None,
                    call_type: None,
                    function: Some(DeltaFunction {
                        name: None,
                        arguments: Some(piece),
                    }),
                }]),
            });
        }
    }
    deltas
}
