//! Native chat routes: `POST /v1/chat` and `POST /v1/chat/stream`.
//!
//! Extracted from `routes.rs`; the router registrations and the `OpenAPI`
//! coverage gate stay there.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::Event, sse::Sse},
};
use honkbox_api_types::{ChatRequest, ChatResponse};
use honkbox_llm_worker::{LlmPipeline, RequestCancel};
use tokio::sync::mpsc;
use tokio_stream::{Stream, wrappers::UnboundedReceiverStream};

use super::AppConfig;
use super::chat_prep::{
    admit_then_register, chat_pipeline, chat_unsupported_message, clamp_ttl_seconds,
    vision_rejection, with_job_id, worker_messages, worker_request_base,
};

/// `POST /v1/chat`: sync worker call inside `spawn_blocking`; the residency
/// lease is held across the call. Only the server edge is async.
#[utoipa::path(
    post,
    path = "/v1/chat",
    request_body = ChatRequest,
    responses(
        (status = 200, description = "Chat completion", body = ChatResponse),
        (status = 400, description = "Unknown model slug"),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Inference failed")
    )
)]
pub(super) async fn post_chat(
    State(config): State<AppConfig>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, (StatusCode, String)> {
    let catalog = config.catalog_snapshot();
    super::validate_model(Some(&req.model), config.llm_model.as_deref(), &catalog)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    // Aliases resolve to their owning registry slug; the swap manager and
    // the worker both key on the slug (an unresolved value is the
    // startup-configured model, same as before aliases existed).
    let slug = super::resolve_model_slug(&req.model, &catalog)
        .map_or_else(|| req.model.clone(), str::to_owned);
    // Chat-capability parity with the completions gate: a catalog entry
    // declaring `chat = false` (a projector `mmproj*.gguf`, a declared
    // embedding) is refused by name; the raw startup-configured model (no
    // catalog entry) passes — the operator declared it by hand.
    if let Some(entry) = super::resolve_model_entry(&slug, &catalog)
        && !entry.capabilities.chat
    {
        return Err((StatusCode::BAD_REQUEST, chat_unsupported_message(&slug)));
    }
    // REQ-8 parity with `/v1/chat/completions`: image content is gated on
    // model vision capability (sibling mmproj or operator override) and
    // `data:` URI payloads — a plain-text 400 on the native route, never a
    // silently flattened image.
    if let Some((message, _)) =
        vision_rejection(super::resolve_model_entry(&slug, &catalog), &req.messages)
    {
        return Err((StatusCode::BAD_REQUEST, message));
    }
    let pipe = chat_pipeline(
        &config,
        &slug,
        clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds),
    )
    .await?;
    // NEW-3: take the GPU slot first, then register the in-flight job. A
    // rejected request never enters the registry (no lifecycle events, no
    // X-Job-Id), and the admission error is returned explicitly rather
    // than with `?`. The blocking FFI call itself is not interruptible —
    // the cancel token is marked but never checked here.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await,
        crate::jobs_registry::JobKind::Chat,
        Some(slug.clone()),
        false,
    ) {
        Ok(pair) => pair,
        Err(error) => return Err(error),
    };
    let worker_req = worker_request_base(
        slug,
        worker_messages(&req.messages),
        req.max_tokens,
        req.temperature,
    )
    .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
    let reply = tokio::task::spawn_blocking(move || {
        let _lease = lease;
        pipe.complete(&worker_req).map_err(|err| {
            let (status, _, message) = super::openai_constraints::llm_error_class(&err);
            (status, message)
        })
    })
    .await
    .map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("chat task failed: {err}"),
        )
    })??;
    Ok(with_job_id(
        Json(ChatResponse {
            backend_id: reply.backend_id,
            text: reply.text,
            tokens: reply.tokens,
            millis: reply.millis,
        })
        .into_response(),
        job.id(),
    ))
}

/// Drop the GPU residency lease immediately (the disconnect path): the slot
/// frees at the token boundary while `abort()` stops the FFI loop one token
/// later. Taking the [`crate::admission::GpuLease`] out of the slot drops
/// it, which releases residency AND pokes the admission queue so a queued
/// waiter wakes now instead of at its deadline.
pub(super) fn free_gpu_lease(slot: &Mutex<Option<crate::admission::GpuLease>>) {
    slot.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
}

/// Bridge a registry job cancel into the per-request `RequestCancel` flag.
/// Without this watcher the downstream reader only notices a cancel at the
/// next token boundary, so a stalled upstream keeps the socket and GPU lease
/// until its own timeout. The watcher is aborted when the SSE body drops.
pub(super) fn watch_job_cancel(
    token: tokio_util::sync::CancellationToken,
    cancel: RequestCancel,
) -> tokio::task::AbortHandle {
    tokio::spawn(async move {
        token.cancelled().await;
        cancel.cancel();
    })
    .abort_handle()
}

/// SSE body wrapper: dropping the body (client disconnect, or the stream
/// ending normally) marks the request cancelled and calls the pipeline's
/// `abort` hook so the synchronous FFI generation stops at its next token
/// step; the worker `JoinHandle` lives exactly as long as the body it feeds.
pub(super) struct CancelOnDrop<S> {
    /// Wrapped token stream.
    pub(super) inner: S,
    /// Set on drop; checked by the worker at each token boundary.
    pub(super) cancelled: Arc<AtomicBool>,
    /// Set when the worker emitted its terminal frame.
    pub(super) done: Arc<AtomicBool>,
    /// Pipeline `abort` hook (atomics-only state: cannot deadlock with the
    /// blocking worker, which holds no lock across `complete_stream`).
    pub(super) pipe: Arc<dyn LlmPipeline>,
    /// Per-request cancel token: only this stream's reader is interrupted.
    pub(super) cancel: RequestCancel,
    /// Registry cancel watcher; aborted when the body ends for any reason.
    pub(super) job_cancel_watcher: Option<tokio::task::AbortHandle>,
    /// Blocking worker task feeding `inner`.
    pub(super) _worker: tokio::task::JoinHandle<()>,
}

impl<S: Stream + Unpin> Stream for CancelOnDrop<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> Drop for CancelOnDrop<S> {
    fn drop(&mut self) {
        if let Some(watcher) = &self.job_cancel_watcher {
            watcher.abort();
        }
        if self.done.load(Ordering::Relaxed) {
            return;
        }
        self.cancelled.store(true, Ordering::Relaxed);
        self.cancel.cancel();
        self.pipe.abort();
    }
}

/// `POST /v1/chat/stream`: same blocking call, tokens forwarded as SSE
/// (`data: {"token":"..."}` + `data: [DONE]`) over an unbounded channel so
/// the sink never blocks the worker thread.
///
/// Disconnect handling: dropping the SSE body sets a cancel flag and calls
/// the pipeline `abort` hook; the worker notices at its **next token
/// boundary** — token delivery stops, the GPU residency lease is freed, and
/// the synchronous FFI loop exits at its next internal checkpoint (per-token
/// decode returns to our loop, so early return needs no FFI-level abort).
#[utoipa::path(
    post,
    path = "/v1/chat/stream",
    request_body = ChatRequest,
    responses(
        (status = 200, description = "SSE token stream"),
        (status = 400, description = "Unknown model slug"),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Inference failed")
    )
)]
pub(super) async fn post_chat_stream(
    State(config): State<AppConfig>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, (StatusCode, String)> {
    let catalog = config.catalog_snapshot();
    super::validate_model(Some(&req.model), config.llm_model.as_deref(), &catalog)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    // Alias → owning slug (same contract as `post_chat`).
    let slug = super::resolve_model_slug(&req.model, &catalog)
        .map_or_else(|| req.model.clone(), str::to_owned);
    // Chat-capability parity with the completions gate (see `post_chat`).
    if let Some(entry) = super::resolve_model_entry(&slug, &catalog)
        && !entry.capabilities.chat
    {
        return Err((StatusCode::BAD_REQUEST, chat_unsupported_message(&slug)));
    }
    // REQ-8 parity with `/v1/chat/completions`: image content is gated on
    // model vision capability and `data:` URI payloads (plain-text 400 on
    // the native route).
    if let Some((message, _)) =
        vision_rejection(super::resolve_model_entry(&slug, &catalog), &req.messages)
    {
        return Err((StatusCode::BAD_REQUEST, message));
    }
    let pipe = chat_pipeline(
        &config,
        &slug,
        clamp_ttl_seconds(config.swap_max_ttl, req.ttl_seconds),
    )
    .await?;
    let worker_req = worker_request_base(
        slug.clone(),
        worker_messages(&req.messages),
        req.max_tokens,
        req.temperature,
    )
    .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancel = RequestCancel::new();
    // Set by the worker after the terminal frame: a normal end must not
    // abort the shared pipeline when the SSE body is dropped.
    let done = Arc::new(AtomicBool::new(false));
    // REQ-6 (Option B): the busy condition must be an HTTP 429, never a 200
    // with an error frame - so the slot is taken HERE through the admission
    // queue (REQ-6 Option A on top: a busy slot grants one bounded wait),
    // before the handler returns the SSE body. A rejection short-circuits
    // as a plain 429 with no registered job and no X-Job-Id.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Llm).await,
        crate::jobs_registry::JobKind::Chat,
        Some(slug),
        true,
    ) {
        Ok(pair) => pair,
        Err(error) => return Err(error),
    };
    // NEW-3: only an admitted stream is registry-visible and
    // interruptible — its cancel token is checked at every token boundary
    // below (the same abort-at-next-boundary path an SSE disconnect takes).
    // The lease sits in a slot the token sink can drain: once the client is
    // gone, the next token boundary frees the slot right away. The handler
    // returns the SSE body without awaiting the worker, so tokens stream as
    // they are produced.
    let lease_slot = Arc::new(Mutex::new(None::<crate::admission::GpuLease>));
    // Separate handle for the SSE body: the worker closure consumes its own
    // clone, the drop path calls `abort` through this one.
    let abort_pipe = Arc::clone(&pipe);
    let worker_cancel = cancel.clone();
    let body_cancel = cancel.clone();
    let worker_cancelled = cancelled.clone();
    let job_token = job.token().clone();
    let job_cancel_watcher = watch_job_cancel(job_token.clone(), cancel.clone());
    let worker_done = done.clone();
    let job_id = job.id().to_owned();
    let worker = tokio::task::spawn_blocking(move || {
        // The job guard rides with the worker: the registry entry lives
        // exactly as long as the stream (the JoinHandle is dropped/detached
        // when the SSE body goes away, the task still unwinds and
        // unregisters).
        let _job = job;
        *lease_slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(lease);
        let result = pipe.complete_stream_cancellable(
            &worker_req,
            &mut |token: &str| {
                if worker_cancelled.load(Ordering::Relaxed) || job_token.is_cancelled() {
                    // Also abort the FFI loop itself, not just delivery
                    // (disconnect, or `POST /v1/jobs/{id}/cancel`).
                    worker_cancel.cancel();
                    pipe.abort();
                    free_gpu_lease(&lease_slot);
                    return;
                }
                let data = serde_json::json!({ "token": token }).to_string();
                if tx.send(Ok(Event::default().data(data))).is_err() {
                    // Receiver gone without the drop guard (drop raced the send).
                    worker_cancelled.store(true, Ordering::Relaxed);
                    worker_cancel.cancel();
                    pipe.abort();
                    free_gpu_lease(&lease_slot);
                }
            },
            &worker_cancel,
        );
        // Set before terminal frames: the body can drop as `[DONE]` is read.
        worker_done.store(true, Ordering::Relaxed);
        // A disconnect between the last token and completion frees here.
        free_gpu_lease(&lease_slot);
        if let Err(err) = result {
            // loop-lore's parser reads `data:` only: a terminal error must
            // be a JSON data frame with no `[DONE]` after it.
            let payload = serde_json::json!({ "error": format!("chat failed: {err}") }).to_string();
            let _ = tx.send(Ok(Event::default().data(payload)));
        } else {
            let _ = tx.send(Ok(Event::default().data("[DONE]")));
        }
    });
    Ok(with_job_id(
        Sse::new(CancelOnDrop {
            inner: UnboundedReceiverStream::new(rx),
            cancelled,
            done,
            pipe: abort_pipe,
            cancel: body_cancel,
            job_cancel_watcher: Some(job_cancel_watcher),
            _worker: worker,
        })
        .into_response(),
        &job_id,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn job_cancel_watcher_sets_only_its_request_cancel() {
        let token = CancellationToken::new();
        let request_cancel = RequestCancel::new();
        let sibling = RequestCancel::new();
        let handle = watch_job_cancel(token.clone(), request_cancel.clone());

        token.cancel();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while !request_cancel.is_cancelled() && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        assert!(
            request_cancel.is_cancelled(),
            "job cancel did not reach the reader"
        );
        assert!(!sibling.is_cancelled(), "sibling request was cancelled");
        handle.abort();
    }
}
