//! Per-kind admission adapter over the residency resource ledger (REQ-6
//! Option A, generalized by TASK-resource-ledger-bounded-queue): when a
//! kind's capacity is busy, ONE request per kind may wait up to
//! `[server] queue_wait_secs` in the ledger's bounded queue for that kind;
//! every further concurrent request gets the immediate 429 (`queue_full`),
//! and a waiter that exceeds its bound gets the 429 (`queue_timeout`) with
//! the shared `Retry-After` hint.
//!
//! Depth-1 policy on purpose (local-first, single GPU): a deep queue would
//! just move the 429 retry storm inside the server and turn UI time-outs
//! into silent multi-second waits. The direct `try_acquire` fast path is
//! preserved — idle capacity is never delayed by queue machinery, and a
//! fresh fast-path arrival CAN win against a queued waiter's wake-up
//! (ordering is best-effort at depth 1, not a promised FIFO). Within one
//! kind the ledger grants strictly FIFO.
//!
//! The 429-vs-503 line (TASK decision): 429 covers every TRANSIENT
//! admission rejection (slot busy with queueing disabled, bounded queue
//! full, queue TTL timeout) — the pre-ledger contract, byte-identical. 503
//! covers the new hard-capacity case where a job's configured demand can
//! never fit the configured capacities (e.g. a VRAM estimate above the
//! budget) — no queue wait can ever grant it, so it is a service-shape
//! error, not a rate-limit. The bounded-queue rejections keep the shared
//! middleware `Retry-After: 1` (the ledger's own hint is one full wait
//! window and would regress the pinned `Retry-After: 1` contract).
//!
//! The wait bound is process config ([`configure`], called from `serve`
//! with the resolved `[server] queue_wait_secs`); the static default is
//! ZERO = queueing disabled (immediate `no_wait` 429, the pre-REQ-6
//! behavior), so tests and CLI paths keep the old semantics unless they opt
//! in. The same bound is installed as the ledger queue TTL, so the ledger's
//! eviction clock and this 429's "gave up waiting after Xs" agree.

use std::sync::{LazyLock, RwLock};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use tokio::sync::Notify;

use crate::metrics::{self, BusyReason};
use honkbox_residency::{AcquireError, EnqueueError, PollOutcome, ResourceKind};

/// Resolved `[server] queue_wait_secs`; `Duration::ZERO` disables queueing.
static QUEUE_WAIT: LazyLock<RwLock<Duration>> = LazyLock::new(|| RwLock::new(Duration::ZERO));

/// Woken (`notify_waiters`) whenever a `GpuLease` frees capacity — one
/// shared notifier for every kind (each waiter re-checks its own ledger
/// ticket, so a wrong-kind wake costs one poll, never correctness).
static LEDGER_NOTIFY: LazyLock<Notify> = LazyLock::new(Notify::new);

/// Install the resolved `[server] queue_wait_secs` (called once at serve
/// startup; `Duration::ZERO` keeps the immediate-429 behavior).
pub(crate) fn configure(wait: Duration) {
    *QUEUE_WAIT
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = wait;
}

/// The configured wait bound (test + message introspection).
fn queue_wait() -> Duration {
    *QUEUE_WAIT
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Reserve capacity for `kind` through the ledger's admission queue. An
/// idle slot is granted immediately (fast path, no queue touch); a busy
/// slot enqueues the bounded per-kind ticket and polls it, then maps the
/// outcome onto the 429 contract (`Err((StatusCode, message))`, the same
/// shape the pre-ledger `acquire` handed its callers) with the
/// busy-rejection metric reason recorded (`no_wait` / `queue_full` /
/// `queue_timeout`). A never-fitting demand maps to the 503 hard-capacity
/// contract instead.
pub(crate) async fn acquire(kind: ResourceKind) -> Result<GpuLease, (StatusCode, String)> {
    let ledger = honkbox_residency::global();
    // Fast path first, always: idle capacity never waits behind machinery.
    match ledger.try_acquire(kind) {
        Ok(lease) => return Ok(GpuLease::held(lease)),
        Err(AcquireError::Oversized { what }) => return Err(hard_rejection(kind, what)),
        Err(AcquireError::Busy) => {}
    }
    let wait = queue_wait();
    if wait.is_zero() {
        // Queue disabled: the legacy immediate rejection (REQ-6 Option B).
        return Err(busy_rejection(BusyReason::NoWait));
    }
    // Claim a place in the kind's bounded queue; overflow is the 429
    // `queue_full` contract (the ledger counted the rejection).
    let ticket = match ledger.enqueue(kind) {
        Ok(ticket) => ticket,
        Err(EnqueueError::QueueFull { .. }) => return Err(busy_rejection(BusyReason::QueueFull)),
        Err(EnqueueError::Oversized { what }) => return Err(hard_rejection(kind, what)),
    };
    let deadline = Instant::now() + wait;
    loop {
        // Poll first: the ledger owns the TTL, so an expired ticket is
        // counted as a ledger eviction AND returns the queue_timeout 429.
        match ticket.poll() {
            PollOutcome::Granted(lease) => return Ok(GpuLease::held(lease)),
            PollOutcome::Expired | PollOutcome::Cancelled => {
                return Err(busy_rejection(BusyReason::QueueTimeout));
            }
            PollOutcome::Waiting => {}
        }
        let now = Instant::now();
        if now >= deadline {
            return Err(busy_rejection(BusyReason::QueueTimeout));
        }
        // A lease drop anywhere in the process could free OUR capacity (the
        // slot pool is shared across kinds), so every waiter wakes on every
        // notify and re-polls; a missed notify only costs one backstop
        // sleep, never the slot.
        tokio::select! {
            () = LEDGER_NOTIFY.notified() => {}
            () = tokio::time::sleep(deadline - now) => {}
        }
    }
}

/// Map a busy outcome onto the 429 contract: metric reason recorded, error
/// message naming what happened (the legacy wording for `no_wait` so the
/// disabled-queue path stays byte-identical with the pre-REQ-6 behavior).
fn busy_rejection(reason: BusyReason) -> (StatusCode, String) {
    metrics::note_gpu_slot_busy(reason);
    let message = match reason {
        BusyReason::NoWait => honkbox_residency::Busy.to_string(),
        BusyReason::QueueFull => {
            "gpu slot busy: another job is running and the admission queue is full; retry later"
                .to_owned()
        }
        BusyReason::QueueTimeout => format!(
            "gpu slot busy: another job is running; gave up waiting after {}s",
            queue_wait().as_secs_f64()
        ),
    };
    (StatusCode::TOO_MANY_REQUESTS, message)
}

/// Map a never-fitting demand onto the hard-capacity contract: 503 (the
/// ledger counted the rejection; this is a config/size error, not a
/// rate-limit — see the module docs for the 429-vs-503 line).
fn hard_rejection(kind: ResourceKind, what: &str) -> (StatusCode, String) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        format!(
            "resource demand for {kind} jobs exceeds the configured capacity ({what}); \
             no admission queue can ever grant it"
        ),
    )
}

/// A held GPU lease that pokes the queue when capacity actually frees — on
/// EVERY drop path (normal handler end, the stream workers' early
/// `free_gpu_lease`, error returns), so a queued waiter wakes promptly
/// instead of at its deadline. Dropping the inner lease releases the ledger
/// demand (its own `Drop`); the notify fires AFTER the release so the woken
/// waiter's poll sees free capacity.
#[derive(Debug)]
pub(crate) struct GpuLease {
    inner: Option<honkbox_residency::Lease<'static>>,
}

impl GpuLease {
    fn held(lease: honkbox_residency::Lease<'static>) -> Self {
        Self { inner: Some(lease) }
    }
}

impl Drop for GpuLease {
    fn drop(&mut self) {
        if self.inner.take().is_some() {
            LEDGER_NOTIFY.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_maps_to_503_naming_the_dimension() {
        let (status, message) = hard_rejection(ResourceKind::Sd, "gpu slots");
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(message.contains("sd"));
        assert!(message.contains("gpu slots"));
        let (status, message) = hard_rejection(ResourceKind::Llm, "vram budget");
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(message.contains("llm"));
        assert!(message.contains("vram budget"));
    }

    #[test]
    fn busy_rejection_keeps_the_legacy_wording() {
        let (status, message) = busy_rejection(BusyReason::NoWait);
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(message, honkbox_residency::Busy.to_string());
        let (status, message) = busy_rejection(BusyReason::QueueFull);
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert!(message.contains("admission queue is full"), "{message}");
        let (status, message) = busy_rejection(BusyReason::QueueTimeout);
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert!(message.contains("gave up waiting"), "{message}");
    }
}
