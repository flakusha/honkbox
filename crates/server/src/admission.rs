//! Bounded admission queue in front of the single GPU slot (REQ-6 Option
//! A): when the residency slot is busy, ONE request may wait up to
//! `[server] queue_wait_secs` for the slot to free; every further concurrent
//! request gets the immediate 429 (`queue_full`), and a waiter that exceeds
//! its bound gets the 429 (`queue_timeout`) with the shared `Retry-After`
//! hint.
//!
//! Depth-1 policy on purpose (local-first, single GPU): a deep queue would
//! just move the 429 retry storm inside the server and turn UI time-outs
//! into silent multi-second waits. The direct `try_reserve` fast path is
//! preserved — an idle slot is never delayed by queue machinery.
//!
//! Ordering honesty: with depth 1 there is at most one waiter, woken when
//! the lease frees; every wake-up (and the deadline) retries `try_reserve`,
//! so a missed notification only costs latency, never correctness. A fresh
//! fast-path arrival CAN win the slot against the queued waiter's wake-up —
//! ordering is best-effort at depth 1, not a promised FIFO.
//!
//! The wait bound is process config ([`configure`], called from `serve`
//! with the resolved `[server] queue_wait_secs`); the static default is
//! ZERO = queueing disabled (immediate `no_wait` 429, the pre-REQ-6
//! behavior), so tests and CLI paths keep the old semantics unless they opt
//! in.

use std::sync::{LazyLock, Mutex, RwLock};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use tokio::sync::Notify;

use crate::metrics::{self, BusyReason};

/// Resolved `[server] queue_wait_secs`; `Duration::ZERO` disables queueing.
static QUEUE_WAIT: LazyLock<RwLock<Duration>> = LazyLock::new(|| RwLock::new(Duration::ZERO));

/// The process-wide admission state (waiter slot + free notification).
static ADMISSION: LazyLock<Admission> = LazyLock::new(Admission::new);

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

/// Reserve the process-wide GPU slot through the admission queue. An idle
/// slot is granted immediately (fast path, no queue touch); a busy slot
/// grants ONE bounded wait, then maps the outcome onto the 429 contract
/// (`Err((StatusCode::TOO_MANY_REQUESTS, message))`, same shape the old
/// `reserve_gpu_slot` handed its callers) with the busy-rejection metric
/// reason recorded (`no_wait` / `queue_full` / `queue_timeout`).
pub(crate) async fn acquire() -> Result<GpuLease, (StatusCode, String)> {
    // Fast path first, always: an idle slot never waits behind machinery.
    if let Ok(lease) = honkbox_residency::global().try_reserve() {
        return Ok(GpuLease::held(lease));
    }
    let wait = queue_wait();
    if wait.is_zero() {
        // Queue disabled: the legacy immediate rejection (REQ-6 Option B).
        return Err(busy_rejection(BusyReason::NoWait));
    }
    match ADMISSION.wait_for_slot(wait).await {
        Ok(lease) => Ok(GpuLease::held(lease)),
        Err(reason) => Err(busy_rejection(reason)),
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

/// A held GPU lease that pokes the queue when the slot actually frees — on
/// EVERY drop path (normal handler end, the stream workers' early
/// `free_gpu_lease`, error returns), so a queued waiter wakes promptly
/// instead of at its deadline. Dropping the inner lease releases the
/// residency slot (its own `Drop`).
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
            ADMISSION.slot_freed.notify_waiters();
        }
    }
}

/// Admission state: the single waiter slot (depth-1 policy) plus the
/// notification a lease drop fires.
struct Admission {
    /// 0 = queue has room, 1 = one waiter already queued.
    waiters: Mutex<usize>,
    /// Woken (`notify_waiters`) whenever a `GpuLease` frees the slot.
    slot_freed: Notify,
}

impl Admission {
    fn new() -> Self {
        Self {
            waiters: Mutex::new(0),
            slot_freed: Notify::new(),
        }
    }

    /// Claim the single waiter slot (`false` = queue full).
    fn try_claim_waiter(&self) -> bool {
        let mut waiters = self
            .waiters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *waiters == 0 {
            *waiters = 1;
            true
        } else {
            false
        }
    }

    /// Release the waiter slot (always; also on the error paths).
    fn release_waiter(&self) {
        *self
            .waiters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = 0;
    }

    /// Wait up to `wait` for the slot, retrying `try_reserve` on every
    /// wake-up (free notification OR the deadline sleep — a missed notify
    /// costs latency, never the slot). Caller has already lost the fast
    /// path; returns the raw lease (the `GpuLease` wrapper is applied by
    /// [`acquire`]) or the busy reason.
    async fn wait_for_slot(
        &self,
        wait: Duration,
    ) -> Result<honkbox_residency::Lease<'static>, BusyReason> {
        if !self.try_claim_waiter() {
            return Err(BusyReason::QueueFull);
        }
        let deadline = Instant::now() + wait;
        let outcome = loop {
            if let Ok(lease) = honkbox_residency::global().try_reserve() {
                break Ok(lease);
            }
            let now = Instant::now();
            if now >= deadline {
                break Err(BusyReason::QueueTimeout);
            }
            tokio::select! {
                () = self.slot_freed.notified() => {}
                () = tokio::time::sleep(deadline - now) => {}
            }
        };
        self.release_waiter();
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waiter_slot_is_depth_one() {
        let admission = Admission::new();
        assert!(admission.try_claim_waiter());
        // The depth-1 policy: a second concurrent waiter is turned away.
        assert!(!admission.try_claim_waiter());
        admission.release_waiter();
        assert!(admission.try_claim_waiter());
        admission.release_waiter();
    }

    #[test]
    fn release_waiter_is_total() {
        let admission = Admission::new();
        assert!(admission.try_claim_waiter());
        // Releasing without a claim (error-path symmetry) stays consistent.
        admission.release_waiter();
        admission.release_waiter();
        assert!(admission.try_claim_waiter());
        admission.release_waiter();
    }
}
