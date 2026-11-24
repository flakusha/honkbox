//! Read-only queue introspection (`GET /v1/queue`).
//!
//! The snapshot separates two kinds of work honestly:
//!
//! - running jobs come from the live [`crate::jobs_registry`] and carry their real
//!   registry ids, oldest first (the registry already sorts by registration
//!   order, so `position` is stable across reads);
//! - queued ledger tickets have NO public id — admission enqueues a private
//!   ticket, not a job — so they are reported only as the aggregate
//!   [`QueueSnapshot::queued`] depth. No ids are invented for them.
//!
//! A few routes still register before admitting (`audio_upload`, `audio_speech`,
//! `audio_transcriptions`, `image_ops_settle`), so a job can briefly be in both
//! views while its ledger ticket waits: the two fields are independent facts,
//! not a disjoint partition. Route-side ordering is the fix, not this snapshot.
//!
//! Everything is read through the same sources `/v1/capabilities` reports
//! (the ledger's `admission_info` and `admission::queue_wait_secs`), and the
//! handler acquires nothing, reserves nothing, and reorders nothing.

use axum::Json;
use honkbox_api_types::{QueueJob, QueueSnapshot};
use honkbox_residency::ResourceKind;

/// `GET /v1/queue`: side-effect-free snapshot of the live running jobs and
/// the bounded admission queue's aggregate depth.
#[utoipa::path(
    get,
    path = "/v1/queue",
    responses((status = 200, description = "Live queue snapshot", body = QueueSnapshot))
)]
pub(crate) async fn get_queue() -> Json<QueueSnapshot> {
    let ledger = honkbox_residency::global();
    // Same ledger snapshot capabilities uses for `queue.gpu_slots` and the
    // per-kind TTL; one read keeps all derived fields mutually consistent.
    let admission = ledger.admission_info();
    let ttl_secs = admission.queue_ttl.as_secs();

    let jobs: Vec<QueueJob> = crate::jobs_registry::global()
        .list()
        .into_iter()
        .enumerate()
        .map(|(position, info)| QueueJob {
            id: Some(info.id),
            kind: info.kind,
            model: info.model,
            state: "running".to_owned(),
            position,
            ttl_secs,
        })
        .collect();

    // Ledger waiters are id-less tickets: count them, never name them.
    let queued: u64 = ResourceKind::ALL
        .iter()
        .map(|kind| ledger.stats(*kind).queue_depth)
        .sum();

    Json(QueueSnapshot {
        jobs,
        gpu_slots: u64::from(admission.gpu_slots),
        queued,
        queue_wait_secs: crate::admission::queue_wait_secs(),
    })
}
