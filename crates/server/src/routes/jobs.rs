//! Job registry routes (`/v1/jobs`, cancel verbs, `/v1/events` SSE),
//! extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only handlers and their helpers live
//! here.

use std::convert::Infallible;

use axum::{
    Json,
    extract::Path as AxumPath,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use honkbox_api_types::{JobCancelResult, JobEvent, JobList};
use tokio_stream::{
    Stream, StreamExt as _,
    wrappers::{BroadcastStream, errors::BroadcastStreamRecvError},
};

/// `GET /v1/jobs` (NEW-3): the in-flight registry — admitted GPU requests
/// (running, or waiting in the admission queue), oldest first. Entries
/// appear when a route admits a request and vanish when it ends; this is a
/// liveness view, not history (see `jobs_registry`).
#[utoipa::path(
    get,
    path = "/v1/jobs",
    responses(
        (status = 200, description = "In-flight jobs", body = JobList),
    )
)]
pub(crate) async fn get_jobs() -> Json<JobList> {
    Json(JobList {
        object: "list".to_owned(),
        data: crate::jobs_registry::global().list(),
    })
}

/// Shared cancel machinery for `POST /v1/jobs/{id}/cancel` and
/// `DELETE /v1/jobs/{id}` (same semantics, two spellings — the REST-shaped
/// verb and the explicit action): mark the job's cooperative-stop token
/// and answer honestly. 200 `{cancelled: bool}` — `true` for EVERY found
/// id, because the token was marked; whether the work actually stops early
/// is the separate `interruptible` contract reported by `GET /v1/jobs`
/// and the job lifecycle events (image generations check between images
/// AND mid-sampling via `sd_cancel_generation`; streaming chat checks at
/// the next token boundary; a single blocking FFI call runs to completion).
/// 404: unknown id (never registered, or the request already ended — the
/// registry holds live work only).
pub(super) fn cancel_job_response(id: &str) -> Response {
    match crate::jobs_registry::global().cancel(id) {
        Some(outcome) => Json(JobCancelResult {
            cancelled: outcome.cancelled,
        })
        .into_response(),
        None => (StatusCode::NOT_FOUND, format!("unknown job id {id:?}")).into_response(),
    }
}

/// `POST /v1/jobs/{id}/cancel` (NEW-3): mark the job's cooperative-stop
/// token. See [`cancel_job_response`] for the honest response contract.
#[utoipa::path(
    post,
    path = "/v1/jobs/{id}/cancel",
    responses(
        (status = 200, description = "Token marked", body = JobCancelResult),
        (status = 404, description = "Unknown job id"),
    )
)]
pub(crate) async fn post_job_cancel(AxumPath(id): AxumPath<String>) -> Response {
    cancel_job_response(&id)
}

/// `DELETE /v1/jobs/{id}`: the REST-shaped spelling of
/// `POST /v1/jobs/{id}/cancel` — deleting an in-flight job means requesting
/// its cooperative cancel, nothing more. Same token machinery, same honest
/// 200 `{cancelled}` / 404 contract (see [`cancel_job_response`]); the
/// DELETE does not wait for the job to end and never removes finished-work
/// records (job bundles under `<data_dir>/jobs/` are a separate concern).
#[utoipa::path(
    delete,
    path = "/v1/jobs/{id}",
    responses(
        (status = 200, description = "Cancel requested (same contract as POST /v1/jobs/{id}/cancel)", body = JobCancelResult),
        (status = 404, description = "Unknown job id"),
    )
)]
pub(crate) async fn delete_job(AxumPath(id): AxumPath<String>) -> Response {
    cancel_job_response(&id)
}

/// `GET /v1/events`: the registry-wide job lifecycle feed as a
/// Server-Sent-Events stream (observability only — it registers no job and
/// touches no GPU slot). Every in-flight GPU job of every kind emits:
///
/// - `event: job_registered` — the job entered the registry (route
///   admitted the request; it may still be waiting in the admission
///   queue);
/// - `event: job_cancelled` — a cancel was REQUESTED for the job (POST
///   `.../cancel` or DELETE `/v1/jobs/{id}`); not a completion event;
/// - `event: job_finished` — the job left the registry (its request ended,
///   any outcome);
/// - `event: job_progress` — a work update: `step`, `total_steps`, and
///   `pct` (percent, 0..=100; omitted when the total is 0);
/// - `event: job_result` — an artifact was produced: `artifact_id`,
///   `sha256`, `size_bytes`. A job may emit several; not a completion
///   signal.
///
/// Each frame's JSON `data` mirrors the SSE `event` line in its `event`
/// field (`JobEvent`): `id`, `kind`, `model` (when known),
/// `started_at_ms`, `interruptible`, and `cancelled` (= a cancel had been
/// requested at that point; on `job_finished` it is `true` for a
/// token-marked job, which for a non-interruptible job still ran to
/// completion), plus the progress/result fields above when applicable.
/// Frames are live-only: a subscriber that falls more
/// than 256 events behind gets an `events_dropped` frame naming the skipped
/// count (honest gap marker, never silent skipping), and (re)connecting
/// replays nothing — `GET /v1/jobs` is the current-state snapshot.
#[utoipa::path(
    get,
    path = "/v1/events",
    responses(
        (status = 200, description = "SSE stream of job events (job_registered / job_cancelled / job_finished / job_progress / job_result; catalogue above)", content_type = "text/event-stream"),
    )
)]
pub(crate) async fn get_events() -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let receiver = crate::jobs_registry::global().subscribe();
    let stream = BroadcastStream::new(receiver).filter_map(|item| match item {
        Ok(event) => Some(Ok::<_, Infallible>(job_event_frame(&event))),
        // This consumer fell behind: surface the gap instead of quietly
        // skipping the missed events. (A closed channel has no error
        // variant here — the stream simply ends, and the registry's sender
        // never closes anyway.)
        Err(BroadcastStreamRecvError::Lagged(skipped)) => {
            let frame = Event::default().event("events_dropped").data(format!(
                "{{\"event\":\"events_dropped\",\"skipped\":{skipped}}}"
            ));
            Some(Ok(frame))
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// One SSE frame for a registry lifecycle event: the `event:` line mirrors
/// the JSON `event` field, so JSON-only consumers can ignore the envelope.
pub(super) fn job_event_frame(event: &JobEvent) -> Event {
    Event::default()
        .event(event.event.label())
        .data(serde_json::to_string(&event).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use honkbox_api_types::JobEventType;

    /// The SSE `event:` line comes from `label()`; every event type,
    /// progress and result included, must have a wire label.
    #[test]
    fn frame_labels_cover_every_event_type() {
        for (kind, label) in [
            (JobEventType::Registered, "job_registered"),
            (JobEventType::Cancelled, "job_cancelled"),
            (JobEventType::Finished, "job_finished"),
            (JobEventType::Progress, "job_progress"),
            (JobEventType::Result, "job_result"),
        ] {
            assert_eq!(kind.label(), label);
        }
    }
}
