//! In-flight job registry (NEW-3, Feature-Survey §4): every GPU route
//! registers its job (id, kind, model, started-at, cancel token) for the
//! lifetime of the request; `GET /v1/jobs` lists the registry and
//! `POST /v1/jobs/{id}/cancel` marks a job's token.
//!
//! Cancellation semantics are **cooperative** by design. GPU inference runs
//! on the blocking pool across one synchronous C FFI call (llama.cpp /
//! sd.cpp / whisper.cpp); true engine-level cancellation needs a per-backend
//! spike (on record in the design docs). Marking the token therefore cannot
//! interrupt a call mid-FFI:
//!
//! - `/v1/images/generations` with `n > 1` checks the token between images
//!   and stops before generating the remaining ones (the one real
//!   mid-flight abort; a queued job aborts before its first image);
//! - the streaming chat routes check the token at every token boundary and
//!   abort the pipeline at the next one (the same path an SSE disconnect
//!   takes);
//! - single-shot calls (one image, sync chat/completions, embeddings,
//!   transcribe, synthesize, translations) run to completion. `cancel`
//!   answers honestly: `cancelled: true` only for jobs that check the token
//!   between work units (`interruptible`).
//!
//! The registry is process-global like the residency ledger. Entries are
//! removed when the route handler's [`JobGuard`] drops, so the registry
//! only ever holds live work — this is a liveness view, not history.
//!
//! Ids reach callers as the `X-Job-Id` response header. On SSE routes the
//! header precedes the stream body, so a client can cancel the very stream
//! it is reading. On sync routes the header arrives with the completed
//! response (too late to cancel that same request) — clients that need to
//! cancel discover ids from `GET /v1/jobs` instead.
//!
//! # Lifecycle events (`GET /v1/events`)
//!
//! Every transition emits a [`JobEvent`] on a process-wide
//! [`tokio::sync::broadcast`] channel (`subscribe`): `job_registered` when
//! a job enters the registry, `job_cancelled` the moment a cancel is
//! REQUESTED, and `job_finished` when the job leaves the registry (the
//! event's `cancelled` flag says whether the token had been marked — for a
//! non-interruptible job that still means the FFI call ran to completion).
//! Emission is fire-and-forget: with no live subscribers the send fails and
//! the event is dropped; a subscriber that falls more than
//! [`EVENT_CHANNEL_CAPACITY`] events behind gets a `Lagged` error (the SSE
//! route surfaces that as an `events_dropped` frame, never a lie by
//! omission). The channel carries transitions only — there is no replay of
//! the current registry on subscribe; `GET /v1/jobs` is the snapshot.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use honkbox_api_types::{JobEvent, JobEventType, JobInfo};

/// Broadcast capacity for lifecycle events, in events (not bytes). A
/// subscriber that falls further behind is `Lagged` — surfaced honestly by
/// the SSE route instead of silently skipped. Small by design: this is a
/// liveness stream, not a history log.
const EVENT_CHANNEL_CAPACITY: usize = 256;

/// The process-wide registry.
static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

/// The shared in-flight job registry.
pub(crate) fn global() -> &'static Registry {
    &REGISTRY
}

/// Modality of the GPU work behind a job (the `kind` field of `JobInfo`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JobKind {
    /// Chat completion (sync `/v1/chat`, `/v1/chat/completions`, or either
    /// streaming variant — the stream variant is the interruptible one).
    Chat,
    /// Embedding batch (`POST /v1/embeddings`).
    Embeddings,
    /// Image generation (`POST /v1/images/generations`).
    Images,
    /// Speech-to-text (`/v1/transcribe`, `/v1/audio/transcriptions`).
    Transcribe,
    /// Text-to-speech (`/v1/synthesize`, `/v1/audio/speech`).
    Synthesize,
    /// Speech translation to English (`/v1/audio/translations`).
    Translations,
}

impl JobKind {
    /// Wire label served by `GET /v1/jobs`.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Embeddings => "embeddings",
            Self::Images => "images",
            Self::Transcribe => "transcribe",
            Self::Synthesize => "synthesize",
            Self::Translations => "translations",
        }
    }
}

/// One registry entry: everything `GET /v1/jobs` reports plus the cancel
/// token the route's worker checks between work units.
#[derive(Debug)]
struct JobEntry {
    id: String,
    token: CancellationToken,
    kind: JobKind,
    model: Option<String>,
    started_at_ms: u64,
    /// Monotonic per-registry insertion order: `started_at_ms` has
    /// millisecond resolution and same-ms ties are otherwise decided by
    /// `HashMap` iteration order (nondeterministic).
    seq: u64,
    /// Whether the route honors the token between work units (multi-image
    /// generations, streaming chat). Surfaced verbatim by `cancel` so the
    /// response never promises an interruption the route cannot deliver.
    interruptible: bool,
}

impl JobEntry {
    /// The event-stream projection of this entry (with the lifecycle
    /// transition and the cancel-requested flag filled in).
    fn to_event(&self, event: JobEventType, cancelled: bool) -> JobEvent {
        JobEvent {
            event,
            id: self.id.clone(),
            kind: self.kind.label().to_owned(),
            model: self.model.clone(),
            started_at_ms: self.started_at_ms,
            interruptible: self.interruptible,
            cancelled,
        }
    }
}

/// Registry of live GPU jobs: id → entry. The map lives behind an `Arc` so
/// each [`JobGuard`] unregisters from ITS OWN registry on drop (a locally
/// built registry in tests behaves exactly like the global one). The
/// lifecycle-event broadcaster rides alongside: `register`, `cancel`, and
/// [`JobGuard::drop`] are the only emitters, all keyed by the registry.
#[derive(Debug)]
pub(crate) struct Registry {
    entries: Arc<Mutex<HashMap<String, JobEntry>>>,
    events: broadcast::Sender<JobEvent>,
    seq: AtomicU64,
}

impl Default for Registry {
    fn default() -> Self {
        let (events, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        Self {
            entries: Arc::default(),
            events,
            seq: AtomicU64::new(0),
        }
    }
}

impl Registry {
    /// Register one job; returns the guard whose drop unregisters it.
    /// `interruptible` must reflect what the route actually checks (see the
    /// module docs) — it is the honesty contract of the cancel endpoint.
    pub(crate) fn register(
        &self,
        kind: JobKind,
        model: Option<String>,
        interruptible: bool,
    ) -> JobGuard {
        let id = format!("job_{}", ulid::Ulid::new());
        let token = CancellationToken::new();
        let started_at_ms = now_ms();
        let entry = JobEntry {
            id: id.clone(),
            token: token.clone(),
            kind,
            model: model.clone(),
            started_at_ms,
            seq: self.seq.fetch_add(1, Ordering::Relaxed),
            interruptible,
        };
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id.clone(), entry);
        // Fire-and-forget: no live subscriber is a normal idle state, not
        // an error (observability never back-pressures GPU work).
        let _ = self.events.send(JobEvent {
            event: JobEventType::Registered,
            id: id.clone(),
            kind: kind.label().to_owned(),
            model,
            started_at_ms,
            interruptible,
            cancelled: false,
        });
        JobGuard {
            entries: Arc::clone(&self.entries),
            events: self.events.clone(),
            id,
            token,
        }
    }

    /// Mark a live job's token. `Some(interruptible)` = the id was found
    /// (freshly marked or already cancelled while still in flight); `None`
    /// = unknown id (never registered, or its request already ended).
    /// Emits `job_cancelled` — a cancel was REQUESTED, not a completion —
    /// on the lifecycle channel when the id was found.
    pub(crate) fn cancel(&self, id: &str) -> Option<bool> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = entries.get_mut(id)?;
        entry.token.cancel();
        // Fire-and-forget (same contract as `register`); the send touches
        // only the broadcast channel, never the entries map, so holding the
        // entries lock here cannot invert any lock order.
        let _ = self
            .events
            .send(entry.to_event(JobEventType::Cancelled, true));
        Some(entry.interruptible)
    }

    /// Live jobs, oldest first (registration order; deterministic even for
    /// same-millisecond registrations).
    pub(crate) fn list(&self) -> Vec<JobInfo> {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut rows: Vec<&JobEntry> = entries.values().collect();
        rows.sort_by_key(|entry| (entry.started_at_ms, entry.seq));
        rows.into_iter()
            .map(|entry| JobInfo {
                id: entry.id.clone(),
                kind: entry.kind.label().to_owned(),
                model: entry.model.clone(),
                started_at_ms: entry.started_at_ms,
                interruptible: entry.interruptible,
            })
            .collect()
    }

    /// Subscribe to job lifecycle events (`GET /v1/events`). Only events
    /// emitted after the subscription are delivered — there is no replay;
    /// `list` is the current-state snapshot.
    pub(crate) fn subscribe(&self) -> broadcast::Receiver<JobEvent> {
        self.events.subscribe()
    }
}

/// Registry id + cancel token + a handle to the OWNING registry; dropping
/// it unregisters the job from that registry, so a guard must live exactly
/// as long as the request that owns the work. The drop emits `job_finished`
/// on the lifecycle channel (`cancelled` = the token had been marked — for
/// a non-interruptible job the work still ran to completion).
#[derive(Debug)]
pub(crate) struct JobGuard {
    entries: Arc<Mutex<HashMap<String, JobEntry>>>,
    events: broadcast::Sender<JobEvent>,
    id: String,
    token: CancellationToken,
}

impl JobGuard {
    /// Registry id (stamped as the `X-Job-Id` response header).
    pub(crate) fn id(&self) -> &str {
        &self.id
    }

    /// Cooperative-stop token; the route checks this between work units.
    pub(crate) fn token(&self) -> &CancellationToken {
        &self.token
    }
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        let entry = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.id);
        if let Some(entry) = entry {
            // The guard's own token says whether a cancel had been marked;
            // what the work actually did with it is the `interruptible`
            // contract (documented on the event DTO).
            let cancelled = self.token.is_cancelled();
            let _ = self
                .events
                .send(entry.to_event(JobEventType::Finished, cancelled));
        }
    }
}

/// Current Unix time in milliseconds (0 if the clock is pre-epoch).
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Local registry instance (the global one is shared with other tests).
    fn fresh() -> Registry {
        Registry::default()
    }

    #[test]
    fn register_list_remove_roundtrip() {
        let registry = fresh();
        let job = registry.register(JobKind::Chat, Some("llm:a".to_owned()), false);
        assert!(job.id().starts_with("job_"));
        let listed = registry.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].kind, "chat");
        assert_eq!(listed[0].model.as_deref(), Some("llm:a"));
        assert!(!listed[0].interruptible);
        assert!(listed[0].started_at_ms > 0);
        drop(job);
        assert!(registry.list().is_empty());
    }

    #[test]
    fn cancel_marks_token_and_reports_interruptible() {
        let registry = fresh();
        let job = registry.register(JobKind::Images, None, true);
        let token = job.token().clone();
        assert_eq!(registry.cancel(job.id()), Some(true));
        assert!(token.is_cancelled());
        // Idempotent: a second cancel still finds the live job.
        assert_eq!(registry.cancel(job.id()), Some(true));
        drop(job);
        // After the request ends the id is gone → the 404 path.
        assert_eq!(registry.cancel("job_unknown"), None);
    }

    #[test]
    fn non_interruptible_jobs_report_honestly() {
        let registry = fresh();
        let job = registry.register(JobKind::Transcribe, None, false);
        assert_eq!(registry.cancel(job.id()), Some(false));
    }

    #[test]
    fn list_orders_oldest_first() {
        let registry = fresh();
        let first = registry.register(JobKind::Chat, None, false);
        let second = registry.register(JobKind::Images, None, true);
        let listed = registry.list();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].kind, "chat");
        assert_eq!(listed[1].kind, "images");
        drop(first);
        drop(second);
    }

    /// Receive one event with a generous bound (a missed send fails with
    /// context at the timeout instead of hanging forever).
    async fn next_event(receiver: &mut broadcast::Receiver<JobEvent>) -> JobEvent {
        tokio::time::timeout(std::time::Duration::from_secs(5), receiver.recv())
            .await
            .expect("lifecycle event within 5s")
            .expect("channel open")
    }

    /// The full lifecycle for one job: registered → cancelled (request)
    /// → finished (token was marked). Event fields mirror the live
    /// `GET /v1/jobs` row.
    #[tokio::test]
    async fn events_report_the_full_lifecycle() {
        let registry = fresh();
        let mut receiver = registry.subscribe();
        let job = registry.register(
            JobKind::Images,
            Some("comfy:checkpoints/s.safetensors".to_owned()),
            true,
        );
        let registered = next_event(&mut receiver).await;
        assert_eq!(registered.event, JobEventType::Registered);
        assert_eq!(registered.event.label(), "job_registered");
        assert_eq!(registered.id, *job.id());
        assert_eq!(registered.kind, "images");
        assert_eq!(
            registered.model.as_deref(),
            Some("comfy:checkpoints/s.safetensors")
        );
        assert!(registered.interruptible);
        assert!(!registered.cancelled);
        assert!(registered.started_at_ms > 0);

        assert_eq!(registry.cancel(job.id()), Some(true));
        let cancelled = next_event(&mut receiver).await;
        assert_eq!(cancelled.event, JobEventType::Cancelled);
        assert_eq!(cancelled.event.label(), "job_cancelled");
        assert_eq!(cancelled.id, *job.id());
        assert!(cancelled.cancelled);
        assert!(cancelled.interruptible);

        drop(job);
        let finished = next_event(&mut receiver).await;
        assert_eq!(finished.event, JobEventType::Finished);
        assert_eq!(finished.event.label(), "job_finished");
        assert_eq!(finished.id, cancelled.id);
        // The token had been marked before the job ended.
        assert!(finished.cancelled);
        assert_eq!(finished.started_at_ms, registered.started_at_ms);
    }

    /// A job that ends without any cancel reports `cancelled: false` on
    /// `job_finished` — the flag is "a cancel had been requested", never
    /// an outcome claim.
    #[tokio::test]
    async fn finish_without_cancel_reports_cancelled_false() {
        let registry = fresh();
        let mut receiver = registry.subscribe();
        let job = registry.register(JobKind::Synthesize, None, false);
        let registered = next_event(&mut receiver).await;
        assert!(!registered.cancelled);
        drop(job);
        let finished = next_event(&mut receiver).await;
        assert_eq!(finished.event, JobEventType::Finished);
        assert!(!finished.cancelled);
        assert!(!finished.interruptible);
        assert!(finished.model.is_none());
    }

    /// Events are broadcast: every subscriber sees the same sequence, and
    /// emitting with NO subscriber is a quiet no-op — observability stays
    /// fire-and-forget, never a panic, never back-pressure.
    #[tokio::test]
    async fn events_fan_out_and_send_without_receivers_is_quiet() {
        let registry = fresh();
        // No subscriber yet: this emit is silently dropped.
        let job = registry.register(JobKind::Chat, None, false);
        let mut first = registry.subscribe();
        let mut second = registry.subscribe();
        drop(job);
        let from_first = next_event(&mut first).await;
        let from_second = next_event(&mut second).await;
        assert_eq!(from_first.event, JobEventType::Finished);
        assert_eq!(from_second.event, JobEventType::Finished);
        assert_eq!(from_first.id, from_second.id);
    }

    /// The serialized wire shape: the SSE `event:` label mirrors the JSON
    /// `event` field (explicit per-variant renames), and `model` is omitted
    /// when unknown (`skip_serializing_if`).
    #[tokio::test]
    async fn event_wire_shape() {
        let registry = fresh();
        let mut receiver = registry.subscribe();
        let job = registry.register(JobKind::Embeddings, None, false);
        let event = next_event(&mut receiver).await;
        let json = serde_json::to_value(&event).expect("serialize");
        assert_eq!(json["event"], "job_registered");
        assert_eq!(json["kind"], "embeddings");
        assert_eq!(json["interruptible"], false);
        assert_eq!(json["cancelled"], false);
        assert!(json.get("model").is_none(), "model omitted: {json}");
        assert!(json["id"].as_str().is_some());
        assert!(json["started_at_ms"].as_u64().is_some());
        drop(job);
    }
}
