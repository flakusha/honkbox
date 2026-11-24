//! In-flight job registry (NEW-3, Feature-Survey §4): every GPU route
//! registers its job (id, kind, model, started-at, cancel token) for the
//! lifetime of the request; `GET /v1/jobs` lists the registry and
//! `POST /v1/jobs/{id}/cancel` marks a job's token.
//!
//! Cancellation semantics are **cooperative** by design. GPU inference runs
//! on the blocking pool across one synchronous C FFI call (llama.cpp /
//! sd.cpp / whisper.cpp). Marking the token cannot itself reach into a
//! call — but backends can expose an engine-level stop the route bridges
//! the token into:
//!
//! - `/v1/images/generations` checks the token between images and stops
//!   before generating the remaining ones, and — since the sd.cpp backend
//!   gained `sd_cancel_generation` — aborts the RUNNING single-image
//!   generation mid-sampling too (a queued job aborts before its first
//!   image; a cancelled image answers 409 `job_cancelled` naming the
//!   interrupted image, never a partial render);
//! - the streaming chat routes check the token at every token boundary and
//!   abort the pipeline at the next one (the same path an SSE disconnect
//!   takes);
//! - single-shot calls that have no engine stop (sync chat/completions,
//!   embeddings, transcribe, synthesize, translations) run to completion.
//!   `cancel` reports both facts separately: `cancelled` (the token was
//!   marked — true for every found id) and `interruptible` (whether the
//!   work can actually stop early). They are never conflated.
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
//! Progress-reporting backends also emit `job_progress` (step / total / pct)
//! and artifact-producing routes emit `job_result` (artifact id / sha256 /
//! size); neither is a lifecycle transition.
//! Emission is fire-and-forget: with no live subscribers the send fails and
//! the event is dropped; a subscriber that falls more than
//! [`EVENT_CHANNEL_CAPACITY`] events behind gets a `Lagged` error (the SSE
//! route surfaces that as an `events_dropped` frame, never a lie by
//! omission). Every emission is also kept in a bounded replay window keyed
//! by its monotonic `seq`, so a reconnect with `?since=` / `Last-Event-ID`
//! replays only what it missed; a cursor older than the window collapses to
//! `events_dropped`. `GET /v1/jobs` stays the snapshot for cursor-less
//! clients.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use honkbox_api_types::{JobArtifactRef, JobEvent, JobEventType, JobInfo, NsfwAnnotation};

/// Broadcast capacity for lifecycle events, in events (not bytes). A
/// subscriber that falls further behind is `Lagged` — surfaced honestly by
/// the SSE route instead of silently skipped. Small by design: this is a
/// liveness stream, not a history log.
const EVENT_CHANNEL_CAPACITY: usize = 256;

/// Cap on artifact refs kept per live job. Frames are never capped; the
/// snapshot only needs enough identity to reconcile a dropped frame.
const MAX_ARTIFACT_REFS: usize = 64;

/// Borrowed selection metadata the producer already holds at emit time.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArtifactFrameMeta<'a> {
    pub name: &'a str,
    pub kind: &'a str,
    pub index: u32,
    pub primary: bool,
}

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
    /// CPU image ops (`POST /v1/image-ops`): decode/transform/encode work
    /// on the blocking pool — no GPU slot; cancel is honored between
    /// inputs (one op batch = many work units).
    ImageOps,
    /// Blender asset jobs (`POST /v1/jobs/blender`): an external Blender
    /// process; cancel is honored at the next STAGE boundary (the recipe
    /// checks a marker file between stages, never mid-bake).
    Blender,
    /// OCR document ingest (`POST /v1/ocr/ingest`): in-process CPU work;
    /// cancel is honored between pages of a document.
    Ocr,
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
            Self::ImageOps => "image_ops",
            Self::Blender => "blender",
            Self::Ocr => "ocr",
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
    /// Monotonic order key: the `seq` stamped on this job `job_registered`
    /// event (`started_at_ms` ties at millisecond resolution; `HashMap`
    /// iteration order would be nondeterministic).
    seq: u64,
    /// Whether the route honors the token — between work units (multi-image
    /// generations, streaming chat) or by bridging it into an engine-level
    /// stop (image generations mid-sampling via `sd_cancel_generation`).
    /// Surfaced verbatim by `cancel` so the response never promises an
    /// interruption the route cannot deliver.
    interruptible: bool,
    /// Artifact identities emitted so far (bounded, emission order).
    artifacts: Vec<JobArtifactRef>,
}

impl JobEntry {
    /// The event-stream projection of this entry (with the lifecycle
    /// transition and the cancel-requested flag filled in).
    fn to_event(&self, event: JobEventType, cancelled: bool) -> JobEvent {
        JobEvent {
            // Placeholder: `EventBus::publish` stamps the real `seq`.
            seq: 0,
            event,
            id: self.id.clone(),
            kind: self.kind.label().to_owned(),
            model: self.model.clone(),
            started_at_ms: self.started_at_ms,
            interruptible: self.interruptible,
            cancelled,
            step: None,
            total_steps: None,
            pct: None,
            artifact_id: None,
            sha256: None,
            size_bytes: None,
            artifact_name: None,
            artifact_kind: None,
            artifact_index: None,
            artifact_primary: None,
            annotations: None,
        }
    }
}

/// The two independent facts a cancel establishes, kept separate so no
/// caller mistakes "the token was marked" for "the work can stop".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CancelOutcome {
    /// The id was live and its cancel token was marked (`job_cancelled`
    /// emitted). Always `true` for a found id.
    pub cancelled: bool,
    /// The route honors the token and can stop before the work completes.
    pub interruptible: bool,
}

/// Registry of live GPU jobs: id → entry. The map lives behind an `Arc` so
/// each [`JobGuard`] unregisters from ITS OWN registry on drop (a locally
/// built registry in tests behaves exactly like the global one). The
/// lifecycle-event broadcaster rides alongside: `register`, `cancel`, and
/// [`JobGuard::drop`] are the only emitters, all keyed by the registry.
/// Central event emit: one `AtomicU64` stamps every event `seq`, appends
/// it to the bounded replay window, and broadcasts it. All emitters
/// (`register`, `cancel`, `JobGuard::drop`, progress/result/annotations)
/// route through [`EventBus::publish`] — no call site stamps its own `seq`.
#[derive(Debug)]
struct EventBus {
    sender: broadcast::Sender<JobEvent>,
    seq: AtomicU64,
    history: Mutex<VecDeque<JobEvent>>,
}

impl EventBus {
    /// Stamp `seq`, retain in the replay window, broadcast fire-and-forget.
    /// Returns the assigned `seq` so `register` can key entry order on it.
    fn publish(&self, mut event: JobEvent) -> u64 {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        event.seq = seq;
        {
            let mut history = self
                .history
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            history.push_back(event.clone());
            while history.len() > EVENT_CHANNEL_CAPACITY {
                history.pop_front();
            }
        }
        // No live subscriber is a normal idle state, not an error
        // (observability never back-pressures GPU work).
        let _ = self.sender.send(event);
        seq
    }

    /// Events after `cursor`, oldest first. `Err(skipped)` when the cursor
    /// predates the retained window: the caller owes one `events_dropped`
    /// gap marker naming `skipped` instead of a replay.
    fn replay_since(&self, cursor: u64) -> Result<Vec<JobEvent>, u64> {
        let history = self
            .history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(oldest) = history.front().map(|event| event.seq) else {
            return Ok(Vec::new());
        };
        if cursor < oldest {
            let latest = history.back().map_or(oldest, |event| event.seq);
            return Err(latest - cursor);
        }
        Ok(history
            .iter()
            .filter(|event| event.seq > cursor)
            .cloned()
            .collect())
    }
}

#[derive(Debug)]
pub(crate) struct Registry {
    entries: Arc<Mutex<HashMap<String, JobEntry>>>,
    bus: Arc<EventBus>,
}

impl Default for Registry {
    fn default() -> Self {
        let (events, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        Self {
            entries: Arc::default(),
            bus: Arc::new(EventBus {
                sender: events,
                seq: AtomicU64::new(0),
                history: Mutex::new(VecDeque::new()),
            }),
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
        // Central emit stamps `seq`; the entry keys its oldest-first order
        // on the registered event `seq` (timestamps tie at ms resolution).
        let assigned = self.bus.publish(JobEvent {
            seq: 0,
            event: JobEventType::Registered,
            id: id.clone(),
            kind: kind.label().to_owned(),
            model: model.clone(),
            started_at_ms,
            interruptible,
            cancelled: false,
            step: None,
            total_steps: None,
            pct: None,
            artifact_id: None,
            sha256: None,
            size_bytes: None,
            artifact_name: None,
            artifact_kind: None,
            artifact_index: None,
            artifact_primary: None,
            annotations: None,
        });
        let entry = JobEntry {
            id: id.clone(),
            token: token.clone(),
            kind,
            model,
            started_at_ms,
            seq: assigned,
            interruptible,
            artifacts: Vec::new(),
        };
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id.clone(), entry);
        // The bounded job-kind counter (no user data reaches the label).
        crate::metrics::note_job_started(kind.label());
        JobGuard {
            entries: Arc::clone(&self.entries),
            bus: Arc::clone(&self.bus),
            id,
            token,
        }
    }

    /// Mark a live job's token and return BOTH facts separately. `Some` =
    /// the id was found (freshly marked or already cancelled in flight):
    /// `cancelled` says the token was marked, `interruptible` says the route
    /// can actually stop early. `None` = unknown id (never registered, or
    /// its request already ended). Emits `job_cancelled` — a cancel was
    /// REQUESTED, not a completion — when the id was found.
    pub(crate) fn cancel(&self, id: &str) -> Option<CancelOutcome> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = entries.get_mut(id)?;
        entry.token.cancel();
        let cancelled = entry.token.is_cancelled();
        let interruptible = entry.interruptible;
        // Central emit (same contract as `register`); publish touches
        // only the bus, never the entries map, so holding the entries lock
        // here cannot invert any lock order.
        self.bus
            .publish(entry.to_event(JobEventType::Cancelled, cancelled));
        Some(CancelOutcome {
            cancelled,
            interruptible,
        })
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
                artifacts: (!entry.artifacts.is_empty()).then(|| entry.artifacts.clone()),
            })
            .collect()
    }

    /// Subscribe to job lifecycle events (`GET /v1/events`). Only events
    /// emitted after the subscription are delivered live; pair with
    /// [`Registry::replay_since`] for cursor resume on reconnect.
    pub(crate) fn subscribe(&self) -> broadcast::Receiver<JobEvent> {
        self.bus.sender.subscribe()
    }

    /// Missed events after `cursor` from the retained window, oldest first
    /// (`GET /v1/events` reconnect replay). `Err(skipped)` when the cursor
    /// predates the window — the route owes one `events_dropped` frame.
    pub(crate) fn replay_since(&self, cursor: u64) -> Result<Vec<JobEvent>, u64> {
        self.bus.replay_since(cursor)
    }

    /// Emit an incremental `job_progress` event for a live job. Fire-and-forget
    /// like every other event; an unknown (already finished) id is a no-op.
    pub(crate) fn emit_progress(&self, id: &str, step: u32, total_steps: u32) {
        let mut event = {
            let entries = self
                .entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(entry) = entries.get(id) else {
                return;
            };
            entry.to_event(JobEventType::Progress, false)
        };
        event.step = Some(step);
        event.total_steps = Some(total_steps);
        // Percentage of work done; no percentage exists without a total.
        event.pct = (total_steps > 0)
            .then(|| ((f64::from(step) / f64::from(total_steps)) * 100.0).clamp(0.0, 100.0));
        // The entries lock is already released; send only touches the channel.
        self.bus.publish(event);
    }

    /// Emit a `job_result` event for a live job: one produced artifact,
    /// identified by its content hash and its producer's own selection
    /// metadata. The bounded snapshot gets the same identity so a dropped
    /// frame can be reconciled. Fire-and-forget; unknown id is a no-op.
    pub(crate) fn emit_result(
        &self,
        id: &str,
        artifact_id: &str,
        sha256: &str,
        size_bytes: u64,
        meta: ArtifactFrameMeta<'_>,
    ) {
        let artifact = JobArtifactRef {
            artifact_id: artifact_id.to_owned(),
            name: meta.name.to_owned(),
            kind: meta.kind.to_owned(),
            index: meta.index,
            primary: meta.primary,
        };
        let mut event = {
            let mut entries = self
                .entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(entry) = entries.get_mut(id) else {
                return;
            };
            if entry.artifacts.len() < MAX_ARTIFACT_REFS {
                entry.artifacts.push(artifact.clone());
            }
            entry.to_event(JobEventType::Result, false)
        };
        event.artifact_id = Some(artifact.artifact_id);
        event.sha256 = Some(sha256.to_owned());
        event.size_bytes = Some(size_bytes);
        event.artifact_name = Some(artifact.name);
        event.artifact_kind = Some(artifact.kind);
        event.artifact_index = Some(artifact.index);
        event.artifact_primary = Some(artifact.primary);
        self.bus.publish(event);
    }

    /// Re-emit a live job's `job_result` frame carrying its NSFW
    /// annotations. Reuses the same event path as `emit_result` — fire and
    /// forget, unknown id is a no-op — so no other emitter changes.
    pub(crate) fn emit_annotations(&self, id: &str, annotations: Vec<NsfwAnnotation>) {
        let mut event = {
            let entries = self
                .entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(entry) = entries.get(id) else {
                return;
            };
            entry.to_event(JobEventType::Result, false)
        };
        event.annotations = Some(annotations);
        self.bus.publish(event);
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
    bus: Arc<EventBus>,
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
            self.bus
                .publish(entry.to_event(JobEventType::Finished, cancelled));
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
    fn cancel_marks_token_and_reports_both_facts() {
        let registry = fresh();
        let job = registry.register(JobKind::Images, None, true);
        let token = job.token().clone();
        let both = CancelOutcome {
            cancelled: true,
            interruptible: true,
        };
        assert_eq!(registry.cancel(job.id()), Some(both));
        assert!(token.is_cancelled());
        // Idempotent: a second cancel still finds the live job.
        assert_eq!(registry.cancel(job.id()), Some(both));
        drop(job);
        // After the request ends the id is gone → the 404 path.
        assert_eq!(registry.cancel("job_unknown"), None);
    }

    /// F6 regression: a non-interruptible job still answers `cancelled:
    /// true` because its token WAS marked; `interruptible: false` is the
    /// separate fact. The old `Some(false)` answer made routes report
    /// `cancelled: false` for a job it had just cancelled.
    #[tokio::test]
    async fn non_interruptible_cancel_reports_token_marked_separately() {
        let registry = fresh();
        let mut events = registry.subscribe();
        let job = registry.register(JobKind::Transcribe, None, false);
        assert_eq!(
            next_event(&mut events).await.event,
            JobEventType::Registered
        );
        assert_eq!(
            registry.cancel(job.id()),
            Some(CancelOutcome {
                cancelled: true,
                interruptible: false,
            })
        );
        assert!(job.token().is_cancelled());
        let cancelled = next_event(&mut events).await;
        assert_eq!(cancelled.event, JobEventType::Cancelled);
        assert!(cancelled.cancelled, "job_cancelled marks the token");
        assert!(!cancelled.interruptible, "work cannot stop early");
        drop(job);
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

    #[tokio::test]
    async fn progress_and_result_events_carry_their_payloads() {
        let registry = fresh();
        let mut receiver = registry.subscribe();
        let job = registry.register(JobKind::Images, None, true);
        let _ = next_event(&mut receiver).await; // job_registered

        registry.emit_progress(job.id(), 1, 4);
        let progress = next_event(&mut receiver).await;
        assert_eq!(progress.event, JobEventType::Progress);
        assert_eq!(progress.step, Some(1));
        assert_eq!(progress.total_steps, Some(4));
        assert_eq!(progress.pct, Some(25.0));
        assert!(progress.artifact_id.is_none());

        registry.emit_progress(job.id(), 9, 4);
        assert_eq!(next_event(&mut receiver).await.pct, Some(100.0));
        registry.emit_progress(job.id(), 1, 0);
        assert_eq!(next_event(&mut receiver).await.pct, None);

        registry.emit_result(
            job.id(),
            "art_1",
            "abc123",
            42,
            ArtifactFrameMeta {
                name: "final_00.png",
                kind: "image",
                index: 0,
                primary: true,
            },
        );
        let result = next_event(&mut receiver).await;
        assert_eq!(result.event, JobEventType::Result);
        assert_eq!(result.artifact_id.as_deref(), Some("art_1"));
        assert_eq!(result.sha256.as_deref(), Some("abc123"));
        assert_eq!(result.size_bytes, Some(42));
        assert_eq!(result.artifact_name.as_deref(), Some("final_00.png"));
        assert_eq!(result.artifact_kind.as_deref(), Some("image"));
        assert_eq!(result.artifact_index, Some(0));
        assert_eq!(result.artifact_primary, Some(true));
        assert!(result.step.is_none());

        // An id that is not live emits nothing.
        registry.emit_result(
            "job_unknown",
            "art_2",
            "def",
            1,
            ArtifactFrameMeta {
                name: "final_00.png",
                kind: "image",
                index: 1,
                primary: false,
            },
        );
        assert!(receiver.try_recv().is_err());
    }

    /// Result frames select one primary artifact; the live snapshot carries
    /// the same identities and loses them when the guard evicts the entry.
    #[tokio::test]
    async fn result_refs_snapshot_primary_and_eviction() {
        let registry = fresh();
        let mut receiver = registry.subscribe();
        let job = registry.register(JobKind::Blender, None, false);
        let _ = next_event(&mut receiver).await; // job_registered

        registry.emit_result(
            job.id(),
            "art_preview",
            "prev",
            3,
            ArtifactFrameMeta {
                name: "preview.webp",
                kind: "preview",
                index: 0,
                primary: false,
            },
        );
        registry.emit_result(
            job.id(),
            "art_model",
            "model",
            9,
            ArtifactFrameMeta {
                name: "output.glb",
                kind: "model",
                index: 1,
                primary: true,
            },
        );
        let first = next_event(&mut receiver).await;
        let second = next_event(&mut receiver).await;
        assert_eq!(first.artifact_name.as_deref(), Some("preview.webp"));
        assert_eq!(first.artifact_kind.as_deref(), Some("preview"));
        assert_eq!(first.artifact_index, Some(0));
        assert_eq!(first.artifact_primary, Some(false));
        assert_eq!(second.artifact_name.as_deref(), Some("output.glb"));
        assert_eq!(second.artifact_kind.as_deref(), Some("model"));
        assert_eq!(second.artifact_index, Some(1));
        assert_eq!(second.artifact_primary, Some(true));

        let listed = registry.list();
        let refs = listed[0].artifacts.as_ref().expect("artifacts");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].artifact_id, "art_preview");
        assert_eq!(refs[1].artifact_id, "art_model");
        assert_eq!(refs.iter().filter(|row| row.primary).count(), 1);

        // The annotation re-emit is artifact-less: no second primary.
        registry.emit_annotations(job.id(), Vec::new());
        let annotations = next_event(&mut receiver).await;
        assert!(annotations.annotations.is_some());
        assert!(annotations.artifact_id.is_none());
        assert!(annotations.artifact_primary.is_none());
        assert_eq!(
            registry.list()[0]
                .artifacts
                .as_ref()
                .expect("artifacts")
                .len(),
            2
        );

        drop(job);
        assert!(registry.list().is_empty(), "eviction drops the snapshot");
    }

    /// A manifest with many outputs cannot grow a live entry without bound.
    #[test]
    fn result_refs_are_bounded() {
        let registry = fresh();
        let job = registry.register(JobKind::Blender, None, false);
        for index in 0..70 {
            registry.emit_result(
                job.id(),
                &format!("art_{index}"),
                "sha",
                1,
                ArtifactFrameMeta {
                    name: "output.glb",
                    kind: "model",
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    primary: index == 0,
                },
            );
        }
        let refs = registry.list()[0]
            .artifacts
            .as_ref()
            .expect("artifacts")
            .clone();
        assert_eq!(refs.len(), MAX_ARTIFACT_REFS);
        assert_eq!(refs[0].artifact_id, "art_0");
        assert_eq!(refs[MAX_ARTIFACT_REFS - 1].artifact_id, "art_63");
    }

    /// A consumer built for the original three result keys still decodes a
    /// frame; the additive keys are simply absent on the wire.
    #[test]
    fn legacy_result_frame_decodes_without_new_keys() {
        #[derive(serde::Deserialize)]
        struct Legacy {
            event: String,
            artifact_id: Option<String>,
            sha256: Option<String>,
            size_bytes: Option<u64>,
        }

        let registry = fresh();
        let mut receiver = registry.subscribe();
        let job = registry.register(JobKind::Images, None, true);
        let registered = receiver.try_recv().expect("registered frame");
        let registered_json = serde_json::to_value(&registered).expect("serialize");
        for key in [
            "artifact_name",
            "artifact_kind",
            "artifact_index",
            "artifact_primary",
        ] {
            assert!(
                registered_json.get(key).is_none(),
                "{key} absent on non-result frames: {registered_json}"
            );
        }
        registry.emit_result(
            job.id(),
            "art_1",
            "abc",
            3,
            ArtifactFrameMeta {
                name: "final_00.png",
                kind: "image",
                index: 0,
                primary: true,
            },
        );
        let event = receiver.try_recv().expect("result frame");
        let json = serde_json::to_value(&event).expect("serialize");
        assert_eq!(json["artifact_name"], "final_00.png");
        let legacy: Legacy = serde_json::from_value(json).expect("legacy decode");
        assert_eq!(legacy.event, "job_result");
        assert_eq!(legacy.artifact_id.as_deref(), Some("art_1"));
        assert_eq!(legacy.sha256.as_deref(), Some("abc"));
        assert_eq!(legacy.size_bytes, Some(3));
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

        assert_eq!(
            registry.cancel(job.id()),
            Some(CancelOutcome {
                cancelled: true,
                interruptible: true,
            })
        );
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
    /// Reconnect replay: `replay_since` returns exactly the missed tail,
    /// oldest-first with monotonic `seq`. Five registrations stamp seq
    /// 0..=4 on the fresh registry, so a cursor of 1 selects seq 2, 3, 4.
    #[test]
    fn replay_since_returns_missed_tail_oldest_first() {
        let registry = fresh();
        // Hold the guards: dropping a job emits `job_finished` and would
        // pollute the replay window under test.
        let _held = (0..5)
            .map(|_| registry.register(JobKind::Chat, None, false))
            .collect::<Vec<_>>();
        let tail = registry.replay_since(1).expect("cursor inside window");
        assert_eq!(tail.len(), 3, "seq 2,3,4 follow cursor 1");
        let seqs = tail.iter().map(|event| event.seq).collect::<Vec<_>>();
        assert_eq!(seqs, [2, 3, 4], "oldest-first, no gaps");
        assert!(
            tail.iter()
                .all(|event| event.event == JobEventType::Registered),
            "only registrations were emitted"
        );
    }

    #[test]
    fn replay_since_at_head_is_empty_ok() {
        let registry = fresh();
        let _held = registry.register(JobKind::Chat, None, false);
        assert!(registry.replay_since(0).expect("cursor at head").is_empty());
        // A cursor past the head is also empty, not an error.
        assert!(
            registry
                .replay_since(10_000)
                .expect("cursor past head")
                .is_empty()
        );
    }

    /// A cursor predating the retained window is an `Err(skipped)` naming
    /// the dropped count — the route owes a gap marker, not a replay.
    /// `EVENT_CHANNEL_CAPACITY + 10` registrations evict seq 0..=9, so
    /// cursor 0 reports `latest - cursor` (265) as skipped.
    #[test]
    fn replay_since_predating_window_reports_skipped() {
        let registry = fresh();
        let _held = (0..EVENT_CHANNEL_CAPACITY + 10)
            .map(|_| registry.register(JobKind::Chat, None, false))
            .collect::<Vec<_>>();
        match registry.replay_since(0) {
            Err(skipped) => assert_eq!(skipped, 265),
            Ok(_) => panic!("cursor 0 predates the window"),
        }
        // The window edge itself still replays: cursor 10 selects seq 11
        // oldest-first.
        let edge = registry.replay_since(10).expect("window edge replays");
        assert_eq!(edge.len(), EVENT_CHANNEL_CAPACITY - 1);
        assert_eq!(edge[0].seq, 11);
        assert!(
            edge.windows(2).all(|pair| pair[0].seq + 1 == pair[1].seq),
            "monotonic, no gaps"
        );
    }
}
