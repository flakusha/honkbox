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

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio_util::sync::CancellationToken;

use honkbox_api_types::JobInfo;

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

/// Registry of live GPU jobs: id → entry. The map lives behind an `Arc` so
/// each [`JobGuard`] unregisters from ITS OWN registry on drop (a locally
/// built registry in tests behaves exactly like the global one).
#[derive(Debug, Default)]
pub(crate) struct Registry {
    entries: Arc<Mutex<HashMap<String, JobEntry>>>,
    seq: AtomicU64,
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
        let entry = JobEntry {
            id: id.clone(),
            token: token.clone(),
            kind,
            model,
            started_at_ms: now_ms(),
            seq: self.seq.fetch_add(1, Ordering::Relaxed),
            interruptible,
        };
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id.clone(), entry);
        JobGuard {
            entries: Arc::clone(&self.entries),
            id,
            token,
        }
    }

    /// Mark a live job's token. `Some(interruptible)` = the id was found
    /// (freshly marked or already cancelled while still in flight); `None`
    /// = unknown id (never registered, or its request already ended).
    pub(crate) fn cancel(&self, id: &str) -> Option<bool> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = entries.get_mut(id)?;
        entry.token.cancel();
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
}

/// Registry id + cancel token + a handle to the OWNING registry; dropping
/// it unregisters the job from that registry, so a guard must live exactly
/// as long as the request that owns the work.
#[derive(Debug)]
pub(crate) struct JobGuard {
    entries: Arc<Mutex<HashMap<String, JobEntry>>>,
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
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.id);
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
}
