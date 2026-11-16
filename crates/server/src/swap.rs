//! LLM swap manager (P1 of the llama-swap parity plan): load-on-demand +
//! idle-TTL/LRU eviction for chat pipelines.
//!
//! sd-rip loads pipelines in-process behind `Arc<dyn LlmPipeline>`, so the
//! llama-swap process/proxy machinery collapses to a cache map: an entry is
//! the loaded pipeline plus its idle bookkeeping. Semantics ported from
//! llama-swap §1.4 (`.tmp/research/ReviewedLlamaSwap-Parity.md`):
//!
//! - **Load-on-demand**: a request for uncached slug S constructs S via the
//!   [`LlmLoader`] seam and caches it; cached slugs reuse the pipeline. A
//!   second request never re-constructs.
//! - **TTL**: idle time since the entry's last use; a request resets the
//!   timer. An optional per-request `keep_alive` overrides the configured
//!   budget for that bump (NEW-8). `keep_alive = 0` makes the entry evictable
//!   at the very next sweep.
//! - **LRU beyond `llm_max_loaded`**: enforced *before* a slow load, not
//!   after — the previous resident is dropped first, so the GPU is never
//!   double-committed mid-swap (parity §6.2). `llm_max_loaded = 1` (the
//!   default) is classic swap: the incoming model replaces all residents.
//! - **Eviction trigger**: lazy (before loads) plus a periodic reaper
//!   ([`SwapManager::spawn_reaper`], started by `serve`) for pure idle
//!   expiry; sweep granularity bounds how soon an expired entry actually
//!   frees.
//!
//! Arc lifetime (in-flight safety): eviction removes the entry from the map,
//! dropping only the cache's `Arc`. Every in-flight request holds its own
//! clone, so the pipeline stays alive until the last holder drops —
//! llama-swap's `trackedServe` drain, for free (parity §1.2). `LlamaCppLlm`
//! owns its `llama-cpp-4` model handles, so the final drop returns the
//! model's memory to the allocator.
//!
//! Concurrency: constructions serialize on an async load gate; the cache is
//! re-checked after acquiring (double-checked load), so racing requests for
//! the same slug load once. Loads running while another request *infers* are
//! not prevented — the residency slot is still the only inference gate and
//! 429 semantics are unchanged; admission scheduling that also gates loads
//! is the P2 queue's job (parity §5 P2).
//!
//! Scope: LLM only. SD/audio stay startup-loaded — their pipeline
//! constructors differ (diffusion-rs/whisper/kokoro assemblies) and each
//! needs its own eviction cost model; tight scope per the P1 ticket.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use honkbox_api_types::{ModelCatalog, ModelKind};
use honkbox_llm_worker::{LlamaCppLlm, LlmOptions, LlmPipeline};
use tokio::sync::Mutex as AsyncMutex;

/// Slug → pipeline constructor seam. Production: catalog-backed llama.cpp
/// load ([`catalog_llm_loader`]); tests: counting stubs. Called on the
/// blocking pool — may take seconds-to-minutes for real weights.
pub type LlmLoader = Arc<dyn Fn(&str) -> Result<Arc<dyn LlmPipeline>, SwapError> + Send + Sync>;

/// Tunables mirroring the `[swap]` config section (`config::SwapConfig`).
#[derive(Debug, Clone)]
pub struct SwapSettings {
    /// Idle time before an entry is evictable; the default applied when a
    /// request carries no `keep_alive` override.
    pub llm_ttl: Duration,
    /// Warm-pipeline capacity; LRU evicts beyond it. `1` = classic swap.
    pub llm_max_loaded: usize,
}

/// Swap failures, mapped to HTTP by the routes (`UnknownSlug` → 400, the
/// same contract as `validate_model`; `Load` → 500).
#[derive(Debug)]
pub enum SwapError {
    /// Slug absent from the catalog, or present but not LLM weights.
    UnknownSlug(String),
    /// Pipeline construction failed; the cache stays consistent (nothing
    /// inserted, other entries untouched).
    Load(String),
}

impl fmt::Display for SwapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSlug(msg) | Self::Load(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for SwapError {}

/// One cached pipeline with its idle bookkeeping.
struct Entry {
    /// The loaded pipeline; dropping the last clone frees the model.
    pipe: Arc<dyn LlmPipeline>,
    /// Last request touch (or load time); the TTL window start.
    last_use: Instant,
    /// Idle budget for this entry: the configured TTL, or the `keep_alive`
    /// override of the most recent touch.
    ttl: Duration,
}

/// LLM pipeline cache: load-on-demand, per-entry idle TTL, LRU capacity.
pub struct SwapManager {
    settings: SwapSettings,
    loader: LlmLoader,
    /// Cache map; short critical sections only (std `Mutex`), never held
    /// across loads or awaits.
    entries: Mutex<HashMap<String, Entry>>,
    /// Construction lock: held across the slow load so concurrent requests
    /// for uncached slugs serialize instead of racing (`tokio` mutex — the
    /// guard is held across `spawn_blocking().await`).
    load_gate: AsyncMutex<()>,
}

impl SwapManager {
    /// Build a manager around a constructor seam.
    #[must_use]
    pub fn new(settings: SwapSettings, loader: LlmLoader) -> Self {
        Self {
            settings,
            loader,
            entries: Mutex::new(HashMap::new()),
            load_gate: AsyncMutex::new(()),
        }
    }

    /// Pre-warm a cache entry with an already-built pipeline (the startup
    /// eager load / llama-swap `preload` analog): the configured slug answers
    /// instantly, and the TTL reaper may free it after the idle window. A
    /// present entry wins — seeding never clobbers a loaded pipeline.
    pub fn seed(&self, slug: &str, pipe: Arc<dyn LlmPipeline>) {
        let mut entries = self.lock_entries();
        entries.entry(slug.to_owned()).or_insert_with(|| Entry {
            pipe,
            last_use: Instant::now(),
            ttl: self.settings.llm_ttl,
        });
    }

    /// Pipeline for `slug`, loading it on demand. `keep_alive` (seconds)
    /// overrides the configured TTL for this touch; `Some(0)` makes the
    /// entry evictable at the next sweep.
    ///
    /// # Errors
    /// [`SwapError::UnknownSlug`] when the loader rejects the slug;
    /// [`SwapError::Load`] when construction fails (cache unchanged).
    pub async fn get(
        &self,
        slug: &str,
        keep_alive: Option<u64>,
    ) -> Result<Arc<dyn LlmPipeline>, SwapError> {
        let ttl = keep_alive.map_or(self.settings.llm_ttl, Duration::from_secs);
        if let Some(pipe) = self.touch(slug, ttl) {
            return Ok(pipe);
        }
        // Slow path: serialize constructions, then re-check — a racing
        // request may have loaded the slug while we waited on the gate.
        let gate = self.load_gate.lock().await;
        if let Some(pipe) = self.touch(slug, ttl) {
            drop(gate);
            return Ok(pipe);
        }
        // Make room *before* the load: drop idle-expired entries, then LRU
        // down to capacity - 1 so the incoming model never double-commits
        // the GPU with a resident it is replacing (parity §6.2). In-flight
        // requests keep their clones alive until they finish.
        self.trim_for_load();
        let loader = Arc::clone(&self.loader);
        let owned_slug = slug.to_owned();
        let pipe = tokio::task::spawn_blocking(move || loader(&owned_slug))
            .await
            .map_err(|err| SwapError::Load(format!("model load task failed: {err}")))??;
        self.lock_entries().insert(
            slug.to_owned(),
            Entry {
                pipe: Arc::clone(&pipe),
                last_use: Instant::now(),
                ttl,
            },
        );
        drop(gate);
        Ok(pipe)
    }

    /// Bump last-use + TTL budget for a cached slug; `None` when absent.
    fn touch(&self, slug: &str, ttl: Duration) -> Option<Arc<dyn LlmPipeline>> {
        let mut entries = self.lock_entries();
        entries.get_mut(slug).map(|entry| {
            entry.last_use = Instant::now();
            entry.ttl = ttl;
            Arc::clone(&entry.pipe)
        })
    }

    /// Evict every entry idle beyond its TTL budget. Called by the reaper;
    /// exposed for deterministic tests (they pass short TTLs and call this
    /// directly instead of sleeping on the reaper period).
    ///
    /// Returns the number of entries evicted.
    pub fn evict_idle(&self) -> usize {
        let now = Instant::now();
        let mut entries = self.lock_entries();
        let before = entries.len();
        entries.retain(|_slug, entry| now.duration_since(entry.last_use) <= entry.ttl);
        before - entries.len()
    }

    /// Pre-load capacity trim: TTL-expired entries go first, then LRU down
    /// to `llm_max_loaded - 1` (keeping room for the incoming load). Called
    /// under the load gate with the target slug absent from the map.
    fn trim_for_load(&self) {
        let keep = self.settings.llm_max_loaded.saturating_sub(1);
        let now = Instant::now();
        let mut entries = self.lock_entries();
        entries.retain(|_slug, entry| now.duration_since(entry.last_use) <= entry.ttl);
        trim_lru(&mut entries, keep);
    }

    /// Short-lock helper mapping poisoning back into a guard (a poisoned
    /// swap-cache mutex means a panic already tore through; failing loudly
    /// beats serving an inconsistent cache).
    fn lock_entries(&self) -> std::sync::MutexGuard<'_, HashMap<String, Entry>> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Drop the least-recently-used entries until at most `keep` remain.
fn trim_lru(entries: &mut HashMap<String, Entry>, keep: usize) {
    if entries.len() <= keep {
        return;
    }
    // Owned slugs: the eviction loop mutates the map the order was built
    // from.
    let mut order: Vec<(String, Instant)> = entries
        .iter()
        .map(|(slug, entry)| (slug.clone(), entry.last_use))
        .collect();
    order.sort_by_key(|&(_, last_use)| last_use);
    for (slug, _) in order {
        if entries.len() <= keep {
            break;
        }
        entries.remove(&slug);
    }
}

/// Periodic idle sweep: evicts entries whose TTL expired between requests.
/// The period is half the TTL clamped to `[1s, 30s]` — short TTLs sweep
/// often enough to feel immediate, long TTLs never busy-loop. Eviction is
/// best-effort bookkeeping: a missed sweep only delays a free, never
/// corrupts state.
#[must_use]
pub fn spawn_reaper(manager: Arc<SwapManager>) -> tokio::task::JoinHandle<()> {
    let period =
        (manager.settings.llm_ttl / 2).clamp(Duration::from_secs(1), Duration::from_secs(30));
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(period);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            let evicted = manager.evict_idle();
            if evicted > 0 {
                tracing::debug!("swap: evicted {evicted} idle LLM pipeline(s)");
            }
        }
    })
}

/// Production loader: resolve a catalog slug to its on-disk path via the
/// current `/v1/models` snapshot (stays fresh across
/// `POST /v1/models/refresh`) and load it with `LlamaCppLlm` + default
/// options. Non-LLM catalog entries (STT/TTS/SD weights) and unknown slugs
/// are [`SwapError::UnknownSlug`] — a 400 at the route.
#[must_use]
pub fn catalog_llm_loader(catalog: Arc<RwLock<Arc<ModelCatalog>>>) -> LlmLoader {
    Arc::new(move |slug: &str| {
        let snapshot = catalog
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let entry = snapshot.data.iter().find(|entry| entry.id == slug);
        let Some(entry) = entry else {
            return Err(SwapError::UnknownSlug(format!(
                "unknown model {slug:?}: not a catalog slug (see GET /v1/models)"
            )));
        };
        if entry.kind != ModelKind::Llm {
            return Err(SwapError::UnknownSlug(format!(
                "model {slug:?} is not LLM weights; the chat pipeline loads `llm:` slugs only"
            )));
        }
        let path = std::path::PathBuf::from(&entry.path);
        match LlamaCppLlm::load(&path, &LlmOptions::default()) {
            Ok(pipe) => Ok(Arc::new(pipe) as Arc<dyn LlmPipeline>),
            Err(err) => Err(SwapError::Load(format!(
                "loading {slug:?} from {} failed: {err}",
                path.display()
            ))),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the settings plumbing: clone + equality on the tunables the
    /// config layer feeds in (real behavior lives in the integration tests).
    #[test]
    fn settings_clone_roundtrip() {
        let settings = SwapSettings {
            llm_ttl: Duration::from_secs(600),
            llm_max_loaded: 1,
        };
        let cloned = settings.clone();
        assert_eq!(cloned.llm_ttl, settings.llm_ttl);
        assert_eq!(cloned.llm_max_loaded, settings.llm_max_loaded);
    }
}
