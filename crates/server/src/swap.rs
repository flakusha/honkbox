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
//!   timer. A per-request `ttl_seconds` (NEW-8, clamped to `max_ttl` at the
//!   route) raises *that residency's* budget: the effective window is
//!   `max(remaining, own)`, re-capped at `max_ttl`. Windows never shrink on
//!   a touch, so `ttl_seconds = 0` (Ollama use-then-unload semantics) only
//!   bites once the previously granted budget is spent — the anti-thrash
//!   shield that keeps mixed-TTL traffic from reload-per-request ping-pong
//!   (parity §6.3). Eviction is lazy, so `ttl = 0` frees at the next sweep,
//!   never mid-request.
//! - **Pin** (`pinned` slugs from `[swap].pin`): never auto-evict (TTL or
//!   LRU) and ignore per-request `ttl_seconds` entirely — pin outranks
//!   client advice. Pinning past `llm_max_loaded` accepts the co-residency.
//! - **LRU beyond `llm_max_loaded`**: enforced *before* a slow load, not
//!   after — the previous resident is dropped first, so the GPU is never
//!   double-committed mid-swap (parity §6.2). `llm_max_loaded = 1` (the
//!   default) is classic swap: the incoming model replaces all residents.
//! - **Eviction trigger**: lazy (before loads) plus a periodic reaper
//!   ([`SwapManager::spawn_reaper`], started by `serve`) for pure idle
//!   expiry; sweep granularity bounds how soon an expired entry actually
//!   frees.
//! - **Explicit unload** (REQ-5): [`SwapManager::evict_by_slug`] removes one
//!   resident on API request (`POST /v1/models/{slug}/unload` and the
//!   llama-swap-compat `/api/models/unload` forms); pinned residents are
//!   refused ([`EvictOutcome::Pinned`] → 409 at the route); non-resident ids
//!   are an idempotent no-op. [`SwapManager::residency`] is the readout
//!   behind `GET /v1/memory` / `GET /running`.
//!
//! Metrics: load commits, removals, capacity evictions, and load durations
//! are emitted through [`crate::metrics`] with kind `llm` (the manager's
//! only modality — [`catalog_llm_loader`] refuses every other kind).
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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use honkbox_api_types::{ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_llm_worker::{
    EmbedPipeline, KvQuant, LlamaCppEmbed, LlamaCppLlm, LlmOptions, LlmPipeline,
};
use tokio::sync::Mutex as AsyncMutex;

use crate::config::{CacheConfig, KvQuantConfig};
use crate::metrics::{
    LoadReason, UnloadReason, note_cache_hit, note_cache_miss, note_model_load,
    note_model_load_duration, note_model_unload, note_swap_eviction,
};

/// The manager's only modality: [`catalog_llm_loader`] refuses every other
/// catalog kind, so all swap-metric observations carry this kind. Per-kind
/// swap actors for sd/stt/tts (P2) will pass their own kinds.
const SWAP_KIND: ModelKind = ModelKind::Llm;

/// Slug → pipeline constructor seam. Production: catalog-backed llama.cpp
/// load ([`catalog_llm_loader`]); tests: counting stubs. Called on the
/// blocking pool — may take seconds-to-minutes for real weights.
pub type LlmLoader = Arc<dyn Fn(&str) -> Result<Arc<dyn LlmPipeline>, SwapError> + Send + Sync>;

/// Tunables mirroring the `[swap]` config section (`config::SwapConfig`).
#[derive(Debug, Clone)]
pub struct SwapSettings {
    /// Idle time before an entry is evictable; the default applied when a
    /// request carries no `ttl_seconds` advice.
    pub llm_ttl: Duration,
    /// Warm-pipeline capacity; LRU evicts beyond it. `1` = classic swap.
    pub llm_max_loaded: usize,
    /// Ceiling on any per-entry residency window: per-request `ttl_seconds`
    /// advice is capped here (the route clamps first; this re-caps
    /// defensively). Also caps a configured `llm_ttl` above it.
    pub max_ttl: Duration,
    /// Slugs that never auto-evict and ignore per-request `ttl_seconds`
    /// entirely (pin outranks client advice).
    pub pinned: HashSet<String>,
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
    /// Idle budget for this entry: the configured TTL, or the `ttl_seconds`
    /// advice of the most recent touch, `max`-ed with the remaining window
    /// and capped at `max_ttl`. Unused for pinned entries (they never
    /// TTL-evict).
    ttl: Duration,
    /// Load time in Unix seconds (the `/v1/memory` `loaded_at` readout;
    /// `Instant` has no epoch anchor).
    loaded_at: u64,
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

/// One resident entry of the [`SwapManager`] readout (`GET /v1/memory` /
/// `GET /running` data source).
#[derive(Debug, Clone)]
pub struct ResidentEntry {
    /// Owning registry slug.
    pub slug: String,
    /// Load time (Unix seconds) - captured at insert, not derived from
    /// `Instant` (which has no epoch anchor).
    pub loaded_at: u64,
    /// Idle-window end (Unix seconds); `None` for pinned entries, which
    /// carry no TTL budget.
    pub expires_at: Option<u64>,
    /// `[swap].pin` member.
    pub pinned: bool,
    /// Best-effort in-flight flag: `Arc::strong_count(&pipe) > 1` - some
    /// holder besides the manager owns the pipeline right now (an
    /// in-flight request's clone). Racy by nature (the count can change
    /// between read and response), but honest for a diagnostic readout:
    /// `false` never claims activity, `true` always reflects a real extra
    /// holder at snapshot time.
    pub in_use: bool,
}

/// Outcome of [`SwapManager::evict_by_slug`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictOutcome {
    /// A resident entry was removed (the in-flight `Arc` clones keep the
    /// pipeline alive until the last holder drops - no request is aborted).
    Evicted,
    /// The slug had no resident entry (idempotent no-op).
    NotLoaded,
    /// The slug is resident but `[swap].pin`-pinned: refusal is explicit,
    /// the route maps this to a 409.
    Pinned,
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
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
        entries.entry(slug.to_owned()).or_insert_with(|| {
            note_model_load(SWAP_KIND, LoadReason::Preload);
            Entry {
                pipe,
                last_use: Instant::now(),
                ttl: self.settings.llm_ttl.min(self.settings.max_ttl),
                loaded_at: epoch_secs(),
            }
        });
    }

    /// Pipeline for `slug`, loading it on demand. `ttl_seconds` (NEW-8)
    /// raises this residency's budget to `max(remaining, own)`, capped at
    /// `max_ttl`; `Some(0)` = use-then-unload, effective once the remaining
    /// budget is spent. Pinned slugs ignore the advice entirely.
    ///
    /// # Errors
    /// [`SwapError::UnknownSlug`] when the loader rejects the slug;
    /// [`SwapError::Load`] when construction fails (cache unchanged).
    pub async fn get(
        &self,
        slug: &str,
        ttl_seconds: Option<u64>,
    ) -> Result<Arc<dyn LlmPipeline>, SwapError> {
        let requested = ttl_seconds.map_or(self.settings.llm_ttl, Duration::from_secs);
        if let Some(pipe) = self.touch(slug, requested) {
            return Ok(pipe);
        }
        // Slow path: serialize constructions, then re-check — a racing
        // request may have loaded the slug while we waited on the gate.
        let gate = self.load_gate.lock().await;
        if let Some(pipe) = self.touch(slug, requested) {
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
        let started = Instant::now();
        let pipe = tokio::task::spawn_blocking(move || loader(&owned_slug))
            .await
            .map_err(|err| SwapError::Load(format!("model load task failed: {err}")))??;
        note_model_load(SWAP_KIND, LoadReason::Request);
        note_model_load_duration(SWAP_KIND, started.elapsed());
        note_cache_miss(SWAP_KIND);
        self.lock_entries().insert(
            slug.to_owned(),
            Entry {
                pipe: Arc::clone(&pipe),
                last_use: Instant::now(),
                ttl: requested.min(self.settings.max_ttl),
                loaded_at: epoch_secs(),
            },
        );
        drop(gate);
        Ok(pipe)
    }

    /// Bump last-use + window budget for a cached slug; `None` when absent.
    /// The effective window is `max(remaining, own)`, capped — advice never
    /// shortens a window another touch granted (`ttl_seconds = 0` cannot
    /// thrash an entry that still has budget left). Pinned slugs keep their
    /// configured window (they never TTL-evict anyway).
    fn touch(&self, slug: &str, requested: Duration) -> Option<Arc<dyn LlmPipeline>> {
        let mut entries = self.lock_entries();
        let hit = entries.get_mut(slug).map(|entry| {
            let now = Instant::now();
            if !self.settings.pinned.contains(slug) {
                let remaining = entry.ttl.saturating_sub(now.duration_since(entry.last_use));
                entry.ttl = remaining.max(requested).min(self.settings.max_ttl);
            }
            entry.last_use = now;
            Arc::clone(&entry.pipe)
        });
        if hit.is_some() {
            note_cache_hit(SWAP_KIND);
        }
        hit
    }

    /// Evict every non-pinned entry idle beyond its TTL budget. Called by
    /// the reaper; exposed for deterministic tests (they pass short TTLs and
    /// call this directly instead of sleeping on the reaper period).
    ///
    /// Returns the number of entries evicted.
    pub fn evict_idle(&self) -> usize {
        let now = Instant::now();
        let mut entries = self.lock_entries();
        let before = entries.len();
        entries.retain(|slug, entry| {
            if self.settings.pinned.contains(slug)
                || now.duration_since(entry.last_use) <= entry.ttl
            {
                return true;
            }
            note_model_unload(SWAP_KIND, UnloadReason::Ttl);
            false
        });
        before - entries.len()
    }

    /// Evict one slug on explicit request (`POST /v1/models/{slug}/unload`):
    /// resident non-pinned entries are removed and counted as
    /// `UnloadReason::Explicit`; pinned residents are refused
    /// ([`EvictOutcome::Pinned`] - the route answers 409); an absent slug is
    /// an idempotent [`EvictOutcome::NotLoaded`]. In-flight requests keep
    /// their `Arc` clones - eviction only drops the cache's own handle, so
    /// running inference is never aborted (llama-swap `trackedServe` drain
    /// semantics, for free).
    pub fn evict_by_slug(&self, slug: &str, reason: UnloadReason) -> EvictOutcome {
        let mut entries = self.lock_entries();
        if !entries.contains_key(slug) {
            return EvictOutcome::NotLoaded;
        }
        if self.settings.pinned.contains(slug) {
            return EvictOutcome::Pinned;
        }
        entries.remove(slug);
        note_model_unload(SWAP_KIND, reason);
        EvictOutcome::Evicted
    }

    /// Evict every resident non-pinned entry (the llama-swap-compat
    /// `POST /api/models/unload` bulk form). Pinned residents are skipped,
    /// never refused. Returns the number of entries evicted.
    pub fn evict_all_unpinned(&self, reason: UnloadReason) -> usize {
        let mut entries = self.lock_entries();
        let before = entries.len();
        entries.retain(|slug, _| self.settings.pinned.contains(slug));
        let evicted = before - entries.len();
        for _ in 0..evicted {
            note_model_unload(SWAP_KIND, reason);
        }
        evicted
    }

    /// Residency snapshot (`GET /v1/memory` / `GET /running`): exactly the
    /// entries the manager considers resident, under one short read lock.
    /// Sorted by slug for deterministic output.
    #[must_use]
    pub fn residency(&self) -> Vec<ResidentEntry> {
        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let now = Instant::now();
        let entries = self.lock_entries();
        let mut out: Vec<ResidentEntry> = entries
            .iter()
            .map(|(slug, entry)| {
                let pinned = self.settings.pinned.contains(slug);
                let expires_at = if pinned {
                    None
                } else {
                    let remaining = entry.ttl.saturating_sub(now.duration_since(entry.last_use));
                    Some(now_epoch + remaining.as_secs())
                };
                ResidentEntry {
                    slug: slug.clone(),
                    loaded_at: entry.loaded_at,
                    expires_at,
                    pinned,
                    in_use: Arc::strong_count(&entry.pipe) > 1,
                }
            })
            .collect();
        out.sort_by(|a, b| a.slug.cmp(&b.slug));
        out
    }

    /// Configured warm-set capacity (`GET /v1/memory`'s `capacity` block):
    /// `(llm_max_loaded, llm_ttl_secs)`.
    #[must_use]
    pub fn capacity(&self) -> (usize, u64) {
        (
            self.settings.llm_max_loaded,
            self.settings.llm_ttl.as_secs(),
        )
    }

    /// Pre-load capacity trim: TTL-expired entries go first, then LRU down
    /// to `llm_max_loaded - 1` (keeping room for the incoming load). Pinned
    /// entries never auto-evict. Called under the load gate with the target
    /// slug absent from the map.
    fn trim_for_load(&self) {
        let keep = self.settings.llm_max_loaded.saturating_sub(1);
        let now = Instant::now();
        let mut entries = self.lock_entries();
        entries.retain(|slug, entry| {
            if self.settings.pinned.contains(slug)
                || now.duration_since(entry.last_use) <= entry.ttl
            {
                return true;
            }
            note_model_unload(SWAP_KIND, UnloadReason::Ttl);
            false
        });
        trim_lru(&mut entries, keep, &self.settings.pinned);
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
/// Pinned slugs never auto-evict (pin outranks capacity: the user accepted
/// the co-residency when pinning); every capacity swap-out counts an
/// unload (`evict`) plus a `honkbox_swap_evictions_total` observation.
fn trim_lru(entries: &mut HashMap<String, Entry>, keep: usize, pinned: &HashSet<String>) {
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
        if pinned.contains(&slug) {
            continue;
        }
        entries.remove(&slug);
        note_model_unload(SWAP_KIND, UnloadReason::Evict);
        note_swap_eviction(SWAP_KIND);
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
            // The resource ledger's queue-TTL sweep rides the same heartbeat
            // (TASK-resource-ledger-bounded-queue): waiters that outlived
            // their bound are evicted even if their polling request went
            // away. Best-effort, like the idle evictions above.
            let swept = honkbox_residency::global().sweep_expired();
            if swept > 0 {
                tracing::debug!("ledger: swept {swept} expired queue ticket(s)");
            }
        }
    })
}

/// The worker options for a slug: the declared cache knobs, else the
/// backend defaults, plus the global VRAM ceiling the KV cache is clamped
/// to by the worker.
pub(crate) fn llm_options(cache: CacheConfig, vram_budget_bytes: Option<u64>) -> LlmOptions {
    let defaults = LlmOptions::default();
    LlmOptions {
        n_ctx: cache.n_ctx.unwrap_or(defaults.n_ctx),
        kv_quant: cache.kv_quant.map_or(defaults.kv_quant, kv_quant_of),
        offload_kqv: cache.offload.unwrap_or(defaults.offload_kqv),
        vram_budget_bytes,
        ..defaults
    }
}

/// The worker-side spelling of a configured KV quantization.
fn kv_quant_of(config: KvQuantConfig) -> KvQuant {
    match config {
        KvQuantConfig::F16 => KvQuant::F16,
        KvQuantConfig::Q8_0 => KvQuant::Q8_0,
    }
}

/// The per-slug LLM cache policy: the knobs the loader applies when it
/// builds a pipeline, plus the VRAM ceiling the KV cache is clamped to.
///
/// The policy is derived from the operator config (`[[models.entry]] cache`
/// per slug, `[models] vram_budget_bytes` as the global ceiling). An empty
/// policy = the backend defaults, so a deployment that configures nothing
/// behaves exactly as before.
#[derive(Debug, Clone, Default)]
pub struct CachePolicy {
    /// Per-slug knobs, keyed by the registry slug (`llm:…`).
    by_slug: BTreeMap<String, CacheConfig>,
    /// The VRAM ceiling for the KV cache (`[models] vram_budget_bytes`).
    vram_budget_bytes: Option<u64>,
}

impl CachePolicy {
    /// The policy for a config: the per-slug knobs keyed by the registry
    /// slug, plus the global VRAM ceiling.
    #[must_use]
    pub fn new(by_slug: BTreeMap<String, CacheConfig>, vram_budget_bytes: Option<u64>) -> Self {
        Self {
            by_slug,
            vram_budget_bytes,
        }
    }

    /// The knobs for a slug: the declared ones, or the empty default.
    fn for_slug(&self, slug: &str) -> CacheConfig {
        self.by_slug.get(slug).copied().unwrap_or_default()
    }

    /// The knobs for a model the operator addresses by id (an alias) or by
    /// slug: the catalog resolves the request to the registry slug the
    /// knobs are keyed by.
    ///
    /// Integration-test binaries that include this module without
    /// exercising it need the allow (same pattern as the stub loaders).
    #[allow(dead_code)]
    #[must_use]
    pub fn for_request(&self, catalog: &ModelCatalog, requested: &str) -> CacheConfig {
        catalog
            .data
            .iter()
            .find(|entry| entry.id == requested || entry.slug == requested)
            .and_then(|entry| self.by_slug.get(&entry.slug))
            .copied()
            .unwrap_or_default()
    }
}

/// Production loader: resolve a catalog slug to its on-disk path via the
/// current `/v1/models` snapshot (stays fresh across
/// `POST /v1/models/refresh`) and load it with `LlamaCppLlm` (plus the
/// sibling mmproj projector when one sits next to the weights — vision
/// chat). Non-LLM catalog entries (STT/TTS/SD weights) and unknown slugs
/// are [`SwapError::UnknownSlug`] — a 400 at the route.
#[must_use]
pub fn catalog_llm_loader(
    catalog: Arc<RwLock<Arc<ModelCatalog>>>,
    policy: CachePolicy,
) -> LlmLoader {
    Arc::new(move |slug: &str| {
        let snapshot = catalog
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        // Match the served `id` (operator alias) or the registry slug: the
        // chat routes resolve aliases to the owning slug before calling the
        // loader, and the raw slug keeps working either way.
        let entry = snapshot
            .data
            .iter()
            .find(|entry| entry.id == slug || entry.slug == slug);
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
        // The cache knobs for this slug (an entry that declares none keeps
        // the backend defaults). The VRAM ceiling is global - it is the
        // operator's declared ceiling for the whole box, not a per-slug one.
        let cache = policy.for_slug(&entry.slug);
        let mut opts = llm_options(cache, policy.vram_budget_bytes);
        // Vision chat (REQ-8): a sibling mmproj projector upgrades the
        // pipeline to multimodal (same discovery + same `VISION_COMPILED`
        // flag the catalog's `capabilities.vision` derives from); none is
        // text-only.
        opts.mmproj_path = honkbox_llm_worker::VISION_COMPILED
            .then(|| honkbox_registry::sibling_mmproj(&path))
            .flatten();
        match LlamaCppLlm::load(&path, &opts) {
            Ok(pipe) => Ok(Arc::new(pipe) as Arc<dyn LlmPipeline>),
            Err(err) => Err(SwapError::Load(format!(
                "loading {slug:?} from {} failed: {err}",
                path.display()
            ))),
        }
    })
}

/// POST /v1/embeddings loader seam (REQ-4): a resolved catalog entry →
/// embedding pipeline. Production: [`catalog_embed_loader`] (llama.cpp from
/// the entry's on-disk path, capability-gated); tests: [`stub_embed_loader`]
/// (deterministic stub, entry ignored). Called on the blocking pool.
pub type EmbedLoader =
    Arc<dyn Fn(&ModelCatalogEntry) -> Result<Arc<dyn EmbedPipeline>, String> + Send + Sync>;

/// Production embeddings loader: llama.cpp embedding model from the entry's
/// path. Re-gates `capabilities.embeddings` (defense-in-depth: the route's
/// resolution already checked — a chat model handed here is refused, never
/// loaded for garbage token output). Integration-test binaries that include
/// this module without exercising it need the allow (same pattern as the
/// stub loaders).
#[allow(dead_code)]
pub fn catalog_embed_loader() -> EmbedLoader {
    Arc::new(|entry: &ModelCatalogEntry| {
        if !entry.capabilities.embeddings {
            return Err(format!(
                "model {:?} is not embeddings-capable (capabilities.embeddings = false)",
                entry.id
            ));
        }
        LlamaCppEmbed::load(Path::new(&entry.path), &LlmOptions::default())
            .map(|pipe| Arc::new(pipe) as Arc<dyn EmbedPipeline>)
            .map_err(|err| err.to_string())
    })
}

/// Test loader: the deterministic [`StubEmbed`], entry ignored. Unused in
/// the server binary itself (integration tests only), hence the allow —
/// same pattern as the reserved `UnloadReason` slots.
#[allow(dead_code)]
pub fn stub_embed_loader() -> EmbedLoader {
    Arc::new(|_| Ok(Arc::new(honkbox_llm_worker::StubEmbed)))
}

/// Test helper: an `AppConfig.embed` cache with the stub loader and a long
/// idle window (tests never exercise eviction; unit tests do it with short
/// TTLs directly).
#[allow(dead_code)]
pub fn stub_embed_cache() -> EmbedCache {
    EmbedCache::new(stub_embed_loader(), Duration::from_hours(24))
}

/// Cached-pipeline map entry: the pipeline plus its load/last-use instant
/// (the idle-TTL window start).
type EmbedEntry = (Arc<dyn EmbedPipeline>, Instant);
/// Slug → pipeline map behind the capacity-1 cache.
type EmbedEntries = HashMap<String, EmbedEntry>;

/// POST /v1/embeddings pipeline cache (REQ-4): load-once per slug, idle
/// TTL reuse on hits, capacity 1 with evict-on-conflict.
///
/// Deliberately **not** the [`SwapManager`]: that type is
/// `LlmPipeline`-bound and chat-shaped (seeded swap decisions, pin lists,
/// `/v1/memory` residency). An embedding model never shares a warm set or
/// a lease with a chat model, so a small dedicated cache — keyed by slug,
/// at most one resident embedding pipeline, evicted after the idle window
/// — is the honest minimal shape. `ttl` reuses `[swap] max_ttl` (the
/// operator's one residency knob) rather than inventing a new setting.
pub struct EmbedCache {
    loader: EmbedLoader,
    entries: Mutex<EmbedEntries>,
    ttl: Duration,
}

impl EmbedCache {
    /// Cache with `loader` and the idle-TTL reuse window.
    pub fn new(loader: EmbedLoader, ttl: Duration) -> Self {
        Self {
            loader,
            entries: Mutex::new(HashMap::new()),
            ttl,
        }
    }

    /// Production cache: the catalog-backed llama.cpp loader, the `[swap]`
    /// `max_ttl` idle window (the operator's one residency knob).
    #[must_use]
    #[allow(dead_code)]
    pub fn catalog(max_ttl: Duration) -> Self {
        Self::new(catalog_embed_loader(), max_ttl)
    }

    /// The cached pipeline for `entry`'s slug, loading it on a miss (or an
    /// idle-expired hit) via the configured loader. Loading runs on the
    /// caller's thread — the route invokes this from `spawn_blocking`.
    ///
    /// # Errors
    ///
    /// The loader's error string (capability refusal, load failure); the
    /// cache stays consistent — nothing is inserted on failure.
    pub fn get(&self, entry: &ModelCatalogEntry) -> Result<Arc<dyn EmbedPipeline>, String> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((pipe, loaded_at)) = entries.get(entry.slug.as_str()) {
            if loaded_at.elapsed() < self.ttl {
                return Ok(Arc::clone(pipe));
            }
            // Idle-expired: evict (drop the pipeline = free the model) and
            // fall through to a fresh load.
            entries.remove(entry.slug.as_str());
        }
        // Capacity 1: any other resident embedding pipeline is the LRU
        // victim by construction.
        entries.clear();
        let load_start = Instant::now();
        let pipe = (self.loader)(entry)?;
        crate::metrics::note_model_load(ModelKind::Embedding, crate::metrics::LoadReason::Request);
        crate::metrics::note_model_load_duration(ModelKind::Embedding, load_start.elapsed());
        entries.insert(entry.slug.clone(), (Arc::clone(&pipe), Instant::now()));
        Ok(pipe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cache policy keys the knobs by slug; an undeclared slug keeps the
    /// backend defaults, and the loader's options follow the knobs plus the
    /// global VRAM ceiling.
    #[test]
    fn cache_policy_keys_the_knobs_by_slug() {
        let mut knobs = BTreeMap::new();
        knobs.insert(
            "llm:a.gguf".to_owned(),
            CacheConfig {
                n_ctx: Some(4096),
                kv_quant: Some(KvQuantConfig::Q8_0),
                offload: Some(false),
            },
        );
        let policy = CachePolicy::new(knobs, Some(8_000_000_000));
        let declared = policy.for_slug("llm:a.gguf");
        assert_eq!(declared.n_ctx, Some(4096));
        assert_eq!(declared.kv_quant, Some(KvQuantConfig::Q8_0));
        assert_eq!(declared.offload, Some(false));
        // An undeclared slug keeps the defaults.
        assert_eq!(policy.for_slug("llm:other.gguf"), CacheConfig::default());

        let opts = llm_options(declared, policy.vram_budget_bytes);
        assert_eq!(opts.n_ctx, 4096);
        assert_eq!(opts.kv_quant, KvQuant::Q8_0);
        assert!(!opts.offload_kqv);
        assert_eq!(opts.vram_budget_bytes, Some(8_000_000_000));

        // No knobs declared -> the backend defaults, with the ceiling.
        let opts = llm_options(CacheConfig::default(), Some(42));
        assert_eq!(opts.n_ctx, LlmOptions::default().n_ctx);
        assert_eq!(opts.kv_quant, KvQuant::F16);
        assert!(opts.offload_kqv);
        assert_eq!(opts.vram_budget_bytes, Some(42));
    }

    /// The catalog resolves a request (an alias or the slug) to the knobs
    /// keyed by the registry slug.
    #[test]
    fn cache_policy_resolves_a_request_to_its_slug() {
        let mut knobs = BTreeMap::new();
        knobs.insert(
            "llm:a.gguf".to_owned(),
            CacheConfig {
                n_ctx: Some(4096),
                ..CacheConfig::default()
            },
        );
        let policy = CachePolicy::new(knobs, None);
        let catalog = ModelCatalog {
            object: "list".to_owned(),
            data: vec![ModelCatalogEntry {
                id: "alpha".to_owned(),
                slug: "llm:a.gguf".to_owned(),
                kind: ModelKind::Llm,
                owned_by: "local".to_owned(),
                context_length: None,
                max_output: None,
                thinking: false,
                tool_calling: false,
                modalities: vec!["text".to_owned()],
                capabilities: honkbox_api_types::ModelCapabilities::for_kind(ModelKind::Llm),
                category: "root".to_owned(),
                path: "/models/a.gguf".to_owned(),
                bytes: 1,
                metadata: None,
            }],
        };
        assert_eq!(policy.for_request(&catalog, "alpha").n_ctx, Some(4096));
        assert_eq!(policy.for_request(&catalog, "llm:a.gguf").n_ctx, Some(4096));
        // An unrequested model gets the defaults.
        assert_eq!(
            policy.for_request(&catalog, "other"),
            CacheConfig::default()
        );
    }

    /// Pin the settings plumbing: clone + equality on the tunables the
    /// config layer feeds in (real behavior lives in the integration tests).
    #[test]
    fn settings_clone_roundtrip() {
        let settings = SwapSettings {
            llm_ttl: Duration::from_secs(600),
            llm_max_loaded: 1,
            max_ttl: Duration::from_hours(24),
            pinned: HashSet::from(["llm:pin.gguf".to_owned()]),
        };
        let cloned = settings.clone();
        assert_eq!(cloned.llm_ttl, settings.llm_ttl);
        assert_eq!(cloned.llm_max_loaded, settings.llm_max_loaded);
        assert_eq!(cloned.max_ttl, settings.max_ttl);
        assert_eq!(cloned.pinned, settings.pinned);
    }
}
