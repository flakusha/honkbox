//! Minimal Prometheus text exposition (`GET /metrics`, feature-survey NEW-9).
//!
//! Hand-rolled on purpose: `AtomicU64` statics in the hot path (no mutex, no
//! per-request allocation), pre-declared histogram buckets, and a
//! vLLM-vocabulary subset — `honkbox_requests_total{route,code}`,
//! `honkbox_request_duration_seconds`,
//! `honkbox_gpu_slot_busy_rejections_total`, `honkbox_build_info{version}`.
//!
//! Swap/residency events are labeled by **kind, not slug**
//! (NEW-items §4): the label set is closed by `api-types::ModelKind` (7
//! kinds), so every kind-labeled grid is fixed at compile time and series
//! counts stay bounded no matter how large the catalog grows —
//! `honkbox_model_loads_total{kind,reason}` and
//! `honkbox_model_unloads_total{kind,reason}` are 7 × 3 = 21 series each,
//! `honkbox_swap_evictions_total{kind}` is 7, and
//! `honkbox_model_load_duration_seconds{kind}` re-uses the existing 11
//! buckets (7 × 13 lines with `count`/`sum`). The cache counters follow the
//! same rule: `honkbox_cache_hits_total{kind}` and
//! `honkbox_cache_misses_total{kind}` are 7 series each. Unpopulated kinds
//! render `0`;
//! the swap manager serves `llm` today, other kinds light up when per-kind
//! swap actors land. Request metrics stay slug-free by design (the safe
//! label is `kind`: 100 models × route × code would multiply `requests_total`
//! into the thousands).

use std::fmt::Write as _;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use honkbox_api_types::ModelKind;

/// Known route labels (bounded cardinality); parameterized control routes
/// collapse to their template, everything else counts under `"other"` so
/// arbitrary fallback paths cannot explode the series set.
const ROUTES: [&str; 42] = [
    "/health",
    "/api/version",
    "/v1/openapi.json",
    "/v1/models",
    "/v1/models/refresh",
    "/v1/models/{slug}/unload",
    "/v1/memory",
    "/v1/chat",
    "/v1/chat/stream",
    "/v1/chat/completions",
    "/v1/transcribe",
    "/v1/synthesize",
    "/running",
    "/api/models/unload",
    "/api/models/unload/{model}",
    "/metrics",
    "/v1/apps",
    "/v1/apps/{id}",
    "/v1/apps/{id}/start",
    "/v1/apps/{id}/stop",
    "/comfyui/status",
    "/comfyui/queue",
    "/comfyui/history/{id}",
    "/comfyui/view",
    "/v1/presets",
    "/v1/presets/{id}",
    "/v1/status/warnings",
    "/v1/jobs/blender",
    "/v1/capabilities",
    "/v1/catalog",
    "/v1/external-tools",
    "/v1/external-tools/{id}",
    "/v1/external-tools/reload",
    "/v1/downloads",
    "/v1/downloads/{id}",
    "/v1/artifacts",
    "/v1/artifacts/{id}/bytes",
    "/v1/queue",
    "/v1/jobs/{id}/bundle.zip",
    "/v1/jobs/{id}/content/{name}",
    "/v1/jobs/records",
    "/v1/jobs/records/{id}",
];

/// Status codes tracked verbatim; any other code counts under `"other"`.
const CODES: [&str; 7] = ["200", "400", "404", "415", "429", "500", "503"];

/// Histogram bucket edges in nanoseconds (Prometheus-client defaults).
const BUCKET_EDGES_NS: [u64; 11] = [
    5_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
    100_000_000,
    250_000_000,
    500_000_000,
    1_000_000_000,
    2_500_000_000,
    5_000_000_000,
    10_000_000_000,
];

/// Bucket `le` labels, matching `BUCKET_EDGES_NS` one for one.
const BUCKET_LES: [&str; 11] = [
    "0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1.0", "2.5", "5.0", "10.0",
];

/// Known routes + the `"other"` catch-all row.
const ROUTE_SLOTS: usize = ROUTES.len() + 1;
/// Known codes + the `"other"` catch-all column.
const CODE_SLOTS: usize = CODES.len() + 1;
/// Pre-declared histogram buckets.
const BUCKETS: usize = BUCKET_EDGES_NS.len();

/// Kind labels, matching `api-types::ModelKind` serde names one for one
/// (`KIND_LABELS[kind_slot(k)]`). The closed 9-kind enum is the cardinality
/// bound; the grids below never grow with the catalog.
const KIND_LABELS: [&str; 9] = [
    "llm",
    "sd",
    "controlnet",
    "lora",
    "embedding",
    "vae",
    "stt",
    "tts",
    "ocr",
];
/// Job-kind labels for `honkbox_jobs_total`, one per closed `JobKind`
/// (`crates/server/src/jobs_registry.rs`). The vocabulary is fixed at
/// compile time, so the series count never grows with the traffic and no
/// user data (ids, paths, prompts) can reach the label.
const JOB_KIND_LABELS: [&str; 9] = [
    "chat",
    "embeddings",
    "images",
    "transcribe",
    "synthesize",
    "translations",
    "image_ops",
    "blender",
    "ocr",
];

/// Load reasons (`request` = load-on-demand, `preload` = startup seed,
/// `manual` = reserved for a future explicit load endpoint).
const LOAD_REASONS: [&str; 3] = ["request", "preload", "manual"];
/// Unload reasons (`ttl` = idle expiry, `evict` = LRU capacity swap-out,
/// `explicit` = reserved for a future unload endpoint).
const UNLOAD_REASONS: [&str; 3] = ["ttl", "explicit", "evict"];

const KIND_SLOTS: usize = KIND_LABELS.len();
const LOAD_REASON_SLOTS: usize = LOAD_REASONS.len();
const UNLOAD_REASON_SLOTS: usize = UNLOAD_REASONS.len();

/// GC removal reasons (`reason` label of the two GC counters below).
const GC_REASONS: [&str; 4] = ["expired", "size_cap", "part_sweep", "temp_sweep"];
const GC_REASON_SLOTS: usize = GC_REASONS.len();

/// Why a job artifact was reclaimed by startup GC.
#[derive(Debug, Clone, Copy)]
pub(crate) enum GcReason {
    /// Job dir past its retention TTL.
    Expired,
    /// Oldest-first job dir evicted to satisfy the byte cap.
    SizeCap,
    /// Crash-leftover `*.part` file.
    PartSweep,
    /// Stale `$TMPDIR/honkbox-*.png` scratch file.
    TempSweep,
}

fn gc_reason_slot(reason: GcReason) -> usize {
    match reason {
        GcReason::Expired => 0,
        GcReason::SizeCap => 1,
        GcReason::PartSweep => 2,
        GcReason::TempSweep => 3,
    }
}

/// Why a model was loaded (emitted at the swap manager's load-commit).
#[derive(Debug, Clone, Copy)]
pub(crate) enum LoadReason {
    /// A request asked for an uncached slug.
    Request,
    /// Startup pre-warm seed (llama-swap `preload` analog).
    Preload,
    /// Reserved: an explicit operator/API-triggered load.
    #[allow(dead_code)] // reserved slot in the bounded exposition set
    Manual,
}

/// Why a model was unloaded (emitted at every swap-manager removal site).
#[derive(Debug, Clone, Copy)]
pub(crate) enum UnloadReason {
    /// Idle TTL expired (reaper sweep or pre-load trim).
    Ttl,
    /// An explicit operator/API-triggered unload (`POST /v1/models/{slug}/unload`).
    Explicit,
    /// LRU capacity trim made room for an incoming load.
    Evict,
}

fn kind_slot(kind: ModelKind) -> usize {
    match kind {
        ModelKind::Llm => 0,
        ModelKind::Sd => 1,
        ModelKind::ControlNet => 2,
        ModelKind::Lora => 3,
        ModelKind::Embedding => 4,
        ModelKind::Vae => 5,
        ModelKind::Stt => 6,
        ModelKind::Tts => 7,
        ModelKind::Ocr => 8,
    }
}

fn load_reason_slot(reason: LoadReason) -> usize {
    match reason {
        LoadReason::Request => 0,
        LoadReason::Preload => 1,
        LoadReason::Manual => 2,
    }
}

fn unload_reason_slot(reason: UnloadReason) -> usize {
    match reason {
        UnloadReason::Ttl => 0,
        UnloadReason::Explicit => 1,
        UnloadReason::Evict => 2,
    }
}

/// One route row of `requests_total` cells.
type CodeCells = [AtomicU64; CODE_SLOTS];

/// `requests_total` cells: `[route][code]`, zero-initialized at first use.
static REQUESTS: LazyLock<[CodeCells; ROUTE_SLOTS]> =
    LazyLock::new(|| [const { [const { AtomicU64::new(0) }; CODE_SLOTS] }; ROUTE_SLOTS]);

/// `jobs_total` cells: one per job kind.
static JOBS: LazyLock<[AtomicU64; JOB_KIND_LABELS.len()]> =
    LazyLock::new(|| [const { AtomicU64::new(0) }; JOB_KIND_LABELS.len()]);

/// Hand-rolled cumulative histogram over `BUCKET_EDGES_NS`.
struct Histogram {
    /// Observations total (equals the `+Inf` bucket).
    count: AtomicU64,
    /// Observed nanoseconds total.
    sum_ns: AtomicU64,
    /// Per-bucket cumulative counts (`le` edges, `+Inf` = `count`).
    buckets: [AtomicU64; BUCKETS],
}

impl Histogram {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum_ns: AtomicU64::new(0),
            buckets: [const { AtomicU64::new(0) }; BUCKETS],
        }
    }

    /// One observation: count + sum + every bucket at or above the value.
    fn observe(&self, nanos: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_ns.fetch_add(nanos, Ordering::Relaxed);
        for (slot, edge) in BUCKET_EDGES_NS.iter().enumerate() {
            if nanos <= *edge {
                self.buckets[slot].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Global request-duration histogram (no labels; route breakdown lives on
/// `requests_total` only, to keep the series set small).
static REQUEST_DURATION: LazyLock<Histogram> = LazyLock::new(|| Histogram {
    count: AtomicU64::new(0),
    sum_ns: AtomicU64::new(0),
    buckets: [const { AtomicU64::new(0) }; BUCKETS],
});

/// Why a 429 GPU-slot rejection happened (`reason` label of
/// `honkbox_gpu_slot_busy_rejections_total`). `no_wait` = the admission
/// queue is disabled (or the fast path lost); `queue_full` / `queue_timeout`
/// are the REQ-6 Option A admission-queue rejections (see `admission`).
#[derive(Debug, Clone, Copy)]
pub(crate) enum BusyReason {
    /// Rejected immediately - no queue configured (queue off; REQ-6
    /// Option B behavior).
    NoWait,
    /// The admission queue already had its one waiter.
    QueueFull,
    /// The admission wait exceeded its `[server] queue_wait_secs` bound.
    QueueTimeout,
}

/// Busy rejection reason labels (bounded exposition set).
const BUSY_REASONS: [&str; 3] = ["no_wait", "queue_full", "queue_timeout"];

const BUSY_REASON_SLOTS: usize = BUSY_REASONS.len();

/// 429 GPU-slot rejections (residency `Busy`), per reason.
static GPU_SLOT_BUSY_REJECTIONS: LazyLock<[AtomicU64; BUSY_REASON_SLOTS]> =
    LazyLock::new(|| [const { AtomicU64::new(0) }; BUSY_REASON_SLOTS]);

/// `loads_total` cells: `[kind][load reason]`, zero-initialized at first use.
static MODEL_LOADS: LazyLock<[[AtomicU64; LOAD_REASON_SLOTS]; KIND_SLOTS]> =
    LazyLock::new(|| [const { [const { AtomicU64::new(0) }; LOAD_REASON_SLOTS] }; KIND_SLOTS]);

/// `unloads_total` cells: `[kind][unload reason]`.
static MODEL_UNLOADS: LazyLock<[[AtomicU64; UNLOAD_REASON_SLOTS]; KIND_SLOTS]> =
    LazyLock::new(|| [const { [const { AtomicU64::new(0) }; UNLOAD_REASON_SLOTS] }; KIND_SLOTS]);

/// `swap_evictions_total` cells: `[kind]` (capacity-driven swap-outs).
static SWAP_EVICTIONS: LazyLock<[AtomicU64; KIND_SLOTS]> =
    LazyLock::new(|| [const { AtomicU64::new(0) }; KIND_SLOTS]);

/// `cache_hits_total` cells: `[kind]` (a request served from a resident
/// pipeline - nothing was loaded).
static CACHE_HITS: LazyLock<[AtomicU64; KIND_SLOTS]> =
    LazyLock::new(|| [const { AtomicU64::new(0) }; KIND_SLOTS]);

/// `cache_misses_total` cells: `[kind]` (no pipeline was resident, so one
/// was loaded).
static CACHE_MISSES: LazyLock<[AtomicU64; KIND_SLOTS]> =
    LazyLock::new(|| [const { AtomicU64::new(0) }; KIND_SLOTS]);

/// Reclaimed bytes by GC reason.
static GC_RECLAIMED_BYTES: LazyLock<[AtomicU64; GC_REASON_SLOTS]> =
    LazyLock::new(|| [const { AtomicU64::new(0) }; GC_REASON_SLOTS]);
/// Removed entries by GC reason.
static GC_ENTRIES: LazyLock<[AtomicU64; GC_REASON_SLOTS]> =
    LazyLock::new(|| [const { AtomicU64::new(0) }; GC_REASON_SLOTS]);

/// Per-kind pipeline-construction duration histograms (the same 11 buckets
/// as the global request duration).
static LOAD_DURATION: LazyLock<[Histogram; KIND_SLOTS]> =
    LazyLock::new(|| [const { Histogram::new() }; KIND_SLOTS]);

/// Request route label for a raw URI path: the path itself for known
/// routes, the template for parameterized routes, and `"other"` for
/// anything else (fallback hits, escapes, probes). Dynamic ids and file
/// names never reach a label.
fn has_single_id(path: &str, prefix: &str) -> bool {
    path.strip_prefix(prefix)
        .is_some_and(|rest| !rest.is_empty() && !rest.contains('/'))
}

pub(crate) fn route_label(path: &str) -> &'static str {
    ROUTES
        .iter()
        .find(|route| **route == path)
        .copied()
        .or_else(|| {
            if path.starts_with("/v1/models/") && path.ends_with("/unload") {
                Some("/v1/models/{slug}/unload")
            } else if path.starts_with("/api/models/unload/") {
                Some("/api/models/unload/{model}")
            } else if path.starts_with("/v1/apps/") && path.ends_with("/start") {
                Some("/v1/apps/{id}/start")
            } else if path.starts_with("/v1/apps/") && path.ends_with("/stop") {
                Some("/v1/apps/{id}/stop")
            } else if has_single_id(path, "/v1/apps/") {
                Some("/v1/apps/{id}")
            } else if path.starts_with("/comfyui/history/") {
                Some("/comfyui/history/{id}")
            } else if has_single_id(path, "/v1/presets/") {
                Some("/v1/presets/{id}")
            } else if has_single_id(path, "/v1/external-tools/") {
                Some("/v1/external-tools/{id}")
            } else if has_single_id(path, "/v1/downloads/") {
                Some("/v1/downloads/{id}")
            } else if has_single_id(path, "/v1/jobs/records/") {
                Some("/v1/jobs/records/{id}")
            } else if path
                .strip_prefix("/v1/artifacts/")
                .and_then(|rest| rest.strip_suffix("/bytes"))
                .is_some_and(|id| !id.is_empty() && !id.contains('/'))
            {
                Some("/v1/artifacts/{id}/bytes")
            } else if path
                .strip_prefix("/v1/jobs/")
                .and_then(|rest| rest.strip_suffix("/bundle.zip"))
                .is_some_and(|id| !id.is_empty() && !id.contains('/'))
            {
                Some("/v1/jobs/{id}/bundle.zip")
            } else if path
                .strip_prefix("/v1/jobs/")
                .and_then(|rest| rest.strip_suffix("/content/"))
                .is_some_and(|rest| {
                    let mut parts = rest.split('/');
                    parts.next().is_some_and(|id| !id.is_empty())
                        && parts.next().is_some_and(|name| !name.is_empty())
                        && parts.next().is_none()
                })
            {
                Some("/v1/jobs/{id}/content/{name}")
            } else {
                None
            }
        })
        .unwrap_or("other")
}

fn route_slot(route: &str) -> usize {
    ROUTES
        .iter()
        .position(|known| *known == route)
        .unwrap_or(ROUTES.len())
}

fn code_slot(code: u16) -> usize {
    CODES
        .iter()
        .position(|label| label.parse::<u16>().is_ok_and(|known| known == code))
        .unwrap_or(CODES.len())
}

/// Count one request: route/code counter + one duration observation. A
/// handful of relaxed atomic adds; no allocation, no lock.
pub(crate) fn record_request(route: &'static str, code: u16, elapsed: Duration) {
    REQUESTS[route_slot(route)][code_slot(code)].fetch_add(1, Ordering::Relaxed);
    let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    REQUEST_DURATION.observe(nanos);
}

/// Count one 429 GPU-slot-busy rejection (call at the rejection site).
pub(crate) fn note_gpu_slot_busy(reason: BusyReason) {
    GPU_SLOT_BUSY_REJECTIONS[busy_reason_slot(reason)].fetch_add(1, Ordering::Relaxed);
}

fn busy_reason_slot(reason: BusyReason) -> usize {
    match reason {
        BusyReason::NoWait => 0,
        BusyReason::QueueFull => 1,
        BusyReason::QueueTimeout => 2,
    }
}

/// Count one model load-commit (call at the swap manager's load commit).
pub(crate) fn note_model_load(kind: ModelKind, reason: LoadReason) {
    MODEL_LOADS[kind_slot(kind)][load_reason_slot(reason)].fetch_add(1, Ordering::Relaxed);
}

/// Count one model unload (call at every swap-manager removal site).
pub(crate) fn note_model_unload(kind: ModelKind, reason: UnloadReason) {
    MODEL_UNLOADS[kind_slot(kind)][unload_reason_slot(reason)].fetch_add(1, Ordering::Relaxed);
}

/// Count one capacity-driven eviction (call at the LRU swap-out site).
pub(crate) fn note_swap_eviction(kind: ModelKind) {
    SWAP_EVICTIONS[kind_slot(kind)].fetch_add(1, Ordering::Relaxed);
}

/// Count one admitted job of the given kind. The label is the closed
/// `JobKind` vocabulary; an unknown label is ignored, so the series set
/// stays bounded whatever the caller passes.
pub(crate) fn note_job_started(label: &str) {
    if let Some(slot) = JOB_KIND_LABELS.iter().position(|known| *known == label) {
        JOBS[slot].fetch_add(1, Ordering::Relaxed);
    }
}

/// Count one cache hit (call when a request is served from a resident
/// pipeline).
pub(crate) fn note_cache_hit(kind: ModelKind) {
    CACHE_HITS[kind_slot(kind)].fetch_add(1, Ordering::Relaxed);
}

/// Count one cache miss (call when a pipeline had to be loaded).
pub(crate) fn note_cache_miss(kind: ModelKind) {
    CACHE_MISSES[kind_slot(kind)].fetch_add(1, Ordering::Relaxed);
}

/// Count one successful GC removal and the bytes it reclaimed.
pub(crate) fn note_gc_reclaim(reason: GcReason, bytes: u64) {
    let slot = gc_reason_slot(reason);
    GC_ENTRIES[slot].fetch_add(1, Ordering::Relaxed);
    GC_RECLAIMED_BYTES[slot].fetch_add(bytes, Ordering::Relaxed);
}

/// Observe one pipeline-construction duration (call around the swap
/// manager's blocking load).
pub(crate) fn note_model_load_duration(kind: ModelKind, elapsed: Duration) {
    let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    LOAD_DURATION[kind_slot(kind)].observe(nanos);
}

/// Render the cache counters: the hits and the misses, each labeled by
/// kind (a bounded grid: one series per kind).
fn push_cache_families(out: &mut String) {
    push_family(
        out,
        "honkbox_cache_hits_total",
        "Requests served from a resident pipeline, by kind.",
        "counter",
    );
    for (kind, cell) in KIND_LABELS.iter().zip(CACHE_HITS.iter()) {
        let _ = writeln!(
            out,
            "honkbox_cache_hits_total{{kind=\"{kind}\"}} {}",
            cell.load(Ordering::Relaxed)
        );
    }

    push_family(
        out,
        "honkbox_cache_misses_total",
        "Pipelines loaded because none was resident, by kind.",
        "counter",
    );
    for (kind, cell) in KIND_LABELS.iter().zip(CACHE_MISSES.iter()) {
        let _ = writeln!(
            out,
            "honkbox_cache_misses_total{{kind=\"{kind}\"}} {}",
            cell.load(Ordering::Relaxed)
        );
    }
}

/// Render the swap/residency families: the kind-labeled grids (bounded by
/// `KIND_SLOTS`, see the module docs) plus the per-kind load-duration
/// histogram. Split out of [`exposition`] to keep both functions readable.
fn push_swap_families(out: &mut String) {
    push_family(
        out,
        "honkbox_model_loads_total",
        "Swap-manager model load commits by kind and reason.",
        "counter",
    );
    let kinds = KIND_LABELS.iter().copied();
    let reasons = LOAD_REASONS.iter().copied();
    for (kind, row) in kinds.zip(MODEL_LOADS.iter()) {
        for (reason, cell) in reasons.clone().zip(row.iter()) {
            let _ = writeln!(
                out,
                "honkbox_model_loads_total{{kind=\"{kind}\",reason=\"{reason}\"}} {}",
                cell.load(Ordering::Relaxed)
            );
        }
    }

    push_family(
        out,
        "honkbox_model_unloads_total",
        "Swap-manager model removals by kind and reason.",
        "counter",
    );
    let kinds = KIND_LABELS.iter().copied();
    let reasons = UNLOAD_REASONS.iter().copied();
    for (kind, row) in kinds.zip(MODEL_UNLOADS.iter()) {
        for (reason, cell) in reasons.clone().zip(row.iter()) {
            let _ = writeln!(
                out,
                "honkbox_model_unloads_total{{kind=\"{kind}\",reason=\"{reason}\"}} {}",
                cell.load(Ordering::Relaxed)
            );
        }
    }

    push_family(
        out,
        "honkbox_swap_evictions_total",
        "Pipelines evicted for LRU capacity (swap-outs) by kind.",
        "counter",
    );
    for (kind, cell) in KIND_LABELS.iter().zip(SWAP_EVICTIONS.iter()) {
        let _ = writeln!(
            out,
            "honkbox_swap_evictions_total{{kind=\"{kind}\"}} {}",
            cell.load(Ordering::Relaxed)
        );
    }

    push_cache_families(out);

    push_family(
        out,
        "honkbox_model_load_duration_seconds",
        "Swap-manager pipeline construction duration in seconds, by kind.",
        "histogram",
    );
    for (kind, hist) in KIND_LABELS.iter().zip(LOAD_DURATION.iter()) {
        let count = hist.count.load(Ordering::Relaxed);
        for (le, bucket) in BUCKET_LES.iter().zip(hist.buckets.iter()) {
            let _ = writeln!(
                out,
                "honkbox_model_load_duration_seconds_bucket{{kind=\"{kind}\",le=\"{le}\"}} {}",
                bucket.load(Ordering::Relaxed)
            );
        }
        let _ = writeln!(
            out,
            "honkbox_model_load_duration_seconds_bucket{{kind=\"{kind}\",le=\"+Inf\"}} {count}"
        );
        let sum_ns = hist.sum_ns.load(Ordering::Relaxed);
        // Integer math only (the repo avoids float casts): 9 fractional digits.
        let _ = writeln!(
            out,
            "honkbox_model_load_duration_seconds_sum{{kind=\"{kind}\"}} {}.{:09}",
            sum_ns / 1_000_000_000,
            sum_ns % 1_000_000_000
        );
        let _ = writeln!(
            out,
            "honkbox_model_load_duration_seconds_count{{kind=\"{kind}\"}} {count}"
        );
    }
}

/// Render the GC counters: one fixed reason label per bounded slot.
fn push_gc_families(out: &mut String) {
    push_family(
        out,
        "honkbox_gc_reclaimed_bytes_total",
        "Bytes reclaimed by startup garbage collection, by reason.",
        "counter",
    );
    for (reason, cell) in GC_REASONS.iter().zip(GC_RECLAIMED_BYTES.iter()) {
        let _ = writeln!(
            out,
            "honkbox_gc_reclaimed_bytes_total{{reason=\"{reason}\"}} {}",
            cell.load(Ordering::Relaxed)
        );
    }

    push_family(
        out,
        "honkbox_gc_entries_total",
        "Entries removed by startup garbage collection, by reason.",
        "counter",
    );
    for (reason, cell) in GC_REASONS.iter().zip(GC_ENTRIES.iter()) {
        let _ = writeln!(
            out,
            "honkbox_gc_entries_total{{reason=\"{reason}\"}} {}",
            cell.load(Ordering::Relaxed)
        );
    }
}

/// Render the Prometheus text exposition (version 0.0.4) for every series.
pub(crate) fn exposition() -> String {
    let mut out = String::new();

    push_family(&mut out, "honkbox_build_info", "Build metadata.", "gauge");
    let _ = writeln!(
        out,
        "honkbox_build_info{{version=\"{}\"}} 1",
        env!("CARGO_PKG_VERSION")
    );

    push_family(
        &mut out,
        "honkbox_requests_total",
        "HTTP requests by route and status code.",
        "counter",
    );
    let routes = ROUTES.iter().copied().chain(["other"]);
    let codes = CODES.iter().copied().chain(["other"]);
    for (route, row) in routes.zip(REQUESTS.iter()) {
        for (code, cell) in codes.clone().zip(row.iter()) {
            let _ = writeln!(
                out,
                "honkbox_requests_total{{route=\"{route}\",code=\"{code}\"}} {}",
                cell.load(Ordering::Relaxed)
            );
        }
    }

    push_family(
        &mut out,
        "honkbox_gpu_slot_busy_rejections_total",
        "Requests rejected with 429 because the GPU slot was busy, by reason (queue reasons reserved).",
        "counter",
    );
    for (reason, cell) in BUSY_REASONS
        .iter()
        .copied()
        .zip(GPU_SLOT_BUSY_REJECTIONS.iter())
    {
        let _ = writeln!(
            out,
            "honkbox_gpu_slot_busy_rejections_total{{reason=\"{reason}\"}} {}",
            cell.load(Ordering::Relaxed)
        );
    }

    push_family(
        &mut out,
        "honkbox_request_duration_seconds",
        "Request duration in seconds.",
        "histogram",
    );
    let hist = &*REQUEST_DURATION;
    let count = hist.count.load(Ordering::Relaxed);
    for (le, bucket) in BUCKET_LES.iter().zip(hist.buckets.iter()) {
        let _ = writeln!(
            out,
            "honkbox_request_duration_seconds_bucket{{le=\"{le}\"}} {}",
            bucket.load(Ordering::Relaxed)
        );
    }
    let _ = writeln!(
        out,
        "honkbox_request_duration_seconds_bucket{{le=\"+Inf\"}} {count}"
    );
    let sum_ns = hist.sum_ns.load(Ordering::Relaxed);
    // Integer math only (the repo avoids float casts): 9 fractional digits.
    let _ = writeln!(
        out,
        "honkbox_request_duration_seconds_sum {}.{:09}",
        sum_ns / 1_000_000_000,
        sum_ns % 1_000_000_000
    );
    let _ = writeln!(out, "honkbox_request_duration_seconds_count {count}");

    push_family(
        &mut out,
        "honkbox_jobs_total",
        "Admitted jobs by kind (closed vocabulary; no user data).",
        "counter",
    );
    for (kind, cell) in JOB_KIND_LABELS.iter().zip(JOBS.iter()) {
        let _ = writeln!(
            out,
            "honkbox_jobs_total{{kind=\"{kind}\"}} {}",
            cell.load(Ordering::Relaxed)
        );
    }

    push_gc_families(&mut out);
    push_swap_families(&mut out);
    out
}

/// HELP + TYPE header lines for one metric family.
fn push_family(out: &mut String, name: &str, help: &str, kind: &str) {
    let _ = write!(out, "# HELP {name} {help}\n# TYPE {name} {kind}\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_map_known_and_unknown() {
        assert_eq!(route_label("/v1/chat"), "/v1/chat");
        assert_eq!(route_label("/nope/xyz"), "other");
        assert_eq!(route_slot("other"), ROUTES.len());
        assert_eq!(code_slot(429), 4);
        assert_eq!(code_slot(418), CODES.len());
    }

    /// Numeric tail of the first exposition line with the given prefix.
    fn series_value(body: &str, prefix: &str) -> u64 {
        body.lines()
            .find(|line| line.starts_with(prefix))
            .unwrap_or_else(|| panic!("missing series line: {prefix}"))
            .rsplit(' ')
            .next()
            .expect("series line has a value")
            .parse()
            .expect("series value parses as u64")
    }

    #[test]
    fn exposition_counts_requests_and_buckets() {
        // Parallel unit tests share the process-wide statics: assert >=, not ==.
        record_request("/v1/models", 200, Duration::from_millis(12));
        record_request("/v1/chat/stream", 200, Duration::from_nanos(5_000_000));
        let body = exposition();

        assert!(body.contains("# TYPE honkbox_requests_total counter"));
        let models_line = "honkbox_requests_total{route=\"/v1/models\",code=\"200\"}";
        assert!(series_value(&body, models_line) >= 1);
        let stream_line = "honkbox_requests_total{route=\"/v1/chat/stream\",code=\"200\"}";
        assert!(series_value(&body, stream_line) >= 1);
        // 12ms lands in le=0.025 (and above); 5ms exactly on the le=0.005 edge.
        let le25 = "honkbox_request_duration_seconds_bucket{le=\"0.025\"}";
        assert!(series_value(&body, le25) >= 1);
        let le005 = "honkbox_request_duration_seconds_bucket{le=\"0.005\"}";
        assert!(series_value(&body, le005) >= 1);
        assert!(body.contains("# TYPE honkbox_request_duration_seconds histogram"));
        assert!(body.contains("honkbox_request_duration_seconds_bucket{le=\"+Inf\"}"));
        // Family presence only: the cumulative sum crosses 1s when other
        // tests in this binary recorded routed requests first (serial -j1),
        // so a value-shape assert here would be ordering-dependent.
        assert!(body.contains("honkbox_request_duration_seconds_sum "));
        assert!(series_value(&body, "honkbox_request_duration_seconds_count") >= 2);

        assert!(body.contains("# TYPE honkbox_gpu_slot_busy_rejections_total counter"));
        note_gpu_slot_busy(BusyReason::NoWait);
        assert!(
            series_value(
                &exposition(),
                "honkbox_gpu_slot_busy_rejections_total{reason=\"no_wait\"}"
            ) >= 1
        );
    }

    /// The closed job-kind vocabulary the exposition must render (kept
    /// local: `metrics` must not depend on `jobs_registry`).
    const EXPECTED_JOB_KINDS: [&str; 9] = [
        "chat",
        "embeddings",
        "images",
        "transcribe",
        "synthesize",
        "translations",
        "image_ops",
        "blender",
        "ocr",
    ];

    #[test]
    fn jobs_total_renders_bounded_grid() {
        note_job_started("blender");
        note_job_started("not_a_job_kind");
        let body = exposition();

        assert!(body.contains("# TYPE honkbox_jobs_total counter"));
        // Counters are process-global: assert >=, never ==.
        assert!(series_value(&body, "honkbox_jobs_total{kind=\"blender\"}") >= 1);
        // The full bounded grid renders (unpopulated kinds expose their `0`).
        for kind in EXPECTED_JOB_KINDS {
            let prefix = format!("honkbox_jobs_total{{kind=\"{kind}\"}}");
            assert!(body.contains(&prefix), "missing job series: {prefix}");
        }
        // An unknown label adds no series.
        assert!(!body.contains("not_a_job_kind"));
        let series = body
            .lines()
            .filter(|line| line.starts_with("honkbox_jobs_total{"))
            .count();
        assert_eq!(series, EXPECTED_JOB_KINDS.len());
    }

    #[test]
    fn build_info_is_present() {
        let body = exposition();
        assert!(body.starts_with("# HELP honkbox_build_info"));
        assert!(body.contains("# TYPE honkbox_build_info gauge"));
        assert!(body.contains("honkbox_build_info{version=\""));
        assert!(body.ends_with('\n'));
    }

    #[test]
    fn kind_slots_match_model_kind_order() {
        assert_eq!(kind_slot(ModelKind::Llm), 0);
        assert_eq!(kind_slot(ModelKind::Tts), 7);
        assert_eq!(kind_slot(ModelKind::Ocr), KIND_SLOTS - 1);
        assert_eq!(load_reason_slot(LoadReason::Manual), LOAD_REASONS.len() - 1);
        assert_eq!(
            unload_reason_slot(UnloadReason::Evict),
            UNLOAD_REASONS.len() - 1
        );
    }

    /// The kind-labeled families render a full bounded grid (7 kinds x
    /// reasons), so unpopulated kinds still expose their `0` series.
    #[test]
    fn kind_labeled_families_render_bounded_grids() {
        note_model_load(ModelKind::Llm, LoadReason::Request);
        note_model_load(ModelKind::Tts, LoadReason::Preload);
        note_model_unload(ModelKind::Llm, UnloadReason::Ttl);
        note_model_unload(ModelKind::Sd, UnloadReason::Evict);
        note_swap_eviction(ModelKind::Llm);
        note_cache_hit(ModelKind::Llm);
        note_cache_miss(ModelKind::Llm);
        note_model_load_duration(ModelKind::Llm, Duration::from_millis(25));
        let body = exposition();

        assert!(body.contains("# TYPE honkbox_model_loads_total counter"));
        assert!(
            series_value(
                &body,
                "honkbox_model_loads_total{kind=\"llm\",reason=\"request\"}"
            ) >= 1
        );
        assert!(
            series_value(
                &body,
                "honkbox_model_loads_total{kind=\"tts\",reason=\"preload\"}"
            ) >= 1
        );
        // Unpopulated slots still render (bounded pre-declared grid).
        assert!(body.contains("honkbox_model_loads_total{kind=\"vae\",reason=\"manual\"} 0"));

        assert!(body.contains("# TYPE honkbox_model_unloads_total counter"));
        assert!(
            series_value(
                &body,
                "honkbox_model_unloads_total{kind=\"llm\",reason=\"ttl\"}"
            ) >= 1
        );
        assert!(
            series_value(
                &body,
                "honkbox_model_unloads_total{kind=\"sd\",reason=\"evict\"}"
            ) >= 1
        );

        assert!(body.contains("# TYPE honkbox_swap_evictions_total counter"));
        assert!(series_value(&body, "honkbox_swap_evictions_total{kind=\"llm\"}") >= 1);
        assert!(body.contains("honkbox_swap_evictions_total{kind=\"stt\"} 0"));

        assert!(body.contains("# TYPE honkbox_cache_hits_total counter"));
        assert!(series_value(&body, "honkbox_cache_hits_total{kind=\"llm\"}") >= 1);
        assert!(body.contains("honkbox_cache_hits_total{kind=\"stt\"} 0"));
        assert!(body.contains("# TYPE honkbox_cache_misses_total counter"));
        assert!(series_value(&body, "honkbox_cache_misses_total{kind=\"llm\"}") >= 1);

        assert!(body.contains("# TYPE honkbox_model_load_duration_seconds histogram"));
        // 25ms lands in le=0.025 (and above), llm row only.
        let le25 = "honkbox_model_load_duration_seconds_bucket{kind=\"llm\",le=\"0.025\"}";
        assert!(series_value(&body, le25) >= 1);
        assert!(
            body.contains("honkbox_model_load_duration_seconds_bucket{kind=\"sd\",le=\"+Inf\"} 0")
        );
        assert!(body.contains("honkbox_model_load_duration_seconds_count{kind=\"llm\"}"));
        assert!(body.contains("honkbox_model_load_duration_seconds_sum{kind=\"llm\"} 0."));
    }
}
