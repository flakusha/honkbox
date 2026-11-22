//! Capability/VRAM-aware resource ledger (TASK-resource-ledger-bounded-queue);
//! pure in-memory math, no backends, no HTTP, no async runtime.
//!
//! The ledger models a fixed set of capacities (GPU slots, VRAM bytes, RAM
//! bytes, CPU threads; a `0` budget leaves that dimension unmanaged) and a
//! fixed [`ResourceKind`] vocabulary (`sd`, `llm`, `ocr`, `embed`, `rag`,
//! `comfy`, `blender`, `3d`, `training`). Each kind carries a per-job [`Demand`] (GPU
//! slots + byte/thread estimates), a bounded admission-queue cap, and a
//! queue TTL. Reservation happens BEFORE model load / external process
//! start; a bounded per-kind queue absorbs bursts, `cancel` removes a queued
//! waiter before it can ever acquire, and the TTL expires waiters that
//! waited past their bound (counted as evictions).
//!
//! The RAII [`Lease`] API is preserved: dropping a lease (or an explicit
//! [`Lease::release`]) frees exactly the demand it committed — across
//! cancel, timeout, and engine-crash drop paths alike. The legacy
//! single-slot probe ([`Ledger::try_reserve`], [`Ledger::is_busy`], [`Busy`])
//! keeps its pre-ledger semantics (kind `sd`, one shared GPU slot) so the
//! server's `/v1/memory` readout and the CLI one-shots behave identically.
//!
//! Metrics are per-instance counters over the fixed kind grid (bounded
//! labels: kinds and three fixed rejection reasons — never user data),
//! rendered by [`Ledger::exposition`] as a Prometheus text fragment. The
//! server binary shares the thin [`global`] wrapper; tests always build
//! their own [`Ledger::new`] instance — never the global — so suites stay
//! order-independent.
//!
//! Init scope: a single in-process GPU pool (`try_reserve`/drop, NO fake GPU
//! math). Tier detection and measured VRAM sizes arrive later; workers never
//! touch this crate.

use std::collections::VecDeque;
use std::fmt;
use std::fmt::Write as _;
use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// Process-wide ledger shared by every server route in this process. Tests
/// NEVER use this — build a [`Ledger::new`] instance per test instead (the
/// global carries cross-test state; per-test instances keep suites
/// order-independent).
static GLOBAL: LazyLock<Ledger> = LazyLock::new(Ledger::new);

/// The shared process-wide ledger (the server binary's singleton; a thin
/// wrapper over [`Ledger`] so tests and other binaries can stay isolated).
pub fn global() -> &'static Ledger {
    &GLOBAL
}

/// The closed admission-kind vocabulary. The label set is fixed at compile
/// time: every ledger metric grid is 9 kinds wide no matter how large the
/// model catalog grows, and no user-controlled string ever becomes a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// Stable-diffusion image pipelines (`/v1/images/*`, the CLI `generate`).
    Sd,
    /// Chat/LLM pipelines (`/v1/chat*`, `/v1/chat/completions`).
    Llm,
    /// OCR pipelines (reserved; no route serves it yet).
    Ocr,
    /// Embedding pipelines (`/v1/embeddings`).
    Embed,
    /// RAG indexing jobs (reserved; CPU-only by default).
    Rag,
    /// `ComfyUI` renders (external process; reserved).
    Comfy,
    /// Blender jobs (external process; reserved).
    Blender,
    /// Generic 3D tooling (external process; reserved).
    ThreeD,
    /// Long LoRA/fine-tune jobs (external Python sidecar). Exclusive: a
    /// training lease never starts beside other work, and other work never
    /// starts beside it.
    Training,
}

impl ResourceKind {
    /// Every kind, in the fixed label order used by [`Ledger::exposition`]
    /// and the per-kind settings arrays.
    pub const ALL: [ResourceKind; 9] = [
        ResourceKind::Sd,
        ResourceKind::Llm,
        ResourceKind::Ocr,
        ResourceKind::Embed,
        ResourceKind::Rag,
        ResourceKind::Comfy,
        ResourceKind::Blender,
        ResourceKind::ThreeD,
        ResourceKind::Training,
    ];

    /// The fixed metric/config label for this kind (never user data).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            ResourceKind::Sd => "sd",
            ResourceKind::Llm => "llm",
            ResourceKind::Ocr => "ocr",
            ResourceKind::Embed => "embed",
            ResourceKind::Rag => "rag",
            ResourceKind::Comfy => "comfy",
            ResourceKind::Blender => "blender",
            ResourceKind::ThreeD => "3d",
            ResourceKind::Training => "training",
        }
    }

    /// Slot in the fixed 9-wide arrays (`ALL` order).
    const fn slot(self) -> usize {
        self as usize
    }

    /// Parse a config label back into a kind (the `[ledger]` per-kind
    /// tables).
    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|kind| kind.label() == label)
    }
}

impl fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// What one job of a kind consumes while its [`Lease`] is held. All values
/// are admission-time ESTIMATES: VRAM/RAM numbers bound concurrent
/// commitments against the configured budgets, they never measure anything.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Demand {
    /// Shared GPU slots held for the job's whole run.
    pub gpu_slots: u32,
    /// Estimated VRAM the job pins (0 = no VRAM claim).
    pub vram_bytes: u64,
    /// Estimated RAM the job pins (0 = no RAM claim).
    pub ram_bytes: u64,
    /// CPU worker threads the job occupies.
    pub cpu_threads: u32,
}

impl Demand {
    /// The default demand per kind. Every kind that holds the shared GPU
    /// slot in the server today (`sd`, `llm`, `embed` routes, plus the
    /// reserved GPU-bound kinds) reserves one slot — preserving the
    /// pre-ledger single-slot contention exactly; `rag` is CPU-only by
    /// default. VRAM/RAM stay unclaimed (0) until an operator configures
    /// budgets and per-kind estimates (`[ledger] vram_demand`), so the
    /// default changes nothing for a single-tenant dev flow.
    #[must_use]
    pub fn for_kind(kind: ResourceKind) -> Self {
        let gpu_slots = match kind {
            ResourceKind::Rag => 0,
            _ => 1,
        };
        Self {
            gpu_slots,
            ..Self::default()
        }
    }
}

/// The box's resource ceilings. A `0` byte/thread budget leaves that
/// dimension unmanaged (no accounting, never a rejection); GPU slots are
/// always managed — they are the ledger's core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capacities {
    /// Shared GPU slots (default 1: the single-GPU studio shape).
    pub gpu_slots: u32,
    /// Total VRAM budget in bytes; 0 = unmanaged.
    pub vram_bytes: u64,
    /// Total RAM budget in bytes; 0 = unmanaged.
    pub ram_bytes: u64,
    /// CPU thread budget; 0 = unmanaged.
    pub cpu_threads: u64,
    /// Pressure margin for the OOM near-miss metric: an admission whose
    /// committed VRAM lands within this many bytes of the budget counts a
    /// near miss. 0 = the metric fires only at exact budget fill.
    pub oom_headroom_bytes: u64,
}

impl Default for Capacities {
    fn default() -> Self {
        Self {
            gpu_slots: 1,
            vram_bytes: 0,
            ram_bytes: 0,
            cpu_threads: 0,
            oom_headroom_bytes: 0,
        }
    }
}

impl Capacities {
    /// `true` when `demand` alone can never fit these capacities (a hard
    /// rejection — no queue wait can ever grant it; the server maps this to
    /// a 503 naming the binding dimension).
    fn oversized(&self, demand: Demand) -> bool {
        let Demand {
            gpu_slots,
            vram_bytes,
            ram_bytes,
            cpu_threads,
        } = demand;
        u64::from(gpu_slots) > u64::from(self.gpu_slots)
            || (self.vram_bytes > 0 && vram_bytes > self.vram_bytes)
            || (self.ram_bytes > 0 && ram_bytes > self.ram_bytes)
            || (self.cpu_threads > 0 && u64::from(cpu_threads) > self.cpu_threads)
    }

    /// `true` when `committed + demand` exceeds capacity right now (a
    /// transient rejection — the busy path; queueing or a later retry can
    /// grant it).
    fn exhausted(&self, committed: &Committed, demand: Demand) -> bool {
        let Demand {
            gpu_slots,
            vram_bytes,
            ram_bytes,
            cpu_threads,
        } = demand;
        u64::from(committed.gpu_slots.saturating_add(gpu_slots)) > u64::from(self.gpu_slots)
            || (self.vram_bytes > 0 && committed.vram_bytes + vram_bytes > self.vram_bytes)
            || (self.ram_bytes > 0 && committed.ram_bytes + ram_bytes > self.ram_bytes)
            || (self.cpu_threads > 0
                && u64::from(committed.cpu_threads.saturating_add(cpu_threads)) > self.cpu_threads)
    }
}

/// Currently committed (leased) demand totals.
#[derive(Debug, Default, Clone, Copy)]
struct Committed {
    gpu_slots: u32,
    vram_bytes: u64,
    ram_bytes: u64,
    cpu_threads: u32,
}

impl Committed {
    fn add(&mut self, demand: Demand) {
        self.gpu_slots += demand.gpu_slots;
        self.vram_bytes += demand.vram_bytes;
        self.ram_bytes += demand.ram_bytes;
        self.cpu_threads += demand.cpu_threads;
    }

    /// Remove one lease's demand (saturating: an underflow must never panic
    /// the ledger — it would corrupt admission for every later job).
    fn sub(&mut self, demand: Demand) {
        self.gpu_slots = self.gpu_slots.saturating_sub(demand.gpu_slots);
        self.vram_bytes = self.vram_bytes.saturating_sub(demand.vram_bytes);
        self.ram_bytes = self.ram_bytes.saturating_sub(demand.ram_bytes);
        self.cpu_threads = self.cpu_threads.saturating_sub(demand.cpu_threads);
    }
}

/// Per-kind admission settings (queue cap, TTL, per-job demand).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KindSettings {
    /// Bounded queue length for this kind (>= 1; the depth-1 default keeps
    /// the pre-ledger admission policy).
    pub queue_cap: usize,
    /// Queue TTL: a waiter un-admitted after this long is expired (evicted,
    /// counted in `honkbox_ledger_queue_evictions_total`) instead of waiting
    /// forever. `Duration::ZERO` = never expires (the ledger default; the
    /// server always installs its `[server] queue_wait_secs` bound here so
    /// the 429 message's "gave up waiting after Xs" and the actual eviction
    /// agree on one knob).
    pub ttl: Duration,
    /// The per-job resource demand (see [`Demand::for_kind`]).
    pub demand: Demand,
    /// Exclusive kinds cannot start beside any live lease, and no other kind
    /// can start beside them. Used by long training jobs so interactive
    /// generation is never silently displaced or starved mid-run.
    pub exclusive: bool,
}

/// Whole-ledger configuration: capacities + the per-kind settings array.
/// Built once at serve startup (`Ledger::configure`) or per test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerConfig {
    /// Resource ceilings.
    pub capacities: Capacities,
    /// Per-kind settings, indexed by [`ResourceKind::ALL`] order
    /// (sd, llm, ocr, embed, rag, comfy, blender, 3d, training).
    pub kinds: [KindSettings; 9],
}

impl Default for LedgerConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl LedgerConfig {
    /// Defaults: one GPU slot, unmanaged byte/thread budgets, queue cap 1
    /// per kind, no queue TTL, and the default per-kind demands — the exact
    /// pre-ledger admission shape.
    #[must_use]
    pub fn new() -> Self {
        Self {
            capacities: Capacities::default(),
            kinds: ResourceKind::ALL.map(|kind| KindSettings {
                queue_cap: 1,
                ttl: Duration::ZERO,
                demand: Demand::for_kind(kind),
                exclusive: kind == ResourceKind::Training,
            }),
        }
    }

    /// Settings for one kind.
    #[must_use]
    pub fn kind(&self, kind: ResourceKind) -> &KindSettings {
        &self.kinds[kind.slot()]
    }

    /// Mutable settings for one kind (config builders).
    pub fn kind_mut(&mut self, kind: ResourceKind) -> &mut KindSettings {
        &mut self.kinds[kind.slot()]
    }
}

/// Rejection while another job holds the last GPU slot (the legacy
/// single-slot error; unchanged wording so existing 429 bodies stay
/// byte-identical).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Busy;

impl fmt::Display for Busy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gpu slot busy: another job is running; retry later")
    }
}

impl std::error::Error for Busy {}

/// `try_acquire` rejection: the transient busy case queues upstream, the
/// oversized case can never fit (the server maps it to a 503).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquireError {
    /// Capacity is committed to other jobs right now; retry or queue.
    Busy,
    /// The demand alone exceeds the configured capacity — no future free
    /// window can ever grant it. `what` names the binding dimension.
    Oversized {
        /// The binding capacity dimension ("gpu slots", "vram budget", …).
        what: &'static str,
    },
}

impl fmt::Display for AcquireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy => Busy.fmt(f),
            Self::Oversized { what } => write!(
                f,
                "resource demand exceeds the configured ledger capacity ({what}); \
                 no queue can ever grant it"
            ),
        }
    }
}

impl std::error::Error for AcquireError {}

impl From<AcquireError> for Busy {
    fn from(_: AcquireError) -> Self {
        // The legacy probe keeps its exact old error regardless of the new
        // rejection taxonomy (default config never produces `Oversized` for
        // the `sd` one-slot demand anyway).
        Busy
    }
}

/// `enqueue` rejection: the bounded queue for the kind is full (the server
/// maps this to the 429 `queue_full` contract with the shared
/// `Retry-After`) or the demand is oversized (503).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueError {
    /// The kind's bounded queue is at its cap. `retry_after` hints when the
    /// earliest queue slot frees (the configured TTL).
    QueueFull {
        /// When the earliest queue slot frees (the kind's queue TTL).
        retry_after: Duration,
    },
    /// Hard capacity rejection (see [`AcquireError::Oversized`]).
    Oversized {
        /// The binding capacity dimension ("gpu slots", "vram budget", …).
        what: &'static str,
    },
}

impl fmt::Display for EnqueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull { .. } => write!(
                f,
                "gpu slot busy: another job is running and the admission queue is full; \
                 retry later"
            ),
            Self::Oversized { what } => AcquireError::Oversized { what }.fmt(f),
        }
    }
}

impl std::error::Error for EnqueueError {}

/// Proof of reservation; dropping (or [`Lease::release`]) frees exactly the
/// demand the reservation committed — on every path (normal end, cancel
/// mid-job, engine crash unwinding the handler).
#[derive(Debug)]
pub struct Lease<'a> {
    ledger: &'a Ledger,
    kind: ResourceKind,
    demand: Demand,
    since: Instant,
}

impl Lease<'_> {
    /// The kind this lease runs under (read-only probe).
    #[must_use]
    pub fn kind(&self) -> ResourceKind {
        self.kind
    }

    /// Free the reservation NOW, consuming the lease (the explicit form of
    /// dropping it; the body is `Drop`, so consuming keeps double-release
    /// impossible by construction).
    pub fn release(self) {}
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.ledger
            .release_lease(self.kind, self.demand, self.since);
    }
}

/// Outcome of polling a queued [`Ticket`].
#[derive(Debug)]
pub enum PollOutcome<'a> {
    /// The ticket reached its queue head and capacity was free: a lease is
    /// granted and the ticket is consumed from the queue.
    Granted(Lease<'a>),
    /// Still queued (behind an earlier waiter of the same kind, or capacity
    /// busy).
    Waiting,
    /// The ticket waited past its TTL and was evicted (counted in
    /// `honkbox_ledger_queue_evictions_total`).
    Expired,
    /// The ticket was cancelled before admission (or is unknown) — it never
    /// acquired and never will.
    Cancelled,
}

/// A queued reservation request: a place in its kind's bounded queue.
/// Dropping the ticket cancels it — a dropped ticket can never later
/// acquire a lease, and an explicit [`Ticket::cancel`] removes it before
/// admission (a queued job cancelled before admission never acquires).
#[derive(Debug)]
pub struct Ticket<'a> {
    ledger: &'a Ledger,
    id: u64,
    kind: ResourceKind,
}

impl<'a> Ticket<'a> {
    /// The kind this ticket queued under.
    #[must_use]
    pub fn kind(&self) -> ResourceKind {
        self.kind
    }

    /// Check the ticket: granted / still waiting / expired / cancelled.
    /// Polling is non-blocking; its only side effects are the grant and
    /// expiry transitions themselves.
    pub fn poll(&self) -> PollOutcome<'a> {
        self.ledger.poll_ticket(self.id, self.kind)
    }

    /// Cancel the queued ticket (idempotent). Returns `true` when it was
    /// still queued and is now removed without ever acquiring; `false` once
    /// it was already admitted (a running job's cancel releases its LEASE —
    /// the ticket is inert), expired, or cancelled.
    pub fn cancel(&self) -> bool {
        self.ledger.cancel_ticket(self.id, self.kind)
    }

    /// `true` while the ticket sits in its kind's queue (never admitted).
    #[must_use]
    pub fn is_queued(&self) -> bool {
        self.ledger.ticket_queued(self.id, self.kind)
    }
}

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        self.ledger.cancel_ticket(self.id, self.kind);
    }
}

/// One queued waiter (ledger-side state; the ticket holds only the id, so
/// cancel/expiry have a single source of truth under the lock).
#[derive(Debug)]
struct Waiter {
    id: u64,
    enqueued_at: Instant,
}

/// Per-kind counters (fixed 9-wide grids; bounded labels only). Mutated
/// only under the ledger lock, so gauges always agree with the queue/lease
/// state any reader can observe.
#[derive(Debug, Default)]
struct Counters {
    admissions: [u64; 9],
    rejections_busy: [u64; 9],
    rejections_queue_full: [u64; 9],
    rejections_capacity: [u64; 9],
    evictions: [u64; 9],
    oom_near_misses: [u64; 9],
    queue_depth: [u64; 9],
    running: [u64; 9],
    wait_ns: [u64; 9],
    wait_count: [u64; 9],
    hold_ns: [u64; 9],
    hold_count: [u64; 9],
}

/// Read-only counter snapshot for one kind (test + readout surface).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KindStats {
    /// Leases granted (fast path + queue grants).
    pub admissions: u64,
    /// Transient busy rejections on the fast path.
    pub rejections_busy: u64,
    /// Bounded-queue-full rejections.
    pub rejections_queue_full: u64,
    /// Hard oversized (never-fits) rejections.
    pub rejections_capacity: u64,
    /// Queued waiters evicted past their TTL.
    pub evictions: u64,
    /// Admissions landing within the OOM headroom of the VRAM budget.
    pub oom_near_misses: u64,
    /// Currently queued waiters (gauge).
    pub queue_depth: u64,
    /// Currently held leases (gauge).
    pub running: u64,
    /// Total queue-wait time before admission.
    pub wait_total: Duration,
    /// Number of queue-wait observations.
    pub wait_count: u64,
    /// Total lease hold time.
    pub hold_total: Duration,
    /// Number of lease hold observations.
    pub hold_count: u64,
}

/// Ledger-internal state, all under one mutex (admission is a cold path:
/// one lock per reserve/release, never per generated token).
#[derive(Debug)]
struct Inner {
    config: LedgerConfig,
    committed: Committed,
    /// Per-kind FIFO queues (`ResourceKind::ALL` order).
    queues: [VecDeque<Waiter>; 9],
    next_ticket: u64,
    /// When the last lease of a kind was released (`idle_for` hook source).
    last_release: [Option<Instant>; 9],
    counters: Counters,
}

/// Snapshot of the ledger's admission-facing configuration (the values a
/// capability document needs to report: shared slots, the queue TTL, and
/// one row per kind's bounded queue length).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerAdmission {
    /// Shared GPU slots.
    pub gpu_slots: u32,
    /// Queue TTL installed per kind (`Duration::ZERO` = never expires).
    pub queue_ttl: Duration,
    /// `(kind, queue_cap)` per kind, in [`ResourceKind::ALL`] order.
    pub queue_caps: Vec<(ResourceKind, usize)>,
}

/// Capability/VRAM-aware resource ledger with per-kind bounded admission
/// queues. Share by reference (`global()` or a local instance).
#[derive(Debug)]
pub struct Ledger {
    inner: Mutex<Inner>,
}

/// Lock helper: poisoned mutexes must never wedge admission (a panic in one
/// request must not 503 every later request), so poisoning is healed.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Default for Ledger {
    fn default() -> Self {
        Self::new()
    }
}

impl Ledger {
    /// A ledger with the default configuration (one GPU slot, unmanaged
    /// byte/thread budgets, queue cap 1, no TTL — the pre-ledger shape).
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(LedgerConfig::new())
    }

    /// A ledger with an explicit configuration (tests, serve startup).
    #[must_use]
    pub fn with_config(config: LedgerConfig) -> Self {
        Self {
            inner: Mutex::new(Inner {
                config,
                committed: Committed::default(),
                queues: [const { VecDeque::new() }; 9],
                next_ticket: 0,
                last_release: [const { None }; 9],
                counters: Counters::default(),
            }),
        }
    }

    /// Install a resolved configuration. Serve startup calls this once
    /// before the router accepts traffic; mid-traffic swaps are not
    /// supported (already-queued tickets keep their original bounds).
    pub fn configure(&self, config: LedgerConfig) {
        lock(&self.inner).config = config;
    }

    // ------------------------------------------------------------------
    // Legacy single-slot API (preserved verbatim in behavior).
    // ------------------------------------------------------------------

    /// Reserve the GPU slot, or `Err(Busy)` while it is held. The legacy
    /// single-slot probe (kind `sd`, one shared slot): identical semantics
    /// to the pre-ledger ledger, so the CLI one-shots and the `/v1/memory`
    /// readout keep their exact behavior.
    pub fn try_reserve(&self) -> Result<Lease<'_>, Busy> {
        self.try_acquire(ResourceKind::Sd).map_err(Busy::from)
    }

    /// `true` while NO GPU slot is free (read-only probe for the
    /// `/v1/memory` readout; never reserves). With the default one-slot
    /// capacity this is exactly the old single-slot busy flag.
    #[must_use]
    pub fn is_busy(&self) -> bool {
        let inner = lock(&self.inner);
        u64::from(inner.committed.gpu_slots) >= u64::from(inner.config.capacities.gpu_slots)
    }

    // ------------------------------------------------------------------
    // Per-kind admission.
    // ------------------------------------------------------------------

    /// Reserve capacity for `kind` immediately (the admission fast path) or
    /// reject: [`AcquireError::Busy`] when other jobs hold the capacity
    /// right now (the caller may retry or enqueue),
    /// [`AcquireError::Oversized`] when the demand can never fit (a 503
    /// upstream — queueing cannot help).
    ///
    /// This bypasses the per-kind queues by design: an idle capacity is
    /// never delayed behind queue machinery, and a fresh fast-path arrival
    /// CAN win against a queued waiter's wake-up (best-effort ordering, as
    /// documented since REQ-6).
    pub fn try_acquire(&self, kind: ResourceKind) -> Result<Lease<'_>, AcquireError> {
        let mut inner = lock(&self.inner);
        let demand = inner.config.kinds[kind.slot()].demand;
        if Self::exclusive_conflict(&inner, kind) {
            inner.counters.rejections_busy[kind.slot()] += 1;
            return Err(AcquireError::Busy);
        }
        if inner.config.capacities.oversized(demand) {
            inner.counters.rejections_capacity[kind.slot()] += 1;
            return Err(AcquireError::Oversized {
                what: Self::oversized_what(&inner.config, demand),
            });
        }
        if inner.config.capacities.exhausted(&inner.committed, demand) {
            inner.counters.rejections_busy[kind.slot()] += 1;
            return Err(AcquireError::Busy);
        }
        Ok(Self::grant(self, &mut inner, kind))
    }

    /// Join the bounded admission queue for `kind`: `Ok(Ticket)` to poll
    /// for a grant, [`EnqueueError::QueueFull`] when the kind's queue is at
    /// its configured cap (the caller rejects with 429 + the shared
    /// `Retry-After`), [`EnqueueError::Oversized`] when the demand can
    /// never fit (503). The ticket's TTL is the kind's configured `ttl`.
    pub fn enqueue(&self, kind: ResourceKind) -> Result<Ticket<'_>, EnqueueError> {
        let mut inner = lock(&self.inner);
        let demand = inner.config.kinds[kind.slot()].demand;
        if inner.config.capacities.oversized(demand) {
            inner.counters.rejections_capacity[kind.slot()] += 1;
            return Err(EnqueueError::Oversized {
                what: Self::oversized_what(&inner.config, demand),
            });
        }
        let queue_cap = inner.config.kinds[kind.slot()].queue_cap;
        let retry_after = inner.config.kinds[kind.slot()].ttl;
        if inner.queues[kind.slot()].len() >= queue_cap {
            inner.counters.rejections_queue_full[kind.slot()] += 1;
            return Err(EnqueueError::QueueFull { retry_after });
        }
        inner.next_ticket += 1;
        let id = inner.next_ticket;
        inner.queues[kind.slot()].push_back(Waiter {
            id,
            enqueued_at: Instant::now(),
        });
        inner.counters.queue_depth[kind.slot()] =
            u64::try_from(inner.queues[kind.slot()].len()).unwrap_or(u64::MAX);
        Ok(Ticket {
            ledger: self,
            id,
            kind,
        })
    }

    /// Exclusive kinds (long training) never share the box with any other
    /// live lease; conversely no other kind starts beside one. The running
    /// counters are per-kind and updated under the same lock as admission.
    fn exclusive_conflict(inner: &Inner, kind: ResourceKind) -> bool {
        if inner.config.kinds[kind.slot()].exclusive {
            return inner.counters.running.iter().any(|&running| running > 0);
        }
        inner
            .config
            .kinds
            .iter()
            .enumerate()
            .any(|(slot, settings)| settings.exclusive && inner.counters.running[slot] > 0)
    }

    /// Commit a grant under the lock (capacity already checked by the
    /// caller). `ledger` is passed explicitly so the lease can carry the
    /// outer reference (a `&mut Inner` cannot rebuild it).
    fn grant<'a>(ledger: &'a Ledger, inner: &mut Inner, kind: ResourceKind) -> Lease<'a> {
        let demand = inner.config.kinds[kind.slot()].demand;
        inner.committed.add(demand);
        inner.counters.admissions[kind.slot()] += 1;
        inner.counters.running[kind.slot()] += 1;
        let lease = Lease {
            ledger,
            kind,
            demand,
            since: Instant::now(),
        };
        Self::note_vram_pressure(inner, kind);
        lease
    }

    /// Name the binding dimension of an oversized demand (for the 503).
    fn oversized_what(config: &LedgerConfig, demand: Demand) -> &'static str {
        let caps = &config.capacities;
        if u64::from(demand.gpu_slots) > u64::from(caps.gpu_slots) {
            "gpu slots"
        } else if caps.vram_bytes > 0 && demand.vram_bytes > caps.vram_bytes {
            "vram budget"
        } else if caps.ram_bytes > 0 && demand.ram_bytes > caps.ram_bytes {
            "ram budget"
        } else {
            "cpu threads"
        }
    }

    /// OOM near-miss hook: an admission whose committed VRAM lands within
    /// `oom_headroom_bytes` of a managed budget. Estimates, not
    /// measurements — the counter is an early-warning signal, nothing more.
    fn note_vram_pressure(inner: &mut Inner, kind: ResourceKind) {
        let caps = inner.config.capacities;
        if caps.vram_bytes == 0 {
            return;
        }
        let headroom = caps.oom_headroom_bytes.min(caps.vram_bytes);
        if inner.committed.vram_bytes >= caps.vram_bytes - headroom {
            inner.counters.oom_near_misses[kind.slot()] += 1;
        }
    }

    /// Release one lease's demand (from `Lease::drop`) + hold-time metric.
    fn release_lease(&self, kind: ResourceKind, demand: Demand, since: Instant) {
        let mut inner = lock(&self.inner);
        inner.committed.sub(demand);
        inner.last_release[kind.slot()] = Some(Instant::now());
        let slot = kind.slot();
        inner.counters.running[slot] = inner.counters.running[slot].saturating_sub(1);
        inner.counters.hold_count[slot] += 1;
        let held = u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX);
        inner.counters.hold_ns[slot] = inner.counters.hold_ns[slot].saturating_add(held);
    }

    /// Poll a queued ticket (see [`Ticket::poll`]). Grant rule: the ticket
    /// is at ITS kind's queue head, un-expired, and capacity is free — then
    /// it commits and leaves the queue (strict FIFO within a kind; the
    /// fast path may still barge, a queued waiter may not).
    fn poll_ticket(&self, id: u64, kind: ResourceKind) -> PollOutcome<'_> {
        let mut inner = lock(&self.inner);
        let slot = kind.slot();
        let Some(waiter_index) = inner.queues[slot].iter().position(|w| w.id == id) else {
            return PollOutcome::Cancelled;
        };
        if waiter_index != 0 {
            return PollOutcome::Waiting;
        }
        let ttl = inner.config.kinds[slot].ttl;
        let enqueued_at = inner.queues[slot][0].enqueued_at;
        if ttl > Duration::ZERO && enqueued_at.elapsed() >= ttl {
            inner.queues[slot].pop_front();
            inner.counters.evictions[slot] += 1;
            inner.counters.queue_depth[slot] =
                u64::try_from(inner.queues[slot].len()).unwrap_or(u64::MAX);
            return PollOutcome::Expired;
        }
        let demand = inner.config.kinds[slot].demand;
        if inner.config.capacities.exhausted(&inner.committed, demand)
            || Self::exclusive_conflict(&inner, kind)
        {
            return PollOutcome::Waiting;
        }
        inner.queues[slot].pop_front();
        inner.counters.queue_depth[slot] =
            u64::try_from(inner.queues[slot].len()).unwrap_or(u64::MAX);
        inner.counters.wait_count[slot] += 1;
        let waited = u64::try_from(enqueued_at.elapsed().as_nanos()).unwrap_or(u64::MAX);
        inner.counters.wait_ns[slot] = inner.counters.wait_ns[slot].saturating_add(waited);
        PollOutcome::Granted(Self::grant(self, &mut inner, kind))
    }

    /// Cancel a queued ticket: remove it before it can ever acquire.
    /// Returns `true` when it was queued and is now gone.
    fn cancel_ticket(&self, id: u64, kind: ResourceKind) -> bool {
        let mut inner = lock(&self.inner);
        let slot = kind.slot();
        let len_before = inner.queues[slot].len();
        inner.queues[slot].retain(|w| w.id != id);
        let removed = inner.queues[slot].len() != len_before;
        if removed {
            inner.counters.queue_depth[slot] =
                u64::try_from(inner.queues[slot].len()).unwrap_or(u64::MAX);
        }
        removed
    }

    /// `true` while the ticket id sits in its kind's queue.
    fn ticket_queued(&self, id: u64, kind: ResourceKind) -> bool {
        lock(&self.inner).queues[kind.slot()]
            .iter()
            .any(|w| w.id == id)
    }

    // ------------------------------------------------------------------
    // Idle-TTL / shutdown hooks (`/free`-style sweeps, process stop).
    // ------------------------------------------------------------------

    /// Expire every queued ticket whose TTL passed, across all kinds (the
    /// sweep hook for the swap reaper / a `/free`-style endpoint). Returns
    /// the number evicted; each counts in
    /// `honkbox_ledger_queue_evictions_total`.
    pub fn sweep_expired(&self) -> usize {
        let mut inner = lock(&self.inner);
        let now = Instant::now();
        let mut evicted = 0;
        for slot in 0..inner.queues.len() {
            let ttl = inner.config.kinds[slot].ttl;
            if ttl.is_zero() {
                continue;
            }
            let (gone, depth) = {
                let queue = &mut inner.queues[slot];
                let before = queue.len();
                queue.retain(|w| now.duration_since(w.enqueued_at) < ttl);
                (before - queue.len(), queue.len())
            };
            if gone > 0 {
                let gone64 = u64::try_from(gone).unwrap_or(u64::MAX);
                inner.counters.evictions[slot] += gone64;
                inner.counters.queue_depth[slot] = u64::try_from(depth).unwrap_or(u64::MAX);
                evicted += gone;
            }
        }
        evicted
    }

    /// Cancel every queued ticket (the process-stop hook): waiters observe
    /// [`PollOutcome::Cancelled`] instead of hanging on a dying process.
    /// Returns the number cancelled.
    pub fn cancel_all(&self) -> usize {
        let mut inner = lock(&self.inner);
        let mut cancelled = 0;
        for slot in 0..inner.queues.len() {
            cancelled += inner.queues[slot].len();
            inner.queues[slot].clear();
            inner.counters.queue_depth[slot] = 0;
        }
        cancelled
    }

    /// Idle duration for `kind`: `Some(elapsed)` since the kind's last
    /// lease was released while no lease is held (the `/free`-style readout
    /// for an idle-TTL unload decision), `None` while the kind is running
    /// (or never ran — there is nothing idle to unload yet).
    #[must_use]
    pub fn idle_for(&self, kind: ResourceKind) -> Option<Duration> {
        let inner = lock(&self.inner);
        if inner.counters.running[kind.slot()] > 0 {
            return None;
        }
        inner.last_release[kind.slot()].map(|at| at.elapsed())
    }

    /// Counter snapshot for one kind.
    #[must_use]
    pub fn stats(&self, kind: ResourceKind) -> KindStats {
        let inner = lock(&self.inner);
        let c = &inner.counters;
        let slot = kind.slot();
        KindStats {
            admissions: c.admissions[slot],
            rejections_busy: c.rejections_busy[slot],
            rejections_queue_full: c.rejections_queue_full[slot],
            rejections_capacity: c.rejections_capacity[slot],
            evictions: c.evictions[slot],
            oom_near_misses: c.oom_near_misses[slot],
            queue_depth: c.queue_depth[slot],
            running: c.running[slot],
            wait_total: Duration::from_nanos(c.wait_ns[slot]),
            wait_count: c.wait_count[slot],
            hold_total: Duration::from_nanos(c.hold_ns[slot]),
            hold_count: c.hold_count[slot],
        }
    }

    /// Read the current admission-facing configuration (`Ledger::configure`
    /// swaps it; serve startup installs the resolved `[ledger]` settings) —
    /// the values a capability document needs to report: shared slots, the
    /// queue TTL, and one row per kind's bounded queue length.
    #[must_use]
    pub fn admission_info(&self) -> LedgerAdmission {
        let inner = lock(&self.inner);
        LedgerAdmission {
            gpu_slots: inner.config.capacities.gpu_slots,
            queue_ttl: inner
                .config
                .kinds
                .first()
                .map_or(Duration::ZERO, |settings| settings.ttl),
            queue_caps: ResourceKind::ALL
                .iter()
                .map(|kind| (*kind, inner.config.kinds[kind.slot()].queue_cap))
                .collect(),
        }
    }

    /// Render the ledger's Prometheus text fragment (version 0.0.4): the
    /// fixed 9-kind grids for gauges/counters plus queue-wait and lease-hold
    /// summaries. Labels are the closed kind vocabulary and three fixed
    /// rejection reasons — never user data (asserted by the
    /// `metrics_labels_contain_no_user_data` test). Integer math only (the
    /// repo avoids float casts).
    #[must_use]
    pub fn exposition(&self) -> String {
        let inner = lock(&self.inner);
        let c = &inner.counters;
        let mut out = String::new();

        push_grid(
            &mut out,
            "honkbox_ledger_queue_depth",
            "Waiters currently queued per kind.",
            "gauge",
            &c.queue_depth,
        );
        push_grid(
            &mut out,
            "honkbox_ledger_running",
            "Leases currently held per kind.",
            "gauge",
            &c.running,
        );
        push_grid(
            &mut out,
            "honkbox_ledger_admissions_total",
            "Reservations granted per kind.",
            "counter",
            &c.admissions,
        );
        push_grid(
            &mut out,
            "honkbox_ledger_queue_evictions_total",
            "Queued waiters evicted past their TTL per kind.",
            "counter",
            &c.evictions,
        );
        push_grid(
            &mut out,
            "honkbox_ledger_oom_near_misses_total",
            "Admissions landing within the OOM headroom of the VRAM budget, per kind.",
            "counter",
            &c.oom_near_misses,
        );

        push_family(
            &mut out,
            "honkbox_ledger_rejections_total",
            "Rejected reservations per kind and reason (`busy` = fast-path capacity \
             miss, `queue_full` = bounded queue at cap, `capacity` = demand can never \
             fit).",
            "counter",
        );
        for kind in ResourceKind::ALL {
            let slot = kind.slot();
            let _ = writeln!(
                out,
                "honkbox_ledger_rejections_total{{kind=\"{}\",reason=\"busy\"}} {}",
                kind.label(),
                c.rejections_busy[slot]
            );
            let _ = writeln!(
                out,
                "honkbox_ledger_rejections_total{{kind=\"{}\",reason=\"queue_full\"}} {}",
                kind.label(),
                c.rejections_queue_full[slot]
            );
            let _ = writeln!(
                out,
                "honkbox_ledger_rejections_total{{kind=\"{}\",reason=\"capacity\"}} {}",
                kind.label(),
                c.rejections_capacity[slot]
            );
        }

        push_summary(
            &mut out,
            "honkbox_ledger_queue_wait_seconds",
            "Time queued waiters spent before admission, per kind.",
            &c.wait_ns,
            &c.wait_count,
        );
        push_summary(
            &mut out,
            "honkbox_ledger_lease_hold_seconds",
            "Lease hold time per kind.",
            &c.hold_ns,
            &c.hold_count,
        );
        out
    }
}

/// HELP + TYPE header lines for one metric family.
fn push_family(out: &mut String, name: &str, help: &str, kind: &str) {
    let _ = write!(out, "# HELP {name} {help}\n# TYPE {name} {kind}\n");
}

/// Render one fixed 8-kind grid (one series per kind; unpopulated kinds
/// render `0`).
fn push_grid(out: &mut String, name: &str, help: &str, kind: &str, cells: &[u64; 9]) {
    push_family(out, name, help, kind);
    for (label, cell) in ResourceKind::ALL.iter().zip(cells.iter()) {
        let _ = writeln!(out, "{name}{{kind=\"{}\"}} {cell}", label.label());
    }
}

/// Render one per-kind summary (`_sum`/`_count`; integer nanosecond math,
/// 9 fractional digits like the server's histograms).
fn push_summary(out: &mut String, name: &str, help: &str, ns: &[u64; 9], count: &[u64; 9]) {
    push_family(out, name, help, "summary");
    for (kind, (total, n)) in ResourceKind::ALL.iter().zip(ns.iter().zip(count.iter())) {
        let _ = writeln!(
            out,
            "{name}_sum{{kind=\"{}\"}} {}.{:09}",
            kind.label(),
            total / 1_000_000_000,
            total % 1_000_000_000
        );
        let _ = writeln!(out, "{name}_count{{kind=\"{}\"}} {n}", kind.label());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// VRAM-budgeted config: 1 GPU slot, `vram` VRAM budget, 10% headroom.
    fn vram_config(vram: u64, sd_vram: u64, llm_vram: u64) -> LedgerConfig {
        let mut config = LedgerConfig::new();
        config.capacities.vram_bytes = vram;
        config.capacities.oom_headroom_bytes = vram / 10;
        config.kind_mut(ResourceKind::Sd).demand.vram_bytes = sd_vram;
        config.kind_mut(ResourceKind::Llm).demand.vram_bytes = llm_vram;
        config
    }

    // ------------------------------------------------------------------
    // Preserved legacy behavior (pre-ledger unit tests).
    // ------------------------------------------------------------------

    #[test]
    fn second_reserve_fails_until_drop() {
        let ledger = Ledger::new();
        let lease = ledger.try_reserve().expect("first reserve");
        assert_eq!(ledger.try_reserve().unwrap_err(), Busy);
        drop(lease);
        assert!(ledger.try_reserve().is_ok());
    }

    /// Updated for the ledger: the unconditional `Ledger::release()` is gone
    /// (it cannot know WHICH lease's demand to free); explicit release is
    /// now `Lease::release(self)`. Contract change disclosed in the ticket
    /// report — no production caller ever used the old method.
    #[test]
    fn explicit_lease_release_frees_slot() {
        let ledger = Ledger::new();
        let lease = ledger.try_reserve().expect("first reserve");
        assert!(ledger.is_busy());
        lease.release();
        assert!(!ledger.is_busy());
        assert!(ledger.try_reserve().is_ok());
    }

    #[test]
    fn holder_blocks_other_threads() {
        let ledger = Ledger::new();
        let _lease = ledger.try_reserve().expect("first reserve");
        let blocked =
            std::thread::scope(|s| s.spawn(|| ledger.try_reserve().is_err()).join().unwrap());
        assert!(blocked);
    }

    /// The defaults preserve the pre-ledger topology exactly: `embed` (which
    /// serves a GPU pipeline today) contends with an `sd` lease for the one
    /// shared slot, and every budget stays unmanaged.
    #[test]
    fn default_config_preserves_single_slot_topology() {
        let config = LedgerConfig::new();
        assert_eq!(config.capacities.gpu_slots, 1);
        assert_eq!(config.capacities.vram_bytes, 0);
        assert_eq!(config.kind(ResourceKind::Embed).demand.gpu_slots, 1);
        assert_eq!(config.kind(ResourceKind::Rag).demand.gpu_slots, 0);
        assert_eq!(config.kind(ResourceKind::Sd).queue_cap, 1);
        assert_eq!(config.kind(ResourceKind::Sd).ttl, Duration::ZERO);

        let ledger = Ledger::with_config(config);
        let lease = ledger.try_reserve().expect("sd holds the slot");
        assert!(ledger.is_busy());
        assert_eq!(
            ledger.try_acquire(ResourceKind::Embed).unwrap_err(),
            AcquireError::Busy
        );
        drop(lease);
        assert!(ledger.try_acquire(ResourceKind::Embed).is_ok());
    }

    /// Kind labels round-trip and stay inside the fixed vocabulary.
    #[test]
    fn kind_labels_round_trip() {
        for kind in ResourceKind::ALL {
            assert_eq!(ResourceKind::from_label(kind.label()), Some(kind));
        }
        assert_eq!(ResourceKind::from_label("audio"), None);
        assert_eq!(ResourceKind::from_label("sd"), Some(ResourceKind::Sd));
        assert_eq!(ResourceKind::ThreeD.to_string(), "3d");
        assert_eq!(ResourceKind::Training.to_string(), "training");
    }

    /// Training is exclusive even on a box with spare GPU slots: no drift
    /// into interactive work while a long sidecar run holds the ledger.
    #[test]
    fn training_lease_is_exclusive() {
        let mut config = LedgerConfig::new();
        config.capacities.gpu_slots = 2;
        let ledger = Ledger::with_config(config);
        let chat = ledger.try_acquire(ResourceKind::Llm).expect("chat slot");
        assert_eq!(
            ledger.try_acquire(ResourceKind::Training).unwrap_err(),
            AcquireError::Busy,
            "training cannot start beside interactive work"
        );
        drop(chat);
        let training = ledger
            .try_acquire(ResourceKind::Training)
            .expect("training slot");
        assert_eq!(
            ledger.try_acquire(ResourceKind::Sd).unwrap_err(),
            AcquireError::Busy,
            "interactive work cannot start beside training"
        );
        drop(training);
        assert!(ledger.try_acquire(ResourceKind::Sd).is_ok());
    }

    // ------------------------------------------------------------------
    // Ticketed queues: cancel / TTL / bounded caps.
    // ------------------------------------------------------------------

    /// Acceptance: a queued job cancelled before admission NEVER acquires a
    /// lease — even once the capacity it was waiting for frees up.
    #[test]
    fn queued_cancel_never_acquires() {
        let ledger = Ledger::new();
        let lease = ledger.try_reserve().expect("sd holds the slot");
        let ticket = ledger.enqueue(ResourceKind::Llm).expect("cap 1 queue");
        assert!(ticket.is_queued());
        assert!(ticket.cancel());
        // Idempotent: a second cancel is a no-op.
        assert!(!ticket.cancel());
        assert!(!ticket.is_queued());

        drop(lease); // the waited-for capacity frees
        assert!(matches!(ticket.poll(), PollOutcome::Cancelled));
        // The slot is genuinely free for a fresh acquisition — the cancelled
        // ticket did not silently take it. The fresh lease is HELD while the
        // stats assert runs (a temporary would drop and free immediately).
        let fresh = ledger.try_acquire(ResourceKind::Llm).expect("slot free");
        assert_eq!(ledger.stats(ResourceKind::Llm).running, 1);
        assert_eq!(ledger.stats(ResourceKind::Llm).admissions, 1);
        assert_eq!(ledger.stats(ResourceKind::Llm).evictions, 0);
        drop(fresh);
    }

    /// Acceptance: a RUNNING job's cancel releases its lease (the token
    /// path drops the lease; RAII frees the demand), and a ticket cancelled
    /// after admission is inert — capacity frees exactly once.
    #[test]
    fn running_cancel_releases() {
        let ledger = Ledger::new();
        let lease = ledger.try_acquire(ResourceKind::Llm).expect("running");
        assert_eq!(ledger.stats(ResourceKind::Llm).running, 1);
        assert!(ledger.is_busy());

        // The ticket view of an ADMITTED job: once `poll` granted its lease,
        // the ticket is inert — cancel reports false (the RUNNING job's
        // cancel releases its LEASE, not the ticket).
        let mut config = LedgerConfig::new();
        config.kind_mut(ResourceKind::Llm).queue_cap = 2;
        let other = Ledger::with_config(config);
        let ticket = other.enqueue(ResourceKind::Llm).expect("queued");
        let PollOutcome::Granted(admitted) = ticket.poll() else {
            panic!("idle ledger must grant the queued head");
        };
        assert!(!ticket.is_queued());
        assert!(!ticket.cancel(), "already-admitted job: nothing to cancel");
        assert_eq!(other.stats(ResourceKind::Llm).running, 1);
        drop(admitted); // the running job's cancel = lease drop → released
        assert_eq!(other.stats(ResourceKind::Llm).running, 0);

        // Cancelling the RUNNING job = dropping its lease.
        drop(lease);
        assert_eq!(ledger.stats(ResourceKind::Llm).running, 0);
        assert!(!ledger.is_busy());
        assert!(ledger.try_reserve().is_ok());
    }

    /// Acceptance: no queue grows past its configured cap — the overflow is
    /// rejected with a `Retry-After` hint equal to the queue TTL.
    #[test]
    fn queue_cap_rejects_with_retry_after() {
        let mut config = LedgerConfig::new();
        config.kind_mut(ResourceKind::Sd).queue_cap = 2;
        config.kind_mut(ResourceKind::Sd).ttl = Duration::from_secs(7);
        let ledger = Ledger::with_config(config);

        let lease = ledger.try_reserve().expect("slot busy for the test");
        let first = ledger.enqueue(ResourceKind::Sd).expect("cap 2");
        let second = ledger.enqueue(ResourceKind::Sd).expect("cap 2");
        assert_eq!(
            ledger.enqueue(ResourceKind::Sd).unwrap_err(),
            EnqueueError::QueueFull {
                retry_after: Duration::from_secs(7)
            }
        );
        assert_eq!(ledger.stats(ResourceKind::Sd).rejections_queue_full, 1);
        assert_eq!(ledger.stats(ResourceKind::Sd).queue_depth, 2);

        drop(first);
        drop(second);
        assert!(
            ledger.enqueue(ResourceKind::Sd).is_ok(),
            "dropped tickets freed their queue slots"
        );
        drop(lease);
    }

    /// Acceptance: dropping the lease frees capacity on every path; the
    /// hold-time metric records the run.
    #[test]
    fn lease_drop_frees_capacity() {
        let ledger = Ledger::new();
        {
            let _lease = ledger.try_reserve().expect("first");
            assert!(ledger.is_busy());
            assert_eq!(ledger.stats(ResourceKind::Sd).running, 1);
        }
        assert!(!ledger.is_busy());
        assert_eq!(ledger.stats(ResourceKind::Sd).running, 0);
        {
            let _lease = ledger.try_reserve().expect("second");
        }
        let stats = ledger.stats(ResourceKind::Sd);
        assert_eq!(stats.admissions, 2);
        assert_eq!(stats.hold_count, 2);
        assert_eq!(stats.running, 0);
    }

    /// Acceptance: TTL eviction — a waiter past its bound is expired
    /// (evicted), never granted afterwards; the sweep hook finds the same
    /// class of tickets.
    #[test]
    fn ttl_eviction() {
        let mut config = LedgerConfig::new();
        config.kind_mut(ResourceKind::Sd).ttl = Duration::from_millis(40);
        let ledger = Ledger::with_config(config);

        let lease = ledger.try_reserve().expect("slot busy for the test");
        let ticket = ledger.enqueue(ResourceKind::Sd).expect("queued");
        assert!(matches!(ticket.poll(), PollOutcome::Waiting));
        std::thread::sleep(Duration::from_millis(90));
        assert!(matches!(ticket.poll(), PollOutcome::Expired));
        assert!(!ticket.is_queued());
        assert_eq!(ledger.stats(ResourceKind::Sd).evictions, 1);
        assert_eq!(ledger.stats(ResourceKind::Sd).queue_depth, 0);

        // The sweep hook expires the same class of tickets.
        let ticket2 = ledger.enqueue(ResourceKind::Sd).expect("queued");
        std::thread::sleep(Duration::from_millis(90));
        assert_eq!(ledger.sweep_expired(), 1);
        assert!(matches!(ticket2.poll(), PollOutcome::Cancelled));
        assert_eq!(ledger.stats(ResourceKind::Sd).evictions, 2);

        // An expired ticket never grabs the freed slot.
        drop(lease);
        assert!(matches!(ticket.poll(), PollOutcome::Cancelled));
        assert!(ledger.try_reserve().is_ok());
    }

    /// FIFO within a kind: the queued head grants once capacity frees, the
    /// queue-wait metric records the wait, and the next waiter never jumps
    /// ahead.
    #[test]
    fn queued_waiter_grants_fifo_when_capacity_frees() {
        let mut config = LedgerConfig::new();
        config.kind_mut(ResourceKind::Sd).queue_cap = 2;
        let ledger = Ledger::with_config(config);

        let lease = ledger.try_reserve().expect("slot busy for the test");
        let first = ledger.enqueue(ResourceKind::Sd).expect("queued");
        let second = ledger.enqueue(ResourceKind::Sd).expect("queued");

        assert!(matches!(first.poll(), PollOutcome::Waiting));
        // The LATER waiter is refused a grant while the head waits (FIFO).
        drop(lease);
        let PollOutcome::Granted(granted) = first.poll() else {
            panic!("head waiter must grant once capacity frees");
        };
        assert_eq!(granted.kind(), ResourceKind::Sd);
        drop(granted);
        // Now the second waiter is at the head and capacity is free again.
        assert!(matches!(second.poll(), PollOutcome::Granted(_)));

        let stats = ledger.stats(ResourceKind::Sd);
        assert_eq!(stats.admissions, 3);
        assert_eq!(stats.wait_count, 2);
        assert_eq!(stats.queue_depth, 0);
    }

    /// The process-stop hook cancels every queued waiter in one call.
    #[test]
    fn cancel_all_frees_every_queue() {
        let mut config = LedgerConfig::new();
        config.kind_mut(ResourceKind::Sd).queue_cap = 2;
        config.kind_mut(ResourceKind::Llm).queue_cap = 2;
        let ledger = Ledger::with_config(config);
        let lease = ledger.try_reserve().expect("slot busy for the test");
        let sd = ledger.enqueue(ResourceKind::Sd).expect("queued");
        let llm = ledger.enqueue(ResourceKind::Llm).expect("queued");
        assert_eq!(ledger.cancel_all(), 2);
        assert!(matches!(sd.poll(), PollOutcome::Cancelled));
        assert!(matches!(llm.poll(), PollOutcome::Cancelled));
        assert_eq!(ledger.stats(ResourceKind::Sd).queue_depth, 0);
        assert_eq!(ledger.stats(ResourceKind::Llm).queue_depth, 0);
        drop(lease);
    }

    /// The idle hook reads None while running/never-used, and the elapsed
    /// idle time once the last lease releases.
    #[test]
    fn idle_for_tracks_release() {
        let ledger = Ledger::new();
        assert_eq!(ledger.idle_for(ResourceKind::Sd), None);
        let lease = ledger.try_reserve().expect("running");
        assert_eq!(ledger.idle_for(ResourceKind::Sd), None);
        drop(lease);
        let idle = ledger.idle_for(ResourceKind::Sd).expect("idle now");
        assert!(idle < Duration::from_secs(1));
    }

    // ------------------------------------------------------------------
    // VRAM awareness.
    // ------------------------------------------------------------------

    /// A demand that can never fit is a hard rejection EVEN when the box is
    /// completely idle (the server maps this to a 503 — queueing cannot
    /// help).
    #[test]
    fn oversized_demand_rejected_even_when_idle() {
        let ledger = Ledger::with_config(vram_config(100, 10, 200));
        let err = ledger.try_acquire(ResourceKind::Llm).unwrap_err();
        assert_eq!(
            err,
            AcquireError::Oversized {
                what: "vram budget"
            }
        );
        assert_eq!(ledger.stats(ResourceKind::Llm).rejections_capacity, 1);
        assert_eq!(
            ledger.enqueue(ResourceKind::Llm).unwrap_err(),
            EnqueueError::Oversized {
                what: "vram budget"
            }
        );
    }

    /// Managed VRAM budgets gate admission: a fitting-but-tight admission
    /// counts an OOM near miss; a second commitment that would exceed the
    /// budget is refused until the first releases.
    #[test]
    fn vram_budget_gates_admission_and_counts_near_misses() {
        let ledger = Ledger::with_config(vram_config(100, 95, 30));
        // 95 of 100 with headroom 10: admitted, and a near miss (95 >= 90).
        let lease = ledger.try_acquire(ResourceKind::Sd).expect("fits");
        assert_eq!(ledger.stats(ResourceKind::Sd).oom_near_misses, 1);
        // 95 + 30 > 100: transient refusal (the busy path).
        assert_eq!(
            ledger.try_acquire(ResourceKind::Llm).unwrap_err(),
            AcquireError::Busy
        );
        drop(lease);
        assert!(ledger.try_acquire(ResourceKind::Llm).is_ok());
    }

    // ------------------------------------------------------------------
    // Metrics: bounded labels only, never user data.
    // ------------------------------------------------------------------

    /// Every label value the ledger can ever render is from the closed kind
    /// vocabulary or the fixed reason set. The ledger API takes no strings
    /// from requests at all, so this holds by construction — the test pins
    /// it by exercising every counter family and parsing the exposition.
    #[test]
    fn metrics_labels_contain_no_user_data() {
        const KINDS: [&str; 9] = [
            "sd", "llm", "ocr", "embed", "rag", "comfy", "blender", "3d", "training",
        ];
        const REASONS: [&str; 3] = ["busy", "queue_full", "capacity"];

        let mut config = vram_config(100, 95, 200);
        config.kind_mut(ResourceKind::Sd).queue_cap = 1;
        config.kind_mut(ResourceKind::Sd).ttl = Duration::from_millis(30);
        let ledger = Ledger::with_config(config);

        // Exercise every family: admissions, busy + queue_full + capacity
        // rejections, evictions, near misses, gauges, both summaries.
        let lease = ledger.try_acquire(ResourceKind::Sd).expect("fits");
        assert_eq!(
            ledger.try_acquire(ResourceKind::Sd).unwrap_err(),
            AcquireError::Busy
        );
        let ticket = ledger.enqueue(ResourceKind::Sd).expect("queued");
        std::thread::sleep(Duration::from_millis(60));
        assert!(matches!(ticket.poll(), PollOutcome::Expired));
        assert_eq!(
            ledger.try_acquire(ResourceKind::Llm).unwrap_err(),
            AcquireError::Oversized {
                what: "vram budget"
            }
        );
        drop(lease);
        let again = ledger.try_acquire(ResourceKind::Sd).expect("fits");
        drop(again);

        let body = ledger.exposition();
        let mut admissions = 0;
        let mut near_misses = 0;
        let mut evictions = 0;
        let mut capacity_rejects = 0;
        for line in body.lines() {
            let Some((series, value)) = line.split_once(' ') else {
                continue; // HELP/TYPE lines
            };
            if series.starts_with('#') || !series.contains('{') {
                continue;
            }
            let labels = &series[series.find('{').expect("series has brace") + 1..series.len() - 1];
            for pair in labels.split(',') {
                let (key, val) = pair.split_once('=').expect("label k=v");
                let val = val.trim_matches('"');
                match key {
                    "kind" => assert!(
                        KINDS.contains(&val),
                        "label value {val:?} is outside the kind vocabulary"
                    ),
                    "reason" => assert!(
                        REASONS.contains(&val),
                        "label value {val:?} is outside the reason set"
                    ),
                    other => panic!("unexpected label {other:?} in {series}"),
                }
            }
            let name = &series[..series.find('{').expect("series has brace")];
            // Summary `_sum` values are dotted seconds — skip their parse.
            if value.contains('.') {
                continue;
            }
            let value: u64 = value.parse().expect("numeric series value");
            match name {
                "honkbox_ledger_admissions_total" => admissions += value,
                "honkbox_ledger_oom_near_misses_total" => near_misses += value,
                "honkbox_ledger_queue_evictions_total" => evictions += value,
                "honkbox_ledger_rejections_total" if labels.contains("capacity") => {
                    capacity_rejects += value;
                }
                _ => {}
            }
        }
        assert_eq!(admissions, 2, "one fast-path grant + one re-acquire");
        assert_eq!(near_misses, 2, "both sd admissions land within headroom");
        assert_eq!(evictions, 1, "the expired waiter");
        assert_eq!(capacity_rejects, 1, "the oversized llm demand");
        assert!(body.contains("# TYPE honkbox_ledger_queue_depth gauge"));
        assert!(body.contains("honkbox_ledger_queue_depth{kind=\"sd\"} 0"));
        assert!(body.contains("# TYPE honkbox_ledger_queue_wait_seconds summary"));
        // The only queue touch in this flow expired before any grant, so the
        // wait summary stays empty.
        assert!(body.contains("honkbox_ledger_queue_wait_seconds_count{kind=\"sd\"} 0"));
        assert!(body.contains("honkbox_ledger_lease_hold_seconds_sum{kind=\"sd\"} 0."));
        assert!(body.ends_with('\n'));
    }
}

/// Property-style hardening for the ledger contract. proptest was judged
/// too heavy for this deliberately dependency-free crate (its dev tree
/// pulls rand/rusty-fork/tempfile), so these tests drive randomized — but
/// SEEDED, hence reproducible — operation sequences through a tiny in-crate
/// LCG and assert the invariants after EVERY step against an independently
/// maintained mirror:
/// - committed demand never exceeds the configured capacities;
/// - gauges (`running`, `queue_depth`) and counters (`admissions`,
///   `wait_count`, `hold_count`) never drift from observable state;
/// - a cancelled/dropped/expired ticket never acquires (it polls
///   `Cancelled` forever and no admission is ever counted for it);
/// - a granted lease frees its demand exactly once, on either exit path
///   (explicit `release` or drop) — double release is impossible by
///   construction (`release` consumes), which the counter parity asserts;
/// - grants within a kind are strictly FIFO (a non-head waiter can never
///   grant before the head);
/// - the TTL sweep evicts EXACTLY the expired waiters — none missing, none
///   extra — and a still-fresh waiter keeps its queue slot.
#[cfg(test)]
mod property_tests {
    use super::*;

    /// `usize` count in the gauge/counter domain (u64).
    fn u64c(n: usize) -> u64 {
        u64::try_from(n).unwrap_or(u64::MAX)
    }

    /// Slot in the fixed 9-wide arrays (the private `slot()` is not
    /// reachable from a sibling test module).
    fn slot_of(kind: ResourceKind) -> usize {
        ResourceKind::ALL
            .iter()
            .position(|k| *k == kind)
            .expect("kind is in ALL")
    }

    /// Tiny deterministic LCG (Knuth's MMIX constants). Fixed seeds keep
    /// every failure reproducible from the seed alone.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed ^ 0x5EED_1D5E_5EED_1D5E)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        /// Uniform in `[0, bound)` (0 for a zero bound).
        fn below(&mut self, bound: u64) -> u64 {
            if bound == 0 {
                0
            } else {
                self.next_u64() % bound
            }
        }

        fn pick_kind(&mut self) -> ResourceKind {
            let idx = self.index(ResourceKind::ALL.len());
            ResourceKind::ALL[idx]
        }

        /// A live index into a collection of the given length.
        fn index(&mut self, len: usize) -> usize {
            usize::try_from(self.below(u64c(len))).unwrap_or(0)
        }
    }

    /// The test-side mirror of the ledger: every invariant is checked
    /// against this independently maintained model, never against another
    /// ledger readout alone.
    struct Mirror {
        config: LedgerConfig,
        /// Committed (gpu slots, vram, ram, cpu threads) sums.
        committed: (u64, u64, u64, u64),
        /// Index-parallel to the test's `leases` slot vector.
        held: Vec<Option<(ResourceKind, Demand)>>,
        /// `(tickets slot, kind)` of every live queued ticket, in enqueue
        /// order (the FIFO reference).
        queued: Vec<(usize, ResourceKind)>,
        grants: [u64; 9],
    }

    impl Mirror {
        fn new(config: LedgerConfig) -> Self {
            Self {
                config,
                committed: (0, 0, 0, 0),
                held: Vec::new(),
                queued: Vec::new(),
                grants: [0; 9],
            }
        }

        fn demand_of(&self, kind: ResourceKind) -> Demand {
            self.config.kind(kind).demand
        }

        fn grant(&mut self, kind: ResourceKind) {
            let demand = self.demand_of(kind);
            self.committed.0 += u64::from(demand.gpu_slots);
            self.committed.1 += demand.vram_bytes;
            self.committed.2 += demand.ram_bytes;
            self.committed.3 += u64::from(demand.cpu_threads);
            self.held.push(Some((kind, demand)));
            self.grants[slot_of(kind)] += 1;
        }

        fn release(&mut self, demand: Demand) {
            self.committed.0 -= u64::from(demand.gpu_slots);
            self.committed.1 -= demand.vram_bytes;
            self.committed.2 -= demand.ram_bytes;
            self.committed.3 -= u64::from(demand.cpu_threads);
        }

        /// The invariants that must hold after EVERY ledger operation.
        fn assert_invariants(&self, ledger: &Ledger, queue_grants: &[u64; 9]) {
            let caps = self.config.capacities;
            assert!(
                self.committed.0 <= u64::from(caps.gpu_slots),
                "gpu slots overcommitted: {} > {}",
                self.committed.0,
                caps.gpu_slots
            );
            if caps.vram_bytes > 0 {
                assert!(
                    self.committed.1 <= caps.vram_bytes,
                    "vram overcommitted: {} > {}",
                    self.committed.1,
                    caps.vram_bytes
                );
            }
            if caps.ram_bytes > 0 {
                assert!(
                    self.committed.2 <= caps.ram_bytes,
                    "ram overcommitted: {} > {}",
                    self.committed.2,
                    caps.ram_bytes
                );
            }
            if caps.cpu_threads > 0 {
                assert!(
                    self.committed.3 <= caps.cpu_threads,
                    "cpu threads overcommitted: {} > {}",
                    self.committed.3,
                    caps.cpu_threads
                );
            }
            assert_eq!(
                ledger.is_busy(),
                self.committed.0 >= u64::from(caps.gpu_slots),
                "is_busy disagrees with the committed gpu slots"
            );
            for kind in ResourceKind::ALL {
                let stats = ledger.stats(kind);
                let slot = slot_of(kind);
                let running = self
                    .held
                    .iter()
                    .flatten()
                    .filter(|(k, _)| *k == kind)
                    .count();
                assert_eq!(
                    stats.running,
                    u64c(running),
                    "running gauge drifted from held leases for {kind}"
                );
                let queued = self.queued.iter().filter(|(_, k)| *k == kind).count();
                assert_eq!(
                    stats.queue_depth,
                    u64c(queued),
                    "queue_depth gauge drifted from live tickets for {kind}"
                );
                assert_eq!(
                    stats.admissions, self.grants[slot],
                    "admissions counter drifted for {kind}"
                );
                assert_eq!(
                    stats.wait_count, queue_grants[slot],
                    "queue-wait count drifted from poll grants for {kind}"
                );
                // Every granted lease releases exactly once (drop or
                // explicit `release`): released == granted - still running.
                assert_eq!(
                    stats.hold_count,
                    self.grants[slot] - u64c(running),
                    "hold count drifted (double release or leak) for {kind}"
                );
            }
        }
    }

    /// Random admit/queue/poll/cancel/release sequences never violate the
    /// ledger contract (per-step mirror assertions; every ticket that
    /// leaves the queue via cancel/drop is dead forever).
    #[test]
    #[allow(clippy::too_many_lines)] // one continuous invariant block per op
    fn random_operation_sequences_preserve_the_ledger_contract() {
        for seed in 0..48u64 {
            let mut rng = Lcg::new(seed);
            let mut config = LedgerConfig::new();
            config.capacities.gpu_slots = u32::try_from(1 + rng.below(2)).unwrap_or(1);
            config.capacities.vram_bytes = 4096 * rng.below(2);
            config.capacities.ram_bytes = 4096 * rng.below(2);
            config.capacities.cpu_threads = rng.below(3);
            for kind in ResourceKind::ALL {
                let settings = config.kind_mut(kind);
                settings.queue_cap = 1 + rng.index(3);
                settings.demand = Demand {
                    gpu_slots: u32::try_from(rng.below(3)).unwrap_or(0),
                    vram_bytes: 1024 * rng.below(4),
                    ram_bytes: 2048 * rng.below(3),
                    cpu_threads: u32::try_from(rng.below(3)).unwrap_or(0),
                };
            }
            let ledger = Ledger::with_config(config.clone());
            let mut mirror = Mirror::new(config);
            let mut leases: Vec<Option<Lease<'_>>> = Vec::new();
            let mut tickets: Vec<Option<Ticket<'_>>> = Vec::new();
            let mut dead: Vec<usize> = Vec::new();
            let mut queue_grants = [0u64; 9];

            for _ in 0..40 {
                match rng.below(7) {
                    // Fast-path acquire.
                    0 => {
                        let kind = rng.pick_kind();
                        if let Ok(lease) = ledger.try_acquire(kind) {
                            mirror.grant(kind);
                            leases.push(Some(lease));
                        }
                    }
                    // Join the bounded queue (rejections change nothing).
                    1 => {
                        let kind = rng.pick_kind();
                        if let Ok(ticket) = ledger.enqueue(kind) {
                            tickets.push(Some(ticket));
                            mirror.queued.push((tickets.len() - 1, kind));
                        }
                    }
                    // Poll a queued ticket: grants are strictly FIFO.
                    2 if !mirror.queued.is_empty() => {
                        let idx = rng.index(mirror.queued.len());
                        let (slot, kind) = mirror.queued[idx];
                        match tickets[slot].as_ref().expect("live ticket").poll() {
                            PollOutcome::Granted(lease) => {
                                let head = mirror
                                    .queued
                                    .iter()
                                    .find(|(_, k)| *k == kind)
                                    .expect("granted ticket was queued");
                                assert_eq!(
                                    *head,
                                    (slot, kind),
                                    "queue grant out of turn for {kind}"
                                );
                                mirror.queued.remove(idx);
                                mirror.grant(kind);
                                queue_grants[slot_of(kind)] += 1;
                                leases.push(Some(lease));
                            }
                            PollOutcome::Waiting => {}
                            PollOutcome::Expired => {
                                panic!("ttl is zero in this scenario; poll must not expire")
                            }
                            PollOutcome::Cancelled => {
                                panic!("mirror says slot {slot} is queued; ledger disagrees")
                            }
                        }
                    }
                    // Explicit cancel: the ticket never acquires afterwards.
                    3 if !mirror.queued.is_empty() => {
                        let idx = rng.index(mirror.queued.len());
                        let (slot, _) = mirror.queued[idx];
                        let cancel = tickets[slot].as_ref().expect("live ticket").cancel();
                        assert!(cancel, "cancel of a queued ticket must report true");
                        mirror.queued.remove(idx);
                        dead.push(slot);
                        assert!(matches!(
                            tickets[slot].as_ref().expect("live ticket").poll(),
                            PollOutcome::Cancelled
                        ));
                    }
                    // Drop a lease (RAII exit path).
                    4 if leases.iter().any(Option::is_some) => {
                        let idx = rng.index(leases.len());
                        if let Some((_, demand)) = mirror.held[idx].take() {
                            drop(leases[idx].take().expect("live lease"));
                            mirror.release(demand);
                        }
                    }
                    // Explicit release (the other exit path).
                    5 if leases.iter().any(Option::is_some) => {
                        let idx = rng.index(leases.len());
                        if let Some((_, demand)) = mirror.held[idx].take() {
                            leases[idx].take().expect("live lease").release();
                            mirror.release(demand);
                        }
                    }
                    // Drop a queued ticket (cancel-on-drop path).
                    6 if !mirror.queued.is_empty() => {
                        let idx = rng.index(mirror.queued.len());
                        let (slot, _) = mirror.queued[idx];
                        let ticket = tickets[slot].take().expect("live ticket");
                        assert!(ticket.is_queued());
                        drop(ticket);
                        mirror.queued.remove(idx);
                    }
                    _ => {}
                }
                mirror.assert_invariants(&ledger, &queue_grants);
            }

            // Drain: every held lease exits (alternating drop / explicit
            // release — both must free exactly once), every remaining
            // ticket cancels on drop.
            for (idx, lease_slot) in leases.iter_mut().enumerate() {
                if let Some((_, demand)) = mirror.held[idx].take() {
                    mirror.release(demand);
                    let lease = lease_slot.take().expect("live lease");
                    if idx % 2 == 0 {
                        drop(lease);
                    } else {
                        lease.release();
                    }
                }
            }
            for (slot, ticket) in tickets.iter_mut().enumerate() {
                // Cancelled tickets stay observable for the final poll.
                if !dead.contains(&slot) {
                    drop(ticket.take());
                }
            }
            mirror.queued.clear();
            mirror.assert_invariants(&ledger, &queue_grants);

            for kind in ResourceKind::ALL {
                let stats = ledger.stats(kind);
                assert_eq!(stats.running, 0, "{kind} leaked a lease");
                assert_eq!(stats.queue_depth, 0, "{kind} leaked a queued ticket");
                assert_eq!(stats.evictions, 0, "no ttl is configured in this scenario");
                assert_eq!(
                    stats.hold_count, stats.admissions,
                    "{kind}: every grant must release exactly once"
                );
                assert_eq!(
                    stats.wait_count,
                    queue_grants[slot_of(kind)],
                    "{kind}: queue-wait observations must match poll grants"
                );
            }
            assert!(!ledger.is_busy());
            for slot in dead {
                assert!(matches!(
                    tickets[slot]
                        .as_ref()
                        .expect("cancelled ticket stays observable")
                        .poll(),
                    PollOutcome::Cancelled
                ));
            }

            // The rendered exposition agrees with the counters (every 8th
            // sequence, a cheap cross-check of the rendering path).
            if seed % 8 == 0 {
                let body = ledger.exposition();
                for kind in ResourceKind::ALL {
                    let stats = ledger.stats(kind);
                    let k = kind.label();
                    for (series, value) in [
                        (
                            format!("honkbox_ledger_queue_depth{{kind=\"{k}\"}}"),
                            stats.queue_depth,
                        ),
                        (
                            format!("honkbox_ledger_running{{kind=\"{k}\"}}"),
                            stats.running,
                        ),
                        (
                            format!("honkbox_ledger_admissions_total{{kind=\"{k}\"}}"),
                            stats.admissions,
                        ),
                        (
                            format!("honkbox_ledger_queue_evictions_total{{kind=\"{k}\"}}"),
                            stats.evictions,
                        ),
                    ] {
                        let line = format!("{series} {value}");
                        assert!(body.contains(&line), "exposition missing {line:?}");
                    }
                }
            }
        }
    }

    /// Queue grants leave a kind's queue strictly in enqueue order: polls
    /// in randomized order only ever grant the head, one grant per capacity
    /// window, no matter how the waiter polls are interleaved.
    #[test]
    fn queue_grants_are_strictly_fifo_within_a_kind() {
        let mut rng = Lcg::new(0xF10);
        for n in 1..=6 {
            let mut config = LedgerConfig::new();
            config.kind_mut(ResourceKind::Sd).queue_cap = n;
            let ledger = Ledger::with_config(config);
            let holder = ledger.try_acquire(ResourceKind::Sd).expect("initial hold");
            let tickets: Vec<Ticket<'_>> = (0..n)
                .map(|_| ledger.enqueue(ResourceKind::Sd).expect("queued"))
                .collect();

            // While capacity is held, ANY poll order yields Waiting for all.
            for _ in 0..(2 * n) {
                let idx = rng.index(n);
                assert!(matches!(tickets[idx].poll(), PollOutcome::Waiting));
                assert!(tickets[idx].is_queued());
            }

            drop(holder);
            // Grant rounds: exactly the head grants per round; later
            // waiters never overtake.
            let mut live: Vec<usize> = (0..n).collect();
            let mut grant_order = Vec::new();
            while !live.is_empty() {
                let start = rng.index(live.len());
                let mut granted = None;
                for offset in 0..live.len() {
                    let idx = live[(start + offset) % live.len()];
                    match tickets[idx].poll() {
                        PollOutcome::Granted(lease) => {
                            assert!(granted.is_none(), "two grants in one capacity window");
                            assert_eq!(idx, live[0], "a non-head waiter overtook the head");
                            granted = Some(lease);
                        }
                        PollOutcome::Waiting => {}
                        other => panic!("unexpected outcome {other:?} for waiter {idx}"),
                    }
                }
                let lease = granted.expect("head must grant once capacity is free");
                grant_order.push(live.remove(0));
                drop(lease);
                assert_eq!(
                    ledger.stats(ResourceKind::Sd).queue_depth,
                    u64c(live.len()),
                    "queue depth must shrink by exactly one per grant"
                );
            }
            assert_eq!(grant_order, (0..n).collect::<Vec<_>>());
            let stats = ledger.stats(ResourceKind::Sd);
            assert_eq!(stats.admissions, u64c(n + 1));
            assert_eq!(stats.wait_count, u64c(n));
            assert_eq!(stats.hold_count, u64c(n + 1));
        }
    }

    /// REQ-6 best-effort ordering: the FAST path may barge past a queued
    /// waiter when capacity is free — but the queue itself never grants out
    /// of turn (asserted by `queue_grants_are_strictly_fifo_within_a_kind`).
    #[test]
    fn fast_path_may_barge_past_a_queued_waiter() {
        let mut config = LedgerConfig::new();
        config.kind_mut(ResourceKind::Sd).queue_cap = 2;
        let ledger = Ledger::with_config(config);
        let waiter = ledger.enqueue(ResourceKind::Sd).expect("queued");
        let barger = ledger
            .try_acquire(ResourceKind::Sd)
            .expect("fast path barges");
        assert!(matches!(waiter.poll(), PollOutcome::Waiting));
        drop(barger);
        let PollOutcome::Granted(lease) = waiter.poll() else {
            panic!("the queued waiter grants once capacity frees");
        };
        assert_eq!(lease.kind(), ResourceKind::Sd);
        drop(lease);
    }

    /// The TTL sweep evicts EXACTLY the expired waiters: none is left
    /// queued past its bound, none still grants, and the freed capacity is
    /// genuinely free.
    #[test]
    fn ttl_sweep_evicts_exactly_expired_waiters() {
        for seed in 0..4u64 {
            let mut rng = Lcg::new(seed);
            let kind = rng.pick_kind();
            let count = 1 + rng.index(4);
            let mut config = LedgerConfig::new();
            config.kind_mut(kind).ttl = Duration::from_millis(100);
            config.kind_mut(kind).queue_cap = count;
            let ledger = Ledger::with_config(config);
            let holder = ledger.try_acquire(kind).expect("hold the slot");
            let mut tickets = Vec::new();
            for _ in 0..count {
                tickets.push(ledger.enqueue(kind).expect("queued"));
            }

            // Nothing expired yet: the sweep is a no-op.
            assert_eq!(ledger.sweep_expired(), 0);
            assert_eq!(ledger.stats(kind).queue_depth, u64c(count));

            std::thread::sleep(Duration::from_millis(160));
            assert_eq!(ledger.sweep_expired(), count, "exactly the expired waiters");
            let stats = ledger.stats(kind);
            assert_eq!(stats.evictions, u64c(count));
            assert_eq!(stats.queue_depth, 0);
            assert_eq!(stats.admissions, 1, "no evicted waiter ever acquired");
            for ticket in &tickets {
                assert!(!ticket.is_queued());
                assert!(matches!(ticket.poll(), PollOutcome::Cancelled));
            }

            drop(holder);
            assert!(ledger.try_acquire(kind).is_ok(), "capacity genuinely free");
        }
    }

    /// Expiry is per-waiter: a waiter that joined LATER survives a sweep
    /// that evicts an older one, keeps its queue slot, and still grants at
    /// the head afterwards.
    #[test]
    fn ttl_sweep_spares_fresh_waiters() {
        let mut config = LedgerConfig::new();
        config.kind_mut(ResourceKind::Llm).ttl = Duration::from_millis(250);
        config.kind_mut(ResourceKind::Llm).queue_cap = 2;
        let ledger = Ledger::with_config(config);
        let holder = ledger.try_acquire(ResourceKind::Llm).expect("hold");
        let stale = ledger.enqueue(ResourceKind::Llm).expect("queued");
        std::thread::sleep(Duration::from_millis(150));
        let fresh = ledger.enqueue(ResourceKind::Llm).expect("queued");
        std::thread::sleep(Duration::from_millis(150));

        assert_eq!(ledger.sweep_expired(), 1, "exactly the stale waiter");
        assert!(matches!(stale.poll(), PollOutcome::Cancelled));
        assert!(fresh.is_queued());
        let stats = ledger.stats(ResourceKind::Llm);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.queue_depth, 1);

        drop(holder);
        let PollOutcome::Granted(lease) = fresh.poll() else {
            panic!("a fresh waiter still grants at the head after older ones were evicted");
        };
        drop(lease);
        assert_eq!(ledger.stats(ResourceKind::Llm).wait_count, 1);
    }

    /// A demand that can never fit is rejected with `Oversized` naming the
    /// binding dimension on BOTH paths (fast + queue), no matter how often
    /// the caller retries, and never counts an admission.
    #[test]
    fn oversized_demand_is_never_admitted_on_any_path() {
        for seed in 0..64u64 {
            let mut rng = Lcg::new(seed);
            let kind = rng.pick_kind();
            let mut config = LedgerConfig::new();
            config.capacities.gpu_slots = u32::try_from(1 + rng.below(2)).unwrap_or(1);
            config.capacities.vram_bytes = 4096 * rng.below(2);
            config.capacities.ram_bytes = 8192 * rng.below(2);
            config.capacities.cpu_threads = 2 * rng.below(2);
            let demand = Demand {
                gpu_slots: u32::try_from(rng.below(3)).unwrap_or(0),
                vram_bytes: 1024 * rng.below(5),
                ram_bytes: 2048 * rng.below(5),
                cpu_threads: u32::try_from(rng.below(3)).unwrap_or(0),
            };
            config.kind_mut(kind).demand = demand;
            let caps = config.capacities;
            let ledger = Ledger::with_config(config);

            // Reference predicate: the binding dimension per the documented
            // precedence (gpu slots, then vram, ram, cpu).
            let gpu_over = u64::from(demand.gpu_slots) > u64::from(caps.gpu_slots);
            let vram_over = caps.vram_bytes > 0 && demand.vram_bytes > caps.vram_bytes;
            let ram_over = caps.ram_bytes > 0 && demand.ram_bytes > caps.ram_bytes;
            let cpu_over = caps.cpu_threads > 0 && u64::from(demand.cpu_threads) > caps.cpu_threads;
            let expected = if gpu_over {
                "gpu slots"
            } else if vram_over {
                "vram budget"
            } else if ram_over {
                "ram budget"
            } else if cpu_over {
                "cpu threads"
            } else {
                ""
            };

            match ledger.try_acquire(kind) {
                Err(AcquireError::Oversized { what }) => {
                    assert_eq!(what, expected, "binding dimension for {demand:?}");
                    assert!(matches!(
                        ledger.enqueue(kind),
                        Err(EnqueueError::Oversized { what }) if what == expected
                    ));
                    for _ in 0..3 {
                        assert!(ledger.try_acquire(kind).is_err(), "never admitted on retry");
                        assert!(matches!(
                            ledger.enqueue(kind),
                            Err(EnqueueError::Oversized { .. })
                        ));
                    }
                    let stats = ledger.stats(kind);
                    assert_eq!(stats.admissions, 0);
                    assert_eq!(stats.running, 0);
                    assert_eq!(
                        stats.rejections_capacity, 8,
                        "all fast-path and queue attempts count as capacity rejections"
                    );
                }
                other => {
                    assert!(
                        expected.is_empty(),
                        "oversized demand {demand:?} was admitted: {other:?}"
                    );
                }
            }
        }
    }
}
