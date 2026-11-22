# TASK: dedicated reservation class for long training jobs

Status: done (2026-11-22)
Priority: P2
Epic: epic-external-tools
Owner: unassigned
Acceptance: long training jobs have an admission class with VRAM/RAM demand,
queue policy and preemption rules; training never auto-starts alongside
interactive chat/image work, and cancel means checkpoint first then kill.
Verification: residency + server tests for the new class, `bun run check`,
`bun run audit:gpu-slot`
Links: `FEAT-3d-training-jobs.md` (policy precedent), `FEAT-fizgig-lora-training-jobs.md`,
`crates/residency/src/lib.rs` (fixed 8-kind `ResourceKind` grid),
`crates/server/src/metrics.rs` (fixed job-kind label grid)
Resources: TBD at implementation

## Change

- Decide whether the closed `ResourceKind` vocabulary gains a ninth
  `Training` kind (metric/config label grids and tests must grow with it) or
  whether an existing reserved external-GPU class is formalized.
- Define demand estimates, concurrency = 1, TTL, and a preemption order that
  keeps interactive generation authoritative.
- Migrates the Fizgig first slice off its temporary `Comfy` class borrowing.

## Landed

Ninth `ResourceKind::Training` (label `training`) grows every fixed grid from
8 to 9. `KindSettings.exclusive` is ONE-WAY: training refuses to start while
ANY lease is live, but a live training lease does not block other kinds —
`rag` (CPU-only) and interactive GPU work keep running. Concurrency 1 holds
with the default single GPU slot; the queue TTL and cancel-with-checkpoint
behavior are unchanged. Fizgig's runner acquires `Training`, and
`ADMITTING_ROUTES` advertises `/v1/fizgig/train`. A later audit closed four
gaps: the runner function was renamed so `bun run audit:gpu-slot` actually
derives the route, the fizgig test takes `SLOT_LOCK`, `Oversized` is checked
before exclusivity again, and stale 8-kind prose/API docs were regenerated.

Evidence: `cargo test --locked --offline -p honkbox-residency` → 23 passed
(incl. `training_defers_to_live_work`), `... -p honkbox-server --test fizgig
--test capabilities --test licenses` → 361 passed, `bun run audit:gpu-slot`
now lists `/v1/fizgig/train` and passes. Full preemption of a running
training job is not implemented — with one slot, interactive GPU work queues
behind it; the one-way rule only guarantees priority at admission.
