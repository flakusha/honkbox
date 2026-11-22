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
8 to 9. `KindSettings.exclusive` makes training refuse to start beside ANY
live lease and refuses other kinds beside it — concurrency 1 even with
multiple GPU slots; the queue TTL and cancel-with-checkpoint behavior are
unchanged. Fizgig's runner now acquires `Training`, and `ADMITTING_ROUTES`
advertises `/v1/fizgig/train`. Evidence: `cargo test --locked --offline
-p honkbox-residency` → 23 passed (incl. new exclusivity test);
`... -p honkbox-server --test capabilities --test fizgig` → 290 passed;
`bun run audit:gpu-slot` green.
