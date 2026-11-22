# TASK: dedicated reservation class for long training jobs

Status: proposed
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
