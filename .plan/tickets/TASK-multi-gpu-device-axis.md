# TASK: Device-aware GPU slots

Status: proposed
Priority: P2
Epic: epic-external-tools
Owner: unassigned
Acceptance: ledger slots carry a device id, demand can express affinity, and
admission never assigns two mutually exclusive jobs to the same device; a
single-device request reports which GPU was picked.
Verification: `bun run audit:gpu-slot`, `cargo test --locked -p
honkbox-residency`, `bun run test:server`, `bun run codegen:check` if DTOs move.
Links: `.tmp/research/multi-gpu-2026-11-20.md` §4.2,
`TASK-device-capacity-gates.md`, `TASK-device-selection-rank-review.md`.
Resources: no real second GPU in CI; tests use injected device inventory.

## Change

- Extend the fixed kind ledger with device id + affinity, keep the single-slot
  default unchanged when no affinity is requested.
- Report the chosen device in the existing job status/capability surface.

## Non-goals

Cross-device batch sharding, `CUDA_VISIBLE_DEVICES` process layouts, a second
scheduler (all in the follow-up recipe ticket).