# TASK: Video job class and admission

Status: proposed
Priority: P2
Epic: epic-video-support
Owner: unassigned
Acceptance: `JobKind::video` has bounded frames/fps/edge caps, a VRAM estimate, TTL, preemption, and a bounded `kind` metric label.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run lint:rs`
Links: `.tmp/research/ReviewedSupport-Video.md` rollout step 3
Resources: per-test job IDs and temp bundle dirs; no shared GPU slot

## Change

- Depends on a green `TASK-video-ffi-spike.md`.
- Register the kind with the residency ledger and queue; never silently share
  the interactive slot with chat/image work.
- Cancel responses state interruptibility honestly until `TASK-sd-ffi-cancel.md`
  is available for video too.

## Acceptance detail

A video job cannot exceed configured budgets; one client cannot starve other
modalities; metrics carry only kind labels, never slug labels.
