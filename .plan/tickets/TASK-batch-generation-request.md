# TASK: Batch generation with contact sheet

Status: proposed
Priority: P2
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: one generation request can request N seeds/frames under one job
record and lease; the bundle contains every result plus a bounded contact-sheet
artifact, and partial failure reports which variants failed.
Verification: `bun run test:server`, `bun run audit:gpu-slot`,
`bun run codegen:check`; a test asserts N results, one lease and sheet bounds.
Links: `.tmp/research/content-experience-gaps-2026-11-20.md` §5 G14,
`FEAT-blender-job-route.md` (turntable strip precedent).
Resources: per-test temp dir; no fixed port.

## Change

- Extend the generation DTO with a bounded count (additive field, codegen).
- Reuse the existing per-job bundle and supervisor; cap N and sheet dimensions
  so one request cannot fill the data dir.

## Non-goals

Cross-request batching or scheduler changes; one request, one lease.