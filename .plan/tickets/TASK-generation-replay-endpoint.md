# TASK: Replay a recorded generation

Status: proposed
Priority: P2
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: `POST /v1/jobs/{id}/replay` reuses the recorded request with
optional bounded overrides, creates a new job id, and reports the old and new
output hashes so determinism can be compared without overwriting the original.
Verification: `bun run test:server`, `bun run codegen:check`,
`bun run audit:gpu-slot`; a fake-backend test asserts a fresh lease and bundle.
Links: `.tmp/research/organic-growth-2026-11-20.md` §B,
`TASK-determinism-contract.md`, `TASK-generation-record-history-api.md`.
Resources: per-test temp data dir.

## Change

- Read the same manifest the history route uses; validate overrides through
  the existing generation contract before admission.
- No cache hit on replay by default: it must actually re-run unless a separate
  hash-cache experiment opts in.

## Non-goals

Cross-backend byte-equality claims, automatic replay on load.