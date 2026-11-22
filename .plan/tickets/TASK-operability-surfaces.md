# TASK: Operability surfaces for jobs and GC

Status: proposed
Priority: P2
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: per-route request/outcome metrics, GC reclaim counters, and a
bounded `GET /v1/status/warnings` document make retention and admission
behavior observable without log scraping; no counter is unbounded by label.
Verification: `bun run test:server` (metrics + status tests),
`bun run codegen:check`, `bun run lint:rs`.
Links: `.tmp/research/organic-growth-2026-11-20.md` §F,
`TASK-queue-introspection.md`.
Resources: per-test temp data dir; port 0 or harness-assigned.

## Change

- Add fixed route labels and reclaim counters next to the existing metrics and
  GC code; no dynamic label from user input.
- Warnings are a small typed document with additive DTO fields.

## Non-goals

New metrics backend, alerting rules, per-user telemetry.