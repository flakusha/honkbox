# TASK: Operability surfaces for jobs and GC

Status: Done
**Scope:** Operability surfaces: bounded route/GC metrics and status warnings
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

## Verification Notes (2026-11-25)

- Static gates in `tree/fix-operability-surfaces`: `cargo fmt --all -- --check`,
  `bunx markdownlint-cli2 .plan/tickets/TASK-operability-surfaces.md`,
  `giwt plan check-links`, and `giwt plan validate --gates naming,linkage,matrix,links`
  all pass (linkage emits only the repo-wide advisory about unbounded epics).
- Implementation evidence: `crates/server/src/metrics.rs` defines the four
  fixed `GcReason` labels and two exposition families;
  `crates/server/src/jobs.rs` calls `note_gc_reclaim` once per successful
  removal; `crates/server/src/status.rs` caps the warning document at 64;
  `crates/server/src/routes.rs` and `crates/server/src/salvo_routes/health_downloads.rs`
  serve the same document from both router stacks.
- Runtime verification is blocked by a pre-existing broken server build.
  `cargo check --manifest-path /home/flak/git-ai/honkbox/Cargo.toml --locked
  -p honkbox-server --bin honkbox` on the clean root checkout fails with the
  same 10 errors as the worktree's
  `cargo test -p honkbox-server --test metrics --test status_warnings`.
  Errors are confined to untouched `routes/blender.rs`, `routes/presets.rs`,
  and `routes/image_ops_settle.rs`; none points at this ticket's files.
  Ticket stays In Progress until that baseline is fixed.
- No api-types or generated bindings changed, so `bun run codegen:check` is
  not implicated by this surface.

**Resolved:** 2026-11-26 registry-driven close: git issue c2fef6c (registry tip: 9d3ad7c Konstantin Fedotov Close issue)
