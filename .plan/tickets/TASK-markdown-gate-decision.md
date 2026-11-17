# TASK: Markdown formatting and link gate decision

Status: postponed
Priority: P3
Epic: epic-docs-tooling
Owner: unassigned
Acceptance: a recorded decision on Markdown formatting/link checking before any hook or CI change.
Verification: none now; after a future change run `bun run spdx`, `bun run check`
Links: `.tmp/research/ReviewedDocs-Organization.md`, `dprint.json`
Resources: none (decision ticket)

## Decision inputs

- `dprint.json` currently has TypeScript + JSON plugins only; Markdown is not
  formatted or checked.
- `docs/` and `.plan/` Markdown now exists, so rot is possible but not yet
  gated. The user deferred pre-commit checks (2026-11-17).
- Options: add dprint Markdown plugin, add a link checker, both, or neither.

## Acceptance detail

No `.githooks/` or CI change by this ticket; record the decision first.
