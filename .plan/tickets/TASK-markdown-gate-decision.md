# TASK: Markdown formatting and link gate decision

Status: done
Priority: P3
Epic: epic-docs-tooling
Owner: unassigned
Acceptance: a recorded decision on Markdown formatting/link checking before any hook or CI change.
Verification: `bun run hooks:check`, `bun run check`
Links: `.tmp/research/ReviewedDocs-Organization.md`, `dprint.json`
Resources: none (decision ticket)

## Decision (2026-11-18)

No dprint Markdown plugin is added now. `.md` is removed from the staged
dprint bucket in `.githooks/pre-commit` (it was dead without the plugin), the
CI step label was corrected, and Markdown is intentionally NOT dprint-gated.
Revisit only when a Markdown formatting gate is requested. The original
deferral below is superseded by user approval of pre-commit hardening on
2026-11-18; see `TASK-precommit-gates.md`.

## Decision inputs

- `dprint.json` currently has TypeScript + JSON plugins only; Markdown is not
  formatted or checked.
- `docs/` and `.plan/` Markdown now exists, so rot is possible but not yet
  gated. The user deferred pre-commit checks (2026-11-17).
- Options: add dprint Markdown plugin, add a link checker, both, or neither.

## Acceptance detail

Original scope: no `.githooks/` or CI change by this ticket; record the
decision first. Superseded by the 2026-11-18 decision above.
