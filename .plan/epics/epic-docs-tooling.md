# EPIC: Docs tooling decisions

Status: in progress
Priority: P3
Owner: unassigned
Research: `.tmp/research/plan-research-promotion-index-2026-11-25.md`,
`.tmp/research/worktree-research-index-2026-11-24.md`

## Scope

The `.plan/` and `docs/README.md` scaffold and pre-commit hardening have landed.
Generated plan indexes exist but currently drift from honkbox headers and crate
paths; repair their generator before treating them as authority. A docs site and
dprint Markdown plugin remain deferred.

## Tickets

- `TASK-plan-index-generator.md` — regenerate epic, feature, interconnect, and
  code-map artifacts from current headers and real repository paths (P2).
- `TASK-docs-site-decision.md` — VitePress (separate MIT dep, Node 22+) vs plain
  Markdown, decided only when there is content to publish.
- `TASK-markdown-gate-decision.md` — dprint Markdown plugin + link check + CI gate.
- `TASK-precommit-gates.md` — hardened pre-commit gates ported from the audited
  loop-lore hook (tool guards, partial-staging guard, captured reports,
  restage verification); generated drift stays in pre-push/CI.

## Acceptance

Each ticket records a decision with cost/runtime tradeoffs before any dependency
or hook change. The user approved pre-commit hardening (2026-11-18), so
`TASK-precommit-gates.md` landed hook changes; no dprint Markdown plugin is
added in this round. Plan indexes become authoritative only after one command
set regenerates them without hand edits.

## Verification

`giwt plan check-links`; scoped `bunx markdownlint-cli2 <changed .plan files>`;
`bun run spdx`, `bun run check` after any future product/hook change. For hook
changes verify with `sh -n .githooks/pre-commit`,
`bun test --isolate scripts/pre-commit-hook.test.ts`, `bun run test:unit`, and
`bun run check`.

## Non-goals

Creating empty `docs/` subdirectories, adding VitePress now, adding a dprint
Markdown plugin in this round, or adopting giwt worktrees.
