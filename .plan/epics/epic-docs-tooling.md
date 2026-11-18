# EPIC: Docs tooling decisions

Status: proposed
Priority: P3
Owner: unassigned
Research: `.tmp/research/ReviewedDocs-Organization.md`,
`ReviewedDocs-Ticketing.md`

## Scope

Deferred decisions only. The `.plan/` and `docs/README.md` scaffold already
landed; a docs site, Markdown formatting plugin, and link checks are **not**
being built in this round.

## Tickets

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
added in this round.

## Verification

`bun run spdx`, `bun run check` after any future change; for hook changes verify with `sh -n .githooks/pre-commit`, `bun test --isolate scripts/pre-commit-hook.test.ts`, `bun run test:unit`, and `bun run check`.

## Non-goals

Creating empty `docs/` subdirectories, adding VitePress now, adding a dprint
Markdown plugin in this round.
