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

## Acceptance

Each ticket records a decision with cost/runtime tradeoffs before any dependency
or hook change; no pre-commit hook is added by this epic.

## Verification

`bun run spdx`, `bun run check` after any future change; none required now.

## Non-goals

Creating empty `docs/` subdirectories, adding VitePress now, editing `.githooks/`.
