# `.plan/` — task tracking

Research: `.tmp/research/ReviewedDocs-Ticketing.md`. Prior art:
`../loop-lore/.plan/README.md`. Minimal shape; no giwt, generated artifacts,
or matrices yet.

## Authority

`AGENTS.md` > `.plan/` > `src/` > `docs/`. Current behavior is what `src/` does;
specs and docs may drift.

## Layout

| Path | Purpose |
| --- | --- |
| `tickets/` | `TASK-|FEAT-|BUG-<slug>.md` implementation/feature/fix specs |
| `epics/` | `epic-<slug>.md` cross-cutting initiatives |

## Ticket header

```
Status: proposed | in progress | done | blocked | postponed
Priority: P0 | P1 | P2 | P3
Epic: epic-<slug> (or none)
Owner: <name or unassigned>
Acceptance: <observable result>
Verification: <exact commands>
Links: <research/docs/DTO/route references>
Resources: <per-test tmp dirs/ports/DB names owned>
```

## Conventions

- Research stays in gitignored `.tmp/research/`. Promote it here only with an
  owner, acceptance criteria, and a verification command.
- Frontend tickets must include `bun run check`, `bun run codegen:check`, and
  a UI surface check; `packages/web-ui/src/gen/` is generated, never hand-edited.
- Tests are designed for parallel execution: unique tmp dirs / port 0 or offsets
  / per-test DB or collection names, deterministic teardown in `afterEach`/
  `finally`, no ordering dependence. State the resource contract in `Resources:`. Serial runs are shakeout only, not the end goal.
- Use plain git worktrees; giwt is not adopted. Pre-commit hooks are not yet
  wired into these tickets (2026-11-17).
- Keep ticket diffs small: one route/DTO seam or one engine adapter per ticket.

## Current epics

- [epic-full-ai-cycle.md](epics/epic-full-ai-cycle.md)
- [epic-blender-assets.md](epics/epic-blender-assets.md)
- [epic-game-engine-export.md](epics/epic-game-engine-export.md)
