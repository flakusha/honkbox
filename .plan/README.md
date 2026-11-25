# `.plan/` — task tracking

## Purpose

`.plan/` stores implementation tickets and cross-cutting epics as plain Markdown.
It is a planning structure, not a project-status report or roadmap. Current
behavior is defined by repository code; ticket status is a claim to verify.

## Authority

`AGENTS.md` > current repository code > this README > epics > tickets > docs.
`.tmp/research/` is gitignored evidence, not a shippable specification.

## Layout

| Path | Purpose |
| --- | --- |
| `tickets/` | `TASK\|FEAT\|BUG-<slug>.md` implementation, feature, and defect specs |
| `epics/` | `epic-<slug>.md` cross-cutting initiatives and dependency groups |
| `epics-index.md` | Generated epic index |
| `tickets/index.json` | Generated ticket index |
| `feature-matrix.md` | Generated feature/category matrix |
| `interconnect.md` | Generated epic/ticket/path relationship index |
| `code-map.json` | Generated code-to-plan ownership map |

Generated artifacts are owned by their generators. Do not hand-edit them.

## Categories

- `TASK`: bounded implementation, migration, spike, or verification work.
- `FEAT`: user-visible capability or backend family.
- `BUG`: correctness defect with a reproduction and regression contract.
- `epic`: related tickets with shared dependencies and acceptance boundaries.

## Ticket header

```text
Status: proposed | in progress | done | blocked | postponed
Priority: P0 | P1 | P2 | P3
Epic: epic-<slug> (or none)
Owner: <name or unassigned>
Acceptance: <observable result>
Verification: <exact commands>
Links: <research/docs/DTO/route references>
Resources: <per-test tmp dirs/ports/DB names owned>
```

## Epic header

```text
Status: proposed | in progress | done | blocked | postponed
Priority: P0 | P1 | P2 | P3
Owner: <name or unassigned>
Research: <evidence references>
```

## Conventions

- Keep one route/DTO seam or one engine adapter per ticket.
- Use source-cited claims and explicit `[UNKNOWN]` markers for unverified behavior.
- Tests use unique temp dirs, port 0 or explicit offsets, per-test databases or
  collections, deterministic teardown, and no ordering dependence.
- Frontend tickets include `bun run check`, `bun run codegen:check`, and a UI
  surface check. `packages/web-ui/src/gen/` is generated, never hand-edited.
- Use plain git worktrees. Hook tests use per-test temp git repos and no shared
  index or hooks path.
- Cross-link tickets and epics by filename. `giwt plan check-links` checks
  Markdown links; `bun run audit:plan-refs` checks code-formatted plan references.

## Validation

- `giwt plan check-links`
- `bun run audit:plan-refs`
- `bunx markdownlint-cli2 <changed .plan markdown files>`
- `bun run codegen:check` after DTO/OpenAPI changes
- Product changes use the narrow per-ticket commands, then `bun run check` /
  `bun run test` as appropriate.
