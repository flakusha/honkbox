# TASK: Make plan indexes regenerate from honkbox sources

Status: proposed
Priority: P2
Epic: epic-docs-tooling
Owner: unassigned
Acceptance: one documented command set regenerates `.plan/epics-index.md`, `.plan/feature-matrix.md`, `.plan/interconnect.md`, and `.plan/code-map.json` from the current 24 epics and ticket headers with correct statuses, epic bindings, counts, and `crates/` source paths; no generated file is hand-edited.
Verification: `giwt plan gen-docs --check`, `giwt plan code-map --check`, `giwt plan links --check`, `giwt plan check-links`, `giwt plan validate --gates format,linkage,naming,spdx,status-vocab,epics-doc,matrix,code-map,links,interconnect`
Links: `.plan/README.md`, `.plan/epics-index.md`, `.plan/feature-matrix.md`, `.plan/interconnect.md`, `.plan/code-map.json`, `.tmp/research/plan-research-promotion-index-2026-11-25.md`
Resources: read-only plan fixtures in per-test temp dirs; no shared index, cache, git issue, or worktree state

## Observed drift (2026-11-25)

- `.plan/epics/` contains 24 epics; `.plan/epics-index.md` reports 23, omits `epic-salvo-migration.md`, and `giwt plan gen-docs --check` reports it stale.
- `.plan/tickets/` contains 193 ticket files; `.plan/feature-matrix.md` reports 201 stale records and `.plan/interconnect.md` reports 178 tickets/23 epics. The matrix freshness gate passes only because it validates the existing artifact shape, not these counts.
- `.plan/code-map.json` and `.plan/interconnect.md` use `src/...` keys, while this repository implements code under `crates/<crate>/src/...`; `giwt plan code-map --check` and `giwt plan links --check` report both artifacts stale.
- Installed `giwt plan` has no `matrix` subcommand. Its format gate expects bold frontmatter/required sections that honkbox tickets do not use (1,290 format errors), and its default status vocabulary does not include the repository's documented `proposed`/`done` values.
- `giwt plan check-links` passes 220 Markdown files. Link validity is not the current defect.

## Change

- Decide whether to adapt giwt configuration/header parsing or use a small honkbox-owned generator. Reuse the existing index schema where possible; do not add a second plan database.
- Parse the repository's actual plain `Status:`, `Priority:`, `Epic:`, `Owner:`, `Acceptance:`, `Verification:`, `Links:`, and `Resources:` headers. Preserve explanatory status suffixes while deriving a canonical status for aggregates.
- Derive code ownership from real repository paths, including `crates/`, `packages/`, `scripts/`, and `vendor_patches/`; never invent a root `src/` tree.
- Generate all four artifacts atomically from one input snapshot. A stale or partially written artifact fails the check instead of being silently accepted.
- Keep `.plan/README.md` as the human authority until the generator gate is green; document the exact command set and failure behavior there.

## Acceptance detail

- Counts match a direct file inventory: 24 epics and the current ticket-file count, with no epic counted as a ticket.
- A known done ticket and a known proposed epic appear in the correct aggregate columns; `epic-salvo-migration.md` appears once.
- `giwt plan code-map --find crates/server/src/swap.rs` (or the chosen equivalent) returns the tickets that actually own that path.
- Running either generator command twice produces byte-identical output; interrupting it leaves the previous complete artifact set intact.
- `giwt plan check-links` remains green and the scoped validation gates above pass without hand edits or `--force`.

## Non-goals

Adopting giwt worktrees, creating git issues, changing ticket status semantics, moving `.plan/` into another repository, or building a docs site.
