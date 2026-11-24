# EPIC: Vector and material assets

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/texture-creation-workflows.md`,
`.tmp/research/shaders-materials-uiux-asset-pipeline-2026-11-20.md`

## Scope

Validate SVG and PBR texture-map artifacts, probe operator-installed material
tools, and keep generators external until a spike proves value. Substance
`.sbs`/`.sbsar` stays an operator-licensed external process, never a Rust parser.

## Tickets

- `TASK-svg-artifact-validation.md` — safe SVG validation + `resvg` preview (P2).
- `TASK-svg-generator-spike.md` — external vector generator spike (P3).
- `TASK-material-map-validation.md` — PBR map set + dimensions + hashes (P2).
- `TASK-material-tool-capability-probe.md` — Material Maker / sbsrender probe (P2).
- `TASK-material-generator-spike.md` — local PBR generation spike (P3).

## Acceptance

SVG has no scripts/external refs before rasterization; PBR sets are complete
and versioned; missing licensed tools yield named 501s, not silent fallbacks.

## Verification

`bun run test:server`, `bun run lint:rs`, `bun run deny`; local fixtures only.

## Non-goals

Embedding Substance/Python/GEGL in Rust, bundling third-party material weights,
and generating SVG natively in the server.
