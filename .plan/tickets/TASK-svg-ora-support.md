# TASK: Safe SVG/ORA read path

Status: proposed
Priority: P2
Epic: epic-binary-formats
Owner: unassigned
Acceptance: SVG and ORA inputs are parsed safely, with scripts/external refs rejected and layers capped before any job is admitted.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`
Links: `.tmp/research/ReviewedBinaryFormats.md`, `TASK-svg-artifact-validation.md`, `TASK-ora-output.md`
Resources: per-test SVG/ORA fixtures; no shared renderer state

## Change

- Reuse SVG validation rules from `epic-vector-materials`; ORA uses the
  container safety rules.
- Convert SVG to PNG via `resvg` when a raster input is required; keep the raw
  artifact and hash in the bundle.
- Cap layer count and decoded dimensions for ORA.

## Acceptance detail

A malicious SVG or ORA is rejected before GPU work; a valid one produces a
deterministic preview or layer set.
