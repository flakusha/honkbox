# TASK: Safe SVG artifact validation

Status: proposed
Priority: P2
Epic: epic-vector-materials
Owner: unassigned
Acceptance: SVG inputs are validated for scripts/external refs and optionally rasterized with `resvg`; unsafe input returns a named 400.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedCreative-SVG.md`
Resources: per-test SVG fixture dirs; no shared renderer state

## Change

- Reject `<script>`, `on*` handlers, external `href`/`url()` references, and
  entity expansion before rasterization.
- Cap element count, nesting depth, and decoded byte size.
- Add `resolution` and output PNG/WebP; keep raw SVG in the job bundle with a hash.

## Acceptance detail

A safe SVG produces a deterministic preview; an unsafe one fails before any
renderer is invoked.
