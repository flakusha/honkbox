# TASK: Embed A1111-style parameters chunk on PNG outputs

Status: proposed
Priority: P2
Epic: epic-output-formats
Owner: unassigned
Acceptance: PNG outputs carry a `parameters` tEXt chunk (prompt/seed/model/dims) readable by A1111-family tooling; pixels and determinism unchanged.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-sd-core`, `bun run codegen:check`
Links: `.tmp/research/image-metadata-provenance-2026-11-25.md` (H-IMG-META-2)
Resources: per-test output dirs; no shared encode cache

## Change

- Re-encode PNG through `png@0.18` (`Info::add_text_chunk`) when a metadata payload is requested; keep `image` for pixels. `png@0.18` is already in `Cargo.lock` (via `image`), so this adds a direct edge, not a new crate version.
- `image@0.25` encoder has no text-chunk API; do not hand-roll chunks.

## Acceptance detail

- Same job re-run produces byte-identical PNG (fixed chunk order, no wall-clock text); chunk absent when no params exist.
