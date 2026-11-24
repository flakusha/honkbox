# TASK: OpenRaster output

Status: proposed
Priority: P2
Epic: epic-editor-integration
Owner: unassigned
Acceptance: editor jobs can emit `result.ora` with stable `stack.xml`, relative paths, and PNG layers, alongside the flat output.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/dcc-closed-loop-honkbox-2026-11-25.md` (H-DCC2; supersedes historical `ReviewedImageEdit-Clients.md` §Layered interchange)
Resources: per-test ORA dirs; no shared layer cache

## Change

- Write ORA zip entries in sorted order with fixed timestamps; no wall-clock
  metadata.
- Cap layer count, decoded dimensions, and uncompressed size.
- Record layer names/order and per-layer hashes in the sidecar.

## Acceptance detail

A produced ORA opens in GIMP/Krita with the expected layer order; a re-run of
the same job produces byte-identical ORA.
