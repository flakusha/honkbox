# TASK: Model kinds server coverage

Status: proposed
Priority: P1
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: each of the 8 new ModelKind values projects through `catalog_entry` / `modality_for` to `ModelCatalogEntry`, `cargo test --locked -p honkbox-server` catalog tests are green, and `gen/openapi.json` contains the new kind strings
Verification: `cargo test --locked -p honkbox-server --test catalog` then `bun run codegen` + `git diff --stat packages/web-ui/src/gen/openapi.json`
Links: `tree/feat-registry-model-kinds` (branch `feat/registry-model-kinds`, commit `2ed7ab7`), `.tmp/research/model-recognition-classification-2026-11-26.md`
Resources: per-test tmp model dirs; no real weights, no downloads, no GPU

## Change

- Reproduce the reported server compile errors in untouched files (`crates/server/src/blender.rs`, `presets.rs`, `image_ops_settle.rs`) first; fix if one-line/obvious, else scope them out of this ticket with a BUG ticket and proceed.
- Add catalog-scoped server tests asserting each of the 8 new kinds maps via `catalog_entry` / `modality_for` to the expected `ModelCatalogEntry` (kind, modality, family grouping from the header-ladder classification).
- Regenerate `packages/web-ui/src/gen/openapi.json` via the repo generator only (never hand-edit).

## Acceptance detail

- Narrowest green: `cargo test --locked -p honkbox-server` catalog tests pass; `rg '<new-kind-string>' packages/web-ui/src/gen/openapi.json` hits for all 8 kinds.
- PII-safe: fixtures use synthetic arch/header bytes, never real checkpoint filenames.
- Licenses stay operator-responsibility; no license text added.
