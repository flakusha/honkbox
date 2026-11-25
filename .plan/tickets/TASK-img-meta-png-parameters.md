# TASK: Embed A1111-style parameters chunk on PNG outputs

Status: In Progress
**Scope:** Emit the A1111 `parameters` tEXt chunk on PNG outputs
Priority: P2
Epic: epic-output-formats
Owner: unassigned
Acceptance: PNG outputs carry a `parameters` tEXt chunk (prompt/seed/model/dims) readable by A1111-family tooling, emitted beside the existing honkbox record chunk; pixels and determinism unchanged.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-sd-core`, `bun run codegen:check`
Links: `.tmp/research/image-metadata-provenance-2026-11-25.md` (H-IMG-META-2); `crates/server/src/image_metadata.rs`
Resources: per-test output dirs; no shared encode cache

## Change (premise corrected 2026-11-25)

- [stale premise removed] Master `80a5836` landed container-level provenance embedding: `crates/server/src/image_metadata.rs` copies container bytes verbatim and never re-encodes pixels (header comment, `image_metadata.rs:1-7`). It inserts iTXt chunks after IHDR via a hand-rolled chunk writer — `embed_png` (`image_metadata.rs:175-246`), `png_text` (`:248-254`), `png_chunk` (`:256-264`), keyword `honkbox_generation_record` (`RECORD_KEYWORD`, `:19`). No `png@0.18` direct dependency and no pixel re-encode is needed or wanted here.
- Add an A1111-style `parameters` tEXt chunk (plain text: prompt, seed, model, dimensions) beside the existing record chunk inside `image_metadata.rs` (`embed_png`), reusing `png_text`/`png_chunk`; fixed chunk order (record, parameters, XMP); no pixel re-encode; no new dependency.
- Keep the existing `MAX_METADATA_BYTES` bound (`image_metadata.rs:17`) applying to the combined payload.

## Acceptance detail

- Same job re-run produces byte-identical PNG (fixed chunk order, no wall-clock text); `parameters` chunk absent when no params exist; the existing record/XMP chunks and their order stay stable.

## Verification Notes (2026-11-25)

- Implemented in `crates/server/src/image_metadata.rs`: `png_parameters()` derives prompt/negative prompt/seed/model/dimensions from the canonical record; `png_plaintext()` writes one `tEXt` chunk; `embed_png()` emits record → parameters → XMP and skips the chunk when no recipe facts exist.
- Inline tests: `png_parameters_chunk_is_deterministic_and_ordered` (chunk order, byte-identical repeat embed, unchanged pixels) and `png_parameters_chunk_is_absent_without_recipe_facts`.
- Sibling ticket `TASK-img-meta-comfyui-pair.md` stays `proposed`; its emit/ingest pair is a separate slice.
- Verified 2026-11-25: `cargo test --locked -p honkbox-server --bin honkbox image_metadata` 12 pass, 0 fail (after `cargo clean -p honkbox-api-types` to clear the shared-target artifact collision); `cargo fmt --all -- --check` clean.
