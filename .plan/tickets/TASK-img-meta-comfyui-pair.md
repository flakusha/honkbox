# TASK: ComfyUI prompt/workflow metadata pair

Status: blocked
Priority: P2
Epic: epic-output-formats
Owner: unassigned
Acceptance: honkbox PNGs carry ComfyUI-readable `prompt`/`workflow` tEXt chunks scoped to what honkbox actually knows; ComfyUI-produced PNGs uploaded to honkbox surface their embedded `prompt`/`workflow`/`parameters` in the job record for debugging.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/image-metadata-provenance-2026-11-25.md` (H-IMG-META-3)
Resources: per-test fixture dirs; no shared image cache

## Change

- Emit: serialized effective prompt template id + resolved params only. honkbox has no ComfyUI graph; emitting a fake graph is dishonest provenance and out of scope. Premise note (2026-11-25): the PNG text-chunk emit seam already exists — `crates/server/src/image_metadata.rs` embeds iTXt chunks without re-encoding pixels (`embed_png` `:175-246`, `png_text` `:248-254`, `RECORD_KEYWORD` `:19`, landed in `80a5836`) — so the `prompt`/`workflow` chunks here extend that seam instead of adding one.
- Ingest: parse `prompt`/`workflow`/`parameters` tEXt keywords from uploads into the job record (debug surface, not execution input); reuse the same chunk scanner rather than a second PNG parser.

## Acceptance detail

- A ComfyUI PNG round-trips its recipe text through honkbox image-ops into the job record; honkbox PNGs opened in ComfyUI show the recorded prompt text.

## Verification Notes (2026-11-25)

- Emit: `png_prompt` and `png_workflow` derive text only from the canonical record; `workflow` is an honest `honkbox.recipe/1` JSON record with no fabricated ComfyUI nodes. Chunks are deterministic, NUL-free, bounded, and ordered record → parameters → prompt → workflow → XMP.
- Ingest: `png_source_annotations` is one bounded PNG tEXt scanner; image-ops copies its `source_prompt`/`source_workflow`/`source_parameters` annotations into `ops.json` before the worker runs. They are debug provenance, never execution input.
- Gate run this turn: `cargo fmt --all -- --check` clean. Product tests intentionally NOT run per operator instruction; run `cargo test --locked -p honkbox-server --bin honkbox image_metadata` before final promotion.
