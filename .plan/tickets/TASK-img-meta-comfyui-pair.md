# TASK: ComfyUI prompt/workflow metadata pair

Status: In Progress
**Scope:** Emit A1111 parameters and ComfyUI recipe text chunks on PNG outputs
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
