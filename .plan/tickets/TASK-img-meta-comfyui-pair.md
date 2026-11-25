# TASK: ComfyUI prompt/workflow metadata pair

Status: proposed
Priority: P2
Epic: epic-output-formats
Owner: unassigned
Acceptance: honkbox PNGs carry ComfyUI-readable `prompt`/`workflow` tEXt chunks scoped to what honkbox actually knows; ComfyUI-produced PNGs uploaded to honkbox surface their embedded `prompt`/`workflow`/`parameters` in the job record for debugging.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/image-metadata-provenance-2026-11-25.md` (H-IMG-META-3)
Resources: per-test fixture dirs; no shared image cache

## Change

- Emit: serialized effective prompt template id + resolved params only. honkbox has no ComfyUI graph; emitting a fake graph is dishonest provenance and out of scope.
- Ingest: parse `prompt`/`workflow`/`parameters` tEXt keywords from uploads into the job record (debug surface, not execution input).

## Acceptance detail

- A ComfyUI PNG round-trips its recipe text through honkbox image-ops into the job record; honkbox PNGs opened in ComfyUI show the recorded prompt text.
