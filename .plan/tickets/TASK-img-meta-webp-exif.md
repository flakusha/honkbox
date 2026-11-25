# TASK: WebP EXIF workflow/prompt write path

Status: done (2026-11-25; decision-only closure, no writer shipped)
Priority: P3
Epic: epic-output-formats
Owner: unassigned
Acceptance: decision recorded: either WebP outputs carry EXIF `workflow`/`prompt` tags like ComfyUI animated WebP, or the ticket closes with a documented reason not to.
Verification: `bun run test:server`
Links: `.tmp/research/image-metadata-provenance-2026-11-25.md` (H-IMG-META-5)
Resources: per-test output dirs; no shared encode cache

## Change

- `image@0.25` already exposes an in-tree EXIF write path (`ImageEncoder::set_exif_metadata`: `jpeg/encoder.rs:757`, `webp/encoder.rs:101-104` via `image-webp`). Spike a round-trip fixture first; new crate or manual VP8X assembly only if the in-tree path proves lossy.
- Parked behind the PNG tickets; file as P3, close as wont-do with rationale if the dep cost exceeds the interop value.

## Acceptance detail

- Either a round-trip test (ComfyUI reads honkbox WebP recipe) or a written wont-do rationale; no half-landed writer.

## Decision (2026-11-25)

- No ComfyUI WebP recipe consumer or round-trip fixture exists in this repo. The only in-tree WebP EXIF write is honkbox's own canonical record (`crates/server/src/image_metadata.rs:525-601`, EXIF chunk assembled at `:580` and appended at `:597`). Adding ComfyUI `workflow`/`prompt` tags would need a new EXIF tag schema plus a fixture that no named consumer reads, so the dep/interop cost exceeds the value.
- PNG outputs carry recipe text through `parameters` and the scoped ComfyUI `prompt`/`workflow` pair; WebP outputs deliberately retain the canonical honkbox record only. Reopen this ticket when a named ComfyUI WebP consumer or fixture proves those tags are actually read.
