# TASK: WebP EXIF workflow/prompt write path

Status: proposed
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
