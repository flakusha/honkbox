# TASK: Decide EXIF/XMP preserve-or-strip on transform

Status: proposed
Priority: P3
Epic: epic-output-formats
Owner: unassigned
Acceptance: each image-op documents and implements one policy: strip-by-default with opt-in preserve, or preserve-always; GPS-bearing EXIF is never forwarded accidentally.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/image-metadata-provenance-2026-11-25.md` (H-IMG-META-4)
Resources: per-test fixture dirs; no shared image cache

## Change

- Current `decode → transform → encode` strips everything including ICC; make that deliberate per-op instead of accidental.
- Privacy default: stripping location-bearing EXIF is a feature; preserving must be an explicit caller opt-in.

## Acceptance detail

- Policy recorded in the ticket and reflected in the sidecar/job record; no silent behavior change for existing callers without a major reason.
