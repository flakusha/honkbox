# TASK: Honor EXIF orientation on image inputs

Status: In Progress
**Scope:** image-ops EXIF orientation honor
Priority: P1
Epic: epic-output-formats
Owner: unassigned
Acceptance: phone-origin JPEG/WebP inputs with EXIF orientation != 1 enter image-ops and generate control-image paths already rotated to match what the viewer shows.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/image-metadata-provenance-2026-11-25.md` (H-IMG-META-1)
Resources: per-test fixture dirs; no shared image cache

## Change

- Read `orientation()` at the two decode call sites (`decode_admitted_input` in `crates/server/src/routes/image_ops_exec.rs`, generate-path control-image decode in `crates/server/src/routes/images_generate.rs`) and apply the EXIF transform before admission.
- `image@0.25` exposes orientation but never auto-applies; no new crate needed.

## Acceptance detail

- A six-orientation EXIF fixture set decodes to identical pixels (orientation normalized); fixture set does not exist yet and ships with this ticket.
