# TASK: Cheap pixel-ops spike

Status: proposed
Priority: P1
Epic: epic-editor-integration
Owner: unassigned
Acceptance: resize/crop/rotate/color-convert/format-convert batch ops run in-process with permissive Rust crates and deterministic output.
Verification: `cargo test --locked -p honkbox-server`, `bun run lint:rs`, `bun run deny`
Links: `.tmp/research/ReviewedImageEdit-Operations.md` §Tier 0
Resources: per-test image fixtures; bounded thread pool, no GPU lease

## Change

- Implement the first op set behind the image-ops route; no new heavy dep.
- Cap edge, batch size, and total bytes; reject-not-clamp on every parameter.
- Record op name and software version in the sidecar.

## Acceptance detail

No LGPL linkage (libvips/GEGL stay external), no unbounded base64 body, and
identical input yields identical bytes.
