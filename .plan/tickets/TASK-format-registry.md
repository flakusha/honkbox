# TASK: Binary format registry

Status: proposed
Priority: P1
Epic: epic-binary-formats
Owner: unassigned
Acceptance: one registry maps canonical format names to MIME, magic bytes, read/write support, and byte/size limits for images, video frames, SVG/ORA, GLB/splats, and opaque blobs.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`
Links: `.tmp/research/ReviewedBinaryFormats.md` §Format inventory
Resources: none (registry table); fixtures generated in runner tmp

## Change

- Define `FormatKind` + `FormatInfo` in `crates/api-types` or server internal
  module; keep generated DTOs in sync.
- Validate magic bytes over extensions; named errors for mismatch/unsupported.
- Add limits for decoded pixels, container entries, and uncompressed bytes.

## Acceptance detail

Two formats with the same extension but different magic bytes are distinguished;
unknown formats are rejected before admission.
