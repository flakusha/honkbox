# TASK: Binary format registry

Status: done
Priority: P1
Epic: epic-binary-formats
Owner: unassigned
Acceptance: one registry maps canonical format names to MIME, magic bytes, read/write support, and byte/size limits for images, video frames, SVG/ORA, GLB/splats, and opaque blobs.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`
Links: `.tmp/research/ReviewedBinaryFormats.md` §Format inventory
Resources: none (registry table); fixtures generated in runner tmp

## Done (2026-11-18, commit 6e1aeb2)

`crates/server/src/formats.rs` is the single per-format source of truth:
`FormatKind` (Image/VideoFrame/Svg/Ora/Glb/Splat/Blob), MIME, advisory extensions,
magic-byte signatures, `FormatBudget` (decoded pixels / container entries /
uncompressed bytes — enforcement wiring is TASK-container-safety's), and per-surface
capability bools (`sd_image_ingest`, `image_ops_input`, `output`,
`output_compression`). 11 rows: png/jpeg/webp live; ppm/pgm are the video-frame
placeholders; svg/ora/glb/ply/splat/blob are inert extension points (flip bools +
teach a validator — no new source of truth). Compile-time `image` codec probes make
a workspace feature flip a compile error in the registry. Wired: capabilities
`formats.read`/`formats.write`, edits image/mask + ControlNet control_image sniff,
image-ops container authority (declared-MIME disagreement = named
`format_mismatch`), `output_format` validation (all three sites), decoder limits.

## Acceptance detail — verified

Two formats with the same extension but different magic bytes are distinguished;
unknown formats are rejected before admission.

- Same-extension/different-magic: `formats::tests::same_extension_different_magic_distinguished`
  (ppm/pgm both carry the umbrella `pnm` extension; `P6`/`P5` sniff distinguishes).
- Unknown rejected before admission: `formats::tests::unknown_bytes_are_rejected_before_admission`
  (structured `FormatReject::Unknown`) + route-level `images::unknown_output_format_is_400`,
  `edits::non_png_image_is_400` (magic over declared content type/filename).
- Gate evidence (2026-11-18): `bun run lint:rs` exit 0; `bun run test:server` exit 0 —
  22 suites, 0 failed (incl. `formats_agree_with_compiled_encoders`,
  `format_registry_matches_the_compiled_encoders`);
  `cargo test --locked -p honkbox-sd-core` 16+4 passed. Constraint for dependents:
  formats.rs stays sd-core-free (link-safety; see module docs +
  `.tmp/intercom/LANE-format-registry.md`).

## Change

- Define `FormatKind` + `FormatInfo` in `crates/api-types` or server internal
  module; keep generated DTOs in sync.
- Validate magic bytes over extensions; named errors for mismatch/unsupported.
- Add limits for decoded pixels, container entries, and uncompressed bytes.

## Acceptance detail

Two formats with the same extension but different magic bytes are distinguished;
unknown formats are rejected before admission.
