# TASK: Cheap pixel-ops spike

Status: done (2026-11-19 — landed with the image-ops route; ticket closed as superseded)
Priority: P1
Epic: epic-editor-integration
Owner: image-ops route lane (landed)
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

## Implementation (landed with the image-ops route — board review 2026-11-19)

Every clause of this ticket is satisfied by the landed `POST /v1/image-ops`
route; the ticket was still `proposed` (status drift), so it is closed here.

- The op set: `ImageOpKind::ALL` = `resize`, `crop`, `rotate`, `grayscale`
  (color-convert), `format` (format-convert) — the closed vocabulary is
  advertised in `/v1/capabilities` (`crates/api-types/src/lib.rs`).
- In-process, permissive crate: the workspace `image` crate only — no
  libvips/GEGL linkage, no new heavy dependency.
- Caps, reject-not-clamp: edge range `[1, 8192]` per side and
  `width * height <= 16_777_216`, `IMAGE_OP_MAX_INPUTS` per batch, the
  router body cap on total bytes.
- Deterministic output: `identical_commands_produce_identical_bytes` (fixed
  Lanczos3 resample + fixed encoders, no timestamps), plus the SVG/ORA
  rasterize determinism tests.
- Sidecar: `ops.json` records the op name and the tool
  (`honkbox-image-ops` / crate version / `image-rs` engine).
