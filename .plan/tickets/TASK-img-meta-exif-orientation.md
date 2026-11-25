# TASK: Honor EXIF orientation on image inputs

Status: in progress (code + fixtures landed on `fix/img-exif-orientation`; bin-test re-run blocked by shared-target stale test-profile server unit — see FIXED intercom note)
**Scope:** image-ops EXIF orientation honor
Priority: P1
Epic: epic-output-formats
Owner: unassigned
Acceptance: phone-origin JPEG/WebP inputs with EXIF orientation != 1 enter image-ops and generate control-image paths already rotated to match what the viewer shows.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/image-metadata-provenance-2026-11-25.md` (H-IMG-META-1)
Resources: per-test fixture dirs; no shared image cache

## Change (landed 2026-11-25, lane `fix/img-exif-orientation`)

- `decode_flat_raster_upright` in `crates/server/src/routes/image_ops_exec.rs` decodes with `ImageReader::into_decoder`, reads `ImageDecoder::orientation()` best-effort, and applies `DynamicImage::apply_orientation`; wired into `decode_admitted_input`'s flat-raster arm.
- Premise corrections (research §1 already struck the second site): there is NO generate-path decode in `images_generate.rs` (zero `ImageReader`/`load_from_memory` hits); control images are PNG-validated (`identify_png` in `images_controlnet.rs`) and PNG carries no EXIF orientation — nothing to do there.
- Locked-version audit (`Cargo.lock`): `image 0.25.10` implements `ImageDecoder::orientation()` for JPEG (`codecs/jpeg/decoder.rs:140`), WebP (`codecs/webp/decoder.rs:95`, EXIF chunk read via `image-webp 0.2.4 decoder.rs:648`), and TIFF (`codecs/tiff.rs:337`); `DynamicImage::apply_orientation` at `images/dynimage.rs:1161`. WebP EXIF orientation IS honored by the locked stack — no WebP gap.
- Admission semantics preserved: header-based pixel cap unchanged (90/270 swaps width/height, pixel count invariant); `OpsInputRecord` hashes remain hashes of SOURCE bytes; records carry no input width/height, so nothing reports stale pre-rotation dims.

## Acceptance detail

- In-tree fixture set (`routes/tests_image_orientation.rs`): 3×2 asymmetric JPEG + spliced APP1 EXIF; orientations 1–8 covered with hand-written pixel expectations (not derived via `apply_orientation`), plus corrupt-EXIF-best-effort and lossless WebP orientation-6 cases.
