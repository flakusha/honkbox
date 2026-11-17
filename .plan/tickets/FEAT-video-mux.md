# FEAT: WebP/WebM muxing for video

Status: proposed
Priority: P3
Epic: epic-video-support
Owner: unassigned
Acceptance: a completed frame sequence can be offered as WebP/WebM via a vendor build-flag and wrapper change, with license gates green.
Verification: `bun run deny`, `bun run spdx`, `bun run test:server`
Links: `.tmp/research/ReviewedSupport-Video.md` rollout step 5; `vendor.manifest.toml:68-76`
Resources: per-test mux output dirs; no shared vendor build artifacts

## Change

- Depends on `FEAT-video-i2v-frames.md`.
- Flip `SD_WEBP=ON` and `SD_WEBM=ON` in `vendor_patches/`; WebM requires WebP
  (upstream CMake `:199-200`), and `libwebp`/`libwebm` are already pinned.
- Wire libwebm per the manifest; register MIME types in the route layer only
  after the wrapper exists.

## Acceptance detail

Container bytes decode in a standard player; license/advisory gates stay green;
no new model download enters CI.
