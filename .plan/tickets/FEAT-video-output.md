# FEAT: Video output (vendor-blocked)

Status: postponed
Priority: P3
Epic: epic-output-formats
Owner: unassigned
Acceptance: not committed for v1 — this ticket records the blocker and the seam.
Verification: none until unblocked
Links: `.tmp/research/ReviewedOutput-Formats.md` §3, §5
Resources: none (research-only until unblocked)

## Blocker

The C library has `generate_video()` / `sd_vid_gen_params_t` and raw frame/audio
results, but `SD_WEBP=OFF` and `SD_WEBM=OFF` in the vendor build, and the vendored
Rust `api.rs` exposes only `generate_image`. Bindings are generated at build time
and no safe wrapper exists. Effort L; needs a `vendor_patches/` change.

## Unblock sequence

1. Patch vendor `Config` fields + `gen_video` wrapper.
2. Add `sd-core` video types behind a feature flag.
3. Choose muxing: re-enable WebM/WebP and link libwebm, or v1 PNG-frame sequence
   in the existing job bundle.
4. Add route + DTOs + registry job kind, then re-open this ticket.
