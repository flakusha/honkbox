# FEAT: Video output (gated on current-pin spike)

Status: postponed
Priority: P3
Epic: epic-video-support
Owner: unassigned
Acceptance: reopen only after `TASK-video-ffi-spike.md` produces one short local clip from a license-cleared H3 or LTX checkpoint on stable-diffusion.cpp `5dcfa79`; then provide owned canonical frames, sidecar and optional audio without adding a route prematurely.
Verification: none until unblocked; then `cargo test --locked -p honkbox-sd-core --features video`, `bun run test:core`, `bun run deny`
Links: `.tmp/research/native-video-backend-refresh-2026-11-25.md`, `.tmp/research/media-masters-audio-video-2026-11-25.md`, `.tmp/research/media-handling-consolidated-2026-11-24.md`
Resources: after unblock, synthetic tiny frame/audio fixtures plus one opted-in local clip; per-run output directories; no CI downloads

## Blocker

The pinned C library already has `sd_ctx_supports_video_generation`,
`sd_vid_gen_params_t`, `generate_video` and
`free_sd_video_generation_result`; its docs cover MiniMax-H3 and LTX-2.3/2.5.
The vendored high-level Rust API exposes only image generation, `sd-core` has no
owned video result, and no local clip has been proven.

`SD_WEBP=OFF` and `SD_WEBM=OFF` disable vendor file muxers, not raw frame/audio
generation. They do not block the spike or a canonical frame-sequence proof.
Effort remains L and needs a `vendor_patches/` + `sd-core` change; no vendor
upgrade is proposed.

## Unblock sequence

1. Green current-pin FFI spike with exact family/mode/frame/fps/edge evidence,
   component hashes and license gate.
2. Add feature-gated `sd-core` owned video types and cleanup. Keep PNG frames +
   `timeline.json` as the canonical master; keep optional audio as PCM/WAV.
3. Add `TASK-video-job-class.md` with hard caps, exclusive heavy-inference
   lease, bounded retention and measured cancellation behavior.
4. Add `FEAT-video-i2v-frames.md` for typed first/last-frame conditioning.
5. Add `FEAT-video-mux.md` as a derived, capability-gated export. Prefer the
   established supervised FFmpeg path; enable vendor WebP/WebM only if measured
   need justifies native codec linkage.
6. Add route + DTOs + capability advertisement last, after the output and job
   contracts are tested.
