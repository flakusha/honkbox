# TASK: Video FFI support spike

Status: proposed
Priority: P2
Epic: epic-video-support
Owner: unassigned
Acceptance: a support probe and one short low-resolution clip run from a locally supplied video checkpoint; no HTTP route is added.
Verification: `bun run test:core`, `cargo test --locked -p honkbox-sd-core`, `bun run deny`
Links: `.tmp/research/ReviewedSupport-Video.md` §2, rollout step 2
Resources: opt-in local checkpoint fixture; per-test clip output dirs; no CI downloads

## Change

- In `vendor_patches/`, wrap `sd_ctx_supports_video_generation`,
  `sd_vid_gen_params_init`, `generate_video`, and
  `free_sd_video_generation_result` (C header `:544,463-496,613-616,600`).
- Expose a support probe in `sd-core` behind a build flag; do not add a route.
- Record exact model family, frame count, fps, and edge used in the probe result.

## Acceptance detail

If the spike fails for the target family, close `FEAT-video-output.md` as
blocked rather than expanding the surface; no route or job kind is created by
this ticket.
