# TASK: Video FFI support spike (current pin)

Status: proposed
Priority: P2
Epic: epic-video-support
Owner: unassigned
Acceptance: a support probe and one short low-resolution clip run from a license-cleared local H3 or LTX checkpoint on stable-diffusion.cpp `5dcfa79`; no HTTP route, DTO, job kind or capability change.
Verification: `cargo test --locked -p honkbox-sd-core --features video video_ffi_spike -- --ignored --nocapture`, `bun run test:core`, `bun run deny`
Links: `.tmp/research/native-video-backend-refresh-2026-11-25.md`; upstream stable-diffusion.cpp docs at `5dcfa79801213f23e27b3e83617350ebb8ea93aa`: https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/5dcfa79801213f23e27b3e83617350ebb8ea93aa/docs/models/minimax_h3.md and https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/5dcfa79801213f23e27b3e83617350ebb8ea93aa/docs/models/ltx2.md
Resources: one operator-supplied local component set; exclusive GPU lease; per-run output directory; no CI downloads or weight bundling

## Verified starting point

The current pin already declares `sd_ctx_supports_video_generation`,
`sd_vid_gen_params_t`, `sd_vid_gen_params_init`, `generate_video`, and
`free_sd_video_generation_result`; its pipeline contains MiniMax-H3 and LTXAV.
The vendored high-level Rust API imports only `generate_image`, and `sd-core`
has no video wrapper. Do not bump the vendor or implement against upstream
master's newer opaque-options C ABI.

## Change

- Extend the existing `vendor_patches/` + `diffusion-rs::api` pattern with a
  safe, feature-gated video wrapper. Use the build-generated bindings; do not
  duplicate the C struct ABI by hand.
- Initialize context and video params with `sd_ctx_params_init` and
  `sd_vid_gen_params_init`; set only fields supported by the selected family.
- Probe with `sd_ctx_supports_video_generation` and
  `sd_get_model_version_name`. A true support probe is not a generation proof.
- Call `generate_video`, copy frames/optional audio/generation info into owned
  Rust values, and call `free_sd_video_generation_result` plus `free_sd_ctx` on
  every path.
- Expose the probe only behind a `video` feature in `sd-core`; add no route.

## Evidence contract

Record the pin and backend, model family/version/mode, component paths and
sha256 hashes, license artifact and operator gate, seed/steps/sampler, requested
and effective width/height/frame count/fps, output frame count and dimensions,
optional audio sample rate/channels/sample count, generation info, elapsed time,
peak memory when measurable, and any cancellation observation. Preserve ordered
PNG frames plus a `timeline.json`; preserve audio as PCM/WAV when present.

Family rules are explicit: H3 uses 24 fps, 32-pixel spatial alignment and
`17k + 5` frames (minimum 5); LTX uses `8k + 1` frames. LTX-2.5 requires the
LTX-specific Gemma 4 encoder and conv video VAE, not stock Gemma 4 or the
unimplemented diffusion decoder. H3 keyframes and Ref2VA references are
mutually exclusive.

## License gate

Before loading weights, record the exact checkpoint license and operator
acceptance. MiniMax-H3 excludes EU/UK/South Korea/USA and requires authorization
above $20M yearly commercial revenue. LTX weights use their community license,
not the Apache-2.0 code license; LTX-2.x requires paid licensing at the $10M
entity threshold subject to its stated exception. Refuse an uncleared fixture;
never download or bundle weights.

## Acceptance detail

One named family produces non-empty ordered frames at the recorded effective
fps and dimensions from the current pin. Audio may be absent for a video-only
proof; when present, its metadata and duration are checked. Repeated success and
failure paths release all native resources. No HTTP route, job kind, DTO or
capability flag is added.

If the target family fails, leave `FEAT-video-output.md` postponed and attach the
exact failure evidence; do not expand the surface or silently switch families.
