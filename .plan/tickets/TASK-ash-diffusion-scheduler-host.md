# TASK: Host-math diffusion schedulers (Euler-a, DPM++-2M) with upstream parity

Status: proposed
Priority: P2
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: Euler-a and DPM++-2M step updates exist as pure-host f32 functions next to `DdimStep`, parity-tested against diffusion-rs on fixed latents and predictions with no new kernels.
Verification: `cargo test --locked -p honkbox-ash-backend`
Links: `.tmp/research/sdcpp-upstream-patterns-2026-11-25.md` §6.1, `crates/ash-backend/src/ddim_sampler.rs`, `crates/ash-backend/src/sampler.rs`
Resources: synthetic latents/predictions only; no GPU, no checkpoint, no download

## Attribution

Scheduler split from `leejet/stable-diffusion.cpp` (MIT — schedule math on host in the denoiser/sampler surface, backbone on device). Method: white-room with numeric cross-check. Step-update functions are honkbox-original host math; diffusion-rs outputs serve as expected values in tests (observed behavior comparison, not code copying). MIT license would permit verbatim reuse with notice, but no upstream source text is imported.

## Change

- Add Euler-a and DPM++-2M step updates as pure-host f32 functions next to `DdimStep`.
- Parity-test against diffusion-rs on fixed latents and predictions (CPU-only).
- Scheduler choice becomes a request parameter only after two schedulers agree with upstream to 1e-5. No new kernels.

## Acceptance criteria

- [ ] Both schedulers match diffusion-rs to 1e-5 on fixed fixtures.
- [ ] No Vulkan kernel, checkpoint, download, or new dependency introduced.
- [ ] Single-scheduler behavior is unchanged until the parameter gates open.
