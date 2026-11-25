# TASK: TAESD tiny decoder as first non-Stub VAE alternative

Status: proposed
Priority: P3
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: TAESD lands as an alternate `VaeDecode` weight-shape with its own golden, behind a flag, as the smallest per-family VAE alternative.
Verification: `cargo test --locked -p honkbox-ash-backend`
Links: `.tmp/research/sdcpp-upstream-patterns-2026-11-25.md` §6.4, `TASK-ash-vae-family-adapters.md`, `crates/ash-backend/src/vae_decode.rs`
Resources: synthetic fixtures only; no GPU, no checkpoint, no download

## Attribution

TRACK-listing from the sd.cpp upstream survey (`leejet/stable-diffusion.cpp`, MIT — tiny per-family VAE alternative). Method: white-room. Weight-shape and tests are honkbox-original against the existing `VaeDecode` seam; no upstream code copied. Related family VAE specs stay owned by `TASK-ash-vae-family-adapters.md`.

## Change

- Add TAESD as an alternate `VaeDecode` weight-shape with its own golden, behind a flag.
- Keep `VaeReal` unchanged; cross-shape misuse fails before dispatch.
- Quality tradeoff is documented, not benchmarked here.

## Acceptance criteria

- [ ] TAESD synthetic decode matches its golden; SD1.5 path unchanged.
- [ ] Flag off means zero behavior change.
- [ ] No GPU, checkpoint, download, or bundled weights in any test.
