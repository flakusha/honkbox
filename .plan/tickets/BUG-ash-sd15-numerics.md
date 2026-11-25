# BUG: Ash SD1.5 timestep and residual numerics

Status: done (2026-11-25; c9a0098, 77f5c81)
Priority: P0
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: Ash uses the canonical half-split sinusoidal timestep layout and applies SiLU to the timestep embedding before every ResBlock `emb_layers.1` projection; CPU-reference tests fail on the current code and pass after the fix.
Verification: `cargo test --locked -p honkbox-ash-backend --features ash-sd sinusoidal`, `cargo test --locked -p honkbox-ash-backend --features ash-sd resnet_timestep`, `cargo test --locked -p honkbox-ash-backend --features ash-sd`
Links: `.tmp/research/ash-promotion-refresh-2026-11-25.md` §2.1, `crates/ash-backend/src/timestep.rs`, `crates/ash-backend/src/unet_real.rs`, `vendor/diffusion-rs/sys/stable-diffusion.cpp/ggml/src/ggml-cpu/ops.cpp`, `vendor/diffusion-rs/sys/stable-diffusion.cpp/src/model/common/block.hpp`, `TASK-ash-backend-promote.md`
Resources: deterministic CPU vectors and existing synthetic Ash tests; no checkpoint, model download, shared GPU state or new runtime.

## Verified defects (2026-11-25)

- **[FACT]** `crates/ash-backend/src/timestep.rs:46-47` interleaves cosine and sine values. The vendored ggml reference writes all cosines to the first half and all sines to the second half (`ggml-cpu/ops.cpp:8574-8583`). The existing test only checks bounds, so it cannot detect the layout mismatch.
- **[FACT]** `crates/ash-backend/src/unet_real.rs:476-480` projects the timestep embedding and adds its bias without SiLU. The vendored ResBlock applies `ggml_silu` before `emb_layers.1` (`src/model/common/block.hpp:147-152`).
- **[FACT]** The VAE scale is not a defect: `vae_real.rs:25,336` multiplies by `1 / 0.18215`, equivalent to the vendored decoder division by `scale_factor = 0.18215` (`auto_encoder_kl.hpp:679,820-827`). Do not change it as part of this ticket.

## Change

- Change `sinusoidal` to the canonical `[cos(freq_0)..cos(freq_half), sin(freq_0)..sin(freq_half)]` layout, including the odd-width zero tail.
- Apply the shared `Silu` pipeline to the ResBlock timestep vector before the `emb_layers.1` matmul and bias.
- Add CPU-reference assertions with distinct frequency/slot values so an interleaved implementation and a missing activation both fail.
- Keep the reduced topology and all unrelated VAE math unchanged.

## Acceptance detail

- [x] `sinusoidal_matches_ggml_half_split` compares exact slot placement against an independent CPU reference.
- [x] `resnet_timestep_projection_applies_silu` compares the projected embedding against `SiLU(time)` followed by the stored linear/bias.
- [x] Existing Ash SD1.5 tests stay green; no public capability or route changes.
- [x] `TASK-ash-backend-promote.md` treats this bug as a promotion blocker.

## Non-goals

Full SD1.5 parity, SD2/SDXL generation, VAE rescaling, performance work, model downloads or HTTP exposure.
