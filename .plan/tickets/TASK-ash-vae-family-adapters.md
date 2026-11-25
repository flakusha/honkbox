# TASK: Ash VAE family adapters

Status: done (2026-11-25; 238736f)
Priority: P2
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: explicit MOSS/FLUX.2-small and Qwen-Image-2.1 VAE specs, key maps and normalization contracts pass synthetic encode/decode and pack/unpack tests without inheriting SD1.5 channels or scale; no family execution or real checkpoint is required.
Verification: `cargo test --locked -p honkbox-ash-backend --features ash-sd --lib vae_family_adapters -- --test-threads=1`, `HONKBOX_ASH_REQUIRE_VULKAN=1 cargo test --locked -p honkbox-ash-backend --features ash-sd --test vae_family_adapters_parity -- --test-threads=1 --nocapture`, `bun run spdx`. The ignored real-artifact command is optional and listed under Verification.
Links: `.tmp/research/ash-family-implementation-index-2026-11-25.md` §4 S8 and §6; `.tmp/research/ash-moss-image-implementation-candidate-2026-11-25.md` §§2,4–6; `.tmp/research/ash-qwen-image21-implementation-candidate-2026-11-25.md` §§2,4–6; `crates/ash-backend/src/vae_real.rs`; `crates/ash-backend/src/checkpoint.rs`; `vendor/diffusion-rs/sys/stable-diffusion.cpp/src/model/vae/auto_encoder_kl.hpp`; `vendor/diffusion-rs/sys/stable-diffusion.cpp/src/model/vae/wan_vae.hpp`; `vendor/diffusion-rs/sys/stable-diffusion.cpp/src/name_conversion.cpp`; `TASK-ash-flow-dit-substrate.md`.
Resources: deterministic tiny synthetic tensors and sparse safetensors fixtures in unique temp dirs; one test-only RAII lease keyed to the selected physical device for Vulkan tests; optional read-only user checkpoint with recorded SHA-256 and license under the same exclusive lease; no download, cache mutation or bundled weights.

## Verified 2026-11-25

- **[FACT]** `VaeReal` is a reduced SD1.5 decoder only: four latent channels, scale `1/0.18215`, square planes, 8× output and three image channels; mid attention and spare up blocks are omitted (`crates/ash-backend/src/vae_real.rs:24-29,108-163,314-375`). It has no family adapter or encoder contract.
- **[FACT]** The pinned MIT reference gives FLUX.2 VAE 32 base latent channels, 2×2 packing to 128 diffusion channels and effective 16× image compression (`vendor/diffusion-rs/sys/stable-diffusion.cpp/src/model/vae/auto_encoder_kl.hpp:540-660`). Its normalization uses 128 per-channel means/stds with scale 1 and shift 0 (`:665-693,773-836`).
- **[FACT]** The same reference gives Qwen-Image-2.1 a 64-channel latent, four-channel RGBA output, 16× compression and its own 64 mean/std values (`vendor/diffusion-rs/sys/stable-diffusion.cpp/src/model/vae/wan_vae.hpp:1077-1123,1388-1428`). It is not the older 16-channel Qwen-Image/Krea VAE.
- **[FACT]** Pinned Qwen key conversion explicitly maps Diffusers `down_blocks`/`up_blocks`, resnets, shortcuts, samplers, quant convolutions, mid blocks and output norms/convolutions (`vendor/diffusion-rs/sys/stable-diffusion.cpp/src/name_conversion.cpp:1253-1360`).
- **[FACT]** `CheckpointMap` strictly validates safetensors headers, shapes, offsets and payload bounds; its generic alias fallback is SD1.5-oriented (`crates/ash-backend/src/checkpoint.rs:90-258`). Family aliases must therefore be explicit, not guessed.
- **[FACT]** Index S8 requires an explicit spec/key map per family and prohibits silent SD1.5 scale/channel reuse. MOSS and Qwen candidates consume this seam.
- **[FACT]** S8 completion adds `crates/ash-backend/src/vae_family_adapters.rs`: family-specific two-block geometry, synthetic encode/decode, named missing-block/shape guards, CPU/GPU parity tests and an opt-in real-artifact header evidence test. License terms are operator-supplied; no gate work.

## Dependencies

- `TASK-ash-flow-dit-substrate.md` is the sequencing dependency for shared image-latent conventions and synthetic proof discipline; this ticket does not execute a DiT.
- `TASK-ash-native-families-index.md` records this ticket as S8.
- Existing `Conv2d`, `ConvTranspose2d`, `GroupNorm`, `Silu`, attention, matmul, add and `CheckpointMap` remain shared implementation seams.
- `FEAT-ash-moss-image-family.md` and `TASK-ash-qwen-image21-attention-proof.md` are consumers, not dependencies. Their admission and execution tickets remain separate.

## Change

- Add a shared `vae` contract whose spec explicitly names latent channels, diffusion channels, spatial scale, image channels/order, alpha behavior, normalization equation/constants, supported direction and an exact required-key map. Provide no SD1.5 default or fallback.
- Add `Flux2SmallVaeSpec` for the MOSS candidate: 32 VAE latent channels, 128 packed diffusion channels, 2×2 packing, 16× scale, RGB output and the pinned 128 mean/std constants. Add `QwenImage21VaeSpec`: 64 latent channels, no 2×2 diffusion pack, 16× scale, RGBA output and the pinned 64 mean/std constants.
- Implement table-driven key maps for the declared layouts. FLUX.2 uses explicit encoder/decoder, quant, resnet, attention, sampler, norm and output keys. Qwen uses the pinned Diffusers-to-original mappings, including `down_blocks`/`up_blocks`, residual/shortcut, sampler, `quant_conv`/`post_quant_conv`, mid-block and output mappings. Reject missing, duplicate, unexpected or shape-mismatched required tensors by name.
- Implement normalization as data from the selected spec. For both declared specs, diffusion-to-VAE is `z * std / scale + mean` and VAE-to-diffusion is `(z - mean) * scale / std`; tests must prove the selected family vectors are used and that SD1.5 `0.18215` is never substituted.
- Implement FLUX.2 pack/unpack with a hand-computed channel/pixel-order golden fixture and exact divisibility checks. Implement Qwen latent pass-through explicitly and reject accidental 128-channel packing.
- Build tiny deterministic encode/decode fixtures with the declared stage topology but reduced widths. Compare one resnet/attention block with an independent CPU reference, then prove full synthetic encode/decode shapes, rectangular inputs, batch preservation and finite outputs. Required fixture keys cannot be skipped to make a test pass.
- Keep `VaeReal` unchanged for SD1.5. New constructors require a family spec; cross-family 4/16/32/64/128-channel misuse must fail before dispatch.
- Add an ignored optional real-artifact test. It requires family, path, SHA-256 and license, verifies the digest before payload/device work, records all four values, acquires the exclusive single-device lease and runs one encode/decode shape check. It is evidence only and is not an acceptance prerequisite.
- Register any new shader in `build.rs` with required notice and first-line provenance. Reuse existing kernels where they satisfy the explicit topology.

## Acceptance criteria

- [ ] Synthetic specs and key-map fixtures prove both families independently, including exact required-key rejection and no generic SD1.5 alias fallback.
- [ ] FLUX.2 tests prove `[1,32,H,W] ↔ [1,128,H/2,W/2]`, effective 16× image shape and exact 2×2 channel/pixel ordering; non-divisible dimensions fail.
- [ ] Qwen tests prove `[1,64,H,W] ↔ [1,4,16H,16W]`, non-square shapes, real alpha preservation and rejection of 2×2 packing.
- [ ] Normalization golden tests use each family's complete mean/std vectors, prove encode/decode inverses within `1e-4`, and fail if SD1.5 scale or another family's constants are selected.
- [ ] Tiny synthetic CPU/Vulkan block parity stays within absolute error `1e-4`; required-Vulkan mode fails rather than silently skipping.
- [ ] Synthetic encode/decode, missing-key, wrong-shape, wrong-family and omitted-block tests pass without a real checkpoint.
- [ ] If run, the ignored real test verifies SHA-256/license and holds one exclusive device lease; its absence does not block ticket completion.
- [ ] No result claims MOSS/Qwen execution, full-family availability, image quality, latency or VRAM fit.

## Verification

```sh
cargo test --locked -p honkbox-ash-backend --features ash-sd \
  --lib vae_family_adapters -- --test-threads=1
HONKBOX_ASH_REQUIRE_VULKAN=1 cargo test --locked \
  -p honkbox-ash-backend --features ash-sd \
  --test vae_family_adapters_parity -- --test-threads=1 --nocapture
HONKBOX_ASH_VAE_FAMILY=flux2-small \
HONKBOX_ASH_VAE_MODEL=/read-only/vae.safetensors \
HONKBOX_ASH_VAE_MODEL_SHA256=<recorded-sha256> \
HONKBOX_ASH_VAE_LICENSE=<recorded-license> \
cargo test --locked -p honkbox-ash-backend --features ash-sd \
  --test vae_family_real -- --ignored --test-threads=1 --nocapture
bun run spdx
```

Repeat the ignored command with `HONKBOX_ASH_VAE_FAMILY=qwen-image21` and that family's read-only artifact. Run real checks alone on an idle host. No command was executed by the filing pass.

## Unknowns and stop conditions

- **[UNKNOWN]** Exact production tensor names, dtypes, shards and omitted-block policy must be revalidated against user-supplied artifacts. Stop on metadata/config disagreement; never guess aliases or silently drop a required block.
- **[UNKNOWN]** Component license terms are independent of the DiT family. Real-artifact checks stop without a recorded license; synthetic acceptance does not imply redistribution permission.
- **[UNKNOWN]** Real-checkpoint numerical tolerance, alpha semantics beyond the pinned reference, quality, performance and one-device memory fit remain unmeasured.
- Stop on normalization drift, pack-order mismatch, missing provenance, incomplete lease coverage or any attempt to reuse SD1.5 constants/channels implicitly.

## Non-goals

MOSS-Image or Qwen-Image generation; DiT/text-encoder execution; HTTP DTOs/routes; admission/availability flags; model downloads, conversion or bundling; tiling, training or video; multi-GPU; full-family quality, speed or memory claims; mandatory real-checkpoint validation for this synthetic substrate ticket.