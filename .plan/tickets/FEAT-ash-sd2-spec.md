# FEAT: Specify SD2-base and prove one Ash step

Status: done (2026-11-26; three focused gates pass, LLVM 22 + /tmp/cargo-honkbox-ash)
Priority: P1
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: a source-cited SD2-base spec names the exact original-single-file tensors/geometry, SD2 header tests distinguish SD1.5/SD2/SDXL, and a tiny F32 fixture proves one predictor call plus one DDIM step before any full-generation claim.
Verification: `cargo test --locked -p honkbox-ash-backend --features ash-sd sd2`, `cargo test --locked -p honkbox-ash-backend --features ash-sd unet_predictor`, `cargo test --locked -p honkbox-ash-backend --features ash-sd ddim_sampler`.
Links: `.tmp/research/ash-promotion-refresh-2026-11-25.md` §3.3; `crates/ash-backend/src/sd.rs:139-176`; `crates/ash-backend/src/model_spec.rs`; https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/stable_diffusion/stable_diffusion_2.md.
Resources: generated tiny F32 safetensors fixture and unique temp directory; CPU reference first; ignored GPU test on one exclusive Vulkan/lavapipe device; optional real checkpoint is read-only and never downloaded or bundled.

## Verified 2026-11-25

- **[FACT]** SD2 detection and `SD2_SINGLE_LEVEL` header/spec acceptance already
  landed. `checkpoint_family::detect` distinguishes SD1.5, SD2 and SDXL;
  `validate`/`infer` check the declared SD2 tensors. Do not re-file basic SD2
  header recognition.
- **[FACT]** `AshSdPipeline::generate` still rejects every non-SD1.5 family
  (`crates/ash-backend/src/sd.rs:316-325`). SD1.5 CLIP geometry and the reduced
  real-weight UNet/VAE are not SD2 implementations.
- **[FACT]** Reusable seams exist: `UnetModelSpec`/`LevelSpec`/`tensor_keys`,
  `UnetPredictor`, `DdimSampler`, F32/F16/BF16 checkpoint decode and
  CPU-reference op tests.
- **[UNKNOWN]** Exact tensor names/geometry for the eventual user checkpoint
  must be confirmed from that artifact; this ticket does not guess aliases or
  claim checkpoint parity.

## Verified 2026-11-26

- **[FACT]** At slice start `CheckpointFamily` had no `Sd2` variant; the earlier
  "SD2 detection landed" note was stale relative to current source. Header-only
  detection now needs the UNet `[*,4,3,3]` input plus a `[77,1024]` CLIP position
  embedding, so SD1.5 `[77,768]`, SDXL markers and bad geometry stay distinct.
- **[FACT]** SD2-base geometry is source-cited from the original `v2-inference.yaml`
  (`model_channels: 320`, `channel_mult: [1,2,4,4]`, `context_dim: 1024`, 4 latent
  channels, 1000 steps) and the diffusers text-encoder config (`hidden_size: 1024`,
  `num_hidden_layers: 23`, `num_attention_heads: 16`). SD2-base predicts epsilon;
  `2-v` v-prediction is out of scope.

## Dependencies

- `FEAT-ash-vulkan-backend` for the shared checkpoint/spec/op primitives.
- No dependency on full SD2 generation, HTTP selection, readiness markers or
  an upscaler. `TASK-ash-family-readiness-markers` may consume the resulting
  manifest later.

## Change

- Add a named SD2-base spec beside `SD15_SINGLE_LEVEL`, scoped to the original
  single-file safetensors layout. Record exact key names, UNet widths,
  OpenCLIP text-tower geometry, cross-attention context width, scheduler and
  prediction semantics with primary-source citations. Do not silently reuse
  `SD15_CLIP` or SD2.1 v-prediction semantics.
- Retain and extend header tests: SD2 `[320,4,3,3]` input acceptance, SD1.5
  distinction, SDXL eight-latent-channel distinction, wrong latent channels,
  wrong kernel geometry and missing named tensors. Header checks must not read
  tensor payloads.
- Add a tiny deterministic F32 fixture that loads the declared input/spec
  roles, runs one `UnetPredictor::predict`, then one `DdimSampler::run` step.
  Compare each stage with an independent CPU reference. The fixture proves
  wiring/numerics only; it is not a scale model or quality test.

## Acceptance criteria

- [x] `SD2_BASE_REDUCED` is source-cited and contains no guessed SD1/SDXL aliases;
  unsupported layout/dtype errors name the tensor and expected shape.
- [x] Header tests accept valid SD2-base geometry and reject wrong latent
  channels, kernel geometry, missing required keys and SDXL input geometry.
- [x] The synthetic one-step fixture matches the CPU predictor/DDIM references
  within the existing f32 op tolerances and uses no bundled model weights.
- [x] An ignored GPU test can run the same tiny fixture on one selected Vulkan
  device; absence of a device skips explicitly rather than fabricating output.
- [x] `AshSdPipeline::generate` still rejects SD2 by name (`sd.rs:379-386`). No
  full-generation, catalog, HTTP-selection or family-availability claim is added.

## Verification

```sh
cargo test --locked -p honkbox-ash-backend --features ash-sd sd2
cargo test --locked -p honkbox-ash-backend --features ash-sd unet_predictor
cargo test --locked -p honkbox-ash-backend --features ash-sd ddim_sampler
```

If an opt-in real-checkpoint header audit is added, give it a distinct ignored
test and an explicit `HONKBOX_ASH_SD2_MODEL` path; record the artifact SHA-256
and license in the test output. Do not make normal tests download weights.

## Non-goals

Full SD2 sampling/text encoding/VAE decode; SD2.1, inpainting or depth variants;
SDXL/FLUX/DiT support; real-checkpoint image-quality parity; automatic model
selection or catalog registration; performance/VRAM claims.

## Verification evidence (2026-11-26)

Build env: LLVM 22 (`PATH`/`LIBCLANG_PATH`), `CARGO_TARGET_DIR=/tmp/cargo-honkbox-ash`,
`RUSTC_WRAPPER=sccache`, `SCCACHE_DIR=/tmp/sccache`, `CCACHE_DIR=/tmp/ccache`.

- `cargo fmt --all -- --check`: clean.
- `cargo test --locked -p honkbox-ash-backend --features ash-sd sd2`: 4 passed, 1 ignored.
- `cargo test --locked -p honkbox-ash-backend --features ash-sd unet_predictor`: 1 passed.
- `cargo test --locked -p honkbox-ash-backend --features ash-sd ddim_sampler`: 2 passed.
- `giwt plan check-links`: 291 files, 0 broken links.
