# FEAT: Ash MOSS-Image single-stream flow DiT

Status: proposed
Priority: P1
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: prove synthetic single-stream block, MOSS Base/SFT manifest, Qwen-conditioned 42-layer flow loop, FLUX.2-small VAE adapter, then opt-in user-supplied Base/SFT gate; the first slice supplies text states directly and loads neither Qwen3.5-9B nor VAE.
Verification: `cargo test --locked -p honkbox-ash-backend --features ash-sd moss_single_block`, `moss_manifest`, `moss_flow_pipeline`, `moss_vae_contract`, `cargo test --locked -p honkbox-ash-backend --features ash-llm,ash-sd moss_conditioning`, opt-in serialized `moss_real` Base/SFT commands and `bun run spdx` in § Verification.
Links: `.tmp/research/ash-moss-image-implementation-candidate-2026-11-25.md`; `.tmp/research/ash-family-implementation-index-2026-11-25.md`; https://github.com/OpenMOSS/MOSS-Image; https://openmoss.ai/MOSS-Image; `vendor/diffusion-rs/sys/stable-diffusion.cpp/src/model/diffusion/moss_image.hpp`.
Resources: per-test unique `.tmp/honkbox-moss-*` directories with deterministic teardown; one exclusive single-device lease for each real-checkpoint run; read-only operator-supplied component manifest with SHA-256 and license per artifact; no downloads, cache mutation, ports, databases, or bundled weights.

## Verified 2026-11-25

- **[FACT]** Current `Op` has 14 variants and no RMSNorm, RoPE, multiply,
  SiTU, AdaLN, or flow-matching step; `ModelSpec` validates static capacity only
  (`crates/ash-backend/src/model.rs:13-106`).
- **[FACT]** Current multi-head attention supports an additive mask but has no
  MOSS 4D position or gate mapping (`crates/ash-backend/src/attention.rs:9-148`).
- **[FACT]** `CheckpointMap` validates safetensors headers and reads payloads on
  demand; it has no MOSS component/key map
  (`crates/ash-backend/src/checkpoint.rs:90-258`).
- **[FACT]** Current tokenizer is CLIP BPE; `AshLlmPipeline` owns a device while
  every verb returns `NotYet` (`crates/ash-backend/src/tokenizer.rs:1-108`;
  `crates/ash-backend/src/llm.rs:29-79`).
- **[FACT]** `VaeReal` is a reduced SD1.5 decoder: four latent channels,
  `1/0.18215` scale, square planes and 8x output. It cannot decode FLUX.2-small
  unchanged (`crates/ash-backend/src/vae_real.rs:24-29,314-376`).
- **[FACT]** Current noise is seeded Gaussian, correcting the candidate note's
  stale uniform-noise description. Current sampling is still DDIM eta=0, not
  MOSS flow Euler; the MOSS noise distribution remains unknown
  (`crates/ash-backend/src/sampler.rs:1-140`).
- **[FACT]** The pinned reference defaults to width 3072, FFN 8192, text width
  4096, head width 128, 128 latent channels and depth 42
  (`vendor/diffusion-rs/sys/stable-diffusion.cpp/src/model/diffusion/moss_image.hpp:16-55`).
  Its SiTU is `4*tanh(gate/4)*sigmoid(gate) * 25*tanh(hidden/25)`, not SwiGLU
  (`src/core/ggml_extend.cpp:120-138`).
- **[FACT]** The reference assigns `{22}` only to the negative branch's skip
  layers for MOSS (`src/pipeline/diffusion_engine.cpp:2842-2847`). Research
  records shift 5, 50 steps and CFG 3 defaults.
- **[FACT]** The pinned stable-diffusion.cpp reference is MIT
  (`vendor/diffusion-rs/sys/stable-diffusion.cpp/LICENSE:1-21`). Current catalog
  marks `flux2-dev-vae` non-commercial and gated
  (`crates/server/src/catalog.json:309-323`); MOSS Apache-2.0 does not transfer to
  that component.
- **[UNKNOWN]** Exact Base/SFT tensor names, shapes, dtypes, shards and runtime
  delta; official tokenizer identity; shared Qwen prefill parity; FLUX.2
  normalization constants; concrete encoder/VAE license terms; MOSS noise
  convention; one-device fit; numerical tolerance, quality, speed and memory.
  Stop rather than guess aliases, drop conditioning, inherit terms, or claim
  readiness from model headers.
- **[FACT]** This filing pass read the cited sources and ran no build, test,
  lint, formatter, download, model execution, or git command.

## Dependencies

- [TASK-ash-flow-dit-substrate](TASK-ash-flow-dit-substrate.md): shared RMS/L2
  norm, RoPE axes, multiply, attention gates, AdaLN, SiTU and flow-step math.
- [TASK-ash-vae-family-adapters](TASK-ash-vae-family-adapters.md): shared VAE
  specification, pack/unpack and normalization seam, without SD1.5 reuse.
- [TASK-ash-tokenizer-framework](TASK-ash-tokenizer-framework.md): official
  tokenizer binding and strict length/template behavior.
- [FEAT-ash-qwen35-dense](FEAT-ash-qwen35-dense.md): the shared
  `tokens -> [T,4096]` Qwen3.5 prefill backend, masks, state and final RMSNorm.
- Existing correctness/promotion gates, safetensors reader, registry/license
  admission, one-device byte planner, `SdPipeline`, final-prediction CFG and
  cancellation seams. Do not create parallel implementations.
- A user-supplied Base component set for the first real gate and a separately
  recorded SFT set for the second; no dependency on GGUF conversion.

## Change

1. **Synthetic single-stream proof first.** Add the smallest DiT substrate and
   CPU reference for `D=64`, four heads of 16, FFN 128, four image sites and
   three caller-supplied text-state rows. Supply latents and text states
   directly; assert the Qwen encoder and VAE loaders are never called. Build
   half-cos/half-sin time features, SiLU time MLP, six shared AdaLN vectors
   (`scale/shift/gate` for attention and FFN), two text-refiner blocks, one
   joint block, final RMSNorm/head modulation/linear and two Euler steps.
   Preserve centered row/column plus text-offset 4D RoPE, Q/K RMSNorm,
   gated/scaled residuals and exact SiTU. Use a test-only skip at fixture index
   0; separately pin production policy to negative-only zero-based layer 22.
2. **MOSS Base/SFT manifest.** Add an internal component manifest over existing
   admission/checkpoint structures, not a new HTTP DTO. Infer width, depth,
   text width and latent geometry from synthetic sparse safetensors; validate
   DiT, Qwen encoder, FLUX.2-small VAE and official tokenizer independently and
   bind their paths, SHA-256 values and licenses. Use one Base/SFT runtime
   mapping until tensor-level evidence proves a delta. Missing, unexpected,
   wrong-shape, unstable-digest or unresolved-license components fail closed
   before device selection or payload work. Headers never imply runnable.
3. **Shared Qwen-conditioned path.** Consume the shared Qwen3.5 prefill backend;
   MOSS owns only its chat-template/tokenizer binding and hidden-state
   consumption. Preserve caller layout JSON as prompt text, produce masked and
   final-normed `[T,4096]` states, enforce the 4096-token bound with explicit
   refusal, and do not add an LM head, sampler, vision/projector, second Qwen
   block set, second GGUF reader or cross-request state reuse.
4. **Full 42-layer flow loop.** Compose the synthetic-proven block into the
   manifest-derived depth. Positive traversal runs every layer; negative
   traversal skips only layer 22. Run separate predictions before
   `uncond + CFG*(cond-uncond)`; CFG=1 never invokes the negative branch. Use
   `sigma(u)=5u/(1+4u)`, `u=1-i/N` and `x_next=x+(sigma_next-sigma)*v`, with
   recorded defaults 50 steps/shift 5/CFG 3. A synthetic depth-23 call trace
   proves 42-layer policy without checkpoint geometry. Reuse `SdPipeline`,
   deterministic seed and cancellation. Obtain a one-device byte plan before
   loading encoder, DiT or VAE; over-budget admission returns
  `VkError::InsufficientCapacity`, with no ad hoc offload or sharding.
5. **FLUX.2-small VAE adapter.** Reuse shared VAE kernels/specification, not
   `VaeRealWeights`. Implement 32-channel latent to 2x2-packed 128-channel
   diffusion latent and inverse, effective /16 image geometry, rectangular
   shape checks and per-channel normalize/denormalize identity. Keep
   normalization constants unknown until sourced; fail closed rather than
   copying SD1.5 scale. Prove synthetic full decode before real decode.
6. **Opt-in real Base/SFT gate.** After slices 1-5 and promotion/readiness
   gates, acquire the existing exclusive single-device lease before device
   selection and hold it for the whole run. Verify every component SHA-256 and
   recorded license first. Run Base at 1024x1024, then SFT; require RGBA output,
   finite pixels and same-device/same-seed repeatability. Record output hashes,
   peak bytes and elapsed time in the unique run directory. No download,
   conversion, bundling or quality claim without a pinned reference.
7. **Provenance.** Preserve OpenMOSS Apache-2.0 copyright/license notices,
   mark modified derived files and propagate upstream `NOTICE` when required.
   Retain the pinned MIT reference copyright/permission notice for translated
   expressions. Record Qwen encoder and FLUX.2 VAE terms separately. New shader
   expressions carry first-line provenance; no CUDA, GPL or proprietary
   reference source is admissible.

## Acceptance criteria

- [ ] Synthetic `moss_single_block` proves finite f32 CPU/Vulkan parity within
  absolute error `1e-4` for the tiny fixture; zero velocity leaves latents
  unchanged, zero Q/K gives uniform attention, and skipping the only block
  equals the final-head-only path.
- [ ] The first slice accepts text states directly and proves neither
  Qwen3.5-9B nor any VAE is loaded.
- [ ] Six-way AdaLN, 4D RoPE axis isolation, time-feature layout, SiTU rather
  than SwiGLU, shifted-flow Euler and malformed-shape pre-dispatch failures
  have independent references and zero/identity controls.
- [ ] `moss_manifest` infers geometry from synthetic sparse safetensors and
  rejects missing, extra, wrong-shape, mismatched-hash and unresolved-license
  Base/SFT components before device/payload work.
- [ ] `moss_conditioning` proves shared Qwen hidden-state consumption, official
  tokenizer/template binding, causal/padding masks, JSON preservation and
  explicit 4096-token refusal without a second Qwen runtime.
- [ ] `moss_flow_pipeline` proves positive 42-layer traversal, negative-only
  skip of layer 22, CFG=1 branch omission, post-prediction CFG, defaults,
  cancellation and shared one-device admission/refusal.
- [ ] `moss_vae_contract` proves 32/128-channel 2x2 pack/unpack, /16 geometry,
  rectangular rejection, normalization identity and synthetic full decode.
- [ ] Ignored `moss_real` runs Base then SFT only with a complete user-supplied
  component manifest, recorded SHA-256/license per artifact and exclusive
  single-device lease; repeatability and finite RGBA outputs pass and measured
  bytes/time are recorded, without quality or availability claims.
- [ ] `bun run spdx` passes with Apache-2.0/MIT notices intact and component
  licenses kept separate.
- [ ] Synthetic suites pass under the default parallel runner with unique
  per-test resources, deterministic teardown and no ordering dependence.

## Verification

All commands are proposed and were not run during filing. Synthetic tests
run with the default parallel runner: each owns a unique temporary directory
and local device/buffer/state resources, has no shared mutable fixture or
ordering dependence, and releases resources deterministically on failure. Only
the ignored real-checkpoint gates serialize, through their exclusive lease.

```sh
cargo test --locked -p honkbox-ash-backend --features ash-sd moss_single_block
cargo test --locked -p honkbox-ash-backend --features ash-sd moss_manifest
cargo test --locked -p honkbox-ash-backend --features ash-llm,ash-sd moss_conditioning
cargo test --locked -p honkbox-ash-backend --features ash-sd moss_flow_pipeline
cargo test --locked -p honkbox-ash-backend --features ash-sd moss_vae_contract
bun run spdx
```

The test-owned component manifest is read-only JSON containing `flavor`
(`base` or `sft`), each component path, SHA-256 and concrete license expression.
It is fixture/admission data, not a public DTO. Each ignored test acquires the
existing exclusive single-device lease itself, failing when unavailable, before
selecting a device. Use a fresh run directory for each flavor:

```sh
run_dir="$(mktemp -d .tmp/honkbox-moss-real.XXXXXX)"
HONKBOX_ASH_MOSS_TEST_TMPDIR="$run_dir" \
HONKBOX_ASH_MOSS_MANIFEST=/read-only/moss-base-components.json \
cargo test --locked -p honkbox-ash-backend --features ash-sd moss_real -- --ignored --test-threads=1 --nocapture

run_dir="$(mktemp -d .tmp/honkbox-moss-real.XXXXXX)"
HONKBOX_ASH_MOSS_TEST_TMPDIR="$run_dir" \
HONKBOX_ASH_MOSS_MANIFEST=/read-only/moss-sft-components.json \
cargo test --locked -p honkbox-ash-backend --features ash-sd moss_real -- --ignored --test-threads=1 --nocapture
```

Run the ignored gates alone on an idle GPU host. Retain their unique evidence
directories; never point two runs at the same directory or device lease.

## Non-goals

Reference-image editing, vision/projectors, edit LoRA, training, audio/other LLM
families, multi-GPU, CPU offload, new HTTP DTOs, bundled/downloaded/converted
assets, a second GGUF parser/cache/tokenizer/residency/registry/availability
mechanism, and any quality, speed, VRAM or product-availability claim. This
ticket does not assert that Base, SFT, Qwen3.5-9B or FLUX.2-small currently run
on Ash.