# FEAT: Pure-Vulkan (`ash`) inference backend

Status: in progress (first slice landed in this worktree, uncommitted 2026-11-22)
Priority: P1
Epic: epic-full-ai-cycle
Owner: ash-vulkan lane
Acceptance: With `experimental-ash-sd` enabled, `AshSdPipeline::generate` renders a deterministic fixture from a user-supplied checkpoint using Vulkan compute only (no CUDA/ComfyUI process); missing device features return named `VkError::InsufficientCapacity`; every shader carries a provenance line (license comes from `REUSE.toml`, which overrides per-file headers in this repo).
Verification: `cargo test --locked -p honkbox-ash-backend --features ash-compute` (first slice, no vendor stack); `cargo test --locked -p honkbox-ash-backend --features experimental-ash-sd`; `bun run test:parity` with a local checkpoint (opt-in); `bun run check`.
Links: `.tmp/research/ash-vulkan-backend-licensing-extensibility-2026-11-22.md` (main checkout), `crates/ash-backend/src/{device,sd}.rs`, `TASK-device-capacity-gates.md`.

## Change

- Add an ops vocabulary (elementwise, matmul, norm/softmax, RoPE, conv2d,
  attention, sampler step, VAE decode slice) with one `ModelSpec` per family:
  graph + tensor-name map. New models add one spec module, not a new pipeline.
- Extend `PhysicalDeviceInfo::admit` to consume a required-capability set;
  optional extensions (e.g. cooperative matrix) get portable fallbacks.
- Compile GLSL → SPIR-V at build time, embed with `include_bytes!`; every source
  file carries a provenance line (original vs derived, upstream path + pinned
  rev). License stays in `REUSE.toml` — this repo has no per-file SPDX headers.
  No runtime shader compiler.
- Wire `server/experimental-ash-sd` in `main.rs` constructor only; keep
  `SdCppPipeline` as default. `stubs.rs` LLM/STT/TTS remain separate follow-ups.

## Non-goals

CUDA/ROCm backends, tensor/model parallelism, bundled weights, byte-level
equality with stable-diffusion.cpp, API/DTO changes.

## Landed — slice 1 (commit a32829a)

`elementwise_add` runs on the selected Vulkan device end to end: GLSL source,
offline glslang compile, embedded SPIR-V, descriptor set + compute pipeline,
host-visible buffers, capacity admission, fenced submit and readback. New
`ash-compute` feature builds this stack without the vendored diffusion-rs graph;
`experimental-ash-sd` forwards to it. Vulkan calls stay in `raw.rs`; dispatch
is serialized by a mutex because descriptor and command buffers are reused.

Evidence: `cargo test --locked -p honkbox-ash-backend --features ash-compute`
→ 18 passed (GPU test exercised 4097 elements on RADV RX 7900 XTX, not skipped);
`cargo clippy ... --all-targets -- -D warnings` → clean; `cargo fmt --check` →
clean. `experimental-ash-sd` full-graph check green: 233 crates compiled,
`Finished dev profile` (13m25s).

## Landed — slice 2 (2026-11-22, worktree)

`MatMul` adds row-major GEMM on the same pipeline scaffold. `compute::Kernel`
now owns the shader/pipeline/descriptors/command buffer and is shared by both
ops; `GpuBuffer` and SPIR-V decoding are shared too. New `shaders/matmul.comp`
uses an 8×8 tile with a shape guard; shape mismatch is rejected before any GPU
allocation. Evidence: 19 tests pass (`matmul_dispatches` ran on GPU, no skip),
clippy `-D warnings` clean, fmt clean, `experimental-ash-sd` graph check green
(1 crate rebuilt incrementally).

## Landed — slice 3 (2026-11-22, worktree)

Capability discovery + admission: `PhysicalDeviceInfo` now records all device
extension names (185 on RX 7900 XTX) and shader float16/int8 bits from core
`vkGetPhysicalDeviceFeatures2`. `CapacityRequirements` gains matching required
flags plus `required_extensions`; `admit` names the first missing capability.
`has_extension` lets optional fast paths fall back without failing admission —
no speculative capability enum or config added.

Evidence: 23 tests pass (4 new pure admission tests); headless smoke on real
GPU reports 185 extensions; clippy `-D warnings` clean; fmt clean;
`experimental-ash-sd` graph check green.

## Landed — slice 4 (2026-11-22, worktree)

`model::{Op, ModelSpec}` introduces the ops vocabulary: one variant per landed
kernel (ElementwiseAdd, MatMul), each declaring a static capability floor.
`ModelSpec::validate(&PhysicalDeviceInfo)` admits every op in the model before
any pipeline is built. Tensor-name map is deferred until a checkpoint exists;
no placeholder fields added. Shared test fixture now reports realistic limits.

Evidence: 27 tests pass (4 new pure model tests, including a failing-capability
case); clippy `-D warnings` clean; fmt clean; `experimental-ash-sd` graph check
green.

## Landed — slice 5 (2026-11-22, worktree)

`generate` now runs its front half: edge/dimension checks, cancel-before-work,
`ModelSpec` admission on the selected device, and real checkpoint container
validation (safetensors header length, GGUF magic, non-empty regular file).
The pipeline holds a resolved checkpoint path via `new(device, path)`. A valid
checkpoint then hits a named "sampling not implemented" error — no fake image.

Actual sampling remains out of reach in one slice: it needs text encoders,
diffusion backbone, VAE and a scheduler. No stubs for those added.

Evidence: 31 tests pass (4 new: supported formats, bad inputs, full generate
ordering); generate test ran on the real GPU; clippy `-D warnings` clean;
fmt clean; full `experimental-ash-sd` test build and graph check green.

## Landed — slice 6 (2026-11-22, worktree)

First diffusion kernel: `Softmax::run(values, rows, cols)` does stable row-wise
softmax with one invocation per row (max subtraction before exp, then
normalize). New `shaders/softmax.comp` carries the required provenance line and
is compiled offline by `build.rs`; the Rust op uses the shared `Kernel`
scaffold and `Op::Softmax` joins the vocabulary. No subgroup optimization yet.

Evidence: 28 tests pass on `ash-compute`, 32 on `experimental-ash-sd`; GPU test
ran on RADV (3 rows = partial workgroup, 1000.0 inputs = overflow guard) and
matched a CPU reference within 1e-6; clippy `-D warnings` clean; fmt clean.

## Landed — slice 7 (2026-11-22, worktree)

`Attention` composes the existing kernels: score matmul → scaled softmax →
output matmul. `Softmax::run_scaled` uses the previously spare binding as an
f32 scale (attention passes `1/sqrt(head_dim)`); `AttentionShape` groups the
four shape values. `K` is supplied pre-transposed — no transpose kernel yet.

Evidence: 30 tests pass on `ash-compute`, 34 on `experimental-ash-sd`; GPU
attention test matched a CPU reference within 1e-4; clippy `-D warnings` clean;
fmt clean. `Op::Attention` joins the vocabulary.

## Landed — slice 8 (2026-11-22, worktree)

`LayerNorm::run(values, rows, cols, gamma, beta, eps)` normalizes each row and
applies per-column affine params (packed gamma/beta in one buffer). New
`shaders/layer_norm.comp` uses one invocation per row; `Op::LayerNorm` joins the
vocabulary. GroupNorm for UNet conv blocks is still absent.

Evidence: 32 tests pass on `ash-compute`, 36 on `experimental-ash-sd`; GPU test
on RADV matched a CPU reference within 1e-4; clippy `-D warnings` clean;
fmt clean.

## Landed — slice 9 (2026-11-22, worktree)

`Conv2d::run(input, weights, bias, ConvShape)` adds direct NCHW convolution:
square kernel, stride, zero padding, per-channel bias, packed weight+bias
buffer, and pure shape validation (including derived output-size checks). New
`shaders/conv2d.comp` is one invocation per output element — no im2col or
tiling yet. `Op::Conv2d` joins the vocabulary.

Evidence: 34 tests pass on `ash-compute`, 38 on `experimental-ash-sd`; 2×2×4×4
convolution with padding matched a CPU reference within 1e-4; clippy
`-D warnings` clean; fmt clean.

Note: a sibling lane added `experimental-ash-llm` to `Cargo.toml` mid-slice,
forcing an approved offline `Cargo.lock` regeneration. That lock change and
the sibling's LLM files are deliberately NOT part of this commit.

## Landed — slice 10 (2026-11-22, worktree)

`Silu::run(values)` adds the elementwise `x / (1 + exp(-x))` activation on one
invocation per element. New `shaders/silu.comp` is offline-compiled and uses
the shared four-buffer scaffold; `Op::Silu` joins the vocabulary and shares
the 4-byte static floor with elementwise add.

Evidence: 35 tests pass on `ash-compute`, 39 on `experimental-ash-sd`; GPU test
on 129 elements (partial workgroup, negative and large inputs) matched a CPU
reference within 1e-5; clippy `-D warnings` clean; fmt clean.

Next: GroupNorm for UNet conv blocks, then VAE decode slice.


## Landed — slice 11 (2026-11-22, worktree)

`GroupNorm::run(values, gamma, beta, shape, eps)` normalizes each `(batch,
group)` over `C/G × H × W` elements, then applies per-channel affine params.
New `shaders/group_norm.comp` uses one invocation per element and recomputes
group statistics per thread — a deliberate O(N × group_size) starting point
with a two-pass reduce as the upgrade path. GroupNorm joins the `Op` vocabulary
with a 24-byte static floor; shape validation rejects zero dims, non-divisible
groups and buffer mismatches.

Evidence: 37 tests pass on `ash-compute`, 41 on `experimental-ash-sd`; GPU test
on a 1×4×2×2 / 2-group tensor matched a CPU reference within 1e-4;
clippy `-D warnings` clean; fmt clean.

## Landed — slice 12 (2026-11-22, worktree)

`ConvTranspose2d::run` adds NCHW transposed convolution for VAE upsampling:
weights `[in_c, out_c, k, k]`, bias `[out_c]`, derived output-size validation,
and one invocation per output element (no atomics for overlapping strides).
`VaeDecode` composes it with `GroupNorm`, `Silu` and a 1×1 `Conv2d` into one
decoder block; all weights come from the caller, so no image is faked and no
model data is bundled. `Op::ConvTranspose2d` joins the vocabulary.

Evidence: 40 tests pass on `ash-compute`, 44 on `experimental-ash-sd`; the
transposed-conv GPU test (2×2 latent → 4×4, kernel 2, stride 2) and the full
decode GPU test both matched CPU references within 1e-4; clippy `-D warnings`
clean; fmt clean.

Next: scheduler/sampling slice, then wire into `generate`.

## Landed — slice 13 (2026-11-22, worktree)

`DdimStep::run` adds the DDIM eta=0 update on the shared Vulkan compute
scaffold: `x_{t-1} = sqrt(a_prev) * (x_t - sqrt(1-a_t) * eps) / sqrt(a_t)
+ sqrt(1-a_prev) * eps`. `DdimAlphas::from_cumprod` derives the four
coefficients from consecutive cumulative alphas and rejects out-of-range or
non-finite input. `seeded_uniform_noise` derives a deterministic latent from a
seed and index using sd-core's `splitmix64` (no second hash copy). `Op` gains a
`SamplingStep` capability floor. `generate` is unchanged: no model prediction
exists yet, so it still names the missing pieces instead of faking an image.

Evidence: 43 tests pass on `ash-compute` and 44 on `experimental-ash-sd`; the
DDIM GPU test matched a CPU reference within 1e-4 on an 8-element latent;
clippy `-D warnings` clean; fmt clean.

## Landed — slice 14 (2026-11-22, worktree)

`Cfg::run` applies classifier-free guidance on the Vulkan compute scaffold:
`uncond + scale * (cond - uncond)`, with scale validated as finite and >= 1.
`TextConditioning` holds caller-supplied positive/negative token embeddings
with a `(tokens, dim)` shape contract and finite-value checks; its `guided`
method routes through the CFG pipeline. No prompt→embedding conversion exists
here: CLIP tokenization needs vocab/merge assets and an encoder checkpoint, so
nothing is derived from a prompt string. `Op::ClassifierFreeGuidance` joins the
vocabulary with an 8-byte floor. `generate` remains unchanged until a text
encoder and diffusion backbone exist. (Committed in f791620 together with
shaders/cfg_guidance.comp, whose build.rs registration and compute.rs embed
predate the source file.)

Evidence: 47 tests pass on `ash-compute` and 52 on `experimental-ash-sd`; the
CFG GPU test matched a CPU reference within 1e-4 at scale 7.5, and scale 1.0
preserved the conditional vector within 1e-6; clippy `-D warnings` clean; fmt
clean.

## Landed — slice 15 (2026-11-22, worktree)

The LLM placeholder becomes a real worker seam: `llm.rs` (new, gated
`experimental-ash-llm`) hosts `AshLlmPipeline` implementing the
`honkbox-llm-worker` `LlmPipeline` trait; `stubs.rs` keeps only the STT/TTS
placeholders. Semantics: this is a placeholder on the direct-Vulkan path,
NOT a deterministic emulator — `complete` and `complete_stream` (and the
cancellable forward) return `LlmError::Load("direct-Vulkan chat is not
implemented yet: ... no output is faked")`; the token sink is never
invoked. `backend_id()` is `"ash-llm"`. No generation ever starts, so
`abort` stays the trait-default no-op and `complete_stream_cancellable`
keeps the documented legacy path. `server/experimental-ash-llm` forwards
to the crate feature; `--llm-backend ash-llm` exists only in feature builds
and constructs `VulkanDevice::headless()` at startup — device failure is a
loud named exit-1, never a silent stub fallback. No new config keys.
`honkbox-llm-worker` is declared by path in both crates (the workspace
dependencies table does not carry it).

Evidence: `cargo test --locked -p honkbox-ash-backend --features
experimental-ash-llm` → 48 passed (5 new: backend id, named unimplemented
complete, stream-without-tokens, cancellable legacy path, device-in-a-box
construction); `--features ash-compute` regression → 43 passed; server
`cargo check --locked` green with and without the feature; workspace
clippy `--features honkbox-server/experimental-ash-llm -- -D warnings`
clean; `cargo fmt --check` clean.

## Landed — slice 16 (2026-11-23, worktree)

The STT/TTS placeholders become real worker seams, same posture as slice
15: `stubs.rs` keeps both types (two small seams, one module) and — behind
new `experimental-ash-stt` / `experimental-ash-tts` crate features (shared
optional `honkbox-audio-worker` path dep) — `AshSttPipeline` implements
`SttPipeline` and `AshTtsPipeline` implements `TtsPipeline`. Semantics:
placeholder on the direct-Vulkan path, NOT a deterministic emulator —
`transcribe` returns `SttError::Model(...)` and `synthesize` returns
`TtsError::Model(...)`, both "direct-Vulkan <modality> seam is not
implemented yet ... no output is faked"; no transcript text or PCM16
samples are ever produced. Backend ids `ash-stt` / `ash-tts`; the Send+Sync
const assertions stay. The module also builds under plain
`experimental-ash-sd` (types reserved, no worker traits) — the module gate
is now `any(experimental-ash-sd, experimental-ash-stt,
experimental-ash-tts)`. Server side: forwarding features, cfg-gated
`SttBackend::AshStt` / `TtsBackend::AshTts` variants, `make_stt`/
`make_tts` arms constructing `VulkanDevice::headless()` with a loud named
exit-1 on device failure — mirroring the slice-15 AshLlm arm — and two
cfg arms in the CLI model-scan matches. No new config keys.

Evidence: `cargo test --locked -p honkbox-ash-backend --features
experimental-ash-stt,experimental-ash-tts` → 51 passed (4 new: backend id
+ named-unimplemented error per seam); `--features ash-compute` regression
→ 47 passed; server `cargo check --locked` green with and without the two
features; workspace clippy `--features
honkbox-server/experimental-ash-stt,honkbox-server/experimental-ash-tts --
-D warnings` clean; `cargo fmt --check` clean.

## Landed — slice 17 (2026-11-23, worktree)

`checkpoint.rs` adds a strict safetensors header reader: 8-byte length prefix
bounded like the catalog probe, JSON tensor entries validated for known dtype,
non-empty non-overflowing shape, offsets matching shape × dtype width, and
contiguous non-overlapping ranges inside the file. `CheckpointMap` exposes the
checkpoint key → tensor slot map and reads a tensor's payload on demand; no
model data is bundled and no GPU work is queued during parsing. `serde` /
`serde_json` are new direct dependencies (both already workspace-pinned).

This is the tensor-name map the extensibility design calls for: a new model
family adds a spec over these slots, not new server routes or call sites.

Evidence: 49 tests pass on `ash-compute` and 54 on `experimental-ash-sd`;
synthetic fixtures cover payload readback and rejection of overlapping ranges,
unknown dtypes and bad headers; clippy `-D warnings` clean; fmt clean.

Next: wire the tensor map into `AshSdPipeline::generate` admission (real
checkpoint shape checks before any kernel is scheduled).

## Landed — slice 18 (2026-11-23, worktree)

`AshSdPipeline::generate` now admits checkpoints through `CheckpointMap`
instead of a header-length-only check. Safetensors must contain the `UNet`
input convolution `model.diffusion_model.input_blocks.0.0.weight` with rank 4,
4 latent input channels, a 3×3 kernel and a floating dtype; malformed headers,
missing tensors and wrong shapes are named before any GPU work is queued.
GGUF is refused by name because it has no safetensors tensor map — no silent
pass-through with unknown weights. No payload is copied and no image is
generated yet; sampling still stops with its named error.

Evidence: 49 tests pass on `ash-compute` and 56 on `experimental-ash-sd`;
clippy `-D warnings` clean with the SD, STT and TTS features; fmt clean.

Next: text encoder seam (CLIP tokenizer assets + embedding lookup), then
sampling wiring.

## Landed — slice 19 (2026-11-23, worktree)

`Embedding::run` gathers one row per token ID from a `vocab × dim` table on
the Vulkan compute scaffold; IDs are range-checked on the host, shapes are
validated, and out-of-range IDs are rejected by name. `TextEncoder::encode`
composes two gathers (token table keyed by caller IDs, positional table keyed
by sequence position) with the existing `ElementwiseAdd` kernel. No prompt is
tokenized here: CLIP BPE needs vocab/merge assets and a license review, so no
embedding is derived from a prompt string. `Op::EmbeddingGather` joins the
vocabulary with a 12-byte floor.

Evidence: 52 tests pass on `ash-compute` and 59 on `experimental-ash-sd`;
gather and composed encoding matched CPU references within 1e-5/1e-6 on the
Vulkan device; clippy `-D warnings` clean with SD/STT/TTS features; fmt clean.

Next: CLIP tokenizer assets + prompt→ID conversion (license review first),
then transformer encoder blocks and sampling wiring.

## Landed — slice 20 (2026-11-23, worktree)

`Gelu::run` adds the tanh-approximated GELU activation on the Vulkan compute
scaffold. `FeedForward::run` composes two matmuls around it
(`w2 · GELU(w1 · x)`) over caller-supplied weight matrices; projections and
biases stay with the caller, so no weights are bundled and no output is faked.
`Op::Gelu` joins the vocabulary with the 4-byte elementwise floor.

This lands before the tokenizer on purpose: CLIP vocab/merge assets are
MIT-licensed upstream but still need a REUSE entry and a vendoring decision;
the MLP compute primitive is asset-free and unblocks encoder blocks either way.

Evidence: 54 tests pass on `ash-compute` and 61 on `experimental-ash-sd`;
GELU and the composed MLP matched CPU references within 1e-4/1e-3 on the
Vulkan device; clippy `-D warnings` clean with SD/STT/TTS features; fmt clean.

Next: CLIP tokenizer asset decision (REUSE entry) + prompt→ID conversion,
then transformer block composition and sampling wiring.

## Landed — slice 21 (2026-11-23, worktree)

`ClipTokenizer` parses caller-supplied `vocab.json` / `merges.txt` and encodes
a prompt to CLIP token IDs with BOS/EOS. `encode` lowercases, collapses
whitespace, splits contractions and letter/non-letter runs, maps UTF-8 bytes
through the published GPT-2/CLIP byte→unicode table, applies ranked merges,
then resolves each token through the vocabulary. Unknown tokens and malformed
assets produce `InvalidData` errors.

Licensing decision: upstream CLIP assets are MIT (verified via openai/CLIP
LICENSE), but no vocab/merge files are vendored — callers point at their own
files — so no REUSE entry or license-header change is needed here. The
byte→unicode algorithm is reimplemented from the published description, not
copied verbatim.

ponytail: segmentation is lowercase + whitespace + Unicode-letter runs; no
ftfy/HTML unescaping or diacritic folding. Add when a prompt corpus shows the
gap.

Evidence: 57 tests pass on `ash-compute` and 64 on `experimental-ash-sd`;
`encode("hello world")` and `encode("it's")` matched hand-built vocab/merge
fixtures exactly; malformed assets and unknown tokens rejected; clippy
`-D warnings` clean with SD/STT/TTS features; fmt clean.

Next: transformer block composition (attention + residual + MLP + norms),
then sampling wiring and end-to-end text conditioning.

## Landed — slice 22 (2026-11-23, worktree)

`Transpose::run` adds a row/column swap kernel for `Kᵀ` (no separate host
round-trip layout juggling). `TransformerBlock::run` composes Q/K/V projections,
single-head attention, output projection, two residual adds and two layer norms
around `FeedForward`. All weights and norm parameters are caller-supplied; no
checkpoint keys are mapped yet.

ponytail: one head per block. Multi-head split/concat needs two more kernels;
add them when a real CLIP checkpoint makes the gap observable.

Evidence: 61 tests pass on `ash-compute` and 68 on `experimental-ash-sd`;
transpose matched a 3×5 CPU reference; the full block matched a CPU reference
for a 2-token/2-dim case within 1e-3; mismatched weights rejected; clippy
`-D warnings` clean with SD/STT/TTS features; fmt clean.

Next: wire token IDs → embedding → stacked blocks → text conditioning, then
DDIM sampling with caller-supplied model weights.

## Landed — slice 23 (2026-11-23, worktree): server selection wiring (f1cedf3)

Self-review found a blocking wiring gap: `experimental-ash-sd` was an empty
server feature and no constructor selected `AshSdPipeline`, so the capability
flag advertised a backend the server could not drive.

Changes: forward the feature to `honkbox-ash-backend/experimental-ash-sd`;
add `ash-sd` to `--sd-backend` and `generate --backend`, constructing the
pipeline with a headless device and a loud exit-1 on device failure; update
the `FeatureFlags` doc and regenerate its TS snapshot. `generate` still stops
with the named sampling error — no fake image is produced.

Evidence: `cargo check --locked -p honkbox-server --features experimental-ash-sd`
→ 0 errors (110 crates; only a pre-existing unused-`unimplemented` warning in
ash stubs when SD-only features are selected). `cargo test --locked
-p honkbox-api-types` → 16 pass; TS snapshot regenerated. OpenAPI snapshot
regeneration → `openapi covers all 53 mounted routes`; both generated
snapshots now carry the updated flag description. `cargo test --locked
-p honkbox-server --features experimental-ash-sd` → 4900 passed (39 suites).

Known nit: with SD-only features, ash stubs warn about unused `unimplemented`;
default-feature clippy gates are unaffected.

## Landed — slice 24 (2026-11-23, worktree)

`ClipTextEncoder::run` chains `TextEncoder::encode` (token + positional table
gather and sum) through a caller-weighted stack of `TransformerBlock`s,
returning `tokens × dim` hidden states. `ClipTextEncoderWeights` carries the
tables, one weight set per layer and a shared MLP width. Layer-count mismatch
and oversized sequences are rejected before dispatch.

ponytail: no causal or padding mask — attention is bidirectional here.
Callers pass exactly the tokens that should attend to each other; add a mask
to `Attention` when a real CLIP checkpoint needs padded prompts.

Evidence: 64 tests pass on `ash-compute` and 71 on `experimental-ash-sd`;
zero-layer output matched a CPU token+position reference within 1e-5, one
layer matched a direct `TransformerBlock::run` within 1e-6, and both rejection
paths were exercised; clippy `-D warnings` clean with SD/STT/TTS features;
fmt clean.

Next: pooled/sequence text conditioning over these hidden states, then DDIM
sampling with caller-supplied model weights.

## Landed — slice 25 (2026-11-23, worktree)

`TextConditioner::condition` encodes positive and negative token ID sequences
through the same `ClipTextEncoder`, mean-pools each `tokens × dim` hidden state
batch on the host, then blends them with `Cfg` classifier-free guidance.
Tokenization stays with the caller, so no asset files are touched and Vulkan
errors are not mixed with `io::Error`.

ponytail: mean pooling on the host instead of a GPU reduction; real CLIP pools
the projected EOS token. Add the projection head and pooling kernel when a
checkpoint's text tower is wired in.

Evidence: 66 tests pass on `ash-compute` and 73 on `experimental-ash-sd`;
scale 1.0 matched CPU pooled embeddings within 1e-5, scale 2.5 matched the
`uncond + scale*(cond - uncond)` reference within 1e-4, invalid scale rejected;
clippy `-D warnings` clean with SD/STT/TTS features; fmt clean.

Next: wire the conditioned embedding into `DdimStep` with caller-supplied
scheduler alphas and latents, then the diffusion backbone.

## Landed — slice 26 (2026-11-23, worktree)

`DdimSampler::run` walks a caller-supplied cumulative-alpha schedule from an
initial latent, calling a `predict` closure once per step and applying the
shared `DdimStep` kernel. The closure receives the current latent and step
index; the conditioned embedding can be captured there once the backbone
lands. Schedules must have at least two finite alphas in (0, 1] and be
non-increasing; predictor errors propagate by name.

No diffusion backbone is bundled — `predict` is the explicit seam, so this
slice never fakes a noise prediction.

Evidence: 68 tests pass on `ash-compute` and 75 on `experimental-ash-sd`;
the loop matched a sequential CPU DDIM reference within 1e-4 with the expected
step indices; short schedules, empty latents, non-monotonic alphas, wrong-
length predictions and closure errors were rejected; clippy `-D warnings`
clean with SD/STT/TTS features; fmt clean.

Next: the diffusion backbone itself — checkpoint tensor mapping for UNet
blocks, then a `predict` implementation closing the conditioning seam.

## Landed — slice 27 (2026-11-23, worktree)

`UnetInputBlock::run` composes the first real backbone stage: NCHW convolution
→ group norm → SiLU from caller-supplied tensors. `UnetInputShape` derives
output spatial size with checked arithmetic; group count, tensor lengths and
kernel geometry are validated before dispatch.

This is one block, not `predict`: no timestep embedding, attention, residual
or down/up path yet. No checkpoint keys are bound here — weights stay with
the caller.

Evidence: 70 tests pass on `ash-compute` and 77 on `experimental-ash-sd`;
the composed block matched a CPU convolution/group-norm/SiLU reference within
1e-3 on a 1×2×3×3 → 1×3×3×3 shape; empty input, invalid groups and oversized
kernel were rejected; clippy `-D warnings` clean with SD/STT/TTS features;
fmt clean.

Next: checkpoint key mapping for this block (safetensors tensors →
`UnetInputWeights`) and a stack of blocks, then the attention/residual path.

## Landed — slice 28 (2026-11-23, worktree)

`load_unet_input` maps four caller-named safetensors keys to the tensors
`UnetInputWeights` needs: kernel `[out_c, in_c, k, k]`, bias/gamma/beta
`[out_c]`. All shapes are validated before any payload bytes are read;
`F32`, `F16` and `BF16` decode to f32 via local bit expansion (std `f16`/`bf16`
are unstable on this toolchain — no dependency added).

No key family is assumed: SD 1.x/2.x and SDXL place the first convolution and
norms under different paths, so guessing would bind wrong tensors silently.

Evidence: 74 tests pass on `ash-compute` and 81 on `experimental-ash-sd`;
F32/F16/BF16 decode matched exact f32 references, missing keys, wrong shapes
and unsupported dtypes rejected; clippy `-D warnings` clean with SD/STT/TTS
features; fmt clean.

Next: stack these blocks with caller-supplied key sets, then the attention and
residual path needed for a real `predict` closure.

## Landed — slice 29 (2026-11-23, worktree)

`UnetStack::run` loads four caller-named tensors per block and runs a fixed
stack in order. Each `UnetStackBlockSpec` carries its own key set and NCHW
geometry; adjacent blocks must chain on channels and spatial size, otherwise
a named `UnetStackError::Shape` is returned before any load. Checkpoint and
dispatch failures keep their own variants instead of being flattened into
strings.

Still the input-style block only — no timestep embedding, attention or
residual path, so this is not yet a `predict` implementation.

Evidence: 76 tests pass on `ash-compute` and 83 on `experimental-ash-sd`;
a two-block stack (2→3→2 channels) matched direct sequential block runs within
1e-6; wrong spec count and broken channel chain rejected; clippy `-D warnings`
clean with SD/STT/TTS features; fmt clean.

Next: attention + residual path inside the UNet stage, then a `predict`
closure taking the conditioned embedding.

## Landed — slice 30 (2026-11-23, worktree)

`UnetAttentionBlock::run` adds the spatial self-attention stage over NCHW
tensors: group norm → `(batch·H·W) × C` sequence → caller-weighted Q/K/V →
single-head `Attention` → output projection → residual add → NCHW reshape.
Host-side layout conversion is used instead of a transpose kernel for the
reshape (small tensors; no extra GPU pass).

ponytail: one head, no cross-attention or timestep embedding. Add them when a
real transformer block checkpoint requires them.

Evidence: 78 tests pass on `ash-compute` and 85 on `experimental-ash-sd`;
the composed block matched a CPU group-norm/attention/residual reference
within 1e-3 on 1×2×2×2 tensors with identity projections; empty input and
wrong projection length rejected; clippy `-D warnings` clean with SD/STT/TTS
features; fmt clean.

Next: timestep embedding and a `predict` closure that consumes the conditioned
embedding, then end-to-end SD generation still needs the down/up sampling and
VAE decode paths.

## Landed — slice 31 (2026-11-23, worktree)

`sinusoidal(timestep, dim, max_period)` builds the cos/sin frequency table on
the host. `TimestepEmbed::run` feeds it through two caller-weighted linear
layers with `SiLU` between them, returning a `channels`-wide timestep vector.

Dimensions and timestep are limited to `u16::MAX` by the f32 conversions; SD
schedules stay well inside that bound. No dependency or shader added for the
frequency table.

Evidence: 80 tests pass on `ash-compute` and 87 on `experimental-ash-sd`;
t=0 produced [1, 0] in the first cos/sin pair, all frequencies stayed in
[-1, 1], and the MLP matched a CPU reference within 1e-4; zero dimensions and
wrong weight length rejected; clippy `-D warnings` clean with SD/STT/TTS
features; fmt clean.

Next: a `predict` closure that broadcasts the timestep embedding into spatial
tokens, adds conditioned text, and runs the UNet stages — then down/up blocks
and VAE decode remain for end-to-end generation.

## Landed — slice 32 (2026-11-23, worktree)

`UnetPredictor::predict` composes the timestep MLP, text-conditioning
broadcast, per-block input stack, and spatial attention into one noise
prediction. Caller supplies checkpoint key sets, projection weights and
geometry; shape errors (empty blocks, misaligned conditioning, attention
channel mismatch) are returned before any payload reads.

ponytail: conditioning must already be projected to latent channels and is
broadcast per spatial position — real SD uses cross-attention. No down/up
blocks or VAE decode yet; this is not end-to-end generation or checkpoint
parity.

Evidence: 82 tests pass on `ash-compute` and 89 on `experimental-ash-sd`;
the composed predictor matched a manual stage-by-stage run within 1e-6;
misaligned conditioning length rejected; clippy `-D warnings` clean with
SD/STT/TTS features; fmt clean.

Next: down/up sampling blocks and VAE decode so a latent can travel through a
full UNet and back to pixels.

## Landed — slice 33 (2026-11-23, worktree)

`UnetDownsample::run` loads `[out_c, in_c, k, k]` kernel and `[out_c]` bias
from a checkpoint and runs a strided `Conv2d` over NCHW input. Output geometry
is derived from input size, stride and padding; missing/misshaped tensors and
zero stride are rejected before payload reads. `load_tensor` is now public for
reuse by other checkpoint-backed stages.

Evidence: 84 tests pass on `ash-compute` and 91 on `experimental-ash-sd`;
a 4×4 → 2×2 stride-2 convolution matched a CPU reference within 1e-6;
missing bias key and zero stride rejected; clippy `-D warnings` clean with
SD/STT/TTS features; fmt clean.

Next: upsample + skip-connection stack, then VAE decode to pixels.

## Landed — slice 34 (2026-11-23, worktree)

`UnetUpsample::run` loads checkpoint kernel/bias, applies the existing
`ConvTranspose2d` to NCHW input, and adds a caller-supplied skip tensor of the
exact output shape. Derived output geometry, missing/misshaped keys, wrong
skip length and zero stride are rejected before dispatch.

ponytail: one conv-transpose + add per stage, no per-stage attention.

Evidence: 86 tests pass on `ash-compute` and 93 on `experimental-ash-sd`;
the GPU output matched the shader-faithful CPU gather plus skip within 1e-6;
missing bias key and wrong skip length rejected; clippy `-D warnings` clean
with SD/STT/TTS features; fmt clean.

Next: wire down/up stages into a full UNet traversal; `VaeDecode` already
exists as a separate one-block decoder component.

## Landed — slice 35 (2026-11-23, worktree)

`UnetTraversal::run` wires the four composed stages: downsample →
`UnetPredictor` bottleneck → upsample with the original latent as skip →
`VaeDecode` block. Encoder/bottleneck/decoder/VAE shapes are checked before
any dispatch; mismatched skip geometry is a named shape error.

ponytail: one down/up level and one VAE decoder block. No multi-scale skips,
no stacked VAE blocks, no real checkpoint spec, so this is not SD parity.

Evidence: 88 tests pass on `ash-compute` and 95 on `experimental-ash-sd`;
the traversal output matched a manual stage-by-stage run within 1e-6 on an
8×8 → 16×16 synthetic path; mismatched skip geometry rejected; clippy
`-D warnings` clean with SD/STT/TTS features; fmt clean.

Next: real-checkpoint model spec (tensor names + level topology), then
wire this traversal to the server seam behind the experimental feature.

## Landed — slice 36 (2026-11-23, worktree)

`model_spec.rs` declares checkpoint tensor names as typed key structs
(`ConvKeys`, `NormKeys`, `ResnetKeys`, `LevelSpec`) plus `UnetModelSpec` for a
whole family. `SD15_SINGLE_LEVEL` names a one-level SD 1.5-style topology;
`tensor_keys` enumerates roles and keys, `validate` reports the first missing
tensor with its role. New model families add a sibling spec constant.

ponytail: names and presence only — no shape inference, key remapping or
server dispatch yet.

Evidence: 90 tests pass on `ash-compute` and 97 on `experimental-ash-sd`;
a synthetic checkpoint containing every declared key validated, removing
`time_embedding.linear_2.bias` produced the named role/key error; topology
counts asserted; clippy `-D warnings` clean with SD/STT/TTS features; fmt
clean.

Next: infer stage shapes from tensor metadata and wire the traversal into
the server behind the experimental feature flag.

## Landed — slice 37 (2026-11-23, worktree)

`shape_infer::infer` reads checkpoint metadata for a spec: entry/down/mid/up/
output conv channels and kernels, timestep linear widths, VAE entry/exit conv
shapes. Rank errors and timestep channel disagreements are named and typed.
Spatial sizes remain caller-supplied.

Evidence: 92 tests pass on `ash-compute` and 99 on `experimental-ash-sd`;
synthetic SD15-spec checkpoint inferred 4 channels / 3×3 kernels / 4→8→4
timestep widths; bad rank and channel mismatch rejected; clippy `-D warnings`
clean with SD/STT/TTS features; fmt clean.

Next: server seam. This is a separate surface (api-types + routes + config + CI
gates) and should land as its own slice with the feature flag disabled by
default.

## Landed — slice 38 (2026-11-23, worktree)

Server seam is now usable without faking images: `AshSdPipeline` already had
CLI/serve constructors behind `experimental-ash-sd`; this slice adds
`sd::inspect_checkpoint(path)`, which validates the declared spec and infers
stage widths from tensor metadata without reserving a GPU slot, plus the
`honkbox models inspect-ash <slug>` CLI subcommand (same feature flag).
`generate` still stops with the named sampling error — no fabricated output.

Evidence: 92 tests pass on `ash-compute`, 100 on `experimental-ash-sd`;
a synthetic spec checkpoint inferred 4 channels / 8-wide timestep hidden and
a missing declared tensor produced the named role error; `cargo check -p
honkbox-server --features experimental-ash-sd --all-targets` clean (two
pre-existing warnings); CLI `--help` shows the new subcommand; clippy
`-D warnings` and fmt clean.

Next: real text-conditioned sampling. Needs CLIP/text-encoder wiring, fused
QKV cross-attention, and a full model loader — not a small slice.

## Landed — slice 39 (2026-11-23, worktree)

`cross_attention.rs` adds `FusedQkv::run` for diffusers-style
`[parts·out, in]` weights (transpose → matmul → per-row column split) and
`CrossAttention::run` for hidden queries × context keys/values, output
projection and residual. Reuses the existing transpose, matmul, attention and
add kernels; no new shader.

Also hardened the downsample test's temp filename with a per-process counter
after a full-suite run hit a same-nanosecond checkpoint collision.

ponytail: single head, no mask. Multi-head split/concat and padded prompts
need their own kernels and a real checkpoint to justify them.

Evidence: 95 tests pass on `ash-compute` and 103 on `experimental-ash-sd`;
fused split matched three independent matmuls within 1e-6; cross-attention
matched a manual decomposition within 1e-6; wrong weight length rejected;
clippy `-D warnings` clean with SD/STT/TTS features; fmt clean.

Next: text-conditioned UNet stage that feeds these context tokens into the
bottleneck, then the sampler loop.

## Landed — slice 40 (2026-11-24, worktree)

`UnetPredictor` now injects text via the cross-attention stage instead of
broadcasting a channel vector: `conditioning` is a flattened
`context_rows × channels` token matrix; `UnetCrossConfig` carries the fused
K/V projections and token count. NCHW ↔ token layout conversion is explicit;
timestep embedding remains additive. Manifold tests cover the single-token
identity case and the traversal composition.

ponytail: single head, no mask, context assumed pre-projected to latent
channels. A real checkpoint decides whether pooled/sequence conditioning and
multi-head attention are worth wiring.

Evidence: 95 tests pass on `ash-compute`, 103 on `experimental-ash-sd`;
server feature check clean (two pre-existing warnings); clippy `-D warnings`
and fmt clean.

Next: sampler loop that repeatedly calls this predictor, then real checkpoint
key mapping from SD 1.5 tensor names.

## Landed — slice 41 (2026-11-24, worktree)

`UnetTraversal::sample` drives the existing eta=0 DDIM loop with the same
down → predict → up stage chain, keeping the diffusion state at latent
resolution and decoding once after the final step. `SampleConfig` bundles
timesteps, context tokens and cumulative alphas; traversal shape validation is
shared with `run`. `UnetStackError` converts into `VkError` for the sampler
closure.

ponytail: fixed eta=0 schedule supplied by the caller, no variance sampling or
scheduler generation. Add those when a real checkpoint needs them.

Evidence: 96 tests pass on `ash-compute`, 104 on `experimental-ash-sd`; test
asserts distinct timesteps change the output, decoded length is correct, and a
short timestep list is rejected; server feature check clean (two pre-existing
warnings); clippy `-D warnings` and fmt clean.

Next: SD 1.5 checkpoint key mapping from tensor names to the local stage specs.
