# TASK: Ash single-stream flow-DiT substrate

Status: proposed
Priority: P1
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: synthetic CPU/Vulkan tests prove one single-stream DiT block, patch embed, AdaLN, per-attention QK norm, 2D/3D positions, timestep conditioning and flow prediction/step/scheduling, including identity controls and pre-dispatch shape failures; no image family executes.
Verification: `cargo test --locked -p honkbox-ash-backend --features ash-sd --lib flow_dit_substrate -- --test-threads=1`, `HONKBOX_ASH_REQUIRE_VULKAN=1 cargo test --locked -p honkbox-ash-backend --features ash-sd --test flow_dit_substrate_parity -- --test-threads=1 --nocapture`, `bun run spdx`.
Links: `.tmp/research/ash-family-implementation-index-2026-11-25.md` §4 S7 and §6; `.tmp/research/ash-moss-image-implementation-candidate-2026-11-25.md` §§4–6; `.tmp/research/ash-qwen-image21-implementation-candidate-2026-11-25.md` §§4–5; `crates/ash-backend/src/model.rs`; `crates/ash-backend/src/attention.rs`; `crates/ash-backend/src/timestep.rs`; `crates/ash-backend/src/sampler.rs`; `TASK-ash-decoder-ops.md`.
Resources: deterministic tiny synthetic tensors; unique temp dirs per test with teardown; one test-only RAII lease keyed to the selected physical device for Vulkan tests, held from before device selection through teardown; no checkpoint, download, fixed port, shared DB or bundled weights.

## Verified 2026-11-25

- **[FACT]** `Op` has 14 variants covering add, matmul, softmax, attention, layer/group norm, convolution, SiLU/GELU, DDIM sampling, CFG, embedding gather and transpose. It has no patch embed, AdaLN, RMS/RoPE, QK norm, multiply or flow step (`crates/ash-backend/src/model.rs:15-44`).
- **[FACT]** Multi-head attention and an additive `seq_q × seq_k` mask already exist; per-attention split/concat is currently host-side (`crates/ash-backend/src/attention.rs:9-29,75-150`).
- **[FACT]** `TimestepEmbed` is an integer-timestep, half-cos/half-sin MLP with two matmuls and SiLU (`crates/ash-backend/src/timestep.rs:31-79`). It is not a continuous flow-sigma embedding or positional RoPE.
- **[FACT]** The sampler is DDIM eta=0 plus seeded Gaussian noise (`crates/ash-backend/src/sampler.rs:49-161`); it is not a flow-matching predictor or scheduler.
- **[FACT]** Index S7 records no current patch embed, AdaLN, QK-normalized DiT block or flow predictor. MOSS and Qwen-Image candidates consume this shared seam rather than owning copies.

## Dependencies

- `TASK-ash-decoder-ops.md` owns shared RMSNorm, RoPE, QK norm, multiply and gated-FFN primitives. This ticket consumes those primitives and owns their DiT wiring; it does not add a second norm/RoPE kernel.
- `TASK-ash-native-families-index.md` is the shared dependency index and records this ticket as S7.
- `BUG-ash-sd15-numerics.md` and `TASK-ash-verification-gates.md` remain correctness/gate owners; this ticket does not move their fixes.
- MOSS `SiTU`, its family-specific modulation key layout and negative layer-22 skip, plus Qwen-Image mixed-segment masks, prefix reuse and Turbo sigma assets, are consumer deltas. They are not prerequisites for the synthetic substrate.

## Change

- Add a shared `dit` module with an explicit `FlowDitSpec` for width, attention count, Q/K width, FFN width, patch volume, latent channels and positional axes. Reject zero, odd or inconsistent dimensions before dispatch.
- Add 2D and 3D patch embedding as deterministic patch flatten/gather followed by the existing matmul path. Preserve batch, channel-major NCHW/NCDHW order and rectangular shapes; require exact token counts and patch-volume divisibility.
- Compose one pre-norm single-stream block: six-way AdaLN scale/shift for attention and FFN, QKV projection, shared per-attention QK norm, positional rotation, additive-mask attention, output projection/gate, FFN/gate and residual adds. Use the shared decoder-op FFN contract; the synthetic fixture names its activation explicitly.
- Add 2D `(row, column)` and 3D `(frame, row, column)` position builders with declared axis widths and centered image coordinates. Keep text-token offsets and family masks outside this shared builder. Zero position must be the rotation identity and each axis must be independently testable.
- Add continuous flow-sigma timestep conditioning with an explicit half-split frequency layout and MLP. Do not silently pass a flow sigma through the integer DDIM timestep API.
- Add a model-agnostic velocity predictor boundary `(latent, sigma, conditioning) -> velocity`, an Euler step `x_next = x + (sigma_next - sigma) * velocity`, and a validated shifted schedule with finite monotonic sigmas and terminal zero. A shift value is data, not a MOSS default.
- Register any new shader in `build.rs`, retain required notices and give translated or original expressions a first-line `// Provenance:` record. Prefer composition of existing kernels; add a shader only when the required operation is absent.
- Add a CPU reference that owns expected vectors independently of the Vulkan implementation, then run the same fixed vectors through Vulkan.

## Acceptance criteria

- [ ] CPU golden tests cover patch flatten/projection, six-way AdaLN, per-attention QK norm, 2D/3D axis isolation, flow-sigma embedding, one complete block and two scheduler steps.
- [ ] Synthetic Vulkan output matches the independent F32 CPU reference within absolute error `1e-4`; required-Vulkan mode fails rather than silently skipping when no usable device exists.
- [ ] Identity controls pass: patch-size-1 identity projection round-trips tokens; zero positions do not rotate; zero Q/K yields uniform attention; zero output projections/gates preserve residual inputs; zero velocity leaves the latent unchanged.
- [ ] Scheduler tests prove the declared shift equation, monotonic descent, terminal sigma zero, exact one-step Euler arithmetic and rejection of non-finite, increasing or malformed grids.
- [ ] Wrong token counts, patch divisibility, Q/K widths, axis splits, conditioning widths, mask lengths and sigma ranges fail before any device dispatch.
- [ ] Results describe one synthetic shared block and scheduler only. They do not claim MOSS/Qwen execution, model quality, latency, VRAM fit or family availability.

## Verification

```sh
cargo test --locked -p honkbox-ash-backend --features ash-sd \
  --lib flow_dit_substrate -- --test-threads=1
HONKBOX_ASH_REQUIRE_VULKAN=1 cargo test --locked \
  -p honkbox-ash-backend --features ash-sd \
  --test flow_dit_substrate_parity -- --test-threads=1 --nocapture
bun run spdx
```

The required-Vulkan test owns the exclusive device lease and records the selected physical device in its output. CPU tests remain runnable without Vulkan. No command was executed by the filing pass.

## Unknowns and stop conditions

- **[UNKNOWN]** Production family geometries, activation selection, dynamic-shift policy and numerical tolerances remain unproven. Stop rather than infer them from marketing metadata.
- **[UNKNOWN]** Qwen-Image Turbo requires an authoritative user-supplied sigma asset and SHA-256; do not guess its grid.
- Stop on CPU/Vulkan drift, incomplete lease coverage, missing provenance or any request to turn this substrate into a family runner without a separate admitted family ticket.

## Non-goals

MOSS-Image or Qwen-Image execution; text encoders; VAE decode; classifier-free guidance branch policy; mixed-segment/prefix-cache attention; Turbo assets; HTTP DTOs/routes; model admission or availability flags; real checkpoints; downloads, conversion or bundled weights; multi-GPU; quality, speed or memory claims.