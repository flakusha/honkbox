# TASK: Add portable Ash decoder operations

Status: proposed
Priority: P1
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: shared `Op` variants and portable f32 kernels implement RMSNorm, RoPE, SwiGLU, per-head QK norm and one-shot causal short convolution, with scalar CPU-reference synthetic tests and named invalid-dimension failures; no fast path lands before measurement.
Verification: `TMPDIR="$(mktemp -d "$PWD/.tmp/ash-decoder-ops.XXXXXX")" cargo test --locked -p honkbox-ash-backend --features ash-compute ash_decoder_ops`; `cargo test --locked -p honkbox-ash-backend --features ash-compute model::tests`; `bun scripts/check-ash-provenance.ts`.
Links: `.tmp/research/ash-family-implementation-index-2026-11-25.md` §4 S4; `crates/ash-backend/src/model.rs`; `crates/ash-backend/src/norm.rs`; `crates/ash-backend/src/mlp.rs`; `crates/ash-backend/src/compute.rs`; `crates/ash-backend/build.rs`; `.tmp/research/ash-lfm-implementation-candidate-2026-11-25.md` §5; `.tmp/research/ash-gemma4-implementation-candidate-2026-11-25.md` §5; `.tmp/research/ash-qwen35plus-implementation-candidate-2026-11-25.md` §4.
Resources: deterministic in-memory f32 tensors; unique gitignored `.tmp/ash-decoder-ops.*` directories; existing headless Vulkan smoke pattern; no checkpoint, weight download, bundled model, fixed port, database or shared mutable state.

## Verified 2026-11-25

- **[FACT]** `Op` currently has add, matmul, softmax, attention, LayerNorm, Conv2d, SiLU, GroupNorm, ConvTranspose2d, DDIM sampling, CFG, embedding gather, GELU and transpose; none of the five requested decoder ops exists (`model.rs:15-44`).
- **[FACT]** `LayerNorm` subtracts the row mean and applies affine scale/shift; it is not RMSNorm (`norm.rs:12-97`). `FeedForward` is a GELU two-projection block, not SwiGLU (`mlp.rs:94-179`).
- **[FACT]** `build.rs` registers the current shader set and none of RMSNorm, RoPE, gated multiply or short convolution (`build.rs:7-22`). `compute::Kernel` and `PhysicalDeviceInfo::admit` are the shared dispatch/admission seams.
- **[FACT]** S4 requires portable f32 implementations, CPU-reference synthetic shapes and invalid head/dimension cases, with capability-gated fast paths only after measurement (`ash-family-implementation-index-2026-11-25.md:67`).
- **[FACT]** LFM requires RMS/QK norm, RoPE, SwiGLU and depth-3 causal convolution; Gemma requires RMSNorm, RoPE and gated FFN; Qwen3.5 requires hidden/Q/K/output RMSNorm, RoPE, QK norm, SwiGLU and short convolution (candidate §§4–5).
- **[UNKNOWN]** Family epsilon values, gain presence, non-standard RoPE variants and convolution state semantics belong to later family adapters. No GPU correctness, throughput or VRAM result exists for these proposed ops.

## Dependencies

- `TASK-ash-native-families-index.md` owns S4 scope and ordering.
- `TASK-ash-gguf-dequant.md` supplies validated f32 operands and shape/range errors from supported quantized tensors. This ticket neither loads GGUF nor adds a dtype path.
- Reuse `Op::requirements`, `ModelSpec::validate`, `compute::Kernel`, `GpuBuffer`, shader registration and existing headless test conventions.
- No dependency on a family graph, KV/recurrent state, sampler or server route.

## Change

- Add `RmsNorm`, `Rope`, `SwiGlu`, `QkNorm` and `ShortConv` variants to the shared `Op` vocabulary, with static storage/workgroup requirements and exhaustive admission updates. Register each new shader in `build.rs` with provenance.
- Implement portable f32 kernels and independent scalar CPU references. Keep all shapes row-major and all buffers caller-owned; no weights or cross-call state survive a dispatch.
- **RMSNorm:** for `[rows, cols]`, compute `x * rsqrt(mean(x^2) + eps) * gain` with `gain[cols]`. Reject empty dimensions, length mismatch, non-finite input/gain and non-finite or non-positive epsilon before dispatch.
- **QK norm:** apply the same RMSNorm math independently to each head of `[rows, heads, head_dim]`, with `gain[head_dim]`. Validate checked `rows * heads * head_dim`, non-zero heads/head dimension and exact input/gain lengths; do not mix values between heads.
- **RoPE:** accept `[rows, heads, head_dim]`, a scalar position offset and even `rotary_dim <= head_dim`. Rotate half-split pairs `(i, i + rotary_dim/2)` with base `theta`; copy the non-rotary tail unchanged. Reject odd/zero rotary dimension, oversized dimension, position overflow and non-finite parameters. Family MRoPE/p-RoPE is not part of this op.
- **SwiGLU:** accept equal-length non-empty `gate` and `up` vectors and return `silu(gate) * up`. Reject length mismatch and non-finite values. Projections, residuals and activation-order variants stay outside this primitive.
- **ShortConv:** implement one-shot depthwise causal convolution for `[rows, channels, time]` with row-major weights `[channels, kernel]`, stride/dilation 1 and left zero padding `kernel - 1`. Output preserves `[rows, channels, time]`. Reject zero dimensions, kernel zero, weight mismatch and non-finite values. Rolling convolution state is deliberately excluded.
- Add synthetic tests for single/multiple rows, one/multiple heads, tail widths, positions 0/1/large, rotary and non-rotary tails, sequence lengths 1/2/3/5 and kernels shorter/longer than the sequence. Compare portable output with the scalar reference using a declared absolute tolerance.
- Add invalid-dimension tests for every op: empty/overflowing shapes, mismatched gain/weight lengths, odd rotary dimensions, non-positive epsilon and non-finite inputs. Errors must name the op and field before any device dispatch.
- Follow the existing headless Vulkan test pattern for dispatch parity when a device is available; retain named skips only for loader/no-device/no-compute-queue conditions. CPU-reference and validation tests must pass without Vulkan.
- Keep the portable path mandatory and optional-feature-free. Any subgroup/cooperative-matrix or other fast path requires a separate measured comparison and fallback proof; this ticket makes no speed claim.

## Acceptance criteria

- [ ] All five `Op` variants have exhaustive requirements, shader registration, portable f32 implementations and provenance.
- [ ] CPU-reference tests prove the equations and shape preservation on bounded synthetic tensors; QK norm never aliases heads.
- [ ] Invalid dimensions and non-finite parameters fail with named errors before dispatch, including overflow-safe element counts.
- [ ] Headless Vulkan parity meets the tolerance declared in the tests, or skips only for the existing unavailable-device cases.
- [ ] No family graph, persistent KV/convolution state, sampler, GGUF loader, new dependency, multi-GPU path or performance/availability claim is added.

## Verification

```sh
mkdir -p .tmp
TMPDIR="$(mktemp -d "$PWD/.tmp/ash-decoder-ops.XXXXXX")" \
  cargo test --locked -p honkbox-ash-backend --features ash-compute ash_decoder_ops
cargo test --locked -p honkbox-ash-backend --features ash-compute model::tests
bun scripts/check-ash-provenance.ts
```

These tests use synthetic tensors only. Any later opt-in real-checkpoint proof must run alone on an idle GPU host, acquire the exclusive single-device lease before device creation, and verify recorded checkpoint SHA-256/license before dispatch.

## Non-goals

Family-specific tensor maps or graphs; GQA/mask builders; KV cache, recurrent state or rolling convolution state; token sampling; GGUF parsing/dequantization; fused full decoder blocks; quantized kernels; multi-GPU; model downloads/bundling; benchmark, latency, VRAM, quality or availability claims.