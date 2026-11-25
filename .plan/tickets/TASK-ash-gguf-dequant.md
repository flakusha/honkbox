# TASK: Add reference GGUF dequantization for the initial Ash dtype set

Status: proposed
Priority: P1
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: independent CPU vectors prove Q4_K decode for Q4_K_S/Q4_K_M containers plus required F16/BF16 conversion, with checked shapes/ranges and named errors for every unsupported or malformed input.
Verification: `TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-ash-backend --features ash-compute gguf_dequant -- --nocapture`, `TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-ash-backend --features ash-compute gguf_reader -- --nocapture`, `bun run spdx`.
Links: `.tmp/research/ash-family-implementation-index-2026-11-25.md` §4 S2 and §5; `.tmp/research/ash-family-evidence-inventory-2026-11-25.md` §2–4; `.tmp/research/ash-qwen35plus-implementation-candidate-2026-11-25.md` §4–5; `.tmp/research/ash-lfm-implementation-candidate-2026-11-25.md` §4–6; `.tmp/research/ash-gemma4-implementation-candidate-2026-11-25.md` §5–6; `.tmp/research/llm-support-lfm-2026-11-24.md`; `.tmp/research/llm-support-gemma4-2026-11-24.md`; `crates/ash-backend/src/unet_checkpoint.rs`; `crates/ash-backend/src/matmul.rs`; `vendor/llama-cpp-sys-4-0.11.0/llama.cpp/ggml/src/ggml-quants.c`; `vendor/llama-cpp-sys-4-0.11.0/llama.cpp/LICENSE`.
Resources: deterministic in-memory blocks and CPU reference vectors; each test owns a unique directory under `TMPDIR` with PID, atomic sequence and timestamp, removed by RAII teardown; no ports, databases, devices, model files or shared globals.

## Verified 2026-11-25

- **[FACT]** Ash has no GGUF dtype/dequantization path. Existing checkpoint
  decode handles F32/F16/BF16 for one UNet input block, and `MatMul` consumes
  row-major `f32` slices (`crates/ash-backend/src/unet_checkpoint.rs:43-105`;
  `crates/ash-backend/src/matmul.rs:21-96`).
- **[FACT]** Pinned GGML uses one `GGML_TYPE_Q4_K` tensor type (code 12) for
  both Q4_K_S and Q4_K_M. A block holds 256 values in 144 bytes: two F16
  super-block scales, 12 packed scale/min bytes and 128 packed quant bytes
  (`ggml/include/ggml.h:390-422`; `ggml/src/ggml-common.h:87-89,323-338`;
  `ggml/src/ggml.c:800-806`).
- **[FACT]** `general.file_type` labels Q4_K_S as 14 and Q4_K_M as 15, but the
  tensor descriptor still carries `GGML_TYPE_Q4_K`. A majority-type label or
  filename cannot substitute for inspecting each tensor dtype.
- **[FACT]** Research records local LFM text Q4_K_M, Qwen3.5 text Q4_K_M and
  Gemma E4B Q4_K_S candidates. The same inventory also records Q5_K_M,
  Q5_K_S and Q8_0 tensors/files; those are outside this initial slice and must
  fail rather than be approximated.
- **[UNKNOWN]** The complete per-tensor dtype mix of the eventual selected
  checkpoints and real-weight numerical tolerance. No checkpoint was opened or
  executed for this ticket. Synthetic proof is sufficient for S2 completion.

## Dependencies

- `TASK-ash-native-families-index.md` owns shared-seam ordering.
- `TASK-ash-gguf-reader.md` supplies validated dtype/shape/range metadata and
  exact on-demand bytes. S2 never reparses GGUF or trusts filename/header
  majority labels in place of tensor descriptors.
- Reuse the existing F32 decode and share the F16/BF16 bit-conversion helpers
  currently private to `unet_checkpoint.rs`; do not maintain duplicate IEEE
  conversion routines.
- Decoder ops, tokenizer, sampling and family graph tickets are consumers, not
  dependencies. Admission/provenance gates still apply before any later family
  loads real weights.

## Change

- Add a type-directed CPU decode seam in the shared checkpoint module. It
  accepts validated tensor metadata plus exactly the declared payload bytes
  and returns flat `f32` values in GGUF `ne` storage order for the existing
  row-major `MatMul` contract. Shape permutation, if needed by a family, is not
  hidden in this decoder.
- Implement Q4_K once for tensor type 12. Decode each 256-value super-block as
  eight groups of 32: unpack the six-bit scale/min fields, decode F16 `d` and
  `dmin`, map low then high quant nibbles in the pinned order, and compute
  `d * scale * q - dmin * min`. Do not create separate S/M wire formats or
  infer tensor dtype from `general.file_type`.
- Reuse shared F16 and BF16 conversion for scalar tensors. Preserve existing
  F32 behavior. F16/BF16 conversion is bit-preserving, including signed zero,
  subnormals and IEEE non-finite values; no clamping or NaN replacement is
  introduced. Reject non-finite Q4_K super-block scales as invalid data.
- Validate before allocation or matmul: exact payload length, nonzero checked
  dimensions, `ne[0] % 256 == 0` for Q4_K, checked block count and exactly
  144 bytes per block. Reject short/long/trailing payloads, element/byte
  overflow and every unsupported dtype with a named error. Never truncate,
  zero-fill, reinterpret as another format or fall back to F16.
- Add checked-in, hand-computable CPU vectors with documented provenance for
  zero, minimum/maximum nibbles, scale/min extraction boundaries, low/high
  nibble ordering, signed scales, multiple blocks and 256/512-element rows.
  Include deterministic mixed-block vectors and F16/BF16 special values.
  Expected values must not be produced by the production decoder.
- Add shape/range tests for 1/256/257/512 elements, multidimensional `ne`
  shapes, malformed blocks, unsupported Q5_K/Q8_0 descriptors and overflow.
  Power-of-two vectors are bit-exact; other Q4_K vectors allow at most 2 ULP
  against the independent scalar equation.
- Add a synthetic integration test that feeds decoded `f32` operands and their
  explicit shapes to a CPU dot-product reference matching `MatMul` semantics.
  This proves the immediate-before-matmul data contract without dispatch,
  device selection, persistent-weight policy or a GPU performance claim.
- If an expression is translated from pinned MIT ggml source, retain its
  notice and record the exact upstream path/revision and modification in the
  repository provenance convention. Do not copy shader files wholesale.

## Acceptance criteria

- [ ] Q4_K CPU output matches independent vectors for every packed scale/min
  boundary and quant-nibble position, including multiple 256-value blocks.
- [ ] A Q4_K_S-labelled and Q4_K_M-labelled generated container with the same
  Q4_K tensor bytes decode identically; conflicting per-tensor dtypes fail.
- [ ] F16/BF16 conversion is bit-exact against IEEE fixtures and reuses the
  existing conversion behavior; F32 safetensors decode remains unchanged.
- [ ] Invalid block lengths, non-multiple row widths, empty/overflowing shapes,
  non-finite Q4_K scales and unsupported dtypes fail before matmul with named
  errors; there is no silent fallback or partial output.
- [ ] Decoded values and shapes satisfy the existing `MatMul` input contract
  against an independent CPU dot-product fixture.
- [ ] Tests require no model download, real checkpoint, Vulkan device or new
  dependency and make no readiness, compatibility, speed or memory claim.

## Verification

```sh
TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-ash-backend --features ash-compute gguf_dequant -- --nocapture
TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-ash-backend --features ash-compute gguf_reader -- --nocapture
TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-ash-backend --features ash-compute unet_checkpoint::tests -- --nocapture
bun run spdx
```

These are proposed commands, not run by the filing pass. No real-checkpoint
test is part of S2. Any later family opt-in real-weight proof must use an
exclusive single-device lease, recorded SHA-256 and license for every supplied
artifact, and serialized execution; it does not expand this ticket scope.

## Non-goals

Quantization/encoding; formats beyond the evidenced initial set; GPU dequant
kernels or optimization; persistent weight residency; family tensor maps,
tokenizer, decoder or sampling; real-checkpoint parity; bundled weights,
downloads or conversion; multi-GPU; availability, quality, latency or VRAM
claims.