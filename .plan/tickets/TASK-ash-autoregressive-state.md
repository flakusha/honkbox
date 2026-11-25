# TASK: Add Ash autoregressive state caches

Status: done
Priority: P1
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: one shared per-sequence state runtime proves KV/GQA and recurrent-state append, rewind, copy and clear; causal, sliding and packed masks; position tracking; prefill/decode equivalence; two-sequence isolation; and cancellation-safe lifetime, without implementing a family model or a second inference cache.
Verification: `cargo test --locked -p honkbox-ash-backend --features ash-llm autoregressive_state_cpu`; ignored synthetic Vulkan proof under `flock` on one named device; `bun run spdx`.
Links: `.tmp/research/ash-family-implementation-index-2026-11-25.md` §4 S5 and §5; `.tmp/research/ash-qwen35plus-implementation-candidate-2026-11-25.md` §§4–5; `.tmp/research/ash-lfm-implementation-candidate-2026-11-25.md` §§5–6; `crates/ash-backend/src/attention.rs`; `crates/ash-backend/src/llm.rs`.
Resources: generated tiny tensors only; unique `.tmp/ash-state.*` test directories with deterministic teardown; one ignored synthetic Vulkan test on one named compute device under an exclusive file lock; no checkpoint, download, fixed port, database or global cache. Consuming-family real-checkpoint tests remain opt-in and must hold an exclusive single-device lease with recorded SHA-256 and license.

## Verified 2026-11-25

- **[FACT]** `Attention` is multi-head and accepts an additive mask, but
  `AttentionShape` has one head count and there is no KV input, GQA mapping,
  mask builder or persistent state (`crates/ash-backend/src/attention.rs:9-29,74-150`).
- **[FACT]** `AshLlmPipeline` owns one device and every verb currently returns
  `LlmError::Load(NotYet)`; no generation state survives a call
  (`crates/ash-backend/src/llm.rs:29-79`).
- **[FACT]** The family index assigns S5 to KV/recurrent caches, GQA,
  causal/sliding/packed masks and position state, with prefill/decode and
  cancellation proof. S5 depends on decoder ops S4 and precedes sampling S6.
- **[FACT]** Qwen3.5 dense needs KV only on full-attention layers plus
  per-sequence convolution-ring and matrix state on Gated DeltaNet layers; LFM2.5
  needs six KV caches and ten short-convolution states. These are consumer
  geometries, not models to implement here.
- **[UNKNOWN]** Real-checkpoint numerics, tolerances, capacity and peak bytes are
  unproven. This ticket makes no availability, quality, latency or VRAM claim.

## Dependencies

- `TASK-ash-decoder-ops.md` — required shared attention/norm/position/recurrent
  math; this ticket owns storage, layout, mapping and lifetime only.
- `TASK-ash-gguf-reader.md` and `TASK-ash-gguf-dequant.md` — consumed by later
  family admission; synthetic state proof must not wait for or duplicate them.
- `TASK-ash-sampling-contract.md` — downstream consumer of committed logits/state.
- `FEAT-ash-qwen35-dense.md` and `FEAT-ash-lfm25-text-decoder.md` — downstream
  consumers that supply layer plans and own real-checkpoint evidence.

## Change

- Add one shared `llm/state` runtime beside the existing attention implementation.
  A validated state plan names each layer as KV, recurrent or stateless and
  declares head counts, widths, window/context capacity, position mode and element
  size. Reject zero widths, non-divisible GQA ratios, overflow and unsupported
  layer plans before allocation or dispatch.
- Own KV storage once per sequence. Store unique K/V heads, map each query head to
  `query_head / group_size`, and feed the existing masked attention primitive;
  never keep a second expanded KV copy. Append prefill chunks and decode tokens in
  logical token order. Sliding-window eviction preserves absolute positions.
- Own recurrent storage once per sequence: convolution ring/history and matrix
  state slots described by the plan. Do not implement Qwen Gated DeltaNet or LFM
  convolution equations here; decoder ops update these slots through one shared
  boundary.
- Build causal, sliding-window and packed-segment additive masks. Packed segments
  have independent positions and block-diagonal visibility; no query may attend
  across a segment boundary or outside its window. Reject a mask row with no
  visible key.
- Track next position and position offset independently of physical cache slots.
  `append` commits tokens; `rewind` restores the exact target state by snapshot or
  replay; `copy` deep-copies buffers, positions and RNG-independent metadata;
  `clear` resets one sequence. Rewind must never approximate recurrent state.
- Prove chunked prefill followed by append-only decode equals one full forward
  pass for logits and final state within a stated tolerance. Interleave two
  sequences and prove writes, rewind, copy and clear on one cannot alias or
  mutate the other.
- Check `RequestCancel` before prefill, between chunks, before each append and
  after device readback. Cancellation discards the request state and returns no
  committed token; buffers are released by RAII on every exit. Report exact
  weight-independent state/scratch bytes to the existing residency demand seam.
- Keep this state request-local. It is pipeline execution state, not a response,
  prefix, prompt or cross-request inference cache.

## Acceptance criteria

- [x] KV tests cover equal and grouped query/KV heads, invalid ratios, append,
      context limits, sliding eviction and absolute-position preservation.
- [x] Causal, sliding and packed masks prevent future-token, out-of-window and
      cross-segment leakage; malformed/all-masked rows fail by name.
- [x] Synthetic KV and recurrent tests prove prefill/decode equivalence at token
      and chunk boundaries, including rewind by snapshot or exact replay.
- [x] Two interleaved sequences remain isolated through append, rewind, copy and
      clear; copied buffers do not alias.
- [x] Cancellation at every boundary prevents later commits, returns
      `LlmError::Cancelled` at the worker seam and releases all state.
- [x] State byte accounting includes KV, recurrent state, positions and scratch;
      no second cache, family graph, speculative decoder or multi-GPU path exists.

## Verification

```sh
mkdir -p .tmp
state_tmp="$(mktemp -d "$PWD/.tmp/ash-state.XXXXXX")"
trap 'rm -rf "$state_tmp"' EXIT
mkdir -p "$state_tmp/cpu" "$state_tmp/gpu"
TMPDIR="$state_tmp/cpu" \
  cargo test --locked -p honkbox-ash-backend --features ash-llm \
  autoregressive_state_cpu
TMPDIR="$state_tmp/gpu" HONKBOX_ASH_TEST_DEVICE='<exact-device-name>' \
  flock -x "$PWD/.tmp/ash-gpu.lock" \
  cargo test --locked -p honkbox-ash-backend --features ash-llm \
  --test ash_autoregressive_state -- --ignored --nocapture --test-threads=1
bun run spdx
```

The ignored Vulkan test uses generated tensors, asserts the selected physical
device name, and runs alone on an idle one-device host. No command is run by the
filing pass.

## Evidence 2026-11-25

- **[FACT]** CPU gate: 7 tests pass, 1 named-Vulkan proof ignored; full `honkbox-ash-backend` suite also passes 157 unit tests and 23 integration tests, with 2 unrelated/proof tests ignored.
- **[FACT]** Named Vulkan gate: `HONKBOX_ASH_TEST_DEVICE='AMD Radeon RX 7900 XTX'` under `flock`; GQA attention output matches CPU within 1e-4.
- **[FACT]** `bun run spdx`: REUSE 601/601 files licensed, 19 Ash shaders with provenance.
- **[FACT]** `cargo clippy` reports no findings in `llm/state`; unrelated pre-existing crate lint debt remains outside this slice.

## Non-goals

Any Qwen/LFM/model graph; GGUF parsing/dequantization; tokenizer or sampler;
family tensor maps; real weights or downloads; response/prefix caching;
cross-request KV reuse; default speculative decoding; hosted llama-server changes;
multi-GPU/tensor parallelism; performance, quality, availability or VRAM claims.
