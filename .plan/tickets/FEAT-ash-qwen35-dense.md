# FEAT: Ash Qwen3.5 dense hybrid text proof

Status: proposed
Priority: P2
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: synthetic CPU-reference tests prove one Qwen3.5 Gated DeltaNet state step, one gated-attention/KV block, and a dense 3:1 motif through finite reference logits; a metadata-driven dense 9B gate admits only a verified user artifact and records an opt-in single-device logits comparison without adding sampling or claiming family availability.
Verification: `TMPDIR="$(mktemp -d "$PWD/.tmp/ash-qwen35-dense.XXXXXX")" cargo test --locked -p honkbox-ash-backend --features ash-llm qwen35_dense_proof`; the isolated ignored-checkpoint command in `## Verification`; `bun scripts/check-ash-provenance.ts`.
Links: `.tmp/research/ash-qwen35plus-implementation-candidate-2026-11-25.md` §§4–9; `.tmp/research/ash-family-implementation-index-2026-11-25.md` §4 S1–S6 and §6; `crates/ash-backend/src/attention.rs`; `crates/ash-backend/src/llm.rs`; `crates/ash-backend/src/model.rs`; `vendor/llama-cpp-sys-4-0.11.0/llama.cpp/src/models/qwen35.cpp`; `vendor/llama-cpp-sys-4-0.11.0/llama.cpp/src/models/delta-net-base.cpp`.
Resources: bounded synthetic f32 tensors; unique gitignored `.tmp/ash-qwen35-dense.*` directories; read-only user GGUF plus reference manifest with recorded SHA-256/license; one private test-only device selector and exclusive single-device lease; no download, bundled weights, fixed port, database, shared cache or multi-GPU.

## Verified 2026-11-25

- **[FACT]** Ash has multi-head attention with an additive `run_masked`, but no causal-mask builder, GQA/cache layout, gate, persistent KV, Gated DeltaNet, recurrence or LLM sampler (`attention.rs:9-29,64-148`; `model.rs:13-44`).
- **[FACT]** `AshLlmPipeline` implements the worker trait but every completion returns a named `Load` error; it emits no token (`llm.rs:29-79`). `CheckpointMap` reads safetensors, not GGUF (`checkpoint.rs:90-195`; `sd.rs:142-175`).
- **[FACT]** Research identifies Qwen3.5-9B as dense `qwen35`: hidden 4096, 32 layers, eight `(GDN, GDN, GDN, gated attention)` motifs, padded vocabulary 248,320 and native context 262,144. These are research facts, not inspected user-checkpoint shapes.
- **[FACT]** The smallest research slice is one synthetic GDN state step plus one gated-attention/KV block. Token and chunk recurrence must agree at lengths 1, 2, 63, 64 and 65 before a dense graph is assembled.
- **[UNKNOWN]** Exact user-GGUF metadata, tensor names/shapes, quant types, tokenizer arrays, numerical tolerances, real-checkpoint quality, throughput and peak bytes. Recheck model-card and pinned-source facts at implementation start.

## Dependencies

- `TASK-ash-native-families-index.md` owns S1–S6 ordering and family scope.
- `TASK-ash-decoder-ops.md` supplies RMSNorm, RoPE, SwiGLU, per-head QK norm and one-shot causal short convolution. It does not supply rolling convolution or recurrent state.
- `TASK-ash-autoregressive-state.md` supplies sequence-scoped KV, convolution-ring and matrix-state allocation/lifetime, append/rewind/copy/clear and isolation proofs.
- `TASK-ash-tokenizer-framework.md` is consumed for the `qwen35` profile: exact IDs, byte spans, special-token policy and supported chat rendering. This ticket adds no second tokenizer.
- The real 9B gate additionally consumes `TASK-ash-gguf-reader.md`, `TASK-ash-gguf-dequant.md`, existing provenance/license gates and `TASK-inference-cache-correctness.md` byte planning. Synthetic block proofs do not wait for weights.
- Existing Ash correctness, verification, selection and readiness tickets remain release dependencies; this ticket does not move their scope.

## Change

### 1. Metadata and bounded synthetic geometry

- Add a `qwen35` manifest adapter at the future `crates/ash-backend/src/llm/qwen35.rs`. Derive layer kinds, widths, heads, head dimensions, RoPE sections, convolution kernel/state sizes and required tensor roles from validated metadata; reject wrong architecture, missing roles, unsupported dtype, overflow and unresolved artifact identity before device/payload work.
- Use tiny deterministic dimensions for CI, including GQA, non-multiple/tail widths and both tied/untied output-head cases. A synthetic manifest may declare the researched 9B geometry for validation, but normal tests must not allocate a 9B model.
- Keep the 3:1 layer schedule data-driven. Do not infer architecture, readiness or quant support from a filename or marketing name.

### 2. Gated DeltaNet state proof

- Compose shared projections, short convolution, SiLU, L2-normalized Q/K, decay/delta recurrence, gated output normalization and output projection. Follow the pinned `delta-net-base` state orientation; do not transpose by guesswork.
- Prove one token updates convolution and matrix state exactly once. Compare token-by-token execution with an independent chunk reference at T=1,2,63,64,65 using the research recurrence and `q` scaled by `1/sqrt(D_k)`.
- Prove prefill/decode equivalence, context-boundary errors, cancellation between steps, save/restore, rewind/clear and two-sequence non-aliasing. Wrong state shapes and non-finite inputs fail before dispatch.

### 3. Gated-attention/KV block proof

- Compose joint Q/gate projection and split, per-head Q/K RMSNorm, text MRoPE, GQA K/V layout, causal masking, scaled softmax, sigmoid gate and output projection.
- Compare cached append-only decode with full recomputation for one/many heads and sequence lengths 1,2,17. Cover the context boundary, cancellation, wrong cache shape and a second sequence; no cross-request reuse.
- Reuse the shared attention, state and decoder ops. Add only the Qwen-specific composition and metadata mapping.

### 4. Dense motif and 9B text gate

- Assemble embedding, two pre-norm residuals per layer, three GDN layers plus one gated-attention layer, final RMSNorm and output head. Compare finite synthetic logits, shape and top-1 IDs with an independent scalar CPU reference.
- For f32 synthetic parity require finite outputs, relative L2 error `<= 1e-4` and max absolute error `<= 1e-3 * max(1, max_abs_reference)`; record actual errors. Real-checkpoint tolerances stay **[UNKNOWN]** until fixed with the supplied pinned reference, and an unset tolerance fails rather than inventing a pass.
- The ignored 9B test verifies every required artifact digest/license, acquires the exclusive single-device lease before device creation, loads one user Q4_K_M text artifact with capped context, and compares a fixed prompt prefill plus bounded greedy-token reference. It records logits shape/finite status/digest/top-1, token IDs, timings, planned demand, peak bytes, device/driver and source pins under its private output directory.
- Greedy token selection in this gate is a comparison mechanism, not a sampler implementation. Streaming, stop rules, tools and worker integration belong to `TASK-ash-sampling-contract.md`.

## Acceptance criteria

- [ ] CPU tests prove one GDN state step and token/chunk equivalence at all five boundary lengths, including state isolation and cancellation.
- [ ] CPU tests prove gated attention/KV append against full recomputation, GQA, causal masking, gate application and named shape/context errors.
- [ ] A tiny dense 3:1 motif produces finite CPU-reference logits and correct output-head geometry without external assets.
- [ ] The 9B manifest gate derives geometry from metadata and rejects wrong architecture, missing tensors, unsupported dtype, overflow and unresolved license/identity before allocation.
- [ ] The ignored real test verifies recorded SHA-256/license, uses one exclusive device lease and emits bounded comparison evidence; absence of a user artifact leaves real parity explicitly unmeasured, never passed.
- [ ] Normal CI contains no real weights, downloads, device lease, model execution or performance/availability claim.

## Verification

```sh
tmp="$(mktemp -d "$PWD/.tmp/ash-qwen35-dense.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
TMPDIR="$tmp" cargo test --locked -p honkbox-ash-backend \
  --features ash-llm qwen35_dense_proof
bun scripts/check-ash-provenance.ts

# Opt-in only; run alone on an idle GPU host, never in CI.
evidence="$(mktemp -d "$PWD/.tmp/ash-qwen35-real.XXXXXX")"
HONKBOX_ASH_QWEN35_MODEL=/read-only/qwen35-9b-q4_k_m.gguf \
HONKBOX_ASH_QWEN35_MODEL_SHA256=<recorded-sha256> \
HONKBOX_ASH_QWEN35_LICENSE=/read-only/qwen35-license.txt \
HONKBOX_ASH_QWEN35_REFERENCE=/read-only/qwen35-reference.json \
HONKBOX_ASH_QWEN35_DEVICE_INDEX=0 \
HONKBOX_ASH_QWEN35_DEVICE_LEASE="$evidence/device.lease" \
HONKBOX_ASH_QWEN35_EVIDENCE_OUT="$evidence/evidence.json" \
TMPDIR="$evidence" cargo test --locked -p honkbox-ash-backend \
  --features ash-llm --test qwen35_dense_real \
  -- --ignored --nocapture --test-threads=1
```

The reference manifest must name its generator command, source/build pin, artifact digests, fixed prompt, context, seed and declared tolerances. Normal tests run in parallel: each owns a unique temp directory and deterministic cleanup, with no shared globals or ordering dependence. Only the ignored hardware test serializes on its exclusive device lease. Archive `evidence.json` before removing `$evidence`; no command here was run during planning.

## Non-goals

Qwen3.6-35B-A3B MoE routing, Qwen3.8-27B execution, MTP/speculative decoding, vision/video/projectors, audio, 122B-class models, CPU-expert offload, expert streaming, multi-GPU, production token sampling/chat routes, new GGUF/tokenizer/cache/residency/registry mechanisms, automatic downloads/conversion, bundled weights/tokenizers, quality/throughput/VRAM claims or public Ash availability.
