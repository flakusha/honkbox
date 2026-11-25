# FEAT: Ash embeddinggemma-2 text proof and gated Gemma 4 E4B decoder

Status: proposed
Priority: P1
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: synthetic non-causal embeddinggemma-2 geometry, pooling, projection and MRL match an independent CPU reference; an opt-in user-supplied text GGUF proves 768-dimensional embeddings; only after that evidence and resolved E4B contracts may the dense E4B text decoder phase begin.
Verification: `TMPDIR="$(mktemp -d "$PWD/.tmp/ash-gemma.XXXXXX")" cargo test --locked -p honkbox-ash-backend --features ash-llm --lib gemma_embedding -- --test-threads=1`, the exclusive-lease commands in §Verification, and `bun run spdx`.
Links: `.tmp/research/ash-gemma4-implementation-candidate-2026-11-25.md`; `crates/ash-backend/src/llm.rs`; `crates/llm-worker/src/embed.rs`; https://huggingface.co/google/embeddinggemma-2; https://gemma.ai/resources/apache-2/.
Resources: CPU-only synthetic fixtures in a unique `.tmp/ash-gemma.*` directory; opt-in real tests alone on one selected compute device under `.tmp/ash-real-device.lock`; read-only user GGUF/reference/license inputs with recorded SHA-256; no downloads, bundled weights or cache mutation.

## Verified 2026-11-25

- **[FACT]** `AshLlmPipeline` holds one device and every completion returns
  `LlmError::Load(NotYet)` without emitting a token
  (`crates/ash-backend/src/llm.rs:41-79`). Ash checkpoint loading is
  safetensors-only; there is no native GGUF, RMSNorm, RoPE, gated FFN, KV state
  or token sampler (`crates/ash-backend/src/checkpoint.rs:90-96,244-258`;
  `.tmp/research/ash-gemma4-implementation-candidate-2026-11-25.md` §4).
- **[FACT]** The current worker embedding seam already specifies fresh-context,
  mean-pooled, non-causal, BOS-prefixed and L2-normalized embeddings
  (`crates/llm-worker/src/embed.rs:65-119,122-140`). Reuse `EmbedPipeline`,
  `EmbedLoader` and the capacity-one cache; add no route or DTO.
- **[FACT]** The primary EmbeddingGemma 2 model card records Apache-2.0, 24
  layers, width 512, 4 query heads, 2 local/1 global KV heads, 5:1 local/global
  attention plus a final global layer, a 1024-token window, gated GELU FFN,
  mean pooling, 512→768 projection, 8,192-token context, and MRL dimensions
  128/256/512 below native 768. It requires BF16 or F32, not F16.
- **[FACT]** Current primary Gemma evidence is Apache-2.0, superseding older
  planning text that treated the Prohibited Use Policy as the model license.
  Redistribution, if separately approved, requires the license, modification
  notices, retained notices and applicable NOTICE propagation.
- **[UNKNOWN]** Real GGUF tensor names/orientations/quant mix; exact norm,
  input-scale, activation and RoPE constants; complete local/global arrays;
  pooling/projection order details; numerical tolerances; VRAM/latency; E4B
  PLE/shared-KV/p-RoPE semantics; supplier terms for any derivative. Resolve
  against a pinned permissive upstream revision and the admitted file, not by
  guessing from the family name.

## Dependencies

- `TASK-ash-native-families-index.md` owns ordering and readiness vocabulary.
- `TASK-ash-gguf-reader.md` and `TASK-ash-gguf-dequant.md` own the one strict
  reader, dtype admission and weight path. This ticket adds no Gemma parser.
- `TASK-ash-tokenizer-framework.md` owns GGUF vocabulary loading, byte/Unicode
  handling, specials and errors. Gemma vocabulary data is family-specific.
- `TASK-ash-decoder-ops.md` owns RMSNorm, RoPE, GQA/masks, gated FFN and shared
  output math. This ticket owns only Gemma graph order and tensor semantics.
- `TASK-ash-sampling-contract.md` is a feature dependency for the conditional
  E4B phase; the one-shot embedding proof neither implements nor invokes it.
- `TASK-ash-autoregressive-state.md` is required only before E4B prefill/decode.
- `TASK-ash-llm-evidence-integration.md` owns shared identity, provenance,
  single-device lease and evidence recording; both real tests consume it.
- Existing `TASK-ash-verification-gates.md`,
  `TASK-ash-family-readiness-markers.md`, `TASK-ash-backend-promote.md`,
  `TASK-model-provenance-contract.md` and `TASK-model-license-registry.md`
  remain the gate, marker, routing and admission owners.

## Change

### G1 — embeddinggemma-2 synthetic encoder and head

- Add a reduced `gemma-embedding2` graph, reusing shared ops and existing
  embedding gather/matmul/attention. Strictly admit 24 layers, width 512,
  4Q with 2-local/1-global KV, 5:1 plus final-global scheduling, the 1024 window,
  non-causal masks and supported dtypes; reject mismatches by name.
- Compare every block boundary and final states with a small independent CPU
  reference. Cover positions and mask boundaries 255/256/511/512/1023/1024,
  unequal Q/KV heads, padding exclusion and malformed schedules.
- Mean-pool valid final states, apply 512→768 projection, then L2-normalize.
  Prove MRL truncation to 128/256/512 and native 768, re-normalizing each
  prefix. Preserve input order and token counts. A zero vector follows the
  existing first-basis-vector policy and must not produce NaN.
- Use BF16 or F32 activations; refuse F16 for this model by name.

### G2 — first proof: text-only real embedding GGUF

- Add an ignored `gemma_embedding_real_gguf` integration test. Before device
  work, verify the supplied SHA-256, base/supplier/derivative identity,
  Apache-2.0 license/NOTICE evidence and one-device capacity.
- Use the shared GGUF reader/dequantizer/tokenizer and the existing
  `EmbedPipeline`. Compare identical token IDs and 768-dimensional vectors
  against an operator-supplied, provenance-recorded compatible CPU or
  llama.cpp reference within tolerances declared before the run.
- Cover short and medium text, Unicode/specials, an 8,192-token boundary case,
  repeat determinism, batch isolation and same-path replacement with a changed
  digest. Record device/driver, precision, quants, context, tolerances and peak
  bytes. Text only: reject media placeholders/projectors rather than dropping
  them. Keep the proof crate-level; no HTTP availability claim.

### G3 — conditional E4B dense text decoder

- Start only after G2 evidence is accepted, shared state/sampling dependencies
  pass, and the unknown E4B contracts are resolved. This phase is a
  reconsideration gate, not implied readiness from the embedding proof.
- Add a synthetic `gemma4` graph for 42 dense layers, width 2560, 8Q/2KV,
  512-token SWA, 5:1 plus final-global scheduling, 256-wide per-layer PLE,
  distinct local/global p-RoPE bases, an explicit shared-KV map for
  `shared_kv_layers=18`, and final-logit softcap 30.
- Prove SWA/global boundaries, PLE isolation, shared-KV aliasing, softcap
  saturation, greedy prefill/decode versus full-sequence CPU, EOS/stop and
  cancellation. Extend `AshLlmPipeline` only after the real graph completes.

### G4 — conditional E4B real text evidence

- Add ignored `gemma4_real_gguf` evidence using a separately supplied,
  hashed and admitted E4B text GGUF under the same exclusive single-device
  lease. Require exact greedy short-prompt token parity with a pinned,
  compatible reference using identical template/token IDs; record sampled
  output without treating it as parity. Start at 8K; 128K requires measurement.

## Acceptance criteria

- [ ] G1 proves the non-causal 24-layer schedule, hybrid local/global GQA,
  valid-token mean pooling, 512→768 projection, unit norm and all MRL prefixes
  against an independent CPU reference, including boundaries and bad metadata.
- [ ] G2 is the first real proof: one user-supplied text GGUF yields admitted,
  reproducible 768-dimensional embeddings with recorded digest/license,
  reference tolerances and exclusive single-device evidence.
- [ ] No family-local GGUF reader, tokenizer framework, KV runtime, sampler,
  cache, registry, residency or availability mechanism is added.
- [ ] G3/G4 remain blocked until G2 evidence and E4B unknowns are accepted;
  synthetic E4B and real greedy parity pass before `AshLlmPipeline` is extended.
- [ ] Results say “text-only embeddinggemma-2 proof” and, only when earned,
  “dense Gemma 4 E4B text proof”; no 26B, multimodal, quality, throughput or
  whole-family readiness claim is made.

## Verification

```sh
test_root="$(mktemp -d "$PWD/.tmp/ash-gemma.XXXXXX")"
trap 'rm -rf "$test_root"' EXIT

TMPDIR="$test_root" cargo test --locked -p honkbox-ash-backend \
  --features ash-llm --lib gemma_embedding -- --test-threads=1

# G3/G4 are conditional; run only after G2 evidence is accepted.
TMPDIR="$test_root" cargo test --locked -p honkbox-ash-backend \
  --features ash-llm --lib gemma4 -- --test-threads=1

# Serialize every real-checkpoint family test on the same repo-local lock.
exec 9>"$PWD/.tmp/ash-real-device.lock"
flock -x 9

HONKBOX_ASH_REAL_GGUF=/read-only/embeddinggemma-2.gguf \
HONKBOX_ASH_REAL_GGUF_SHA256=<recorded-sha256> \
HONKBOX_ASH_REAL_GGUF_LICENSE=/read-only/embeddinggemma-2-LICENSE \
HONKBOX_ASH_EMBED_REFERENCE=/read-only/embedding-reference.json \
HONKBOX_ASH_REFERENCE_SHA256=<recorded-reference-sha256> \
TMPDIR="$test_root" cargo test --locked -p honkbox-ash-backend \
  --features ash-llm --test gemma_embedding_real_gguf \
  -- --ignored --nocapture --test-threads=1

HONKBOX_ASH_REAL_GGUF=/read-only/gemma-4-E4B-it.gguf \
HONKBOX_ASH_REAL_GGUF_SHA256=<recorded-sha256> \
HONKBOX_ASH_REAL_GGUF_LICENSE=/read-only/gemma-4-E4B-LICENSE \
HONKBOX_ASH_CHAT_REFERENCE=/read-only/gemma4-reference.json \
HONKBOX_ASH_REFERENCE_SHA256=<recorded-reference-sha256> \
TMPDIR="$test_root" cargo test --locked -p honkbox-ash-backend \
  --features ash-llm --test gemma4_real_gguf \
  -- --ignored --nocapture --test-threads=1

bun run spdx
```

The ignored tests must fail closed without complete inputs, acquire the shared
RAII exclusive lease before creating exactly one `VulkanDevice`, verify hashes
before dispatch, and release the lease on every exit. Reference JSON must
record its generating tool/revision, model digest, template, token IDs,
sampler and expected values. No command here was run by the filing pass.

## Non-goals

Bundled/downloaded/converted weights or tokenizers; new dependencies; a second
parser/cache/residency/registry; multi-GPU; new routes/DTOs; prefix or
speculative decode; 26B-A4B/MoE; E4B vision/audio; F16 activation; benchmarks,
quality claims, or HTTP/capability promotion owned elsewhere.