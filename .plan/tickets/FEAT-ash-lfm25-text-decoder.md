# FEAT: Ash LFM2.5 dense text decoder proof

Status: proposed
Priority: P1
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: a synthetic LFM2 decoder-block proof covers causal short convolution and GQA against CPU; only then may an opt-in user-supplied LFM2.5-VL-450M language GGUF prove text-only greedy parity on one leased device, without a projector or family-readiness claim.
Verification: `TMPDIR="$(mktemp -d "$PWD/.tmp/ash-lfm25.XXXXXX")" cargo test --locked -p honkbox-ash-backend --features ash-llm --lib lfm2_synthetic_decoder -- --test-threads=1`, the exclusive-lease real-GGUF command in §Verification, and `bun run spdx`.
Links: `.tmp/research/ash-lfm-implementation-candidate-2026-11-25.md`; `crates/ash-backend/src/llm.rs`; `crates/server/src/llm_attached.rs`; https://huggingface.co/LiquidAI/LFM2.5-VL-450M; https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE.
Resources: deterministic CPU fixtures in a unique `.tmp/ash-lfm25.*` directory; opt-in real test alone on one selected compute device under `.tmp/ash-real-device.lock`; read-only user GGUF/reference/license inputs with recorded SHA-256; no download, conversion, mmproj or bundled assets.

## Verified 2026-11-25

- **[FACT]** `AshLlmPipeline` still returns `LlmError::Load(NotYet)` for every
  completion and emits no token (`crates/ash-backend/src/llm.rs:41-79`).
  Attached llama.cpp remains the production text baseline; Ash has no server
  routing arm (`crates/server/src/llm_attached.rs:48-141`;
  `.tmp/research/ash-lfm-implementation-candidate-2026-11-25.md` §4).
- **[FACT]** Ash currently has safetensors checkpoint loading, multi-head
  attention with additive masks, CLIP byte-BPE and DDIM `SamplingStep`, but no
  strict GGUF reader, persistent quantized weights, RMSNorm, RoPE, SwiGLU, KV
  cache, recurrent convolution state or token sampler. SD1.5 execution does not
  establish LLM readiness.
- **[FACT]** The primary LFM2.5-VL-450M card identifies an LFM2.5-350M language
  backbone plus SigLIP2 NaFlex. The text backbone has 16 layers: 10
  double-gated convolution blocks and 6 GQA blocks, vocabulary 65,536 and a
  32,768-token model-card context. This ticket uses only its language GGUF.
- **[FACT]** Audited local-header evidence gives width 1024, FFN 4608, 16 query
  heads, six 8-KV-head attention layers, head dimension 64, `L_cache=3`,
  RMS epsilon `1e-5`, RoPE base `1e6`, BOS 1 and EOS 7
  (`.tmp/research/ash-lfm-implementation-candidate-2026-11-25.md` §5-6).
  The local GGUF header/card context conflict remains unresolved; cap this
  proof at 8,192 tokens.
- **[FACT]** The pinned reference implements `in_proj → B/C/X`,
  `u=transpose(B⊙X)`, two-token causal state, depth-3 depthwise convolution,
  `C⊙conv(u)` and `out_proj`
  (`vendor/llama-cpp-sys-4-0.11.0/llama.cpp/src/models/lfm-base.cpp:9-83`).
  Its dense FFN is bias-free `down(silu(up(x)) * gate(x))`
  (`vendor/llama-cpp-sys-4-0.11.0/llama.cpp/src/models/lfm2.cpp:94-129`).
- **[FACT]** LFM Open License v1.0 permits redistribution subject to notices,
  but §5 conditions Commercial Use on annual revenue not exceeding its
  Threshold, defined as US$10,000,000 or more; use above it is not licensed.
  Local execution of an unredistributed user file does not itself trigger §4
  redistribution duties, while admission, provenance and product policy still
  apply. The owner must resolve eligibility and derivative/safety posture.
- **[UNKNOWN]** Exact supplied-file dtype/shape inventory, template rendering,
  legal eligibility, real VRAM/throughput/tolerances and whether the 8K cap is
  sufficient. A Q4_K_M filename does not prove every tensor is Q4_K_M.

## Dependencies

- `TASK-ash-native-families-index.md` owns ordering and readiness vocabulary.
- `TASK-ash-gguf-reader.md` and `TASK-ash-gguf-dequant.md` own strict metadata,
  bounded reads, every admitted dtype and persistent weights; no LFM parser.
- `TASK-ash-tokenizer-framework.md` owns byte-BPE and special-token mechanics;
  the `gpt2` byte-BPE, `pre=lfm2`, llama3-style Unicode split, BOS and no space
  cleanup are the LFM profile, not a second tokenizer framework.
- `TASK-ash-decoder-ops.md` owns RMSNorm, RoPE, GQA, SwiGLU and shared
  arithmetic. This ticket owns LFM block order, short-convolution semantics and
  tensor/schedule mapping only.
- `TASK-ash-autoregressive-state.md` owns allocation, append/reset/lifetime and
  accounting shared by six KV caches and ten convolution states.
- `TASK-ash-sampling-contract.md` owns greedy/seeded temperature, top-k,
  min-p, repeat penalty, stop handling and UTF-8 streaming; no LFM sampler.
- `TASK-ash-llm-evidence-integration.md` owns shared identity/provenance,
  single-device lease and evidence recording.
- Existing `TASK-ash-verification-gates.md`,
  `TASK-ash-family-readiness-markers.md`, `TASK-ash-backend-promote.md`,
  `TASK-model-provenance-contract.md` and `TASK-model-license-registry.md`
  remain the gate, marker, routing and admission owners.

## Change

### L1 — strict synthetic `lfm2` spec and first decoder-block proof

- Admit only dense `general.architecture=lfm2` text with the verified
  16-layer 10-convolution/6-GQA schedule and expected tensors, including
  `token_embd_norm`; reject `lfm2moe`, decision/ColBERT heads, SWA, non-causal
  attention, unsupported dtypes and shape/schedule mismatches by name.
- Build a reduced CPU-reference graph that records every block boundary. Prove
  the short-convolution equation and a GQA block using shared ops: Q/K/V widths
  `[1024,512,512]`, per-head Q/K RMS norm, RoPE base `1e6`, causal 16Q/8KV
  attention, `wo`, residual, FFN norm and bias-free SwiGLU residual.
- Prove token-by-token, chunked and full-prompt short-convolution equivalence,
  zero-state reset, rollback, boundary tokens and no future-token leakage.
  Reuse the shared KV runtime; do not implement a family cache.

### L2 — tokenizer, hybrid state and honest pipeline

- Consume the shared tokenizer with an LFM profile. Prove exact ASCII,
  Unicode, contraction, digit, CRLF, whitespace, unknown and special-token IDs
  plus byte round trips against the pinned reference. Render only the
  tool-free ChatML-like template embedded in the supplied GGUF; reject tools,
  grammar, JSON constraints and images with named errors.
- Compose the 16-layer graph with shared hybrid state: six KV caches, ten
  two-token convolution states, shared positions, final RMS norm and explicit
  or tied output head. Account separately for weights, KV, convolution state,
  logits/scratch and headroom; cap context at 8,192.
- Complete `AshLlmPipeline` only when it can finish honestly: greedy or seeded
  supported sampling, EOS/max-token stop, usage/timing, valid UTF-8 streaming,
  `RequestCancel → Cancelled`, and `InvalidRequest` for unsupported knobs.
  Sequential requests must not leak state.

### L3 — user-supplied 450M language GGUF real gate

- Add ignored `lfm2_real_gguf`. Accept a text-only LFM2.5-VL-450M derivative
  language GGUF, not an `lfm3` or MoE file. Require base model, supplier,
  derivative/abliteration status, license link/notices, eligibility decision,
  path and SHA-256 before device work. Missing evidence denies admission.
- Acquire the shared exclusive single-device lease, verify the digest and
  strict metadata/tensor/dtype inventory, then run short text prompts at 8K or
  less. Require exact greedy token parity with provenance-recorded reference
  tokens from the same checkpoint, template and sampler on a pinned compatible
  llama.cpp/CPU build. Record sampled runs without parity claims.
- Cover Unicode/specials, repeat determinism, stop/cancel, sequential-request
  isolation, and same-path replacement with a changed digest. Record
  device/driver, reference revision, quants, context, sampler, tolerances and
  peak bytes. Never load or silently drop an `mmproj` or image input.

## Acceptance criteria

- [ ] L1 is the first proof: synthetic short-convolution and GQA blocks match
  an independent CPU reference at every boundary, including causal state,
  chunk/full-prompt equivalence, reset/rollback and malformed metadata.
- [ ] L2 uses the shared tokenizer, decoder ops, state runtime and sampler;
  exact IDs/round trips, 8K accounting, stop/cancel and sequential isolation
  pass without a family-specific framework.
- [ ] L3 proves text-only greedy parity on one admitted, hashed user-supplied
  450M language GGUF under the exclusive lease; license/provenance and reference
  revision are recorded, and changed-path/digest input fails before dispatch.
- [ ] No GGUF parser, tokenizer framework, cache, residency system, registry,
  availability path, sampler, projector or multimodal route is duplicated.
- [ ] Evidence says “experimental LFM2.5 dense text proof”; it does not claim
  LFM2.5 vision, 1.6B, LFM3, production routing, quality or throughput readiness.

## Verification

```sh
test_root="$(mktemp -d "$PWD/.tmp/ash-lfm25.XXXXXX")"
trap 'rm -rf "$test_root"' EXIT

TMPDIR="$test_root" cargo test --locked -p honkbox-ash-backend \
  --features ash-llm --lib lfm2_synthetic_decoder -- --test-threads=1
TMPDIR="$test_root" cargo test --locked -p honkbox-ash-backend \
  --features ash-llm --lib lfm2_pipeline -- --test-threads=1

exec 9>"$PWD/.tmp/ash-real-device.lock"
flock -x 9

HONKBOX_ASH_REAL_GGUF=/read-only/LFM2.5-VL-450M-text.gguf \
HONKBOX_ASH_REAL_GGUF_SHA256=<recorded-sha256> \
HONKBOX_ASH_REAL_GGUF_LICENSE=/read-only/LFM-Open-License-v1.0 \
HONKBOX_ASH_REAL_GGUF_PROVENANCE=/read-only/lfm25-provenance.json \
HONKBOX_ASH_PROVENANCE_SHA256=<recorded-provenance-sha256> \
HONKBOX_ASH_REFERENCE_TOKENS=/read-only/lfm25-reference-tokens.json \
HONKBOX_ASH_REFERENCE_SHA256=<recorded-reference-sha256> \
TMPDIR="$test_root" cargo test --locked -p honkbox-ash-backend \
  --features ash-llm --test lfm2_real_gguf \
  -- --ignored --nocapture --test-threads=1

bun run spdx
```

The ignored test must fail closed without complete inputs, acquire the shared
RAII exclusive lease before creating exactly one `VulkanDevice`, verify hashes
before dispatch, and release the lease on every exit. The reference-token JSON
must record its generating tool/revision, model digest, exact rendered prompt,
token IDs, greedy sampler and expected output IDs. No command here was run by
the filing pass.

## Non-goals

Bundled/downloaded/converted weights, tokenizers or projectors; new
dependencies; a second parser/cache/residency/registry; multi-GPU or batching;
mmproj/SigLIP2/NaFlex, image/video/audio/speech/codecs; MoE, decision heads,
DSpark, prefix reuse, grammar/JSON/tools; 1.6B or production routing; benchmarks
or quality claims.

## Conditional follow-up

LFM3 text is not implemented here. It remains conditional on this proof,
shared SWA/full-attention state and a separate reconsideration of its 24-layer
7-convolution/17-GQA graph, `L_cache=7`, optional sandwich norms, head gate,
mixed full/SWA RoPE and non-causal full-attention prompt pass. LFM3 Omni remains
do-not-start until its text, SWA, modality, audio/codec and license gates pass.