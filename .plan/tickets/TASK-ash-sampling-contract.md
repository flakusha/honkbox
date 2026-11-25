# TASK: Add Ash sampling and stop contract

Status: proposed
Priority: P1
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: Ash consumes shared autoregressive state through the existing `LlmPipeline` seam with deterministic greedy sampling, reproducible seeded temperature/top-k/top-p, EOS/max-token/context stop rules, explicit unsupported tool/grammar policy, and cancellable streaming that never returns partial output as success.
Verification: `cargo test --locked -p honkbox-ash-backend --features ash-llm ash_sampling`; `cargo test --locked -p honkbox-llm-worker sampler`; `cargo test --locked -p honkbox-server openai_constraints`; `bun run codegen:check`.
Links: `.tmp/research/ash-family-implementation-index-2026-11-25.md` §4 S6 and §5; `.tmp/research/ash-qwen35plus-implementation-candidate-2026-11-25.md` §§4–5; `.tmp/research/ash-lfm-implementation-candidate-2026-11-25.md` §§5–6; `crates/ash-backend/src/llm.rs`; `crates/llm-worker/src/lib.rs`; `crates/llm-worker/src/backend.rs`.
Resources: synthetic logits and token IDs only; unique `.tmp/ash-sampling.*` test directories with deterministic teardown; no GPU, checkpoint, download, fixed port, database or global sampler. Consuming-family real-checkpoint tests remain opt-in and must hold an exclusive single-device lease with recorded SHA-256 and license.

## Verified 2026-11-25

- **[FACT]** `AshLlmPipeline` currently returns `LlmError::Load(NotYet)` from every
  verb and never invokes a stream sink (`crates/ash-backend/src/llm.rs:29-79`).
- **[FACT]** `Op::SamplingStep` is DDIM, not token sampling
  (`crates/ash-backend/src/model.rs:15-44`).
- **[FACT]** The worker already carries temperature, seed, top-k and extended
  sampler knobs, plus `RequestCancel` and `complete_stream_cancellable`. Its
  contract says cancellation returns `LlmError::Cancelled`, never a partial reply
  as `Ok` (`crates/llm-worker/src/lib.rs:127-183,381-457`).
- **[FACT]** Neither `SamplerParams` nor `OpenAiChatRequest` currently has
  `top_p`; the OpenAI route documents `top_p` and `stop` as accepted and ignored
  (`crates/api-types/src/lib.rs:1028-1118`;
  `crates/server/src/routes/openai_chat.rs:69-75`). The vendored in-process llama
  binding already exposes `LlamaSampler::top_p`.
- **[FACT]** Index S6 requires token sampling, stop rules, tool/grammar policy and
  worker-seam cancellation after S5. Qwen3.5 calls for greedy first and then seeded
  temperature/top-k/top-p; LFM2.5 calls for EOS/max-token stopping and honest
  unsupported-knob errors.
- **[UNKNOWN]** Real-checkpoint token parity and sampled quality are unproven.
  This ticket makes no model availability, quality or performance claim.

## Dependencies

- `TASK-ash-autoregressive-state.md` — required source of committed logits,
  positions, EOS-aware state lifetime and cancellation boundaries.
- `TASK-ash-tokenizer-framework.md` — required EOS/special-token metadata and
  UTF-8-safe detokenization; no second tokenizer convention.
- `TASK-ash-llm-evidence-integration.md` — downstream owner of real-checkpoint
  parity evidence; this ticket uses synthetic logits only.
- `FEAT-ash-qwen35-dense.md` and `FEAT-ash-lfm25-text-decoder.md` — consumers that
  provide family logits and admission, not sampler implementations.

## Change

- Add one shared Ash token sampler over committed logits. Validate finite
  temperature, `0 < top_p <= 1`, integer top-k and at least one finite logit;
  reject NaN/positive infinity and preserve `-infinity` masks. Temperature zero is
  greedy with lowest-token-id tie-breaking and ignores the seed.
- For stochastic sampling, apply top-k to logits, compute the pre-temperature
  nucleus and retain the smallest probability-sorted prefix reaching top-p, then
  apply temperature and categorical sampling. Use an original dependency-free
  SplitMix64 stream from the request seed; `Some(seed)` is reproducible and `None`
  follows the worker's existing entropy-seeded contract. Golden tests pin the
  algorithm, tie-breaking and token sequence.
- Add `top_p` to the existing `SamplerParams` and OpenAI DTO/conversion. Wire the
  in-process llama.cpp sampler through its existing binding so the shared field
  is not silently ignored. The attached llama-server adapter must reject `top_p`
  locally rather than forwarding or pretending to honor it; no hosted-server
  protocol or process behavior changes.
- Reject unsupported Ash sampler knobs by name with `LlmError::InvalidRequest`
  before prefill: min-p, typical-p, repetition penalty, XTC, DRY and dynamic
  temperature. Do not silently drop them or inherit llama.cpp defaults.
- Stop on the tokenizer's EOS set, `max_tokens` (existing default 512), or context
  overflow. EOS is not emitted or counted as generated text; report prompt and
  generated-token accounting only for a completed response. Context overflow fails
  before generation with `LlmError::Context`.
- Define the initial Ash tool/grammar policy explicitly: `grammar`, `json_schema`,
  `json_object` and `tools_json` are unsupported and return
  `LlmError::InvalidRequest` naming the field before state or device work. No
  unconstrained fallback, fake tool call or ignored constraint is permitted.
- Implement `complete_stream_cancellable` on `AshLlmPipeline`; check cancellation
  before prefill, before each sample, before each sink call and after decode.
  Emit only UTF-8-safe decoded pieces. Cancellation drops request state, returns
  `LlmError::Cancelled`, emits no final success response/SSE terminal event, and
  never presents already-streamed pieces as a successful partial reply.
- Keep greedy and seeded streaming/non-streaming paths token-identical for the
  same synthetic input. No family graph, second inference cache or speculative
  drafting is part of this ticket.

## Acceptance criteria

- [ ] Greedy fixtures prove deterministic argmax and token-id tie-breaking;
      seeded temperature/top-k/top-p fixtures prove exact repeatable token IDs.
- [ ] Boundary tests cover temperature zero, disabled top-k, top-p 1, one-token
      nuclei, ties, all-masked logits, NaN/infinity and invalid parameter ranges.
- [ ] EOS, max-token and context-overflow tests prove correct stop point and
      accounting; EOS never appears in generated text.
- [ ] Unsupported sampler knobs and tool/grammar constraints fail by name before
      prefill; no request field is silently ignored.
- [ ] Streaming and non-streaming calls produce identical tokens for the same
      seed; cancellation before first emission leaves the sink empty, and
      mid-stream cancellation returns `Cancelled` with no success response.
- [ ] `top_p` crosses the worker/OpenAI seam and passes codegen; attached
      llama-server behavior is unchanged apart from an honest local rejection.
- [ ] No family-specific model, real weights, default speculative decoding,
      response/prefix cache, multi-GPU path or hosted llama-server change exists.

## Verification

```sh
mkdir -p .tmp
sampling_tmp="$(mktemp -d "$PWD/.tmp/ash-sampling.XXXXXX")"
trap 'rm -rf "$sampling_tmp"' EXIT
mkdir -p "$sampling_tmp/ash" "$sampling_tmp/worker" "$sampling_tmp/server"
TMPDIR="$sampling_tmp/ash" \
  cargo test --locked -p honkbox-ash-backend --features ash-llm ash_sampling
TMPDIR="$sampling_tmp/worker" \
  cargo test --locked -p honkbox-llm-worker sampler
TMPDIR="$sampling_tmp/server" \
  cargo test --locked -p honkbox-server openai_constraints
bun run codegen:check
```

All tests use generated logits/token fixtures and unique per-test child
directories. No command is run by the filing pass.

## Non-goals

Qwen/LFM/model graphs or tensor maps; GGUF loading; real-checkpoint parity;
user stop-string matching; grammar or tool-call execution; min-p/typical/repeat/
XTC/DRY/dynamic-temperature support; default speculative decoding; a second
inference/response/prefix cache; multi-GPU; hosted llama-server changes; quality,
latency or availability claims.
