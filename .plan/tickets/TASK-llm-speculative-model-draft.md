# TASK: Model-based speculative draft-verify for LLM worker

Status: proposed
Priority: P3
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: a default-off `SpeculativeLlm` wrapper pairs a draft and a target `LlmPipeline`, verifies in one target prefill, and emits draft-acceptance metrics; enablement requires measured acceptance on the operator model pair.
Verification: `cargo test --locked -p honkbox-llm-worker`
Links: `.tmp/research/cuda-strata-inference-2026-11-25.md` §7.2, `TASK-llm-speculative-decoding-spike.md`, `crates/llm-worker/src/lib.rs`
Resources: `StubLlm` pairs only; no GPU, no model downloads

## Attribution

Draft-verify pattern from `Niko1221/Strata` (MIT, MTP drafts) and `ValerioDolci/capy-flash` (Apache-2.0, MTP/DFlash2 in production); acceptance-rate-as-headline-metric from `izembaki-hash/HyperStream` (Apache-2.0, 65.7 pct reported). Method: white-room. The wrapper is implemented against the honkbox `LlmPipeline` trait; no upstream code copied. Follows `TASK-llm-speculative-decoding-spike.md`, which owns draftless ngram-mod and explicitly defers draft models — this ticket is that deferred half, still gated.

## Change

- Add `SpeculativeLlm` over two `LlmPipeline` impls (draft small, target large): draft N tokens, target verifies in one prefill, accept the prefix, resample at divergence.
- Emit draft-acceptance rate, mean accepted run length, and resample count from day one; ship behind a default-off flag.
- Deterministic tests with `StubLlm` pairs (fixed-text draft vs fixed-text target; assert accepted-prefix behavior and resample count).
- No change to the ngram-mod path; no default-on speculative decoding.

## Acceptance criteria

- [ ] Disabled by default; enabled only with documented measured acceptance on the operator pair.
- [ ] Metrics present for every speculative run; no silent fallback to unverified text.
- [ ] No GPU, checkpoint, download, or new dependency in any test.
