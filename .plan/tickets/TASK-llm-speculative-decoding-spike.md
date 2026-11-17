# TASK: Speculative decoding spike (ngram-mod)

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: opt-in draftless `ngram-mod` runs per slug with bounded memory, reports `draft_n`/`draft_n_accepted`, and adds no draft-model VRAM.
Verification: `bun run test:server`, `bun run lint:rs`; local model fixture only
Links: `.tmp/research/ReviewedExpansion-SpeculativeDecoding.md` §Recommended rollout
Resources: per-test model/slot dirs; no shared KV cache or CI downloads

## Change

- Expose `spec_type: none|ngram-mod` in config/DTO; keep default `none`.
- Assert no draft model is loaded and memory stays near the documented ~16 MB
  shared pool; do not exceed `[models] vram_budget_bytes`.
- Surface acceptance counters in bounded `kind` metrics and `/v1/capabilities`;
  measure repetitive and code-like prompts before considering draft models.

## Acceptance detail

No new route, no semantic output claim, and no byte-determinism promise. Draft
models and target-specific families (EAGLE-3/DFlash) remain deferred.
