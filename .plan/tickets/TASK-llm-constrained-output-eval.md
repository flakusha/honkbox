# TASK: Constrained-output eval with checked-in goldens

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: a fixed prompt set with `json_schema`-forced outputs hashes parsed JSON against checked-in goldens on CPU stub and real backend, with sampler-extremes validity proven.
Verification: `cargo test --locked -p honkbox-llm-worker`
Links: `.tmp/research/cuda-strata-inference-2026-11-25.md` §7.5, `.tmp/research/cuda-capyflash-infr-2026-11-25.md` §9.5, `crates/llm-worker/src/backend.rs`
Resources: no GPU, no model downloads, no GPU CI job

## Attribution

Parity-by-benchmark practice from `Niko1221/Strata` (MIT, GSM8K/MMLU-Pro parity claims) and `ValerioDolci/capy-flash` (Apache-2.0, vs-vLLM reference comparisons); structured-output precedent from `ValerioDolci/capy-flash` xgrammar vendoring, matched by the honkbox GBNF path already in `backend.rs`. Method: white-room. Eval harness and goldens are honkbox-original; no upstream code or prompt set copied.

## Change

- Fixed prompt set with `json_schema`-forced outputs (existing GBNF path); hash the parsed JSON; compare against checked-in goldens. Temperature 0 for eval.
- Prove constrained outputs stay valid under sampler-parameter extremes (temperature/top-p/top-k combined with grammar), since eval determinism depends on it. One focused test, not a feature.
- Runs on CPU stub and real backend; no GPU CI job needed.

## Acceptance criteria

- [ ] Goldens match on stub and real backend; mismatch names the prompt and field.
- [ ] Sampler-extremes test proves grammar validity is preserved.
- [ ] No GPU, checkpoint, download, or network in any test.
