# TASK: Split LLM load-plan from run (admit before load)

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: a `LlmLoadPlan` admits to a `Lease` before load, `complete_stream` takes only the request, and over-budget plans are refused before any load with the error named.
Verification: `cargo test --locked -p honkbox-llm-worker`
Links: `.tmp/research/cuda-capyflash-infr-2026-11-25.md` §9.1, `TASK-multi-gpu-device-axis.md`, `crates/llm-worker/src/lib.rs`, `crates/residency/src/lib.rs`
Resources: `StubLlm` only; no GPU, no model downloads

## Attribution

Plan/recipe split from `kryptic-sh/infr` (MIT — one `Backend` trait per model plus a per-arch recipe in `docs/plan.md`). Method: white-room. The `LlmLoadPlan` struct and admit-before-load ordering are honkbox-original, against the existing `LlmPipeline` seam; no upstream code copied. Device affinity defers to `TASK-multi-gpu-device-axis.md`; this ticket adds no device semantics.

## Change

- Add `LlmLoadPlan` (model path, quant, n_ctx budget, offload tiers, mmproj optional) with `admit(plan)` to `Lease` resolved before load.
- `complete_stream` takes only the request; the contention policy keys on the plan, not the backend.
- Plan-admission unit tests with `StubLlm`: over-budget refused before load, exact error named.

## Acceptance criteria

- [ ] No load path bypasses admission; refusal precedes any allocation.
- [ ] llama.cpp, stub, and any future backend share the one admission path.
- [ ] No GPU, checkpoint, download, or new dependency in any test.
