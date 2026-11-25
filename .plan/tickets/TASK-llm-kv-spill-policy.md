# TASK: Paged KV spill policy (Refuse default, SpillRam opt-in)

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: a `KvSpill` policy enum defaults to current `Refuse` behavior, `SpillRam` is operator opt-in with logged spill events, and one test proves over-budget prompts succeed under spill and fail under refuse.
Verification: `cargo test --locked -p honkbox-llm-worker`
Links: `.tmp/research/cuda-capyflash-infr-2026-11-25.md` §9.3, `TASK-llm-prompt-cache.md`, `crates/llm-worker/src/lib.rs`
Resources: stub backends, mmap temp dirs; no GPU, no model downloads

## Attribution

Spill-ladder idea from `ValerioDolci/capy-flash` (Apache-2.0 — VRAM to RAM to disk context cache for oversized contexts). Method: white-room. Policy enum, spill accounting, and tests are honkbox-original against the existing `clamp_n_ctx` seam; no upstream code copied.

## Change

- Add `KvSpill` policy enum (`Refuse` preserves current behavior; `SpillRam` mmaps oldest KV pages to RAM). Default `Refuse`; `SpillRam` is operator opt-in with a logged spill event.
- Implementation order: policy enum plus `Refuse` path first (no-op, documents the seam) before any mmap code.
- Measure spill count and spill bytes in bounded metrics.
- Same test, two policies: prompt just over budget succeeds under `SpillRam`, fails under `Refuse`.

## Acceptance criteria

- [ ] Default behavior is byte-identical to today (Refuse path, clamp + named failure).
- [ ] Spill events are logged and metered; no silent disk use.
- [ ] No GPU, checkpoint, download, or shared global in any test.
