# TASK: Vision-ingest tested contract (fixture + mmproj-less refusal)

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: a fixture test proves image + mmproj + question through `LlamaCppLlm` where vision is compiled, and a mmproj-less model refusal names the reason.
Verification: `cargo test --locked -p honkbox-llm-worker`
Links: `.tmp/research/cuda-strata-inference-2026-11-25.md` §7.3, `TASK-llm-vision-default.md`, `crates/llm-worker/src/lib.rs`, `crates/llm-worker/src/backend.rs`
Resources: tiny image fixture; no GPU, no model downloads

## Attribution

Requirement pattern from `Niko1221/Strata` (MIT), which ships optional image input on its localhost API with an explicit unsupported-model refusal. Method: white-room. No upstream code involved; the seam (`ChatMessage.images`, mmproj marker splice, `LlmError::Vision`) already exists in honkbox.

## Change

- Add a fixture test: tiny image + mmproj + canned question through `LlamaCppLlm` where the `vision` feature is compiled.
- Promote the mmproj-less 501-with-reason to a tested contract: text-only model plus image attachment fails with the reason named, never silent text.
- No new seam, no route change, no fetch behavior change (data-URI-only contract unchanged).

## Acceptance criteria

- [ ] Fixture test passes where `vision` is compiled and skips honestly where it is not.
- [ ] mmproj-less refusal names the model limitation; no vision inference is faked by the stub beyond its existing emulation.
- [ ] No GPU, checkpoint, download, or network fetch in any test.
