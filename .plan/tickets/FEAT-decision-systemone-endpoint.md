# FEAT: decision-model systemone endpoint

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: `POST /v1/systemone` answers typed choice/score/noul questions with zero generated text and advertises `decisions` in `/v1/capabilities`; non-decision models get a named 501, never a silent fallback.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run codegen:check`
Links: llama.cpp `tools/server/README.md` L1911-2051 (contract) + PR #29818; llama-swap v262 + PR #1197 (proxy passes it through unchanged); loop-lore `TASK-llama-cpp-decision-model-client-for-aux-classification` (consumer: per-question timeout + graceful null)
Resources: per-test temp model dirs; no GPU (stub pipeline answers canned decisions or 501)

## Change

- New DTOs in `crates/api-types`: `state` (string | object | chat-message array), `questions{id: choice{instructions, criteria map} | score{instructions, level descs low→high} | noul{...}}`; response `answers{id: choice{choice, probabilities, confidence} | score{...} | noul}`, `usage{input_tokens, output_tokens: 0}`.
- Option caps per model family (52 openjev-style, up to 255 laya-style per llama.cpp); document that probabilities are temperature-scaled and uncalibrated — cutoffs belong per model, not in the server.
- Errors: 400 invalid request (incl. video input — unsupported upstream), 501 non-decision model or image input without projector. Both are terminal for clients (no retry on same model).
- `FeatureFlags.decisions: bool` + per-model modality advertisement so llama.cpp-style discovery (`output_modalities: ["decisions"]`) works without probing routes.

## Acceptance detail

Unit tests cover choice/score/noul parsing, 501/400 degradation, and option-cap rejection; a stub-pipeline test pins `output_tokens: 0`. No generative fallback inside this endpoint — fallback lives in the caller.
