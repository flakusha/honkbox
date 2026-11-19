# EPIC: LLM surface expansion

Status: proposed
Priority: P1
Owner: unassigned
Research: `.tmp/research/ReviewedSupport-LLM.md`

## Scope

Refine the already-implemented OpenAI-shaped chat/completions surface: make
`llm-vision` an honest default after CI parity, add bounded prompt/KV cache
controls, and remove silent tool-choice surprises. Retrieval/rerank stays in
`FEAT-rag-pipeline.md`; no MCP surface.

## Tickets

- `TASK-llm-vision-default.md` — feature default + stale docs (P0).
- `TASK-llm-prompt-cache.md` — bounded per-slug KV cache knobs (P1, done).
- `TASK-llm-tool-choice-contract.md` — document/align tool-choice fidelity (P2).
- `TASK-remote-vision-url-fetch.md` — opt-in SSRF-safe image URL fetch (P2).
- `TASK-llm-speculative-decoding-spike.md` — draftless `ngram-mod` spike (P2).

## Acceptance

Vision advertising matches the linked build and discovered mmproj files; cache
knobs cannot exceed `[models] vram_budget_bytes`; tool-choice behavior is
machine-readable rather than silently clamped.

## Verification

`bun run ci` matrix (Linux Vulkan + Windows/macOS CPU), `bun run test:server`,
`bun run codegen:check`.

## Non-goals

Responses API, reasoning parameters, and parallel tool calls unless a named
consumer requests them. Remote vision URLs and speculative decoding now have
spike tickets above; draft models stay deferred.
