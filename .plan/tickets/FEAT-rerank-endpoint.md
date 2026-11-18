# FEAT: rerank endpoint

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: `POST /rerank` (and `/v1/rerank`) scores documents against a query in Jina/TEI flavor with `top_n` support and advertises `rerank` in `/v1/capabilities`; loop-lore `memory/rerank.ts` can point at it unchanged.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run codegen:check`
Links: llama.cpp `tools/server/README.md` L778-851 (contract); llama-swap README (`/v1/rerank` proxied, `capabilities.reranker` discovery); loop-lore `src/memory/rerank.ts` (consumer shape)
Resources: per-test temp model dirs; no GPU (stub ranker returns canned scores)

## Change

- New DTOs in `crates/api-types`: `query`, `texts[]` (TEI → bare `{index, score}`, `+text` with `return_text`) or `documents[]` (Jina → `{results: [{index, relevance_score}]}`), `top_n`, `model`; usage passthrough.
- Multimodal items (`text` + `image`) accepted structurally; unsupported models get a named 501.
- `FeatureFlags.rerank: bool` so capability discovery works without probing; `/v1/models` entries mark reranker-capable models.

## Acceptance detail

Tests cover TEI and Jina shapes, `top_n` truncation, and the named 501 without a ranker model. No embedding-model fallback inside this endpoint.
