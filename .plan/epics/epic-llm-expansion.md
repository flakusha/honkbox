# EPIC: LLM surface expansion

Status: proposed
Priority: P1
Owner: unassigned
Research: `.tmp/research/llm-runtime-memory-2026-11-25.md`,
`.tmp/research/llm-support-local-roster-2026-11-24.md`,
`.tmp/research/llama-swap-loop-lore-gaps-2026-11-22.md`,
`.tmp/research/ash-family-implementation-index-2026-11-25.md`

## Scope

Refine the already-implemented OpenAI-shaped chat/completions surface: make
cache identity and memory demand correct before persistent prompt-state reuse.
Vision defaulting and tool-choice fidelity are landed. Native Ash LLM families
belong to `epic-ash-native-families.md`; retrieval/rerank stays in
`FEAT-rag-pipeline.md`, and this epic adds no MCP surface.

## Ordered dependencies

1. `TASK-inference-cache-correctness.md` (P0, open): verified weight +
   tokenizer/template/projector identity before cache lookup, safe same-path
   replacement, and weight/working-memory split on the existing ledger.
2. `TASK-llm-vision-default.md` (P0, done): advertising matches the linked
   build and discovered mmproj bytes.
3. `TASK-llm-tool-choice-contract.md` (P2, done): tool-choice fidelity is
   documented and aligned.
4. `TASK-remote-vision-url-fetch.md` (P2): opt-in SSRF-safe image URL fetch.
5. Persistent prefix/KV reuse follow-up to `TASK-llm-prompt-cache.md` (P2, not
   filed): depends on item 1 plus measured long-prefix workloads and bounded KV
   eviction. The done prompt-cache ticket is not evidence that this exists.
6. `TASK-llm-speculative-decoding-spike.md` (P2): draftless `ngram-mod` spike;
   promotion depends on measured prefill/decode behavior, not assumed wins.

## Tickets

- `TASK-inference-cache-correctness.md` — identity, replacement safety, and
  demand split (P0, proposed).
- `TASK-llm-vision-default.md` — feature default + stale docs (P0, done).
- `TASK-llm-prompt-cache.md` — bounded per-slug context/KV knobs and resident
  hit/miss metrics (P1, done); no cross-request KV/prefix cache shipped.
- `TASK-llm-tool-choice-contract.md` — documented tool-choice fidelity (P2, done).
- `TASK-remote-vision-url-fetch.md` — opt-in SSRF-safe image URL fetch (P2).
- `TASK-llm-speculative-decoding-spike.md` — draftless `ngram-mod` spike (P2).
- `FEAT-llama-server-client-backend.md` — opt-in backend for an attached,
  user-installed `llama-server` (landed 2026-11-22).
- `TASK-negative-prompt-preset-library.md` — versioned prompt/style presets
  served from the model registry.

## Current status (2026-11-25)

- [FACT] Bounded context/KV controls and the attached llama-server backend are
  landed. Chat and embedding requests still create fresh contexts; there is no
  cross-request prefix cache.
- [FACT] `SwapManager`/`EmbedCache` still key by slug, and the inline image
  response cache does not include the resolved checkpoint digest. Same-path
  replacement correctness is open.
- [FACT] The residency ledger already has byte/thread demand fields, but LLM
  routes currently claim a slot with zero bytes. The KV clamp covers KV only,
  not weights or total working memory.
- [FACT] Native Ash LLM families and their shared GGUF/decoder/state/sampling
  substrate are owned by `epic-ash-native-families.md`. Its autoregressive
  state is pipeline KV/recurrent state, not a second response cache.
- [UNKNOWN] No honkbox prefix-cache hit rate, memory saving, or latency benefit
  has been measured.
- This planning pass ran no tests, builds, linters, or git commands.

## Acceptance

Vision advertising matches the linked build and discovered mmproj identity;
cache hits require verified model/tokenizer/projector identity; weight and
working bytes are separately accounted within configured budgets; tool-choice
behavior is machine-readable rather than silently clamped. Persistent prefix
reuse is not accepted until it has bounded memory and measured prefill benefit.

## Verification

Run the exact per-ticket commands first, then `bun run ci` matrix (Linux Vulkan +
Windows/macOS CPU), `bun run test:server`, and `bun run codegen:check`. Use
gitignored `.tmp/` scratch dirs, test-local ledgers, and port 0 or in-process
routers; isolate any real-GGUF/GPU run to one selected device.

## Non-goals

Responses API, reasoning parameters, and parallel tool calls unless a named
consumer requests them. Remote vision URLs and speculative decoding have spike
tickets above; draft models stay deferred. No second inference/response cache,
no public fast-path flag before evidence, and no cross-backend byte-equality
claim.
