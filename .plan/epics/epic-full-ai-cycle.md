# EPIC: Full AI cycle

Status: proposed
Priority: P1
Owner: unassigned
Research: `.tmp/research/llm-runtime-memory-2026-11-25.md`,
`.tmp/research/plan-research-promotion-index-2026-11-25.md`,
`.tmp/research/llama-swap-loop-lore-gaps-2026-11-22.md`

## Scope

Foundation first — resource ledger/bounded queues, the determinism contract,
correct inference identity/demand, and measured native-runtime behavior — then
OCR ingest, RAG, ComfyUI workflow pipelines, Blender assets, and deferred 3D
training. honkbox is the local heavy-lifting engine; loop-lore keeps UX and
provider routing (`.tmp/research/llama-swap-loop-lore-gaps-2026-11-22.md`).

## Build order

1. Resource ledger + bounded queues + TTL/cancel + bounded-label metrics. Core
   landed; byte-aware LLM demand and remaining drain/admission follow-ups stay
   explicit.
2. Determinism contract (seed, model hash, engine version, canonical keys). Core
   landed; `TASK-inference-cache-correctness.md` (P0, open) extends identity to
   every inference cache lookup and separates weights from working memory.
3. Native runtime evidence: `TASK-vulkan-coopmat-baseline.md` (P1, open) measures
   capability-gated RDNA3/RDNA4 quantized matmul, full `MUL_MAT_ID` coverage,
   numerical parity, same-window timing, and peak bytes. It depends on item 2
   identity/version pins and blocks public Ash/Vulkan promotion. HTTP Ash
   selection in `TASK-ash-backend-promote.md` follows evidence; CLI Ash generation
   is already wired.
4. OCR ingest (text-layer first, cache, provenance).
5. RAG (embeddings via `/v1/embeddings`, hybrid retrieval, rerank external); any
   future result/prefix cache depends on item 2.
6. ComfyUI workflow pipelines (typed templates, HTTP+WS, cancel, artifacts).
7. Blender assets and deferred NeRF/3DGS.

## Acceptance

- Each item above has a ticket with observable acceptance criteria.
- No GPU allocation/load occurs without a demand-bearing lease; all long work is
  a cancellable job writing into `<data_dir>/jobs/<ulid>/` via `.part` + fsync +
  rename.
- Inference identity is resolved before cache lookup; weights, KV, and scratch
  are not conflated. Eviction readout does not claim a stronger drain barrier
  than the last in-flight `Arc` drop.
- Native fast-path claims require capability logs, full op coverage, numerical
  checks, same-window timings, and peak-memory evidence. No public flag/default
  change lands before that evidence.
- DTO changes keep `bun run codegen:check` green.

## Verification

Run the exact per-ticket commands first, then `bun run check`, `bun run test`,
`bun run codegen:check`, and `bun run lint:rs`. Use gitignored `.tmp/` scratch
dirs, test-local ledgers, port 0 or in-process routers, and one isolated GPU per
hardware measurement.

## Current status (2026-11-25)

- [FACT] Resource-ledger and determinism cores are landed; their tickets record
  remaining follow-ups.
- [FACT] Ash SD crate/CLI generation is wired (`main.rs:976-1023`). HTTP `serve`
  still selects only Stub/SdCpp through `make_sd` (`main.rs:677`;
  `backends.rs:101-128`). HTTP backend selection remains open.
- [FACT] Vendored ggml Vulkan contains RDNA3/RDNA4 integer cooperative-matrix
  `MUL_MAT`/`MUL_MAT_ID` pipelines, but honkbox does not expose the selected path.
  The exact vendor build is not proven to be b11160.
- [FACT] Ash matmul is scalar f32 and `AshLlmPipeline` is an unimplemented
  placeholder; no Ash LLM or honkbox coopmat speedup is established.
- [UNKNOWN] No honkbox RDNA3/RDNA4 timing, numerical-error, expert-routing, or
  peak-memory baseline has been measured.
- This planning pass ran no tests, builds, linters, or git commands.

## Current follow-ups (2026-11-22)

- `BUG-fizgig-family-model-resolution.md` — family-aware VAE/text-encoder
  resolution and pre-admission rejection.
- `BUG-fizgig-artifact-cap-and-gc.md` — bounded bundles, work-dir GC, resume
  state retention.
- `TASK-fizgig-admission-contract-tests.md` — path confinement, terminal-state
  cancel, non-unix and `/v1/jobs` contract gaps.

- `FEAT-fizgig-lora-training-jobs.md` + `TASK-fizgig-cli-probe.md` — landed
  first training slice (the four follow-ups below stay open).
- `TASK-caption-alt-text-sidecar.md`, `TASK-batch-generation-request.md`,
  `TASK-generation-record-history-api.md`, `TASK-generation-replay-endpoint.md`,
  `TASK-operability-surfaces.md` — content-recall and DX gaps from
  `content-experience-gaps-2026-11-20.md` and `organic-growth-2026-11-20.md`.

- `epic-comfyui-managed-runtime.md` — build-order item 6 now sits on the landed
  managed app + typed `/comfyui/*` reads; readiness/history honesty are filed
  there before the workflow pipeline grows a write surface.

## Non-goals

Engine SDKs/MCP inside honkbox; Blender embedded in-process; byte-identical
cross-backend output claims; public fast-path flags or throughput claims before
measurement.
