# EPIC: Full AI cycle

Status: proposed
Priority: P1
Owner: unassigned
Research: `.tmp/research/ReviewedFullCycle-Index.md` and its topic docs

## Scope

Foundation first — resource ledger/bounded queues and the determinism contract —
then OCR ingest, RAG, ComfyUI workflow pipelines, Blender assets, and deferred
3D training. honkbox is the local heavy-lifting engine; loop-lore keeps UX and
provider routing (`.tmp/research/ReviewedFullCycle-LoopLore-Acceleration.md`).

## Build order

1. Resource ledger + bounded queues + TTL/cancel + bounded-label metrics.
2. Determinism contract (seed, model hash, engine version, canonical keys).
3. OCR ingest (text-layer first, cache, provenance).
4. RAG (embeddings via `/v1/embeddings`, hybrid retrieval, rerank external).
5. ComfyUI workflow pipelines (typed templates, HTTP+WS, cancel, artifacts).
6. Blender assets and deferred NeRF/3DGS.

## Acceptance

- Each item above has a ticket with observable acceptance criteria.
- No GPU work without a lease; all long work is a cancellable job writing into
  `<data_dir>/jobs/<ulid>/` via `.part` + fsync + rename.
- DTO changes keep `bun run codegen:check` green.

## Verification

`bun run check`, `bun run test`, `bun run codegen:check`, `bun run lint:rs`.

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

## Non-goals

Engine SDKs/MCP inside honkbox; Blender embedded in-process; byte-identical
cross-backend output claims.
