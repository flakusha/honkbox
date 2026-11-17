# TASK: Resource ledger and bounded queues

Status: proposed
Priority: P0
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: `residency::global()` becomes a capability/VRAM-aware ledger while keeping the RAII `Lease`; per-kind bounded queues with cancel, TTL, and bounded-label metrics.
Verification: `cargo test --locked -p honkbox-residency`, `bun run test:server`, `bun run lint:rs`
Links: `.tmp/research/ReviewedFullCycle-Resource-Queues.md`
Resources: per-test ledger instances; no global slot reused across tests

## Change

- Model resources as GPU slot/VRAM bytes/RAM/CPU threads with a fixed `kind`
  vocabulary (`sd`, `llm`, `ocr`, `embed`, `rag`, `comfy`, `blender`, `3d`).
- Reserve **before** model load or external process start; reject with 429/503
  + `Retry-After` when the bounded queue is full.
- Preserve `Drop`-based lease release across cancel, timeout, and engine crash;
  add idle TTL unload hooks (`/free`, process stop).
- Metrics: queue depth, wait time, running count, lease hold time, admissions,
  rejections, evictions, OOM near-misses. Labels never contain user data.

## Acceptance detail

A queued job cancelled before admission never acquires a lease; a running job
releases its lease on cancel; no queue can grow past its configured cap.
