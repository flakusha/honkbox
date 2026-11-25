# TASK: Persistent prefix/KV reuse

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: cross-request prefix/KV reuse ships behind an explicit opt-in with bounded eviction inside the VRAM budget, verified model/tokenizer/projector identity per lookup, and a measured prefill-latency benefit on long-prefix workloads; default off.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`, `bun run lint:rs`; local checkpoint fixture only, no CI downloads
Links: `TASK-llm-prompt-cache.md`, `TASK-inference-cache-correctness.md`, `.tmp/research/llm-runtime-memory-2026-11-25.md`, `.tmp/research/residency-kv-cache-memory-planning-2026-11-25.md`
Resources: per-test temp model/cache dirs; no shared KV cache slot

## Change

- Blocked on `TASK-inference-cache-correctness.md`: no reuse lookup until
  weight + tokenizer/template/projector identity is verified per entry.
- Prefix match on hashed token prefix bound to the verified identity; no
  cross-slug or cross-identity reuse; same-path replacement invalidates.
- Bounded LRU/size eviction inside `[models] vram_budget_bytes`; releasing
  a lease drops its cache, same as the resident prompt-cache controls.
- Report prefix hits/misses and prefill-latency delta in bounded `kind`
  metrics; ship only with measured, not assumed, long-prefix wins.

## Acceptance detail

- No second inference/response cache and no public fast-path flag before
  evidence; the done prompt-cache ticket is not evidence this exists.
- Opt-in only; absent configuration means per-request fresh contexts.
