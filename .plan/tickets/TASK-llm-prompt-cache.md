# TASK: Bounded prompt/KV cache controls

Status: done
Priority: P1
Epic: epic-llm-expansion
Owner: honkbox-llm-cache
Acceptance: bounded per-slug KV cache controls exist (`[[models.entry]] cache`: `n_ctx`, `kv_quant`, `offload`), never exceed `[models] vram_budget_bytes`, and the cache hits/misses are observable in kind-labeled metrics.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`, `bun run lint:rs`
Links: `.tmp/research/ReviewedSupport-LLM.md` §Remaining table
Resources: per-test temp model/cache dirs; no shared KV cache slot

## Change

- Expose bounded knobs per slug; default off until measured.
- Reuse the swap/residency ledger; no silent co-residency with SD/video work.
- Record cache hits/misses in bounded `kind`-labeled metrics only.

## Acceptance detail

A cache cannot grow beyond the configured VRAM budget; releasing a lease drops
its cache; long-session latency improvement is measured, not assumed.

## Shipped (2026-11-19)

- `crates/llm-worker`: `KvQuant` (`F16` | `Q8_0`), `LlmOptions::{kv_quant,
  offload_kqv, vram_budget_bytes}`, `kv_cache_bytes{,_per_token}` and
  `clamp_n_ctx`. The context size IS the cache budget: the loader clamps
  `n_ctx` to the VRAM ceiling before the llama.cpp context is created, so
  the cache is bounded from the first token (the clamp never returns 0 — a
  budget too small for one token keeps one).
- `crates/server/src/config.rs`: the `[models.entry.cache]` block (`n_ctx`,
  `kv_quant`, `offload`) on a slug-targeting entry; the knobs land in
  `ModelConfig::cache` (keyed by the slug). `n_ctx = 0` and a cache block
  without a slug are config errors.
- `crates/server/src/swap.rs`: `CachePolicy` (the per-slug knobs + the
  global ceiling) and the loader's `llm_options`.
- `crates/server/src/metrics.rs`: `honkbox_cache_hits_total` /
  `honkbox_cache_misses_total` (kind-labeled, bounded grid).
- The knobs are off until an entry declares them (absent keys = the backend
  default). The cache rides the loaded pipeline, so the swap manager's
  unload (idle TTL / LRU eviction) drops it with the pipeline; the request
  path holds no GPU slot between requests, so a resident LLM cache never
  silently co-resides with SD work.
- The seeded (static) pipeline is bounded like the swap-loaded one: the
  catalog resolves the requested id (an alias) to the registry slug the
  knobs are keyed by, and the VRAM ceiling reaches every LLM load.
- The hit/miss counters are the measurement instrument. The latency delta
  itself needs a local checkpoint (`HONKBOX_PARITY_MODEL`) and is read by
  the operator in deployment — it is not asserted in CI.
