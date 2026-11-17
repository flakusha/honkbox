# TASK: Bounded prompt/KV cache controls

Status: proposed
Priority: P1
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: bounded `cache_prompt`/`cache_ram` controls exist, never exceed `[models] vram_budget_bytes`, and show measurable benefit on long chat sessions.
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
