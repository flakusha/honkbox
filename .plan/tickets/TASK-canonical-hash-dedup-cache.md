# TASK: Canonical-hash dedup cache without GPU lease

Status: done (2026-11-20, hash cache lane)
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: the same canonical request hash replays/dedups without acquiring a GPU lease; the cache is bounded with TTL and eviction; distinct hashes never collide.
Verification: `cargo test --locked -p honkbox-server`, `bun run ci`
Links: `TASK-determinism-contract.md`, `routes/image_ops_dedup.rs`
Resources: per-test temp dirs / in-process cache; no shared state; bounded capacity and TTL

## Change

- Reuse the canonical request hash from `TASK-determinism-contract.md` and the
  dedup seam in `routes/image_ops_dedup.rs`. A cache hit replays the stored
  result before lease acquisition so identical requests consume no GPU slot.
- Make the cache bounded: fixed capacity, TTL expiry, and deterministic
  eviction. Cache identity is the full canonical hash — distinct hashes must
  never alias to one entry.
- Keep the cache in-process and per-test isolated; no shared global that leaks
  across parallel tests.

## Acceptance detail

- Two requests with the same canonical hash produce one result and exactly one
  (or zero, on replay) lease acquisition; the second is served from cache.
- Entries expire after TTL and the oldest/expired entries evict at capacity;
  size never exceeds the bound.
- A collision test over distinct canonical hashes returns distinct cached
  results, proving the hash is the full cache key.
