# TASK: Determinism contract

Status: proposed
Priority: P0
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: published reproducibility levels and canonical hash keys; bundle metadata records seed, model hash, engine version, and all sampling/parser params.
Verification: `cargo test --locked -p honkbox-server`, `cargo test --locked -p honkbox-sd-core --test parity`, `bun run check`
Links: `.tmp/research/ReviewedFullCycle-Determinism.md`
Resources: per-test temp bundles; no shared hash cache

## Change

- Define levels L0 (stable IDs/cache keys) through L3 (cross-hardware equality,
  explicitly unsupported); document that deterministic kernels may be slower.
- Canonical JSON (sorted keys, no ULID/timestamp/queue number) + BLAKE3/SHA-256
  over inputs + model revision + engine version + code version.
- Named PRNG streams derived from the root seed by stage; stable topological
  order and RRF tie-break by chunk ID.
- Extend `jobs.rs` metadata with model hash, sampler/scheduler, steps, cfg,
  dimensions, recipe, hires params, LoRA hashes, addon/engine versions.

## Acceptance detail

Same request twice produces identical bundle inputs on the stub backend;
hashing is stable across map insertion order; cross-backend parity stays
structural (no byte-equality assertion).
