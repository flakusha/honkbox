# TASK: Swap entry byte awareness with reasoned eviction

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: swap entries record per-entry bytes and evict under a byte cap (expired, LRU, then KV-before-weights) with a reason, keeping `llm_max_loaded` and pin/drain semantics unchanged.
Verification: `cargo test --locked -p honkbox-server swap`
Links: `.tmp/research/residency-extended-fields-2026-11-25.md` Candidate 2, `crates/server/src/swap.rs`
Resources: per-test tmp dirs; `StubLlm` only; no GPU, no model downloads

## Change

- Add `ResidentBytes` (`weights_bytes`, `projector_bytes`, `kv_bytes`, `measured: bool`) to the swap entry, recorded from the estimator at load (scratch/headroom gate admission only, never stored per entry).
- Add `EvictionReason::{Expired,Lru,BytesPressure,KvOnly,Explicit,Conflict}` threaded through `evict_idle` / `trim_for_load` / `evict_by_slug` / `evict_all_unpinned` / unload routes.
- Byte cap is opt-in/config-gated; when enforced it trims expired entries first, then LRU, then KV-before-weights; `llm_max_loaded` count-cap semantics unchanged and pin/in-flight drain behavior unchanged.

## Acceptance criteria

- [ ] Per-entry bytes recorded at load; `measured` false until a harness measures loads.
- [ ] Eviction order expired, then LRU, then KV-before-weights, with reason surfaced as a bounded label plus additive readout field.
- [ ] `llm_max_loaded` semantics unchanged; no byte-cap default that could evict hot models on uncalibrated estimates.
- [ ] Pin and in-flight drain (`Arc` holders) behavior unchanged; no per-entry prefix-hash or KV-block tables.
- [ ] No GPU, checkpoint, download, or new dependency in any test.

Licensing: neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license.

[UNKNOWN] No harness has calibrated a safe default byte cap for mixed workloads; the cap stays opt-in until estimator calibration lands.
