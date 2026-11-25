# TASK: MemoryDemand estimator struct with labeled estimate/measured source

Status: proposed
Priority: P1
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: `MemoryDemand` with 8 fields and `DemandSource::{Estimated,Clamped,Measured}` maps to `residency::Demand` conservatively and labels estimate vs measured in logs/readouts.
Verification: `cargo test --locked -p honkbox-residency`
Links: `.tmp/research/residency-extended-fields-2026-11-25.md` Candidate 1, `crates/residency/src/lib.rs`, `crates/llm-worker/src/lib.rs`
Resources: per-test tmp dirs; `StubLlm` only; no GPU, no model downloads

## Change

- Add `MemoryDemand` (`weights_bytes`, `projector_bytes`, `kv_bytes`, `scratch_bytes`, `ram_bytes`, `headroom_bytes`, `measured: bool`, `source: DemandSource`) plus `DemandSource::{Estimated, Clamped, Measured}` in the residency seam only.
- Map `MemoryDemand` to existing `residency::Demand` conservatively: never understate VRAM when uncertain; estimator output reduced by `vram_budget_bytes` clamp is labeled `Clamped`.
- Estimator unit tests: decomposition sums, clamp relabels to `Clamped`, readouts/logs carry the estimate-vs-measured label.

## Acceptance criteria

- [ ] `MemoryDemand` has exactly these 8 fields with `u64` byte widths; no per-layer/per-tensor, per-device, or fragmentation fields.
- [ ] `source` is the 3-variant enum only; no free-form provenance string, no confidence float.
- [ ] Mapping never understates VRAM; estimate vs measured is labeled in logs/readouts.
- [ ] No GPU, checkpoint, download, or new dependency in any test.

Licensing: neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license.

[UNKNOWN] Whether `scratch_bytes` can graduate to `Measured` on the Vulkan path without allocator hooks; treat as `Estimated` until a harness proves otherwise.
