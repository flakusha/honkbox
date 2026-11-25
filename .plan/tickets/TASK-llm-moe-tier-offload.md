# TASK: Tiered MoE expert offload for LLM worker

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: per-slug offload tiers admit through residency, cold experts compute on CPU, tier placement is visible in metrics, and no request fails closed when a tier is unavailable.
Verification: `cargo test --locked -p honkbox-llm-worker`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/cuda-strata-inference-2026-11-25.md` §7.1, `TASK-llm-prompt-cache.md`, `crates/llm-worker/src/lib.rs`, `crates/residency/src/lib.rs`
Resources: stub backends only; no GPU, no model downloads

## Attribution

Policy idea from `Niko1221/Strata` (MIT — hot-expert residency with CPU fallback for cache-missed experts) and `RustingStudio/RustingStrata` (custom RustingStrata License 1.0, non-SPDX — ideas only, no code contact, no vendoring; its §2 forbids engine resale). Method: white-room. Tier policy and ledger are reimplemented against the `LlmPipeline` and residency `Lease` seams; no upstream code is copied. MIT sources would permit verbatim reuse with notice, but this ticket requires no copied text.

## Change

- Extend `LlmOptions` with offload tiers (`gpu_layers`, `cpu_hot_experts`, `disk_spill` bool) keyed by the existing `kv_cache_bytes` estimator family. Admission order stays Lease-first, then tier assignment.
- Add a small expert-load ledger beside residency, `(model, layer, expert)` to tier, LRU-evicted under VRAM pressure. No kernel work.
- Wire tiers through the existing llama.cpp split/offload knobs only; no custom CUDA, no NVRTC port. Vulkan-LLM stays deferred.
- Report tier hits/misses in the bounded kind-labeled metrics grid established by `TASK-llm-prompt-cache.md` (`honkbox_cache_hits_total` / `honkbox_cache_misses_total` pattern).

## Acceptance criteria

- [ ] Over-budget tier assignment is refused before load with the exact error named; a budget too small for one token keeps one token (existing clamp contract).
- [ ] Cold experts complete on CPU with no request failure; placement is observable in metrics.
- [ ] Unit tests use `StubLlm` only; no GPU, checkpoint, or download in any test.
