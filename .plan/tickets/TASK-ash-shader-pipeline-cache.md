# TASK: Build-time shaders plus persisted pipeline cache

Status: proposed
Priority: P3
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: cold-start pipeline-creation time is logged per kernel today, stable kernels embed precompiled SPIR-V, and the runtime pipeline-cache blob persists under `data_dir` keyed by driver UUID with the runtime-compile fallback intact.
Verification: `cargo test --locked -p honkbox-ash-backend`, `bun run spdx`
Links: `.tmp/research/vulkan-colibri-hyperstream-2026-11-25.md` §7.1, `crates/ash-backend/src/compute.rs`, `crates/ash-backend/src/device.rs`
Resources: normal CPU test process; no GPU lease required for the logging step

## Attribution

Build-time-compile plus pipeline-cache practice from `maeddesg/vulkanforge` (GPL-3.0-only — pattern only, no code contact; GPL code is never vendored or copied into the AGPL tree). Method: white-room. Ordering (metric, then embedded SPIR-V, then persisted cache) and all build changes are honkbox-original.

## Change

- Ordered: (1) log cold-start pipeline-creation time per kernel today, one metric, no behavior change; (2) embed precompiled SPIR-V for the stable kernels (elementwise, matmul, softmax) via the existing build path; (3) persist and reload the pipeline-cache blob under `data_dir` keyed by driver UUID.
- Each step keeps the runtime-compile fallback (honest on unknown drivers; slow path always works).
- New shaders carry `// Provenance:` lines per the existing `spdx` gate.

## Acceptance criteria

- [ ] Cold-start metric exists before any caching change.
- [ ] Cache hit and fallback paths are both exercised in tests without a GPU.
- [ ] Unknown-driver behavior stays runtime-compile, never an error.
