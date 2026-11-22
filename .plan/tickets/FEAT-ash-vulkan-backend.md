# FEAT: Pure-Vulkan (`ash`) inference backend

Status: in progress (first slice landed in this worktree, uncommitted 2026-11-22)
Priority: P1
Epic: epic-full-ai-cycle
Owner: ash-vulkan lane
Acceptance: With `experimental-ash-sd` enabled, `AshSdPipeline::generate` renders a deterministic fixture from a user-supplied checkpoint using Vulkan compute only (no CUDA/ComfyUI process); missing device features return named `VkError::InsufficientCapacity`; every shader carries a provenance line (license comes from `REUSE.toml`, which overrides per-file headers in this repo).
Verification: `cargo test --locked -p honkbox-ash-backend --features ash-compute` (first slice, no vendor stack); `cargo test --locked -p honkbox-ash-backend --features experimental-ash-sd`; `bun run test:parity` with a local checkpoint (opt-in); `bun run check`.
Links: `.tmp/research/ash-vulkan-backend-licensing-extensibility-2026-11-22.md` (main checkout), `crates/ash-backend/src/{device,sd}.rs`, `TASK-device-capacity-gates.md`.

## Change

- Add an ops vocabulary (elementwise, matmul, norm/softmax, RoPE, conv2d,
  attention, sampler step, VAE decode slice) with one `ModelSpec` per family:
  graph + tensor-name map. New models add one spec module, not a new pipeline.
- Extend `PhysicalDeviceInfo::admit` to consume a required-capability set;
  optional extensions (e.g. cooperative matrix) get portable fallbacks.
- Compile GLSL → SPIR-V at build time, embed with `include_bytes!`; every source
  file carries a provenance line (original vs derived, upstream path + pinned
  rev). License stays in `REUSE.toml` — this repo has no per-file SPDX headers.
  No runtime shader compiler.
- Wire `server/experimental-ash-sd` in `main.rs` constructor only; keep
  `SdCppPipeline` as default. `stubs.rs` LLM/STT/TTS remain separate follow-ups.

## Non-goals

CUDA/ROCm backends, tensor/model parallelism, bundled weights, byte-level
equality with stable-diffusion.cpp, API/DTO changes.

## Landed — slice 1 (commit a32829a)

`elementwise_add` runs on the selected Vulkan device end to end: GLSL source,
offline glslang compile, embedded SPIR-V, descriptor set + compute pipeline,
host-visible buffers, capacity admission, fenced submit and readback. New
`ash-compute` feature builds this stack without the vendored diffusion-rs graph;
`experimental-ash-sd` forwards to it. Vulkan calls stay in `raw.rs`; dispatch
is serialized by a mutex because descriptor and command buffers are reused.

Evidence: `cargo test --locked -p honkbox-ash-backend --features ash-compute`
→ 18 passed (GPU test exercised 4097 elements on RADV RX 7900 XTX, not skipped);
`cargo clippy ... --all-targets -- -D warnings` → clean; `cargo fmt --check` →
clean. `experimental-ash-sd` full-graph check green: 233 crates compiled,
`Finished dev profile` (13m25s).

## Landed — slice 2 (2026-11-22, worktree)

`MatMul` adds row-major GEMM on the same pipeline scaffold. `compute::Kernel`
now owns the shader/pipeline/descriptors/command buffer and is shared by both
ops; `GpuBuffer` and SPIR-V decoding are shared too. New `shaders/matmul.comp`
uses an 8×8 tile with a shape guard; shape mismatch is rejected before any GPU
allocation. Evidence: 19 tests pass (`matmul_dispatches` ran on GPU, no skip),
clippy `-D warnings` clean, fmt clean, `experimental-ash-sd` graph check green
(1 crate rebuilt incrementally).

Next: capability sets with optional-extension fallbacks, ops registry +
`ModelSpec`, then `AshSdPipeline::generate` on a real checkpoint.

