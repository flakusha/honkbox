# TASK: Decide on candle as a Rust inference backend

Status: postponed
Priority: P3
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: the 2026-11-24 research decision stands — candle is not adopted as
a honkbox SD backend while Vulkan is the first-class accelerator. Candle has no
Vulkan backend (CPU/CUDA/Metal/WASM only), so on Linux it would be a slower CPU
fallback duplicating the vendored sd.cpp CPU path. No new dependency is added;
candle-core/nn/transformers 0.10.2 are already transitive via any-tts.
Verification: research record `.tmp/research/candle-backend-option-2026-11-24.md`
(citations to upstream `candle-core/Cargo.toml`, `candle-onnx/Cargo.toml`, local
`Cargo.lock` and `deny.toml`). No build or test was needed for a decision.
Links: `TASK-ash-backend-promote.md`, `../research/ReviewedSupport-SD.md`.
Resources: none. Reopen only on a revisit trigger below.

## Revisit triggers

1. CUDA or Metal becomes a first-class honkbox target; or
2. Candle gains upstream Vulkan support; or
3. The ash backend is abandoned and sd.cpp's CPU path is no longer sufficient.

## Non-goals

No vendoring of candle-examples SD code (example binaries, not a library API),
no duplicate checkpoint/key-mapping layer, no license/NOTICE/deny policy change
(candle is MIT OR Apache-2.0, already allowed by `deny.toml:7-9`).
