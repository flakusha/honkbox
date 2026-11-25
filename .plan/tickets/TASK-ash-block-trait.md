# TASK: Formal Block trait for chain-of-blocks dispatch

Status: proposed
Priority: P3
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: a `Block` trait (weights view in, command buffer plus tensor views through, outputs out) has `UnetInputBlock` and `UnetAttentionBlock` as first impls with per-block parity tests unchanged and zero numerics change.
Verification: `cargo test --locked -p honkbox-ash-backend`
Links: `.tmp/research/vulkan-colibri-hyperstream-2026-11-25.md` §7.3, `crates/ash-backend/src/unet_traversal.rs`, `crates/ash-backend/src/unet_input.rs`, `crates/ash-backend/src/unet_attention.rs`
Resources: existing fixtures only; no GPU, no checkpoint, no download

## Attribution

Chain-of-blocks dispatch pattern from `JustVugg/colibri` (Apache-2.0 — `qwenimage_chain.h` DiT dispatch). Method: white-room. The trait is a re-skin of existing honkbox traversal; no upstream code copied. Apache-2.0 would permit verbatim reuse with notice, but no copied text is needed.

## Change

- Define `Block` (weights view in, command buffer and tensor views through, outputs out).
- Implement for `UnetInputBlock` and `UnetAttentionBlock` first; DiT blocks reuse traversal later.
- Existing per-block parity tests unchanged; trait is a re-skin with zero numerics change.

## Acceptance criteria

- [ ] All existing block parity tests pass unmodified against the trait.
- [ ] No numerics, kernel, dispatch-order, or fixture change.
- [ ] No GPU, checkpoint, download, or new dependency in any test.
