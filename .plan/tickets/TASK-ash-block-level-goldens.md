# TASK: Block-level goldens with first-divergence reporting

Status: proposed
Priority: P1
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: the parity harness dumps per-block tensor hashes behind an env flag, checks in goldens for the stub graph, and on mismatch reports the first diverging block; UNet growth proceeds one gated level at a time.
Verification: `cargo test --locked -p honkbox-sd-core`, `cargo test --locked -p honkbox-ash-backend`
Links: `.tmp/research/sdcpp-upstream-patterns-2026-11-25.md` §6.2 and §6.3, `TASK-ash-verification-gates.md`, `crates/sd-core/tests/parity.rs`, `crates/ash-backend/src/unet_real.rs`
Resources: per-test tmp dirs; canonical checkpoint fixture is owner-supplied, never downloaded

## Attribution

Localization practice synthesized from upstream `leejet/stable-diffusion.cpp` (MIT) backend-verification needs and the honkbox reduced-topology risk in `unet_real`. Method: white-room. Harness change and goldens are honkbox-original; no upstream code copied.

## Change

- Dump per-block tensor hashes behind an env flag (example `ASH_DUMP_BLOCK_HASHES`); check in goldens for the stub graph.
- On mismatch, report the first diverging block. Every future kernel lands with its golden already in the file.
- UNet growth order: one level at a time (second cross-attention layer, then full down/up depth), each level gated by its block-golden before the next begins. `unet_traversal` already composes levels; no parallel graph work.
- Blocking fixture (canonical checkpoint, prompt, seed) is owner-supplied with recorded SHA-256 and license; never downloaded by the harness.

## Acceptance criteria

- [ ] A deliberately perturbed block is reported as the first divergence by name.
- [ ] No growth level lands without its block-golden passing.
- [ ] No checkpoint download, network, or unrecorded artifact in any test.
