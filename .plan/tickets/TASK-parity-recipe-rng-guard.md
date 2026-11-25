# TASK: Parity recipe table (SD1.5 first) plus RNG-mode guard

Status: proposed
Priority: P2
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: a per-family recipe table (checkpoint, prompt, seed, sampler, steps, expected hashes) starts with SD1.5; the harness refuses cross-RNG-mode byte comparison; CI stays build-only.
Verification: `cargo test --locked -p honkbox-sd-core --test parity`, `bun run check-links`
Links: `.tmp/research/sdcpp-upstream-patterns-2026-11-25.md` S4/S12, `.tmp/research/determinism-scheduling-research-2026-11-25.md` D12, `crates/sd-core/tests/parity.rs`, `TASK-ash-block-level-goldens.md`
Attribution: upstream RNG duality (`rng_mt19937.hpp` torch-cpu port, `rng_philox.hpp` torch-cuda port) observed in leejet/stable-diffusion.cpp (MIT). Method: white-room, no upstream code copied.
Resources: synthetic fixtures only; `HONKBOX_PARITY_MODEL` opt-in; no download in any test.

## Change

- Add a recipe table: one row per family (SD1.5 first) pinning checkpoint hash, prompt, seed, sampler/scheduler, steps, and expected structural hashes.
- Add an RNG-mode guard: byte comparison is refused (named failure, not skip) when the two sides ran under different RNG modes; mode is recorded in the recipe row.
- Do not invent the upstream `--rng` flag spelling (unverified post-restructure); name modes honkbox-side until pinned against the vendored rev.

## Acceptance detail

- [ ] SD1.5 recipe row runs green on stub; real-checkpoint row stays opt-in and unmeasured without a local asset.
- [ ] Cross-mode byte comparison fails loudly with the mode names; same-mode stub comparison stays byte-identical.
- [ ] No GPU, checkpoint download, or network in any test.
