# TASK: Promote the ash Vulkan backend slices

Status: proposed
Priority: P1
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: all GPU-verified commits on `feat/ash-vulkan-backend`, including the
pending SDXL-inspection slice, land on master; `FEAT-ash-vulkan-backend.md`
(already committed on the branch) is reconciled with the landed slice record;
`AshSdPipeline::generate` renders a deterministic fixture from a user checkpoint
using Vulkan compute only; missing capabilities return a named
`VkError::InsufficientCapacity`; every shader carries a `// Provenance:` line;
dangling `ReviewedAsh-*.md` references are replaced with live paths.
Verification: `cargo test --locked -p honkbox-ash-backend --features experimental-ash-sd`,
`bun run check`, `bun run ci` (regenerate generated snapshots after merge).
Links: `.tmp/research/ash-vulkan-backend-licensing-extensibility-2026-11-22.md`
§2-§5. Resources: Vulkan GPU required for final acceptance; ownership of the
rebase/merge is a human decision before this ticket starts.

## Verified 2026-11-24 (pre-landing)

Branch `feat/ash-vulkan-backend` @ `fc0ac5f`: 48 ahead / 11 behind master
`07c5adf` (merge base `db95fb3`). Collision files changed on both sides:
`Cargo.lock`, `crates/api-types/src/lib.rs`, `crates/server/src/main.rs`,
`packages/web-ui/src/gen/api-types.ts`, `packages/web-ui/src/gen/openapi.json`.
Pending uncommitted SDXL slice: modified `checkpoint_family.rs`, `lib.rs`,
`server/src/main.rs`; untracked `ash-backend/README.md`, `src/sdxl.rs`,
`tests/sdxl.rs` (per-test unique temp files with `Drop` teardown).
All 13 shaders carry provenance lines. Five real SD 1.5 checkpoints pass
`models inspect-ash`; `generate` still returns a named sampling error, so
fixture rendering is not yet met. 12 dangling `ReviewedAsh-*.md` refs remain in
7 files; the live doc is the research note above. Sweep refs only after merge —
the same files are modified on the branch.

## Change

- Commit or drop the pending SDXL slice; resolve the five collision files,
  regenerate TS/OpenAPI snapshots, then land the branch (rebase/merge chosen by owner).
- Reconcile `FEAT-ash-vulkan-backend.md` slice history with landed commits.
- Close the generate gap: bias path, EOS pooling, fused-QKV parity need a real checkpoint.
- Sweep dangling research references after merge; add the deferred `// Provenance:`
  check to `bun run spdx` (research §3 trigger was a second shader — 13 have landed).

## Non-goals

Replacing the diffusers backend, new licensing policy, dispatch changes.
