# TASK: Promote the ash Vulkan backend slices

Status: proposed
Priority: P1
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: the four GPU-verified commits on `feat/ash-vulkan-backend`, plus
the pending `crates/ash-backend/src/sd.rs` edit, land on master; the
worktree-only `FEAT-ash-vulkan-backend.md` is promoted into `.plan/tickets/`;
`AshSdPipeline::generate` renders a deterministic fixture from a user checkpoint
using Vulkan compute only; missing capabilities return a named
`VkError::InsufficientCapacity`; every shader carries a `// Provenance:` line;
dangling `ReviewedAsh-*.md` references are replaced with live paths.
Verification: `cargo test --locked -p honkbox-ash-backend`, `bun run ci`.
Links: `.tmp/research/ash-vulkan-backend-licensing-extensibility-2026-11-22.md`
§2-§5. Resources: Vulkan GPU required for final acceptance; ownership of the
rebase/merge is a human decision before this ticket starts.

## Change

- Promote the worktree ticket and reconcile its status with the landed slices.
- Land the pending `sd.rs` edit with the branch, then close the named gap.
- Sweep dangling research references.

## Non-goals

Replacing the diffusers backend, new licensing policy, dispatch changes.
