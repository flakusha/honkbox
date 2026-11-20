# TASK: Mesh-from-splat watch (no build)

Status: proposed
Priority: P3
Epic: epic-3d-capture
Owner: unassigned
Acceptance: a dated watch note naming license status + quality bar vs generative meshes; no code.
Verification: none (research only)
Links: `.tmp/research/gen-flows-props-characters-locations-2026-11-20.md` §4
Resources: none

## Change

- Track: SuGaR (Inria custom license — verify text before commercial use),
  2DGS + GOF (no detectable license — unusable until clarified), Postshot
  commercial (external-only).
- Quality bar: meshing assumes trained multi-view-consistent splats;
  IDOL-style single-view `.ply` is out of scope (blobs, not assets).

## Acceptance detail

- Build only if (a) a license clears AND (b) captured-scene demand exists
  AND (c) generative meshes fail the asset. Default: stays a watch item.
