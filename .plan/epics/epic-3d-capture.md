# EPIC: 3D capture and splat viewing

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/ReviewedFullCycle-3D.md`,
`ReviewedFullCycle-Blender-Addons.md` (NeRF/3DGS bridges)

## Scope

Viewer-first: serve a pre-trained Gaussian-splat scene from job artifacts in
the web UI. Training (nerfstudio/gsplat) is explicitly deferred and needs a
separate long-running GPU reservation class.

## Tickets

- `TASK-splat-viewer-spike.md` — WebGL viewer, no training (first slice).
- `FEAT-3d-training-jobs.md` — postponed pending validated demand.

## Acceptance

A viewer spike loads a local pre-trained scene with orbit/zoom, records viewer
license and scene format (PLY/SPZ), and makes no byte-equality claim.

## Verification

Browser check against a static fixture; `bun run typecheck`; no GPU training in v1.

## Non-goals

Byte-identical cross-backend rendering; auto-starting training alongside chat/image.
