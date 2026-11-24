# EPIC: 3D capture and splat viewing

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/splat-generation-rendering-idol-vkgs-2026-11-20.md`,
`.tmp/research/loop-lore-godot-actionables.md`

## Scope

Viewer-first: serve a pre-trained Gaussian-splat scene from job artifacts in
the web UI. Training (nerfstudio/gsplat) is explicitly deferred and needs a
separate long-running GPU reservation class.

## Tickets

- `TASK-splat-viewer-spike.md` — WebGL viewer, no training (first slice).
- `TASK-splat-render-vkgs-probe.md` — prebuilt vkgs headless render + capability probe (P2, in progress: real binary unverified).
- `TASK-splat-turntable-job.md` — sequencefile-driven turntable job (P2, follows the vkgs probe).
- `FEAT-image-to-3d-idol-spike.md` — image-to-3DGS fixture spike (P2, license decision open).
- `FEAT-3d-training-jobs.md` — postponed pending validated demand.

## Acceptance

A viewer spike loads a local pre-trained scene with orbit/zoom, records viewer
license and scene format (PLY/SPZ), and makes no byte-equality claim.

## Verification

Browser check against a static fixture; `bun run typecheck`; no GPU training in v1.

## Non-goals

Byte-identical cross-backend rendering; auto-starting training alongside chat/image.
