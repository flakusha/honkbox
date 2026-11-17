# FEAT: Blender rigging and skinning jobs

Status: proposed
Priority: P2
Epic: epic-blender-assets
Owner: unassigned
Acceptance: OSS-first rig jobs (Rigify metarig generation, automatic weights, high→low weight transfer) with bone/weight reports and a base-rig-only warning.
Verification: headless fixture: generate rig → auto-weights → transfer; `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedFullCycle-Blender-Rigging.md`
Resources: per-job temp `.blend` files; no shared Blender process

## Change

- Built-in APIs: `parent_set(type='ARMATURE_AUTO')`,
  `data_transfer(data_type='VGROUP_WEIGHTS')`, Rigify generator; probe Rigify
  before any paid face/retargeting addon.
- BlenRig 6 is free but GPL-3.0 — external process only. Mixamo Rig importer
  pins Blender 4.2 LTS (unsupported 5.5+).
- Assert bone names/count, zero unweighted vertices, preserved weight groups —
  not raw float equality across Blender versions.

## Acceptance detail

Automatic rigs are labelled base meshes: no hero-character or animation-ready
claim; shoulder/neck/face hand-tweaking remains the artist's job.
