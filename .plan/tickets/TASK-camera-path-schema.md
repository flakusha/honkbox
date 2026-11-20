# TASK: Camera-path preset schema + validator

Status: proposed
Priority: P2
Epic: epic-3d-capture
Owner: unassigned
Acceptance: a documented preset-JSON contract (positions + lookats + timing) with a magic-check validator; vkgs round-trip proven.
Verification: vkgs in-app presets → `--loadCameraPresets` round-trip; `cargo test --locked -p honkbox-server`
Links: `.tmp/research/shaders-materials-uiux-asset-pipeline-2026-11-20.md` §5, `FEAT-location-flythrough-job.md`
Resources: per-test tmp dirs; no shared preset state

## Change

- Contract owned by loop-lore (Path3D gizmo / keyframe-list UI), validated
  honkbox-side like the model gate (magic-check + bounds, named rejects).
- First path source before any UI: vkgs in-app presets exported and fed
  back through `--loadCameraPresets`; nerfstudio/Blender-VFX JSON is the
  proven authoring loop to copy.
- No editor UI in honkbox (web-ui stays a thin viewer).

## Acceptance detail

- A malformed path fails with the field named. Timing is part of the
  contract (frames/total derives from it); honkbox never invents timing.
