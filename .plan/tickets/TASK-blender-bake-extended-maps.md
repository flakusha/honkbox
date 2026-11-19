# TASK: Blender bake extension (metallic/AO/height + thumbnail)

Status: proposed
Priority: P0
Epic: epic-model-feedback
Owner: unassigned
Acceptance: asset jobs bake metallic, AO, and height maps and emit a thumbnail/turntable artifact; manifest records all of it.
Verification: `cargo test --locked -p honkbox-external-tools` (live-Blender cases `ignored`, opt-in)
Links: `.tmp/research/texture-creation-workflows.md` §5, `TASK-bake-job-contract.md`, `FEAT-blender-asset-preflight.md`
Resources: per-test tmp job dirs; no shared Blender process

## Change

- Delta on the landed recipe + `TASK-bake-job-contract.md` (contract first,
  this extends it): new `BakeMap` variants `Metallic`, `Ao`, `Height`, and a
  `Thumbnail` stage (single still + turntable frames for the loop-lore asset
  browser) — all backward-compatible `AssetStage` additions.
- Baked discrete maps are the input to the OIIO ORM pack; roles/colorspaces
  follow the Godot contract (`loop-lore-godot-actionables.md` §4).

## Acceptance detail

- Structural tests only (dimensions, PNG magic, manifest sha chain — same
  style as `parity.rs`); no byte-equality claim across Blender versions.
- Quadriflow-absent degradation and stage-boundary cancel keep working.
