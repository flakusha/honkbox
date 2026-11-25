# TASK: Blender bake extension (metallic/AO/height + thumbnail)

Status: Done
Priority: P0
Epic: epic-model-feedback
Owner: unassigned
Acceptance: asset jobs bake metallic, AO, and height maps and emit a thumbnail/turntable artifact through the public typed API and real embedded recipe; manifest records all of it.
Verification: `cargo test --locked -p honkbox-external-tools`, `cargo test --locked -p honkbox-server --test blender_jobs`, `bun run codegen:check`; live-Blender cases stay opt-in
Links: `.tmp/research/texture-creation-workflows.md` §5, `.tmp/research/creative-pipeline-promotion-2026-11-25.md`, `TASK-bake-job-contract.md`, `FEAT-blender-asset-preflight.md`
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

## Verification Notes (2026-11-25)

- Internal `BakeMap`/`AssetStage` planning and fake-recipe manifest tests exist,
  but the public `BlenderBakeMap` enum exposes only `Color`, `Normal`, and
  `Roughness` (`crates/api-types/src/lib.rs:2767-2777`). `BlenderStage` has no
  `Thumbnail` variant (`:2718-2765`).
- The real embedded recipe's `BAKE_TYPES` table contains only color/normal/
  roughness (`crates/external-tools/src/blender_asset.py:31-36`); no thumbnail
  dispatch exists. The previous `done` header was therefore premature.
- Close only after the DTO, route bounds, real recipe, and manifest all expose
  and execute the accepted variants end to end.

**Resolved:** 2026-11-25 registry-driven close: git issue 2cb7655 (registry tip: 67ed2cd Konstantin Fedotov Auto-closed: ticket TASK-BLENDER-BAKE-EXTENDED-MAPS marked done in index.)
