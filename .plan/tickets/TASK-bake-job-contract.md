# TASK: Bake job contract

Status: done (2026-11-20, bake lane)
Priority: P2
Epic: epic-model-feedback
Owner: unassigned
Acceptance: a headless Blender bake job declares source mesh, UV/map targets, bake type, resolution, and output paths; results are deterministic and logged.
Verification: `bun run test:server`, opt-in local Blender fixture
Links: `.tmp/research/ReviewedCreative-BakingModels.md`
Resources: per-job temp `.blend` and map dirs; no shared Blender process

## Change

- Define input/output JSON contract for normal/AO/light/roughness bakes.
- Invoke Blender through the external process supervisor with `--background`
  and `--python-exit-code`; record binary version and exact argv.
- Reject missing UVs or overlapping targets before GPU work.

## Acceptance detail

A bake job writes maps + sidecar with source hash and settings; cancel kills the
process group; no in-process `bpy`.
