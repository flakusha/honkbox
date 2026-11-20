# FEAT: Location fly-through job (imported scene + preset path)

Status: proposed
Priority: P3
Epic: epic-3d-capture
Owner: unassigned
Acceptance: not committed — needs a real scene source and a loop-lore-authored camera path.
Verification: none until unblocked
Links: `.tmp/research/splat-generation-rendering-idol-vkgs-2026-11-20.md` §4D, `TASK-camera-path-schema.md`
Resources: none (deferred)

## Change

- Scene `.ply`/`.spz` (imported, gated by `TASK-model-format-validation`)
  + preset-path JSON (loop-lore-authored, validated by
  `TASK-camera-path-schema.md`) → vkgs headless frame sequence → ffmpeg →
  `flythrough.mp4` + preview webp in the job bundle.

## Acceptance detail

- Blocked on: a scene source (capture pipeline or licensed pack), the
  preset-JSON contract, and loop-lore path-authoring UI. Training stays
  postponed; this ticket does not unblock or require it.
