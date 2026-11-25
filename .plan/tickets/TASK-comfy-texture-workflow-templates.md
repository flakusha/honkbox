# TASK: Curated ComfyUI texture workflow templates

Status: proposed
Priority: P0
Epic: epic-comfyui-managed-runtime
Owner: unassigned
Acceptance: two pinned, versioned texture templates ship and emit map sets that pass PBR validation.
Verification: `bun run test:server`, `bun run codegen:check` (live-ComfyUI run opt-in/ignored)
Links: `.tmp/research/texture-creation-workflows.md` §4, `FEAT-comfyui-workflow-pipeline.md`, `TASK-material-map-validation.md`
Resources: per-test prompt IDs and temp output dirs; no shared ComfyUI queue state

## Change

- Depends on `FEAT-comfyui-workflow-pipeline.md` (template infra, 501s, lease).
  This ticket adds no infra — it ships the first two admin-installed templates:
  1. `texture-tileable-pbr`: text → seamlessly tiling PBR set (Patina nodes).
  2. `texture-image-to-pbr`: single image → baseColor/normal/rough/metal
     (Ubisoft CHORD nodes; Marigold/Lotus depth→normal fallback documented).
- Template hashes ride the pipeline manifest rule (workflows are code — only
  admin-installed templates); outputs feed the OIIO post-process ticket.

## Acceptance detail

- Both templates produce sets passing `TASK-material-map-validation.md` rules
  (required maps, matching dimensions, colorspace tags, recorded hashes).
- Unit tests mock ComfyUI HTTP; the live run is `ignored`, opt-in only.
