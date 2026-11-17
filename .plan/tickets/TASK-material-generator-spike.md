# TASK: Material generator spike

Status: proposed
Priority: P3
Epic: epic-vector-materials
Owner: unassigned
Acceptance: one local PBR generator path (ComfyUI nodes or Material Maker CLI) produces a valid map set for a named fixture; no route added yet.
Verification: opt-in local fixture; `bun run lint:rs`, `bun run deny`
Links: `.tmp/research/ReviewedCreative-TexturesMaterials.md`
Resources: per-test output dirs; process group killed in `finally`/drop

## Change

- Depends on `TASK-material-tool-capability-probe.md` and
  `TASK-material-map-validation.md`.
- Run through the external process supervisor; record version, license, argv,
  seed, and input hash in the sidecar.
- Compare structural validity only; no byte-equality claim.

## Acceptance detail

A failed or cancelled generation leaves no orphan process and no partial map set
advertised as complete.
