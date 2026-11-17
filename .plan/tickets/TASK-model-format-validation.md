# TASK: 3D model format validation

Status: proposed
Priority: P2
Epic: epic-binary-formats
Owner: unassigned
Acceptance: GLB and splat artifacts are validated against an extension allowlist before export; FBX/USD/OBJ are external-only conversions.
Verification: external `gltf-transform inspect/validate`; `bun run test:server`
Links: `.tmp/research/ReviewedBinaryFormats.md`, `TASK-model-bundle-validation.md`
Resources: per-test model dirs; no shared validator cache

## Change

- Share the allowlist with `TASK-model-bundle-validation.md`; no duplicate rules.
- Validate PLY/SPLAT/SPZ headers and sizes structurally; no in-process splat
  training or rendering.
- Record format, version, validator version, and hashes in the sidecar.

## Acceptance detail

Unsupported compressed extensions fail with named errors; a valid GLB passes
both this gate and the engine-bundle gate.
