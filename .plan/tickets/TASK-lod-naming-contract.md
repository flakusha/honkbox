# TASK: LOD naming contract

Status: proposed
Priority: P3
Epic: epic-model-feedback
Owner: unassigned
Acceptance: generated LOD levels use deterministic names and manifest metadata so engine importers can discover them without heuristics.
Verification: `bun run test:server`, `bun run codegen:check`
Links: `.tmp/research/ReviewedCreative-BakingModels.md`
Resources: per-test LOD fixture dirs; no shared asset database

## Change

- Define `<base>_LOD0..n` naming and record triangle counts, source hash, and
  decimation settings in the manifest.
- Validate monotonic triangle counts and matching material slots.
- Keep this a metadata/naming contract; decimation stays external.

## Acceptance detail

An engine importer can order LODs from the manifest alone; renamed files fail
validation with a named error.
