# TASK: Model bundle validation gate

Status: proposed
Priority: P2
Epic: epic-model-feedback
Owner: unassigned
Acceptance: exported GLB bundles are validated with `gltf-transform` against an explicit extension allowlist before being marked ready.
Verification: external `gltf-transform inspect/validate`; `bun run test:server`
Links: `.tmp/research/ReviewedCreative-BakingModels.md`, `FEAT-game-engine-export-bundles.md`
Resources: per-test bundle dirs; no shared validator cache

## Change

- Run the validator as an external tool; record version and result in the
  job sidecar.
- Reject Draco/meshopt/`KHR_gaussian_splatting` unless explicitly allowlisted;
  keep Bevy extension allowlist as the baseline.
- Keep checksums and manifest as the single source of truth.

## Acceptance detail

A malformed or unsupported bundle fails with a named error before engine import;
no validator dependency enters `Cargo.toml`.
