# TASK: PBR map-set validation

Status: proposed
Priority: P2
Epic: epic-vector-materials
Owner: unassigned
Acceptance: generated material sets are validated for required maps, matching dimensions, color space, and recorded hashes.
Verification: `bun run test:server`, `bun run codegen:check`
Links: `.tmp/research/ReviewedCreative-TexturesMaterials.md`
Resources: per-test map files in runner tmp; no shared texture cache

## Change

- Define required sets (albedo/base color, normal, roughness, metallic, AO) and
  allow partial sets with explicit `maps` metadata.
- Validate dimensions, bit depth, and color-space tags; reject mismatches with
  named 400s.
- Record per-map hash and generator version in the bundle sidecar.

## Acceptance detail

A client can detect an incomplete or mismatched PBR set before engine import.
