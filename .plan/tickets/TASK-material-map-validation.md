# TASK: PBR map-set validation

Status: done
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

## Verification Notes (2026-11-20)

`texture_postprocess::run` probes every input before staging outputs, rejects
duplicate roles, dimension and probed bit-depth mismatches with a named
`pbr_map_set_mismatch` 400, records `missing_required_maps` /
`bit_depth_unvalidated` in the sidecar, and keeps per-map SHA-256 plus tool
versions. Roles define color space; an unreported depth is never claimed as
checked. Full `bun run test` and `bun run check` pass.
