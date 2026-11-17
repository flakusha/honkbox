# FEAT: Engine-agnostic export bundles

Status: proposed
Priority: P2
Epic: epic-game-engine-export
Owner: unassigned
Acceptance: a versioned glTF/texture bundle with checksums and an explicit extension allowlist, consumable by loop-lore's Godot/Bevy/Ren'Py export tiers.
Verification: `gltf-transform inspect/validate <bundle.glb>`, `bun run check`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedFullCycle-GameEngines.md`
Resources: per-test bundle dirs; no shared glTF cache

## Change

- Package outputs from `FEAT-blender-asset-preflight`: GLB + PBR maps + rigs/
  animations + splat files + manifest with asset/engine versions and hashes.
- Hard-gate the glTF extension allowlist (no Draco/meshopt/
  `KHR_animation_pointer`/`KHR_gaussian_splatting` for the Bevy profile);
  splats ride web/Godot profiles only.
- DTO for bundle request/result in `crates/api-types`; regenerate TS/OpenAPI.

## Acceptance detail

No engine SDK, engine build command, or MCP concept in honkbox code or DTOs.
Godot headless import is demonstrated externally with `godot --headless
--path <proj> --import`; loop-lore owns export profiles and MCP calls.
