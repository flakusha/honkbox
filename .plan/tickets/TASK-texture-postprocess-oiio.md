# TASK: Texture post-process (mip/tile/colorspace/ORM via oiiotool)

Status: proposed
Priority: P0
Epic: epic-output-formats
Owner: unassigned
Acceptance: baked/generated PBR maps get mip chains, tiled packaging, per-map colorspace tags, and a Godot-convention ORM pack; every output hashed in the bundle sidecar.
Verification: `cargo test --locked -p honkbox-external-tools`, `bun run deny`
Links: `.tmp/research/texture-creation-workflows.md` §6, `.tmp/research/loop-lore-godot-actionables.md` §2–§4, `FEAT-game-engine-export-bundles.md`
Resources: per-test tmp dirs; no shared tool process; oiiotool absent → tests skip, never fail

## Change

- Probe `oiiotool`/`maketx` as an external binary (subprocess-only, same probe
  pattern as Blender; no linkage, so `deny.toml` is untouched whatever its
  license). Absent binary degrades to a named 501, never a silent skip.
- Post-process step after bake/generate stages (not another `image-ops` op —
  resize/crop already landed in-process): mip-chain generation, tiled packaging,
  per-map colorspace tagging (albedo sRGB, data maps linear), and ORM packing
  (R=AO, G=roughness, B=metallic — the shared Godot/Unreal convention).
- Sidecar records tool version, exact argv, per-map sha256 + dimensions +
  colorspace + role (`albedo|normal|rough|metal|ao|height|orm`), so loop-lore's
  `engine_exports` row and Godot import presets copy instead of re-derive.

## Acceptance detail

- Mip levels halve each step down to 1px; ORM channel order asserted per-pixel
  on a fixture (R=AO, G=rough, B=metal); a normal map mistagged sRGB is a
  hard error, not a warning.
- No new Rust dependency; no GPL/LGPL linkage path (binary-only, like Blender).
