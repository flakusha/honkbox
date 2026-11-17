# FEAT: Blender asset preflight

Status: proposed
Priority: P1
Epic: epic-blender-assets
Owner: unassigned
Acceptance: headless Blender jobs normalize generated maps and geometry into engine-ready assets with a deterministic manifest.
Verification: `cargo test --locked -p honkbox-server`, `bun run codegen:check`, headless fixture bake + retopo run
Links: `.tmp/research/ReviewedFullCycle-Blender-Workstream.md`, `ReviewedFullCycle-Blender-Addons.md`
Resources: per-job temp `.blend` and map dirs; no shared Blender process

## Change

- External Blender process (`-b -P script.py --python-exit-code N`) with typed
  params: UV unpack/pack, quadriflow/voxel remesh with recorded `seed`,
  `object.bake` maps, texture attachment via `ShaderNodeTexImage`, glTF export.
- Probe Blender/addon versions at startup; record them plus script version and
  source hash in the job bundle. Automatic retopo/rig outputs are base meshes
  only — emit the hero-character deformation warning.
- Optional KTX2/WebP conversion via external MIT/Apache-2.0 tools, never Rust deps.

## Acceptance detail

One bake fixture yields PNG maps + JSON sidecar + progress events; cancel works;
GPL addons remain inside Blender and are never linked/vendored into honkbox.
