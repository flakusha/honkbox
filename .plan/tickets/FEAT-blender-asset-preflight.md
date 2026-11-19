# FEAT: Blender asset preflight

Status: done
Priority: P1
Epic: epic-blender-assets
Owner: main session
Acceptance: headless Blender jobs normalize generated maps and geometry into engine-ready assets with a deterministic manifest.
Verification: `cargo test --locked -p honkbox-external-tools` (+ `-- --ignored` for the real-Blender fixture)
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

## Landed (2026-11-19, main session)

`crates/external-tools/src/blender_assets.rs` + `blender_asset.py` (the recipe is
embedded with `include_str!`):

- Typed job: source (primitive or file), stages (`uv_unpack`, `uv_pack`,
  `voxel_remesh`, `quadriflow`, `attach_texture`, `bake`, `export_glb`), output
  dir, prefix, seed.
- Headless run: `blender --background --factory-startup -noaudio --python
  <recipe> --python-exit-code 1 -- <job.json> <result.json>`; progress arrives
  as `HONKBOX_PROGRESS={json}` lines on stdout.
- `manifest.json` in the output dir (`.part` + fsync + rename): manifest
  version, Blender version, recipe version, recipe sha256, the echoed job,
  every artifact with size and sha256, the warnings, the progress events.
- Cancel lands on a stage boundary (the runner drops a marker file, the recipe
  checks it between stages), so a bake is never interrupted half-written; the
  job deadline is the only kill.
- `quadriflow` degrades to a warning where the build dropped the operator
  (Blender 4.5 does); automatic retopology always reports the base-mesh
  warning (no hero-character deformation promise).

Verified: `cargo test --locked -p honkbox-external-tools` (30 passed, 2
ignored) and the ignored real-install fixture against Blender 4.5.15 LTS
(`--ignored`: PNG maps + manifest). Scoped clippy `-D warnings` and scoped fmt
clean. The workspace-wide gate did not run in this session: `/tmp` (the shared
`CARGO_TARGET_DIR`) was 100% full, so the crate was built in an isolated target
directory on the nvme disk.

Other tickets own the rest: the HTTP job route and its sidecar
(`FEAT-blender-job-route.md`), the optional KTX2/WebP conversion, and the
rigging job (`FEAT-blender-rigging.md`).
