# TASK: Splat turntable job (vkgs preset ring)

Status: proposed
Priority: P2
Epic: epic-3d-capture
Owner: unassigned
Acceptance: a `.ply` artifact in → `output_turntable.webp` + frames in the job bundle, same manifest shape as the Blender turntable.
Verification: supervised headless run on a static fixture; `cargo test --locked -p honkbox-server`
Links: `.tmp/research/splat-generation-rendering-idol-vkgs-2026-11-20.md` §4B, `TASK-splat-render-vkgs-probe.md`, `FEAT-blender-job-route.md`
Resources: per-job tmp dirs; frames disposed after webp pack unless kept

## Change

- `kind=splat-render` supervised job: inputs from the artifact store by id,
  generated `job.cfg` (load → N `--activateCameraPreset` × `--screenshot`),
  `--headless 1`, progress = frames/total, cancel = kill group + lease release.
- `_camera.json` (IDOL output) seeds preset 0 = the input view; novel-view
  angle packs reuse the same path with fixed azimuths (E4 anchor pattern).
- Artifacts hashed into the manifest; ffmpeg/webp assembly argv recorded.

## Acceptance detail

- Blocked on the vkgs probe passing on this box. Output bundle shape
  mirrors the Blender job (`output_turntable.webp` + `manifest.json`).
