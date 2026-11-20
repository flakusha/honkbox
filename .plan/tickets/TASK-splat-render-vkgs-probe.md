# TASK: vk_gaussian_splatting renderer probe

Status: proposed
Priority: P2
Epic: epic-3d-capture
Owner: unassigned
Acceptance: prebuilt vkgs binary renders one screenshot from a fixture `.ply` headless; version + supported pipelines recorded.
Verification: `vk_gaussian_splatting --headless 1 --sequencefile fixture.cfg fixture.ply`; `bun run test:server`
Links: `.tmp/research/splat-generation-rendering-idol-vkgs-2026-11-20.md` §2, `epic-external-tools.md`
Resources: per-test tmp dirs; no shared viewer process

## Change

- New `ToolId::SplatRenderer`: probe the prebuilt Linux binary (`--help`/
  version + fixture `.ply` → 1 screenshot), capability rows for pipelines
  that actually run on this box (RADV vs NVIDIA is the open question).
- Record the sequencefile automation contract (SEQUENCE blocks,
  `--activateCameraPreset`, `--pipeline`, `--screenshot`) as the job
  driver's input shape for `TASK-splat-turntable-job.md`.
- Apache-2.0/MIT stack: no `deny.toml` friction even if vendored later;
  stays out of `Cargo.lock` (external binary, Blender precedent).

## Acceptance detail

- Spike step 0 is the RADV run: if raster fails on this box, the ticket
  closes with the failure log and both downstream job tickets stay blocked.
