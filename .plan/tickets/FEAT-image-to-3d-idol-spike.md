# FEAT: Image-to-3D object spike (IDOL sidecar)

Status: proposed
Priority: P2
Epic: epic-3d-capture
Owner: unassigned
Acceptance: a prepared image produces a `.ply` + turntable bundle through a supervised job; licenses recorded.
Verification: fixture image run on CPU-eager; `cargo test --locked -p honkbox-server`
Links: `.tmp/research/splat-generation-rendering-idol-vkgs-2026-11-20.md` §1/§4A, `TASK-python-sidecar-subprocess.md`
Resources: per-test tmp dirs; gated encoders absent → named error, never stub

## Change

- Per-job `supervisor::run` + `residency` GPU-lease lane (`kind=3d-object`):
  uv-managed IDOL env (pinned rev, `uv sync --frozen`), `IdolPipeline` or
  `python -m idol.sample`, inputs from the artifact store by id.
- Bundle: `.ply` + `_input.png` + `_camera.json` + `output_turntable.webp`
  (vkgs preset ring, `TASK-splat-turntable-job.md`) + manifest (model rev,
  seed, mode, gaussians, sha chain).
- VRAM estimate 7.2 GB on the job kind; missing `HF_TOKEN` / gated encoders
  / EU-UK-KR territory (FLUX.2-dev NC terms) = named refusal, never stub.

## Acceptance detail

- License decision on the Tencent/FLUX/DINOv3 stack lands BEFORE the run
  (`product-licenses-*.md` §1); the spike records the decision + AMD-box
  timing (CUDA-only fast path vs eager/CPU) as its primary output.
