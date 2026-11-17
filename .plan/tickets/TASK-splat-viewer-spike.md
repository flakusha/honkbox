# TASK: Gaussian-splat viewer spike

Status: proposed
Priority: P2
Epic: epic-3d-capture
Owner: unassigned
Acceptance: a pre-trained splat scene loads in the web UI with orbit/zoom from a local static fixture.
Verification: `bun run typecheck`, `bun run check`; browser check against fixture
Links: `.tmp/research/ReviewedFullCycle-3D.md`, `ReviewedFullCycle-Blender-Addons.md` (3DGS bridges)
Resources: static fixture under runner tmp; no GPU training, no shared viewer port

## Change

- Embed a WebGL viewer (e.g. Three.js-based GaussianSplats3D or minimal
  antimatter15/splat) served from job artifacts; record the chosen license
  before bundling.
- Accept PLY/SPZ/`.splat` scene files; do not implement training or NeRF
  rendering in this ticket.
- The viewer is read-only; it must not claim byte-identical output across GPUs.

## Acceptance detail

If the spike is not compelling, close the epic's training follow-up as
postponed — no further 3D buildout by default.
