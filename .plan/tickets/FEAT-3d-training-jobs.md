# FEAT: 3D training jobs

Status: postponed
Priority: P3
Epic: epic-3d-capture
Owner: unassigned
Acceptance: not committed — depends on a compelling viewer spike and validated demand.
Verification: none until unblocked
Links: `.tmp/research/ReviewedFullCycle-3D.md`
Resources: none (deferred)

## Blocker / preconditions

- `TASK-splat-viewer-spike.md` must first show a user-visible payoff.
- nerfstudio/gsplat are Python/CUDA stacks (long, multi-hour GPU jobs) and need
  a separate reservation class, checkpoint/resume, and preemption policy.
- Do not share the interactive GPU slot policy; never auto-start training
  alongside chat/image generation.

## Unblock sequence

Define a `kind=3d` job with VRAM estimate, TTL, progress events, cancel =
checkpoint, and artifact export (PLY/SPZ + preview). Then re-open.
