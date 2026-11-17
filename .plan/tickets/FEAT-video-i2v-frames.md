# FEAT: Image-to-video PNG frame sequence

Status: proposed
Priority: P2
Epic: epic-video-support
Owner: unassigned
Acceptance: a generated job PNG can be used as the init image and produces a deterministic PNG frame sequence plus sidecar in the job bundle; no muxing yet.
Verification: `bun run test:server`, `bun run test:core`, `bun run lint:rs`
Links: `.tmp/research/ReviewedSupport-Video.md` rollout step 4
Resources: per-test init images and frame dirs; no shared encoder state

## Change

- Depends on `TASK-video-job-class.md`.
- Reuse the existing job artifact PNG as `init_image`; cap width/height/frames.
- Write frames as PNG into the job bundle with a sidecar recording seed,
  checkpoint, fps, frame count, and parameters.

## Acceptance detail

Frame count and dimensions match the request; frames decode as PNG; cancellation
and artifact inspection work without a container muxer.
