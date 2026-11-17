# TASK: Deterministic OTIO sidecar

Status: proposed
Priority: P2
Epic: epic-media-timeline
Owner: unassigned
Acceptance: frame-sequence job bundles contain `timeline.otio` with clip order, relative artifact paths, frame rate, and duration; identical bundles produce byte-identical sidecars.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedExpansion-TimelineEditing.md` §Recommended boundary
Resources: per-test job bundle dirs; no shared timeline cache

## Change

- Emit OTIO JSON from `FEAT-video-i2v-frames.md` artifacts; no media bytes in
  the sidecar.
- All media references are relative to the bundle root; no absolute paths,
  wall-clock timestamps, or host-specific metadata.
- Keep a stable key order and newline convention; test two runs of the same job
  for byte equality.

## Acceptance detail

A third-party OTIO reader can determine clip order and duration without
reading any other file; bundle artifacts remain the source of truth.
