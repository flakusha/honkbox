# TASK: External edit-process spike

Status: proposed
Priority: P3
Epic: epic-media-timeline
Owner: unassigned
Acceptance: one FFmpeg concat render runs through `TASK-process-supervisor.md` from a local fixture, with binary version, license, and exact args recorded in the bundle.
Verification: local FFmpeg only; `bun run lint:rs`, `bun run deny`
Links: `.tmp/research/ReviewedExpansion-TimelineEditing.md` §Rollout 2
Resources: per-test render dirs; process group killed in `finally`/drop; no CI binary install

## Change

- Use the concat demuxer for same-codec inputs; do not add FFmpeg/MLT as Rust
  dependencies (`deny.toml` bans GPL/LGPL incoming).
- Record operator-selected binary path, version string, license note, and full
  argv in the job sidecar.
- Cancel kills the process group and releases any lease; no orphan processes.

## Acceptance detail

If the spike is unconvincing, `FEAT-timeline-render-job.md` stays research-only;
this ticket adds no route or `JobKind::edit`.
