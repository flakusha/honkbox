# EPIC: Media timeline interchange

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/media-video-normalization-and-repair-2026-11-24.md`,
`.tmp/research/media-audio-normalization-and-cleanup-2026-11-24.md`

## Scope

honkbox emits deterministic editorial *interchange* metadata for generated
media; it does not become an NLE. Job bundles remain the source of truth, with
a relative-path OTIO sidecar describing clip order, frame rate, and duration.
Headless rendering stays an external operator-chosen process.

## Tickets

- `TASK-otio-sidecar.md` — deterministic `timeline.otio` in frame bundles (P2).
- `TASK-external-edit-process-spike.md` — one FFmpeg concat render via the
  process supervisor (P3).

## Acceptance

Sidecar JSON is byte-stable for identical bundles, uses only relative paths,
and contains no wall-clock metadata. No GPL/LGPL code is linked into Rust.

## Verification

`bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run deny`;
external FFmpeg fixture only with recorded binary version and exact args.

## Non-goals

Timeline UI, track model in the DTOs, embedding MLT/FFmpeg as crates, and any
render job class until the spike is green (`FEAT-timeline-render-job.md` remains
research-only).
