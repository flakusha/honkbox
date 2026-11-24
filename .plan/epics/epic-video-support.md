# EPIC: Video support (gated)

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/ReviewedSupport-Video.md`,
`.tmp/research/ReviewedOutput-Formats.md`

## Scope

There is currently zero video code in `crates/`. Move from a documentation
blocker to a real, bounded job class only after an FFI spike proves the pinned
engine can run a local checkpoint. Parent blocker ticket stays `postponed`
until the spike is green.

## Tickets

- `FEAT-video-output.md` — parent blocker, **postponed** (migrated here).
- `TASK-video-ffi-spike.md` — support probe + one short clip (P2).
- `TASK-video-job-class.md` — `JobKind::video`, caps, lease, TTL (P2).
- `FEAT-video-i2v-frames.md` — PNG frame sequence + sidecar, no mux (P2).
- `FEAT-video-mux.md` — WebP/WebM flags + libwebm wiring (P3).

- `FEAT-comfy-video-audio-templates.md` — proposed external ComfyUI adapter for V2A/TV2A; no native video engine.
- `FEAT-video-source-audio-reuse.md` — proposed mix/context reuse of engine-provided audio, with provenance.

## Acceptance

No route or DTO exists until the spike lands; a video job never shares the
interactive GPU slot silently; cancellation states interruptibility honestly.

## Verification

`bun run test:core`, `bun run test:server`, `bun run lint:rs`, `bun run deny`;
opt-in local checkpoint only, no CI downloads. Structural frame checks only.

## Non-goals

Unlimited frames/edge, byte-identical cross-backend output, and external
AnimateDiff/SVDProcess adapters (those belong to `epic-external-tools`).
