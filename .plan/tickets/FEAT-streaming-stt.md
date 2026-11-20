# FEAT: Streaming STT with SSE partials

Status: done (2026-11-20, audio-vad/streaming-STT lane)
Priority: P1
Epic: epic-audio-streaming
Owner: unassigned
Acceptance: partial transcripts stream over SSE for bounded audio windows, with a registered `JobKind`, cooperative cancel, and bounded `kind` metrics.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`, `bun run codegen:check`
Links: `.tmp/research/ReviewedSupport-Audio.md` §Expansion table; whisper.cpp stream example
Resources: per-test SSE connections on assigned ports; per-test audio chunks; no shared GPU lease

## Change

- Depends on `TASK-audio-vad-surface.md`.
- Acquire `admission::acquire()` and hold a bounded residency lease for the
  stream; release on disconnect, cancel, or timeout.
- Register a `JobKind`, emit partial/final events, and cap window length and
  total duration. Keep file-only `/v1/audio/transcriptions` unchanged.

## Acceptance detail

One client cannot starve the slot; cancel stops the stream and releases the
lease; SSE frames are ordered and bounded.
