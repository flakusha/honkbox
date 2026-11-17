# EPIC: Audio streaming and quality

Status: proposed
Priority: P1
Owner: unassigned
Research: `.tmp/research/ReviewedSupport-Audio.md`

## Scope

Build on the existing file-only STT/TTS routes: surface the already-compiled
Silero VAD through bounded streaming STT, and fix the resampler quality debt.
Existing input-cap and output-format tickets stay in `epic-output-formats`.

## Tickets

- `TASK-audio-vad-surface.md` — forward `audio-worker/vad` from server (P1).
- `FEAT-streaming-stt.md` — SSE partial transcripts over bounded windows (P1).
- `TASK-audio-resampler-quality.md` — windowed-sinc or audited crate (P2).

## Acceptance

Streaming jobs use the same admission, lease, cancel, and kind-labeled metrics
as chat/image; VAD is not a standalone toy route; resampler tests cover
8k→16k and 44.1k→16k with no aliasing regression.

## Verification

`cargo test --locked -p honkbox-server`, `bun run test:server`, `bun run lint:rs`;
short local audio fixtures only.

## Non-goals

Diarization, voice cloning, WebSocket/realtime parity, and cloud fallback.
