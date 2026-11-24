# EPIC: Voice identity (consent-gated)

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/model-identity-license-download-2026-11-25.md`,
`.tmp/research/media-audio-normalization-and-cleanup-2026-11-24.md`

## Scope

Speaker diarization and voice cloning are external adapter jobs only. No voice
cloning route may exist before a recorded consent/license policy; diarization
annotations go into transcript sidecars, not OpenAI-compatible DTO fields.
Keep `onnxruntime` and Python out of the Rust workspace.

## Tickets

- `TASK-diarization-spike.md` — local sherpa-onnx fixture + speaker/time JSON
  (P2; depends on `FEAT-streaming-stt.md`).
- `TASK-voice-clone-policy.md` — consent/license policy + named error (P3).

## Acceptance

A reference voice without consent is rejected with a named error; diarization
output is local, versioned, and replayable from the job bundle.

## Verification

`bun run test:server`, `bun run lint:rs`, `bun run deny`; local audio fixtures
and locally supplied adapter models only, no CI downloads.

## Non-goals

Impersonation, cloud voice services, speaker identity database, voice cloning
from arbitrary URLs/STT input, and realtime WebSocket parity.
