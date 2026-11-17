# TASK: Diarization spike (sherpa-onnx)

Status: proposed
Priority: P2
Epic: epic-voice-identity
Owner: unassigned
Acceptance: one local sherpa-onnx fixture produces speaker/time-range JSON in the job bundle for a bounded audio sample; no OpenAI DTO field is added.
Verification: `bun run test:server`, `bun run lint:rs`, `bun run deny`
Links: `.tmp/research/ReviewedExpansion-VoiceIdentity.md` §Diarization landscape
Resources: per-test audio/model fixture dirs; no shared ONNX session; no CI downloads

## Change

- Depends on `FEAT-streaming-stt.md` + `TASK-audio-vad-surface.md`.
- Run sherpa-onnx as an external adapter via `epic-external-tools`; keep
  onnxruntime/Python out of the Rust workspace.
- Write speaker/time annotations into the transcript sidecar with model version
  and config hash; no new wire field without a version bump.

## Acceptance detail

Output is local, replayable, and structurally verifiable; no cloud fallback,
no speaker identity database, and no realtime parity in this ticket.
