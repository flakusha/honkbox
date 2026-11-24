# EPIC: Audio streaming and quality

Status: in progress
Priority: P1
Owner: unassigned
Research: `.tmp/research/media-masters-audio-video-2026-11-25.md`,
`.tmp/research/media-audio-normalization-and-cleanup-2026-11-24.md`,
`.tmp/research/next-tickets-implementation-plan-2026-11-22.md`

## Scope

The core bounded-streaming lane is landed: `/v1/audio/transcriptions?stream=true`
emits ordered SSE partial/final frames over bounded windows, optional Silero VAD
chunks speech, and the audio worker uses a fixed-tap Lanczos-windowed sinc
resampler. File-only transcription and WAV/FLAC/Opus speech output remain
available. This is bounded whole-upload processing, not socket-driven realtime
ASR.

Remaining work is the optional audio.cpp provider/streaming expansion, an
explicit VAD consolidation decision, and durable audio master/derived manifests
owned by `FEAT-media-master-and-derived-artifacts.md` in `epic-output-formats`.
No audio normalization, cleanup or loudness processing is currently implemented;
any future processing must be opt-in and measured.

## Tickets

### Landed core lane

- `TASK-audio-vad-surface.md` — done: optional `audio-worker/vad` forwarded and consumed by streaming STT; no standalone VAD route.
- `FEAT-streaming-stt.md` — done: bounded SSE windows, registry job, cooperative cancel and kind-labeled metrics.
- `TASK-audio-resampler-quality.md` — done: windowed-sinc resampler with fixed DC normalization and 8k→16k/44.1k→16k coverage.
- `TASK-audio-voices-loaded-backend-truth.md` — done 2026-11-22: `/v1/audio/voices` reflects the loaded TTS backend.
- `TASK-audiocpp-vendor-spike.md` — done: offline/linkage spike passed.
- `TASK-audiocpp-stt-provider.md` — done at worker/CLI seam; capability advertisement split to `TASK-audiocpp-capabilities.md`.
- `TASK-audiocpp-capabilities.md` — done: feature/family advertisement is compile-probe truth.

### Remaining expansion

- `TASK-audiocpp-streaming-stt.md` — proposed P3: pull-based audio.cpp streaming sessions behind the existing SSE contract.
- `TASK-audiocpp-tts-provider.md` — proposed P2: verified eSpeak-free audio.cpp TTS family.
- `TASK-audiocpp-vad-decision.md` — proposed P3: retain `ort` Silero or migrate to bundled audio.cpp Silero with identical feature semantics.
- `FEAT-media-master-and-derived-artifacts.md` — proposed P1 in `epic-output-formats`: durable PCM16/WAV master, derived final and deterministic audio metadata.

## Acceptance

Landed streaming jobs use the same admission, lease, cancel and kind-labeled
metrics as chat/image; VAD is not a standalone toy route; resampler tests cover
8k→16k and 44.1k→16k with no aliasing regression. Current hard bounds remain
5 s windows, 60 s total, 2048 frames, an 8-frame channel and a 90 s deadline.
Future audio.cpp streaming must preserve those bounds and release residency on
cancel, disconnect, timeout and failure. Durable audio records must state exact
PCM16 layout/rate/sample count and never claim loudness normalization without a
measured method and explicit applied transform.

## Verification

`cargo test --locked -p honkbox-server`, `bun run test:server`, `bun run lint:rs`;
short local audio fixtures only. Feature-on audio.cpp tests remain opt-in and
use operator-supplied local models, never CI downloads.

## Non-goals

Diarization, voice cloning, WebSocket/socket-driven realtime parity, cloud
fallback, default FFmpeg/PyTorch/normalization dependencies, and silent audio
cleanup, gain changes, limiting or loudness correction.
