# TASK: audio.cpp streaming STT partials

Status: proposed
Priority: P3
Epic: epic-audio-streaming
Owner: unassigned
Acceptance: pull-based audio.cpp streaming sessions (C ABI `stream_policy`/`stream_push`/`stream_finish`) drive SSE partial transcripts for a registered streaming `JobKind`, with cooperative cancel and bounded residency; FEAT-streaming-stt's acceptance is met using an audio.cpp streaming ASR family (Confucius4-R2T2 / Moonshine / Qwen3-ASR candidates).
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`; an SSE integration test asserts ordered, bounded partial frames and lease release on cancel/disconnect.
Links: `.tmp/research/ReviewedAudioCpp.md` §5, §8; `.plan/tickets/FEAT-streaming-stt.md`; `.plan/tickets/TASK-audiocpp-stt-provider.md`; audio.cpp C API streaming section (https://raw.githubusercontent.com/0xShug0/audio.cpp/master/docs/c_api.md)
Resources: per-test SSE connections on assigned ports; per-test audio chunks; one local streaming-family GGUF; no shared GPU lease

## Change

- Gated on `TASK-audiocpp-stt-provider.md` (provider seam) and the vendor
  spike; depends on `TASK-audio-vad-surface.md` for the window/lease shape.
- The binding's safe wrapper exposes streaming sessions (offline + streaming,
  per its README status checklist); events carry `partial_text` read through
  result accessors — no callback crosses FFI (pull-based, matching upstream).
- Admission: acquire and hold a bounded residency lease for the stream;
  release on disconnect, cancel, timeout; cap window length and total
  duration per FEAT-streaming-stt.
- Whisper file-only routes remain unchanged; this is the new stream surface.

## Acceptance detail

One client cannot starve the slot; cancel stops the stream and releases the
lease; partial frames are append-only for families with committed-text
semantics (R2T2) and are labeled family-appropriately otherwise; SSE frames
are ordered and bounded.
