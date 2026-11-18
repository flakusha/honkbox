# TASK: audio.cpp STT provider

Status: proposed
Priority: P2
Epic: epic-audio-streaming
Owner: unassigned
Acceptance: `AudioCppStt` implements `SttPipeline`, selectable via `--stt-backend audio-cpp --stt-model <slug>`, behind a non-default `audio-cpp` cargo feature; `/v1/audio/transcriptions` serves an audio.cpp ASR family (e.g. `qwen3_asr`) end to end; whisper and stub paths byte-identical when the feature is off.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`, `bun run codegen:check`; feature-off build compiles clean (no audio.cpp code linked); a transcript round-trip on a bundled/synthetic sample with a local GGUF fixture.
Links: `.tmp/research/ReviewedAudioCpp.md` §5, §6, §8; `crates/server/src/backends.rs`; `.plan/tickets/TASK-audiocpp-vendor-spike.md`
Resources: per-test temp WAV fixtures; one local audio.cpp ASR GGUF (no CI downloads); admission lease per request as today

## Change

- Gated on `TASK-audiocpp-vendor-spike.md` SPIKE PASS.
- `honkbox-audio-worker`: `stt.rs` gains `AudioCppStt` (session-per-model,
  mutex like `WhisperStt`); `AUDIOCPP_COMPILED: bool = cfg!(feature =
  "audio-cpp")` const following `VAD_COMPILED` (lib.rs:20 pattern).
- Family selection: `TranscribeRequest::model` carries the registry slug
  (`stt:<rel>`), family hint resolved from a compile-time feature-derived
  table; unknown family for this build → named 400 (REQ-8 unknown-part
  precedent), never a silent fallback.
- `backends.rs`: `SttBackend::AudioCpp` variant; `backend_id()`
  `"audiocpp-<family>"`.
- Stub/hermetic default untouched; `stt-vulkan`/whisper unchanged.

## Acceptance detail

Feature-off builds contain no audio.cpp code (link-surface check in the spike
report's method); feature-on builds pass the contract test asserting the route
advertises only families this build compiled; error paths (missing slug,
missing weights, non-16k input) surface as existing `SttError` variants with
named messages.
