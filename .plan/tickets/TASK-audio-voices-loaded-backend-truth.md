# TASK: Make /v1/audio/voices reflect the loaded TTS backend

Status: done (2026-11-22)
- Verified 2026-11-22: `bun run check` and `bun run codegen:check` green; voices dispatch on loaded `backend_id()` with scoped-empty non-Kokoro response (`.tmp/final-integration-verify.log`).
Priority: P2
Epic: epic-audio-streaming
Owner: unassigned
Acceptance: `GET /v1/audio/voices` never presents the static Kokoro voice table as the supported set when a different TTS backend or family is selected; advertised ids are a subset (`⊆`) of the `voice` values `/v1/audio/speech` accepts for that backend — `voice` passes through unvalidated (`crates/server/src/routes/audio_speech.rs:118`), so accepted ⊄ advertised — with an explicit scope (or an empty list) when the backend cannot enumerate.
Verification: `bun run test:server`, `bun run codegen:check` (`test:server` is `cargo test --locked -p honkbox-server`, `package.json:44`). Rewrite the voices contract tests `voices_for_non_kokoro_backends_are_empty` (`crates/server/tests/audio.rs:823-854`) and `voices_for_kokoro_backend_are_all_synthesizable` (`:856-896`, shim/helper `:792-821`): select Kokoro and non-Kokoro/stub backends and assert advertised ⊆ accepted by synthesizing each advertised id. Drop the `AUDIOCPP_TTS_FAMILIES` cross-check (`:850-853`, `:880-883`) — `audio_cpp.tts_families` is a compile-time audio.cpp list, not the loaded backend's voice set. Boundary: `.plan/tickets/TASK-audiocpp-tts-provider.md:16-21` adds family-specific enumeration through `backend_id()` later; this ticket only dispatches on the existing Kokoro id.
Links: `.plan/tickets/TASK-audiocpp-tts-provider.md:16-21`; loop-lore `../loop-lore/.tmp/intercom/FIXED-honkbox-capability-keys-2026-11-21.md` (families must come from worker compile tables, never client hard-coding).
Resources: existing stub TTS backend and in-process test server; no model downloads.

## Evidence (honkbox primary sources, read-only)

- `crates/server/src/routes/audio_voices.rs:54-75` — `get_audio_voices()` dispatches on `config.tts.backend_id()`: `kokoro-cpu` maps `KOKORO_VOICES` through `kokoro_voice_meta` (`:20-41`); every other backend returns `data: []`. Pre-fix code always mapped the static table.
- `crates/audio-worker/src/tts.rs:7-12` — the table's own comment: "A future any-tts enumeration surface should replace this list."
- `crates/api-types/src/lib.rs:1875-1899` — `Voice`/`VoiceList` docs explicitly describe the static, Kokoro-prefix-derived, model-free response.
- `.plan/tickets/TASK-audiocpp-tts-provider.md:16-21` — the future provider contract says voices are family-specific and must NOT clone the static `KOKORO_VOICES` pattern.
- `crates/server/src/routes/capabilities.rs:183-193` — `audio_cpp.tts_families` comes from compile-time worker constants: a different axis from the loaded TTS backend, not this ticket's cross-check.

## Why honkbox, not loop-lore

Which TTS backend is loaded and which voice ids it accepts is server-private state. loop-lore cannot derive it from a static id table, and capability-keys work established that clients must not hard-code audio families. A client picker built on today's response would offer Kokoro voices to a non-Kokoro backend.

## Acceptance detail

- Shipped decision (2026-11-22): dispatch on `config.tts.backend_id()` — `kokoro-cpu` serves the static `KOKORO_VOICES` table (it cannot enumerate), every other backend returns an unscoped `data: []` rather than another backend's ids.
- Every advertised `data[].id` is accepted by `/v1/audio/speech` for the selected backend (advertised ⊆ accepted; `voice` is unvalidated at `routes/audio_speech.rs:118`); no id outside the loaded backend's voice set is advertised.
- No additive scope field and no request filter in this ticket; keep `object: "voice"` and the list envelope stable.
- Regenerate TS bindings/OpenAPI; feature-off and non-Kokoro builds report honest absence via `backend_id()` dispatch. Do not cross-check against `audio_cpp.tts_families` (compile-time families of another backend).

## Operator decision (resolved 2026-11-22)

- Ship the scoped-empty posture now; do not wait for `TASK-audiocpp-tts-provider`. The future provider adds family-specific enumeration through `backend_id()` and never inherits the Kokoro table.
- No additive field, no request filter, and no loop-lore voice picker: an empty `data` is the explicit cannot-enumerate signal. Revisit only if a client must distinguish no voices from an unknown backend.
