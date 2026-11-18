# TASK: Cap SpeechRequest input

Status: done (2026-11-18, LANE-impl-wave)
Priority: P0
Epic: epic-output-formats
Owner: LANE-impl-wave
Acceptance: `/v1/audio/speech` rejects oversized `input` with 400 instead of allowing unbounded text → PCM response growth.
Verification: `cargo test --locked -p honkbox-server`, `bun run lint:rs`
Links: `.tmp/research/ReviewedOutput-Formats.md` §6 (blocking-grade finding)
Resources: no external resources; request-body fixtures only

## Change

- `SpeechRequest.input` is currently an uncapped `String`; route checks only
  `trim().is_empty()`.
- Choose an explicit character cap (or derived PCM duration cap) and reject
  with 400 — matching the repo's reject-not-clamp policy for `cfg_scale` and
  `strength`. Apply the same reasoning to image `prompt` in a follow-up if needed.

## Acceptance detail

A request over the cap returns 400 with a clear message; under-cap requests are
unchanged; no cap is enforced by silent truncation.

## Implementation note (2026-11-18)

- Cap picked (the ticket named none): **4096 characters** — the `OpenAI` TTS
  precedent — `SPEECH_INPUT_MAX_CHARS` in routes.rs, counted via `chars()`
  (characters, not bytes).
- Over-cap is a named 400, code `input_too_long`, message naming the actual
  count and the cap plus "rejected, not truncated". Under-cap requests are
  byte-identical to the pre-ticket behavior; the utoipa 400 description now
  mentions the cap.
- Tests (additive, `tests/audio.rs`): `speech_input_over_cap_is_rejected_not_clamped`
  (4097 ASCII + 4097 CJK chars both 400 — proves character counting) and
  `speech_input_at_cap_still_synthesizes` (4096 chars → 200 WAV).
- Image `prompt` cap deliberately NOT done here — the ticket scopes it as a
  follow-up.
