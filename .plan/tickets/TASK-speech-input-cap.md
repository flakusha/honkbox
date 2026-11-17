# TASK: Cap SpeechRequest input

Status: proposed
Priority: P0
Epic: epic-output-formats
Owner: unassigned
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
