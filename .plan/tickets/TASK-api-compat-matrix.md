# TASK: API compat decision matrix

Status: proposed
Priority: P2
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: each claimed compat row has a fixture proving wire behavior against the native route; broad shims listed as explicit non-goals
Verification: `cargo test --locked -p honkbox-server --test compat_matrix`
Links: `.tmp/research/api-provisioning-extended-fields-2026-11-25.md` §1; `crates/server/src/routes/chat.rs`; `crates/server/src/routes/embeddings.rs`
Resources: per-test tmp dirs; fixture JSON under `tests/fixtures/compat/`; port 0; no shared state

## Change

- Add one fixture per claimed row: `chat-completions-min.json`, `embeddings-min.json`, `stt-min.json`, `tts-min.json`, `images-native.json`, `models-list.json`.
- Claim only the subset in the matrix (Ollama-style chat core, single-input embeddings, whole-upload transcription, single-voice speech, native image DTO, registry-slug model list).
- Keep `keep_alive` TTL duration grammar as [UNKNOWN] operator decision (seconds vs duration-string).
- Explicit non-goals (reject): broad KoboldAI/Ollama/A1111/ComfyUI/Whisper/XTTS all-at-once shims; `functions`/`response_format`/`logit_bias` variants; batch/variable-encoding embeddings; streaming-voice/diarization params; full `voice+speed` matrix; A1111 `/sdapi/v1/*` and ComfyUI `/prompt` passthrough claimed as compat; Ollama `show`/`ps` detail and Modelfile round-trip.

## Acceptance criteria

- [ ] Each claimed matrix row has a fixture file proving wire behavior against the native route.
- [ ] Every non-goal shim from §1 is listed as rejected in the ticket/test, not silently untested.
- [ ] No licensing line needed (no weights/models involved).
