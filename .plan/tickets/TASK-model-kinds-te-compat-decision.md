# TASK: Model kinds text-encoder compatibility decision

Status: proposed
Priority: P3
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: a recorded decision picks the text-encoder compatibility rule (n_embd cross-check vs width-ladder only) and the music-vs-tts boundary for singing/vocoder arch values, with fixture-based acceptance criteria for a future implementer
Verification: `giwt plan check-links` on this file; no code commands (design-only)
Links: `.tmp/research/model-recognition-classification-2026-11-26.md` (UNKNOWN items), `TASK-model-kinds-server-coverage.md`
Resources: no tmp dirs, no weights, no GPU (no code, no fixtures executed)

## Change (design-only; implementation explicitly out of scope)

- Decide: text-encoder compatibility = `n_embd` cross-check against the paired text tower, or width-ladder class only. Record rationale and the rejection case for the loser.
- Decide: music-vs-tts boundary for singing/vocoder arch values — which ModelKind each maps to and why (audition criteria, not weights).
- Specify fixture-based acceptance for the future implementation ticket: synthetic header/arch fixtures, expected kind per fixture, per-test tmp dirs, no real weights/downloads/GPU.

## Acceptance detail

- Both UNKNOWNs from the research doc are resolved to a named rule each; no new UNKNOWNs introduced.
- No code changes in this ticket; implementation belongs to a follow-up ticket.

## Decision (2026-11-26)

### 1. Text-encoder compatibility rule (chosen)

Width/depth ladder as primary (header-only, no sibling needed) + `n_embd`
cross-check against the sibling chat model when present; mismatch →
`compatible:false`.

- Ladder-only: safetensors key cascades (`text_model.encoder.layers.N`,
  `encoder.block.N`, `model.layers.0…` widths) decide text-encoder candidacy
  from the header alone, following the ComfyUI `model_detection.py` /
  `sd.py::detect_te_model` idiom. A TE file with no sibling chat model is
  still classifiable (`compatible:true` when on-ladder).
- Cross-check: when a sibling chat model is present (existing `sibling_mmproj`
  pairing idiom, generalized to chat-model pairing), compare the TE width
  against the chat model's `n_embd` (llama.cpp `mtmd` idiom:
  `llama-model-loader.cpp:581`, `mtmd-model.cpp:133` `clip.*` keys). Mismatch
  → `text-encoder{compatible:false}`, never a silent pass.
- Rejected alternative: width-ladder only with no cross-check. Rejected
  because a ladder match can still be the wrong tower for the paired chat
  model (e.g. SD-1.x-width TE next to an SDXL-width chat tower); without the
  `n_embd` check the mismatch surfaces only as garbage conditioning at
  inference time. Cost of the check is one KV read on an already-paired
  sibling, so the ladder stays cheap and the check only fires when pairing
  data exists.
- Also rejected: `n_embd`-only (sibling mandatory). Rejected because TE files
  are usable standalone in SD pipelines; requiring a sibling would make
  single-file scans unclassifiable and reintroduce the misfile-as-`sd`
  fallback the ladder exists to kill.

### 2. Music-vs-tts boundary (chosen)

Vocoder/singing arch values → `tts` kind, unless the header names a
music-only role, in which case → `music`.

- Default `tts`: GGUF `general.architecture` values in the speech-synthesis
  family (`QWEN3TTS`, `KOKORO`, `MOSHI`, `ZONOS3`, `TINY_VOCA`, …) and any
  singing/vocoder arch value keep the existing `tts` kind. Audition criterion:
  the arch's upstream definition is speech/singing-voice synthesis, even when
  the synthesis path runs through a vocoder or singing-voice model.
- Exception `music`: only when the header (`general.architecture` /
  `general.type`) names a music-generation role whose upstream definition is
  music output, not speech — no filename guessing, per research item 6.
- Rejected alternative: a separate kind per vocoder/singing arch value (or
  routing them to `music` by default). Rejected: YAGNI — no consumer
  distinguishes vocoder-backed TTS from plain TTS today, and defaulting
  singing voices to `music` would hide them from the TTS capability surface
  they actually serve.

### 3. Fixture cases a future implementation must cover

All fixtures synthetic (hand-built GGUF KV maps / safetensors JSON headers in
per-test tmp dirs); no real weights, no downloads, no GPU:

- TE-1 (ladder-only pass): safetensors header with
  `text_model.encoder.layers.0…11` keys, width 768, no sibling file →
  `text-encoder{compatible:true}`.
- TE-2 (cross-check pass): TE-1 header + sibling chat model with `n_embd`
  768 → `text-encoder{compatible:true}`.
- TE-3 (cross-check fail): TE-1 header + sibling chat model with `n_embd`
  4096 → `text-encoder{compatible:false}`.
- TE-4 (off-ladder): safetensors header with `model.layers.0…` keys at a width
  on no known ladder rung → not text-encoder (falls through to `unknown`,
  never `sd`).
- TE-5 (GGUF TE arch): GGUF header with a text-encoder `general.architecture`
  value → `text-encoder{compatible:true}` without any sibling.
- AU-1 (tts arch): GGUF `general.architecture=KOKORO` → `tts`.
- AU-2 (singing/vocoder): GGUF header with a singing/vocoder arch value →
  `tts`, not `music`.
- AU-3 (music-only role): GGUF header with a music-generation-only
  `general.architecture` value → `music`.
- AU-4 (no filename guessing): TTS/vocoder tensor content under a
  music-suggestive filename, header arch naming speech synthesis → `tts`
  (filename never overrides header).
