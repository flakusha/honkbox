# TASK: VLM recognition profile (OCR/handwriting/charts via mtmd)

Status: proposed
Priority: P2
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: recognition requests can select the vision chat path (landed llm-vision) as an engine for kind=ocr (quality mode), handwriting, and chart/diagram description, with `bbox_support=false` honesty.
Verification: `bun run test:server`, `bun run codegen:check`, `bun run check`
Links: `.tmp/research/ReviewedRecognition.md` §1 (VLM row), §Uncertainty; FEAT-ocr-ingest; TASK-recognition-provider-trait
Resources: none beyond existing model roots; chat-route lease semantics unchanged

## Change

- New engine id `vlm-mtmd` in the recognition surface: maps `RecognizeRequest` → vision chat
  messages (data: URI image parts, per FEAT-response content-part support) with fixed prompt +
  sampling profiles per kind; per-page sequential calls; no bbox regions emitted.
- Reuses existing vision gates (mmproj sibling discovery + `VISION_COMPILED`,
  `vision_unsupported` 400); GPU/queue behavior identical to vision chat — no new lease class.
- Capabilities: `ocr.engines` includes `vlm-mtmd` only when a vision-capable model exists; DTO
  marks `bbox_support=false` for this engine so clients cannot expect regions.

## Acceptance detail

Handwriting fixture: VLM transcription ≥ Tesseract baseline or ticket records the delta;
seconds-per-page latency documented in the engine meta (`millis`); labels `kind=ocr|vlm` only,
never prompt text.
