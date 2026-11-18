# TASK: Recognition provider trait and capability surface

Status: proposed
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: one `RecognitionPipeline` trait in `crates/recognition-worker` (backend_id/kinds/recognize/cancel) with a stub backend; recognition DTOs in `crates/api-types`; `recognition` block in `/v1/capabilities` driven by compile probes.
Verification: `cargo test --locked -p honkbox-recognition-worker`, `bun run test:server`, `bun run codegen:check`, `bun run lint:rs`
Links: `.tmp/research/ReviewedRecognition.md` §3, §4; `.plan/tickets/FEAT-ocr-ingest.md`
Resources: none new; stub backend only, no engine process

## Change

- New crate `crates/recognition-worker` mirroring audio-worker: trait `RecognitionPipeline`
  (`backend_id()`, `kinds()`, `recognize()`, `cancel()` with the LlmPipeline cancel contract),
  `StubRecognition`, request/result types, hand-rolled errors.
- DTOs in `crates/api-types`: `RecognizeRequest` (kind, source, language, mode, engine?, dpi/scale
  bounds), `RecognizeResult` (pages, regions with bbox/text/confidence, engine meta,
  `text_layer_used`, `bbox_support`); regenerate `packages/web-ui/src/gen/`.
- Compile probes `OCR_COMPILED`/`BARCODE_COMPILED`/`TAG_COMPILED` (`cfg!`, VAD_COMPILED pattern);
  `capabilities_document` gains `features.recognition` with probe-driven flags and live catalog
  counts; contract test cross-checks flags against the document.
- No route yet beyond a 501 placeholder naming the missing engine capability.

## Acceptance detail

A stub-only build reports `recognition` flags honestly; the capability document cannot drift
from the binary; no metric labels beyond `kind`.
