# TASK: Recognition provider trait and capability surface

Status: landed (stage 1, 2026-11-18 — trait + stub + DTOs + `/v1/recognize` + `recognition`
capabilities block + `RootKind::Ocr` deferred to TASK-ocr-ocrs-backend; route is synchronous
until the OCR document pipeline lands)
Priority: P1
Epic: epic-full-ai-cycle
Owner: recognition lane
Acceptance: one `RecognitionPipeline` trait in `crates/recognition-worker` (backend_id/kinds/recognize/cancel) with a stub backend; recognition DTOs in `crates/api-types`; `recognition` block in `/v1/capabilities` driven by compile probes.
Verification: `cargo test --locked -p honkbox-recognition-worker`, `bun run test:server`, `bun run codegen:check`, `bun run lint:rs`
Links: `.tmp/research/ReviewedRecognition.md` §3, §4; `.plan/tickets/FEAT-ocr-ingest.md`
Resources: none new; stub backend only, no engine process

## Change

- Landed: `crates/recognition-worker` (trait with `cancel()` per the LlmPipeline contract,
  `StubRecognition`, hand-rolled `RecError`), DTOs (`RecKind`, `RecognizeRequest`,
  `RecognizeResult`, `RecognitionPage`/`Region`/`EngineMeta`, `RecognitionFeature`), compile
  probes `BARCODE_COMPILED`, additive-optional `Capabilities.recognition` block (image_ops/tools
  precedent), `POST /v1/recognize` with named 400/501 branches (`invalid_recognize_request`,
  `invalid_recognize_input`, `recognition_kind_unsupported`).
- Contract tests: `crates/server/tests/recognition.rs` (76) + `recognition-worker` unit tests
  (10). Probe agreement asserted in both directions (capabilities flag == route gate).
- Still open under this ticket's successor backends: `OCR_COMPILED`/`TAG_COMPILED` probes land
  with the ocr/tag tickets; `ocr:` catalog slugs land with TASK-ocr-ocrs-backend.

## Acceptance detail

A stub-only build reports `recognition` flags honestly; the capability document cannot drift
from the binary; no metric labels beyond `kind`. Stage-1 shared-gate evidence: worker 10/10,
recognition contract suite 76/76, `bun run test:server` green, `cargo deny` green,
api-types 15/15 (+gen/ regen). Full `lint:rs` deferred on sibling lanes' in-flight files
(artifacts.rs missing import, edits/images needless_pass_by_value, bindings duplicated attr)
— every lane-owned target scoped-green.
