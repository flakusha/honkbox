# TASK: Barcode/QR decode via rxing

Status: landed (stage 1, 2026-11-18 — synchronous `/v1/recognize` route; the job registry /
stable cache-key portion moves to the FEAT-ocr-ingest implementation, per the orchestrator's
stage-1 scoping. Full symbology coverage retained: rxing WITHOUT its `image` feature — the
worker decodes containers with the workspace `image` and feeds rxing luma buffers, resolving
rxing's hard `image =0.25.8` pin against the lock's `image 0.25.10` (diffusion-rs `=0.25.10`).)
Priority: P1
Epic: epic-full-ai-cycle
Owner: recognition lane
Acceptance: `RecKind::Barcode` backend decodes QR + 1D barcodes from image bytes in-process (pure Rust), exposed as a cancellable recognition job with a stable cache key.
Verification: `cargo test --locked -p honkbox-recognition-worker`, `bun run test:server`, `bun run codegen:check`
Links: `.tmp/research/ReviewedRecognition.md` §2 (rank 1); TASK-recognition-provider-trait
Resources: per-test image fixtures (generated in-test: rxing encodes the QR bit matrix, the
workspace `image` crate rasterizes — no committed fixture); no engine process

## Change

- Landed: `rxing` (Apache-2.0, pure Rust zxing port) behind `recognition-worker/barcode` and the
  server `barcode` feature (default-on; server forwards it so capabilities cannot drift).
- Error semantics (tightened over the draft): undecodable container → named 400
  `invalid_recognize_input` (a truncated PNG must not look like "no barcode"), decodable image
  with no barcode → clean empty result `pages: []` (never a 500), empty/missing payload →
  named 400. Result: symbology payload + corner-derived axis-aligned bbox, `confidence: null`
  documented as engine-specific (rxing produces none).
- License/dep evidence: `deny.toml` green (`bun run deny`) with rxing 0.9 + tree
  (chrono/num/once_cell/regex/thiserror/unicode-segmentation + oned derive + encoding_rs,
  all MIT/Apache-2.0 class); no `image` feature → no image pin conflict.
- Cache key (stage 2, FEAT-ocr-ingest): sha256(bytes) + `rxing` + engine version; decode is
  deterministic (asserted by tests) so replay is safe.

## Acceptance detail

Undecodable container answers the named 400; no-barcode answers the clean empty result; metric
labels carry `kind=barcode` only; capabilities `recognition.barcode` == `BARCODE_COMPILED`
(contract-tested both directions). QR round trip proven end to end on a generated fixture.
