# TASK: Barcode/QR decode via rxing

Status: proposed
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: `RecKind::Barcode` backend decodes QR + 1D barcodes from image bytes in-process (pure Rust), exposed as a cancellable recognition job with a stable cache key.
Verification: `cargo test --locked -p honkbox-recognition-worker`, `bun run test:server`, `bun run codegen:check`
Links: `.tmp/research/ReviewedRecognition.md` §2 (rank 1); TASK-recognition-provider-trait
Resources: per-test image fixtures; no engine process

## Change

- Add `rxing` (Apache-2.0, pure Rust zxing port) behind `recognition-worker/barcode`; fallback
  candidate `rqrr` (dual MIT/Apache) only if rxing misses a needed symbology.
- Route registers a cancellable job; result: symbology, payload, corners (bbox), confidence n/a →
  `confidence: null` documented as engine-specific.
- Cache key: sha256(bytes) + `rxing` + engine version; decode is deterministic.

## Acceptance detail

Corrupt/no-barcode input is a clean empty result (not a 500); metric labels carry `kind=barcode`
only; capabilities report `barcode: BARCODE_COMPILED`.
