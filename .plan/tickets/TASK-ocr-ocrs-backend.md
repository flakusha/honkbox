# TASK: OCR backend — ocrs (pure Rust)

Status: proposed
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: Latin-script image OCR with word boxes + confidence via `ocrs`/`rten` (CPU, no foreign builds), models from catalog `ocr:` slugs, FEAT-ocr-ingest cache key honored.
Verification: `cargo test --locked -p honkbox-recognition-worker`, `bun run test:server`, `bun run codegen:check`, `bun run check`
Links: `.tmp/research/ReviewedRecognition.md` §1 (ocrs row), §Uncertainty; FEAT-ocr-ingest; TASK-recognition-provider-trait
Resources: per-test fixture images + cached model dir; no shared engine process

## Change

- Feature `recognition-worker/ocr-ocrs`: `ocrs = "0.14"` + `rten` (both MIT OR Apache-2.0,
  pure Rust — no cmake/foreign build; evidence: crates.io deps API shows rten 0.28.0 with zero
  build-deps, no cc/cmake/bindgen, `lib_links: null`, rten-simd/rten-onnx zero-dep, ocrs
  `build = false`; the `cargo test --locked` verification below is the compile probe).
- Registry: `RootKind::Ocr` + `ModelRoots.ocr_dirs`, slug `ocr:<posix-rel>`, `ModelKind::Ocr`;
  weights (`text-detection*.onnx/.rten`, `text-rec*.onnx/.rten`) are user-downloaded — honkbox
  never bundles them (**weights are CC-BY-SA-4.0**; attribution/share-alike applies to weight
  redistribution only, flagged in docs/NOTICE).
- Accuracy gate: fixture set (clean scan, screenshot, photo) must pass before this becomes any
  default; CJK expected weak → do not advertise as multilingual.
- Cache key: sha256(source) + `ocrs` + model version + language + dpi + mode.

## Acceptance detail

Missing weights ⇒ engine absent from capabilities, request → 501 naming it; unchanged file ⇒
cache hit; cancel mid-page stops promptly; labels `kind=ocr` only.
