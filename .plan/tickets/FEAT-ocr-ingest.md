# FEAT: OCR ingest stage

Status: done (2026-11-20; text-layer-first PDF + pure-Rust ocrs image path,
job/cache/DTOs landed; ocr_ingest + capabilities 293 tests green, codegen
regenerated; PDF raster modes return a named 501 until a licensed renderer)
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: text-layer-first document ingest producing page text, bounding boxes, confidence, engine version, and a stable cache key; registered as a cancellable job.
Verification: `bun run test:server`, `bun run codegen:check`, `bun run check`
Links: `.tmp/research/ReviewedFullCycle-OCR.md`
Resources: per-test input files and cache dirs; no shared OCR engine process

## Change

- Detect an existing PDF text layer before OCR; support OCR modes analogous to
  `pdf_aware_layout_regions`, `layout_regions`, `full_page`, and
  replace-broken-text-layer.
- External engine process/service only; no GPL or heavy Python deps in
  `Cargo.lock`. GPU work takes a lease; CPU fallback and DPI/`scale` are bounded.
- Cache by file hash + engine + language + DPI + engine/parser version.
- DTOs in `crates/api-types`: request, page, region/bbox, result metadata;
  regenerate `packages/web-ui/src/gen/`.

## Acceptance detail

Re-running an unchanged file is a cache hit; cancel stops mid-document; outputs
are written atomically into the job bundle. No file names or prompts in metric
labels (`kind=ocr` only).
