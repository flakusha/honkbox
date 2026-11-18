# TASK: Tesseract CLI engine (external supervisor lane)

Status: proposed — BLOCKED on TASK-external-tool-registry + TASK-process-supervisor
Priority: P2
Epic: epic-external-tools
Owner: unassigned
Acceptance: tesseract runs as a supervised external process (per-page args, timeout, RSS/thread caps, cancel, artifact collection) for 100+ language OCR; missing binary → 501 naming the tool.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`
Links: `.tmp/research/ReviewedRecognition.md` §1 (Tesseract row + license note); TASK-external-tool-registry; TASK-process-supervisor; FEAT-ocr-ingest
Resources: per-job temp working dirs; process groups killed in `finally`/drop

## Change

- Register tool `tesseract` in the external-tool registry (startup probe `tesseract --version`,
  cached, refresh on reload); availability reported with version; no workspace linking.
- License note for docs/NOTICE: engine Apache-2.0; Leptonica dependency is **BSD-2-clause-style
  (not GPL — resolves the "libpococr?" concern)**; `tessdata_fast` traineddata Apache-2.0. Nothing
  enters `Cargo.lock`.
- Execution via supervisor: `tesseract <page.png> stdout -l <lang> tsv` (or hOCR) per page; parse
  TSV → regions with word boxes + confidence; deterministic temp dirs; cancel kills the process
  group and releases any lease.
- In-process FFI (`tesseract`/`tesseract-sys`, both MIT) explicitly rejected: in-process C engine
  link violates the external-engine convention.

## Acceptance detail

Cancelled/failed/timeout jobs leave no orphan process, release the lease, and still write log +
sidecar; cache key includes tesseract version + traineddata identity; missing tool surfaces as
501 naming `tesseract`.
