# TASK: OCR backend — PaddleOCR ONNX via ort

Status: proposed
Priority: P2
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: PP-OCRv4/v5 det+rec(+cls) ONNX models run on the pinned `ort` (CPU default; GPU EP only under a residency lease) producing line/word boxes + confidence, incl. CJK.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`, `bun run codegen:check`
Links: `.tmp/research/ReviewedRecognition.md` §1 (Paddle row + RapidOCR license declaration); FEAT-ocr-ingest
Resources: per-test fixtures; model cache dir; GPU lease per device-capability gates when CUDA EP enabled

## Change

- Feature `recognition-worker/ocr-onnx` reusing the in-tree `ort =2.0.0-rc.15` pin (vad pattern).
- Implement DB-det postprocess in Rust: threshold binarize → connected components/contours →
  unclip (polygon offset) → rotated-rect crops; CTC greedy decode + dictionary for rec; cls angle
  pass optional per request.
- Weights: official PaddlePaddle HF ONNX cards or RapidOCR registry entries — **Apache-2.0
  (authoritative RapidOCR declaration verified)**; record source URL + SHA-256 per catalog entry
  (registry db already supports metadata probes).
- Lease: CPU builds take none; CUDA EP build acquires the residency lease for the recognize
  duration; DPI/scale bounds enforced per FEAT-ocr-ingest.

## Acceptance detail

CJK fixture beats the ocrs backend or the ticket records why; cache key includes model file
hash; cancelled job releases the lease; `kind=ocr` labels only.
