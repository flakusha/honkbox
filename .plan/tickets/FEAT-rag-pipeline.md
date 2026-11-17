# FEAT: Local RAG pipeline

Status: proposed
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: ingest → chunk → embed → hybrid retrieve → rerank → cited answer, with versioned index and a labelled local eval gate.
Verification: `bun run check`, `bun run test`, `bun run codegen:check`
Links: `.tmp/research/ReviewedFullCycle-RAG.md`, depends on `FEAT-ocr-ingest.md`

## Change

- Start with sqlite-vec + FTS5 and a custom RRF tie-break by chunk ID; LanceDB
  only when corpus size justifies Arrow machinery. Store under `<data_dir>/rag/`
  with `.part` + fsync + rename.
- Embeddings via honkbox `/v1/embeddings` (verified route). Rerank uses the local
  llama.cpp engine; honkbox has **no** `/v1/rerank` route today.
- Version parser, chunker, embedding model, and reranker; re-embed on any change.
- Evaluate on 30–50 labelled local queries (recall@k / nDCG@10 + latency) before
  enabling rerank by default.

## Acceptance detail

Citations carry chunk IDs; retrieved text never becomes a system instruction;
embedding/rerank batches hold leases and are cancellable jobs.
