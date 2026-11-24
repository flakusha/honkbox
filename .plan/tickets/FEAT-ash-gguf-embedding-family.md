# FEAT: GGUF embedding and rerank family on ash

**Status:** Not Started
**Priority:** medium
**Effort:** Medium
**Epic:** epic-full-ai-cycle
**Tags:** ash, vulkan, gguf, embedding
**Caveats:** TOCTOU, security, authorization, parallelism, lock/unlock lifecycle, partial failure, rollback, stale state, data races, idempotency, observability

**Summary:**

Reuse the in-tree GGUF tensor-info reader to serve encoder-only embedding and rerank models through ash.

**Context:**

Discovery index §2.12 and the next-networks note mark this the cheapest genuinely new ash family. Inference only: weights stay user-supplied, so there is no product license issue.

**Acceptance Criteria:**

- Q8_0 dequant, RMSNorm, RoPE and pooling produce repeatable embeddings from a local GGUF.
- The existing GGUF reader is reused; no second parser, route or DTO is added.

**Verification:**

- `cargo test --locked -p honkbox-ash-backend --features ash-sd` with one local embedding GGUF fixture.

**Links:**

- `.tmp/research/ash-next-wave-promotion-candidates-2026-11-24.md` (candidate 6)
- `ash-feature-discovery-index-2026-11-24.md` §2.12

**Non-goals:**

- No bundled weights, license gate for inference-only weights, or new API surface.
