# TASK: Inference cache identity and memory-demand correctness

Status: proposed
Priority: P0
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: cache lookup is preceded by verified weight plus tokenizer/template/projector identity; existing caches never replay or retain a pipeline for changed bytes; weight and activation/working-memory demand are separately recorded and reserved; collision and same-path replacement tests pass.
Verification: `TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-server --test inference_cache_correctness -- --test-threads=1`; `TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-llm-worker --lib kv_cache -- --test-threads=1`; `bun run test:server`; `bun run codegen:check`; `bun run lint:rs`
Links: `.tmp/research/llm-runtime-memory-2026-11-25.md`; `.plan/tickets/TASK-llm-prompt-cache.md`; `.plan/tickets/TASK-determinism-contract.md`; `.plan/tickets/FEAT-residency-control-plane.md`; `crates/server/src/swap.rs:250-288,604-678,699-767`; `crates/server/src/determinism.rs`; `crates/server/src/routes/images_generate.rs:376-395`; `crates/llm-worker/src/backend.rs:119-152,429-432`; `crates/residency/src/lib.rs:126-159`
Resources: per-test gitignored `.tmp/` model/sidecar/cache dirs; in-process router/oneshot or port 0; test-local `Ledger`, never `global()`; stub pipelines by default (no GPU); any opt-in real-GGUF test serializes and selects one device with `GGML_VK_VISIBLE_DEVICES`.

## Premise

- `SwapManager` and `EmbedCache` are keyed by slug. A same-path replacement or
  changed sibling `mmproj` can reuse the old loaded pipeline.
- Inline image response keys include backend/slug/catalog/request hashes, but
  not the resolved checkpoint digest. Bundle provenance resolves the digest
  only later.
- Completed downloads already have verified `<target>.sha256.json` sidecars;
  `determinism.rs` already has canonical JSON and SHA-256. Reuse both.
- Chat and embedding requests create fresh llama.cpp contexts. There is no
  cross-request KV cache to fix here, and this ticket must not add one.
- The existing KV formula estimates activation state only. It ignores grouped
  query head widths and does not include weights, projector state, logits,
  decode batches, vision bitmaps, or allocator headroom.

## Change

### 1. Resolve identity before every cache lookup

- Resolve alias/catalog entry to concrete weight path(s), then resolve the
  tokenizer/chat-template source and any sibling projector path.
- Require a verified SHA-256 for every identity-bearing artifact before a hit
  can be returned. Use the existing download sidecar only when it describes
  the current file; otherwise stream the file once and persist through the
  existing sidecar seam. Missing/unverifiable identity bypasses the cache; it
  never falls back to slug-only reuse.
- Embedded GGUF tokenizer/template metadata is identified explicitly as
  embedded under the weight digest. An external tokenizer/template or mmproj
  contributes its own digest. Do not fabricate a separate digest for metadata
  that has no independent artifact.
- Build a domain-separated canonical identity record containing task/role,
  ordered artifact digests and roles, tokenizer/template source, projector
  digest, output-affecting load/runtime options, engine version, and code
  version. Hash it through the existing canonical JSON + SHA-256 seam.
- Keep the slug as the public alias. Key the existing resident entries and the
  existing inline response cache by identity. Do not add a second cache or a
  second identity database.

### 2. Make replacement and races safe

- Revalidate the resolved identity immediately before inserting a loaded
  pipeline or cached response. If the file generation/digest changed during
  lookup/load, evict and retry once or fail explicitly; never insert bytes A
  under identity B.
- A catalog refresh that changes the concrete path/digest must miss even when
  the slug is unchanged.
- A changed mmproj invalidates vision-chat identity even if the base GGUF is
  unchanged. A changed context/KV/offload option that can change output or
  memory is part of identity.

### 3. Separate demand components

- Return an internal demand breakdown from the loader with distinct
  `weights_bytes`, `projector_bytes`, `kv_bytes`, and `scratch_bytes` (including
  an explicitly named headroom component). Each component states whether it
  is measured, derived, or a conservative estimate.
- Reuse `kv_cache_bytes{,_per_token}` and the real model dimensions; preserve
  its documented conservative grouped-query behavior until an exact per-layer
  K/V calculation is available. Do not label that estimate measured.
- Map the placement-aware total into the existing `residency::Demand` byte
  fields. CPU-only placement uses RAM; offloaded KV/weights use VRAM; never
  count the same bytes in both dimensions.
- Acquire the demand-bearing lease before model load/allocation on chat,
  embeddings, and future cached inference paths. Loading a pipeline while
  holding only a slot is insufficient once byte budgets are configured.
- Expose the split additively in `GET /v1/memory` per resident: retain the
  existing weight `bytes` estimate and add `working_bytes` (KV + scratch +
  headroom, not weights). Regenerate bindings; keep absent values explicitly
  absent rather than inventing zero.
- Preserve the residency drain contract: eviction removes the map entry, but
  an in-flight `Arc` may retain memory until drop. Demand/readout wording must
  not claim a stronger drain barrier.

## Acceptance detail

The new `crates/server/tests/inference_cache_correctness.rs` proves:

1. Two different same-size files at one path, before and after replacement,
   produce different identities and force a resident reload.
2. Two distinct files with the same slug/path metadata cannot share a cached
   pipeline or inline response.
3. Changing only mmproj bytes changes vision identity and evicts the old
   pipeline.
4. Changing tokenizer/template source or an output-affecting option changes
   identity; embedded metadata is represented honestly.
5. Canonical identity/input tests reject ambiguous concatenation and cover
   key-order stability plus distinct-model/distinct-input vectors.
6. A replacement racing load cannot commit the stale pipeline; the request
   either retries against the new identity or fails by name.
7. A configured VRAM/RAM budget rejects before load when weight + working
   demand cannot fit, and releases exactly the committed demand on every
   error/cancel/drop path.
8. The same model with a larger context reports larger KV/working bytes but
   unchanged weight bytes; offload placement does not double-count bytes.
9. Existing image response-cache behavior still bypasses admission on a true
   identity hit and misses after a checkpoint replacement.

No numerical equality is asserted across backends; this ticket asserts cache
identity, accounting, and isolation only.

## Non-goals

- Persistent KV/prefix caching, llama-server attachment changes, or new
  response caches.
- New hash dependencies, a global digest service, or public backend flags.
- Cross-backend byte-identical output claims.
