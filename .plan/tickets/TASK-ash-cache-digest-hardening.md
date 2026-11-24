# TASK: Harden ash cache key with model and tokenizer digests

**Status:** Done (2026-11-24, implementation on feat/ash-next-wave)
**Priority:** high
**Effort:** Small
**Epic:** epic-full-ai-cycle
**Tags:** ash, cache, determinism
**Caveats:** TOCTOU, security, authorization, parallelism, lock/unlock lifecycle, partial failure, rollback, stale state, data races, idempotency, observability

**Summary:**

Include model and tokenizer digests in the existing image cache key before lookup so distinct inputs cannot collide.

**Context:**

Discovery index §2.7 says the request-hash cache already exists; only digest hardening is missing. A second cache is explicitly a no-change verdict.

**Acceptance Criteria:**

- `model_sha256` is computed before cache lookup and appended to the canonical key.
- Tokenizer asset digests are appended when ash is active; distinct models or tokenizers never collide.

**Verification:**

- `cargo test --locked -p honkbox-server --features ash-sd` plus one focused collision test with distinct model digests.

**Links:**

- `.tmp/research/ash-next-wave-promotion-candidates-2026-11-24.md` (candidate 2)
- `ash-feature-discovery-index-2026-11-24.md` §2.7

**Non-goals:**

- No new cache layer, residency change, route or DTO change.

## Verification evidence (2026-11-24)

`images_generate.rs` now resolves `model_sha256` before cache lookup and folds it plus the ash tokenizer digest into `generation_cache_key`. Tokenizer assets are streamed through a 16 KiB `file_sha256` from `HONKBOX_ASH_CLIP_VOCAB_DIR` (both files); unset or missing assets use the stable `none` marker, never a fabricated hash. Unit tests cover content-sensitive file digests and distinct model/tokenizer keys. Focused server tests: 8 cache tests plus both ash conditioning guards passed; `cargo clippy -p honkbox-server --all-targets -D warnings` clean after declaring the undeclared ash-* features in the server manifest.
