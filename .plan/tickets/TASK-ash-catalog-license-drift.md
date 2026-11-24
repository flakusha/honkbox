# TASK: Fix ash-related catalog license drift

**Status:** Not Started
**Priority:** low
**Effort:** Small
**Epic:** epic-docs-tooling
**Tags:** ash, licensing, catalog
**Caveats:** TOCTOU, security, authorization, parallelism, lock/unlock lifecycle, partial failure, rollback, stale state, data races, idempotency, observability

**Summary:**

Correct stale catalog license data and add the missing `ash` NOTICE row through the license owner.

**Context:**

The licensing note records a 404 SDXL `license_url`, FLUX.1-schnell `gated` drift and an `ash` NOTICE gap. These are data fixes, not a change of license policy.

**Acceptance Criteria:**

- SDXL `license_url` points to a live `LICENSE.md`.
- FLUX.1-schnell gated status is verified against the catalog row.
- The `ash` NOTICE row is added by the license owner.

**Verification:**

- `bun run spdx`, catalog data tests and license-owner sign-off for the NOTICE row.

**Links:**

- `.tmp/research/ash-next-wave-promotion-candidates-2026-11-24.md` (candidate 7)
- `ash-vulkan-backend-licensing-extensibility-2026-11-24.md` §3-§5

**Non-goals:**

- No license-policy rewrite, model redistribution or vendored GPL code.
