# BUG: capabilities advertises rerank without a ranker model

Status: done (2026-11-22)
- Verified 2026-11-22: `bun run check` and `bun run codegen:check` green; `features.rerank` derives from `rerank_supported()` (`.tmp/final-integration-verify.log`).
Priority: P1
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: `features.rerank` is `true` exactly when `POST /v1/rerank` reaches the ranker seam without answering the named no-ranker 501; a python-enabled server with no ranker slug reports `false`, so clients do not route rerank traffic to a route that answers a named 501. A scoring round-trip is not asserted here: the harness has no sidecar venv.
Verification: `cargo test --locked -p honkbox-server` (includes the `rerank_capability` test binary: `python-disabled`, `python-enabled-no-slug`, `python-enabled-with-slug`), `bun run codegen:check`, `bun run lint:rs`. The three states use a separate test binary with re-exec per state (`crates/server/tests/rerank_capability.rs:205-226`) because `runtime_config::FACTS` is a first-wins `OnceLock` (`crates/server/src/runtime_config.rs:40,46`); the enabled+slug state asserts the true flag and absence of the 501 (`:191-200`), not a score.
Links: `../loop-lore/.tmp/intercom/FIXED-honkbox-rerank-2026-11-22.md:150-168`; `.plan/tickets/FEAT-rerank-endpoint.md:16`; `crates/server/tests/capabilities.rs:977-979`; loop-lore `src/memory/rerank.ts:82-100,145-191`.
Resources: one process per state, a unique temp config dir per state with teardown after each run; per-test config stubs only, no GPU or external model download.

## Evidence (honkbox primary sources, read-only)

- `crates/api-types/src/lib.rs:246-248` — the wire contract documents `rerank` as "whether a rerank route is mounted and backed by a model".
- `crates/server/src/routes/capabilities.rs:210` — current tree derives `rerank` from `super::runtime_config::rerank_supported()`; pre-fix (`python_enabled()`) consulted no ranker-model fact, so the working-tree edit is what closes this ticket.
- `crates/server/src/runtime_config.rs:125-136` — the shared gate `rerank_supported() = rerank_backed(python_enabled(), python_allowlist())`: Python enabled AND a non-empty first allowlist slug; `python_enabled()` alone reads only the config boolean.
- `crates/server/src/routes/rerank.rs:144-146` — without ranker support the route returns the named 501 (text at `:97-105`); `:150-153` still rejects an empty model slug. Flag and route guard must share one predicate.
- `crates/server/src/routes.rs:456-457` — `/v1/rerank` and its trailing-slash twin are both mounted, sharing that guard.
- `crates/server/tests/capabilities.rs:977-979` — the snapshot pins `rerank: false` with no ranker slug while `python_sidecar` tracks the Python switch; the old `rerank == python_sidecar` equality hid the no-model drift.

## Why honkbox, not loop-lore

The flag is derived in honkbox's capabilities document; loop-lore only reads it (`src/integrations/honkbox/capabilities.ts`) and has no way to know a ranker model is absent without probing each route. Fixing the derivation preserves the "every true flag has a working route" rule for all consumers.

## Acceptance detail

- Derive `features.rerank` from the same predicate the route uses to decide the 501: python enabled AND a non-empty allowlist slug configured. Loadability is not checked here. Do not add a request-time model load to `GET /v1/capabilities`.
- Leave `features.python_sidecar` semantics unchanged (`python_enabled()` is honest for that flag).
- Add the three-state contract test above and assert the wire snapshot still uses the existing `rerank: bool` field (no schema change, no `api_version` bump).

## Operator decision

- Configured, non-empty allowlist slug is the minimal honest rule: `features.rerank` is true only when Python is enabled and the first allowlist slug is non-empty. A catalog/loadability check is deferred; the sidecar still enforces per-request membership.

## Status note

Status is `done` (2026-11-22): the derivation edits (`runtime_config::rerank_supported`, the capabilities flag, the shared rerank guard) and the `rerank_capability` binary are committed and the full gate is green. The test verifies the contract; the deferred item is only a real scoring round-trip with a sidecar fixture.

## Test-binary note

`crates/server/tests/rerank_capability.rs` gives each state its own child process because `runtime_config::FACTS` is a first-wins `OnceLock`. The python-enabled + slug state asserts the true flag plus route admission (no named 501) rather than a score: this harness has no sidecar venv, so no scoring result is faked. A real scoring round-trip needs a sidecar fixture and stays deferred.
