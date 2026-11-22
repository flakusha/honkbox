# FEAT: residency drain honesty in the shipped readout

Status: done (2026-11-22)
- Verified 2026-11-22: `bun run check` and `bun run codegen:check` green; regenerated bindings cover `idle_secs` and `features.residency_control` (`.tmp/final-integration-verify.log`).
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: `GET /v1/memory` and `GET /running` expose an additive `idle_secs` per resident (seconds since the pipeline's last use) alongside the existing best-effort `in_use` flag, and the readout contract states plainly that an evicted slug leaves the map at once while its pipeline can stay alive until the last in-flight `Arc` drops — "not listed" is not "drained". No new routes and no lease-holder field; an additive `features.residency_control` capability flag is derived from swap-manager presence.
Verification: `cargo test --locked -p honkbox-server`, `bun run codegen:check`, `bun run lint:rs`; tests cover idle-seconds projection, additive decode of both readout DTOs, and unchanged pin/unload outcomes.
Links: `.tmp/research/cli-app-management-llama-server-2026-11-22.md:19`; `crates/server/src/swap.rs:43-48,131-132,159-177,329-339,360-386`; `crates/api-types/src/lib.rs:2170-2197,2238-2248`; `crates/server/src/routes/runtime.rs`; `crates/server/src/routes/models.rs:120-149`; `crates/server/src/routes.rs:384-387,425-427,464-467`; `crates/server/src/routes/capabilities.rs:205-207`; `FEAT-cli-app-manager.md`.
Resources: per-test temp data dirs and stub LLM pipelines; test server on a port offset/port 0, no new fixed port; no GPU.

## Verified premise

`.tmp/research/cli-app-management-llama-server-2026-11-22.md:19`: `SwapManager` already implements llama-swap-style TTL/LRU/pin/unload for in-process pipelines only. The control surface already ships: `GET /running` and `GET /v1/memory` (`routes/runtime.rs`), `POST /api/models/unload`, `POST /api/models/unload/{model}`, `POST /v1/models/{slug}/unload` (`routes/models.rs`), registered in `routes.rs:384-387,425-427` behind the router-wide bearer layer (`:464-467`). Proposed `GET /v1/residency` / `POST /v1/residency/release` would duplicate these and are dropped. `FEAT-cli-app-manager.md` manages a CHILD llama-server app; this ticket only makes honkbox's OWN ledger honest.

## Real gap / change

- `evict_by_slug` removes the map entry immediately (`swap.rs:329-339`) while the pipeline survives until the last `Arc` holder drops (`swap.rs:43-48`). A `/running` poller waiting for the id to disappear therefore reports "done" too early; document that reading and add `idle_secs` so callers can distinguish a fresh entry from one near expiry.
- `last_use` is private (`swap.rs:131-132`) and not projected into `ResidentEntry` (`swap.rs:159-177`). Compute `idle_secs` in `SwapManager::residency` (`swap.rs:360-386`) and add it additively to `ResidentModel` (`api-types:2170-2197`) and `RunningModel` (`api-types:2238-2248`); regenerate bindings via `bun run codegen:check`.
- No lease holder: residency leases are per `ResourceKind` (`crates/residency/src/lib.rs`), not per model slug. Keep the best-effort per-slug `in_use` already derived from `Arc::strong_count` (`swap.rs:170-176,381`); never promise a drain barrier.
- Capability flag: the additive `features.residency_control` derives from swap-manager presence (`config.swap.is_some()`), never hardcoded `true` (precedent `artifact_delivery: config.data_dir.is_some()` at `routes/capabilities.rs:207`); the route set does not change.
- Queue wording: a subsequent request for the released slug re-loads it on demand; that is the documented behavior, not a silent re-load by a queued waiter.
