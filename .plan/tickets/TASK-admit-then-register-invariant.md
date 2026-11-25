# TASK: Admit-then-register invariant

Status: proposed
Priority: P1
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: ordering-invariant lint/test enforces admit-before-register on GPU-slot routes
Verification: `cargo test --locked -p honkbox-server --test admit_before_register`
Links: `.tmp/research/determinism-extended-fields-2026-11-25.md`, `TASK-queue-introspection.md`, `crates/server/src/jobs.rs`
Resources: per-test tmp dirs; port 0; no shared state

## Change

- Enforce admit-before-register ordering on GPU-slot routes via a lint/test: a lease must be held before a registry entry is created, closing the queued-vs-running overlap window.
- Cover the routes known to register before admitting today [UNKNOWN: full route list pending one code read to confirm current ordering].

## Acceptance criteria

- [ ] A test fails if any GPU-slot route registers before admitting.
- [ ] `GET /v1/jobs` and `GET /v1/queue` agree after the fix with no transient overlap.
- [ ] Scope stays one invariant seam; no scheduler policy changes.

Neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license.
