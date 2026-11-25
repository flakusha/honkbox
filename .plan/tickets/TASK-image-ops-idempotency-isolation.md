# TASK: Image-ops idempotency key isolation

Status: proposed
Priority: P3
Epic: epic-editor-integration
Owner: unassigned
Acceptance: ledger key is the (consumer, grant, attempt) triple; foreign keys rejected before ledger; foreign key never replays foreign bytes
Verification: `cargo test --locked -p honkbox-server --test image_ops_idempotency`
Links: `.tmp/research/api-provisioning-extended-fields-2026-11-25.md` §3; `crates/server/src/routes/image_ops_dedup.rs`; `crates/server/src/routes/image_ops_route.rs`
Resources: per-test tmp dirs; fixture JSON; port 0; no shared state

## Change

- Ledger key becomes the `(consumer, grant, attempt)` triple, not bare `request_id`: `request_id` stays opaque (max 128 chars, `[A-Za-z0-9_-]`), `consumer` is registry-checked, `grant` is the authorizing grant token id, `attempt` is the client attempt counter/uuid.
- Same key + changed command returns `409 idempotency-conflict` with `{expected_hash, got_hash}` over canonical command bytes; retry with the same triple replays.
- Unknown consumer/grant returns `401`/`403` before ledger lookup and creates no ledger entry.
- Bounded LRU + TTL eviction retained; evicted-key reuse re-executes honestly and is documented.

Neural-network weight/tokenizer license and inference-use applicability are the responsibility of the operator; honkbox records digest/source and never clears, grants, or assumes a license.

## Acceptance criteria

- [ ] Same triple replays; new attempt is a new key; changed command under the same key returns 409 with `{expected_hash, got_hash}`.
- [ ] Auth failures (401/403) happen before ledger lookup and leave no ledger entry.
- [ ] A foreign key never replays foreign bytes.
