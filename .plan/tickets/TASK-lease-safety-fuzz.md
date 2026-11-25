# TASK: Lease safety fuzz

Status: proposed
Priority: P2
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: property/fuzz driver over ledger acquire/cancel/timeout/drop sequences returns slot counts to baseline on every path
Verification: `cargo test --locked -p honkbox-residency --lib lease_fuzz`
Links: `.tmp/research/api-provisioning-extended-fields-2026-11-25.md` §5; `.plan/tickets/TASK-resource-ledger-bounded-queue.md`; `crates/residency/src/lib.rs`
Resources: per-test ledger instances; no global slot reused across tests; port 0; no shared state

## Change

- Runtime artifact only (test/driver code); the audit-hook static gate is untouched.
- Fuzz driver: single-threaded model of demand counters plus the real `Ticket`/`Lease` types; no GPU needed; assert baseline after drain.
- Cover sequences F1-F9: immediate drop, pre-admission cancel, queue TTL timeout (counts as eviction, `evicted` +1), normal admit-then-drop freeing exact `Demand`, engine-crash teardown, `Oversized` with unchanged queue, `QueueFull` + `Retry-After` hint with order preserved, idempotent double-drop/double-cancel, randomized interleave with baseline oracle.
- Exact TTL/eviction-counter names are [UNKNOWN] until the residency follow-up lands; keep them symbolic.

## Acceptance criteria

- [ ] Every F1-F9 sequence ends with all demand counters at baseline.
- [ ] Oversized/QueueFull paths leave queue state/order unchanged with the specified errors.
- [ ] No changes to the audit-hook static gate.
