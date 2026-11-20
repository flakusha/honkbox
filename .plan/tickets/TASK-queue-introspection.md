# TASK: Read-only queue introspection

Status: done (2026-11-20, queue introspection lane)
Priority: P1
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: read-only `GET /v1/queue` snapshot of the bounded GPU queue (job ids, kind, state, position, TTL) derived from the live ledger/registry; no reservation and no reordering.
Verification: `bun run test:server`, `bun run codegen:check`
Links: `TASK-resource-ledger-bounded-queue.md`, `FEAT-capabilities-api.md`
Resources: read-only; acquires no GPU lease; no shared mutable state

## Change

- Add a read-only `GET /v1/queue` route that snapshots the bounded GPU queue
  from the live resource ledger and external-tool registry — the same source of
  truth used for admission, not a parallel copy.
- Return job ids, kind, state, queue position, and TTL per entry. The handler
  must not reserve a slot, mutate state, or reorder the queue.
- Expose the DTO in the generated OpenAPI surface so the capabilities contract
  can advertise queue visibility.

## Acceptance detail

- The snapshot reflects live ledger/registry state and agrees with admission
  decisions; two reads without intervening work return the same queue.
- The route is side-effect free: no lease is acquired, no entry is removed, and
  ordering is unchanged after a read.
- Scope stays one route/DTO seam; reservation and reordering are out of scope.
