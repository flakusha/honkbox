# TASK: Mesh provider discovery cache

Status: proposed
Priority: P2
Epic: epic-mesh-integration
Owner: unassigned
Acceptance: `/mesh/v1/models` is cached with bounded TTL and exposed as availability observations, not reservations.
Verification: `bun run test:server`, `bun run lint:rs`
Links: `.tmp/research/ReviewedMesh-MeshLLmIntegration.md` §3; `MODEL_DISCOVERY.md`
Resources: per-test stub server on assigned port; no shared cache

## Change

- Fetch on config load and explicit refresh; bounded TTL and entry cap.
- Store model id, provider id, workload class, `requires`, timestamp.
- Never treat a discovered model as admitted; admission still happens per request.

## Acceptance detail

A stale cache entry is labelled stale; a failed refresh keeps last-known data
with an error field instead of silently dropping providers.
