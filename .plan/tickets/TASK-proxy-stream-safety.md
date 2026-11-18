# TASK: Proxy stream safety

Status: done (mesh-proxy lane, commit e5f1a72 — client foundation, stream rules, no-lease proof; route mounting deferred to TASK-reroute-policy)
Priority: P2
Epic: epic-mesh-integration
Owner: unassigned
Acceptance: one framed upstream connection per request, SSE relayed without buffering, cancel propagated, and no lease held for a remote-only turn.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedMesh-ProxyRerouting.md` §Stream rules
Resources: mesh stub on assigned port; per-test connection tracking

## Change

- Use `Connection: close` per request; no shared upstream multiplexing in v1.
- Stream SSE frames in order; no silent buffering or re-framing.
- Upstream cancel on client disconnect or `DELETE /v1/jobs/{id}`.

## Acceptance detail

A cancelled client request terminates the upstream response; no orphan stream
remains; no local GPU lease is acquired.
