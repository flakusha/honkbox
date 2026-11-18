# TASK: Mesh client spike (chat only)

Status: done (mesh-proxy lane, commit e5f1a72 — [mesh] config + client + SSE/cancel/header contract tests; chat route consumption deferred to TASK-reroute-policy per lane scope)
Priority: P2
Epic: epic-mesh-integration
Owner: unassigned
Acceptance: opt-in upstream config forwards one `/v1/chat/completions` turn to a mesh-compatible endpoint with SSE and cancel propagation.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedMesh-MeshLLmIntegration.md`, `.tmp/research/ReviewedCreative-MeshRemoteAPI.md`
Resources: mesh stub on assigned port; per-test temp config, no live network

## Change

- Add `upstreams.mesh` config (base URL + bearer token, default off).
- Proxy only chat completions in this spike; no embeddings, no reroute.
- Hold no local GPU lease for a remote turn; record provider/model/scope in the
  job sidecar.

## Acceptance detail

A remote answer arrives as a normal OpenAI-shaped response; cancel reaches the
upstream; no default fallback exists.
