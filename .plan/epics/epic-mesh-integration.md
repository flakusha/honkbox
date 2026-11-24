# EPIC: Mesh integration and external routing

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/media-transport-federation-2026-11-20.md`,
`.tmp/research/gen-flows-props-characters-locations-2026-11-20.md`

## Scope

honkbox as an opt-in mesh **client** for chat/embeddings, with truthful external
provider advertisement and bounded fallback policy. The reverse direction
(honkbox as a mesh federated provider) is tracked as a mesh-lane assessment and
is not a honkbox runtime feature by default.

## Tickets

- `TASK-mesh-capability-advertisement.md` — external provider flags (P1).
- `TASK-mesh-client-spike.md` — chat-only upstream proxy (P2).
- `TASK-mesh-provider-discovery.md` — bounded `/mesh/v1/models` cache (P2).
- `TASK-reroute-policy.md` — retries/fallbacks/cooldowns, fail-closed (P2).
- `TASK-proxy-stream-safety.md` — one connection/request, SSE, cancel (P2).
- `TASK-external-api-catalog-links.md` — RFC 8631/9727 links (P3).
- `TASK-honkbox-federated-provider-assessment.md` — mesh-lane, likely non-feature (P3).

## Acceptance

No default cloud fallback; private scopes and provider pins fail closed; a
remote turn holds no local GPU lease; `/v1/capabilities` is the canonical source
for external-provider state.

## Verification

`bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run lint:rs`;
mesh stub only, no live network.

## Non-goals

Mesh identity/keypair storage, payment settlement, MCP surface in honkbox, and
default provider plugins inside honkbox.
