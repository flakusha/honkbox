# TASK: Bounded reroute policy

Status: done
Priority: P2
Epic: epic-mesh-integration
Owner: unassigned
Acceptance: retries/fallbacks are bounded, scope-aware, cooldown-limited, and fail closed on private scope or provider pins.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedMesh-ProxyRerouting.md`
Resources: multiple mesh stubs on assigned ports; per-test cooldown state

## Change

- Explicit retry budget, model-group fallback order, and per-provider cooldown.
- Private scope (`@private`, `x-mesh-scope: private`, `/private/v1`) never falls
  back to public nodes.
- Malformed `x-mesh-target`/`x-mesh-exclude` → 400; unavailable/conflicting →
  409; never silently reroute.

## Acceptance detail

Retry budget is never exceeded; a blocked target produces a named error; metrics
carry only bounded `kind` labels, never URLs or tokens.

## Verification Notes (2026-11-20)

Implemented `POST /v1/mesh/reroute`: capped budget (1..4), model-group fallback
order, per-provider cooldown, private scope fail-closed on unknown/public pins,
strict header parsing without echo, and bounded metric kinds. `bun run check`,
`bun run test:server`, and the 73-test `mesh_reroute` contract pass. Budget means
distinct candidates per turn; one dead upstream is not re-hit within a turn.
