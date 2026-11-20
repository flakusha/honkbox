# TASK: Per-consumer API scopes keyed by env name

Status: done (2026-11-20, auth lane)
Priority: P2
Epic: epic-authn-authz
Owner: unassigned
Acceptance: optional scope map keyed by api-key env NAME; scopes gate route families; a missing scope returns a named 403; legacy single-key mode is unchanged.
Verification: `bun run test:server`, `bun run audit:gpu-slot`
Links: `content-experience-gaps-2026-11-20.md` §7, `FEAT-capabilities-api.md`
Resources: per-test temp dirs; per-test env/key names; no shared state; no GPU lease acquired

## Change

- Add an optional scope map keyed by the api-key environment variable NAME, so
  each consumer can be granted a subset of route families. With no map
  configured, the legacy single-key mode behaves exactly as before.
- Gate route families by scope at the auth layer. A request whose key lacks the
  required scope returns a named 403 rather than a generic auth failure.
- Keep the scope definition in config, not code, and do not add per-route
  special cases beyond family-level checks.

## Acceptance detail

- A consumer with a scoped key can reach only its granted route families; a
  missing scope yields the named 403 with the required scope identified.
- With no scope map, every existing single-key request is authorized exactly as
  before — the change is opt-in.
- Scoped tests use distinct env/key names per test and acquire no GPU lease,
  keeping parallel runs isolated and the GPU-slot audit clean.
