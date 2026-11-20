# TASK: Capability-advertised short-lived signed content URLs

Status: done (2026-11-20, auth lane)
Priority: P2
Epic: epic-authn-authz
Owner: unassigned
Acceptance: capability-advertised short-lived signed content URLs scoped to one job, HMAC with a server-held secret, enforced expiry, no token in query logs, and bearer auth still works.
Verification: `bun run test:server`, `bun run codegen:check`
Links: `content-experience-gaps-2026-11-20.md` §7, `FEAT-capabilities-api.md`
Resources: per-test temp dirs; ephemeral HMAC secret per test; no shared state

## Change

- Add a capability row advertising short-lived signed content URLs, derived from
  config/runtime rather than prose.
- Issue an HMAC-signed URL scoped to a single job and a short expiry, keyed by a
  server-held secret. Enforce expiry at request time so a URL stops working
  after its lifetime even if the job still exists.
- Carry the credential in a header rather than the query string so it is absent
  from query logs; keep the existing bearer-token path working unchanged.

## Acceptance detail

- A signed URL fetches only its own job's content; a URL for job A cannot read
  job B, and an expired URL is rejected.
- The signature uses a server-held secret; the token never appears in query
  strings or access logs.
- Existing bearer auth continues to authorize the same routes; the capability
  row agrees with the runtime signing configuration.
