# TASK: Read-only generation record/history API

Status: done (2026-11-24)
Priority: P2
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: a consumer can fetch the recorded prompt, seed, model hash and
engine version for a finished job through a read-only route, with a bounded
pagination surface and no job-state mutation.
Verification: `bun run test:server`, `bun run codegen:check`, `bun run lint:rs`.
Links: `.tmp/research/content-experience-gaps-2026-11-20.md` §1 G2,
`FEAT-artifact-delivery-and-job-result-events.md`.
Resources: per-test temp data dir.

## Change

- Read the bundle `final_00.json` sidecar rather than scraping `request.json` in the client; image bundles carry no `manifest.json`, so the content route cannot resolve them. Return a typed DTO with additive fields.
- Keep scope bounded to a single job plus a paged list; loop-lore owns boards
  and collections.

## Non-goals

Re-running a job (separate replay ticket), editing records, global search.