# TASK: Honest, retryable Fizgig probe and capability contract

Status: proposed
Priority: P1
Epic: epic-external-tools
Owner: unassigned
Acceptance: the startup probe does not block an axum worker, a failed probe is
retried under a bounded TTL instead of being cached for the process lifetime,
and no capability row claims capacity for a route that cannot admit jobs.
Ticket verification statements match the code that shipped.
Verification: `bun run test:server`, `bun run lint:rs`,
`bun run codegen:check`, plus a capability test with a failing then recovering
probe.
Links: `crates/server/src/routes/fizgig.rs:100-113,204-233`,
`crates/server/src/routes/capabilities.rs:46-60`,
`.plan/tickets/TASK-training-reservation-class.md`,
`.tmp/intercom/FIXED-fizgig-fix-round.md`.
Resources: per-test temp dir for the probe; no fixed port.

## Change

- Move `supervisor::run` off the async path (`spawn_blocking` or startup-only
  probe) and replace the permanent `OnceLock<Result<…>>` with a bounded
  retry/TTL cache.
- Either add the real admitting route to `ADMITTING_ROUTES` or move the job to
  the dedicated training class from `TASK-training-reservation-class`; do not
  leave a `reserved: true` row that a hidden route now serves.
- Correct the verification claims in `FEAT-fizgig-lora-training-jobs.md` and
  `TASK-fizgig-cli-probe.md` to the gates actually run, with log paths.

## Non-goals

Changing upstream Fizgig's own CLI or installing the pinned checkout.