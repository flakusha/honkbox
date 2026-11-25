# TASK: Scheduler lifecycle spec

Status: proposed
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: every 429/503 admission path constructs one AdmissionOutcome variant and one JobPhase transition
Verification: `cargo test --locked -p honkbox-server --test scheduler_lifecycle`
Links: `.tmp/research/determinism-extended-fields-2026-11-25.md`, `crates/server/src/determinism.rs`, `crates/server/src/jobs.rs`
Resources: per-test tmp dirs; port 0; no shared state

## Change

- Add internal `AdmissionOutcome` enum (`AcquiredFast`, `AcquiredAfterQueue`, `RejectedOversized`, `RejectedNoWait`, `RejectedQueueFull`, `RejectedTimeout`, `CancelledQueued`) plus additive diagnostics DTO (`outcome`, `queue_wait_secs`, `retry_after_secs`, `queue_position`); wire mapping unchanged.
- Add internal `JobPhase` enum (`Queued`, `Admitted`, `Running`, `CancelRequested`, `Finished`, `Cancelled`, `Rejected`) mapped onto existing job event names; `GET /v1/jobs` entries gain additive `phase` derived from the cancellation token at snapshot time.
- Keep `Cancelled` terminal only when early stop is actually observed; otherwise surface `Finished` with a `cancelled_token` flag [UNKNOWN: exact flag name pending owner decision].

## Acceptance criteria

- [ ] Every 429/503 path constructs one `AdmissionOutcome` variant.
- [ ] Each variant maps to its status code plus reason string in a unit test.
- [ ] Non-interruptible cancel-then-complete surfaces `Finished` with `cancelled_token=true`, never `Cancelled`.
- [ ] Wire stays additive: no existing response field renamed or removed.

Neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license.
