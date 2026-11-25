# TASK: Cancel checkpoint taxonomy

Status: proposed
Priority: P2
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: per-backend checkpoint class published in capabilities for every job kind
Verification: `cargo test --locked -p honkbox-server --test cancel_checkpoints`
Links: `.tmp/research/determinism-extended-fields-2026-11-25.md`, `TASK-queue-introspection.md`, `crates/server/src/jobs.rs`
Resources: per-test tmp dirs; port 0; no shared state

## Change

- Publish one additive checkpoint-class string per job kind in capabilities/diagnostics: `none` | `boundary` | `token_poll` | `engine_abort` | `process_group_kill`.
- Pin the `interruptible` flag per kind in a test alongside its checkpoint class [UNKNOWN: one backend registry flag value pending one code read].

## Acceptance criteria

- [ ] Each of the 9 job kinds has a published checkpoint class plus cancel disposition.
- [ ] A test pins `interruptible` per kind.
- [ ] Cooperative-cancel semantics hold: a cancel mark alone never implies interruption.

Neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license.
