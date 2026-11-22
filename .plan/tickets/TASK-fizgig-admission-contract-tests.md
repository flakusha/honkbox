# TASK: Close Fizgig admission and lifecycle contract gaps

Status: done (2026-11-22)
Priority: P2
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: tests cover allowlist confinement (outside-root paths rejected with
a named 400), a cancel arriving after stage exit does not overwrite a completed
run, non-unix paths do not leave a phantom `queued` record, and Fizgig jobs are
either visible to `GET /v1/jobs` + registry cancel or that exclusion is stated
in the first-slice ticket.
Verification: `cargo test --locked --offline -p honkbox-server --test fizgig`,
`cargo test --locked --offline -p honkbox-server --test cancel`.
Links: `crates/server/tests/fizgig.rs`, `crates/server/src/routes/fizgig.rs:419-425`,
`crates/server/src/routes/fizgig_run.rs:264-267`,
`crates/server/src/jobs_registry.rs:81-106`.
Resources: per-test temp dirs; port 0 or harness-assigned ports.

## Change

- Add path-confinement cases for `dataset_dir`, `dit`, `vae`, `text_encoder`.
- Add a terminal-state guard to cancel and assert it in the fake-CLI test.
- Add the non-unix 501 path assertion; decide and document the `/v1/jobs`
  contract.

## Non-goals

New GPU classes or license gates (separate tickets).

## Landed

Fake-CLI test covers all four caller paths outside their roots (named 400),
late cancel on a terminal job (named 409) and `/v1/jobs` exclusion. The
non-unix branch now returns 501 **before** inserting a job record; the
assertion itself is compile-gated out on this unix host. `FEAT-fizgig-lora-
training-jobs.md` documents the generic-registry exclusion. Evidence:
`cargo test --locked --offline -p honkbox-server --test fizgig` → part of the
205-passing run with `--test licenses`.