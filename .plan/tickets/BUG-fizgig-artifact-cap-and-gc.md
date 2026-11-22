# BUG: Bound Fizgig artifacts, GC the work dir, retain resume state

Status: done (2026-11-22)
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: a 1000-epoch job cannot copy 1001 full LoRA files into a bundle,
cannot evict other jobs via the shared size cap, and leaves the upstream state
dir recorded for resume; `<data_dir>/fizgig/<id>` is swept by GC.
Verification: `cargo test --locked --offline -p honkbox-server --test fizgig`
with a fake CLI emitting many epoch artifacts; `bun run audit:gpu-slot`.
Links: `.tmp/research/cli-app-launch-management-other-apps-2026-11-22.md` §5.3,
`crates/server/src/routes/fizgig_run.rs:318-338`,
`crates/server/src/jobs.rs:487-546`, `crates/server/src/main.rs:396`.
Resources: per-test temp data dir; no fixed port.

## Change

- Commit only the final `<output_name>.safetensors` plus a bounded newest-epoch
  set (or a documented byte cap), not every `*.safetensors`.
- Add `<data_dir>/fizgig` to the existing GC sweep with the same retention
  policy as `jobs`; record the upstream state dir path in the job manifest.
- Keep hashing bounded to the copied artifacts.

## Non-goals

Implementing resume itself; that follows in `POST …/resume` work.

## Landed

`commit()` copies only the final `<output_name>.safetensors` plus the 3 newest
epoch checkpoints and hashes exactly those; the upstream `--save_state` dir is
recorded in the manifest as `state_dir`. `jobs::gc_jobs` sweeps
`<data_dir>/fizgig` with the job retention policy. Fake CLI emits 5 epochs +
final; bundle test proves cap and state record. Evidence: 205 passed across
`--test fizgig --test licenses`; `cargo test ... --bin honkbox
gc_jobs_sweeps_fizgig_work_dirs` → 1 passed.