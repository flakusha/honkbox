# BUG: Bound Fizgig artifacts, GC the work dir, retain resume state

Status: proposed
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