# TASK: External process supervisor

Status: done (commit fa0fefb, 2026-11-19)
Priority: P0
Epic: epic-external-tools
Owner: honkbox-process-supervisor lane (2026-11-19)
Acceptance: one supervisor starts/stops external engines with explicit args, timeout, CPU/RAM limits, cancel, and artifact collection.
Verification: `cargo test --locked -p honkbox-server`, `bun run lint:rs`
Links: `.tmp/research/ReviewedFullCycle-Tool-Management.md`, `ReviewedFullCycle-Blender-Workstream.md`
Resources: per-job temp working dirs; process groups killed in `finally`/drop

## Change

- Own child lifecycle for Blender (`-b -P script.py --python-exit-code N`),
  ComfyUI, and future tools; no `bpy` embedded in-process.
- Cap threads (`-t`), device selection, and RSS (cgroup v2 where available);
  hold a GPU lease for GPU-backed work via the resource ledger.
- Cancel kills the process group; deterministic teardown on success, failure,
  and timeout; collect stdout/stderr and output globs into the job bundle.

## Acceptance detail

A cancelled job leaves no orphan process and releases its lease; a failed job
still writes its log and sidecar; no fixed temp path is shared between tests.

## Implementation note (2026-11-19, process-supervisor lane)

- New module `crates/server/src/supervisor.rs` (unix-only, declared in
  `main.rs` behind `#[cfg(unix)]` + `#[allow(dead_code)]` — the supervisor
  lands ahead of its first caller, `FEAT-blender-job-route`).
- `supervisor::run(Spec) -> Result<Outcome, RunError>`: the child starts in
  its OWN process group (`CommandExt::process_group`), the group is killed
  through `rustix::process::kill_process_group` on EVERY exit path (success,
  failure, timeout, cancel, RSS cap) and the child is reaped before `run`
  returns. `rustix` is the safe door to `killpg`: the workspace forbids
  `unsafe`, so the syscall cannot be a hand-rolled `libc` call. This closes
  the gap the registry ticket left open (its probe kills the direct child
  only).
- Bounds: wall-clock timeout, optional RSS cap (the group LEADER's resident
  set via procfs), optional `CancellationToken`, capped stdout/stderr
  (64 KiB per stream by default; the child never blocks on a full pipe) and
  the artifact globs collected after the run.
- The GPU lease is handed IN (`Spec::lease`) and held for the child's whole
  lifetime; it is released the moment the child is reaped, so a cancelled job
  never keeps the slot.
- Tool flags (`-t`, device selection) stay EXPLICIT caller args: the
  supervisor never builds a shell string and never guesses tool flags. The
  bundle write (log + sidecar) stays with the job route, which owns the
  `.part` + fsync + rename flow.
- Stated ceilings: per-leader RSS instead of cgroup v2 accounting (that
  needs a delegated cgroup), unix-only (Windows needs job objects).
- New direct deps: `rustix` (process groups) + `glob` (artifact collection);
  both were already in `Cargo.lock`, so no new download.

## Verification (observed 2026-11-19, commit fa0fefb)

- `cargo test --locked -p honkbox-server` -> 2681 passed (31 suites) at HEAD.
- `cargo test --locked -p honkbox-server --test supervisor` -> 58 passed (1
  suite: the 8 supervisor contracts below plus the 50 inline unit tests of the
  `#[path]`-included `src/{config,jobs,metrics}.rs` modules): logs survive a
  failed run, a missing program is a spawn error, the timeout kills the
  group, a cancel leaves no orphan, the RSS cap kills the group, artifacts
  are collected by glob, the log cap never blocks the child, a cancelled job
  releases its lease.
- `cargo clippy --locked -p honkbox-server --all-targets -- -D warnings` ->
  clean (whole package, every test target included).
- `cargo clippy --workspace --all-targets --locked -- -D warnings` -> 0 errors
  at HEAD (the only warning line is the llama-cpp-sys build script's patch
  notice).
- `rustfmt --edition 2024 --check crates/server/src/supervisor.rs
  crates/server/tests/supervisor.rs` -> clean. The workspace-wide fmt check
  still reports pre-existing drift in `backends.rs` / `model_formats.rs` and
  the size-lint lane's in-flight `routes/*.rs` — outside this ticket.
- The shared working tree does NOT compile while the size-lint lane's
  `routes/` split is in flight: 9 compile errors, all in
  `crates/server/src/routes.rs`, `crates/server/src/routes/*.rs` and the
  `main.rs:530` use of `routes::refresh_catalog` (re-export privacy E0364 x6,
  the not-yet-moved `refresh_models` E0432, `ApiDoc::openapi` E0599). None of
  them is in this slice, so the runs above were taken at HEAD in a scratch
  worktree (`git worktree add --detach`) to prove the COMMITTED tree.
