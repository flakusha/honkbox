# FEAT: Managed lifecycle for user-installed CLI apps

Status: done (2026-11-22)
- Verified 2026-11-22 (`.tmp/managed-app-verify.log`): `--test apps` 162/162 pass; `cargo check --all-targets` 0 errors; three launch modes (`path`/`installation`/`script`), four `/v1/apps*` admin routes, four typed bearer-gated `/comfyui` read routes, process-group teardown; no router mode, auto-start or auto-restart. `bun run deny` not claimed — pre-existing RUSTSEC advisory.
Priority: P2
Epic: epic-external-tools
Owner: unassigned
Acceptance: an operator can start/stop/restart a declared CLI server (first target: `llama-server`) from honkbox; honkbox owns readiness, capped logs, process-group teardown, GPU lease lifetime and idle unload. Attach-only discovery keeps working when no app is managed.
Verification: `bun run test:server`, `bun run lint:rs`, `bun run codegen:check`, `bun run deny`
Links: `.tmp/research/cli-app-management-llama-server-2026-11-22.md`, `TASK-llama-server-attach.md`
Resources: unix-only first; `rustix` process-group kill + `CancellationToken` + the fixed 8-kind ledger are already in the tree.

## Change

- New manager (`crates/server/src/apps.rs`) + admin routes
  (`GET /v1/apps`, `GET /v1/apps/{id}`, `POST /v1/apps/{id}/start|stop`),
  mirroring `DownloadManager`'s tracked-task shape. Reuse `rustix::process::kill_process_group`;
  do not reuse `supervisor::run` — it is a run-to-completion primitive and
  kills on exit, the opposite of a daemon's contract.
- Readiness polls `/health` (timeout default 120 s, floor 15 s, progress
  restarts the countdown). `failed` carries a capped stderr tail. No automatic
  restart: an unattended restart loop is unattended code execution.
- Prefer a UNIX socket under `<data_dir>/apps/<id>/` on unix; operator-set
  loopback TCP elsewhere. Force generated API key, loopback bind,
  `--cors-origins localhost`, `--no-webui`, no `--tools`/`--agent`/MCP flags;
  keep the argv set allowlisted and free of request-supplied paths.
- Acquire `ResourceKind::Llm` (or `Comfy`) before spawn, hold for the process
  lifetime, release on stop/crash/shutdown. Reuse the existing `[swap]` TTL
  semantics for idleness; do not add a second scheduler.
- Named 501 on non-unix (precedent: `routes/python_sidecar.rs`). Add typed
  DTOs in `api-types` with additive fields, regenerate TS, add a fixed route
  label in `metrics.rs`.

## Acceptance detail

- A managed app that crashes or hangs drops out of the capability surface;
  no row, flag, or admitted GPU slot outlives a dead process.
- Stop kills the whole process group on every path, including server
  shutdown; no orphaned helper survives a cancel.
- Binary selection comes from config/env only; admin scope guards start/stop;
  the app never sees honkbox's bearer and is never publicly reachable.

## Non-goals

llama-server router mode / llama-swap as a dependency (duplicate GPU
scheduler), auto-install, generic command DSL, Windows job objects, app UI.
