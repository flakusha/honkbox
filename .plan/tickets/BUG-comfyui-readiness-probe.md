# BUG: managed ComfyUI readiness probes a route it does not serve

Status: done (2026-11-24)
- Verified 2026-11-24: `cargo test --locked -p honkbox-server --test apps`
  162 passed / 0 failed. `Health::Tcp` now carries the probe path: llama-server
  `/health`, ComfyUI `/queue`. The fake ComfyUI serves only `/queue` and 404s
  `/health`; the existing argv/readiness test is the regression. No DTO change,
  so `codegen:check` was not run for that change alone. Later full-tree
  verification on 2026-11-24 ran the complete server suite and workspace clippy
  clean. Live upstream check remains opt-in and was not run.
Priority: P0
Epic: epic-comfyui-managed-runtime
Owner: unassigned
Acceptance: a managed `comfyui` app with an unmodified upstream reaches `ready`, so `/comfyui/*` stops returning 501; llama-server keeps `/health`.
Verification: `cargo test --locked -p honkbox-server --test apps`, `bun run check`
Links: `.tmp/research/comfyui-managed-instance-review-2026-11-24.md` §2; `FEAT-cli-app-manager.md`
Resources: per-test fake server; no shared ComfyUI instance

## Change

- `crates/server/src/apps.rs:663-672`: choose the readiness path by app/launch
  mode. `llama_server` keeps `GET /health`; ComfyUI probes a route its upstream
  actually registers (`GET /queue` or `GET /system_stats`). Keep the 200 =
  ready, 503 = loading (deadline restart with the 3x hard ceiling) semantics.
- Update the test fixtures so the fake ComfyUI serves the pinned upstream's
  route set and explicitly 404s `/health`. The test must fail against the
  current implementation and pass after the fix.
- Keep unix-socket probing unchanged for llama-server.

## Acceptance detail

- `POST /v1/apps/comfyui/start` on a real upstream leads to state `ready`, a
  non-empty endpoint and `GET /comfyui/status` returning `available: true`
  with `source: "managed"`.
- A dead/reloading ComfyUI still fails readiness within the configured budget
  and clears endpoint/pid/lease; no false-ready from a 404.
- Evidence: pinned sibling checkout `/home/flak/git-ai/comfyui` (0.46.0)
  registers no `/health` route; before this fix the `/health` probe made a
  healthy ComfyUI look `starting` for up to 3x120 s and then `failed`.
