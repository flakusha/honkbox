# TASK: document the managed ComfyUI setup contract

Status: done (2026-11-24)
- Implemented: `probe_version()` reads `__version__` from the operator
  checkout's `comfyui_version.py` for installation/script launches (non-fatal
  when absent) with a test fixture proving the venv `--version` is not used;
  `[apps.comfyui]` example + README topology/rev pins written.
- Verified 2026-11-24: full `cargo test --locked -p honkbox-server`,
  `codegen:check`, workspace clippy, docs/format gates.
Priority: P1
Epic: epic-comfyui-managed-runtime
Owner: unassigned
Acceptance: an operator can configure managed ComfyUI from repo docs alone, with pinned revisions and an honest version claim.
Verification: `cargo test --locked -p honkbox-server --test apps`, `bun run check`
Links: `.tmp/research/comfyui-uv-pytorch-build-cycle-2026-11-20.md`;
`comfyui-managed-instance-review-2026-11-24.md` §3; `FEAT-cli-app-manager.md`
Resources: docs/example config only; no install command run by honkbox

## Change

- Add an `[apps.comfyui]` example to `honkbox.example.toml` covering `path`,
  `installation` and `script`, loopback host/port, health timeout and explicit
  start (no auto-start/restart).
- Document in README `Deployment topology`: honkbox runs an operator-owned
  checkout via its `.venv/bin/python`; honkbox does NOT run `uv sync`, install
  ComfyUI, or pin revisions at runtime. Record the fork rev and its upstream
  base rev in the docs; rev changes are deliberate bumps, as with
  `vendor.manifest.toml`. Point ComfyUI model dirs at the same model store as
  `[core]` rather than duplicating weights.
- Fix `probe_version()` for `installation`/`script`: read the pinned version
  file in the operator checkout (upstream `comfyui_version.py`) instead of
  reporting the venv's Python version. Non-fatal if absent.

## Acceptance detail

- `GET /v1/apps/comfyui` reports the ComfyUI version, not `Python 3.x`, when a
  version file exists; absence leaves the field unset.
- Docs state the no-install/no-restart contract and show one complete config;
  no operator has to guess `.venv` layout or which rev is supported.
- No code path starts `uv`, fetches a repo, or accepts a path from HTTP.
