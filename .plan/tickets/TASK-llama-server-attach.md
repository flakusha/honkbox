# TASK: Attach a user-installed llama-server

Status: done (2026-11-22; probe/config landed, 6-tool capability contract green)
Priority: P2
Epic: epic-external-tools
Owner: honkbox CLI-app lane (2026-11-22)
Acceptance: a user-installed `llama-server` on PATH (or at `HONKBOX_LLAMA_SERVER`) is discovered by the startup probe; when `[tools] llama_server_url` is set, the tool claims the `openai_compat` capability only after `/health` returns 200 and `/v1/models` exposes a `data` array. No process is started, stopped, or restarted.
Verification: `cargo test --locked --offline -p honkbox-server` → capability
tests cover all six tools and all suites pass; workspace clippy clean
(`.tmp/integration-review-server-test.log`, `.tmp/integration-review-clippy.log`).
Adapter harness: 6/6 pass (`.tmp/llama-harness/test2.log`).
Links: `.tmp/research/cli-app-management-llama-server-2026-11-22.md`
Resources: probe + config only; no new dependency, no new route, no new DTO.

## Change

- New `crates/external-tools/src/llama_server.rs`: `binary()` (env override →
  `llama-server` on PATH), `--version` probe via the existing `probe_version`,
  and an async HTTP probe of `/health` + `/v1/models` using the registry's
  shared `reqwest::Client` (so it never blocks a runtime worker).
- `ToolId::LlamaServer` + `Capability::OpenAiCompat`; `ToolId::ALL` grows 5 → 6;
  `Registry::reload` probes the binary and endpoint concurrently with the
  other tools, and caches the combined honest status.
- `[tools] llama_server_url` (optional) threads through `FileTools` →
  `ExternalToolsConfig` → the probe crate's `Config`. No URL = binary-only
  status with no capability claim.
- Capability/test contract updated for the sixth tool row. No `api-types`
  change: `ExternalTool` already carries id/version/capabilities/availability.

## Acceptance detail

A missing binary is `NotInstalled`; a broken binary or unreachable/invalid
endpoint is `ProbeFailed` with the reason; `GET /v1/external-tools` and the
capabilities document report all six tools in stable order. A 503 `/health`
("Loading model") is not readiness and claims nothing. The 501 semantics and
admin scope of `/v1/external-tools/{id}` are unchanged. No route serves chat
through this tool yet — that is `FEAT-llama-server-client-backend.md`.

## Non-goals

Managed start/stop, port allocation, process-group teardown, auto-restart,
router mode, app UI (all in `FEAT-cli-app-manager.md` or out of scope).
