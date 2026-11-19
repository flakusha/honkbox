# TASK: External tool registry

Status: done
Priority: P0
Epic: epic-external-tools
Owner: honkbox-external-tools lane (2026-11-19)
Acceptance: one capability surface reports installed external engines (Blender binary + Cycles devices, ComfyUI `/system_stats` + `/object_info`, future tools) with versions and missing-tool 501s.
Verification: `bun run test:server`, `bun run codegen:check`
Links: `.tmp/research/ReviewedFullCycle-Tool-Management.md`
Resources: probes only; no shared engine process or fixed socket path

## Change

- Register tools at startup from config/PATH; probe once, cache result, refresh
  on explicit reload.
- DTO in `crates/api-types`: tool id, version, capabilities, availability reason;
  regenerate TS/OpenAPI.
- No generic “external app manager” UI; bounded metric labels (`kind` only).
- Missing tool or capability returns 501 naming the tool, not a stack trace.

## Acceptance detail

Capabilities are read-only and deterministic for a given install; no tool is
linked into the Rust workspace.

## Implementation note (2026-11-19, external-tools lane)

- Discovery half landed as the new crate `crates/external-tools`
  (`honkbox-external-tools`): `ToolId` / `Capability` / `Status` /
  `Unavailable`, a Blender CLI probe (`--version` plus the headless Cycles
  device probe), ComfyUI `/system_stats` + `/object_info` probes (workflow is
  claimed only when the node catalog is present), probe-once caching and an
  explicit `reload`. `Unavailable::NotInstalled` stays distinct from
  `Unavailable::ProbeFailed(reason)`, and a capability is claimed only after
  the probe that proves it succeeded.
- Gates observed: `cargo test -p honkbox-external-tools` -> 15 passed (2 suites);
  `cargo clippy -p honkbox-external-tools --all-targets -- -D warnings` -> no
  issues; `cargo fmt -p honkbox-external-tools -- --check` -> clean. Commits
  `b01de24` (the crate) + `45be1f0` (error-path coverage added by the
  self-review: probe timeout, non-zero exit, missing Cycles marker, HTTP
  500, invalid JSON, missing version, trailing-slash base URL, invalid URL).
- Known limitation, stated in the crate docs: a timed-out probe kills the
  direct child only, not a process group (`killpg` is `libc`, and the
  workspace forbids `unsafe` everywhere).
- Server half landed: `[tools]` config (`blender_path`, `comfyui_url`,
  `timeout_secs`) feeds `Registry::discover` at startup; `ExternalTool` /
  `ExternalToolsCapability` DTOs are exported to TS/OpenAPI; the documented
  `external_tools` block appears in `Capabilities`; `GET /v1/external-tools`,
  `GET /v1/external-tools/{id}` and `POST /v1/external-tools/reload` read and
  refresh the cached snapshot. A missing tool or unproven capability returns
  a named 501 (`external tool {id} unavailable: …`), never a stack trace.
- Read-only test harnesses that do not start the server see both known tools
  marked `not installed` — no probe is spawned from a test process.
- The process supervisor (`TASK-process-supervisor`) is the other half of
  this epic; it needs a process-group kill design that the workspace-wide
  `unsafe_code = "forbid"` rules out (`killpg` lives in `libc`).
