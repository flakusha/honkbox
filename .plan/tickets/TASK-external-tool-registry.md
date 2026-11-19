# TASK: External tool registry

Status: in progress
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
  device probe), a ComfyUI `/system_stats` probe, probe-once caching and an
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
- Server half still open: the DTO in `crates/api-types`, the capabilities
  surface and the 501 for a missing tool need `api-types` + `routes.rs`, both
  owned by the review-fixes lane (IN PROGRESS, 2026-11-19). The ticket stays
  `in progress` until that wiring lands.
- Wiring point for the follow-up: `Registry::discover(config).await` at
  startup, `Registry::reload().await` behind the explicit reload path;
  `Config` carries the configured Blender path and ComfyUI base URL.
- The process supervisor (`TASK-process-supervisor`) is the other half of
  this epic; it needs a process-group kill design that the workspace-wide
  `unsafe_code = "forbid"` rules out (`killpg` lives in `libc`).
