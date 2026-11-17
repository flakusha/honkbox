# TASK: External tool registry

Status: proposed
Priority: P0
Epic: epic-external-tools
Owner: unassigned
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
