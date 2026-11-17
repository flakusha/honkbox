# EPIC: External tool registry and process supervision

Status: proposed
Priority: P0
Owner: unassigned
Research: `.tmp/research/ReviewedFullCycle-Tool-Management.md`,
`ReviewedFullCycle-Resource-Queues.md`

## Scope

One capability-aware registry for Blender, ComfyUI, and future external engines
plus a single process supervisor: discover versions/devices, start/stop, pass
args, enforce timeout and CPU/RAM limits, kill on cancel, collect artifacts.
“Managing apps” is not a feature of its own — it is this plumbing.

## Tickets

- `TASK-external-tool-registry.md` — capability discovery + 501 on missing tool.
- `TASK-process-supervisor.md` — lifecycle, limits, cancel, artifact dir.

## Acceptance

All external engines report capabilities through one DTO; no engine is linked
into the Rust workspace; leases and cancellation flow through the resource
ledger (`TASK-resource-ledger-bounded-queue.md`).

## Verification

`bun run test:server`, `bun run codegen:check`, `bun run lint:rs`.

## Non-goals

giwt-style manager UI; MCP tools; embedded `bpy`.
