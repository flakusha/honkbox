# EPIC: External tool registry and process supervision

Status: in progress (2026-11-22 — registry/supervisor landed; CLI-app follow-ups open)
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
- `TASK-llama-server-attach.md` — attach a user-installed `llama-server`
  (landed; six-tool capability contract green).
- `FEAT-cli-app-manager.md` — managed start/stop lifecycle (landed 2026-11-22).
  ComfyUI follow-ups are tracked in `epic-comfyui-managed-runtime.md`.
- `FEAT-llama-server-client-backend.md` — consume the attached server through
  the LLM seam (landed; also tracked in `epic-llm-expansion`).
- `TASK-multi-gpu-device-axis.md` — device ids and affinity on ledger slots,
  keeping the single-slot default unchanged.
- `TASK-fizgig-probe-honesty.md` — bounded retry, no blocked axum worker,
  honest capability rows; follow-up to the attach/probe slices.
- `TASK-training-reservation-class.md` — dedicated training GPU class so
  capability rows never reserve a hidden route.

## Acceptance

All external engines report capabilities through one DTO; no engine is linked
into the Rust workspace; leases and cancellation flow through the resource
ledger (`TASK-resource-ledger-bounded-queue.md`).

## Verification

`bun run test:server`, `bun run codegen:check`, `bun run lint:rs`.

## Non-goals

giwt-style manager UI; MCP tools; embedded `bpy`.
