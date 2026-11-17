# EPIC: Cross-app heavy-lifting contract

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/ReviewedFullCycle-LoopLore-Acceleration.md`,
`ReviewedFullCycle-GameEngines.md`

## Scope

honkbox remains the local compute side for `loop-lore`: OpenAI-compatible chat,
images, jobs/events, downloads, and deterministic bundles. loop-lore keeps UX,
provider routing, engine projects, and MCP calls. No MCP surface in honkbox.

## Tickets

- `TASK-loop-lore-provider-smoke.md` — one end-to-end provider/job smoke path.

## Acceptance

A loop-lore Custom provider pointed at honkbox completes one chat turn and one
image job with observable `/v1/jobs` + `/v1/events` state; no double GPU admission.

## Verification

Run both apps locally; `curl /health`, `/v1/models`; inspect `/v1/jobs`, `/v1/events`,
`/metrics` with bounded `kind` labels.

## Non-goals

Adding MCP to honkbox; owning loop-lore's story/runtime state.
