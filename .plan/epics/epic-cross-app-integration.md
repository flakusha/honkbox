# EPIC: Cross-app heavy-lifting contract

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/llama-swap-loop-lore-gaps-2026-11-22.md`,
`.tmp/research/loop-lore-godot-actionables.md`,
`.tmp/research/dcc-closed-loop-honkbox-2026-11-25.md` (honkbox half; loop-lore half in
`../loop-lore/.tmp/research/honkbox-dcc-additional-processing-2026-11-25.md`)

## Scope

honkbox remains the local compute side for `loop-lore`: OpenAI-compatible chat,
images, jobs/events, downloads, and deterministic bundles. loop-lore keeps UX,
provider routing, engine projects, and MCP calls. No MCP surface in honkbox.

## Tickets

- `TASK-loop-lore-provider-smoke.md` — one end-to-end provider/job smoke path.

## Acceptance

A loop-lore Custom provider pointed at honkbox completes one chat turn and one
image job with observable `/v1/jobs` + `/v1/events` state; no double GPU admission.
Closed-loop runs additionally require manifest-last publication on the honkbox side
and the loop-lore DevelopmentRun/version/approval spine; SSE remains a hint, never
publication proof.

## Verification

Run both apps locally; `curl /health`, `/v1/models`; inspect `/v1/jobs`, `/v1/events`,
`/metrics` with bounded `kind` labels.

## Non-goals

Adding MCP to honkbox; owning loop-lore's story/runtime state.
