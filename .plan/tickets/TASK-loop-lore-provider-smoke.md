# TASK: loop-lore provider smoke path

Status: proposed
Priority: P2
Epic: epic-cross-app-integration
Owner: unassigned
Acceptance: a loop-lore Custom provider pointed at honkbox completes one chat turn and one image job with observable job/events state.
Verification: run both apps locally; `curl /health`, `/v1/models`; inspect `/v1/jobs`, `/v1/events`, `/metrics`
Links: `.tmp/research/ReviewedFullCycle-LoopLore-Acceleration.md`
Resources: loopback ports assigned by the harness; per-run temp data dirs for both apps

## Change

- Configure honkbox bearer token and load one model; register loop-lore Custom
  OpenAI-compatible provider at `/v1/chat/completions`.
- Submit one image via `/v1/images/generations`; confirm job registration,
  progress/finish events, and cancel mid-run.
- Compare honkbox queue/lease metrics with loop-lore's scheduler events: one
  side owns GPU admission, never both.

## Acceptance detail

No MCP surface added to honkbox; loop-lore keeps provider routing, circuit
breaker, and requestClass labels.
