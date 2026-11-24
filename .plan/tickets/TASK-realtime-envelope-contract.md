# TASK: Transport-agnostic realtime envelope contract

Status: proposed
Priority: P2
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: one documented JSON envelope set shared by SSE, WS and (later)
WT: message/token deltas, generation progress, presence/seen-state, plus
transport capability flags (binary/datagram); `GET /v1/jobs` stays the
snapshot, `/v1/events` transition-only semantics (incl. `events_dropped`
with skip count) are preserved verbatim in the contract.
Verification: contract test asserting the WS route and the SSE route emit
identical envelopes for the same stub run; `bun run codegen:check` green
(DTOs flow through `crates/api-types` → `packages/web-ui/src/gen/`, never
hand-edited)
Links: loop-lore `.tmp/research/ws-wt-realtime-honkbox-2026-11-24.md` §§2-5;
loop-lore `src/transport/` (`TransportProtocol.WebTransport` enum,
`ws → wt` upgrade path already declared client-side);
`FEAT-artifact-delivery-and-job-result-events.md`
Resources: none (doc + contract test only).

## Scope

1. Write the envelope spec: event names, required/optional fields, cancel and
   terminal states, `events_dropped` backpressure frame, capability flags.
2. Express every envelope as a DTO in `crates/api-types` so TS types generate
   into `packages/web-ui/src/gen/` via the existing pipeline.
3. Add the parity contract test (SSE output == WS output for the same stub
   run). Mount WS after `TASK-salvo-websocket-chat-stream.md`; WT falls out
   of the same envelopes later.

## Non-goals

New endpoints, queue-semantics changes, binary/audio payload design (text
JSON only in v1; OpenAI-Realtime-style event lifecycle if voice ever needs it).
