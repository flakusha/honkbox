# TASK: Salvo WebSocket chat-stream route (mirror of SSE)

Status: proposed
Priority: P2
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: one WS route served through salvo emitting the same token stream
as `POST /v1/chat/stream` SSE — same chunk envelope, same bearer/JWT/scopes
contract enforced before the upgrade, same cooperative-cancel semantics; SSE
stays the baseline fallback.
Verification: `cargo test --locked -p honkbox-server --test websocket_chat_stream`
(port-0 listener, `Drop` teardown, per-test owned DB), `bun run check`,
`bun run codegen:check`
Links: loop-lore `.tmp/research/ws-wt-realtime-honkbox-2026-11-24.md` §§1-5;
loop-lore `.tmp/research/ws-wt-realtime-honkbox-salvo-addendum-2026-11-24.md`;
`TASK-salvo-migration-decision.md` (revisit trigger 2)
Resources: per-test port 0, owned in-memory DB, deterministic teardown in
`finally`/`Drop`; no shared ports.

## Precondition

Starts only after the salvo cutover serves traffic (`main.rs:serve()` still
runs `axum::serve` on master; the salvo router is built but not serving).
Axum has no native WS story worth keeping — this route lands on salvo only.

## Scope

1. Enable salvo `websocket` feature (currently `oapi server server-handle
   socket2 http1 compression affix-state` — no `websocket`, no `quinn`).
2. One route, e.g. `GET /v1/chat/stream/ws`, mounted **inside** the hooped API
   router so the existing `RequireBearer` hoop runs before
   `WebSocketUpgrade::new().upgrade` — no new auth code, same 401
   OpenAI-shaped rejections, same probe exemptions.
3. Emit the byte-identical chunk envelope the SSE route emits (OpenAI-chunk
   shape the ecosystem already consumes); close frame on job terminal state;
   client close == cooperative cancel, same as SSE disconnect.
4. One integration test through a real port-0 listener (not oneshot): connect,
   authenticate, stream one stub completion, assert envelope parity with the
   SSE route output for the same prompt.

## Non-goals

Job-progress WS (separate ticket if wanted), binary/audio frames over WS,
WebTransport (`TASK-salvo-webtransport-spike.md`), removing SSE.
