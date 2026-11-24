# TASK: Salvo WebTransport spike (quinn, gated, not shipped)

Status: proposed
Priority: P3
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: a spike proving WT serves from the salvo acceptor — `quinn`
feature on, `QuinnListener::join(TcpListener.rustls())` on one port, one
datagram echo plus one bidirectional-stream echo behind bearer auth, with a
recorded verdict on LAN viability (self-signed cert trust install, UDP
pass-through) and a measured check-cost comparison; nothing ships.
Verification: `cargo test --locked -p honkbox-server --test webtransport_spike`
(port-0 listener, `Drop` teardown), `bun run check`; spike code deleted or
kept behind a default-off feature — no new served surface
Links: loop-lore `.tmp/research/ws-wt-realtime-honkbox-salvo-addendum-2026-11-24.md`
§6 (upstream docs: `salvo.rs/guide/features/hello-h3`,
`salvo.rs/guide/features/webtransport`,
`docs.rs/salvo_core/1.2.1/salvo_core/webtransport`);
`TASK-salvo-migration-decision.md` (revisit trigger 1: WAN/QUIC need)
Resources: per-test port 0 UDP+TCP, self-signed cert generated per-test-run
in a temp dir, deterministic teardown; no shared ports.

## Precondition

Starts after `TASK-salvo-websocket-chat-stream.md` lands (salvo serving +
`websocket` feature proven). This spike answers revisit trigger 1 with
measurement instead of premise.

## Scope

1. `salvo = { features = ["quinn"] }` in a spike branch, `req.web_transport_mut()`
   with `accept_datagram()` / `accept_bi()` echo handlers.
2. Record: TLS 1.3 cert story for a LAN box (self-signed + trust install vs
   local CA), UDP/443 router behavior, Safari/caniuse re-evidence for the
   epic's `enableWebTransport`-style checklist, envelope parity with §contract.
3. Verdict only: ship-decision goes through an amendment of the migration
   epic's HTTP/3 + WebTransport non-goals — this ticket explicitly does not
   amend them.

## Non-goals

Shipping WT, HTTP/3 for REST, flag-day migration, hand edits to generated
`packages/web-ui/src/gen/` files.
