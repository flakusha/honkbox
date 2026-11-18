# FEAT: Response compression

Status: done (2026-11-18, LANE-impl-wave)
Priority: P2
Epic: epic-output-formats
Owner: LANE-impl-wave
Acceptance: gzip/brotli response compression for ordinary JSON/binary routes, with SSE chat excluded.
Verification: `cargo test --locked -p honkbox-server`, `bun run lint:rs`
Links: `.tmp/research/ReviewedOutput-Formats.md` §4-5
Resources: no shared ports beyond the test server's assigned port 0/offset

## Change

- `tower-http` is already in the lock transitively (reqwest), but no
  `CompressionLayer` is mounted; add it as a direct dep with the compression
  feature at the router layer (`routes.rs` router assembly).
- Exclude `/v1/events` and `/v1/chat/stream` — SSE must remain unbuffered.
- Verify `Accept-Encoding` negotiation and that base64 image payloads are not
  double-compressed unexpectedly.

## Acceptance detail

Compressed responses decompress to byte-identical bodies; SSE frames arrive
uncompressed and in order; no new heavy dependency tree.

## Implementation note (2026-11-18)

- `tower-http = { version = "0.6", default-features = false, features =
  ["compression-gzip", "compression-br"] }` (workspace dep; predicates are
  always compiled in 0.6 — there is no `compression-filter` feature).
  Encoders arrive via async-compression over the lock-resident
  flate2/brotli — no new heavy tree; deny.toml licenses (MIT/Apache-2.0)
  already allowed, no exception needed.
- Mounted in `app()` INSIDE `track_metrics` so status/`Retry-After`
  bookkeeping still sees the final response.
- SSE exclusion: `DefaultPredicate` vetoes `text/event-stream` by
  CONTENT TYPE — strictly stronger than path exclusion: both SSE surfaces
  (`/v1/events`, `/v1/chat/completions`+`stream:true` / `/v1/chat/stream`)
  and any future SSE route are covered. `image/*` bodies are likewise never
  compressed (the ticket's double-compression concern), gRPC excluded too.
- Size floor (ticket named none — the pick): `COMPRESSION_MIN_BYTES = 1024`
  via `SizeAbove`, documented at the constant; sub-1 KiB bodies pass
  through.
- Tests (new `tests/compression.rs`): gzip on `/v1/openapi.json`
  decompresses BYTE-IDENTICAL to the uncompressed body (flate2 dev-dep);
  brotli negotiates; `/health` below the floor stays uncompressed; unknown
  encoding (zstd) passes through; `/v1/events` with `Accept-Encoding: gzip`
  is uncompressed AND a lifecycle frame arrives WHILE a gated generation is
  parked mid-flight (the unbuffered-streaming proof); the chat SSE surface
  is header-asserted uncompressed (same predicate arm; disclosed in the
  test comment).
- Byte-compat: no existing route body changed; compression is
  client-negotiated only (`Accept-Encoding`), so `Accept-Encoding`-less
  clients observe identical bytes.
