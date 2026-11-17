# FEAT: Response compression

Status: proposed
Priority: P2
Epic: epic-output-formats
Owner: unassigned
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
