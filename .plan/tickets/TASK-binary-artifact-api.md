# TASK: Binary artifact transport

Status: proposed
Priority: P2
Epic: epic-binary-formats
Owner: unassigned
Acceptance: large binary inputs use bounded chunked/multipart upload and downloads support `Range` with streaming hashes.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedBinaryFormats.md` §Transport and safety
Resources: per-test artifact dirs; no shared port beyond test server offset

## Change

- Add `POST /v1/artifacts` (chunked) and `GET /v1/artifacts/{id}/bytes`.
- Store artifacts by content hash under the data dir; `.part` + rename like
  downloads; cap total and per-chunk bytes.
- No arbitrary client-supplied path; no base64 for large payloads.

## Acceptance detail

An interrupted upload resumes by chunk id; a ranged GET returns correct bytes
and `Content-Range`; hashes match the manifest.
