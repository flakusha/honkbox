# FEAT: Downloadable job bundle zip

Status: done
Priority: P2
Epic: epic-output-formats
Owner: main session
Acceptance: `GET /v1/jobs/{id}/bundle.zip` returns a deterministic zip of top-level `<data_dir>/jobs/<ulid>/` files; `GET /v1/jobs/{id}/content/{name}` streams only `manifest.json`, `output.glb` or `output_turntable.webp`. Both validate the 26-char ULID, cap entries/bytes and never resolve host paths or symlinks.
Verification: `cargo test --locked -p honkbox-server`, `bun run codegen:check`, `bun run deny`
Links: `.tmp/research/ReviewedOutput-Formats.md` §4
Resources: per-test temp job dirs; zip files written to runner tmp only

## Change

- The `zip` workspace dependency (license/advisory gate in `bun run deny`)
  builds the archive from the top-level regular files of the existing job
  directory; nested staging trees and symlinks are skipped.
- Sort entry names before writing so identical job content yields identical
  archives (fixed timestamp, permissions and compression, no wall clock).
- Both GETs sit under the existing bearer layer and share an OpenAI-style
  error envelope. Response bounds are explicit: 256 entries, 512 MiB
  uncompressed, 512 MiB compressed, and a 512 MiB single-file cap — a GET
  response is not bounded by the request body limit.

## Acceptance detail

Two archives of the same job are byte-identical. Routes already exist for
artifact bytes (`/v1/downloads`, `/v1/artifacts/{id}/bytes`); this endpoint is
new because a job bundle is a directory snapshot, not a promoted artifact.

## Implementation (2026-11-19)

- `crates/server/src/routes/job_content.rs`: ULID validator, closed filename
  allowlist, `symlink_metadata` on the single-file path, `spawn_blocking` ZIP
  build with sorted names and invariant metadata, reject-not-clamp 413s.
- `crates/server/tests/job_content.rs`: byte-for-byte determinism across
  different creation order/mtimes, nested-file exclusion, allowlist and ULID
  rejections, 404/413 envelopes, path non-leakage, and a Unix symlink 404.
