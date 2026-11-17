# FEAT: Downloadable job bundle zip

Status: proposed
Priority: P2
Epic: epic-output-formats
Owner: unassigned
Acceptance: a GET route returns a zip of `<data_dir>/jobs/<ulid>/` contents with deterministic entry order.
Verification: `cargo test --locked -p honkbox-server`, `bun run codegen:check`
Links: `.tmp/research/ReviewedOutput-Formats.md` §4
Resources: per-test temp job dirs; zip files written to runner tmp only

## Change

- Add a zip dependency with license/advisory check; build the archive from the
  existing job directory (`request.json`, `final_00.*`, sidecar, metrics, log).
- Sort entry names before writing so identical job content yields identical
  archives (no timestamps from wall clock in entry metadata).
- Register the route under the existing bearer layer; bounded response size via
  existing body-limit policy on requests, an explicit entry-count cap on archives.

## Acceptance detail

Two archives of the same job are byte-identical; no job-content route exists
today, so this is a new GET endpoint, not a change to `/v1/jobs` list.
