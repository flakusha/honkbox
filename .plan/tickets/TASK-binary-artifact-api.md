# TASK: Binary artifact transport

Status: done (commit f15b460, 2026-11-18)
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

## Verification (observed 2026-11-18, commit f15b460)

- `cargo test --locked -p honkbox-server --test artifacts` -> "cargo test:
  87 passed (1 suite, 0.05s)" (re-observed on the shared tree after sibling
  lanes landed).
- `bun run test:server` -> full honkbox-server suite green, exit 0 (every
  suite 0 failed; artifacts: "test result: ok. 87 passed; 0 failed").
- `bun run codegen:check` -> api-types "15 passed", coverage "openapi
  covers all 36 mounted routes"; the final `git diff --exit-code` step is
  satisfied by this commit carrying the regenerated gen files.
- `bun run lint:rs` -> clippy passes every file of this lane after
  pedantic fixes (duration units, format_collect, cast_possible_truncation,
  doc backticks in tests/artifacts.rs; needless_pass_by_value in the
  landed image_ops.rs harness). Residual failures are the audio lane's
  in-flight main.rs edits (generate() too_many_lines, match_same_arms) -
  outside this ticket.

