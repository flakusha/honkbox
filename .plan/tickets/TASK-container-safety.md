# TASK: Container safety limits

Status: proposed
Priority: P1
Epic: epic-binary-formats
Owner: unassigned
Acceptance: zip/ORA/GLB containers are bounded by entry count, uncompressed size, and path rules; traversal and decompression bombs are rejected before extraction.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`
Links: `.tmp/research/ReviewedBinaryFormats.md` §Transport and safety
Resources: per-test malformed fixture corpus in runner tmp; no shared extraction dir

## Change

- Reject absolute paths, `..`, drive letters, and symlink entries.
- Cap total uncompressed bytes and per-entry bytes; stream hash while reading.
- Apply the same rules to ORA, job zip bundles, and GLB external buffers.

## Acceptance detail

A crafted archive cannot write outside the job dir or exhaust memory; named
errors identify the violated limit.
