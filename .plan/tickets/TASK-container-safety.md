# TASK: Container safety limits

Status: done (2026-11-18, honkbox-container-safety)
Priority: P1
Epic: epic-binary-formats
Owner: honkbox-container-safety
Acceptance: zip/ORA/GLB containers are bounded by entry count, uncompressed size, and path rules; traversal and decompression bombs are rejected before extraction.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`
Links: `.tmp/research/ReviewedBinaryFormats.md` §Transport and safety
Resources: per-test malformed fixture corpus in runner tmp; no shared extraction dir

## Change

- Reject absolute paths, `..`, drive letters, and symlink entries.
- Cap total uncompressed bytes and per-entry bytes; stream hash while reading.
- Apply the same rules to ORA, job zip bundles, and GLB external buffers.

## What landed

- `crates/server/src/container.rs` — the bounded container reader:
  `ContainerLimits { max_entries, max_entry_bytes, max_total_bytes }` and a
  NAMED `ContainerError` (`TooManyEntries`, `EntryTooLarge`,
  `TotalTooLarge`, `UnsafePath { name, reason }`, `Malformed { detail }`,
  `Unsupported { detail }`) whose message always names the limit or the
  rule that was broken.
  - `Container::open_zip` validates the CENTRAL DIRECTORY (entry count,
    declared per-entry size, declared total) before a single entry byte is
    read;
  - `Container::read_entry` caps the DECOMPRESSED bytes as they stream
    (64 KiB chunks) and hashes them with SHA-256 while streaming — the
    declared size is only a claim, so the streamed counters are what stop a
    decompression bomb;
  - path rules: empty name, NUL byte, backslash separator, absolute path,
    Windows drive letter, `..` component, symlink entry — each reported as
    `UnsafePath` with the rule named;
  - `validate_glb`: the GLB header, the chunk lengths, the per-chunk byte
    cap and a JSON chunk pointing at an EXTERNAL buffer are all rejected;
  - the reader is central-directory-only by construction (it never uses
    `ZipStreamReader` and never extracts an entry by name), so the
    local-header/central-directory name-mismatch advisory
    (RUSTSEC-2026-0548) does not apply to this path.
- `crates/server/tests/container_safety.rs` — the crafted corpus (20
  tests): every fixture is built in memory (no fixture files, no temp
  dirs, no shared state — so the corpus is parallel-safe and
  order-independent) and every rejection asserts the NAMED variant, never
  a generic failure.
- The ORA read path (`crates/server/src/ora.rs`) reads through this
  reader, so it inherits the entry-count, per-entry, total-size and path
  rules for free.

## Acceptance detail

A crafted archive cannot write outside the job dir or exhaust memory; named
errors identify the violated limit. Verified: the reader never writes an
entry to disk (it decompresses in memory and counts the streamed bytes),
the bomb fixture (8 MiB declared, 8 MiB of zeros) is stopped by the
STREAMED counters even when its declared size is rewritten to understate
it, and the traversal/symlink/drive-letter/backslash/NUL fixtures each
produce their own named variant.

## Verification

- `crates/server/tests/container_safety.rs`: 20 tests, all green (run in the
  isolation harness `.tmp/verify-wave`, which compiles the same module
  file; see `.tmp/intercom/FIXED-container-safety.md`).
- `bun run deny` exit 0 (licenses/bans/advisories ok) and `bun run spdx`
  exit 0 on the new dependency set (`zip`, `sha2`).

## Deferred

- The job-bundle ZIP writer does not exist yet (`FEAT-job-bundle-zip`,
  proposed); when it lands, the same `Container::open_zip` rules are the
  enforcement point for it — the reader is the shared seam, so no new rule
  set is needed.
