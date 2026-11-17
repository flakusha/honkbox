# TASK: Make llm-vision an honest default

Status: proposed
Priority: P0
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: `llm-vision` is enabled by default where the vendor hash-link patch is supported, with an explicit opt-out for link-portability; stale `api-types` comment removed.
Verification: `bun run ci` matrix, `cargo build --locked -p honkbox-server --features llm-vision`, `bun run codegen:check`
Links: `.tmp/research/ReviewedSupport-LLM.md` §Remaining table; `Cargo.toml:91-97`, `server/Cargo.toml:55`
Resources: per-build target dirs; no shared model cache

## Change

- Move `llm-vision` into default features after the Linux + Windows/macOS CI
  matrix check; the vendor hash-link patch already landed in `591efc0`.
- Update `api-types/src/lib.rs:88-90` comment that claims no vision backend.
- `TASK-support-doc-truth.md` owns README wording; avoid duplicate edits.
- Catalog still advertises vision only when a sibling `mmproj*.gguf` is found.

## Acceptance detail

A default build reports vision capability accurately; builds that opt out do
not advertise it; data-URI-only image URLs remain the SSRF-safe contract.
