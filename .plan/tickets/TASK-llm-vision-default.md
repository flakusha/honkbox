# TASK: Make llm-vision an honest default

Status: done
Priority: P0
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: `llm-vision` is enabled by default where the vendor hash-link patch is supported, with an explicit opt-out for link-portability; stale `api-types` comment removed.
Verification: done — `llm-vision` is in default features (`crates/server/Cargo.toml:60`, `default = ["vulkan", "llm-vision"]`, with the `experimental-ash-sd`-style opt-out story: a feature-less build omits it); the stale "Honkbox ships no vision backend" `api-types` comment replaced with the vision-capable-model wording (`crates/api-types/src/lib.rs:88-93`); playground gating rides `capabilities.vision` (`packages/web-ui/src/playground.ts`).
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
