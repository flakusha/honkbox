# TASK: Capability flag/route agreement test

Status: proposed
Priority: P0
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: for every advertised `true` flag, the corresponding route exists in OpenAPI and returns a non-501 result for valid stub-backed input; negative cases hold for default features.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedSupport-Capabilities.md` §Verification contract
Resources: per-test feature-configured server instances on assigned ports; no shared catalog

## Change

- Build two test servers: default features and `llm-vision` compiled.
- Assert default build: `vision.built = false`, `video = false`,
  `controlnet = false`; compiled build: `vision.built = true` but
  `vision.models = 0` without a scanned mmproj.
- Cross-check each `true` flag against the generated OpenAPI path list and a
  valid request to the stub backend; no named 501 allowed.

## Acceptance detail

A future feature flag cannot be advertised without a working route, and a
working route cannot be hidden by a stale false flag.
