# TASK: Capability flag/route agreement test

Status: done (2026-11-18, honkbox-capabilities)
Priority: P0
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: for every advertised `true` flag, the corresponding route exists in OpenAPI and returns a non-501 result for valid stub-backed input; negative cases hold for default features.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedSupport-Capabilities.md` §Verification contract
Resources: per-test feature-configured server instances on assigned ports; no shared catalog

> Done 2026-11-18 (honkbox-capabilities), one adaptation: instead of two
> fixed server fixtures, the suite's assertions READ the compile probes and
> hold under either feature set, so the same binary is run twice —
> `cargo test -p honkbox-server` (default, `vision.built = true` exercised
> live through the vision chat gate) and `--no-default-features` (text-only
> build: `vision.built = false` + named `vision_unsupported` 400 via the
> same test). Text-only finding folded into the contract: the vision chat
> gate keys on the CATALOG entry's `capabilities.vision` (the same flag
> `vision.models` counts), so an operator-declared vision entry serves in
> a text-only build (a real pipeline fails loudly at model load) while a
> vision-less entry always answers the named 400 — in text-only builds
> that is every scan-derived entry's shape (derived vision collapses
> without the mtmd path). Observed: capabilities suite passes in default
> (68) and `--no-default-features` (68) configs; the REST of the suite
> does not compile text-only yet (`tests/completions.rs:976` pins
> `VISION_COMPILED == true` at compile time — sibling-owned, untouched).
> True flags (jobs/events/lora + formats) are cross-checked
> against the generated OpenAPI path list AND live stub-backed requests
> (generations round-trips per advertised write format, decoded-magic
> asserted); `video = false` <-> no `/v1/videos` path. Negative cases:
> stub-backed ControlNet advertises `false` (its valid-pair 501), vision
> `models = 0` without a vision catalog entry.

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
