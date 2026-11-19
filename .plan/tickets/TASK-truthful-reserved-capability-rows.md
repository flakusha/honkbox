# TASK: Truthful reserved capability rows

Status: done (2026-11-19, truthful-reserved-rows lane)
Priority: P1
Epic: epic-capabilities-contract
Owner: truthful-reserved-rows lane (2026-11-19)
Acceptance: `/v1/capabilities` reports `blender`/`3d` as reserved capacity only — never as callable job kinds while the closed job registry has no route for them; a flag/route agreement test fails when a reserved row is advertised as callable.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run codegen:check`.
Links: `../loop-lore/.tmp/research/blender-fullcycle-production-editing-2026-11-18.md` §1.1 and §5; `crates/residency/src/lib.rs:54-100` (`admission_info()` at :911-925), `crates/server/src/routes.rs:691-698`, `crates/server/src/jobs_registry.rs:78-111`; `FEAT-capabilities-api.md`, `TASK-capabilities-contract-test.md`.
Resources: per-test config stubs; no shared catalog.

## Change

- Mark reserved `blender`/`3d` queue rows with an explicit additive `reserved: true` field in the Capabilities DTO, or remove them from the advertised job list until a route exists. An additive field must not bump `api_version`.
- Extend the flag/route agreement test: every advertised callable job kind must have an admitting route, and every reserved row must be asserted non-callable.
- Keep consumer-side gating on `FeatureFlags`, not on queue rows. No consumer may treat a reserved row as an executable job.

## Acceptance detail

Once `FEAT-blender-job-route.md` lands and a route admits `blender`, the same test flips that row to callable without a prose edit. The epic's standing rule — every advertised `true` flag has a usable route — holds for both states.

## Implementation (2026-11-19)

- `QueueKindCapacity` gained the additive field `reserved: bool` (`api_version`
  stays `"v1"` — a reserved row hides a job class, it does not change one).
- The flag is DERIVED, never hand-set per row: `routes::ADMITTING_ROUTES`
  maps an admission kind to the route that admits a job of that kind, and a
  kind without an entry is advertised as reserved.
  - callable: `sd` (`/v1/images/generations`), `llm` (`/v1/chat/completions`),
    `embed` (`/v1/embeddings`), `rag` (`/v1/image-ops` — the CPU-only bucket
    the image-ops and audio routes take; the fixed 8-kind vocabulary has no
    CPU kind).
  - reserved: `ocr`, `comfy`, `blender`, `3d` — the ledger carries their
    capacity, no route accepts a job of that kind.
- `reserved_rows_agree_with_the_admitting_routes` (crates/server/tests/
  capabilities.rs) cross-checks every queue row against the generated
  `OpenAPI` document: a row that claims an admitting route must find that
  path in the document, and a kind with no route must be advertised as
  reserved. Verified by mutation: flipping the flag to `false` fails the
  test with `ocr has no admitting route but is advertised as callable`.
- Flipping a row to callable when its route lands is ONE line in
  `ADMITTING_ROUTES` — no prose edit, no test edit.
