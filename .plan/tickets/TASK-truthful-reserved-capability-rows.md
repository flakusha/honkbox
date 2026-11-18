# TASK: Truthful reserved capability rows

Status: proposed
Priority: P1
Epic: epic-capabilities-contract
Owner: unassigned
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
