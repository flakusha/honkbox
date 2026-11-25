# TASK: Capability honesty gate

Status: done
Priority: P0
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: enforced route caps (Blender input/output byte caps, timeout bound, stage non-empty rule; poly-count cap if any) are advertised in `GET /v1/capabilities` and a test pins payload values to the route-enforced bounds. The flag/route-agreement, Blender-probe-truth, Workflow-row and OIIO-omission items already hold on master (see Evidence); this ticket now covers only the caps-advertising delta.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server --test capabilities`, `bun run codegen:check`
Links: `.tmp/research/dcc-closed-loop-honkbox-2026-11-25.md` (H-DCC6); `.plan/tickets/TASK-truthful-reserved-capability-rows.md`; `.plan/tickets/FEAT-capabilities-api.md`; `crates/server/src/routes.rs`; `crates/external-tools/src/lib.rs`
Resources: per-test config stubs; no shared catalog; no live probes

## Change (narrowed 2026-11-25; items 1-4 landed, see Evidence)

- [landed] Flag/route agreement test: `crates/server/tests/capabilities.rs:742-790` (`reserved_rows_agree_with_the_admitting_routes`) asserts callable ⇔ mounted admitting route and reserved ⇔ non-callable.
- [landed] Blender capability follows the probed engine; unprobed is honest absence: `crates/server/tests/capabilities.rs:824-868` (`blender_flag_agrees_with_the_probe_registry`).
- [landed] `Workflow` row means `/object_info` probe only: `crates/external-tools/src/lib.rs:560-561` ("A version alone does not prove workflows") and the probe requires a non-empty node catalog (`lib.rs:507-584`). Template approval cannot be implied today because no workflow-template surface exists yet (`FEAT-comfyui-workflow-pipeline`); template admission stays a separate gate by construction.
- [landed] OIIO omission stays explicit: `ToolId::Oiio` exists (`crates/external-tools/src/lib.rs:68-69`) but is excluded from `ToolId::ALL` (`lib.rs:83-90`) and discovery hard-fails with `unreachable!("OIIO is not registry-discovered")` (`lib.rs:384`); landing commit `80cd121` (ancestor of master) added the empty-`object_info` refusal test.
- [open] Enforced caps (byte, stage, timeout) are NOT advertised: route bounds live only in `crates/server/src/routes/blender.rs:35-42` (`MAX_INPUT_BYTES` 512 MiB, `MAX_OUTPUT_BYTES` 2 GiB, `MAX_TIMEOUT_SECS` 3600) and are enforced at `blender.rs:248-259,273-280,300-303`, while the capability payload carries only queue caps and body-byte limits (`crates/server/src/routes/capabilities.rs:253-259`, `crates/api-types/src/lib.rs:329,367-377`). No poly-count cap exists anywhere in `crates/` [UNKNOWN whether one is wanted]; advertise what is enforced, name poly-count only if it is added.
- Covers the H0 package from `engine-integration-research-2026-11-24/honkbox-heavy-lifting.md` (H0 landed except caps advertising).

## Acceptance detail (remaining delta only)

1. The capability payload (or the Blender tool entry in it) lists the enforced input-byte, output-byte and timeout bounds with values equal to `crates/server/src/routes/blender.rs:35-42`.
2. A test fails when the advertised bound drifts from the route constant.
3. Anything not enforced is not advertised (no poly-count row unless a poly-count cap lands).

Note: `TASK-blender-terminal-hardening` is in flight in `tree/fix-blender-terminal-contract` and may reshape Blender capability/status reporting; if it moves the bound constants or the payload shape, rebase this delta's file targets onto its landing before implementing.

## Evidence (audit 2026-11-25, lane plan-stale-p0-reconcile)

| Acceptance item | Verdict | Evidence |
| --- | --- | --- |
| Callable kinds have admitting routes; reserved rows non-callable | met | `crates/server/tests/capabilities.rs:749` |
| Blender flag follows probed engine | met | `crates/server/tests/capabilities.rs:831-868` |
| Workflow row is probe-only | met | `crates/external-tools/src/lib.rs:560-561` |
| OIIO omission explicit | met | `crates/external-tools/src/lib.rs:68-69,83-90,384` |
| Caps advertised alongside capability | NOT met | bounds only at `crates/server/src/routes/blender.rs:35-42`; payload lacks them (`crates/server/src/routes/capabilities.rs:253-259`) |

Landing: agreement/blender/workflow/OIIO behavior landed through `feat/p0-dcc-honesty-fastwins` (`80cd121` `test(ext-tools): empty object_info never claims Workflow`, an ancestor of master).
Verification commands were not run in this lane (docs-only lane, no cargo build); the code and test lines above were read directly on master `0b89706`.

## Verification Notes (2026-11-25)

- Implemented the caps delta: `BlenderLimitsCapability` on `Capabilities` (`crates/api-types/src/lib.rs`), route-sourced `blender::limits()` (`crates/server/src/routes/blender.rs`), payload wiring (`routes/capabilities.rs`), and OpenAPI schema registration (`routes/openapi.rs`).
- Drift test: `crates/server/tests/capabilities.rs::blender_limits_match_enforced_route_bounds` compares all four advertised values with `blender::limits()` and rejects a poly-count field.
- Verified 2026-11-25: `cargo test --locked -p honkbox-api-types export_ts` 1 pass; OpenAPI regenerated from the locally built server (64 mounted routes covered); `cargo test --locked -p honkbox-server --test capabilities` 199 pass, 0 fail; `bun run typecheck` clean; `bun test --isolate` 46 pass; `cargo fmt --all -- --check` clean.

## Non-goals

- New capability rows for FreeCAD/MuseScore/REAPER/Natron/QGIS/Scribus (gated on H-ADD1–H-ADD6 preconditions).
- Second capability registry or agent framework.
