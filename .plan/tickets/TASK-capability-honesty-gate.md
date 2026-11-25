# TASK: Capability honesty gate

Status: proposed
Priority: P0
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: every advertised callable job kind has an admitting route; every reserved row is asserted non-callable; Blender flag follows probed engine; Workflow row is never template/pack approval; OIIO omission from discovery stays explicit; enforced caps listed
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server --test capabilities`, `bun run codegen:check`
Links: `.tmp/research/dcc-closed-loop-honkbox-2026-11-25.md` (H-DCC6); `.plan/tickets/TASK-truthful-reserved-capability-rows.md`; `.plan/tickets/FEAT-capabilities-api.md`; `crates/server/src/routes.rs`; `crates/external-tools/src/lib.rs`
Resources: per-test config stubs; no shared catalog; no live probes

## Change

- Extend the flag/route agreement test: every advertised callable job kind must resolve to a mounted admitting route; every reserved row must assert non-callable.
- Blender capability follows the probed engine version; unprobed engine is honest absence, never a version guess.
- `Workflow` capability row means `/object_info` probe only, never template/pack approval or version compatibility.
- OIIO `ToolId` exists but stays omitted from `ToolId::ALL` discovery until a real adapter lands; omission is documented, not a gap.
- Enforced caps (poly-count, byte, stage, timeout) are advertised alongside the capability, not buried in route code.
- Covers the H0 package from `engine-integration-research-2026-11-24/honkbox-heavy-lifting.md`.

## Acceptance detail

1. Mutating any advertised row to callable without its route fails the agreement test.
2. Blender row without a probed engine reports unavailable with a named reason.
3. Workflow row never implies template approval; template admission is a separate gate.
4. Caps in the capability payload match route-enforced bounds.

## Non-goals

- New capability rows for FreeCAD/MuseScore/REAPER/Natron/QGIS/Scribus (gated on H-ADD1–H-ADD6 preconditions).
- Second capability registry or agent framework.
