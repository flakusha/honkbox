# TASK: Ash sigma tables

Status: proposed
Priority: P3
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: versioned sigma grids ship with named presets and CPU golden vectors; zero overlap with step-update functions
Verification: `cargo test --locked -p honkbox-ash --lib sigma_tables`
Links: `.tmp/research/api-provisioning-extended-fields-2026-11-25.md` §4 (sibling: scheduler-host ticket); `crates/ash/` scheduler seam
Resources: per-test tmp dirs; fixture JSON; port 0; no shared state

## Change

- Sigma-tables side only: versioned float tables as deterministic numeric data, `schedule_sha256` recorded in bundle provenance, plus block golden vectors on CPU.
- Named presets only (import + pin + golden); never hand re-derive upstream tables.
- Sibling boundary: the scheduler-host ticket owns step order, callback/progress, interrupt, and sampler dispatch; this ticket owns no step-update functions and re-implements no host.
- Contract between sides: `scheduler:{name, version}` + `sigmas_sha256` recorded in `transforms[]`; mismatch returns `409`.

Neural-network weight/tokenizer license and inference-use applicability are the responsibility of the operator; honkbox records digest/source and never clears, grants, or assumes a license.

## Acceptance criteria

- [ ] Named sigma presets pinned with `schedule_sha256` and CPU golden vectors passing.
- [ ] No step-update/scheduler-host functions added or duplicated in this ticket.
- [ ] Digest mismatch surfaces as 409 per the `transforms[]` contract.
