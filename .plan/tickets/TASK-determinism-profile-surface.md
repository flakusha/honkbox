# TASK: Determinism profile surface

Status: proposed
Priority: P2
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: additive determinism profile field on capabilities and job records with stub vs GPU values differing
Verification: `cargo test --locked -p honkbox-server --test determinism_profile`
Links: `.tmp/research/determinism-extended-fields-2026-11-25.md`, `TASK-determinism-contract.md`, `crates/api-types`, `crates/server/src/determinism.rs`
Resources: per-test tmp dirs; port 0; no shared state

## Change

- Add additive `determinism_profile` string field (`contract` | `seeded` | `bitexact` | `nondeterministic`) plus `deterministic_kernels_available` / `deterministic_kernels_slower` flags on the per-backend capabilities entry.
- Record the profile plus `seed` on job/bundle records; stub jobs record `seeded`, seedless ops bundles record `contract` with null seed, GPU records default to best-effort `seeded` [UNKNOWN: GPU certification level pending measurement].
- `bitexact` is never default and never claimed without pinned-version evidence.

## Acceptance criteria

- [ ] Stub vs GPU capability entries report differing profile values.
- [ ] Stub jobs record `seeded`; seedless ops bundles record `contract` with null seed.
- [ ] `bitexact` is unreachable without pinned engine/driver versions.

Neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license.
