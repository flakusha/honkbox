# TASK: Restore Ash verification and provenance gates

Status: done (2026-11-25; 9e3e39b)
Priority: P1
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: every Ash command referenced by package scripts, CI and hooks exists; `bun run spdx` includes shader provenance; the stale server `ash-sd` CI check is removed or retargeted until promotion lands.
Verification: `bun run audit:ash-lints`, `bun run spdx`, `bun run test:ash`, `bun run test:ash:parity`, `bun run hooks:check`, `bun run check`
Links: `.tmp/research/ash-promotion-refresh-2026-11-25.md` §2.1, `package.json`, `.github/workflows/ci.yml`, `.githooks/pre-push`, `scripts/check-ash-lint-tables.ts`, `scripts/check-ash-provenance.ts`, `crates/ash-backend/tests/ash_parity.rs`, `TASK-ash-backend-promote.md`
Resources: normal CPU test process; parity remains ignored/opt-in and skips without local checkpoint/vocab assets; no GPU lease or downloads.

## Verified drift (2026-11-25)

- **[FACT]** `package.json` defines `spdx = reuse lint` but has no `audit:ash-lints`, `test:ash` or `test:ash:parity` scripts.
- **[FACT]** `.github/workflows/ci.yml:344` calls missing `bun run audit:ash-lints`; `.githooks/pre-push:49,60` calls missing `bun run test:ash` and `bun run test:ash:parity`.
- **[FACT]** The CI reuse job is named “REUSE + shader provenance” but `bun run spdx` never invokes existing `scripts/check-ash-provenance.ts`.
- **[FACT]** `.github/workflows/ci.yml:317` checks `honkbox-server --features ash-sd`, while `crates/server/Cargo.toml:91` defines only empty `experimental-ash-sd`; the command cannot exercise a real server Ash path.

## Change

- Add `audit:ash-lints` for `scripts/check-ash-lint-tables.ts`.
- Add `test:ash` for the Ash crate SD/compute test graph and `test:ash:parity` for the ignored `ash_parity` integration test.
- Make `spdx` run both `reuse lint` and `scripts/check-ash-provenance.ts`.
- Remove the stale allow-failure server feature check or retarget it to an existing compile-reachable crate feature; server reachability belongs to `TASK-ash-backend-promote.md`.
- Keep CI and hook commands aligned with package scripts; do not add duplicate command definitions inside workflows.

## Acceptance detail

- [x] `bun run audit:ash-lints`, `bun run test:ash` and `bun run test:ash:parity` resolve and run.
- [x] `bun run spdx` fails when an Ash shader lacks `// Provenance:`.
- [x] CI and pre-push no longer reference missing package scripts.
- [x] No CI step claims server Ash reachability before the server feature/constructor exists.

## Non-goals

Implementing Ash server selection, changing backend numerics, downloading checkpoints, enabling parity by default or weakening existing gates.
