# TASK: Checked-in capabilities contract fixture

Status: done (2026-11-22)
Priority: P2
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: a versioned golden key-path snapshot of
`/v1/capabilities` lives in the repo; a server test fails when a key is added,
removed or renamed, including nested `modality`, `features.decisions`,
`features.audio_cpp` and the full feature-key set (the `blender` gap is the
live example). Array contents and compile-dependent scalars are excluded so
one fixture holds for every feature set. The file is a copy fixture, not a
submodule, so loop-lore can assert its parser against the same contract. Verification: `cargo test --locked -p honkbox-server --test capabilities`,
`bun run codegen:check`. Links: `.tmp/research/harness-giwt-vcs-integration-2026-11-20.md`
§3 S8, §4 S12. Resources: read-only fixture; no GPU, no new dependency.

## Change

- Capture the capabilities key paths as a checked-in `.paths` fixture.
- Add one sorted-path-set test; drift requires an explicit reviewed fixture update.
- Add the fixture path to the capabilities ticket/README cross-reference.

## Non-goals

Runtime schema negotiation, submodules, cross-repo CI orchestration.


## Verification Notes (2026-11-22)

- `cargo test --locked -p honkbox-server --test capabilities capabilities_key_paths_match_golden_fixture` — pass (default features).
- Same command with `--no-default-features` — pass; one fixture covers both feature sets.
- Fixture: `crates/server/tests/fixtures/capabilities-v1.paths` (63 key paths).
- Full `bun run check && bun run test` blocked by concurrent, uncommitted
  `crates/server/src/apps.rs` / `src/routes/apps.rs` / `src/nsfw.rs` edits from
  another lane; none of those files are touched by this ticket.
