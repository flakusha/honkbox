# TASK: Hardened pre-commit gates

Status: done
Priority: P1
Epic: epic-docs-tooling
Owner: unassigned
Acceptance: `.githooks/pre-commit` gates staged files correctly: quoted paths survive spaces, staged renames-with-edit are covered, missing bun/cargo produce named tool-unavailable failures, staged files under `packages/web-ui/src/gen/` block the commit, formatter/linter failures replay the captured report, and a successful dprint run is verified by restaging before commit proceeds.
Verification: `sh -n .githooks/pre-commit`, `bun test --isolate scripts/pre-commit-hook.test.ts`, `bun run test:unit`, `bun run check`
Links: `../loop-lore/.githooks/pre-commit`, `.plan/README.md`
Resources: per-test temp git repos; no shared core.hooksPath, index, port, DB, or fixture path

## Change

- `.githooks/pre-commit` (POSIX sh): OS-agnostic TMPDIR with EXIT-trap cleanup; staged-file list uses quoted paths and covers renamed-with-edit; per-gate tool guards distinguish missing tools (bunx/cargo) from findings; dprint fmt + restage with captured failure report replay; oxlint with captured report; workspace `cargo fmt --all -- --check`; `sh -n` syntax gate; hook self-check; generated-file tripwire for `packages/web-ui/src/gen/`; summary `N check(s) failed - commit blocked`.
- `package.json`: `hooks:install` / `hooks:check` scripts.
- CI: step label corrected to reflect that Markdown is not dprint-gated (no dprint Markdown plugin this round, per `TASK-markdown-gate-decision.md`).
- Tests: `scripts/pre-commit-hook.test.ts` — per-test temp git repos, parallel-safe per the `.plan/README.md` contracts.

Gates and formatters were skipped during this documentation slice; the orchestrator runs the verification commands once all slices land.
