# TASK: Hardened pre-commit gates

Status: done
Priority: P1
Epic: epic-docs-tooling
Owner: unassigned
Acceptance: `.githooks/pre-commit` gates staged files correctly: quoted paths survive spaces, staged renames-with-edit are covered, partially staged or worktree-deleted paths are rejected before tool gates, missing bun/cargo produce named tool-unavailable failures, formatter/linter failures replay the captured report, and a successful dprint run is verified by restaging before commit proceeds. Generated-file drift stays in pre-push/CI via `codegen:check`.
Verification: `sh -n .githooks/pre-commit`, `bun test --isolate scripts/pre-commit-hook.test.ts`, `bun run test:unit`, `bun run check`
Links: `../loop-lore/.githooks/pre-commit`, `.plan/README.md`
Resources: per-test temp git repos; no shared core.hooksPath, index, port, DB, or fixture path

## Change

- `.githooks/pre-commit` (POSIX sh): OS-agnostic TMPDIR with EXIT-trap cleanup; staged-file list uses quoted paths and covers renamed-with-edit; partially staged or worktree-missing paths fail before any tool reads them; per-gate tool guards distinguish missing tools (bunx/cargo) from findings; dprint fmt + restage with captured failure report replay; oxlint with captured report; workspace `cargo fmt --all -- --check`; `sh -n` syntax gate; hook self-check; summary `N check(s) failed - commit blocked`. Generated files pass pre-commit; `codegen:check` in the pre-push hook and CI remains the drift gate.
- `package.json`: `hooks:install` / `hooks:check` scripts.
- CI: step label corrected to reflect that Markdown is not dprint-gated (no dprint Markdown plugin this round, per `TASK-markdown-gate-decision.md`).
- Tests: `scripts/pre-commit-hook.test.ts` — per-test temp git repos, parallel-safe per the `.plan/README.md` contracts.

Gates and formatters were skipped during this documentation slice; the orchestrator runs the verification commands once all slices land.
