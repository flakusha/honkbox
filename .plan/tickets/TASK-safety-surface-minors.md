# TASK: Minors-facing safety surface for generation

Status: done (2026-11-22)
- Verified 2026-11-22 (`.tmp/minors-prompt-gate-verify.log`): 153-gate tests pass; combined 153+180+174 image suites pass; targeted clippy and rustfmt clean. Consumers map the existing additive `capabilities.nsfw` row (available/policy/threshold), not a new FeatureFlags row; the minors prompt gate is default-deny with a fixed named reason.
Priority: P1
Epic: epic-trust-safety
Owner: unassigned
Acceptance: image-generation routes refuse explicit minors-facing requests with
a named error, pass outputs through an NSFW score gate whose decision is
reported (never silently dropped), and expose a capability flag that a
multi-user consumer can map to its own age gate.
Verification: `bun run test:server`, `bun run codegen:check`, `bun run lint:rs`,
plus route tests for block, flag and disabled-engine cases.
Links: `.tmp/research/content-experience-gaps-2026-11-20.md` §3 G7,
`TASK-nsfw-score-signal.md`, `TASK-face-detection-policy.md`.
Resources: per-test temp data dir; no fixed port.

## Change

- Wire the existing NSFW score signal into image routes as a policy gate with a
  bounded reason string; add a prompt blocklist for the obvious cases.
- Add an additive `FeatureFlags` row only if the consumer needs it; regenerate
  TS via codegen, never hand-edit `packages/web-ui/src/gen/`.
- Keep the policy opt-in for operators but default-deny on minors-facing input.

## Non-goals

Building age-gate UI (loop-lore owns it), classifier training, CSAM hashing.