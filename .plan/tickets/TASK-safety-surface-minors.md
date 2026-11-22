# TASK: Minors-facing safety surface for generation

Status: proposed
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