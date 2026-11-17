# TASK: Render-critique loop fixture

Status: proposed
Priority: P3
Epic: epic-model-feedback
Owner: unassigned
Acceptance: a loop-lore-side multimodal agent can critique fixed multi-view renders and apply bounded edits to the scene spec, with every version retained in the job bundle.
Verification: local fixture with a stub critique provider; `bun run test:server`
Links: `.tmp/research/ReviewedCreative-FeedbackLoop.md`
Resources: per-test render dirs and stub provider on assigned port; no live model

## Change

- Depends on `TASK-scene-spec-schema.md`.
- Render front/side/top views from the spec; feed images + spec to a stub
  critic; apply only allowlisted field edits.
- Store each spec version, critic output, and render hashes in the bundle.

## Acceptance detail

No unbounded edit loop, no GUI automation, and no byte-equality claim across
renderers; loop-lore owns agent state, honkbox owns artifacts.
