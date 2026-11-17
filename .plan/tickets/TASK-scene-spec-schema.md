# TASK: Versioned scene-spec schema

Status: proposed
Priority: P2
Epic: epic-model-feedback
Owner: unassigned
Acceptance: a versioned JSON scene spec describes objects, transforms, materials, and render views so critique loops can be replayed without an LLM.
Verification: `bun run codegen:check`, `bun run test:server`
Links: `.tmp/research/ReviewedCreative-FeedbackLoop.md`
Resources: per-test spec fixtures in runner tmp; no shared scene state

## Change

- Define a minimal schema with `schema_version`, objects, asset references by
  hash, transforms, material ids, and fixed camera viewpoints.
- Validate references against the job bundle; reject unknown fields.
- Generate deterministic JSON with stable key order; no wall-clock metadata.

## Acceptance detail

Two runs from the same spec and bundle produce the same multi-view render set
(structurally), and the spec can be inspected without a model.
