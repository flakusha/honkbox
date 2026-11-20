# TASK: Redistribution-restricted pack guard (QAL)

Status: proposed
Priority: P1
Epic: epic-binary-formats
Owner: unassigned
Acceptance: QAL-licensed (and similar no-redistribution) packs can never enter bundles or serve paths; attempts fail named.
Verification: `cargo test --locked -p honkbox-server` (gate unit tests)
Links: `.tmp/research/product-licenses-include-vs-api-2026-11-20.md` §1, `TASK-model-format-validation.md`
Resources: per-test model dirs; no shared validator cache

## Change

- Filename/license-list gate beside the model-format gate: Quaternius QAL
  v1.0 (commercial games OK, **no redistribution as assets**) + any future
  no-redistribution pack ids. Rejection names the pack + term.
- Applies to bundle export AND serve paths; dev-time local use is unaffected
  (the restriction is on redistribution, not use).
- Shares the sidecar convention (`validation_sidecar` provenance shape).

## Acceptance detail

- A valid CC0 pack passes untouched; a QAL pack fails with the term named,
  never a silent skip. List lives with the registry, not inline.
