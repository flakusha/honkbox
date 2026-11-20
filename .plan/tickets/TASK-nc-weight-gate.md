# TASK: Non-commercial weight gate (NC-flagged artifacts)

Status: proposed
Priority: P2
Epic: epic-binary-formats
Owner: unassigned
Acceptance: NC-licensed weights produce NC-flagged artifacts; commercial pipelines refuse them by default with a named error.
Verification: `cargo test --locked -p honkbox-server` (gate unit tests)
Links: `.tmp/research/product-licenses-include-vs-api-2026-11-20.md` §1, `TASK-model-license-registry.md`
Resources: per-test model dirs; no shared validator cache

## Change

- Weights with NC terms (Zero123++ weights CC-BY-NC 4.0; FLUX.2-dev VAE
  BFL-NC where it appears) stamp their artifacts non-commercial in the
  manifest; any commercial-flagged pipeline consuming them fails closed.
- Wires into the license registry's `commercial_terms` field; no separate
  list (unlike the QAL filename guard — weights are registry slugs).

## Acceptance detail

- Non-commercial research use is unaffected; the gate only binds
  commercial-flagged consumption. Research pipelines record the NC flag
  for audit.
