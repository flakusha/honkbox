# TASK: Admission dedupe analysis

Status: proposed
Priority: P3
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: design-only dedupe-key field list plus must-NOT-coalesce counterexamples; implementation explicitly gated
Verification: `giwt plan check-links`
Links: `.tmp/research/determinism-extended-fields-2026-11-25.md`, `TASK-determinism-contract.md`
Resources: per-test tmp dirs; port 0; no shared state

## Change

- Design-only: define the in-flight admission dedupe key (content hash plus backend id, determinism profile, and output-format dims) with singleflight-style in-flight-only coalescing semantics; finished-artifact reuse is out of scope (separate cache decision).
- Document must-NOT-coalesce counterexamples: same content hash on different backends, streaming vs sync delivery expectations, side-effecting requests, and volatile-key-only differences that MUST coalesce [UNKNOWN: key completeness pending owner review].

## Acceptance criteria

- [ ] Dedupe-key field list is published in the ticket.
- [ ] Each counterexample states share vs no-share behavior.
- [ ] Implementation is explicitly gated: no product code in this ticket.

Neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license.
