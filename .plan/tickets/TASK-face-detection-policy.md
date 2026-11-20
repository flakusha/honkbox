# TASK: Face/person detection privacy policy decision

Status: done
Priority: P3
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: a written policy decision (ship / ship-gated-off / never) for face/person detection, recorded before any engine research or design proceeds.
Verification: none (policy artifact reviewed in session)
Links: `.tmp/research/ReviewedRecognition.md` §2 (rank 7 flag)
Resources: none

## Change

- Decision memo covering: consent, local-only retention, default-off gating, capabilities
  advertisement, downstream export implications for an AGPL local-first studio.
- No engine selection, no code, no DTOs in this ticket; any future implementation ticket links
  this decision.

## Acceptance detail

Policy decided and linked; recognition surface carries no face/person kind until then.

## Verification Notes (2026-11-20)

Decision recorded in
`.tmp/research/face-detection-policy-decision-2026-11-20.md`:
**ship-gated-off**. Five gates are mandatory before any implementation:
per-subject consent, local-only bounded retention, compile- plus runtime
default-off gating, honest capabilities advertisement, and exclusion from
bundles/exports unless explicitly declared. Current recognition surface remains
`Barcode` + `Ocr` (`crates/recognition-worker/src/lib.rs`); no engine, DTO or
route was added by this policy ticket.
