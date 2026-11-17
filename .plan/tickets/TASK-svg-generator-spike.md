# TASK: SVG generator spike

Status: proposed
Priority: P3
Epic: epic-vector-materials
Owner: unassigned
Acceptance: one external vector generator produces an SVG from a prompt or image and the artifact passes `TASK-svg-artifact-validation.md`; no native SVG model in honkbox.
Verification: opt-in local fixture; `bun run lint:rs`
Links: `.tmp/research/ReviewedCreative-SVG.md` (VectorFusion, SVGDreamer, OmniSVG)
Resources: per-test generator output dirs; no CI model download

## Change

- Try one external generator through the process supervisor; record binary,
  version, license, and exact argv in the sidecar.
- Judge output by structural validity (parses, safe, preview matches request),
  not byte equality.

## Acceptance detail

If the spike is unconvincing, close the epic's generation follow-up as postponed;
no route is added by this ticket.
