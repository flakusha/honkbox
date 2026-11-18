# TASK: Image tagging/classification spike via ort

Status: proposed
Priority: P3
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: a spike ticket outcome — one classification ONNX model (MobileNet or CLIP-class) runs in-process under ort, tags attached to recognition results, with the per-weight license recorded before any download default flips on.
Verification: `bun run test:server`, `bun run check`
Links: `.tmp/research/ReviewedRecognition.md` §2 (rank 4); TASK-recognition-provider-trait
Resources: spike fixture images; model cache dir

## Change

- Spike scope only: pick model, verify HF card license (several CLIP checkpoints carry non-standard
  tags — record in ticket + NOTICE), measure CPU latency, decide feature promotion.
- `RecKind::Tag` backend behind `TAG_COMPILED`; top-k labels with scores; no route if spike fails
  the value/cost bar — close with findings instead of shipping a half seam.

## Acceptance detail

Spike concludes promote/defer with recorded latency + license evidence; labels `kind=tag` only.
