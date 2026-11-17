# TASK: Remove stale video/vision claims

Status: proposed
Priority: P0
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: README no longer claims video support or video job types in `sd-core`, and all modality claims match compile-time defaults.
Verification: `git diff --check`, documentation review; `bun run codegen:check` if DTO comments change
Links: `.tmp/research/ReviewedSupport-CurrentState.md`, `ReviewedSupport-Video.md` §1
Resources: none (documentation-only ticket)

## Change

- `README.md:3` says “image/video generation”; zero `video` code exists in
  `crates/` at HEAD `591efc0`. Remove or qualify as planned/postponed.
- `README.md:70` claims `sd-core` holds video job types — false; point to
  `FEAT-video-output.md` (postponed) instead.
- Keep the `llm-vision` opt-in wording accurate until
  `TASK-llm-vision-default.md` lands; do not promise video behavior.

## Acceptance detail

No code behavior changes; this ticket only removes claims that would cause a
client or planner to build against a non-existent surface.
