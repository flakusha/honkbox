# TASK: Remove stale video/vision claims

Status: done (2026-11-18, honkbox-openapi-coverage)
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

## Landed

README claims checked against the code (`crates/` at this commit):

- `README.md:3` — "image/video generation" → "image generation". No video
  job class exists: the only `video` in the tree is the deliberate
  `capabilities.video = false` flag (`api-types/src/lib.rs:110`) and the
  `FormatKind::VideoFrame` format-registry placeholder for a future epic.
- `README.md:70` — `crates/sd-core` no longer claims video job types.
- `README.md:74` — `crates/registry` is "model-dir scan, weight metadata,
  SQLite catalog cache"; the hf-hub half is a documented follow-up in
  `registry/src/db.rs`, and downloads live in `crates/server/src/downloads.rs`.
- `README.md:116` — the `X-Job-Id` claim is now scoped to ADMITTED requests:
  a request refused before admission (bad body, model without the required
  capability) never registers a job and carries no header. The header is
  stamped only where a job id exists (`routes.rs`, `with_job_id` call sites).
- `README.md:222` (Docs) — the pointer to `.tmp/research/Reviewed*.md` was
  dead in a fresh clone (`.tmp/` is gitignored scratch). It now points at
  `AGENTS.md`, `.plan/`, and the generated-and-served API reference
  (`GET /v1/openapi.json`), and labels the research directory as local
  scratch.

Verification: `bun run check` exit 0; `bun run test` exit 0; `git diff --check`
clean. Documentation-only, no code behavior change.
