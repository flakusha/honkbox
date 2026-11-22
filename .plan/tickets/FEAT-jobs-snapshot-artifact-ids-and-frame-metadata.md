# FEAT: Jobs snapshot artifact ids and frame metadata

Status: done
Priority: P2
Epic: epic-binary-formats
Owner: unassigned
Acceptance: duplicate of `TASK-job-result-artifact-selection-metadata.md` (gap report `.tmp/research/llama-swap-loop-lore-gaps-2026-11-22.md` G6/G7). Implemented once: `job_result` frames carry additive `artifact_name`, `artifact_kind`, `artifact_index`, `artifact_primary`; the live `GET /v1/jobs` entry carries a bounded (64) `artifacts: Vec<JobArtifactRef>` snapshot so a dropped frame is recoverable without a manifest round trip. Exactly one primary per job (images index 0, edits singleton, Blender first `model` with index-0 fallback; annotation re-emit artifact-less). No `api_version` bump; old fields unchanged.
Verification (2026-11-22): exact commands/results in `.tmp/artifact-metadata-verify.log`. `rustfmt --edition 2024 --check` exit 0; `cargo test --locked --offline -p honkbox-api-types export_ts` 1 passed; `cargo test --locked --offline -p honkbox-server --bin honkbox jobs_registry` 12 passed; `cargo test --locked --offline -p honkbox-server --test jobs --test edits --test blender_jobs --test images` 662 passed / 4 suites; `bun scripts/codegen-openapi-check.ts` wrote openapi.json covering 53 routes; `bun run codegen:check` passes all codegen steps but its final `git diff --exit-code -- packages/web-ui/src/gen/` fails only on uncommitted generated diffs (expected, no hand-edits); clippy `-D warnings` on both crates exit 0 (only a dependency build-script warning); `dprint fmt packages/web-ui/src/gen` exit 0; `API_VERSION` still `"v1"`.
Links: `TASK-job-result-artifact-selection-metadata.md`; `.tmp/research/next-tickets-implementation-plan-2026-11-22.md` §C.
