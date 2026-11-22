# BUG: Fizgig cache stages resolve VAE/text-encoder per family

Status: proposed
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: the latents/text cache stages pass the VAE/text-encoder file the
selected family actually needs; SDXL keeps using the DiT fallback because its
checkpoint carries both, while every other allowlisted family is rejected with
a named 400 before a lease or job record is created when it lacks an explicit
`vae`/`text_encoder`. Operator configuration carries the per-family mapping.
Verification: `cargo test --locked --offline -p honkbox-server --test fizgig`
with a fake CLI asserting argv for one SDXL and one non-SDXL family, plus
`bun run lint:rs`.
Links: `.tmp/research/fizgig-training-python-sidecar-2026-11-22.md`,
upstream pinned tag `v9.8.4` (`src/fizgig/families/cache.py`, `sdxl.py`),
`crates/server/src/routes/fizgig_run.rs:99-115`.
Resources: no new port; fake CLI writes under a per-test temp dir.

## Change

- Replace `unwrap_or(&plan.dit)` in both cache-stage argv builders with a
  family-aware resolver backed by `[fizgig] families` config (or an explicit
  per-family VAE/TE-required table). Default to "required" for unknown families.
- Validate required paths in `post_fizgig_train` before admission and return a
  named 400, never a 202 that fails in stage 1.
- Extend `FizgigTrainRequest` docs only if the contract changes; no new DTO.

## Non-goals

Model download, auto-detection of files inside checkpoints, router mode.