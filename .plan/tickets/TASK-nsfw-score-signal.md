# TASK: NSFW score signal on rendered images

Status: done (2026-11-20, NSFW score lane)
Priority: P0
Epic: epic-trust-safety
Owner: unassigned
Acceptance: every rendered image gets `{nsfw_score, classes, model_version}` in bundle annotations and in the job result; default policy is `flag`, never silent drop; no blocking without explicit config.
Verification: `cargo test --locked -p honkbox-server`, `bun run ci`
Links: `content-experience-gaps-2026-11-20.md` §8, `TASK-face-detection-policy.md`
Resources: per-test temp dirs; stub classifier; no shared state

## Change

- Attach a classifier seam to the render result path so every rendered image
  records `{nsfw_score, classes, model_version}` in both the bundle sidecar
  annotations and the job result DTO.
- Default policy `flag`: annotations are always present and surfaced to the
  consumer, which enforces its own policy. Dropping or withholding an image by
  default is not allowed; blocking happens only under explicit config.
- Keep the signal engine-side and local; no remote calls and no face/person
  detection kind (that remains gated by `TASK-face-detection-policy.md`).

## Acceptance detail

- With a stub classifier, every render's bundle annotations and job result
  carry the three fields with the stub's score/classes/version.
- Default config flags only — no config means no image is dropped and no job is
  blocked; a blocking policy requires an explicit setting.
- Per-test temp dirs and a stub classifier keep parallel tests isolated; no
  shared classifier state or port.
