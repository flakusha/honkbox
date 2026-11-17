# TASK: Inpaint mask semantics

Status: proposed
Priority: P1
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: mask mode (replace vs inpaint) is explicit, tested, and recorded in the job sidecar with a mask hash.
Verification: `bun run test:server`, `bun run test:core`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedSupport-SD.md` expansion table; `routes.rs:3273-3350,3431-3637`
Resources: per-test image/mask temp files; no shared vendor context

## Change

- Audit how the multipart `mask` reaches the vendor mask path; add explicit
  mode tests for replacement and inpainting.
- Reject mismatched dimensions/bad masks before admission with 400.
- Record mask hash + mode in `BundleInputs`/sidecar so a job is reproducible.

## Acceptance detail

A generated edit can be replayed from its sidecar without guessing which mask
mode produced it; no silent vendor default is relied upon.
