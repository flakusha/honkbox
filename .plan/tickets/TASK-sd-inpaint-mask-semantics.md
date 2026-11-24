# TASK: Inpaint mask semantics

Status: done (2026-11-24)
- Explicit mode: `Some("inpaint")` with lowercase mask sha256 when a mask is
  supplied; `Some("replace")` with no hash otherwise. Both recorded in
  `request.json` and `input.json`; mismatched mask IHDR dims are a 400 before
  admission (`crates/server/src/routes/images_edit.rs:366-383`).
- Verified 2026-11-24: `cargo test --locked -p honkbox-server --test edits`
  184/184 (asserts both modes, hash, dimension rejection); server clippy clean;
  `cargo fmt --check` shows no diff in `tests/edits.rs`.
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

## Review note (2026-11-24)

Top next in-master item while `feat/ash-vulkan-backend` is cooked: P1, no
external binary, no DTO/schema drift, no sibling lane claim. Earlier P1/P2
waves (rerank, job metadata, edits delivery, voices) are all done. Resource
contract: each test owns a unique temp image/mask file pair with `Drop`
teardown; no shared vendor context or GPU. See
`.tmp/research/master-next-items-while-ash-cooks-2026-11-24.md`.
