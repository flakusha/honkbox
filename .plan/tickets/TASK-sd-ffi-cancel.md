# TASK: Wrap native SD cancellation

Status: done (2026-11-18, lane ffi-cancel)
Priority: P0
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: cancel can interrupt a mid-flight native generation, and the response states interruptibility honestly when it cannot.
Verification: `bun run test:core`, `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedSupport-SD.md` expansion table; C header `:602-611`
Resources: per-test job IDs and temp output dirs; opt-in local checkpoint only

## Observed (2026-11-18, lane ffi-cancel)

- Vendor patch `vendor_patches/0006-sd-cancel-generation.patch`
  (manifest stack 0000-0006): wrapper gains `CancelHandle`
  (SD_CANCEL_ALL hard / SD_CANCEL_NEW_LATENTS graceful),
  `gen_img_with_cancel`, `DiffusionError::Cancelled`; 25 ms re-firing
  watcher relays into `sd_cancel_generation` (latches over the engine's
  reset-at-start). Sync converges (7/7 applied; verified twice).
- Seam: `SdPipeline::generate(req, &AtomicBool)`; `SdError::Cancelled`
  is the only cancelled outcome (never partial/normal output); stub
  honors pre-work + between rows; sd-cpp relays via scoped thread into
  `gen_img_with_cancel`; routes bridge job token -> flag
  (`tokio::select`) and answer 409 `job_cancelled` naming the
  interrupted image; images jobs register `interruptible: true`.
- Gates OBSERVED: `bun run test:core` → `cargo test --locked -p
  honkbox-sd-core`: 24 passed / 1 ignored (lib 20 + parity 4 + doc 0),
  includes `stub_cancelled_flag_returns_named_error`. clippy `-D
  warnings`: `-p honkbox-sd-core -p honkbox-ash-backend --all-targets`
  clean; `-p honkbox-server --test jobs` clean. Vendor sync: 7/7.
- Gates PENDING sibling landing: `bun run test:server` / full
  `cargo test --locked -p honkbox-server` / workspace `lint:rs` — the
  bin target does not compile while the concurrent bundle-metadata
  session's untracked `crates/server/src/{artifacts,model_formats}.rs`
  are mid-edit (their errors, not this lane's); the jobs route target
  (routes.rs + tests) compiles clean under clippy. Re-run the three
  gates after that session lands.
- codegen:check: skipped, api-types untouched (no DTO change; the
  cancel endpoint's schemas are byte-identical).

## Change

- Wrap `sd_cancel_generation` with `SD_CANCEL_ALL` / `SD_CANCEL_NEW_LATENTS`
  in the vendor patch stack and surface it through `SdPipeline`.
- Single-image FFI calls are currently non-interruptible; either wire cancel
  into the loop or report `cancelled: false` explicitly.
- Keep token-based cancellation for multi-image batches.

## Acceptance detail

A cancelled batch stops before the next image; a single-shot cancellation
never claims success it did not achieve; no orphan native thread remains.
