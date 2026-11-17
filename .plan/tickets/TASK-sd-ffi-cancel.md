# TASK: Wrap native SD cancellation

Status: proposed
Priority: P0
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: cancel can interrupt a mid-flight native generation, and the response states interruptibility honestly when it cannot.
Verification: `bun run test:core`, `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedSupport-SD.md` expansion table; C header `:602-611`
Resources: per-test job IDs and temp output dirs; opt-in local checkpoint only

## Change

- Wrap `sd_cancel_generation` with `SD_CANCEL_ALL` / `SD_CANCEL_NEW_LATENTS`
  in the vendor patch stack and surface it through `SdPipeline`.
- Single-image FFI calls are currently non-interruptible; either wire cancel
  into the loop or report `cancelled: false` explicitly.
- Keep token-based cancellation for multi-image batches.

## Acceptance detail

A cancelled batch stops before the next image; a single-shot cancellation
never claims success it did not achieve; no orphan native thread remains.
