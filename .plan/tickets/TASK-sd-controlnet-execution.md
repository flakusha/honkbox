# TASK: Execute ControlNet requests

Status: implemented — parked on shared-file reconciliation; lane gates below
Owner: honkbox-controlnet-exec
Priority: P0
Epic: epic-sd-native-backend
Acceptance: a valid ControlNet request runs through the vendored engine instead of returning 501 `controlnet_unsupported`.
Verification (observed 2026-11-18): `cargo test --locked -p honkbox-sd-core` 22 passed / 0 failed (incl. new `control_type_from_name_uses_the_engine_name_table`, `stage_control_threads_type_and_weights_and_refuses_missing_weights` — the latter asserts the type lands on the vendor config via `Config::control_type()`); `bun scripts/vendor-patches.ts sync` converges (6 applied / 0 now, re-run + `--dry-run` report "tree is converged"); `cargo check --locked -p honkbox-server --all-targets` — zero errors in every lane file; `cargo clippy -p honkbox-sd-core -p honkbox-api-types -p honkbox-ash-backend --all-targets` — 0 issues. PENDING tree-green (blocked ONLY by the binary-formats wave's in-flight `src/formats.rs`/`src/svg.rs`/`src/ora.rs`/image-ops hunks, not this lane): `test:server` (so the new `controlnet_type_execution_contract` jobs test is written but NOT yet executed), workspace `lint:rs` (routes.rs clippy unverified), `codegen:check`. Commit parked because `routes.rs`/`main.rs` also carry the integration lane's in-flight hunks.
Links: `.tmp/research/ReviewedSupport-SD.md` §Partial; `routes.rs` `resolve_controlnet`; `vendor_patches/0005-gen-img-control-type.patch`
Resources: per-test control image temp files; one opt-in local checkpoint fixture, no CI downloads

## Change (landed 2026-11-18)

- `Txt2ImgRequest::control_type` (`sd-core`): named engine type rides the
  existing pair; omitted stays vendor default `none` (preprocessed map,
  byte-identical prior behavior).
- `vendor_patches/0005-gen-img-control-type.patch`: vendor `Config` gains
  `control_type` (builder default NONE) threaded into `sd_img_gen_params_t`
  (was hardcoded NONE); generation now verifies the weights are actually
  loaded (`sd_ctx_has_control_net`) and hot-loads them otherwise
  (`sd_ctx_load_control_net`), unloading after the render
  (`sd_ctx_unload_control_net`); load/unload failures are a loud
  `DiffusionError::ControlNet` naming the weights path — no silent
  uncontrolled renders. `parse_control_type` wraps `str_to_sd_control_type`
  (engine name table incl. aliases) exposed with test-visible getters.
- sd-core staging refuses a weights path that does not resolve to a file
  (the engine would log-and-render-unconditioned) and threads the named
  type; `control_type_from_name` keeps validation on the vendor's table.
- HTTP: `ImagesGenerationRequest::control_type` (engine names/aliases);
  unknown name → 400 `unknown_control_type`; stray `control_type` without a
  pair → 400 `controlnet_pair`; all prior 400s byte-identical; stub-pipeline
  501 `controlnet_unsupported` preserved; `/v1/capabilities` comment updated
  (execution now real; flag logic unchanged).

## Acceptance detail

CapturingSd contract test (`controlnet_type_execution_contract`) is written
to prove a cnet-named request reaches the pipeline with a NON-zero control
type and the weights path; it executes once the tree is green (the jobs
test target currently aborts on the binary-formats wave's in-flight
`formats.rs`/`svg.rs` errors — none in this lane's files). The seam-level
proof that RUNS today: the sd-core staging test asserts the named type on
the vendor config object (`Config::control_type()` getter) beside the
weights path and conditioning-image path. A missing weight returns a named
error, never a 501; structural parity checks stay green.
