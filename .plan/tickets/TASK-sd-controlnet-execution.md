# TASK: Execute ControlNet requests

Status: done
Owner: honkbox-controlnet-exec
Priority: P0
Epic: epic-sd-native-backend
Acceptance: a valid ControlNet request runs through the vendored engine instead of returning 501 `controlnet_unsupported`.
Verification (all observed 2026-11-18, tree at `cf52aa2`+`3d95265`): `cargo test --locked -p honkbox-server --test jobs controlnet` — 5 passed, 76 filtered, INCLUDING the executed `controlnet_type_execution_contract` (non-zero control type + weights path reach the seam; alias, case-sensitivity, unknown-name, and stray-type branches); `bun run test:server` — 26 suites, 0 failed; `cargo test --locked -p honkbox-sd-core` — 22 passed, 1 ignored (opt-in parity); `bun run lint:rs` — exit 0 (workspace clippy, all targets, `-D warnings`); `bun run codegen:check` — exit 0 after the `openapi.json` regen (`3d95265`; the first run caught the stale document missing the `control_type` schema); `bun scripts/vendor-patches.ts sync` — converges (6 applied / 0 now, `--dry-run` "tree is converged"). Work landed as `cf52aa2` (orchestrator split the lane's hunks out of the shared files).
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

`controlnet_type_execution_contract` (crates/server/tests/jobs.rs) EXECUTED
AND PASSING: a cnet-named request reaches the pipeline with a NON-zero
control type (`sd_control_type_t::SD_CONTROL_TYPE_CANNY`) beside the
weights path; the engine alias (`lineart`) resolves onto the same type;
unknown and wrong-case names are 400 `unknown_control_type` pre-slot; a
stray type without a pair is 400 `controlnet_pair`. Engine-side, the
d-core staging test asserts the named type on the vendor config object
(`Config::control_type()` getter) beside the weights path, and that a
missing weights file is refused by name. A missing weight returns a named
error, never a 501; structural parity checks stay green.
