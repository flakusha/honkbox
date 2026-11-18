# TASK: Determinism contract

Status: in-progress — determinism core LANDED 2026-11-18 (see Landed/Pending below); provenance wiring pending
Priority: P0
Epic: epic-full-ai-cycle
Owner: honkbox-determinism (agent), 2026-11-18
Acceptance: published reproducibility levels and canonical hash keys; bundle metadata records seed, model hash, engine version, and all sampling/parser params.
Verification: `cargo test --locked -p honkbox-server`, `cargo test --locked -p honkbox-sd-core --test parity`, `bun run check`
Links: `.tmp/research/ReviewedFullCycle-Determinism.md`
Resources: per-test temp bundles; no shared hash cache

## Change

- Define levels L0 (stable IDs/cache keys) through L3 (cross-hardware equality,
  explicitly unsupported); document that deterministic kernels may be slower.
- Canonical JSON (sorted keys, no ULID/timestamp/queue number) + BLAKE3/SHA-256
  over inputs + model revision + engine version + code version.
- Named PRNG streams derived from the root seed by stage; stable topological
  order and RRF tie-break by chunk ID.
- Extend `jobs.rs` metadata with model hash, sampler/scheduler, steps, cfg,
  dimensions, recipe, hires params, LoRA hashes, addon/engine versions.

## Acceptance detail

Same request twice produces identical bundle inputs on the stub backend;
hashing is stable across map insertion order; cross-backend parity stays
structural (no byte-equality assertion).

## Landed (2026-11-18, agent honkbox-determinism)

- `crates/server/src/determinism.rs` (NEW): L0-L3 reproducibility levels
  as module docs — L3 (cross-hardware/cross-backend byte equality)
  explicitly UNSUPPORTED, no byte-equality assertion may exist anywhere;
  deterministic-kernels-may-be-slower note. Canonical JSON (compact,
  UTF-8-byte-sorted keys, `VOLATILE_KEYS` dropped: ULID/job/request ids,
  wall-clock keys, queue positions) + SHA-256 canonical hash (`sha2`,
  already a direct workspace dep — no new hash dep, no manifest/lock
  touch). `bundle_record`/`bundle_record_hash`: the canonical input
  record (model, seed, sampler/scheduler, steps, cfg, dimensions,
  negative prompt, strength, recipe, hires pass, LoRA entries +
  `record_version`/`code_version`/`backend_id`), version-scoped by
  design. Named PRNG stage streams: `stage_stream(root_seed, stage)` via
  documented KDF `SHA-256("honkbox/prng/v1" || seed_le || stage)`,
  counter-mode draws (`next_u64`, exact-representable `next_f32`),
  `child(label)` sub-streams. `sort_score_desc_id_asc`: RRF-stable
  ordering (score desc, chunk id asc on ties). Pure functions, no
  retrofit of existing pipelines.
- `crates/server/src/jobs.rs`: module registration (`#[path] pub mod
  determinism` — main.rs was integration-owned) + `inputs_sha256`
  appended to `request.json` and `final_00.json` (append-only, 4 hunks).
- `crates/server/tests/determinism.rs` (NEW): harness includes ONLY
  src/jobs.rs, so it runs independently of sibling lanes' compile state.
  18 tests: byte-identical bundle inputs for the same stub request;
  insertion-order-stable hashing (two textual key orders + two Map
  insertion orders); sorted/compact/volatile-free canonical form;
  version-pinned record + input-change sensitivity; full-request
  (loras/hires/recipe) key equality; stage-stream determinism/children;
  RRF tie-break order; empty-input SHA-256 reference pin.

Gates observed: `cargo test --locked -p honkbox-server --test
  determinism` 18/18 GREEN; `cargo clippy --locked -p honkbox-server
  --test determinism -- -D warnings` 0 errors; `cargo check --locked -p
  honkbox-server --bin honkbox` 0 errors.

## Pending

- Provenance wiring: `model_sha256` + `engine_version` are reserved
  (null) keys in `bundle_record`; wire them at the BundleInputs
  construction sites — routes.rs txt2img (~:3653), routes.rs img2img
  (~:4152), main.rs (~:708) — from the download records (HF LFS publishes
  sha256; `downloads.rs` stores it in the `.part.meta.json` sidecar) and
  the vendor engine version. The record schema is fixed; wiring only
  fills the nulls.
- Bundle records for non-image kinds (LLM/audio streams, embeddings) on
  the same `bundle_record` shape.
- Workspace gate state at landing: `bun run test:server` GREEN (all
  suites pass) after this lane declared the missing `svg`/`ora`/`raster`/
  `container` mods in the two route-harnesses the svg-ora wave missed
  (tests/validator_properties.rs, tests/swap.rs — mechanical, additive).
  `bun run lint:rs` still fails on 15 pedantic lints, ALL in sibling
  files: container.rs (unreadable_literal x2, struct_field_names,
  too_many_lines, needless_pass_by_value, collapsible_if), ora.rs
  (unnested_or_patterns, doc_markdown, struct_field_names,
  too_many_lines, cast_sign_loss), svg.rs (struct_field_names,
  redundant_closure_for_method_calls, chunks_exact_to_as_chunks),
  routes.rs (too_many_lines). Zero lint errors in determinism.rs,
  jobs.rs, tests/determinism.rs (scoped clippy -D warnings clean).
  Those fixes belong to the container-safety / svg-ora lanes.
- `bun run codegen:check` not run: this lane touched no api-types or
  codegen surface (a dirty packages/web-ui/src/gen/api-types.ts in the
  tree predates/postdates sibling work — not this lane's artifact).

Note: the linked research doc
`.tmp/research/ReviewedFullCycle-Determinism.md` is absent from the tree
(gitignored scratch, cleaned between waves); the ticket text plus the
determinism module docs are now the authoritative L0-L3 definition.
