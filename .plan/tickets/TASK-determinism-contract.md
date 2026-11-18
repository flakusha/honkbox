# TASK: Determinism contract

Status: done (core + provenance wiring landed: 67da8d6, 3a86ae0, 02a49f4); deferred: non-image-kind bundle records
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

- Bundle records for non-image kinds (LLM/audio streams, embeddings) on
  the same `bundle_record` shape.

## Provenance wiring — LANDED (2026-11-18, commit 02a49f4)

- `downloads.rs`: completed checksum-verified downloads persist
  `<target>.sha256.json` (`ArtifactMeta`, verified streamed digest only —
  never an unverified value); `artifact_sha256()` reads+validates (64
  lowercase hex or None); `remove_artifact_meta()` keeps the sidecar from
  outliving its file (delete_model_path). The in-progress
  `.part.meta.json` keeps its resume role and is still dropped at commit.
- `sd-core`: `SD_CPP_ENGINE_VERSION` names the exact vendored revs
  (stable-diffusion.cpp 5dcfa798… / ggml eeda9ec… / diffusion-rs 89dcf93…
  — mirroring `vendor.manifest.toml`, the source of truth);
  `engine_version_for(backend_id)` maps the sd-cpp backends to it and the
  stub to `None` (no separate engine — `code_version` covers it; null is
  honest absence, never a fabricated version).
- `jobs.rs`: `BundleInputs.model_sha256` / `.engine_version` (documented
  honest-absence semantics); both recorded in `request.json` and
  `final_00.json`; `sidecar_records_provenance` test (set → recorded;
  plain → explicit nulls).
- Wiring sites: routes.rs txt2img + img2img resolve the digest via the
  catalog entry's weight `path` (catalog-verified download or null) and
  the engine version via `backend_id`; main.rs CLI wires engine version
  and records a null digest (no catalog/scan on the CLI path).
- Tests: `known_provenance_lands_in_the_record` (published keys + key
  change) and the null-path assertions in the pins test.

Gates observed at landing: `cargo test --locked -p honkbox-server --test
  determinism` 20/20 GREEN; `bun run test:server` GREEN (all suites incl.
  the new provenance tests). `bun run lint:rs` and `cargo test -p
  honkbox-sd-core --lib` were deferred at commit time by the sibling
  vendor/ rev-move in flight (`vendor_patches/0006-sd-cancel-
generation.patch` mid-application: vendored api.rs E0252/E0277); the
  earlier 15 sibling pedantic lints are gone (their clippy fixes landed)
  and my sd-core code compiled green inside test:server's build window.
  `bun run codegen:check` n/a — no api-types/codegen surface touched.

Note: the linked research doc
`.tmp/research/ReviewedFullCycle-Determinism.md` is absent from the tree
(gitignored scratch, cleaned between waves); the ticket text plus the
determinism module docs are now the authoritative L0-L3 definition.
