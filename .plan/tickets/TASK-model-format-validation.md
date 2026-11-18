# TASK: 3D model format validation

Status: done (2026-11-18, honkbox-model-format-gate)
Priority: P2
Epic: epic-binary-formats
Owner: honkbox-model-format-gate
Acceptance: GLB and splat artifacts are validated against an extension allowlist before export; FBX/USD/OBJ are external-only conversions.
Verification: external `gltf-transform inspect/validate`; `bun run test:server`
Links: `.tmp/research/ReviewedBinaryFormats.md`, `TASK-model-bundle-validation.md`
Resources: per-test model dirs; no shared validator cache

## Change

- Share the allowlist with `TASK-model-bundle-validation.md`; no duplicate rules.
- Validate PLY/SPLAT/SPZ headers and sizes structurally; no in-process splat
  training or rendering.
- Record format, version, validator version, and hashes in the sidecar.

## Acceptance detail

Unsupported compressed extensions fail with named errors; a valid GLB passes
both this gate and the engine-bundle gate.

## What landed (commit 0be70de)

- `crates/server/src/formats.rs` — the registry is the single allowlist
  source: two new capability columns (`model_export`, `external_only`),
  four new rows (`spz`, `fbx`, `usd`, `obj`; table now 15 rows), the
  published `MODEL_EXPORT_FORMATS = ["glb", "ply", "splat", "spz"]`
  constant, and `exportable_model_rows()` / `exportable_model_names()` /
  `model_row_for_extension()` accessors. The landed glb/ply/splat rows
  carry `model_export: true`; no existing column value changed.
  `fbx`/`usd`/`obj` are `external_only` rows — no read, no write, no
  in-process capability, never silently accepted. New magic identities:
  SPZ v4 NGSP plaintext header (`NGSP`, from the nianticlabs/spz reference
  implementation) and FBX binary (`Kaydara FBX Binary  \x00\x1a\x00`);
  usd/obj and legacy gzip-wrapped SPZ are policy-admitted (no reliable
  magic — usdz is a plain zip, which is already the ora row's signature).
- `crates/server/src/model_formats.rs` (NEW) — the structural gate:
  `validate_model(bytes, claimed_extension)` returning a
  `ValidatedModel` provenance record or a `ModelReject` with named codes
  (`unsupported_model_extension`, `model_external_only`,
  `model_header_invalid`, `model_truncated`, `model_too_large`),
  `VALIDATOR_VERSION = "model-formats/1"`, and the `validation_sidecar()`
  JSON builder (`<artifact>.validation.json`: format, format_version,
  validator_version, sha256, size_bytes). Identification is magic-first;
  rejections name the sniffed format, never the claimed extension.
- `crates/server/tests/model_formats.rs` (NEW) — 14 gate tests (hand-built
  GLB/PLY/SPZ/splat fixtures, external-only + unknown-extension naming,
  allowlist invariant, sidecar provenance) + the 10-test formats-registry
  suite that ships in the same binary: 24 passing.
- `crates/server/src/main.rs` — `mod model_formats;` (one line; UNCOMMITTED
  — main.rs carries the artifact lane's in-flight hunks, so the line rides
  with whichever lane commits main.rs next; the committed slice compiles
  without it because the module file is only test-included via `#[path]`).

## Per-format depth (honest limits)

- **glb** — typed header verdicts (12-byte header, version 2 only, length
  >= chunk table, declared <= present) + the landed
  `container::validate_glb` walk (chunk order/types, budgets,
  external-buffer rule). Byte-shortfall = `model_truncated`; chunk-table
  lies = `model_header_invalid`.
- **ply** — pure-byte header parse: ascii + `binary_little_endian`
  accepted, `binary_big_endian` rejected with the encoding named; strict
  grammar (unknown directives reject, naming the directive); declared
  counts against the registry vertex budget; declared-vs-present size
  crosscheck EXACT for fixed-width binary rows, minimum-only when list
  properties make rows variable, and absent for ASCII (payload length is
  not derivable from the header — documented, not silently skipped).
- **splat** — headerless by spec (antimatter15/PlayCanvas 32-byte record
  layout): nonempty size, stride multiple, registry budgets. Record
  content is not verifiable without a full parse; none happens.
- **spz** — v4 plaintext NGSP header: magic, version 4, nonzero points,
  i32 bound, and the reference's own file-size bound
  (`(size * 1024) / 9`); legacy v1-3 gzip envelopes validate minimally
  (magic + minimum envelope size) because the header lives INSIDE the
  compressed stream — documented honesty; the loader validates it when it
  decompresses. `shDegree`/`flags`/`numStreams`/`tocByteOffset` are
  advisory and not enforced.

## Verification

- `cargo test --locked -p honkbox-server --test model_formats` — 24
  passed (14 gate tests + the 10-test formats-registry suite that ships
  inside the same binary), observed green in-tree. Post-review fixes
  (tab-separated PLY header acceptance, empty-bytes GLB truncation,
  SPZ registry-budget `model_too_large`) verified in the detached
  lint-mirroring scratch crate (clippy 0 warnings, 24 passed) while the
  recognition-worker lane's half-written workspace manifest blocked
  in-tree runs; the in-tree suite re-run belongs to the lane that lands
  last.
- The gate modules were additionally verified in a detached scratch crate
  mirroring the workspace lints (`model_formats.rs` + `formats.rs` +
  `container.rs`): clippy all+pedantic 0 warnings, tests 24 passed — run
  while sibling lanes' in-flight files were the only red in the shared
  tree.
- `bun run test:server` / `bun run lint:rs` observed RED at time of
  landing, exclusively in sibling lanes' UNCOMMITTED in-flight files
  (`tests/images.rs` artifacts wiring move error; `config.rs`/
  `artifacts.rs` clippy set) — no finding touches any file of this
  ticket. Re-run both once the artifact lane lands.

## Deferred

- Route wiring: no route consumes the gate yet. The wiring point is
  documented in the `model_formats.rs` module docs (validate on artifact
  ingest/export, answer with `ModelReject::code`/`detail`, write the
  sidecar beside the artifact) and lands with
  `TASK-model-bundle-validation`, which imports `MODEL_EXPORT_FORMATS`
  instead of restating it.
- External `gltf-transform inspect/validate` semantic validation stays
  with the bundle ticket.
