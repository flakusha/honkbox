# TASK: Image-ops command DTO

Status: done (2026-11-18, LANE-impl-wave)
Priority: P1
Epic: epic-editor-integration
Owner: LANE-impl-wave
Acceptance: a bounded command DTO covers `op`, inputs, params, output format, client identity, and idempotency key with reject-not-clamp validation.
Verification: `bun run codegen:check`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedImageEdit-CommandProtocol.md`
Resources: none (DTO ticket)

## Change

- Add `ImageOpRequest` + `ImageOpKind` to `crates/api-types`; regenerate TS/OpenAPI.
- Allow only data-URI or artifact-id inputs; no paths, `file://`, or URLs.
- Bound seed, strength, scale, layer count, batch size, and output formats.
- `request_id` deduplicates retries.

## Acceptance detail

Unknown fields and out-of-range values return named 400s; default values never
silently clamp an explicit request.

## Implementation note (2026-11-18)

- DTO family in api-types: `ImageOpKind` (closed 5-op vocabulary: resize,
  crop, rotate, grayscale, format; `ALL` drives the capabilities list),
  `ImageOpCommand` (externally tagged: param ops as single-key objects,
  no-param ops as bare strings; param structs `deny_unknown_fields`),
  `ImageOpInput` (`data_uri` | `artifact` — artifact = bundle ULID + index;
  no paths/`file://`/URLs anywhere, nothing is fetched), `ImageOpRequest`
  (`deny_unknown_fields`; `request_id` required, 1..=128 chars; `client`
  optional 1..=128, sidecar-recorded identity; `output_format` png/jpeg/webp;
  `output_compression` JPEG 0-100), plus `ImageOpResize`/`Crop`/`Rotate`,
  `ImageOpArtifact`, `ImageOpOutput`, `ImageOpResponse`, `ImageOpsCapability`
  (additive `Capabilities.image_ops`). export_ts + ApiDoc registered.
- Bounds (reject-not-clamp, named 400 codes `invalid_image_op_request`,
  `invalid_image_op_request_id`, `invalid_image_op_inputs`,
  `invalid_image_op_params`, `invalid_crop_rect`,
  `unsupported_output_format`, `invalid_output_compression`,
  `image_op_request_id_conflict`): resize edges [1,8192] with the
  16_777_216-px target cap; crop rects ≥1px (image-bounds checked against
  the real pixels); rotate only 90/180/270; batch 1..=8; decoded inputs
  bounded by strict decoder caps (16384 edges, 128 MiB alloc) plus the pixel
  cap. Seed/strength/scale/layer count do not exist in the op vocabulary —
  there is nothing to bound beyond the batch; the ticket's bound list maps
  onto batch size + output formats + pixel/edge caps.
- `request_id` dedup: process-global bounded ledger (≤128 entries, ≤64 MiB
  retained, oldest-first eviction) keyed by (request_id, sha256 over the
  re-serialized canonical request). Same key + same command replays the
  first response byte-identically (single-flight `OnceCell`, no double
  execution even under a concurrent race); same key + different command is
  a named 400 conflict; failures are forgotten so genuine retries re-execute.
- codegen: api-types.ts + openapi.json regenerated (convergence observed via
  codegen:check; the final git-diff step needs the gen files committed —
  no-commit lane, so convergence-of-regeneration is the honest result).
