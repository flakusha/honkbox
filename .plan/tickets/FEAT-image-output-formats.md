# FEAT: JPEG/WebP image output over HTTP

Status: done
Priority: P1
Epic: epic-output-formats
Owner: unassigned
Acceptance: `/v1/images/generations` and `/v1/images/edits` accept `output_format: png|jpeg|webp` and emit matching bytes/sidecars; optional `output_compression` quality knob.
Verification: done — `crates/server/tests/images.rs`: `jpeg_output_format_returns_jpeg_bytes`, `webp_output_format_returns_webp_bytes` (lossless), `bundle_sidecar_agrees_with_output_format`, `unknown_output_format_is_400`, `output_compression_semantics`, `default_output_format_unchanged_png`; `crates/server/tests/edits.rs`: `jpeg_output_format_and_bundle_agree`, `output_format_and_compression_parity`; inline `crates/sd-core/src/lib.rs`: `output_format_naming_and_mime_agree`, `jpeg_quality_knob_changes_bytes_and_default_matches_90`.
Links: `.tmp/research/ReviewedOutput-Formats.md` §1
Resources: per-test temp job dirs via runner tmp facility; no shared output paths

## Change

- Add `output_format` to `ImagesGenerationRequest` (default `png`); keep
  `response_format: b64_json` unchanged for wire compatibility.
- Thread through `generate_image_bytes` and both `BundleInputs` literals;
  `jobs.rs` already names `final_00.{png,jpg,webp}`.
- Quality knob follows OpenAI's `output_compression` precedent; JPEG q90 and
  lossless WebP are today's hardcoded defaults. Lossy WebP needs a new encoder
  dep — separate ticket, not this one.

## Acceptance detail

A generated JPEG response decodes as JPEG; a WebP response decodes losslessly;
bundle sidecar extension and MIME agree with the requested format.
