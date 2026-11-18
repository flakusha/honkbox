# TASK: Advertise format capabilities

Status: done (2026-11-18, honkbox-capabilities)
Priority: P1
Epic: epic-binary-formats
Owner: unassigned
Acceptance: `/v1/capabilities` lists readable/writable formats and size limits derived from compiled decoders and configured tools.
Verification: `bun run codegen:check`, `bun run test:server`
Links: `.tmp/research/ReviewedBinaryFormats.md`, `FEAT-capabilities-api.md`
Resources: per-test config/tool stubs; no shared catalog

> Done 2026-11-18 (honkbox-capabilities): `formats.{read,write,limits}`
> derived, not prose — `write` from `honkbox_sd_core::OutputFormat::ALL`
> (+ new `compression_applies()`: JPEG-only knob, PNG deterministic, WebP
> lossless-only), `read` = the PNG magic-sniff the ingest validators really
> run (edits `image`/`mask`, `control_image`; vision pass-through bytes
> deliberately unlisted), `limits` = JPEG quality 0-100 range + the router
> body ceiling (`AUDIO_BODY_LIMIT_BYTES`, exposed `pub(crate)` so the test
> cross-checks doc vs code). Agreement test round-trips EVERY advertised
> write format through `/v1/images/generations` and asserts the decoded
> container magic — a missing encoder can never appear in `write`.

## Change

- Add `formats: { read: [], write: [], limits: {...} }` to the capabilities DTO.
- Derive from compile features + registry + configured external tools; do not
  hard-code a prose list.
- Add a flag/format agreement test against the registry.

## Acceptance detail

A client can choose an output format without trial requests; a missing decoder
never appears in `write`.
