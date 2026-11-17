# TASK: Advertise format capabilities

Status: proposed
Priority: P1
Epic: epic-binary-formats
Owner: unassigned
Acceptance: `/v1/capabilities` lists readable/writable formats and size limits derived from compiled decoders and configured tools.
Verification: `bun run codegen:check`, `bun run test:server`
Links: `.tmp/research/ReviewedBinaryFormats.md`, `FEAT-capabilities-api.md`
Resources: per-test config/tool stubs; no shared catalog

## Change

- Add `formats: { read: [], write: [], limits: {...} }` to the capabilities DTO.
- Derive from compile features + registry + configured external tools; do not
  hard-code a prose list.
- Add a flag/format agreement test against the registry.

## Acceptance detail

A client can choose an output format without trial requests; a missing decoder
never appears in `write`.
