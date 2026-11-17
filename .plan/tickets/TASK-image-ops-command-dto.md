# TASK: Image-ops command DTO

Status: proposed
Priority: P1
Epic: epic-editor-integration
Owner: unassigned
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
