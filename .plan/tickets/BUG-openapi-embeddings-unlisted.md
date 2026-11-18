# BUG: `/v1/embeddings` missing from the served OpenAPI document

Status: done
Priority: P1
Epic: epic-capabilities-contract
Owner: honkbox-openapi-coverage
Acceptance: every mounted route appears in the generated OpenAPI document served at `GET /v1/openapi.json`, and a route that drops out of the derive's `paths(...)` list fails the codegen gate instead of silently shrinking the reference.
Verification: `bun run codegen:check`, `bun run check`, `bun run lint:rs`
Links: `crates/server/src/routes.rs` (`#[openapi(paths(...))]`, `post_embeddings`), `scripts/codegen-openapi-check.ts`, `packages/web-ui/src/gen/openapi.json`
Resources: no new runtime resources; the gate reads `crates/server/src/routes.rs` and the generated document

## Problem

`POST /v1/embeddings` was mounted (the `/v1/embeddings` route in `routes::app`)
and carried a `#[utoipa::path]` annotation on `post_embeddings`, but it was
never listed in the derive's `#[openapi(paths(...))]` list. The generated
document therefore omitted the endpoint entirely: the served API reference
(`GET /v1/openapi.json`) and the checked-in
`packages/web-ui/src/gen/openapi.json` both lacked a route that the server
actually serves.

`codegen:check` could not see this class of defect: it proves the committed
file matches the generator output, and an unlisted route regenerates
identically every time.

## Change

- `crates/server/src/routes.rs`: added `post_embeddings` to the
  `#[openapi(paths(...))]` list. The document now carries the endpoint with
  its request body, `EmbeddingsRequest`/`EmbeddingsResponse` schemas, and its
  400/404/429/500 responses.
- `scripts/codegen-openapi-check.ts`: after generating, the script compares the
  mounted route table (`.route("<path>"` in `routes.rs`) against the paths in
  the generated document and fails with the missing paths named, pointing at
  the `paths(...)` list.
- `packages/web-ui/src/gen/openapi.json`: regenerated (never hand-edited).

## Verification

- `bun run codegen:check` — reports `openapi covers all 33 mounted routes`
  and the committed file matches the generator output.
- Negative control (`.tmp/intercom/scripts/codegen-coverage-control.sh`):
  unlisting `post_embeddings` from the derive makes the gate fail with
  `1 mounted route(s) missing from the OpenAPI document: /v1/embeddings`.
- `bun run check` exit 0; `bun run lint:rs` exit 0; `bun run test` exit 0.
