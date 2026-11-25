# honkbox documentation

Plain `.md` in this directory is the canonical published form — no docs site,
no new dependencies, no CI gates beyond the link check. See
[`meta/docs-shipping.md`](meta/docs-shipping.md) for the shipping review
(plain `.md` vs GitHub Pages vs VitePress).

## Authority

`AGENTS.md` > current repository code > README > epics > tickets > docs.
Here, README means `.plan/README.md`. Current behavior is what the code does;
ticket status is a claim to verify. Anything here that disagrees with a
higher-authority source is wrong or aspirational and must be marked as such.

## Contents

| Path | Content |
| --- | --- |
| `guide/configuration.md` | Data dir, config file, model roots, env overrides, auth, logs, registry cache |
| `guide/operations.md` | Jobs, cancellation, SSE events, deployment topology, ComfyUI read surface |
| `guide/media-capabilities.md` | ControlNet, vision chat (mmproj), audio formats |
| `guide/content-provenance.md` | Embedded image records and unsigned C2PA-shaped job manifests |
| `guide/migrating.md` | Moving state from sd-rip / `~/.honkbox` layouts |
| `meta/docs-shipping.md` | Shipping/generation review: plain `.md` vs Pages vs VitePress |

Planned, created when content exists: `spec/` (decisions and contracts),
`reference/` (CLI/API/config reference — link generated OpenAPI, never copy
it), `research/` (promoted `.tmp/research/Reviewed*.md` with an owner and
consumer).

`.plan/` owns task state; `docs/` explains. Cross-link by epic/ticket ID instead
of duplicating status. Do not copy `packages/web-ui/src/gen/`, `NOTICE`, or
OpenAPI into prose.

## API version policy

`api_version` on `GET /v1/capabilities` is the **wire-contract version**
(`"v1"`), deliberately independent of the crate `version` (which tracks
releases, not the wire shape):

- **Bump `api_version` only for an incompatible wire change**: a DTO field
  removal or type change, an error-code change, or a semantic change to an
  advertised capability flag.
- **Purely additive optional fields land under the existing version** (serde
  `skip_serializing_if` keeps them absent on older servers); the
  `codegen:check` idempotence gate keeps the generated schema honest.
- There is deliberately **no `min_client_api_version`**: the server serves
  exactly one contract and additive fields are feature-detected by presence,
  so a min field would duplicate `api_version` until the first breaking bump.
  Introduce it only if a bump ever ships with a real compatibility window.

Clients discover the rule machine-readably: it is embedded in the
`Capabilities` schema description (`crates/api-types/src/lib.rs`), which
flows into the served `GET /v1/openapi.json`.

## Current sources

- Repository rules: `AGENTS.md`
- Work items: `.plan/README.md` and its tickets/epics
- Canonical design research: `.tmp/research/` (gitignored scratch; promote on
  stability, not on length)
- Rust API docs: `bun run doc:rs`
- Generated TS reference: `bun run docs:api` (gated by `bun run docs:api:check`)

## Tooling

`bun run docs:check` (`scripts/docs-links.ts`) verifies relative file targets in
`README.md` + `docs/**/*.md`; it runs inside `bun run check` and the CI `Rustdoc`
job. Anchor fragments, external URLs, and links inside `.plan/` are outside its
scope. dprint handles TS and JSON only — Markdown formatting is not gated.
