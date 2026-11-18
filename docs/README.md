# honkbox documentation

Research: `.tmp/research/ReviewedDocs-Organization.md`. No docs site, new
dependencies, or CI gates are added by this scaffold.

## Authority

`AGENTS.md` > `.plan/` > `src/` > `docs/`. Current behavior is what `src/` does.
Anything here that disagrees with `AGENTS.md` or `src/` is wrong or aspirational
and must be marked as such.

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

## Planned structure (create directories when content exists)

| Directory | Content |
| --- | --- |
| `spec/` | Design decisions and contracts; drift banner required |
| `guide/` | Operator/developer how-to: data dir, models, CLI, routes |
| `reference/` | CLI/API/config reference; link generated OpenAPI, never copy it |
| `research/` | Promoted `.tmp/research/Reviewed*.md` with an owner and consumer |
| `meta/` | Reviews, ADRs, retrospectives |

`.plan/` owns task state; `docs/` explains. Cross-link by epic/ticket ID instead
of duplicating status. Do not copy `packages/web-ui/src/gen/`, `NOTICE`, or
OpenAPI into prose.

## Current sources

- Repository rules: `AGENTS.md`
- Work items: `.plan/README.md` and its tickets/epics
- Canonical design research: `.tmp/research/` (gitignored scratch; promote on
  stability, not on length)
- Rust API docs: `bun run doc:rs`

## Tooling note

Vite is already a devDependency for `packages/web-ui`, but it does not provide
docs routing/theme/search. VitePress is a separate MIT package requiring Node
22+; adopting it is a deliberate future decision. dprint currently handles TS
and JSON only — Markdown formatting/link checks are not gated yet.
