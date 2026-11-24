# Docs shipping review: plain .md vs GitHub Pages vs VitePress (2026-11-24)

Decision: **plain `.md` in-repo is canonical**; generated API references stay
generated and linked, never copied. Revisit a site only when docs outgrow
single-page reading or need search.

## Views compared

| View | Cost | What it gives over plain .md | Verdict |
| --- | --- | --- | --- |
| Plain `.md` in `docs/` (GitHub renders) | Zero deps, bun-only policy intact | Baseline | **Canonical now** |
| GitHub Pages, no-build (Jekyll/`_config.yml`, same `.md`) | One workflow file + Pages switch; no new package | Public URL for the same files | Cheapest upgrade when a remote exists; no repo — no workflow yet |
| TypeDoc HTML + rustdoc (already built) | Already paid (`docs:api`, `doc:rs`) | Searchable API reference | Keep as the *reference* view; link from `docs/`, never duplicate into prose |
| VitePress | New MIT package, **Node 22+** (repo is bun-only, Bun >= 1.4.4), theme/routing/search config, build+deploy workflow | Theme, sidebar, full-text search | **Deferred**: needs Node 22 alongside bun, a second toolchain for what is currently 5 pages |

## Why not VitePress yet

- Toolchain: VitePress requires Node 22+; the repo standardizes on Bun and
  documents one runtime. Adopting VitePress means every contributor and CI job
  carries two JS runtimes for docs alone.
- Scale: `docs/` is five prose pages plus generated references. GitHub's own
  `.md` rendering plus TypeDoc search covers reading and lookup.
- Trigger to revisit: a remote with Pages enabled plus either >15 prose pages
  or repeated "can't find X" complaints — then prefer the no-build Pages path
  first, VitePress only if theming/search is the actual gap.

## Generation support in this branch

- `bun run docs:check` — relative-link gate over `README.md` + `docs/**/*.md`
  (`scripts/docs-links.ts`): every relative link target must exist. Wired into
  `bun run check`.
- `bun run docs:api:check` (existing) — generated TypeDoc HTML exists where the
  server serves it and carries source doc comments.
- `bun run codegen:check` (existing) — `packages/web-ui/src/gen/` fresh vs
  `api-types`; the machine-readable API reference.
- Anchor (`#section`) checking is deliberately out: headings drift faster than
  files, and a failing gate on every heading rename trains contributors to
  ignore the gate. Revisit on evidence.
