# TASK: Docs site decision

Status: postponed
Priority: P3
Epic: epic-docs-tooling
Owner: unassigned
Acceptance: a recorded decision on docs site tooling, made only when there is content to publish.
Verification: none now; after a future change run `bun run check`
Links: `.tmp/research/ReviewedDocs-Organization.md`
Resources: none (decision ticket)

## Decision inputs

- Vite `^8.3.8` already exists for `packages/web-ui`, but it provides no docs
  routing/theme/search.
- VitePress is a separate MIT package requiring Node.js 22+; honkbox declares
  Bun >=1.4.4. Adding it is a new dep and a new runtime assumption.
- Plain in-repo Markdown with no site remains the zero-cost option.

## Acceptance detail

No dependency added by this ticket; record the chosen option and rationale in
`docs/README.md` before any site buildout.
