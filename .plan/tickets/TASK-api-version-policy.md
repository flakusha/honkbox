# TASK: API version bump policy

Status: proposed
Priority: P1
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: a documented rule for when `api_version` changes, kept separate from crate `version`.
Verification: `bun run spdx`, documentation review (no code gate)
Links: `.tmp/research/ReviewedSupport-Capabilities.md` rule 1
Resources: none (documentation ticket)

## Change

- Record in `docs/README.md`: bump `api_version` on DTO field changes, error
  code changes, or advertised-flag semantics changes; additive optional fields
  may use the existing `codegen:check` compatibility path.
- State that crate `version` tracks releases, not the wire contract.
- Add the rule to the capabilities DTO description so clients see it.

## Acceptance detail

No dependency or hook change; the policy is discoverable from both docs and
the generated OpenAPI description.
