# TASK: API version bump policy

Status: done
Priority: P1
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: a documented rule for when `api_version` changes, kept separate from crate `version`.
Verification: `bun run spdx`, documentation review (no code gate)
Links: `.tmp/research/ReviewedSupport-Capabilities.md` rule 1
Resources: none (documentation ticket)

> Done 2026-11-18 (honkbox-api-version): all 3 change items landed. The rule
> lives on `API_VERSION` + the `Capabilities` schema description in
> `crates/api-types/src/lib.rs` (flows into the served `OpenAPI` description,
> so clients discover it; bump on DTO field removals/type changes, error-code
> changes, or advertised-flag semantics; additive optional fields ride the
> existing `codegen:check` path; crate `version` tracks releases, not the
> wire contract). The `docs/README.md` mirror ALSO landed here — the
> precommit-hooks lane closed (`9e3b3c8`) before this ticket finished, so
> Main assigned the docs section to this lane (see `docs/README.md`, "API
> version policy"). `min_client_api_version` deliberately ABSENT: rationale
> recorded on `API_VERSION` + README — the server serves exactly one
> contract and additive fields are feature-detected by presence, so a min
> field would duplicate `api_version` until the first breaking bump;
> introduce only if a bump ships with a real compatibility window. New test
> `version_policy_is_discoverable_from_the_contract` (capabilities.rs) pins
> both the OpenAPI schema description and the README section
> (`include_str!` fails the build if the section is deleted).
>
> Verification 2026-11-18: `bun run lint:rs` exit 0; `bun run test:server`
> exit 0 (22 suites, 0 failed); `bun run codegen:check` green (api-types
> 15/15, `openapi.json` regen byte-identical, 33 routes covered);
> `bun run spdx` compliant (227/227 files); explicit run:
> `version_policy_is_discoverable_from_the_contract` +
> `api_version_is_the_wire_contract` — 2 passed, 0 failed.

## Change

- Record in `docs/README.md`: bump `api_version` on DTO field changes, error
  code changes, or advertised-flag semantics changes; additive optional fields
  may use the existing `codegen:check` compatibility path.
- State that crate `version` tracks releases, not the wire contract.
- Add the rule to the capabilities DTO description so clients see it.

## Acceptance detail

No dependency or hook change; the policy is discoverable from both docs and
the generated OpenAPI description.
