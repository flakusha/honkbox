# TASK: Tool-choice fidelity contract

Status: done
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: `tool_choice: required|any|<name>` clamping and ignored `parallel_tool_calls` are machine-readable instead of silent.
Verification: `bun run codegen:check`, `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedSupport-Capabilities.md` §Current state; `api-types/src/lib.rs:456-486`
Resources: per-test stub provider instances; no shared tool registry

## Done (2026-11-18, commit 6e1aeb2)

`ToolCallingCapability` DTO published as `Capabilities.tools` (honored vocabulary
`auto|none`, clamped `required|any|<name>` → auto, named-function clamp,
`parallel_tool_calls: false` + rationale). Mapping derives from the
`HONORED_MODES`/`CLAMPED_MODES` consts via `ToolCallingCapability::chat_routes()` —
no literals in routes.rs. Request docs disclose the clamp machine-readably
(accepted, never a 400); OpenAPI + TS gen regenerated. `tool_choice: none` and
GBNF-enforced structured output unchanged; clamp consult intact at
`crates/server/src/routes.rs:2481` (`ToolChoiceSpec::suppresses_tools`), derivation
at `routes.rs:754` via `crates/api-types/src/lib.rs:745-758` (consts at 704/707,
struct at 727).

## Acceptance detail — verified

No request is silently downgraded: either the route honors the choice or the
contract says it does not before the client sends it.

- `capabilities::capabilities_disclose_tool_choice_fidelity` (DTO carries both
  vocabularies before a request is sent), `completions::tool_choice_none_and_clamps`
  (none honored; required/any/named clamped to auto — behavior matches the
  disclosed contract), `completions::parallel_tool_calls_ignored` (ignored, not
  rejected).
- Gate evidence (2026-11-18, re-run independently by the tool lane on the merged
  tree): `bun run lint:rs` exit 0; `bun run test:server` exit 0, 0 failed;
  `cargo fmt --all -- --check` exit 0; `bun run codegen:check`: api-types 15/15,
  regen idempotent — pre-commit exit 1 was the uncommitted gen diff only
  (documented signature), resolved by commit 6e1aeb2; post-commit run exits 0.

## Change

- Minimum: document the clamp in DTO/OpenAPI descriptions and include a
  fidelity note in `/v1/capabilities` (`TASK-api-version-policy.md` semantics).
- Optional follow-up if a named consumer needs it: implement exact tool choice
  and streamed parallel fragments against the model template.
- Keep `tool_choice: none` and GBNF-enforced structured output unchanged.

## Acceptance detail

No request is silently downgraded: either the route honors the choice or the
contract says it does not before the client sends it.
