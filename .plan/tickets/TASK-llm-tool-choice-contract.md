# TASK: Tool-choice fidelity contract

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: `tool_choice: required|any|<name>` clamping and ignored `parallel_tool_calls` are machine-readable instead of silent.
Verification: `bun run codegen:check`, `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedSupport-Capabilities.md` §Current state; `api-types/src/lib.rs:456-486`
Resources: per-test stub provider instances; no shared tool registry

## Change

- Minimum: document the clamp in DTO/OpenAPI descriptions and include a
  fidelity note in `/v1/capabilities` (`TASK-api-version-policy.md` semantics).
- Optional follow-up if a named consumer needs it: implement exact tool choice
  and streamed parallel fragments against the model template.
- Keep `tool_choice: none` and GBNF-enforced structured output unchanged.

## Acceptance detail

No request is silently downgraded: either the route honors the choice or the
contract says it does not before the client sends it.
