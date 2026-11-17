# TASK: Material tool capability probe

Status: proposed
Priority: P2
Epic: epic-vector-materials
Owner: unassigned
Acceptance: installed material tools (Material Maker, ComfyUI nodes, operator-licensed `sbsrender`) report versions/capabilities and missing tools yield named 501s.
Verification: `bun run test:server`, `bun run codegen:check`
Links: `.tmp/research/ReviewedCreative-TexturesMaterials.md` §Substance SAT
Resources: probe-only; no shared tool process or fixed socket

## Change

- Extend the external tool registry with a `material` capability kind.
- Probe headless versions and supported export formats once; cache with a
  bounded TTL and explicit refresh.
- Never parse `.sbs`/`.sbsar` natively; invoke `sbsrender` as an external process
  only if the operator supplies a licensed binary.

## Acceptance detail

Capabilities are machine-readable through `/v1/capabilities`; a missing licensed
tool is reported without disabling unrelated features.
