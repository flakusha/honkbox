# TASK: Federated provider assessment (mesh lane)

Status: proposed
Priority: P3
Epic: epic-mesh-integration
Owner: mesh lane
Acceptance: a recorded decision on whether a mesh-side plugin should register honkbox as an OpenAI-compatible inference provider; no honkbox runtime change without a named consumer.
Verification: documentation decision review; no honkbox code gate
Links: `.tmp/research/ReviewedMesh-FederatedProvider.md`
Resources: none (cross-app assessment)

## Change

- Mesh plugin manifest would declare `openai_http(http://honkbox:8080/v1)` in
  its `inference` section; mesh owns the data plane, plugin owns lifecycle.
- Virtual-model ids map to honkbox slugs; endpoint health is separate from
  plugin health; private scopes still fail closed.
- Default verdict: **non-feature for honkbox**; honkbox-side mesh client tickets
  are the active path.

## Acceptance detail

Decision is recorded in the mesh lane; honkbox gains no plugin host, SDK, or ABI.
