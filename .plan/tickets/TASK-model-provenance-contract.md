# TASK: model identity and provenance contract

Status: proposed
Priority: P1
Epic: epic-binary-formats
Owner: unassigned
Acceptance: every catalog model has an honest, persisted provenance verdict derived from local header/path/digest evidence and the existing license registry; ambiguous identity stays unknown; component/base relationships and territory/revenue policy results survive restart; no probe fetches weights or license text and no restricted weights are auto-fetched or bundled; existing server-side read/write/admin authorization stays fail-closed and provenance adds no direct DB or credential-bearing response path.
Verification: `bun run test:registry`, `cargo test --locked -p honkbox-registry`, `bun run test:server`, `bun run codegen:check`
Links: `.tmp/research/model-identity-license-download-2026-11-25.md` §§3–7, `TASK-model-license-registry.md`, `TASK-nc-weight-gate.md`, `TASK-sd-base-model-tags.md`, `TASK-download-options-contract.md`, `TASK-content-record-provenance.md`, `FEAT-capabilities-api.md`, `TASK-capabilities-contract-test.md`
Resources: tiny local GGUF/safetensors/JSON fixtures, per-test registry DB and model roots, immutable fixture license bytes; no network, live Hub token, shared registry or model cache

## Change

- Add one versioned provenance record per existing registry slug, persisted in
  `crates/registry` with a `SCHEMA_VERSION` bump using the existing
  rebuildable-cache migration convention. Re-probe bounded headers from live
  files; keep the prior verdict only as explicitly stale cache evidence when the
  file is absent. Do not add a second registry or policy engine.
- Record `probe_version`, `probed_at`, file size, optional verified `file_sha256`,
  provider, redacted source URL, repo id, full revision, source path and download
  id. A digest is present only when actually verified or supplied by trusted local
  metadata; never derive it from a name/header. Do not hash every large weight by
  default during a catalog scan.
- Resolve identity in this order: (1) local GGUF `general.name` + architecture or
  exact safetensors identity metadata; (2) curated registry match on kind, repo,
  full revision, path and digest when known; (3) canonical HF-cache
  `repo@full-sha/path` as corroboration or a low-confidence fallback. A path is a
  tie-break, not proof. For safetensors, inspect `__metadata__` identity keys
  (`general.name`, `model_name`, `model_id`) separately from the existing base
  keys (`base_model`, `ss_base_model`, `ss_base_model_version`,
  `modelspec.base_model`); preserve raw values. A base-model tag alone is only a
  family hint, not exact checkpoint identity.
- Persist `identity_status` (`identified` | `ambiguous` | `unknown`), `confidence`
  (`high` | `medium` | `low` | `unknown`), stable `reason_code`, and evidence
  sources. Header+path+digest agreement is high; exact header only or
  header+path without a digest is medium; path-only is low;
  missing/unreadable/conflicting/multiple evidence is unknown. Unknown raw
  base-model tags remain raw/unknown, never normalized by guess.
- Persist base-model and component edges: `derived_from`, `component_of`,
  `projector_for`, and `adapter_for`, with child slug/artifact id, role, required
  flag, revision/digest and license binding. Record Qwen encoders, mmproj, VAE,
  tokenizer/config and LoRA components without treating a permissive component as
  relicensing a restricted composite. An adapter with unknown base is ambiguous.
- Bind each code/weights/adapter/output scope to the existing registry's license
  identifier (`license`), `license_url`, source URL, weight SHA and attribution.
  Add `license_sha256` for the exact immutable license-document bytes only when
  those bytes are locally curated; otherwise null. License-document SHA and
  weight SHA are separate fields. Never fetch a license to fill a missing hash.
- Reuse `catalog::licenses` lookup and `catalog::license_gate` evaluation. Extend
  the existing record/config only as needed for structured revenue band and
  commercial-authorization declarations; the gate remains the sole policy
  implementation. Persist the gate/registry version, evaluated record hash,
  declared territory, commercial posture, revenue band, authorization result,
  final `allowed` | `refused` | `unknown`, and named reason.
- Model restricted-policy fixtures on primary terms: LTX-2.x paid authorization
  at **at least $10M** annual entity revenue except qualifying non-commercial use;
  MiniMax H3 authorization at **more than $20M** yearly commercial revenue and
  territory exclusion of EU/UK/KR/USA. Unknown revenue, authorization or
  territory fails closed for policy-sensitive serving/bundling.
- Default every discovered local file to `user_supplied`. Do not infer provider,
  revision, upstream identity, license, revenue or authorization from a filename.
  Local-only files retain the existing warning-only serving posture but have no
  redistribution/license claim. Curated serving or bundling that requires
  provenance refuses unknown/ambiguous identity with a named
  `model_provenance_*` error before GPU admission or bundle writes.
- Expose additive provenance fields on `/v1/models` and bundle sidecars from the
  same record. Reuse `determinism::bundle_record` so `inputs_sha256` covers the
  verdict, immutable evidence and component digests. Exclude volatile
  `probed_at` and `download_id` from the canonical projection; include schema,
  probe and gate versions plus policy result. Do not create a competing
  output-provenance format (`TASK-content-record-provenance.md` remains the
  embedding task).

## Acceptance detail

- Fixture matrix covers GGUF name/architecture, safetensors exact identity
  metadata, safetensors known/unknown base-only tags, exact HF-cache path,
  header-only, path-only, header/path conflict, duplicate registry matches,
  malformed/truncated header, missing digest and unsupported format. Expected
  status/confidence/reason are asserted exactly.
- Provenance survives registry DB close/reopen with the same verdict, reason and
  evidence hash. A changed size/digest/path invalidates the old verdict and
  schedules a re-probe; an offline cached absent file is marked stale, not fresh.
  An older-schema cache rebuilds without fabricating provenance.
- Keep registry access process-local and operator-scoped: no HTTP handler takes
  a caller-selected DB path, and no tenant ownership is invented. Startup,
  admin-authorized `/v1/models/refresh`, write-authorized model deletion, and
  the internal continuation of an admin-authorized download are the only
  registry mutation paths. Model,
  component and capability reads remain behind the existing `read` scope;
  mutations retain `write`/`admin` scope checks in both Axum and Salvo.
- Component fixtures prove projector/encoder/VAE/adapter relations, required
  component absence, and restrictive-parent license propagation. A Qwen Apache
  encoder inside MiniMax H3 does not make the H3 composite Apache-2.0.
- Registry fixtures bind license id/URL/document SHA/weight SHA independently.
  Missing license text hash stays null and triggers no network request.
- Pure policy fixtures cover allowed/refused/unknown territory, NC, LTX revenue
  threshold and MiniMax revenue/authorization cases. Assertions call the existing
  gate and compare the persisted policy result; no family-specific second gate.
- Unknown curated provenance and ambiguous adapter lineage refuse before a fake
  GPU/job/bundle hook runs. User-supplied local-only files remain visible with
  unknown provenance and are excluded from bundle candidates. Canonical-input
  fixtures prove probe-time/download-id changes do not alter `inputs_sha256`,
  while identity, digest, component or policy-result changes do.
- A fake network guard fails the test if catalog refresh/probe attempts any HTTP.
  Restricted/gated fixtures record zero fetch attempts and zero bundle writes.
- Both router stacks reject missing/invalid credentials, expired or badly signed
  JWTs, wrong configured issuer/audience and insufficient read/write/admin scope
  before any registry read or mutation. Successful env-key fixtures prove DTO
  parity; Axum-only JWT acceptance is not claimed as Salvo parity, and this ticket
  adds no token issuance/revocation engine. Response projections never include
  token values, unredacted source URLs or caller-selected DB paths; capability
  omission is not used as an authorization control.
- After DTO implementation, `/v1/capabilities` advertises the provenance schema,
  status/reason vocabulary and implemented download evidence only. Each
  license/provenance row carries a stable registry slug or manifest record id,
  scope and policy result; anonymous rows cannot be joined to a model. Reserved
  or unknown values are not advertised as identified/licensed; OpenAPI and the
  capability contract test include every additive field.

## Non-goals

No automatic Hub/card/license fetch, remote attestation, legal determination,
weight bundling, signature service, output-ownership guarantee, or per-user
tenancy. Exact remote revision/digest remains unknown unless local evidence
supplies it. Future tenancy requires explicit per-resource authorization.