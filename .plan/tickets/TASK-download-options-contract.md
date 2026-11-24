# TASK: provider-neutral download options contract

Status: proposed
Priority: P1
Epic: epic-external-tools
Owner: unassigned
Acceptance: `DownloadRequest` has a provider-neutral, explicitly validated option vocabulary; ordinary legacy `repo`+`file` and direct `url` requests keep their behavior; full-revision and expected-digest plans cannot publish partial or unprobed model trees; unsupported options and secrets are rejected without leakage; existing server-side admin authorization remains fail-closed with no state mutation on rejection.
Verification: `bun run test:server`, `bun run codegen:check`, `bun run check`
Links: `.tmp/research/model-identity-license-download-2026-11-25.md` §§6–7, `.tmp/research/video-audio-reuse-and-download-options-2026-11-24.md`, `FEAT-download-provider-adapters.md`, `TASK-model-provenance-contract.md`, `crates/server/src/downloads.rs`, `crates/api-types/src/lib.rs:2327-2345`
Resources: per-test temp model roots and staged trees, fake HTTP/metadata fixtures, process-environment indirection fixtures; no live Hub traffic or shared cache

## Change

- Add optional `provider` (`hf` | `direct` | `comfyui_node` | `llama_server`),
  `revision`, `path`, `allow_patterns`, `ignore_patterns`, `destination_root`,
  `token_env`, `expected_sha256`, `dry_run`, and `resume`. Keep `file` as the
  legacy single-file spelling; supplying both `file` and `path` is ambiguous.
- Publish a provider option-support matrix. Legacy HF and direct forms retain
  their current semantics. `comfyui_node` and `llama_server` are reserved values
  and reject until `FEAT-download-provider-adapters` implements them. Reject an
  unknown provider or any option the selected provider cannot honor before
  enqueue; never silently ignore it.
- A non-default HF `revision` must be exactly 40 lowercase hex characters (a
  full commit SHA). Reject branches, tags, short hashes, whitespace and mixed
  case. Legacy omitted revision may still resolve `main`, but provenance records
  it as mutable rather than inventing a commit. Persist the resolved full SHA in
  plan/provenance metadata when an adapter knows it.
- `expected_sha256` must be exactly 64 lowercase hex characters. A supplied
  digest and source-published digest must agree before enqueue. Direct downloads
  with a supplied digest enforce it; mismatch deletes staging and never commits.
- Define a plan/commit boundary for single files and staged trees. A multi-file
  plan writes a sibling staging tree, verifies every expected digest and required
  component, runs the offline provenance/header/relationship probe, writes its
  sidecars, then publishes with atomic no-replace semantics. Missing, duplicate,
  mismatched or unprobeable required components leave no final tree. If a target
  appears between probe and commit, preserve it and never overwrite it.
- `dry_run` returns resolved revision, file list, sizes, expected digests and
  option rejections without writing staging/final files and without creating a
  download id. `resume: false` discards stale staging; `resume: true` may reuse it
  only when source, revision, destination and expected digest all match.
- `path` and `destination_root` are confined to allowlisted managed model
  subroots, never arbitrary caller paths. Reject absolute paths, `..`, separators
  used as traversal, control characters and symlink escapes before enqueue;
  revalidate confinement without following symlinks at staging and commit.
- `token_env` matches `[A-Z_][A-Z0-9_]*` and is a name, never a credential.
  Resolve its value only inside the provider adapter at fetch time. DTOs, status,
  sidecars, logs, metrics and errors contain only the name/redacted source, never
  the value. Reject token/header/credential fields, URL userinfo, URL fragments,
  and known credential-bearing query parameters before enqueue/fetch. The
  case-insensitive rejected-query set includes `token`, `access_token`, `api_key`,
  `key`, `signature`, `sig`, `credential`, `password`, `secret`, `X-Amz-Signature`,
  `X-Amz-Credential` and `X-Amz-Security-Token`. Keep any ordinary query only in
  the internal fetch plan; remove it from `DownloadState.source` and defensively
  redact URL-bearing transport errors. Ordinary legacy URLs retain their wire
  form; credential-bearing URLs are refused.
- Single-flight identity includes provider, normalized internal source,
  revision, destination and expected digest, not just the target path. Only
  identical plans join; different revisions/digests for one target must not share
  progress or bytes.
- A pre-existing target is not `done` when revision/digest/provenance was
  requested: verify the existing bytes/tree or refuse. Never overwrite or claim
  it merely because a second request arrived.
- Provenance/license policy remains in `catalog::licenses` and
  `catalog::license_gate`. This ticket performs no fetching. Restricted, gated,
  unknown or ambiguous weights are never auto-fetched or bundled; adapters must
  obtain an explicit operator authorization and a passing existing-gate result.
- Preserve the existing server-side boundary: `/v1/downloads*` remains inside
  both routers’ auth middleware and requires `admin` scope in bearer mode. Do not
  add an unauthenticated route or bypass, expose the registry DB to callers, or
  weaken the non-loopback `auth = none` bind refusal. The post-download rescan is
  an internal process-local continuation of an admin-authorized operation.

## Acceptance detail

- Existing ordinary no-new-field HF LFS verification and 200/206 resume
  fixtures still pass unchanged; existing single-flight and cancellation
  behavior is preserved.
- Table-driven validator tests cover every provider/option combination, both
  `file` and `path`, short/non-hex revisions, malformed/conflicting digests,
  unsupported snapshot options, and credential-like fields.
- A sentinel credential supplied through process-environment indirection is
  absent from request/state/sidecars, captured logs, metrics and error strings.
  Userinfo, fragment and each rejected-query fixture fails before a fake provider
  records any fetch; ordinary-query status strips the query, and reqwest-style
  transport errors containing the original URL are redacted.
- A local staged-tree fixture commits only after all digests and component probes
  pass; each missing/mismatched/invalid-component case leaves no final tree. A
  target-appears-before-commit race preserves the target, and two plans with the
  same filename but different revision/digest never join single-flight.
- Dry-run leaves the temp root empty and returns no cancellable id.
- Restricted/ambiguous license fixtures refuse before a fake provider records any
  fetch; no weight bytes are added to the repository or test fixtures.
- Axum and Salvo negative-auth fixtures reject missing/invalid credentials,
  expired/bad-signature/wrong-issuer/wrong-audience JWTs, and non-admin scopes
  with 401/403 before enqueue, cancellation, deletion or registry mutation.
  Successful admin env-key fixtures prove the same DTO behavior on both stacks;
  the existing Axum-only JWT acceptance path is not silently claimed as Salvo
  parity, and this ticket adds no token issuance or revocation semantics.
- `/v1/capabilities` advertisement is implemented only after the corresponding
  validator/route behavior lands; reserved providers/options are not advertised
  as supported.

## Non-goals

No provider adapter, live network fetch, managed-engine flag passthrough, model
bundle, or CI model download in this ticket. Those belong to
`FEAT-download-provider-adapters.md`.
