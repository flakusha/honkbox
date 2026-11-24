# FEAT: ComfyUI workflow pipelines through typed slots

Status: proposed
Priority: P0
Epic: epic-comfyui-managed-runtime
Owner: unassigned
Acceptance: typed submit/cancel/events/artifacts for immutable API-format
templates through `POST /v1/jobs/comfyui`, with no raw workflow proxy.

- Submit only an immutable admin-authored API-format template plus closed
  typed slot values through `POST /v1/jobs/comfyui`; the registry label is
  `comfy_workflow`.
- Preflight every required `class_type` with a targeted upstream lookup and
  return 501 naming a missing class before `/prompt` is called.
- Use one fresh canonical `prompt_id` and `client_id` per execution; observe a
  private `/ws` connection and accept only events for that prompt.
- Recover completion and declared outputs from `/history/{prompt_id}` and
  `/view`; socket loss or unknown submission never causes blind resubmission.
- Cancel only the job's exact upstream id with targeted `/queue` and
  `/interrupt`; never clear the queue, globally interrupt or call `/free`.
- Copy only declared output references into an atomic job bundle and
  content-addressed artifacts, with bounded streaming, hashes and no raw
  workflow graph or server paths in public responses.
- Honor the managed runtime's existing lifetime lease; serialize managed
  submissions and take one detached Comfy lease per job.
- Observable fake-server tests cover the contract; live smoke is ignored by
  default and requires an operator-pinned dedicated ComfyUI instance.
Verification: `cargo test --locked -p honkbox-server --test comfy_workflows`,
`cargo test --locked -p honkbox-server --test apps`,
`cargo test --locked -p honkbox-server --test job_content`,
`bun run codegen:check`, `bun run check`; live smoke is opt-in/ignored.
Links: research, prior reviews, companion policy gate, epic and primary API docs.

- Research: `.tmp/research/comfyui-workflow-sidecar-2026-11-25.md`
- Prior reviews: `.tmp/research/comfyui-endpoint-review-2026-11-24.md`,
  `.tmp/research/comfyui-managed-instance-review-2026-11-24.md`,
  `.tmp/research/comfyui-prior-findings-reconciliation-2026-11-24.md`
- Companion policy gate: `.plan/tickets/TASK-comfy-custom-node-capability-gate.md`
- Epic: `.plan/epics/epic-comfyui-managed-runtime.md`
- API routes: <https://docs.comfy.org/development/comfyui-server/comms_routes>
- API examples: <https://docs.comfy.org/development/comfyui-server/api-examples>
Resources: runtime/job/artifact seams, isolated fake fixtures and one Comfy lease.

- `crates/server/src/apps.rs`, `crates/server/src/routes/apps.rs`
- `crates/server/src/jobs_registry.rs`, `crates/server/src/jobs.rs`
- `crates/server/src/routes/job_content.rs`, `crates/server/src/artifacts.rs`
- `crates/server/src/routes/blender_exec.rs`, `crates/server/src/metrics.rs`
- `crates/server/src/routes/capabilities.rs`
- `crates/server/src/determinism.rs`, `crates/server/src/routes.rs`
- `crates/server/src/config.rs`, `crates/residency/src/lib.rs`
- One managed execution permit or one detached `ResourceKind::Comfy` lease
- One reviewed WebSocket client transport (`tokio-tungstenite`; current reqwest
  has no WS feature)
- No public wire proxy, new scheduler, second runtime manager or hand-edited
  generated file

## Context

[FACT] `routes.rs:428-436` currently mounts only ComfyUI start, status, queue,
history and view routes. `routes/apps.rs:1-9` deliberately excludes `/prompt`,
`/ws`, `/interrupt`, `/queue` writes and node-graph bodies.

[FACT] `Manager::start` acquires `ResourceKind::Comfy` before spawn and the
supervisor holds it until child reap. Default residency has one GPU slot, so
workflow submission cannot acquire that same default slot again.

[FACT] typed reads already drop raw history graphs, report progress and mark
detached discovery as `detached-unverified`. The missing feature is a closed
write/execution/artifact pipeline, not another manager or general proxy.

## Scope

## Public contract

- Add `POST /v1/jobs/comfyui`, matching the existing specialized
  `POST /v1/jobs/blender` pattern; `GET /v1/jobs` is list-only today. Add
  `JobKind::ComfyUiWorkflow` with wire label `comfy_workflow`, generated API
  types and OpenAPI. No request `kind` field, `POST /comfyui/prompt`, raw graph
  field or upstream URL/path parameter.
- Request schema: `template_id`, `template_version`, optional
  `template_sha256`, and named `ComfySlotValue`s only; body cap 64 KiB. Deny
  unknown fields, unknown slots, arbitrary JSON, URLs and filesystem paths.
- Slot kinds: bounded `text`, `u64`, `bool`, manifest `enum`, and existing
  honkbox `artifact` id. Use `BTreeMap<String, ComfySlotValue>` for deterministic
  order; entry count cannot exceed the manifest slot count and unknown names
  fail. `ComfySlotValue` is a closed tagged enum with
  `deny_unknown_fields`; reject missing required slots and client nulls, and
  apply defaults only from the manifest. Artifact slots resolve only stored
  sha256 artifacts.
- Illustrative request (template ids are operator-defined):

  ```json
  {
    "template_id": "txt2img-basic",
    "template_version": "1",
    "slots": {
      "prompt": {"kind": "text", "value": "a lighthouse"},
      "seed": {"kind": "u64", "value": 42},
      "input_image": {"kind": "artifact", "value": {"sha256": "<64 lowercase hex>"}}
    }
  }
  ```

- After typed validation, acquire the runtime permit/lease, preflight classes,
  then register `queued` and create a separate ULID bundle directory. A failed
  preflight drops the permit and registers no job. Return
  `202 {job_id, status, bundle}` plus `X-Job-Id`, matching the Blender route.
  Always close with an honest terminal status and `job_finished`; bundle
  manifest remains the durable post-SSE record.
- Errors after 202 are terminal job states. Immediate errors: 400 invalid
  slot, 404 unknown template/artifact, 409 digest mismatch/app stopping, 413
  bound exceeded, 429 runtime busy, 501 missing class, 502 upstream fault,
  503 no data dir, 504 execution/recovery deadline.
- Wire the new kind through every closed surface: `JobKind::label`,
  `metrics::JOB_KIND_LABELS`, the `("comfy", "/v1/jobs/comfyui")` entry in
  `capabilities::ADMITTING_ROUTES`, OpenAPI, the complete `JobInfo`/`JobEvent`
  kind vocabulary (including existing `blender`/`ocr` and new `comfy_workflow`)
  and generated API types. Register it `interruptible: true` because the worker
  honors the registry token and sends targeted upstream cancellation.

## Immutable template manifest

Templates live in an operator-writable, honkbox-owned data directory; they
are not client uploads and are not fetched at runtime. The server loads and
validates them at startup before exposing `(id, version)`; adding a version
requires a restart. Each `(id, version)` owns a separate directory; if that
directory already exists with different hashes, startup fails rather than
overwriting it. This slice adds no HTTP template-management route.

Required fields:

```text
manifest_version, id, version, manifest_sha256, graph_sha256,
api_graph_file, required_class_types[], forbidden_class_types[],
slots{}, outputs[],
runtime_policy { auto_install: false, runtime_fetch: false }
```

- Reuse `jobs::determinism::canonical_hash` and `sha256_hex`. Store lowercase
  SHA-256. `manifest_sha256` hashes canonical manifest bytes excluding itself;
  `graph_sha256` hashes the exact API-format graph file.
- The API graph is at most 1 MiB. `(id, version)` is immutable. A supplied
  digest mismatch returns 409 before
  upstream admission. `api_graph_file` must be a plain relative path inside the
  template directory with no symlink traversal. Load validation rejects
  non-object nodes, missing/empty `class_type`, malformed links, undeclared
  slot targets, duplicate slot names and no declared output.
- `required_class_types` must equal the graph's class set; it must be disjoint
  from operator-enumerated `forbidden_class_types`. A slot can replace only its
  declared node input; it cannot add nodes, alter `class_type`, change links or
  choose model/output paths.
- Each output names a node id, declared media key (`images`, `gifs`, or a typed
  `audio` shape proven by tests), allowed extensions and count/byte bounds.
- Graph format is API format: numeric string node ids, `class_type`, `inputs`,
  and links such as `["4", 0]`; browser/workflow JSON is rejected.
- Node packs, weights, refs, patches and install policy remain operator-owned.
  The manifest policy is an operator assertion, not HTTP-verifiable proof.

## Upstream execution

1. Resolve the upstream from the ready managed `Manager` when configured,
   otherwise from the detached external-tools row; never from the request.
   Acquire a managed execution permit or detached Comfy lease. In managed mode,
   reuse the process lease held by `Manager`; never call ledger acquire for
   `ResourceKind::Comfy` again. A second managed execution gets 429.
2. Query `GET /object_info/{class}` once per required class, percent-encoded,
   with a 1 MiB reply cap. Exact class key present is visibility only. `{}` or
   absent key is 501 naming the class; non-200/invalid JSON is 502. Do not use
   full `/object_info` as a compatibility guarantee. Only after preflight
   succeeds, register the job and create its separate ULID bundle directory;
   rejection above leaves no registry entry or bundle.
3. Generate canonical UUID `prompt_id` and UUID `client_id`. Open private
   `GET /ws?clientId={client_id}` before submit; 1 MiB text-frame cap, discard
   binary previews, enforce idle/read deadlines.
4. Resolve artifact slots and stream `POST /upload/image` multipart parts with
   field `image`, generated safe filename, `type=input`, empty subfolder and
   no client path. Derive the extension only from the slot's `accept_media` and
   stored content type; reject mismatches. Require the returned
   `{name,subfolder,type}` to match the
   generated reference. Per-file upload bound is no greater than ComfyUI's
   100 MB default and the template cap, with 512 MiB total staged input per
   job; never buffer whole media.
5. `POST /prompt` a bounded 1 MiB API graph body with `prompt`, `prompt_id` and
   `client_id`. On timeout mark `submission_unknown` and reconcile by id; never
   resubmit or reuse the id. Record `number` only as diagnostic data.
6. Filter WS text events to `execution_start`, `executing`, `progress`,
   `executed`, `execution_error`, `execution_interrupted`,
   `execution_success`. Require exact `data.prompt_id` where present; drop
   foreign/missing ids and all custom/binary frames. `executing node:null` and
   `executed.output` are hints, not final artifact truth.
7. On socket loss, poll `/history/{prompt_id}` with bounded backoff and a
   deadline. `{}` means not-yet-available, never success. Use `/queue` only to
   reconcile the exact id; never infer another client's state.
8. Parse terminal `status.status_str` or `completed`. Accept only declared
   output node ids/media keys and `type=output` references. Keep the existing
   single-component filename/subfolder policy; invalid references fail closed.
   Drop raw `prompt`, meta and host paths from public records.
9. Stream each `/view` to a sibling `.part`, enforce per-file and aggregate
   limits, hash while writing, fsync and rename. Per-file and total bundle
   limits reuse `routes::job_content::MAX_CONTENT_BYTES`,
   `MAX_UNCOMPRESSED_BYTES` (512 MiB) and `MAX_ZIP_ENTRIES` (256); expose the
   latter two `pub(super)` rather than duplicate constants. Register the same
   bytes through a streaming artifact import,
   not `put_bytes` on whole media. Follow `routes::blender_exec::commit`'s
   artifact-first ordering and write path-free `manifest.json` last.
10. Bundle names are generated `comfy_<node>_<index>.<ext>` and must satisfy
    `routes::job_content::validate_content_name` (plain name, at most 128
    bytes). Manifest records
    job id, bundle id, template id/version/both hashes, slot values, upstream
    id/client id, declared node/filename/subfolder/type, bundle name, media kind, bytes,
    sha256 and artifact id. Public delivery uses artifact ids and listed
    bundle names only.

## Cancellation, stop and cleanup

- Job cancel marks the existing registry token. Before submit it makes no
  upstream call. After submit it sends `POST /queue {"delete":[id]}` and
  `POST /interrupt {"prompt_id":id}`; both targeted requests close the
  queued-to-running race. Never send an empty/malformed interrupt body, which
  upstream interprets as global, and never call queue clear or `/free`.
- Cancel/recovery is idempotent. A late WS frame cannot turn cancellation or
  failure into success. Hold the permit/lease through terminal commit or
  bounded cleanup.
- `POST /v1/apps/comfyui/stop` returns 409 while a workflow execution permit is
  held. Operator cancels the job first. Crash/readiness loss fails owned jobs;
  server shutdown cancels tasks before child teardown.
- Optional post-commit `POST /history {"delete":[prompt_id]}` is best effort
  and targeted. Failure cannot invalidate a committed bundle. It does not
  delete operator-owned output files.
- `/free` is global and outside the per-job path. Detached servers may serve
  other clients, so no automatic free is added.

## Security and policy

- Preserve router bearer auth; template files remain operator-only with no
  HTTP lifecycle route. Upstream client remains loopback-only, credential-free,
  `no_proxy`, redirects
  disabled and ambient proxy disabled.
- No raw workflow JSON proxy, node graph passthrough, client upstream URL,
  arbitrary filename/model path, extension registry or Manager endpoint.
- No `pip`, `git clone`, custom-node install, weight download or generic URL
  fetch. Runtime-fetching/install classes must be listed by the operator in
  `forbidden_class_types` and are rejected at load; URL-valued slots are
  rejected by schema. Operator must pre-stage weights and disable pack
  auto-fetch.
- Never claim `/object_info` proves pack version/digest. Pair this feature with
  `TASK-comfy-custom-node-capability-gate.md`.
- Native H3/LTX remains an alternative typed media adapter. It must not weaken
  class preflight, license gates, hashes or the no-proxy boundary.

## Acceptance criteria

## Observable fake-server tests

`comfy_workflows` tests follow the existing integration-test house pattern:
include the needed server modules with `#[path = "../src/..."]` as
`tests/blender_jobs.rs` and `tests/apps.rs` do; build a per-test `AppConfig`
and axum router; upload real fixture artifacts through `POST /v1/artifacts`;
and drive routes with `tower::ServiceExt::oneshot`.

Each fake ComfyUI binds `127.0.0.1:0` and keeps the listener alive, as
`tests/downloads.rs::MockServer` and `tests/mesh_reroute.rs::spawn_upstream`
do. Do not copy `tests/apps.rs::free_port`'s bind-then-release pattern. Parse
bounded HTTP request heads/bodies, record requests in test-owned
`Arc<Mutex<Vec<_>>>`, and use the same WebSocket transport as production for
a real handshake and masked/unmasked frame exchange. The fake owns a
shutdown/abort handle and `Drop` closes the listener and joins its tasks; no
detached infinite fixture thread. Each test owns an RAII temp dir, UUIDs and
fake queue; no static/shared ComfyUI queue, process or port.

Design the pipeline seam for parallel tests: pass `&jobs_registry::Registry`
and `&honkbox_residency::Ledger` (plus a resolved upstream/client and per-test
template/artifact roots) into execution. Production passes the existing
globals; protocol tests construct local `Registry::default()` and
`Ledger::new()`. `Registry` already documents that a locally built registry
behaves exactly like the global one (`jobs_registry.rs:204-207`), and `Lease`
borrows its ledger, so cleanup is deterministic on every return path.

Only the small route/auth/OpenAPI subgroup may touch the process-wide router
slots; serialize that subgroup with a binary-local `ROUTE_LOCK`, restore
`set_manager_for_tests(None)`/`set_registry_for_tests(None)` and key env vars,
and follow `tests/apps.rs:75-81,405-474`. Managed start/stop tests that must use
the global ledger take the existing `LEASE_LOCK` pattern and release every
lease (`tests/supervisor.rs:187-215`). All HTTP/WS/history/cancel/artifact
protocol tests run in parallel with local state. No test-order dependency and
no suite-wide serialization.

1. **Happy path:** fake `/object_info/KSampler` exposes the class; `/upload/image`
   validates generated input; `/ws?clientId=...` records the id; `/prompt`
   asserts exact graph/slots/ids; history returns `{}` then a terminal record;
   `/view` returns known bytes. Assert 202, `X-Job-Id`, separate bundle id,
   queued/running/ready progress, exact `job_finished`, one bundle, listed
   generated filenames, hashes,
   artifact ids and no graph/path leakage.
2. **Filtering:** inject binary preview, custom event, foreign `prompt_id`, and
   matching progress/interrupted/error/success frames. Assert foreign/binary
   data cannot change progress, create artifacts or complete a different job.
3. **Preflight:** present, absent, empty `{}`, non-200 and malformed class
   catalogs have distinct outcomes. Absent exact class returns 501 naming it
   with zero `/prompt` calls, no registry entry and no bundle; non-200 is 502.
4. **Template admission:** missing/empty `class_type`, unknown required class,
   link/target mismatch, unknown slot/field, raw graph request and digest
   mismatch are rejected before `/prompt`; request digest mismatch returns 409.
   Reusing an installed `(id, version)` with changed bytes fails startup rather
   than overwriting. Two slot maps inserted in opposite order produce the same
   canonical slot hash.
5. **Recovery:** force WS close after submit; history polling reaches terminal
   exactly once. Assert one `/prompt` call, no duplicate artifact and no
   resubmit. Unknown history remains non-success until deadline.
6. **Unknown submit:** fake accepts `/prompt` then drops the response; recovery
   reconciles the original id without issuing a second submit.
7. **Bounds:** a request over 64 KiB is 413 with no job. A tiny template cap
   makes an oversized upload fail before
   `/prompt` without a huge fixture. Unit-test the bounded reader with a small
   limit; the HTTP fixture advertises an over-cap `Content-Length`. Also cover
   excessive output count, unexpected node/media key, non-output type,
   traversal/symlink-shaped reference and invalid upload response; each fails
   the job without a publishable bundle.
8. **Cancel race:** queue a job, then transition to running and cancel. Assert
   exact id in both targeted requests, no empty interrupt body, no clear/free,
   cancelled terminal state, and a decoy upstream job remains untouched.
9. **Lease:** on a test-local `Ledger`, detached job holds
   `ResourceKind::Comfy` through commit and a competing native acquire is
   denied while held. A managed fixture pre-acquires the runtime lease;
   workflow succeeds without a second acquire; concurrent second workflow gets
   429. Every lease is dropped/asserted released in teardown.
10. **Lifecycle:** stop returns 409 during active workflow; after cancel, stop
    tears down the child and releases the lease. Simulated crash/readiness
    loss produces terminal failure and no orphan child/task.
11. **Atomicity:** fail one output after another was copied; assert no
    completed bundle/manifest or artifact registration is exposed.
12. **Route/wiring (isolated `ROUTE_LOCK` subgroup):** unauthenticated
    `POST /v1/jobs/comfyui` is 401 and registers no job; OpenAPI serves the path
    and typed request; the
    capability queue has a callable `comfy` resource row (`reserved: false`);
    `GET /v1/jobs` and lifecycle events use the `comfy_workflow` job label; the bounded
    `honkbox_jobs_total` vocabulary includes it. `GET /v1/jobs` remains
    list-only and there is no generic job-submit route.

`apps` tests retain managed lifecycle/readiness/detached-freshness coverage and
add the execution-permit stop case. `job_content` tests prove manifest-last
publishing and the existing bounds still apply.

## Live opt-in smoke

Ignored unless `HONKBOX_COMFY_LIVE=1`. Requires an operator-pinned dedicated
loopback ComfyUI instance and output dir; never use a shared queue.

- Record ComfyUI version, required node classes, operator pack refs/patches/
  digests, pre-staged weights/license rows and policy audit in the run record.
- Submit one tiny pinned template; assert one prompt and exact ids, then force
  the observer socket closed and verify history recovery and declared output
  copy.
- Submit a second id and cancel it while a decoy upstream id is pending;
  assert decoy progress and files remain unchanged.
- Confirm no Manager/install/download/fetch endpoint was requested and no
  unexpected egress occurred under the operator's blocked-egress policy.
- On failure retain sanitized diagnostics only; never upload graphs, inputs or
  weights to an external service.

## Verification

Commands to run after implementation (not run for this ticket):

```sh
cargo test --locked -p honkbox-server --test comfy_workflows
cargo test --locked -p honkbox-server --test apps
cargo test --locked -p honkbox-server --test job_content
bun run codegen:check
bun run check
HONKBOX_COMFY_LIVE=1 cargo test --locked -p honkbox-server \
  --test comfy_workflows -- --ignored --nocapture
```
