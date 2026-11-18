# TASK: Gate the GPU-slot serialization contract

Status: done
Priority: P1
Epic: epic-docs-tooling
Owner: honkbox-openapi-coverage
Acceptance: a server test that issues a request to a GPU-contending route without serializing on a lock fails `bun run check`; the GPU-contending route list is derived from the server source, not hand-maintained.
Verification: `bun run audit:gpu-slot`, `bun run check`, `bun run test`
Links: `scripts/audit-gpu-slot.ts`, `crates/server/src/routes.rs` (`admission::acquire`), `crates/residency/src/lib.rs` (`Demand::for_kind`), `crates/server/tests/{compression,embeddings,audio,models}.rs`
Resources: no runtime resources; the gate only reads `crates/server/src/routes.rs` and `crates/server/tests/*.rs`

## Problem

The residency ledger's GPU slot is ONE process-global counter
(`capacities.gpu_slots = 1`), so two tests that both take a GPU lease race —
the loser gets a 429 (`gpu slot busy`) or blocks on the admission queue. The
server test binaries therefore serialize their GPU requests on a per-binary
lock (`SLOT_LOCK`, `OP_LOCK`, `LEDGER_LOCK`), and the contract is documented
in each binary's module docs.

That contract was held only by convention: a new test that requested a GPU
route without taking a lock would fail only intermittently, under load, and
the failure would look like a flaky product bug rather than a test-contract
violation.

## Change

- `scripts/audit-gpu-slot.ts` (new): for every `#[tokio::test]` in
  `crates/server/tests/*.rs`, it asks whether the test issues a request to a
  GPU-contending route (directly or through a helper) and whether it
  serializes (holds a lock, directly or through a helper). An unserialized
  GPU request fails the gate.
  - The GPU-contending route list is DERIVED from `crates/server/src/routes.rs`:
    a route whose handler (or a handler it calls) acquires a non-`Rag`
    resource kind takes a GPU slot. `Rag` demand is `gpu_slots: 0`
    (`Demand::for_kind`), so it never contends — which is why the image-ops
    route needs no lock. A new GPU route is therefore covered the moment it
    is mounted, with no list to maintain.
  - A GPU route counts as requested when its URI appears outside assertion
    macros and `for x in [...]` route lists (those document the OpenAPI
    document, they are not calls). Both `.uri("/v1/...")` and
    `raw_post(app, "/v1/...", ...)` are caught.
  - The lock check is deliberately broad (any `.lock()` call counts): a false
    positive would block every developer's build, so the gate biases toward
    missing a gap over inventing one.
- `package.json`: `audit:gpu-slot` script; wired into `check` (after
  `lint:rs`).
- Exemptions (`GATED_BEFORE_SLOT`): a request refused BEFORE the slot is
  taken needs no lock, but the exemption is honored only while the test PINS
  the invariant by asserting the response carries no `X-Job-Id` (no job
  registered → the request never reached the ledger). Three tests are
  exempted, and each now carries that pin:
  - `embeddings.rs:embeddings_model_refused_by_chat_completions` (chat route
    gates on the model capability),
  - `models.rs:transcribe_validates_model_first` (the transcribe route
    validates the model first),
  - `audio.rs:synthesize_input_over_cap_is_rejected` (the synthesize route
    checks the input cap first).
- `README.md`: the `audit:gpu-slot` script and the extended `check` chain.

## Verification


## Why the `check` chain stays sequential

The chain in `package.json` looks like it could run its steps concurrently
(`format` 0.02s, `lint` 0.04s, `typecheck` 0.72s, `lint:rs` 0.21s,
`audit:gpu-slot` 0.33s, `codegen:check` 0.54s — measured warm). It must not:
`codegen:check` REWRITES `packages/web-ui/src/gen/{api-types.ts,openapi.json}`
on every run (verified: the mtimes move even when the bytes do not), and
`format` (dprint check) and `typecheck` (tsc) read exactly those files. Run
concurrently, they would read a half-written file and fail intermittently.
`lint:rs` and `codegen:check` both drive cargo, which serializes them on the
target-directory lock anyway. The chain is therefore sequential by data
dependency, not by habit — noted in `scripts/codegen-openapi-check.ts` too.
- `bun run audit:gpu-slot` → `OK: every GPU-requesting test is serialized by
  a lock` (11 GPU-contending routes derived).
- Negative controls (`.tmp/intercom/scripts/audit-gpu-slot-control.sh`):
  removing a `SLOT_LOCK` from a GPU test fails the gate naming both affected
  tests; removing a pinning assertion fails the gate with `exemption lost`.
- `bun run check` exit 0; `bun run test` exit 0; `bun run lint:rs` exit 0.
