# FEAT: Fizgig LoRA training jobs (first slice)

Status: done (first slice + all 6 follow-ups landed 2026-11-22)
Priority: P2
Epic: epic-full-ai-cycle
Owner: fizgig-training lane
Acceptance: `POST /v1/fizgig/train` accepts a typed request, validates family
and dataset/model paths against the allowlists, writes a dataset TOML, runs
the three Fizgig CLI stages under the supervisor, holds one external-GPU slot
for the whole run, and commits LoRAs + manifest to a job bundle;
`GET /v1/fizgig/jobs/{id}` reports state and `POST …/cancel` writes Fizgig's
`.pause_requested` sentinel, then hard-kills the process group after
`HONKBOX_FIZGIG_GRACE_SECS` (default 30 s) if the CLI ignores it. Bounded
per-stage stdout/stderr are committed into the job bundle; terminal job records
are capped at 200.
Verification (original 135/135 result still stands; see addenda for the
follow-up wave): `CARGO_TARGET_DIR=.tmp/cargo-target
LIBCLANG_PATH=/usr/lib/llvm/22/lib64 cargo test --locked --offline
-p honkbox-server --test fizgig` → 135/135 pass
(`.tmp/fizgig-final-test.log`); rustfmt clean. Full `bun run check` is green
after the follow-up wave (workspace clippy included).
Links: `.tmp/research/fizgig-training-python-sidecar-2026-11-22.md` §4,
`TASK-fizgig-cli-probe.md`, `crates/server/src/routes/python_sidecar.rs`
Resources: per-test temp data dirs and a fake CLI script; no GPU, no network,
no ports; one GPU-slot lease per live job

## Change

- First slice: one family at a time via allowlist, one dataset directory,
  one lease, explicit argv, bundle manifest, soft cancel.
- Reuse `supervisor::run`, `admission::acquire`, `jobs::write_atomic` and the
  capability honesty rule — no second process runner or artifact path.
- Out of scope here: queue persistence across restarts, resume endpoint,
  HTTP-sidecar daemon, workbench tools, and a dedicated `Training`
  reservation class (`TASK-training-reservation-class`).
- The route is off unless the probe passes; a disabled or unprobed backend
  answers a named 501 and creates no job.

## Review reconciliation (2026-11-22)

Green gates prove compilation and tests, not the upstream CLI contract. Four
follow-ups are open before this slice can be called complete:

- `BUG-fizgig-family-model-resolution.md` — non-SDXL families need explicit
  VAE/text-encoder paths; the current DiT fallback fails stage 1 after a 202.
- `BUG-fizgig-artifact-cap-and-gc.md` — per-epoch LoRA copies are uncapped and
  the work dir is outside GC.
- `TASK-fizgig-probe-honesty.md` — blocking probe, permanent failure cache,
  and the borrowed `Comfy` capability row.
- `TASK-fizgig-admission-contract-tests.md` — allowlist confinement, terminal
  cancel race, non-unix phantom record, `/v1/jobs` visibility.

The 135/135 targeted test result still stands for the shipped code.

## Contract decisions (2026-11-22 follow-up wave)

- Fizgig jobs are **not** listed by `GET /v1/jobs` and are not cancellable
  through the generic registry: they are process-local, one-lease training
  runs tracked only by `/v1/fizgig/jobs/{id}` and its cancel route. This is
  deliberate for the first slice; no generic registry entry is created.
- The CLI probe is refreshed on the blocking pool with a bounded TTL
  (`HONKBOX_FIZGIG_PROBE_TTL_SECS`, default 60s); a failed probe flips the
  capability false and is retried without restart. `available()` serves the
  last verdict until the refresh lands, so the capability never claims a
  route that cannot admit.
- Follow-up wave (all six review tickets closed 2026-11-22): per-family VAE/
  text encoder resolution, artifact cap + work-dir GC with pause-state
  retention, probe honesty + retry, admission/lifecycle contract tests, the
  ninth exclusive `Training` reservation class, and the curated weight
  license/territory gate. See each ticket's landed section.
