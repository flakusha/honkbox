# FEAT: Fizgig LoRA training jobs (first slice)

Status: in progress (first slice landed; 4 open follow-ups from the 2026-11-22 review)
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
Verification: `CARGO_TARGET_DIR=.tmp/cargo-target LIBCLANG_PATH=/usr/lib/llvm/22/lib64
cargo test --locked --offline -p honkbox-server --test fizgig` → 135/135 pass
(`.tmp/fizgig-final-test.log`); rustfmt clean. Lane files clippy-clean;
`cargo clippy -p honkbox-server --all-targets -- -D warnings` is blocked only
by `routes/recognition.rs:59` (too_many_lines, another lane).
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
