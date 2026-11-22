# TASK: Fizgig CLI probe and supervised three-stage smoke

Status: done
Priority: P2
Epic: epic-external-tools
Owner: fizgig-training lane
Acceptance: an opt-in env-gated probe checks the pinned Fizgig checkout's
`src/fizgig/families/train.py --help` through the supervised subprocess path;
the capability row turns true only when the probe passes; a stub CLI proves
the latents → text → train stage chain, explicit argv, artifact collection and
soft-cancel sentinel without downloading torch or models.
Verification: `cargo test --locked -p honkbox-server --test fizgig`,
`cargo clippy --workspace --all-targets -- -D warnings`, `bun run codegen:check`
Links: `.tmp/research/fizgig-training-python-sidecar-2026-11-22.md` §1-2/§4,
`crates/server/src/routes/python_sidecar.rs` (argv + supervisor precedent),
`.plan/tickets/TASK-python-sidecar-subprocess.md`
Resources: per-test temp dirs under the test data dir; no model downloads,
no ports, no GPU slot for the probe itself

## Change

- Env-only config (`HONKBOX_FIZGIG_*`), off by default: root, venv, family
  allowlist, dataset/model path roots, timeout. No `config.rs` change in this
  slice — file-backed `[fizgig]` is a follow-up if operators need it.
- Probe the pinned CLI with a bounded supervised `--help` call on the
  blocking pool; cache the verdict for `HONKBOX_FIZGIG_PROBE_TTL_SECS`
  (default 60s, bounded 1..=3600). A failing probe is a named 501 and is
  retried after the TTL, never a silent fallback or a permanent failure.
- The three-stage invocation is built as explicit argv arrays (never a shell)
  and shares one exclusive `Training` lease for the whole run.
- Note the ceiling: probe checks the CLI's presence, not model files or CUDA.
  Upgrade path: `TASK-fizgig-license-gate` + a real installed-model probe.

## Verification addendum (2026-11-22)

`cargo test --locked --offline -p honkbox-server --test fizgig --test licenses`
→ 205 passed (fake-CLI lifecycle + probe failure/recovery + license refusals);
`cargo test --locked --offline -p honkbox-server --test capabilities` → part
of a 290-passing run with `--test fizgig`; workspace `bun run check` green.
Real pinned-CLI fidelity still requires an installed checkout + checkpoint;
no such smoke was run here.
