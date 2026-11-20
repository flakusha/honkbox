# TASK: Off-by-default Python sidecar subprocess seam

Status: done (2026-11-20, rerank sidecar lane)
Priority: P2
Epic: epic-external-tools
Owner: unassigned
Acceptance: off-by-default `[python]` config, a per-job supervised subprocess seam with argv built without a shell, lease + bundle + cancel wiring, no new Rust dependency, and a capability row only after a passing probe.
Verification: `cargo test --locked -p honkbox-server`, `bun run lint:rs`
Links: `python-sidecar-pytorch-2026-11-20.md` §3-4
Resources: per-test temp dirs / port 0; stub interpreter injected via PATH; no shared state

## Change

- Add an off-by-default `[python]` config section. With it disabled the seam is
  unreachable and no subprocess is spawned.
- Add a per-job supervised subprocess seam following `TASK-process-supervisor.md`:
  build argv as an explicit array with no shell, route lease acquisition, the
  bundle artifact dir, and cancellation through the existing ledger. Link no
  Python runtime into Rust and add no Rust dependency (binary/subprocess only).
- Advertise the Python capability row only after a startup probe passes; a
  missing or failing interpreter keeps the row false and the route at a named
  501.

## Acceptance detail

- Default config spawns nothing and advertises no Python capability.
- With a stub interpreter on PATH, a job runs through argv without shell
  interpolation, receives a lease and bundle dir, and is killed on cancel.
- The capability row flips true only on a passing probe; `bun run lint:rs`
  stays clean and no new dependency is added.
