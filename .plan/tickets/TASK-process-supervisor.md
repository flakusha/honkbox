# TASK: External process supervisor

Status: proposed
Priority: P0
Epic: epic-external-tools
Owner: unassigned
Acceptance: one supervisor starts/stops external engines with explicit args, timeout, CPU/RAM limits, cancel, and artifact collection.
Verification: `cargo test --locked -p honkbox-server`, `bun run lint:rs`
Links: `.tmp/research/ReviewedFullCycle-Tool-Management.md`, `ReviewedFullCycle-Blender-Workstream.md`
Resources: per-job temp working dirs; process groups killed in `finally`/drop

## Change

- Own child lifecycle for Blender (`-b -P script.py --python-exit-code N`),
  ComfyUI, and future tools; no `bpy` embedded in-process.
- Cap threads (`-t`), device selection, and RSS (cgroup v2 where available);
  hold a GPU lease for GPU-backed work via the resource ledger.
- Cancel kills the process group; deterministic teardown on success, failure,
  and timeout; collect stdout/stderr and output globs into the job bundle.

## Acceptance detail

A cancelled job leaves no orphan process and releases its lease; a failed job
still writes its log and sidecar; no fixed temp path is shared between tests.
