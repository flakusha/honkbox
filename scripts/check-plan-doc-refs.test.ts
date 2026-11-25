import { expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Exercises the real scripts/check-plan-doc-refs.ts gate against disposable
// fixture trees.
//
// Resource contract: each test owns ONE unique temp directory created with
// mkdtemp under the OS temp root and removes it in finally. The gate only reads
// that tree, so parallel tests share no mutable state and need no ordering.
//
// TODO(env): bun-only - add an executor adapter when a node test target lands
// Named Bun-only APIs: bun:test, import.meta.dir.

const SCRIPT = join(import.meta.dir, "check-plan-doc-refs.ts");

function withRepo(run: (root: string) => void): void {
  const root = mkdtempSync(join(tmpdir(), "honkbox-plan-refs-"));
  try {
    mkdirSync(join(root, ".plan", "tickets"), { recursive: true });
    mkdirSync(join(root, ".plan", "epics"), { recursive: true });
    run(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

function writeTicket(root: string, name: string, body: string): void {
  writeFileSync(join(root, ".plan", "tickets", name), body);
}

function runGate(root: string): { code: number; stdout: string; stderr: string; } {
  const result = spawnSync("bun", [SCRIPT, root], { encoding: "utf8" });
  return {
    code: result.status ?? -1,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
  };
}

test("accepts a plan tree whose references resolve", () => {
  withRepo((root) => {
    writeTicket(root, "TASK-ok.md", "Links: `.plan/tickets/TASK-ok.md`\n");

    const result = runGate(root);

    expect(result.code).toBe(0);
    expect(result.stdout).toContain("all intra-repo plan references resolve");
  });
});

test("fails on a dangling intra-repo plan reference", () => {
  withRepo((root) => {
    writeTicket(root, "TASK-broken.md", "Links: `.plan/tickets/TASK-missing.md`\n");

    const result = runGate(root);

    expect(result.code).toBe(1);
    expect(result.stderr).toContain("DANGLING");
    expect(result.stderr).toContain("TASK-missing.md");
  });
});

test("does not fail on a cross-repo plan reference", () => {
  withRepo((root) => {
    writeTicket(
      root,
      "TASK-cross-repo.md",
      "Links: loop-lore `.plan/tickets/TASK-consumer.md` (the consumer).\n",
    );

    expect(runGate(root).code).toBe(0);
  });
});

test("reports a not-yet-existing source path as advisory, not a failure", () => {
  withRepo((root) => {
    writeTicket(root, "TASK-planned.md", "The new `crates/server/tests/planned.rs` proves:\n");

    const result = runGate(root);

    expect(result.code).toBe(0);
    expect(result.stdout).toContain("advisory:");
  });
});

test("skips gitignored .tmp references instead of failing on them", () => {
  withRepo((root) => {
    writeTicket(root, "TASK-research.md", "Links: `.tmp/research/note.md`\n");

    const result = runGate(root);

    expect(result.code).toBe(0);
    expect(result.stdout).toContain("tmp_refs_skipped=1");
  });
});

test("reports an epic status claim that disagrees with the ticket header", () => {
  withRepo((root) => {
    writeTicket(root, "TASK-shipped.md", "Status: done\nPriority: P1\nEpic: epic-test\n");
    writeFileSync(
      join(root, ".plan", "epics", "epic-test.md"),
      "- `TASK-shipped.md` — P1, proposed; a slice description.\n",
    );

    const result = runGate(root);

    expect(result.stdout).toContain("epic_status_drift=1");
    expect(result.stdout).toContain("claimed \"proposed\", header says \"done\"");
  });
});

test("does not read prose words as status claims", () => {
  withRepo((root) => {
    writeTicket(root, "TASK-open.md", "Status: done\nPriority: P1\nEpic: epic-test\n");
    writeFileSync(
      join(root, ".plan", "epics", "epic-test.md"),
      "- `TASK-open.md` — no blocked axum worker; the upgrade is proposed.\n",
    );

    const result = runGate(root);

    expect(result.stdout).toContain("epic_status_drift=0");
  });
});
