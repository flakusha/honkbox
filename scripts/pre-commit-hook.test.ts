import { afterEach, expect, test } from "bun:test";
import * as fsSync from "node:fs";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

// Exercises the real .githooks/pre-commit script against disposable git repos.
// All tool invocations are per-test shims; real dprint/oxlint/cargo are never
// called by these tests.

// TODO(env): bun-only - add an executor adapter when a node test target lands
// Named Bun-only APIs: bun:test, Bun.spawnSync, Bun.which, import.meta.dir.

const REPO_DIR = path.resolve(import.meta.dir, "..");
const HOOK_PATH = path.join(REPO_DIR, ".githooks", "pre-commit");
const TMP_PREFIX = path.join(os.tmpdir(), "honkbox-hook-");
const ORIGINAL_PATH = process.env.PATH ?? "";

// The hook still needs these utilities in tests that remove the PATH entries
// containing bunx or cargo, so they are symlinked by absolute path.
const SYSTEM_TOOLS = [
  "awk",
  "basename",
  "bash",
  "cat",
  "chmod",
  "cp",
  "cut",
  "date",
  "diff",
  "dirname",
  "env",
  "find",
  "git",
  "grep",
  "head",
  "ln",
  "mkdir",
  "mktemp",
  "mv",
  "rm",
  "sed",
  "sh",
  "sort",
  "tail",
  "tee",
  "tr",
  "wc",
];

type Result = { code: number | null; out: string; };
type DprintMode = "pass" | "fail" | "rewrite";
type LintMode = "pass" | "fail";
type CargoMode = "pass" | "fail";

let activeTmpDir: string | null = null;

// Single-quote escaping for generated shell literals (a literal ' -> '\'').
function shellQuote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

function isExecutable(file: string): boolean {
  try {
    fsSync.accessSync(file, fsSync.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

// Used only by the "tool unavailable" tests: drop PATH entries that really
// contain the tool so `command -v` cannot find it in the ambient environment.
function withoutTool(original: string, tool: string): string {
  return original
    .split(path.delimiter)
    .filter((entry) => entry.length > 0 && !isExecutable(path.join(entry, tool)))
    .join(path.delimiter);
}

type Harness = {
  dir: string;
  bin: string;
  env: Record<string, string | undefined>;
  git: (args: string[]) => Result;
  hook: () => Result;
  stage: (rel: string, content: string) => Promise<void>;
  commit: () => Result;
  calls: () => Promise<string>;
  bunx: (dprint: DprintMode, oxlint: LintMode) => Promise<void>;
  cargo: (mode: CargoMode) => Promise<void>;
};

async function setup(): Promise<Harness> {
  const dir = await fs.mkdtemp(TMP_PREFIX);
  activeTmpDir = dir;

  const bin = path.join(dir, "bin");
  await fs.mkdir(bin);
  for (const tool of SYSTEM_TOOLS) {
    const source = Bun.which(tool);
    if (source) {
      await fs.symlink(source, path.join(bin, tool));
    }
  }

  const gitConfig = path.join(dir, "gitconfig");
  await fs.writeFile(
    gitConfig,
    "[user]\n\tname = Hook Test\n\temail = hook@example.invalid\n[commit]\n\tgpgSign = false\n",
  );

  const env = {
    ...process.env,
    GIT_CONFIG_GLOBAL: gitConfig,
    GIT_CONFIG_SYSTEM: "/dev/null",
    GIT_CONFIG_NOSYSTEM: "1",
    PATH: `${bin}${path.delimiter}${ORIGINAL_PATH}`,
  };

  function run(args: string[]): Result {
    const res = Bun.spawnSync(args, { cwd: dir, env });
    const stdout = res.stdout ? new TextDecoder().decode(res.stdout) : "";
    const stderr = res.stderr ? new TextDecoder().decode(res.stderr) : "";
    return { code: res.exitCode, out: `${stdout}${stderr}` };
  }

  const log = path.join(bin, "calls.log");

  async function writeShim(name: string, body: string): Promise<void> {
    const script = `#!/bin/sh\nprintf '%s\\n' "${name} $*" >> ${shellQuote(log)}\n${body}\n`;
    const file = path.join(bin, name);
    await fs.writeFile(file, script);
    await fs.chmod(file, 0o755);
  }

  async function bunx(dprint: DprintMode, oxlint: LintMode): Promise<void> {
    const dprintBody = {
      pass: "exit 0",
      fail: "printf '%s\\n' 'dprint: synthetic finding' >&2\nexit 1",
      rewrite: [
        "shift",
        "for f in \"$@\"; do",
        "  [ \"$f\" = fmt ] && continue",
        "  [ \"$f\" = --allow-no-files ] && continue",
        "  [ \"$f\" = -- ] && continue",
        "  printf '%s\\n' '// formatted by hook test' > \"$f\"",
        "done",
        "exit 0",
      ].join("\n"),
    }[dprint];
    const oxlintBody = {
      pass: "exit 0",
      fail: "printf '%s\\n' 'oxlint: synthetic finding' >&2\nexit 1",
    }[oxlint];
    await writeShim(
      "bunx",
      `case "$1" in\ndprint)\n${dprintBody} ;;\noxlint)\n${oxlintBody} ;;\nesac\nexit 0\n`,
    );
  }

  async function cargo(mode: CargoMode): Promise<void> {
    const body = mode === "fail"
      ? [
        "if [ \"$1\" = fmt ] && [ \"$2\" = --all ] && [ \"$3\" = -- ] && [ \"$4\" = --check ]; then",
        "  printf '%s\\n' 'cargo fmt: synthetic finding' >&2",
        "  exit 1",
        "fi",
      ].join("\n")
      : "";
    await writeShim("cargo", `${body}\nexit 0\n`);
  }

  async function stage(rel: string, content: string): Promise<void> {
    const abs = path.join(dir, rel);
    await fs.mkdir(path.dirname(abs), { recursive: true });
    await fs.writeFile(abs, content);
    expect(run(["git", "add", "--", rel]).code).toBe(0);
  }

  async function calls(): Promise<string> {
    try {
      return await fs.readFile(log, "utf8");
    } catch {
      return "";
    }
  }

  expect(run(["git", "init", "-q"]).code).toBe(0);

  return {
    dir,
    bin,
    env,
    git: run,
    hook: () => run(["sh", HOOK_PATH]),
    stage,
    commit: () => run(["git", "commit", "-q", "-m", "base"]),
    calls,
    bunx,
    cargo,
  };
}

afterEach(async () => {
  const dir = activeTmpDir;
  activeTmpDir = null;
  if (dir && dir.startsWith(TMP_PREFIX) && dir !== os.tmpdir()) {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test("no staged files skips every file gate", async () => {
  const h = await setup();
  const result = h.hook();
  expect(result.code).toBe(0);
  expect(result.out).toContain("No staged files for file gates - skipping");
  expect(await h.calls()).toBe("");
});

test("dprint failure on a staged ts file reports and blocks", async () => {
  const h = await setup();
  await h.bunx("fail", "pass");
  await h.stage("a.ts", "export const value = 1;\n");

  const result = h.hook();
  expect(result.code).toBe(1);
  expect(result.out).toContain("FAIL: dprint fmt - report:");
  expect(result.out).toContain("1 check(s) failed - commit blocked");
});

test("dprint success rewrites the staged file and restages it", async () => {
  const h = await setup();
  await h.bunx("rewrite", "pass");
  await h.stage("a.ts", "export const value = 1;\n");

  const result = h.hook();
  expect(result.code).toBe(0);
  expect(result.out).toContain("PASS: dprint fmt (restaged)");

  const staged = h.git(["git", "diff", "--cached", "--", "a.ts"]);
  expect(staged.code).toBe(0);
  expect(staged.out).toContain("// formatted by hook test");
});

test("oxlint failure reports and blocks", async () => {
  const h = await setup();
  await h.bunx("pass", "fail");
  await h.stage("a.ts", "export const value = 1;\n");

  const result = h.hook();
  expect(result.code).toBe(1);
  expect(result.out).toContain("FAIL: oxlint - report:");
});

test("cargo fmt failure reports and receives the exact check argv", async () => {
  const h = await setup();
  await h.bunx("pass", "pass");
  await h.cargo("fail");
  await h.stage("c.rs", "fn main() {}\n");

  const result = h.hook();
  expect(result.code).toBe(1);
  expect(result.out).toContain("FAIL: cargo fmt - report:");
  expect((await h.calls()).split("\n")).toContain("cargo fmt --all -- --check");
});

test("staged shell syntax error is reported by sh -n", async () => {
  const h = await setup();
  await h.bunx("pass", "pass");
  await h.cargo("pass");
  await h.stage("broken.sh", "if true; then\n  echo hi\n");

  const result = h.hook();
  expect(result.code).toBe(1);
  expect(result.out).toContain("FAIL: sh -n syntax error in broken.sh");
  expect(result.out).toContain("1 check(s) failed - commit blocked");
});

test("missing bunx reports the tool guard instead of running a finder", async () => {
  const h = await setup();
  await h.cargo("pass");
  h.env.PATH = `${h.bin}${path.delimiter}${withoutTool(ORIGINAL_PATH, "bunx")}`;
  await h.stage("a.ts", "export const value = 1;\n");

  const result = h.hook();
  expect(result.code).toBe(1);
  expect(result.out).toContain("FAIL: tool unavailable: bunx - run 'bun install'");
  expect(await h.calls()).not.toContain("dprint");
  expect(await h.calls()).not.toContain("oxlint");
});

test("missing cargo reports the tool guard instead of running a finder", async () => {
  const h = await setup();
  await h.bunx("pass", "pass");
  h.env.PATH = `${h.bin}${path.delimiter}${withoutTool(ORIGINAL_PATH, "cargo")}`;
  await h.stage("c.rs", "fn main() {}\n");

  const result = h.hook();
  expect(result.code).toBe(1);
  expect(result.out).toContain("FAIL: tool unavailable: cargo - install the Rust toolchain");
  expect(await h.calls()).not.toContain("cargo fmt");
});

test("staged shell file with spaces in its name passes sh -n", async () => {
  const h = await setup();
  await h.bunx("pass", "pass");
  await h.cargo("pass");
  await h.stage("my script.sh", "#!/bin/sh\necho hi\n");

  const result = h.hook();
  expect(result.code).toBe(0);
  expect(result.out).toContain("PASS: sh -n");
});

test("rename with an edit is still gated by the cargo fmt check", async () => {
  const h = await setup();
  await h.bunx("pass", "pass");
  await h.cargo("fail");
  await h.stage("old.rs", "fn main() {}\n");
  expect(h.git(["git", "mv", "old.rs", "new.rs"]).code).toBe(0);
  await h.stage("new.rs", "fn main() { let changed = true; }\n");

  const result = h.hook();
  expect(result.code).toBe(1);
  expect(result.out).toContain("FAIL: cargo fmt - report:");
  expect((await h.calls()).split("\n")).toContain("cargo fmt --all -- --check");
});

test("staged generated files are rejected", async () => {
  const h = await setup();
  await h.bunx("pass", "pass");
  await h.stage("packages/web-ui/src/gen/api-types.ts", "// hand edit\n");

  const result = h.hook();
  expect(result.code).toBe(1);
  expect(result.out).toContain(
    "FAIL: generated files staged under packages/web-ui/src/gen/ - run codegen, never hand-edit",
  );
});

test("staged generated non-code file blocks despite skip branch", async () => {
  const h = await setup();
  await h.bunx("pass", "pass");
  await h.stage("packages/web-ui/src/gen/blob.bin", "binary-ish\n");

  const result = h.hook();
  expect(result.code).toBe(1);
  expect(result.out).toContain("FAIL: generated files staged under packages/web-ui/src/gen/");
  expect(result.out).toContain("1 check(s) failed - commit blocked");
});

test("dprint shim receives operands after the -- separator", async () => {
  const h = await setup();
  await h.bunx("rewrite", "pass");
  await h.stage("-weird.ts", "export const value = 1;\n");

  const result = h.hook();
  expect(result.code).toBe(0);
  expect(result.out).toContain("PASS: dprint fmt (restaged)");
  expect(await h.calls()).toContain("-- -weird.ts");
});

test("staged deletion of a non-generated .rs file passes", async () => {
  const h = await setup();
  await h.bunx("pass", "pass");
  await h.cargo("pass");
  await h.stage("plain.rs", "fn main() {}\n");
  expect(h.commit().code).toBe(0);
  expect(h.git(["git", "rm", "--", "plain.rs"]).code).toBe(0);

  const result = h.hook();
  expect(result.code).toBe(0);
  expect(result.out).toContain("No staged files for file gates - skipping");
});

test("staged deletion of a generated .ts file blocks", async () => {
  const h = await setup();
  await h.bunx("pass", "pass");
  await h.stage("packages/web-ui/src/gen/api.ts", "// generated\n");
  expect(h.commit().code).toBe(0);
  expect(
    h.git(["git", "rm", "--", "packages/web-ui/src/gen/api.ts"]).code,
  ).toBe(0);

  const result = h.hook();
  expect(result.code).toBe(1);
  expect(result.out).toContain(
    "FAIL: staged deletion of generated file under packages/web-ui/src/gen/",
  );
  expect(result.out).toContain("1 check(s) failed - commit blocked");
});

test("independent failures are counted separately", async () => {
  const h = await setup();
  await h.bunx("fail", "pass");
  await h.cargo("pass");
  await h.stage("a.ts", "export const value = 1;\n");
  await h.stage("broken.sh", "if true; then\n  echo hi\n");

  const result = h.hook();
  expect(result.code).toBe(1);
  expect(result.out).toContain("FAIL: dprint fmt - report:");
  expect(result.out).toContain("FAIL: sh -n syntax error in broken.sh");
  expect(result.out).toContain("2 check(s) failed - commit blocked");
});
