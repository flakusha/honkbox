// Dynamic fetch + patch stack for vendor/diffusion-rs.
//
// vendor/ is gitignored scratch, never committed. `sync` converges it from
// the pins in vendor.manifest.toml: shallow-fetch the diffusion-rs checkout
// at [vendor].rev, materialize each [[vendor.nested]] copy inside it at its
// own rev (upstream tracks them as git submodules; the fetch lands them as
// plain files so the tree builds with no submodule step), prune the listed
// dirs, then apply vendor_patches/ in manifest order.
//
// Idempotence model (mirrors custom_nodes_deps.py minus its branch tracking:
// every source here is a detached rev, never a branch): per managed repo,
// fetch only when .git is missing or HEAD != rev; a rev move under a dirty
// repo is refused loudly, never discarded. Per patch: already-applied
// (reverse --check passes) skips, else forward --check then apply; anything
// else is a loud refusal naming the patch. Re-runnable and self-healing:
// rm -rf vendor/ + sync always converges from empty.
//
// --dry-run performs NO network and NO mutation: manifest parse + patch
// files exist + per-repo would-fetch/would-move + per-patch
// would-apply/already-applied, exiting non-zero when sync would change or
// refuse anything. Escape hatch: rm -rf vendor/ + sync (no --force flag).
//
// Network-free after the first fetch: a converged tree needs no network.
// Usage: bun scripts/vendor-patches.ts sync [--dry-run]

import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";

const ROOT = join(import.meta.dir, "..");
const MANIFEST_NAME = "vendor.manifest.toml";
const PATCHES_DIR = "vendor_patches";
const VENDOR_DIR = "vendor";
const TOP_NAME = "diffusion-rs";
const NODE = "vendor";

// spawnSync env wants defined values; absent entries are simply inherited.
const BASE_ENV = process.env as Record<string, string>;

interface Nested {
  path: string;
  url: string;
  rev: string;
  prune: string[];
}

interface Manifest {
  url: string;
  rev: string;
  patches: string[];
  nested: Nested[];
}

interface GitResult {
  rc: number;
  out: string;
  err: string;
}

function git(cwd: string, args: string[]): GitResult {
  const proc = Bun.spawnSync(["git", ...args], {
    cwd,
    env: BASE_ENV,
    stdout: "pipe",
    stderr: "pipe",
  });
  return {
    rc: proc.exitCode,
    out: new TextDecoder().decode(proc.stdout).trim(),
    err: new TextDecoder().decode(proc.stderr).trim(),
  };
}

function fail(message: string): never {
  console.error(`${NODE}: ERROR: ${message}`);
  process.exit(1);
}

function isFullSha(s: string): boolean {
  return /^[0-9a-f]{40}$/.test(s);
}

function parseNested(value: unknown): Nested[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error("manifest: [vendor].nested must be a list");
  return value.map((entry: unknown, i: number) => {
    const where = `manifest: [vendor].nested[${i}]`;
    if (typeof entry !== "object" || entry === null) throw new Error(`${where} must be a table`);
    const { path, url, rev, prune } = entry as Record<string, unknown>;
    if (typeof path !== "string" || !path.startsWith(`${TOP_NAME}/`)) {
      throw new Error(`${where}.path must start with "${TOP_NAME}/"`);
    }
    if (typeof url !== "string" || url === "") throw new Error(`${where}.url must be a string`);
    if (typeof rev !== "string" || !isFullSha(rev)) {
      throw new Error(`${where}.rev must be a full commit SHA`);
    }
    if (prune !== undefined && (!Array.isArray(prune) || !prune.every((p) => typeof p === "string"))) {
      throw new Error(`${where}.prune must be a list of strings`);
    }
    return { path, url, rev, prune: (prune as string[] | undefined) ?? [] };
  });
}

function parseManifest(text: string): Manifest {
  let data: unknown;
  try {
    data = Bun.TOML.parse(text);
  } catch (error) {
    throw new Error(`manifest: cannot parse ${MANIFEST_NAME}`, { cause: error });
  }
  if (typeof data !== "object" || data === null || !("vendor" in data)) {
    throw new Error("manifest: missing [vendor] table");
  }
  const table: unknown = data.vendor;
  if (typeof table !== "object" || table === null) throw new Error("manifest: [vendor] must be a table");
  for (const key of Object.keys(table)) {
    if (key !== "url" && key !== "rev" && key !== "patches" && key !== "nested") {
      throw new Error(`manifest: unknown [vendor] key: ${key}`);
    }
  }
  const { url, rev, patches, nested } = table as Record<string, unknown>;
  if (typeof url !== "string" || url === "") throw new Error("manifest: [vendor].url must be a string");
  if (typeof rev !== "string" || !isFullSha(rev)) {
    throw new Error("manifest: [vendor].rev must be a full commit SHA");
  }
  if (!Array.isArray(patches) || !patches.every((p: unknown) => typeof p === "string")) {
    throw new Error("manifest: [vendor].patches must be a list of strings");
  }
  return { url, rev, patches: [...patches], nested: parseNested(nested) };
}

function readLines(path: string): string[] {
  const full = path.startsWith("/") ? path : join(ROOT, path);
  return readFileSync(full, "utf8").split("\n");
}

// Repo-relative paths a patch stack touches, from its ---/+++ headers.
// Handles git-style (a/ b/) and traditional unified diffs; the /dev/null
// side of a create/delete names no repo path and is skipped.
function patchTouched(path: string): string[] {
  const touched = new Set<string>();
  for (const line of readLines(path)) {
    if (line === "--- /dev/null" || line === "+++ /dev/null") continue;
    if (line.startsWith("--- ")) touched.add(line.slice(4).replace(/^(a\/|b\/)/, "").trim());
    else if (line.startsWith("+++ ")) touched.add(line.slice(4).replace(/^(a\/|b\/)/, "").trim());
  }
  return [...touched];
}

interface Repo {
  // dir relative to ROOT, e.g. vendor/diffusion-rs
  dir: string;
  url: string;
  rev: string;
  prune: string[];
  top: boolean;
}

// Every status entry must be explained: patch-touched files, or (nested
// only) deletions under a prune dir. Anything else is user dirt.
function dirtRefusal(repo: Repo, allowed: Set<string>): string | null {
  // core.quotePath=false: keep spaces/UTF-8 literal so prune-prefix matching sees real paths.
  const status = git(join(ROOT, repo.dir), [
    "-c",
    "core.quotePath=false",
    "status",
    "--porcelain",
    "--ignore-submodules=all",
  ]);
  if (status.rc !== 0) return `${repo.dir}: cannot read worktree status: ${status.err || status.out}`;
  const bad: string[] = [];
  for (const line of status.out.split("\n")) {
    if (line === "") continue;
    // Staged-only entries print as `X<space>path` (e.g. `D docs/gguf.md`
    // left by a prune rm); everything else is `XY<space>path`.
    const stagedOnly = line[0] !== " " && line[0] !== "?";
    const code = line.slice(0, 2);
    let p = stagedOnly ? line.slice(2) : line.slice(3);
    const arrow = p.indexOf(" -> ");
    if (arrow !== -1) p = p.slice(arrow + 4);
    // Porcelain C-quotes paths with spaces (quotePath only covers non-ASCII).
    if (p.startsWith('"') && p.endsWith('"')) {
      p = p.slice(1, -1).replace(/\\\\/g, "\\").replace(/\\"/g, '"');
    }
    // Patch paths are ROOT-relative; status paths are repo-relative.
    if (allowed.has(join(repo.dir, p))) continue;
    if (!repo.top && (code === " D" || code === "D ") && repo.prune.some((d) => p.startsWith(`${d}/`))) continue;
    bad.push(p);
  }
  if (bad.length === 0) return null;
  const shown = bad.slice(0, 5).map((p) => JSON.stringify(p)).join(", ");
  const more = bad.length > 5 ? ` (+${bad.length - 5} more)` : "";
  return `${repo.dir}: dirty worktree not explained by the patch stack: ${shown}${more}; revert or rm -rf ${VENDOR_DIR}/ + sync`;
}

function pruneRepo(repo: Repo): void {
  for (const rel of repo.prune) {
    rmSync(join(ROOT, repo.dir, rel), { recursive: true, force: true });
  }
}

// Converge one managed repo to its pinned rev. Returns "same" | "moved" |
// "created", or "would-fetch"/"would-move" under --dry-run (no mutation).
function ensureRepo(repo: Repo, dryRun: boolean): string {
  const dir = join(ROOT, repo.dir);
  if (!existsSync(join(dir, ".git"))) {
    if (dryRun) return "would-fetch";
    mkdirSync(dir, { recursive: true });
    let result = git(dir, ["init", "-q", "-b", "main"]);
    if (result.rc !== 0) fail(`cannot init ${repo.dir}: ${result.err || result.out}`);
    result = git(dir, ["remote", "add", "origin", repo.url]);
    if (result.rc !== 0) {
      rmSync(dir, { recursive: true, force: true });
      fail(`cannot set origin for ${repo.dir}: ${result.err || result.out}`);
    }
    result = git(dir, ["fetch", "--depth", "1", "origin", repo.rev]);
    if (result.rc !== 0) {
      rmSync(dir, { recursive: true, force: true });
      fail(`cannot fetch ${repo.url}@${repo.rev.slice(0, 12)} (network?): ${result.err || result.out}`);
    }
    result = git(dir, ["checkout", "-q", repo.rev]);
    if (result.rc !== 0) {
      rmSync(dir, { recursive: true, force: true });
      fail(`cannot check out ${repo.rev} in ${repo.dir}: ${result.err || result.out}`);
    }
    pruneRepo(repo);
    return "created";
  }
  const head = git(dir, ["rev-parse", "HEAD"]);
  if (head.rc !== 0 || head.out === "") {
    fail(`${repo.dir}: no HEAD (interrupted fetch?); rm -rf ${VENDOR_DIR}/ + sync to recover`);
  }
  if (head.out === repo.rev) return "same";
  if (dryRun) return "would-move";
  return "moved";
}

const argv = Bun.argv.slice(2);
const command = argv[0];
const dryRun = argv.includes("--dry-run");
if (command !== "sync" || argv.some((a) => a !== "sync" && a !== "--dry-run")) {
  console.error("usage: bun scripts/vendor-patches.ts sync [--dry-run]");
  process.exit(2);
}

const manifestPath = join(ROOT, MANIFEST_NAME);
if (!existsSync(manifestPath)) fail(`manifest not found: ${MANIFEST_NAME}`);
let manifest: Manifest;
try {
  manifest = parseManifest(readFileSync(manifestPath, "utf8"));
} catch (error) {
  fail(error instanceof Error ? error.message : String(error));
}

// Entries are relative to vendor_patches/; an absolute path is used literally.
const files = manifest.patches.map((n) => (n.startsWith("/") ? n : join(PATCHES_DIR, n)));
const missing = files.find((f) => !existsSync(f.startsWith("/") ? f : join(ROOT, f)));
if (missing !== undefined) fail(`declared patch file not found: ${missing}`);

const repos: Repo[] = [
  { dir: join(VENDOR_DIR, TOP_NAME), url: manifest.url, rev: manifest.rev, prune: [], top: true },
  ...manifest.nested.map((n) => ({ dir: join(VENDOR_DIR, n.path), url: n.url, rev: n.rev, prune: n.prune, top: false })),
];

const allowed = new Set(files.flatMap(patchTouched));

if (dryRun) {
  let changes = 0;
  for (const repo of repos) {
    const dir = join(ROOT, repo.dir);
    if (!existsSync(join(dir, ".git"))) {
      console.log(`${NODE}: would fetch ${repo.url}@${repo.rev} (+${repo.prune.length} prune)`);
      changes++;
      continue;
    }
    const head = git(dir, ["rev-parse", "HEAD"]);
    if (head.rc !== 0 || head.out === "") fail(`${repo.dir}: no HEAD; rm -rf ${VENDOR_DIR}/ + sync to recover`);
    if (head.out !== repo.rev) {
      console.log(`${NODE}: would move ${repo.dir} ${head.out.slice(0, 12)}..${repo.rev.slice(0, 12)}`);
      changes++;
      continue;
    }
    const refusal = dirtRefusal(repo, allowed);
    if (refusal !== null) fail(refusal);
  }
  if (!existsSync(join(ROOT, repos[0].dir, ".git"))) {
    console.log(`${NODE}: would apply ${files.length} patch(es) after fetch`);
    process.exit(1);
  }
  const done = appliedPrefix();
  let pending = 0;
  for (const [i, file] of files.entries()) {
    if (i < done) {
      console.log(`${NODE}: already applied: ${file}`);
    } else if (git(ROOT, ["apply", "--check", "--", file]).rc === 0) {
      console.log(`${NODE}: would apply: ${file}`);
      pending++;
    } else {
      fail(`patch does not apply (and is not applied): ${file}; rm -rf ${VENDOR_DIR}/ + sync, then report`);
    }
  }
  if (changes > 0 || pending > 0) {
    console.log(`${NODE}: dry-run: sync would change the tree`);
    process.exit(1);
  }
  console.log(`${NODE}: dry-run: tree is converged`);
  process.exit(0);
}

for (const repo of repos) {
  const state = ensureRepo(repo, false);
  const dir = join(ROOT, repo.dir);
  if (state === "moved") {
    const refusal = dirtRefusal(repo, allowed);
    if (refusal !== null) {
      const at = git(dir, ["rev-parse", "HEAD"]).out.slice(0, 12);
      fail(`${repo.dir} at ${at} has local changes; move to ${repo.rev.slice(0, 12)} refused: ${refusal}`);
    }
    const result = git(dir, ["fetch", "--depth", "1", "origin", repo.rev]);
    if (result.rc !== 0) fail(`cannot fetch ${repo.url}@${repo.rev.slice(0, 12)} (network?): ${result.err || result.out}`);
    const checkout = git(dir, ["checkout", "-q", repo.rev]);
    if (checkout.rc !== 0) fail(`cannot check out ${repo.rev} in ${repo.dir}: ${checkout.err || checkout.out}`);
    pruneRepo(repo);
    console.log(`${NODE}: moved ${repo.dir} to ${repo.rev.slice(0, 12)}`);
  } else if (state === "created") {
    console.log(`${NODE}: fetched ${repo.dir} @ ${repo.rev.slice(0, 12)}`);
  }
}

// Prune every run (idempotent): a restored pruned dir must not linger.
for (const repo of repos) pruneRepo(repo);

for (const repo of repos) {
  const refusal = dirtRefusal(repo, allowed);
  if (refusal !== null) fail(refusal);
}

// Largest applied prefix, scanned top-down: an earlier patch's reverse
// context is shadowed once a later patch touches the same file, so a
// bottom-up "is it applied?" check misfires on overlapping stacks.
// Stacks land in order, so a reverse-applying patch implies its predecessors.
function appliedPrefix(): number {
  for (let i = files.length - 1; i >= 0; i--) {
    if (git(ROOT, ["apply", "--check", "--reverse", "--", files[i]]).rc === 0) return i + 1;
  }
  return 0;
}
const appliedThisRun: string[] = [];
const skipped = appliedPrefix();
for (const file of files.slice(skipped)) {
  const check = git(ROOT, ["apply", "--check", "--", file]);
  if (check.rc !== 0) {
    for (const done of [...appliedThisRun].reverse()) {
      git(ROOT, ["apply", "--reverse", "--", done]);
    }
    fail(`patch failed to apply: ${file}: ${check.err || check.out}; stack reverted`);
  }
  const result = git(ROOT, ["apply", "--", file]);
  if (result.rc !== 0) {
    for (const done of [...appliedThisRun].reverse()) {
      git(ROOT, ["apply", "--reverse", "--", done]);
    }
    fail(`patch failed to apply: ${file}: ${result.err || result.out}; stack reverted`);
  }
  appliedThisRun.push(file);
}
console.log(`${NODE}: patches already applied (${skipped}), applied now (${appliedThisRun.length}), total ${files.length}`);
