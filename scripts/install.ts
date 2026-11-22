// User-level install for honkbox: build the release binary, place it under
// $XDG_DATA_HOME/honkbox/bin, and link ~/.local/bin/honkbox to it.
// Runtime state stays in the XDG state dir; this script never writes there.
//
// Usage: bun scripts/install.ts [--prefix DIR] [--bindir DIR] [--dry-run]
//                              [--no-build] [--force]

import {
  chmodSync,
  copyFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readlinkSync,
  rmSync,
  symlinkSync,
} from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";

const ROOT = join(import.meta.dir, "..");
const BIN = process.platform === "win32" ? "honkbox.exe" : "honkbox";

export interface Layout {
  prefix: string;
  bindir: string;
  binary: string;
  link: string;
}

export interface FileOptions {
  dryRun?: boolean;
  force?: boolean;
}

/** Resolve the install layout from HOME / XDG_DATA_HOME. Pure: no process env default. */
export function defaultLayout(env: NodeJS.ProcessEnv): Layout {
  const home = env.HOME;
  if (!home || !isAbsolute(home)) {
    throw new Error("install: HOME must be set to an absolute path");
  }
  const xdgData = env.XDG_DATA_HOME;
  const prefix = xdgData && isAbsolute(xdgData)
    ? join(xdgData, "honkbox")
    : join(home, ".local/share/honkbox");
  const bindir = join(home, ".local/bin");
  return { prefix, bindir, binary: join(prefix, "bin", BIN), link: join(bindir, BIN) };
}

/** Parse the shared layout flags; extra flags are left to the caller. */
export function parseLayout(
  argv: string[],
  env: NodeJS.ProcessEnv,
): { layout: Layout; dryRun: boolean; rest: string[]; } {
  const layout = defaultLayout(env);
  const rest: string[] = [];
  let dryRun = false;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i] ?? "";
    if (arg === "--dry-run") {
      dryRun = true;
      continue;
    }
    if (arg === "--prefix" || arg === "--bindir") {
      const value = argv[++i];
      if (!value || !isAbsolute(value)) {
        throw new Error(`install: ${arg} needs an absolute path`);
      }
      if (arg === "--prefix") {
        layout.prefix = value;
        layout.binary = join(value, "bin", BIN);
      } else {
        layout.bindir = value;
        layout.link = join(value, BIN);
      }
      continue;
    }
    rest.push(arg);
  }
  return { layout, dryRun, rest };
}

/** Where cargo puts the release binary (honours CARGO_TARGET_DIR). */
export function releaseBinary(root: string, env: NodeJS.ProcessEnv): string {
  const target = env.CARGO_TARGET_DIR
    ? resolve(root, env.CARGO_TARGET_DIR)
    : join(root, "target");
  return join(target, "release", BIN);
}

/** Runtime state path, reported for visibility; never created by the installer. */
export function dataDirHint(env: NodeJS.ProcessEnv): string {
  const xdgState = env.XDG_STATE_HOME;
  if (xdgState && isAbsolute(xdgState)) {
    return join(xdgState, "honkbox");
  }
  const home = env.HOME;
  if (!home || !isAbsolute(home)) {
    throw new Error("install: HOME must be set to an absolute path");
  }
  return join(home, ".local/state/honkbox");
}

type Kind = "missing" | "link" | "other";

function pathKind(path: string): Kind {
  try {
    return lstatSync(path).isSymbolicLink() ? "link" : "other";
  } catch {
    return "missing";
  }
}

function linkTarget(path: string): string | null {
  try {
    return readlinkSync(path);
  } catch {
    return null;
  }
}

/** Copy the built binary and create the ~/.local/bin symlink. */
export function installFiles(layout: Layout, source: string, opts: FileOptions = {}): void {
  const linkKind = pathKind(layout.link);
  const linkIsForeign = linkKind !== "missing" && linkTarget(layout.link) !== layout.binary;
  const binaryIsLink = pathKind(layout.binary) === "link";

  if (opts.dryRun) {
    console.log(`[dry-run] install ${source} -> ${layout.binary}`);
    console.log(`[dry-run] link ${layout.link} -> ${layout.binary}`);
    if (linkIsForeign && !opts.force) {
      console.log(`[dry-run] would refuse: ${layout.link} points somewhere else (use --force)`);
    }
    if (binaryIsLink && !opts.force) {
      console.log(`[dry-run] would refuse: ${layout.binary} is a symlink (use --force)`);
    }
    return;
  }

  if (!existsSync(source)) {
    throw new Error(`install: release binary missing: ${source} (run without --no-build)`);
  }
  if (binaryIsLink && !opts.force) {
    throw new Error(`install: refusing to overwrite symlink ${layout.binary} (use --force)`);
  }
  if (linkIsForeign && !opts.force) {
    throw new Error(`install: refusing to replace ${layout.link} (use --force)`);
  }

  mkdirSync(dirname(layout.binary), { recursive: true });
  mkdirSync(layout.bindir, { recursive: true });
  if (binaryIsLink) {
    rmSync(layout.binary);
  }
  copyFileSync(source, layout.binary);
  chmodSync(layout.binary, 0o755);
  if (linkKind !== "missing") {
    rmSync(layout.link);
  }
  symlinkSync(layout.binary, layout.link);
}

async function run(args: string[], cwd: string): Promise<void> {
  const proc = Bun.spawn(args, { cwd, stdout: "inherit", stderr: "inherit" });
  const code = await proc.exited;
  if (code !== 0) {
    throw new Error(`install: ${args.join(" ")} failed (exit ${code})`);
  }
}

/** Same order as the CI release job: vendor sync, web assets, release binary. */
export async function buildRelease(root: string = ROOT): Promise<void> {
  const bun = process.execPath;
  await run([bun, "scripts/vendor-patches.ts", "sync"], root);
  await run([bun, "run", "build:web"], root);
  await run([bun, "run", "build:release"], root);
}

/** Remove only the link; --purge also removes the install prefix, never state. */
export function uninstallFiles(
  layout: Layout,
  opts: { dryRun?: boolean; purge?: boolean; } = {},
): void {
  const target = linkTarget(layout.link);
  if (target !== null) {
    if (target !== layout.binary) {
      throw new Error(`uninstall: ${layout.link} points elsewhere; refusing to remove`);
    }
    if (opts.dryRun) {
      console.log(`[dry-run] remove ${layout.link}`);
    } else {
      rmSync(layout.link);
    }
  }
  if (opts.purge) {
    if (opts.dryRun) {
      console.log(`[dry-run] remove ${layout.prefix}`);
    } else {
      rmSync(layout.prefix, { recursive: true, force: true });
    }
  }
}

export async function main(argv: string[] = Bun.argv.slice(2)): Promise<void> {
  if (process.platform === "win32") {
    throw new Error("install: POSIX symlinks required; Windows is not supported");
  }
  const { layout, dryRun, rest } = parseLayout(argv, process.env);
  let build = true;
  let force = false;
  for (const arg of rest) {
    if (arg === "--no-build") {
      build = false;
    } else if (arg === "--force") {
      force = true;
    } else {
      throw new Error(`install: unknown option ${arg}`);
    }
  }

  if (build && !dryRun) {
    await buildRelease();
  } else {
    console.log(build ? "[dry-run] would build the release binary" : "skipping build (--no-build)");
  }

  const source = releaseBinary(ROOT, process.env);
  installFiles(layout, source, { dryRun, force });
  if (!dryRun) {
    console.log(`installed ${layout.binary}`);
    console.log(`linked ${layout.link} -> ${layout.binary}`);
  }
  console.log(`runtime data dir: ${dataDirHint(process.env)} (override with --data-dir)`);
}

if (import.meta.main) {
  await main().catch((err: unknown) => {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  });
}
