// Installer filesystem tests: temp dirs only, never touch the real HOME.
import { afterEach, expect, test } from "bun:test";
import {
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readlinkSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { defaultLayout, installFiles, releaseBinary, uninstallFiles } from "./install";
import type { Layout } from "./install";

const BIN = process.platform === "win32" ? "honkbox.exe" : "honkbox";
const temps: string[] = [];

function sandbox(): { root: string; source: string; layout: Layout; } {
  const root = mkdtempSync(join(tmpdir(), "honkbox-install-"));
  temps.push(root);
  const build = join(root, "build");
  mkdirSync(build, { recursive: true });
  const source = join(build, BIN);
  writeFileSync(source, "#!/bin/sh\n");
  const prefix = join(root, "share/honkbox");
  const bindir = join(root, "bin");
  return {
    root,
    source,
    layout: { prefix, bindir, binary: join(prefix, "bin", BIN), link: join(bindir, BIN) },
  };
}

afterEach(() => {
  for (const dir of temps.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("default layout honours XDG_DATA_HOME", () => {
  const env = { HOME: "/home/test", XDG_DATA_HOME: "/home/test/xdg" };
  const layout = defaultLayout(env);
  expect(layout.prefix).toBe(join("/home/test/xdg", "honkbox"));
  expect(layout.bindir).toBe(join("/home/test", ".local/bin"));
  expect(layout.link).toBe(join("/home/test/.local/bin", BIN));
  expect(defaultLayout({ HOME: "/home/test" }).prefix).toBe(
    join("/home/test/.local/share/honkbox"),
  );
});

test("release binary path honours CARGO_TARGET_DIR", () => {
  const custom = join(tmpdir(), "honkbox-target");
  expect(releaseBinary("/repo", { CARGO_TARGET_DIR: custom })).toBe(
    join(custom, "release", BIN),
  );
  expect(releaseBinary("/repo", {})).toBe(join("/repo", "target", "release", BIN));
});

test("install copies the binary and links it into bindir", async () => {
  const { source, layout } = sandbox();
  installFiles(layout, source);
  expect(await Bun.file(layout.binary).text()).toBe("#!/bin/sh\n");
  expect(lstatSync(layout.binary).mode & 0o111).toBe(0o111);
  expect(readlinkSync(layout.link)).toBe(layout.binary);
});

test("dry run prints actions without touching disk", () => {
  const { source, layout } = sandbox();
  installFiles(layout, source, { dryRun: true });
  expect(existsSync(layout.binary)).toBe(false);
  expect(existsSync(layout.link)).toBe(false);
});

test("refuses to replace a foreign link without force", () => {
  const { root, source, layout } = sandbox();
  mkdirSync(layout.bindir, { recursive: true });
  symlinkSync(join(root, "elsewhere"), layout.link);
  expect(() => installFiles(layout, source)).toThrow(/refusing to replace/);
  installFiles(layout, source, { force: true });
  expect(readlinkSync(layout.link)).toBe(layout.binary);
});

test("uninstall removes only the link; purge removes the prefix, not state", () => {
  const { root, source, layout } = sandbox();
  installFiles(layout, source);
  const state = join(root, "state/honkbox");
  mkdirSync(state, { recursive: true });

  uninstallFiles(layout);
  expect(existsSync(layout.link)).toBe(false);
  expect(existsSync(layout.binary)).toBe(true);
  expect(existsSync(state)).toBe(true);

  uninstallFiles(layout, { purge: true });
  expect(existsSync(layout.prefix)).toBe(false);
  expect(existsSync(state)).toBe(true);
});

test("uninstall refuses a link pointing elsewhere", () => {
  const { root, layout } = sandbox();
  mkdirSync(layout.bindir, { recursive: true });
  const foreign = join(root, "other");
  symlinkSync(foreign, layout.link);
  expect(() => uninstallFiles(layout)).toThrow(/points elsewhere/);
  expect(readlinkSync(layout.link)).toBe(foreign);
});

function copyFallback(
  layout: Layout,
  source: string,
  warnings: string[] = [],
): void {
  installFiles(layout, source, {
    fallbackToCopy: true,
    link: () => {
      throw new Error("EPERM");
    },
    warn: (message) => {
      warnings.push(message);
    },
  });
}

test("symlink failure warns and copies the command on Windows", async () => {
  const { source, layout } = sandbox();
  const warnings: string[] = [];
  copyFallback(layout, source, warnings);
  expect(warnings.join(" ")).toMatch(/symlink failed.*copying/);
  expect(lstatSync(layout.link).isSymbolicLink()).toBe(false);
  expect(await Bun.file(layout.link).text()).toBe("#!/bin/sh\n");
  expect(existsSync(join(dirname(layout.binary), ".honkbox-link-copy"))).toBe(true);
});

test("uninstall removes a marker-managed copy", () => {
  const { source, layout } = sandbox();
  copyFallback(layout, source);
  uninstallFiles(layout);
  expect(existsSync(layout.link)).toBe(false);
  expect(existsSync(layout.binary)).toBe(true);
});

test("uninstall refuses an unmanaged regular file at the link path", () => {
  const { layout } = sandbox();
  mkdirSync(layout.bindir, { recursive: true });
  writeFileSync(layout.link, "unrelated");
  expect(() => uninstallFiles(layout)).toThrow(/not an installer-managed copy/);
});
