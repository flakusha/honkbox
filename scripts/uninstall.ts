// Removes the ~/.local/bin/honkbox link created by scripts/install.ts.
// --purge also deletes the install prefix; runtime state is never touched.
//
// Usage: bun scripts/uninstall.ts [--prefix DIR] [--bindir DIR] [--dry-run] [--purge]

import { parseLayout, uninstallFiles } from "./install";

export async function main(argv: string[] = Bun.argv.slice(2)): Promise<void> {
  const { layout, dryRun, rest } = parseLayout(argv, process.env);
  let purge = false;
  for (const arg of rest) {
    if (arg === "--purge") {
      purge = true;
    } else {
      throw new Error(`uninstall: unknown option ${arg}`);
    }
  }
  uninstallFiles(layout, { dryRun, purge });
  if (!dryRun) {
    console.log(`uninstalled ${layout.link}`);
  }
}

if (import.meta.main) {
  await main().catch((err: unknown) => {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  });
}
