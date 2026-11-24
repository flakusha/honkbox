#!/usr/bin/env bun
// Gate: every ash shader source declares where it came from.
//
// The design note requires a `// Provenance:` line on each shader (original or
// derived, with upstream path and pinned rev when derived). All 13 current
// shaders comply; this check stops a new source from landing without one.
// License stays in REUSE.toml — this repo has no per-file SPDX headers.
//
// Exit code 0 = all shaders annotated, 1 = missing annotation or empty source dir.

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const SHADERS_DIR = "crates/ash-backend/shaders";
const MARKER = "// Provenance:";

const sources = readdirSync(SHADERS_DIR, { withFileTypes: true })
  .filter((entry) => entry.isFile() && /\.(comp|glsl)$/.test(entry.name))
  .map((entry) => join(SHADERS_DIR, entry.name));

if (sources.length === 0) {
  console.error(`${SHADERS_DIR}: no shader sources found`);
  process.exit(1);
}

const missing = sources.filter((path) => !readFileSync(path, "utf8").includes(MARKER));
if (missing.length > 0) {
  console.error(`Missing ${MARKER} in:`);
  for (const path of missing) {
    console.error(`  ${path}`);
  }
  process.exit(1);
}

console.log(`${sources.length} ash shaders carry ${MARKER}`);
