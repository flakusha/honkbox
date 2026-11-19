// Opt-in effective-code-line budget, adapted from ../loop-lore's
// scripts/check-file-size.ts. Blank and comment-only lines are free; a
// `// size-allow: N` header replaces the budget. Tests/generated are exempt.
// `--strict` makes overruns fail.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const DEFAULT_BUDGET = 250;
const CEILING = 300;
const HEADER_CHARS = 512;
const ROOTS = ["crates", "packages/web-ui/src", "scripts"];
const EXEMPT = /(^|\/)(gen|tests)(\/|$)/;
const ALLOW = /^\/\/\s*size-allow:\s*(\d+)\s*$/m;

function markComments(text: string, rust: boolean): Uint8Array {
  const marked = new Uint8Array(text.length);
  let i = 0, blockDepth = 0, quote = "", rawHashes = -1;
  while (i < text.length) {
    if (blockDepth > 0) {
      if (text.startsWith("/*", i)) {
        blockDepth++;
        i += 2;
        continue;
      }

      if (text.startsWith("*/", i)) {
        blockDepth--;
        i += 2;
        continue;
      }

      marked[i] = 1;
      i++;
      continue;
    }

    if (rawHashes >= 0) {
      const close = `"${"#".repeat(rawHashes)}`;
      if (text.startsWith(close, i)) {
        i += close.length;
        rawHashes = -1;
        continue;
      }

      i++;
      continue;
    }

    if (quote) {
      if (text[i] === "\\") {
        i += 2;
        continue;
      }

      if (text[i] === quote) quote = "";
      i++;
      continue;
    }

    if (text.startsWith("//", i)) {
      const end = text.indexOf("\n", i);
      marked.fill(1, i, end < 0 ? text.length : end);
      i = end < 0 ? text.length : end;
      continue;
    }

    if (text.startsWith("/*", i)) {
      blockDepth = 1;
      i += 2;
      continue;
    }

    const raw = /^r(#*)"/.exec(text.slice(i));
    if (raw?.[0]) {
      rawHashes = raw[1]!.length;
      i += raw[0].length;
      continue;
    }

    const ch = text[i]!;
    // Rust lifetimes (`'a`) are not string literals; treating them as quotes
    // would swallow later `//` comments and over-count code lines.
    if (ch === "\"" || ch === "`" || (!rust && ch === "'")) quote = ch;
    i++;
  }

  return marked;
}

function codeLines(text: string, rust: boolean): number {
  const marked = markComments(text, rust);
  let code = false;
  let total = 0;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "\n") {
      if (code) total++;
      code = false;
      continue;
    }

    if (/\s/.test(text[i]!)) continue;
    if (!marked[i]) code = true;
  }

  if (code) total++;
  return total;
}

function files(dir: string): string[] {
  let entries: string[];
  try {
    entries = readdirSync(dir, { withFileTypes: true }).map((e) => join(dir, e.name));
  } catch {
    return [];
  }

  return entries.flatMap((entry) => /\.(rs|ts|tsx|js|mjs)$/.test(entry) ? [entry] : files(entry));
}

const strict = process.argv.includes("--strict");
const offenders: string[] = [];
let scanned = 0;
for (const root of ROOTS) {
  for (const file of files(root)) {
    if (EXEMPT.test(file) || /\.test\.tsx?$/.test(file)) continue;
    const text = readFileSync(file, "utf8");
    if (text.includes("DO NOT EDIT MANUALLY")) continue;
    scanned++;
    const code = codeLines(text, file.endsWith(".rs"));
    const allow = ALLOW.exec(text.slice(0, HEADER_CHARS))?.[1];
    const budget = allow ? Number(allow) : code > DEFAULT_BUDGET ? CEILING : DEFAULT_BUDGET;
    if (code > budget) offenders.push(`${code} > ${budget}  ${file}`);
  }
}

console.log(`scanned ${scanned} files; ${offenders.length} over budget`);
for (const line of offenders.sort()) console.log(line);
if (strict && offenders.length > 0) process.exit(1);
