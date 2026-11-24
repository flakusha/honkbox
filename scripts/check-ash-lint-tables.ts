#!/usr/bin/env bun
// Guard: the ash crate's local lint tables must mirror the workspace tables.
//
// Cargo rejects `lints.workspace = true` together with per-lint overrides, so
// crates/ash-backend/Cargo.toml hand-copies [workspace.lints] and flips one
// lint (`unsafe_code = "allow"`, confined by a crate-root deny). This check
// fails when that copy drifts; it compares semantic keys, not comment text.
// Exit code 0 = mirrored, 1 = drift.

import { readFileSync } from "node:fs";

const ROOT = "Cargo.toml";
const ASH = "crates/ash-backend/Cargo.toml";
const UNSAFE_CODE = new Set(["unsafe_code=\"allow\"", "unsafe_code=\"forbid\""]);

function table(text: string, header: string): string[] {
  const lines = text.split("\n");
  const start = lines.findIndex((line) => line.trim() === header);
  if (start < 0) {
    return [];
  }
  const entries: string[] = [];
  for (let i = start + 1; i < lines.length; i += 1) {
    const line = lines[i] ?? "";
    if (line.trim().startsWith("[")) {
      break;
    }
    const value = line.split("#")[0]?.trim().replace(/\s+/g, "") ?? "";
    if (value !== "") {
      entries.push(value);
    }
  }
  return entries;
}

const root = readFileSync(ROOT, "utf8");
const ash = readFileSync(ASH, "utf8");
const pairs = [
  {
    label: "rust",
    expected: table(root, "[workspace.lints.rust]"),
    actual: table(ash, "[lints.rust]"),
  },
  {
    label: "clippy",
    expected: table(root, "[workspace.lints.clippy]"),
    actual: table(ash, "[lints.clippy]"),
  },
];

const drift: string[] = [];
for (const { label, expected, actual } of pairs) {
  if (expected.length === 0 || actual.length === 0) {
    drift.push(`${label}: missing lint table`);
    continue;
  }
  const want = expected.filter((line) => !UNSAFE_CODE.has(line)).sort();
  const have = actual.filter((line) => !UNSAFE_CODE.has(line)).sort();
  if (JSON.stringify(want) !== JSON.stringify(have)) {
    drift.push(`${label}: workspace=${JSON.stringify(want)} ash=${JSON.stringify(have)}`);
  }
}

if (drift.length > 0) {
  console.error("ash lint tables drifted from workspace:");
  for (const line of drift) {
    console.error(`  ${line}`);
  }
  process.exit(1);
}

console.log("ash lint tables mirror workspace (unsafe_code exception)");
