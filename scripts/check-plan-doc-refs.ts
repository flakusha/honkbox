#!/usr/bin/env bun
// Gate: every `.plan/` document's code-formatted repo references resolve.
//
// `giwt plan check-links` validates Markdown links only. Plan cross-links are
// code-formatted (`.plan/tickets/X.md`), so a renamed or deleted sibling plan
// document breaks them silently. This gate fails on a dangling intra-repo
// `.plan/` reference.
//
// Advisory only (reported, never failed):
//   - `crates/`/`docs/` paths, because a proposed ticket may legitimately name
//     a file that does not exist yet ("the new `crates/server/tests/...`").
//   - epic/ticket ownership drift: an epic indexes the subset of its tickets
//     it tracks, so "this epic does not list that ticket" is not a defect.
//   - a ticket an epic names but that has no file, when the epic says the
//     name is research-only.
//
// `.tmp/` references are gitignored and unverifiable in CI; they are counted
// and reported, never failed.
//
// Usage: bun scripts/check-plan-doc-refs.ts [repo-root]
// Exit 0 = no dangling intra-repo plan reference, 1 = at least one.
//
// Resource contract: this gate is READ-ONLY over the repository it is given.
// It allocates no port, no database, no lock, and no shared temp state. The
// optional root argument exists so each test can point it at its OWN
// disposable fixture tree; a fixed-path fixture would race under a
// parallel runner.

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { relative, resolve } from "node:path";

const root = resolve(process.argv[2] ?? resolve(import.meta.dir, ".."));
const planDir = resolve(root, ".plan");

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = resolve(dir, entry);
    if (statSync(full).isDirectory()) out.push(...walk(full));
    else if (entry.endsWith(".md")) out.push(full);
  }
  return out;
}

const repoPrefixes = ["crates/", "scripts/", "packages/", "docs/", ".plan/"];
const dangling: string[] = [];
const advisory: string[] = [];
let checked = 0;
let tmpRefs = 0;

for (const file of walk(planDir)) {
  const text = readFileSync(file, "utf8");
  for (const match of text.matchAll(/`([^`\n]+)`/g)) {
    const path = match[1].trim().split(/[:#]/)[0];
    if (!path || /[\s*{}[\]<>]/.test(path)) continue;
    if (path.startsWith(".tmp/")) {
      tmpRefs++;
      continue;
    }
    if (!repoPrefixes.some((prefix) => path.startsWith(prefix))) continue;
    checked++;
    if (existsSync(resolve(root, path))) continue;
    const line = text.slice(0, match.index).split("\n").length;
    const entry = `${relative(root, file)}:${line}: ${path}`;
    const context = text.slice(Math.max(0, match.index - 80), match.index);
    const crossRepo = /\b(loop-lore|looplore)\b[^`]{0,20}$/i.test(context);
    if (path.startsWith(".plan/") && !crossRepo) dangling.push(entry);
    else advisory.push(entry);
  }
}

// --- Epic <-> ticket ownership sync (advisory, counted not itemized) -------
// An epic indexes the subset of its tickets it tracks, so "this epic does not
// list that ticket" is not a defect. A ticket an epic names but that has no
// file may be a research-only forward reference. Both are counted so the
// number stays visible without flooding every gate run.
const ticketsDir = resolve(planDir, "tickets");
const epicsDir = resolve(planDir, "epics");
const ticketFiles = readdirSync(ticketsDir).filter((f) => f.endsWith(".md"));
const epicFiles = readdirSync(epicsDir).filter((f) => f.endsWith(".md"));

let orphanTicketRefs = 0;
for (const epicFile of epicFiles) {
  const text = readFileSync(resolve(epicsDir, epicFile), "utf8");
  for (const match of text.matchAll(/`([A-Z]+-[a-z0-9-]+\.md)`/g)) {
    if (!ticketFiles.includes(match[1])) orphanTicketRefs++;
  }
}

let unlistedTickets = 0;
for (const ticketFile of ticketFiles) {
  const text = readFileSync(resolve(ticketsDir, ticketFile), "utf8");
  const header = text.split("\n").slice(0, 20).join("\n");
  const epic = /^Epic:\s*(epic-[a-z0-9-]+)/m.exec(header)?.[1];
  if (!epic) continue;
  const epicFile = `${epic}.md`;
  if (!epicFiles.includes(epicFile)) {
    unlistedTickets++;
    continue;
  }
  if (!readFileSync(resolve(epicsDir, epicFile), "utf8").includes(ticketFile)) unlistedTickets++;
}

// --- Epic status claims vs ticket headers (advisory) ----------------------
// An epic bullet often restates a ticket's status ("P0, proposed"). The ticket
// header is the single source of truth, so a restated status that disagrees
// with its header is stale documentation. Reported, not gated: an epic may
// legitimately describe a slice rather than the ticket.
const STATUS_WORDS = ["done", "proposed", "in progress", "postponed", "blocked"];
const ticketStatus = new Map<string, string>();
for (const ticketFile of ticketFiles) {
  const header = readFileSync(resolve(ticketsDir, ticketFile), "utf8")
    .split("\n")
    .slice(0, 20)
    .join("\n");
  const raw = /^Status:\s*(.+)$/m.exec(header)?.[1]?.trim() ?? "";
  const word = STATUS_WORDS.find((w) => raw.toLowerCase().startsWith(w));
  if (word) ticketStatus.set(ticketFile, word);
}

// A claim counts only in an explicit status position, so prose like "no
// blocked axum worker" or "the upgrade is proposed" is not read as a status.
const CLAIM_PATTERNS = [
  /P[0-3]\s*,\s*(done|proposed|in progress|postponed|blocked)\b/i,
  /—\s*(done|proposed|in progress|postponed|blocked)\b/i,
  /\*\*(done|proposed|in progress|postponed|blocked)\*\*/i,
];

const statusDrift: string[] = [];
for (const epicFile of epicFiles) {
  const text = readFileSync(resolve(epicsDir, epicFile), "utf8");
  const lines = text.split("\n");
  lines.forEach((line, index) => {
    const ref = /`([A-Z]+-[a-z0-9-]+\.md)`/.exec(line)?.[1];
    if (!ref) return;
    const actual = ticketStatus.get(ref);
    if (!actual) return;
    for (const pattern of CLAIM_PATTERNS) {
      const claimed = pattern.exec(line)?.[1]?.toLowerCase();
      if (!claimed) continue;
      if (claimed === actual) break;
      statusDrift.push(
        `epics/${epicFile}:${index + 1}: ${ref} claimed "${claimed}", header says "${actual}"`,
      );
      break;
    }
  });
}

console.log(
  `plan_md_files=${walk(planDir).length} repo_refs_checked=${checked} tmp_refs_skipped=${tmpRefs}`,
);
console.log(
  `advisory: missing_path=${advisory.length} orphan_ticket_refs=${orphanTicketRefs}`
    + ` tickets_not_indexed_by_epic=${unlistedTickets} epic_status_drift=${statusDrift.length}`,
);
for (const entry of statusDrift.sort()) console.log(`status drift: ${entry}`);
for (const entry of dangling.sort()) console.error(`DANGLING: ${entry}`);

if (dangling.length > 0) {
  console.error(`${dangling.length} dangling plan reference(s)`);
  process.exit(1);
}
console.log("all intra-repo plan references resolve");
