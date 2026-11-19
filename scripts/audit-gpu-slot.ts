#!/usr/bin/env bun
// Audit the server test suites for GPU-slot serialization gaps.
//
// The residency ledger's GPU slot is ONE process-global counter
// (`capacities.gpu_slots = 1`), so two tests that both take a GPU lease
// race: the loser gets a 429 (or blocks on the admission queue). Every
// test binary therefore serializes its GPU requests on a per-binary lock.
//
// This script is the gate that keeps that contract from rotting:
//
//   - the GPU-contending route list is DERIVED from
//     `crates/server/src/routes.rs` (a route whose handler acquires a
//     non-`Rag` resource kind takes a GPU slot; `Rag` demand is
//     `gpu_slots: 0`, so it never contends), so a new GPU route is
//     covered the moment it is mounted;
//   - for every `#[tokio::test]`, it asks whether the test issues a request
//     to one of those routes (directly or through a helper) and whether it
//     serializes (holds a lock, directly or through a helper);
//   - an unserialized GPU request fails the build.
//
// Detection notes:
//   - A GPU route counts as requested when its URI string appears in the
//     function body OUTSIDE assertion macros and `for x in [...]` route lists
//     (those document the OpenAPI document, they are not calls). This catches
//     both `.uri("/v1/...")` and `raw_post(app, "/v1/...", ...)`.
//   - Helpers are resolved transitively: a helper that calls a locked helper
//     is locked; a helper that calls a GPU-requesting helper requests the GPU.
//   - The lock check is deliberately broad (any `.lock()` call counts): a
//     false positive would block every developer's build, so the gate biases
//     toward missing a gap over inventing one.
//
// Exit code 0 = no findings, 1 = findings.

import { readdirSync } from "node:fs";

const ROUTES_SRC = "crates/server/src/routes.rs";
const ROUTES_DIR = "crates/server/src/routes";
const TESTS_DIR = "crates/server/tests";

/**
 * Routes whose handler refuses the request BEFORE it reaches the slot, keyed
 * by `file.rs:test_name`. The value is the marker the test must assert to
 * pin the invariant: the exemption is honored only while the pin is there, so
 * it cannot rot silently.
 */
const GATED_BEFORE_SLOT: Record<string, string> = {
  // The chat route gates on the model capability before it registers a job
  // or acquires the slot, so the request never touches the ledger.
  "embeddings.rs:embeddings_model_refused_by_chat_completions": "x-job-id",
  // The transcribe route validates the model before it registers a job.
  "models.rs:transcribe_validates_model_first": "x-job-id",
  // The synthesize route checks the input cap before it registers a job.
  "audio.rs:synthesize_input_over_cap_is_rejected": "x-job-id",
  // The blender route refuses a malformed manifest, an unstored mesh, a
  // wrong digest and a missing engine BEFORE it registers a job, so the
  // request never reaches the ledger.
  "blender_jobs.rs:the_request_shape_is_checked_before_the_store": "x-job-id",
  "blender_jobs.rs:an_unstored_mesh_is_a_404": "x-job-id",
  "blender_jobs.rs:a_digest_mismatch_is_a_400": "x-job-id",
  "blender_jobs.rs:without_an_engine_the_route_is_a_named_501": "x-job-id",
};

const FN_RE = /^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)/gm;
const CALL_RE = /\b([a-z_][A-Za-z0-9_]*)\s*\(/g;
const STR_RE = /"(?:[^"\\]|\\.)*"/g;
const ROUTE_RE = /\.route\(\s*"([^"]+)"\s*,/g;
const ACQUIRE_RE = /admission::acquire\(/g;
const RESOURCE_KIND_RE = /ResourceKind::([A-Za-z]+)/;
const HANDLER_RE = /\b([a-z_][A-Za-z0-9_]*)\s*\(\s*([a-z_][A-Za-z0-9_]*)/g;

/** Blank out comments and string literals (or comments only). */
function blank(src: string, keepStrings: boolean): string {
  let out = "";
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i]!;
    if (c === "/" && src[i + 1] === "/") {
      // Keep the length: the cleaned text is sliced with the SAME offsets as
      // the original, so a dropped character would shift every body.
      while (i < n && src[i] !== "\n") {
        out += " ";
        i++;
      }
      continue;
    }
    if (c === "/" && src[i + 1] === "*") {
      while (i < n && !(src[i] === "*" && src[i + 1] === "/")) {
        out += src[i] === "\n" ? "\n" : " ";
        i++;
      }
      out += "  ";
      i += 2;
      continue;
    }
    if (c === "\"") {
      const m = STR_RE.exec(src.slice(i));
      STR_RE.lastIndex = 0;
      if (m) {
        out += keepStrings ? m[0] : " ".repeat(m[0].length);
        i += m[0].length;
        continue;
      }
    }
    out += c;
    i++;
  }
  return out;
}

/** Index just past the pair opened at `start` (`(` or `[`). */
function matchPair(src: string, start: number): number {
  const opener = src[start]!;
  const closer = opener === "(" ? ")" : "]";
  let depth = 0;
  let i = start;
  while (i < src.length) {
    const c = src[i]!;
    if (c === "\"") {
      const m = STR_RE.exec(src.slice(i));
      STR_RE.lastIndex = 0;
      if (m) {
        i += m[0].length;
        continue;
      }
    }
    if (c === opener) depth++;
    else if (c === closer) {
      depth--;
      if (depth === 0) return i + 1;
    }
    i++;
  }
  return src.length;
}

/** Remove assertion macro bodies and `for x in [...]` route lists. */
function stripDocContext(src: string): string {
  let out = "";
  let i = 0;
  // Sticky: the pattern must match AT the cursor, not anywhere after it.
  const assert = /\bassert(?:_eq|_ne|_matches)?!\s*\(/y;
  const forList = /\bfor\s+[A-Za-z_][A-Za-z0-9_]*\s+in\s*\[/y;
  while (i < src.length) {
    // Sticky regexes keep `lastIndex` across calls; reset so each match must
    // start AT the cursor.
    assert.lastIndex = 0;
    forList.lastIndex = 0;
    const m = assert.exec(src.slice(i)) ?? forList.exec(src.slice(i));
    if (m) {
      const end = matchPair(src, i + m[0].length - 1);
      out += " ".repeat(end - i);
      i = end;
      continue;
    }
    out += src[i];
    i++;
  }
  return out;
}

interface FnItem {
  name: string;
  body: string;
  clean: string;
  attrs: string;
}

/** Every `fn` item in a Rust source file, with its brace-matched body. */
function fnItems(src: string): Map<string, FnItem> {
  const clean = blank(src, false);
  const out = new Map<string, FnItem>();
  FN_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = FN_RE.exec(clean)) !== null) {
    const name = m[1]!;
    const open = clean.indexOf("{", m.index + m[0].length);
    if (open === -1) continue;
    let depth = 0;
    let j = open;
    for (; j < clean.length; j++) {
      if (clean[j] === "{") depth++;
      else if (clean[j] === "}") {
        depth--;
        if (depth === 0) break;
      }
    }
    const head = clean.slice(Math.max(0, m.index - 200), m.index);
    const attrs = [...head.matchAll(/#\[([A-Za-z0-9_:]+)\]/g)].pop()?.[1] ?? "";
    out.set(name, { name, body: src.slice(open, j + 1), clean: clean.slice(open, j + 1), attrs });
  }
  return out;
}

/** The identifiers called from a body. */
function callees(clean: string): Set<string> {
  return new Set([...clean.matchAll(CALL_RE)].map((m) => m[1]!));
}

/**
 * The GPU-contending routes, derived from the server source: a route whose
 * handler (or a handler it calls) acquires a non-`Rag` resource kind takes a
 * GPU slot. `Rag` demand is `gpu_slots: 0` (`Demand::for_kind`), so it never
 * contends with the shared slot.
 */
function gpuRoutes(sources: string[]): { uris: string[]; handlers: string[]; } {
  // Handlers live in `routes/*.rs` after the size-lint split; registrations
  // stay in routes.rs. Merge every module so `fns.has(handler)` still holds.
  const fns = new Map<string, FnItem>();
  for (const src of sources) {
    for (const [name, item] of fnItems(src)) fns.set(name, item);
  }
  const acquiresGpu = new Map<string, boolean>();
  const resolves = (name: string, seen: Set<string>): boolean => {
    const cached = acquiresGpu.get(name);
    if (cached !== undefined) return cached;
    if (seen.has(name)) return false;
    seen.add(name);
    const fn = fns.get(name);
    if (!fn) return false;
    let gpu = false;
    // The resource kind is named inside the acquire call, after the
    // argument: `admission::acquire(honkbox_residency::ResourceKind::Llm)`.
    for (const m of fn.clean.matchAll(ACQUIRE_RE)) {
      const kind = RESOURCE_KIND_RE.exec(fn.clean.slice(m.index, m.index + 160))?.[1];
      if (kind !== undefined && kind !== "Rag") gpu = true;
    }
    if (!gpu) {
      for (const callee of callees(fn.clean)) {
        if (fns.has(callee) && resolves(callee, seen)) {
          gpu = true;
          break;
        }
      }
    }
    acquiresGpu.set(name, gpu);
    return gpu;
  };

  // The handler travels WITH its route: the two lists are printed as
  // pairs, so de-duplicating and sorting them apart would print a route
  // next to the wrong handler.
  const byUri = new Map<string, string>();
  ROUTE_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = ROUTE_RE.exec(routesSrc)) !== null) {
    const uri = m[1]!;
    const open = routesSrc.indexOf("(", m.index + m[0].length - 1);
    const args = routesSrc.slice(m.index + m[0].length, matchPair(routesSrc, open));
    for (const h of args.matchAll(HANDLER_RE)) {
      // Group 1 is the axum verb (`get`, `post`, `delete`), group 2 the handler.
      const name = h[2]!;
      if (!fns.has(name)) continue;
      if (resolves(name, new Set())) {
        byUri.set(uri, name);
        break;
      }
    }
  }
  const pairs = [...byUri.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1));
  return {
    uris: pairs.map(([uri]) => uri),
    handlers: pairs.map(([, handler]) => handler),
  };
}

/** Findings for one test file. */
async function auditFile(path: string, uris: readonly string[]): Promise<string[]> {
  const src = await Bun.file(path).text();
  const fns = fnItems(src);
  const findings: string[] = [];

  const resolves = (name: string, seen: Set<string>): { gpu: boolean; locked: boolean; } => {
    if (seen.has(name)) return { gpu: false, locked: false };
    seen.add(name);
    const fn = fns.get(name);
    if (!fn) return { gpu: false, locked: false };
    let gpu = uris.some((u) => stripDocContext(blank(fn.body, true)).includes(u));
    let locked = /\.lock\s*\(\s*\)/.test(fn.clean);
    for (const callee of callees(fn.clean)) {
      if (!fns.has(callee)) continue;
      const r = resolves(callee, seen);
      gpu = gpu || r.gpu;
      locked = locked || r.locked;
    }
    return { gpu, locked };
  };

  for (const fn of fns.values()) {
    if (fn.attrs !== "tokio::test" && fn.attrs !== "test") continue;
    const own = resolves(fn.name, new Set());
    if (!own.gpu) continue;
    if (own.locked) continue;

    const key = `${path.split("/").pop()}:${fn.name}`;
    const pin = GATED_BEFORE_SLOT[key];
    if (pin !== undefined) {
      if (fn.body.includes(pin)) continue;
      findings.push(`${key} — exemption lost: the no-slot invariant is no longer pinned`);
    } else {
      findings.push(`${key} — unserialized GPU request`);
    }
  }
  return findings;
}

const routesSrc = await Bun.file(ROUTES_SRC).text();
const routeFiles = readdirSync(ROUTES_DIR)
  .filter((file) => file.endsWith(".rs"))
  .sort();
const routeSources = [
  routesSrc,
  ...(await Promise.all(routeFiles.map((file) => Bun.file(`${ROUTES_DIR}/${file}`).text()))),
];
const { uris, handlers } = gpuRoutes(routeSources);
if (uris.length === 0) {
  console.error("honkbox gpu-slot audit: no GPU-contending route derived — the parser is broken");
  process.exit(1);
}
console.log(`GPU-contending routes derived from ${ROUTES_SRC}:`);
uris.forEach((uri, i) => console.log(`  ${uri} (${handlers[i] ?? "?"})`));

// The per-file audits are independent and read-only: their reads run
// concurrently, and the (fast, CPU-bound) parsing stays on this thread.
const files = readdirSync(TESTS_DIR).sort().filter((file) => file.endsWith(".rs"));
const findings = (await Promise.all(files.map((file) => auditFile(`${TESTS_DIR}/${file}`, uris))))
  .flat();

if (findings.length > 0) {
  console.error("\nGPU-SLOT SERIALIZATION FINDINGS:");
  for (const f of findings) console.error(`  ${f}`);
  console.error("\nserialize the request on a per-binary lock (SLOT_LOCK / OP_LOCK), or pin the");
  console.error("no-slot invariant in GATED_BEFORE_SLOT and assert it in the test");
  process.exit(1);
}
console.log("\nOK: every GPU-requesting test is serialized by a lock");
