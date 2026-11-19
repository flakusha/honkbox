export {};

// Regenerates packages/web-ui/src/gen/openapi.json from the server's
// OpenAPI export. Drift is gated by the `git diff --exit-code` step in
// the `codegen:check` script — this file always writes.
//
// This script REWRITES the generated files on every run (even when the
// content is unchanged), so it must not run concurrently with the checks
// that READ them: `format` (dprint check) and `typecheck` (tsc) would see a
// half-written file. That is why the `check` chain in package.json is
// sequential and ends with this step.
const out = "packages/web-ui/src/gen/openapi.json";
const proc = Bun.spawn(
  ["cargo", "run", "-q", "--locked", "-p", "honkbox-server", "--", "openapi"],
  {
    stderr: "inherit",
    stdout: "pipe",
  },
);
const body = await new Response(proc.stdout).text();
const code = await proc.exited;
if (code !== 0) {
  console.error(`honkbox codegen: openapi export failed (exit ${code})`);
  process.exit(code);
}
await Bun.write(out, body.endsWith("\n") ? body : `${body}\n`);
console.log(`wrote ${out}`);

// Coverage: every mounted route must appear in the generated document. The
// `git diff --exit-code` drift gate above cannot see this class — a route
// missing from the `#[openapi(paths(...))]` list regenerates identically every
// time, so the served reference just silently loses the endpoint.
const routesSrc = await Bun.file("crates/server/src/routes.rs").text();
const mounted = [...routesSrc.matchAll(/\.route\(\s*"([^"]+)"/g)].map((m) => m[1] as string);
const documented = new Set(Object.keys((JSON.parse(body) as { paths?: object; }).paths ?? {}));
const missing = [...new Set(mounted)].filter((path) => !documented.has(path)).sort();
if (missing.length > 0) {
  console.error(
    `honkbox codegen: ${missing.length} mounted route(s) missing from the OpenAPI document: ${
      missing.join(", ")
    }`,
  );
  console.error(
    "add them to the `#[openapi(paths(...))]` list in crates/server/src/routes/openapi.rs",
  );
  process.exit(1);
}
console.log(`openapi covers all ${new Set(mounted).size} mounted routes`);
