export {};

// Regenerates packages/web-ui/src/gen/openapi.json from the server's
// OpenAPI export. Drift is gated by the `git diff --exit-code` step in
// the `codegen:check` script — this file always writes.
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
