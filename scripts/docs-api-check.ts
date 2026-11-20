export {};

// Generated API docs must exist where the server serves them, and must carry
// source doc comments — a hollow TypeDoc run passes no useful signal.
const required = [
  "web-dist/api-docs/index.html",
  "web-dist/api-docs/types/OcrIngestRequest.html",
] as const;

for (const file of required) {
  if (!(await Bun.file(file).exists())) {
    console.error(`api docs: missing ${file}`);
    process.exit(1);
  }
}

const page = await Bun.file(required[1]).text();
if (!page.includes("Path-free input")) {
  console.error("api docs: source doc comments did not reach the output");
  process.exit(1);
}

console.log(`api docs: ${required.length} files verified, doc comments present`);
