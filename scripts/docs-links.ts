// docs:check — every relative link in README.md + docs/**/*.md must resolve.
// Skips http(s)/mailto/#-only targets. Anchor fragments are not checked:
// headings drift faster than files (see docs/meta/docs-shipping.md).
import { Glob } from "bun";

const root = new URL("..", import.meta.url).pathname;
const files: string[] = ["README.md"];
for await (const entry of new Glob("docs/**/*.md").scan(root)) {
  files.push(entry);
}

const linkRe = /\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g;
let broken = 0;

for (const file of files) {
  const text = await Bun.file(`${root}/${file}`).text();
  for (const match of text.matchAll(linkRe)) {
    const target = match[1];
    if (target === undefined || /^(https?:|mailto:|#)/.test(target)) continue;
    const hash = target.indexOf("#");
    const bare = hash < 0 ? target : target.slice(0, hash);
    if (bare === "") continue;
    const resolved = new URL(bare, `file://${root}/${file}`).pathname;
    if (!(await Bun.file(resolved).exists())) {
      console.error(`${file}: broken link ${target}`);
      broken++;
    }
  }
}

if (broken > 0) {
  console.error(`docs:check: ${broken} broken link(s)`);
  process.exit(1);
}
console.log(`docs:check: ${files.length} files, links ok`);
