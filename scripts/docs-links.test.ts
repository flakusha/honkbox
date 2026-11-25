import { afterEach, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { checkDocsLinks } from "./docs-links";

const roots: string[] = [];
const scratch = join(import.meta.dir, "..", ".tmp");

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

async function fixture(files: Record<string, string>, prefix = "docs-links-"): Promise<string> {
  await mkdir(scratch, { recursive: true });
  const root = await mkdtemp(join(scratch, prefix));
  roots.push(root);
  await mkdir(join(root, "docs"), { recursive: true });
  await Promise.all(
    Object.entries(files).map(async ([name, contents]) => {
      const path = join(root, name);
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, contents);
    }),
  );
  return root;
}

test("accepts nested files and directory targets", async () => {
  const root = await fixture({
    "README.md": "[nested](docs/guide/start.md)\n[directory](docs/guide/)\n",
    "docs/guide/start.md": "# Start\n",
  });
  expect(await checkDocsLinks(root)).toEqual({ files: 2, errors: [] });
});

test("accepts parenthesized paths, angle destinations, and optional titles", async () => {
  const root = await fixture({
    "README.md": [
      "[api](docs/api(v2).md \"API\")",
      "[angle](<docs/space name.md> 'Space')",
      "[title](docs/api(v2).md (API))",
      "[escaped](docs/api\\(v2\\).md)",
      "[nested [label]](docs/api(v2).md)",
      "",
    ].join("\n"),
    "docs/api(v2).md": "# API\n",
    "docs/space name.md": "# Space\n",
  });
  expect(await checkDocsLinks(root)).toEqual({ files: 3, errors: [] });
});

test("accepts full, collapsed, and shortcut reference links", async () => {
  const root = await fixture({
    "README.md": [
      "[full][guide]",
      "[collapsed][]",
      "[shortcut]",
      "",
      "[guide]: docs/guide.md \"Guide\"",
      "[collapsed]: <docs/api(v2).md> 'API'",
      "[shortcut]: docs/",
      "",
    ].join("\n"),
    "docs/guide.md": "# Guide\n",
    "docs/api(v2).md": "# API\n",
  });
  expect(await checkDocsLinks(root)).toEqual({ files: 3, errors: [] });
});

test("reports a missing target with file, line, and destination", async () => {
  const root = await fixture({ "README.md": "# Test\n\n[missing](docs/missing.md)\n" });
  expect(await checkDocsLinks(root)).toEqual({
    files: 1,
    errors: ["README.md:3: broken link docs/missing.md"],
  });
});

test("decodes percent-encoded paths", async () => {
  const root = await fixture({
    "README.md": "[space](docs/space%20name.md)\n",
    "docs/space name.md": "# Space\n",
  });
  expect(await checkDocsLinks(root)).toEqual({ files: 2, errors: [] });
});

test("checks links when the repository root contains spaces", async () => {
  const root = await fixture(
    {
      "README.md": "[guide](docs/guide.md?plain=1#install)\n",
      "docs/guide.md": "# Install\n",
    },
    "docs links ",
  );
  expect(await checkDocsLinks(root)).toEqual({ files: 2, errors: [] });
});

test("skips external targets, anchors, and Markdown examples", async () => {
  const root = await fixture({
    "README.md": [
      "[web](https://example.com)",
      "[cdn](//example.com/docs)",
      "[mail](mailto:test@example.com)",
      "[anchor](#install)",
      "[guide](docs/guide.md?plain=1#install)",
      "[`code label`](docs/guide.md)",
      "`[inline example](docs/missing.md)`",
      "\\[escaped](docs/missing.md)",
      "<!-- [comment](docs/missing.md) -->",
      "<a href=\"https://example.com\">external</a>",
      "",
      "```md",
      "[fenced](docs/missing.md)",
      "```not-a-closer",
      "[still fenced](docs/missing.md)",
      "```",
      "~~~~",
      "[tilde fenced](docs/missing.md)",
      "~~~~",
      "    [indented](docs/missing.md)",
      "",
    ].join("\n"),
    "docs/guide.md": "# Install\n",
  });
  expect(await checkDocsLinks(root)).toEqual({ files: 2, errors: [] });
});

test("surfaces malformed and unsupported relative syntax", async () => {
  const root = await fixture({
    "README.md": [
      "[bad title](docs/guide.md nope)",
      "[missing][unknown]",
      "[bad encoding](docs/%ZZ.md)",
      "[unclosed](<docs/guide.md",
      "<a href=\"docs/guide.md\">HTML</a>",
      "",
    ].join("\n"),
    "docs/guide.md": "# Guide\n",
  });
  expect((await checkDocsLinks(root)).errors).toEqual([
    "README.md:1: malformed inline link: [bad title](docs/guide.md nope)",
    "README.md:3: malformed percent-encoding in link docs/%ZZ.md",
    "README.md:4: malformed inline link: [unclosed](<docs/guide.md",
    "README.md:2: undefined reference unknown",
    "README.md:5: unsupported HTML href docs/guide.md: <a href=\"docs/guide.md\">HTML</a>",
  ]);
});
