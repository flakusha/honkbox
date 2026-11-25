// docs:check — every relative link in README.md + docs/**/*.md must resolve.
// External schemes, //hosts, root-relative paths, and fragments are skipped.
// Code (fenced, indented, or inline) and HTML comments are not live Markdown.
// HTML hrefs are deliberately unsupported and reported rather than silently ignored.
import { Glob } from "bun";
import { stat } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";

type InlineLink = { inner: string; end: number; };
type InlineStart = { index: number; open: number; };
type CheckResult = { files: number; errors: string[]; };

const defaultRoot = join(import.meta.dir, "..");

function blank(value: string): string {
  return value.replace(/[^\n]/g, " ");
}

function liveMarkdown(text: string): string {
  let fence: string | undefined;
  const withoutBlocks = text
    .split("\n")
    .map((line) => {
      if (fence !== undefined) {
        const closing = /^(?: {0,3}>[ \t]?)* {0,3}(`{3,}|~{3,})[ \t\r]*$/.exec(line)?.[1];
        if (closing !== undefined && closing[0] === fence[0] && closing.length >= fence.length) {
          fence = undefined;
        }
        return blank(line);
      }
      const opening = /^(?: {0,3}>[ \t]?)* {0,3}(`{3,}|~{3,})/.exec(line)?.[1];
      if (opening !== undefined) {
        fence = opening;
        return blank(line);
      }
      return /^(?: {4}|\t)/.test(line) ? blank(line) : line;
    })
    .join("\n");
  return withoutBlocks.replace(/<!--[\s\S]*?(?:-->|$)/g, blank);
}

function codeRanges(text: string): Array<[number, number]> {
  const ranges: Array<[number, number]> = [];
  let open: { start: number; length: number; } | undefined;
  for (const match of text.matchAll(/`+/g)) {
    const start = match.index ?? 0;
    if (isEscaped(text, start)) continue;
    if (open === undefined) open = { start, length: match[0].length };
    else if (match[0].length === open.length) {
      ranges.push([open.start, start + match[0].length]);
      open = undefined;
    }
  }
  return ranges;
}

function inside(index: number, ranges: Array<[number, number]>): boolean {
  return ranges.some(([start, end]) => index >= start && index < end);
}

function isEscaped(text: string, index: number): boolean {
  let slashes = 0;
  for (let i = index - 1; text[i] === "\\"; i--) slashes++;
  return slashes % 2 === 1;
}

function lineAt(text: string, index: number): number {
  return text.slice(0, index).split("\n").length;
}

function syntaxError(file: string, text: string, index: number, message: string): string {
  const end = text.indexOf("\n", index);
  const excerpt = text.slice(index, end < 0 ? text.length : end).trim();
  return `${file}:${lineAt(text, index)}: ${message}: ${excerpt}`;
}

function labelKey(label: string): string {
  return label.trim().replace(/\s+/g, " ").toLowerCase();
}

function readInline(text: string, open: number): InlineLink | undefined {
  let depth = 1;
  let quote = "";
  let angle = false;
  for (let i = open + 1; i < text.length; i++) {
    const ch = text[i]!;
    if (ch === "\\") {
      i++;
      continue;
    }
    if (ch === "\n") return;
    if (quote !== "") {
      if (ch === quote) quote = "";
      continue;
    }
    if (angle) {
      if (ch === ">") angle = false;
      continue;
    }
    if (ch === "<" && text.slice(open + 1, i).trim() === "") {
      angle = true;
      continue;
    }
    if ((ch === "\"" || ch === "'") && /\s/.test(text[i - 1] ?? "")) {
      quote = ch;
      continue;
    }
    if (ch === "(") depth++;
    else if (ch === ")" && --depth === 0) return { inner: text.slice(open + 1, i), end: i };
  }
}

// ponytail: backward bracket scan; fine for docs, revisit if bracket-heavy files grow.
function nextInline(
  text: string,
  code: Array<[number, number]>,
  from: number,
): InlineStart | undefined {
  for (let close = from; close < text.length - 1; close++) {
    if (text[close] !== "]" || text[close + 1] !== "(") continue;
    if (inside(close, code) || isEscaped(text, close)) continue;
    let depth = 1;
    for (let open = close - 1; open >= from; open--) {
      if (inside(open, code) || isEscaped(text, open)) continue;
      if (text[open] === "]") depth++;
      else if (text[open] === "[" && --depth === 0) return { index: open, open: close + 1 };
    }
  }
}
function parseDestination(value: string): string | undefined {
  const input = value.trim();
  if (input === "") return "";
  let target: string;
  let rest: string;
  if (input.startsWith("<")) {
    const end = input.indexOf(">", 1);
    if (end < 0 || /[\n<>]/.test(input.slice(1, end))) return;
    target = input.slice(1, end);
    rest = input.slice(end + 1).trim();
  } else {
    let depth = 0;
    let end = 0;
    for (; end < input.length; end++) {
      const ch = input[end]!;
      if (ch === "\\" && end + 1 < input.length) {
        end++;
        continue;
      }
      if (/\s/.test(ch)) break;
      if (ch === "(") depth++;
      else if (ch === ")") {
        if (depth === 0) break;
        depth--;
      }
    }
    if (depth !== 0) return;
    target = input.slice(0, end);
    rest = input.slice(end).trim();
  }
  if (rest !== "" && !/^(?:"[^"\n]*"|'[^'\n]*'|\([^()\n]*\))$/.test(rest)) return;
  return target.replace(/\\([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])/g, "$1");
}

function external(target: string): boolean {
  return /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(target);
}

async function targetError(
  root: string,
  file: string,
  line: number,
  target: string,
): Promise<string | undefined> {
  const bare = target.split(/[?#]/, 1)[0] ?? "";
  if (bare === "" || external(bare) || bare.startsWith("/")) return;
  let decoded: string;
  try {
    decoded = decodeURIComponent(bare);
  } catch {
    return `${file}:${line}: malformed percent-encoding in link ${target}`;
  }
  if (isAbsolute(decoded)) return;
  try {
    await stat(resolve(dirname(join(root, file)), decoded));
  } catch (error) {
    const code = (error as { code?: string; }).code;
    if (code === "ENOENT" || code === "ENOTDIR") return `${file}:${line}: broken link ${target}`;
    return `${file}:${line}: cannot inspect link ${target} (${code ?? "unknown error"})`;
  }
}

export async function checkDocsLinks(root: string): Promise<CheckResult> {
  const files: string[] = ["README.md"];
  const glob = new Glob("docs/**/*.md");
  for await (const entry of glob.scan({ cwd: root, onlyFiles: true })) {
    files.push(relative(root, resolve(root, entry)));
  }
  files.sort();
  const errors: string[] = [];
  for (const file of files) {
    const text = liveMarkdown(await Bun.file(join(root, file)).text());
    const code = codeRanges(text);
    const definitions = new Map<string, string>();
    const definitionRe = /^ {0,3}\[([^\]\n]+)\]:[ \t]*(.*)$/gm;
    for (const match of text.matchAll(definitionRe)) {
      const index = match.index ?? 0;
      const target = parseDestination(match[2] ?? "");
      if (target === undefined) {
        errors.push(syntaxError(file, text, index, "malformed reference definition"));
        continue;
      }
      const key = labelKey(match[1]!);
      if (!definitions.has(key)) definitions.set(key, target);
      const error = await targetError(root, file, lineAt(text, index), target);
      if (error !== undefined) errors.push(error);
    }

    let referenceText = text;
    let from = 0;
    let start: InlineStart | undefined;
    while ((start = nextInline(text, code, from)) !== undefined) {
      const { index, open } = start;
      from = index + 1;
      if (inside(index, code) || isEscaped(text, index)) continue;
      const inline = readInline(text, open);
      if (inline === undefined) {
        errors.push(syntaxError(file, text, index, "malformed inline link"));
        referenceText = referenceText.slice(0, index)
          + blank(text.slice(index, open + 1))
          + referenceText.slice(open + 1);
        continue;
      }
      from = inline.end + 1;
      referenceText = referenceText.slice(0, index)
        + blank(text.slice(index, inline.end + 1))
        + referenceText.slice(inline.end + 1);
      const target = parseDestination(inline.inner);
      if (target === undefined) {
        errors.push(syntaxError(file, text, index, "malformed inline link"));
        continue;
      }
      const error = await targetError(root, file, lineAt(text, index), target);
      if (error !== undefined) errors.push(error);
    }

    const referenceRe = /\[([^\]\n]+)\](?:\[([^\]\n]*)\])?/g;
    for (const ref of referenceText.matchAll(referenceRe)) {
      const index = ref.index ?? 0;
      if (inside(index, code) || isEscaped(referenceText, index)) continue;
      const full = ref[2] !== undefined;
      const after = referenceText[index + ref[0].length];
      if (!full && (after === "(" || after === ":")) continue;
      const label = full && ref[2] ? ref[2] : ref[1]!;
      if (full && !definitions.has(labelKey(label))) {
        errors.push(`${file}:${lineAt(referenceText, index)}: undefined reference ${label}`);
      }
    }

    const htmlRe = /<a\b[^>]*\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi;
    for (const html of text.matchAll(htmlRe)) {
      const index = html.index ?? 0;
      if (inside(index, code)) continue;
      const target = (html[1] ?? html[2] ?? html[3] ?? "").trim();
      const bare = target.split(/[?#]/, 1)[0] ?? "";
      if (bare === "" || external(bare) || bare.startsWith("/")) continue;
      errors.push(syntaxError(file, text, index, `unsupported HTML href ${target}`));
    }
  }
  return { files: files.length, errors };
}

if (import.meta.main) {
  const result = await checkDocsLinks(defaultRoot);
  for (const error of result.errors) console.error(error);
  if (result.errors.length > 0) {
    console.error(`docs:check: ${result.errors.length} broken or unsupported link(s)`);
    process.exitCode = 1;
  } else {
    console.log(`docs:check: ${result.files} files, links ok`);
  }
}
