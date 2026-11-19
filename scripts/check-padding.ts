// Opt-in equivalent of loop-lore's ESLint `padding-line-between-statements`
// rule: one blank line after every multiline statement. honkbox has no ESLint,
// so this uses the already-installed TypeScript compiler API. `--fix` inserts
// the missing blank lines.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import ts from "typescript";

const ROOTS = ["scripts", "packages/web-ui/src"];
const EXEMPT = /(^|\/)gen\//;
const fix = process.argv.includes("--fix");
const targets = process.argv.slice(2).filter((arg) => !arg.startsWith("--"));
type Offense = { line: number; pos: number; };

function findOffenses(file: string, text: string): Offense[] {
  const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
  const found: Offense[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isSourceFile(node) || ts.isBlock(node) || ts.isModuleBlock(node)) {
      const statements = node.statements;
      for (let i = 1; i < statements.length; i++) {
        const previous = statements[i - 1]!;
        const next = statements[i]!;
        const start = source.getLineAndCharacterOfPosition(previous.getStart(source)).line;
        const end = source.getLineAndCharacterOfPosition(previous.getEnd()).line;
        if (end === start) continue;
        const nextStart = source.getLineAndCharacterOfPosition(next.getStart(source)).line;
        if (nextStart <= end + 1) {
          found.push({
            line: end + 2,
            pos: text.lastIndexOf("\n", next.getStart(source) - 1) + 1,
          });
        }
      }
    }

    ts.forEachChild(node, visit);
  };

  visit(source);
  return found;
}

function files(dir: string): string[] {
  let entries: string[];
  try {
    entries = readdirSync(dir, { withFileTypes: true }).map((e) => join(dir, e.name));
  } catch {
    return [];
  }

  return entries.flatMap((entry) => /\.(ts|tsx|js|mjs)$/.test(entry) ? [entry] : files(entry));
}

let total = 0;
for (const root of ROOTS) {
  for (const file of files(root)) {
    if (EXEMPT.test(file) || /\.test\.tsx?$/.test(file)) continue;
    if (targets.length > 0 && !targets.some((target) => file.startsWith(target))) continue;
    const text = readFileSync(file, "utf8");
    const offenses = findOffenses(file, text);
    total += offenses.length;
    if (fix && offenses.length > 0) {
      let patched = text;
      for (const pos of [...new Set(offenses.map((o) => o.pos))].sort((a, b) => b - a)) {
        patched = `${patched.slice(0, pos)}\n${patched.slice(pos)}`;
      }

      writeFileSync(file, patched);
    } else {
      for (const offense of offenses) {
        console.log(`${file}:${offense.line}: missing blank line after multiline statement`);
      }
    }
  }
}

console.log(fix ? `fixed ${total} padding site(s)` : `${total} padding site(s) need a blank line`);
if (!fix && total > 0) process.exit(1);
