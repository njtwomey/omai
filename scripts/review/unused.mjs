/**
 * Triages the core exports no other source file uses (the list `surface.mjs` counts) by deterministic signals, into
 * (a) deliberate public API, (b) building blocks used only inside their own file, (c) dead:
 *
 *   node scripts/api-surface.ts --full <file>
 *   node scripts/review/unused.mjs <full surface file> [rows.tsv]
 *
 * A signal makes an export (a): a test file names it, its TSDoc has `@concept` or `@index`, or a doc (structure.md, the
 * methods catalogue, any other markdown outside code-review and history) names it; a type or interface that its own
 * file uses is part of the signature of what it configures. Without a signal, an export that its own file uses is (b)
 * (a candidate for a private `_` module) and one that nothing uses is (c). When in doubt it is (a). Prints counts by
 * domain and class; with a second argument, writes one row per export (the table the findings file is made from).
 */
import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join, relative } from 'node:path'
import { CORE, isBarrel, isData, isTest, LIB, listFor, ROOT, strip, walk } from './_shared.mjs'

const rows = readFileSync(process.argv[2], 'utf8')
  .split('\n')
  .filter(Boolean)
  .map((line) => {
    const [id, kind] = line.split('\t')
    const [module, name] = id.split('#')
    const [pkg, domain = ''] = module.split('/')
    return { kind, module, name, pkg, domain }
  })
  .filter((e) => e.pkg === 'omai-engine' && CORE.has(e.domain))

// Where each name is mentioned: source files, test files, markdown files.
const word = /[A-Za-z_]\w*/g
const mentions = { source: new Map(), test: new Map(), docs: new Map(), structure: new Map(), catalogue: new Map() }
const wordsOf = (text) => new Set(text.match(word) ?? [])
const sourceFiles = [...walk(join(ROOT, 'packages')), ...walk(join(ROOT, 'apps'))]
for (const f of sourceFiles) {
  if (isBarrel(f) || isData(f)) continue
  const kind = isTest(f) ? 'test' : 'source'
  for (const w of wordsOf(readFileSync(f, 'utf8'))) listFor(mentions[kind], w).push(f)
}
const entriesOf = (dir) => {
  try {
    return readdirSync(dir, { withFileTypes: true })
  } catch {
    return []
  }
}
const markdown = (dir, found = []) => {
  for (const entry of entriesOf(dir)) {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) {
      if (!['code-review', 'history', 'archive', 'node_modules'].includes(entry.name)) markdown(path, found)
    } else if (entry.name.endsWith('.md')) found.push(path)
  }
  return found
}
for (const f of markdown(join(ROOT, 'docs'))) {
  const where = relative(join(ROOT, 'docs'), f)
  const bucket = where === 'structure.md' ? 'structure' : where.startsWith('methods-catalog') ? 'catalogue' : 'docs'
  for (const w of wordsOf(readFileSync(f, 'utf8'))) listFor(mentions[bucket], w).push(f)
}

// Where each name is declared (an export of a source file under the engine) and its doc block.
const declarations = new Map()
const declaration =
  /^export\s+(?:declare\s+)?(?:async\s+)?(?:function\*?|const|let|class|abstract class|interface|type|enum)\s+([A-Za-z_$][\w$]*)/
for (const f of walk(LIB)) {
  if (isTest(f) || isData(f) || isBarrel(f)) continue
  const lines = readFileSync(f, 'utf8').split('\n')
  lines.forEach((line, i) => {
    const m = declaration.exec(line)
    if (m !== null) listFor(declarations, m[1]).push({ file: f, line: i, lines })
  })
}

const docBlock = (lines, i) => {
  let j = i - 1
  while (j >= 0 && /^\s*(\/\*\*|\*|\*\/|\/\/|@)/.test(lines[j])) j--
  return lines.slice(j + 1, i).join('\n')
}

const unused = rows.filter((e) => (mentions.source.get(e.name)?.length ?? 0) <= 1)
const out = []
for (const e of unused) {
  const where = (declarations.get(e.name) ?? []).find((d) => relative(LIB, d.file).startsWith(`${e.domain}/`))
  if (where === undefined) continue
  const doc = docBlock(where.lines, where.line)
  const code = strip(where.lines.join('\n'))
  const declLines = where.lines.filter((l) => declaration.exec(l)?.[1] === e.name).length
  const uses = (code.match(new RegExp(`\\b${e.name.replace(/\$/g, '\\$')}\\b`, 'g')) ?? []).length - declLines
  const tests = new Set(mentions.test.get(e.name) ?? []).size
  const docs = {
    structure: (mentions.structure.get(e.name) ?? []).length > 0,
    catalogue: (mentions.catalogue.get(e.name) ?? []).length > 0,
    other: (mentions.docs.get(e.name) ?? []).length > 0,
  }
  const concept = /@concept\b/.test(doc)
  const index = /@index\b/.test(doc)
  const example = /@example\b/.test(doc)
  const typeLike = e.kind === 'interface' || e.kind === 'type'
  const signal = tests > 0 || concept || index || docs.structure || docs.catalogue || docs.other
  const cls = signal || (typeLike && uses > 0) ? 'a' : uses > 0 ? 'b' : 'c'
  out.push({
    ...e,
    file: relative(ROOT, where.file),
    line: where.line + 1,
    tests,
    concept,
    index,
    example,
    docs,
    uses,
    cls,
  })
}

const tally = new Map()
for (const r of out) listFor(tally, r.domain).push(r.cls)
console.log(`UNUSED CORE EXPORTS: ${unused.length} (classified ${out.length}) of ${rows.length}`)
const total = { a: 0, b: 0, c: 0 }
for (const [domain, list] of [...tally].sort((x, y) => y[1].length - x[1].length)) {
  const n = { a: 0, b: 0, c: 0 }
  for (const c of list) n[c]++
  for (const c of 'abc') total[c] += n[c]
  console.log(`  ${domain.padEnd(11)} ${String(list.length).padStart(4)}  a ${n.a}  b ${n.b}  c ${n.c}`)
}
console.log(`  total       ${String(out.length).padStart(4)}  a ${total.a}  b ${total.b}  c ${total.c}`)
const why = { tests: 0, concept: 0, index: 0, docs: 0, typeUsed: 0 }
for (const r of out.filter((x) => x.cls === 'a')) {
  if (r.tests > 0) why.tests++
  else if (r.concept) why.concept++
  else if (r.index) why.index++
  else if (r.docs.structure || r.docs.catalogue || r.docs.other) why.docs++
  else why.typeUsed++
}
console.log(
  `  (a) by first signal: tests ${why.tests}, @concept ${why.concept}, @index ${why.index}, docs ${why.docs}, a type its file uses ${why.typeUsed}`,
)
if (process.argv[3] !== undefined) {
  const header = 'class\tdomain\tmodule\tname\tkind\tfile\tline\ttests\tconcept\tindex\texample\tdocs\tuses'
  const body = out.map((r) =>
    [
      r.cls,
      r.domain,
      r.module,
      r.name,
      r.kind,
      r.file,
      r.line,
      r.tests,
      r.concept,
      r.index,
      r.example,
      Object.entries(r.docs)
        .filter(([, v]) => v)
        .map(([k]) => k)
        .join('+'),
      r.uses,
    ].join('\t'),
  )
  writeFileSync(process.argv[3], [header, ...body].join('\n'))
}
