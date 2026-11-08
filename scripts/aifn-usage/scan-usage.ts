/**
 * Scan ml-concepts for imports of the aifn packages and write `docs/aifn/data/usage.json`.
 *
 *   node scripts/aifn-usage/scan-usage.ts
 *
 * Every `.mdx`, `.tsx` and `.ts` file of `content/notes`, `site/src`, `lab`, `scripts` and `plugins` is read for static
 * imports and re-exports (named, default, namespace with its `ns.member` uses, side-effect), dynamic `import()` and the
 * worker calls `call('module/export')`. A note is a folder with an `index.mdx`; it uses what its own files import and
 * what the files they import (relatively, or through the `@/` alias of the site) import in turn, so the code in
 * `_shared/` folders counts for every note that uses it. Nothing is written outside `docs/aifn/data/`.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { git, type ImportUse, ML, type UsageData, type UsedSymbol, writeJson } from './lib.ts'

const AIFN_SPEC = /^(aifn-compute|aifn-render|aifn-methods)(?:\/(.*))?$/
const CODE = /\.(mdx|tsx|ts|mjs|js)$/
const IMPORT = /^[ \t]*(import|export)\s+(type\s+)?([^;'"]*?)\s*from\s+['"]([^'"]+)['"]/gm
const SIDE_EFFECT = /^[ \t]*import\s+['"]([^'"]+)['"]/gm
const DYNAMIC = /\bimport\(\s*['"]([^'"]+)['"]\s*\)/g
const CALL = /\bcall\(\s*['"]([\w./-]+)['"]/g

const MAX_LISTED = 25

type Tree = 'notes' | 'lab' | 'site' | 'scripts' | 'plugins'

function walk(dir: string, out: string[] = []): string[] {
  if (!existsSync(dir)) return out
  for (const name of readdirSync(dir).sort()) {
    if (name === 'node_modules' || name === 'dist' || name.startsWith('.')) continue
    const p = join(dir, name)
    if (statSync(p).isDirectory()) walk(p, out)
    else if (CODE.test(name)) out.push(p)
  }
  return out
}

function stripComments(s: string): string {
  return s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '')
}

/** The names a clause imports: `{ a, b as c, type d }`, `* as ns`, a default, or a default with a list. */
function clauseNames(clause: string): { names: { name: string; typeOnly: boolean }[]; namespace?: string } {
  const c = stripComments(clause).trim()
  const names: { name: string; typeOnly: boolean }[] = []
  let namespace: string | undefined
  const ns = /\*\s*as\s+(\w+)/.exec(c)
  if (ns) namespace = ns[1]
  if (c === '*') namespace = ''
  const braces = /\{([\s\S]*)\}/.exec(c)
  if (braces) {
    for (const part of (braces[1] as string).split(',')) {
      const m = /^\s*(type\s+)?([\w$]+)(?:\s+as\s+[\w$]+)?\s*$/.exec(part)
      if (m) names.push({ name: m[2] as string, typeOnly: m[1] !== undefined })
    }
  }
  const def = /^([\w$]+)\s*(?:,|$)/.exec(c)
  if (def && !c.startsWith('type ') && !c.startsWith('{') && !c.startsWith('*')) {
    names.push({ name: 'default', typeOnly: false })
  }
  return namespace === undefined ? { names } : { names, namespace }
}

interface FileInfo {
  readonly file: string
  readonly uses: ImportUse[]
  readonly locals: string[]
  readonly workerCalls: string[]
  readonly dynamic: string[]
}

function parseFile(file: string, siteSrc: string): FileInfo {
  const text = readFileSync(file, 'utf8')
  const uses: ImportUse[] = []
  const locals: string[] = []
  const dynamic: string[] = []
  const addLocal = (spec: string): void => {
    if (spec.startsWith('.')) locals.push(resolve(dirname(file), spec))
    else if (spec.startsWith('@/')) locals.push(join(siteSrc, spec.slice(2)))
  }
  for (const m of text.matchAll(IMPORT)) {
    const spec = m[4] as string
    const a = AIFN_SPEC.exec(spec)
    if (!a) {
      addLocal(spec)
      continue
    }
    const { names, namespace } = clauseNames(m[3] as string)
    const type = m[2] !== undefined
    const sub = a[2] ?? ''
    for (const n of names)
      uses.push({ package: a[1] as string, subpath: sub, name: n.name, typeOnly: type || n.typeOnly })
    if (namespace !== undefined) {
      // `import * as ns`: the members used as `ns.name` stand for the names; `export * from` re-exports everything.
      const members = new Set<string>()
      if (namespace !== '')
        for (const x of text.matchAll(new RegExp(`\\b${namespace}\\.([A-Za-z_$][\\w$]*)`, 'g')))
          members.add(x[1] as string)
      if (members.size === 0) uses.push({ package: a[1] as string, subpath: sub, name: '*', typeOnly: type })
      for (const name of members) uses.push({ package: a[1] as string, subpath: sub, name, typeOnly: type })
    }
  }
  for (const m of text.matchAll(SIDE_EFFECT)) {
    const a = AIFN_SPEC.exec(m[1] as string)
    if (a) uses.push({ package: a[1] as string, subpath: a[2] ?? '', name: '(side-effect)', typeOnly: false })
    else addLocal(m[1] as string)
  }
  for (const m of text.matchAll(DYNAMIC)) {
    const a = AIFN_SPEC.exec(m[1] as string)
    if (a) {
      dynamic.push(m[1] as string)
      uses.push({ package: a[1] as string, subpath: a[2] ?? '', name: '*', typeOnly: false })
    } else addLocal(m[1] as string)
  }
  const workerCalls = /useComputed|aifn-render/.test(text) ? [...text.matchAll(CALL)].map((m) => m[1] as string) : []
  return { file, uses, locals, workerCalls, dynamic }
}

function resolveLocal(base: string, files: ReadonlySet<string>): string | undefined {
  for (const c of [base, `${base}.ts`, `${base}.tsx`, `${base}.mdx`, `${base}/index.ts`, `${base}/index.tsx`]) {
    if (files.has(c)) return c
  }
  return undefined
}

/** Scan ml-concepts and return the usage (every note index kept). */
export function scanUsage(): UsageData {
  const trees: Record<Tree, string[]> = {
    notes: walk(join(ML, 'content', 'notes')),
    lab: walk(join(ML, 'lab')),
    site: walk(join(ML, 'site', 'src')),
    scripts: walk(join(ML, 'scripts')),
    plugins: walk(join(ML, 'plugins')),
  }
  const treeOf = new Map<string, Tree>()
  for (const [t, fs] of Object.entries(trees)) for (const f of fs) treeOf.set(f, t as Tree)
  const all = new Set(treeOf.keys())
  const siteSrc = join(ML, 'site', 'src')
  const info = new Map<string, FileInfo>()
  for (const f of all) info.set(f, parseFile(f, siteSrc))

  // Notes and their closures.
  const noteDirs = trees.notes
    .filter((f) => f.endsWith('/index.mdx'))
    .map((f) => dirname(f))
    .sort()
  const noteDirSet = new Set(noteDirs)
  const slugs = noteDirs.map((d) => d.slice(d.lastIndexOf('/') + 1))
  const subFolderOwned = (f: string, dir: string): boolean => {
    if (!f.startsWith(`${dir}/`)) return false
    let d = dirname(f)
    while (d !== dir) {
      if (noteDirSet.has(d)) return false
      d = dirname(d)
    }
    return true
  }
  const resolved = new Map<string, string[]>()
  for (const [f, i] of info) {
    const out: string[] = []
    for (const l of i.locals) {
      const r = resolveLocal(l, all)
      if (r) out.push(r)
    }
    resolved.set(f, out)
  }
  const closure = (start: readonly string[]): Set<string> => {
    const seen = new Set<string>()
    const stack = [...start]
    while (stack.length > 0) {
      const f = stack.pop() as string
      if (seen.has(f)) continue
      seen.add(f)
      for (const r of resolved.get(f) ?? []) stack.push(r)
    }
    return seen
  }
  // Index the files of each note once.
  const ownerNote = new Map<string, number>()
  noteDirs.forEach((d, idx) => {
    for (const f of trees.notes) if (subFolderOwned(f, d)) ownerNote.set(f, idx)
  })

  interface Acc {
    package: string
    subpath: string
    name: string
    notes: Set<number>
    files: number
    lab: number
    site: number
    scripts: number
    typeOnly: boolean
    examples: string[]
  }
  const acc = new Map<string, Acc>()
  const touch = (u: ImportUse): Acc => {
    const id = `${u.package}${u.subpath ? `/${u.subpath}` : ''}#${u.name}`
    let a = acc.get(id)
    if (!a) {
      a = {
        package: u.package,
        subpath: u.subpath,
        name: u.name,
        notes: new Set(),
        files: 0,
        lab: 0,
        site: 0,
        scripts: 0,
        typeOnly: true,
        examples: [],
      }
      acc.set(id, a)
    }
    return a
  }
  // Direct uses per file.
  for (const [f, i] of info) {
    const tree = treeOf.get(f) as Tree
    const seenHere = new Set<string>()
    for (const u of i.uses) {
      const a = touch(u)
      const key = `${u.package}/${u.subpath}#${u.name}`
      if (!u.typeOnly) a.typeOnly = false
      if (seenHere.has(key)) continue
      seenHere.add(key)
      if (tree === 'notes') a.files++
      else if (tree === 'lab') a.lab++
      else if (tree === 'site') a.site++
      else a.scripts++
    }
  }
  // Notes through closures.
  const noteUses: Set<string>[] = []
  const callsOfNote: Set<string>[] = []
  let notesUsing = 0
  const fileNotes = new Map<string, number>()
  noteDirs.forEach((_d, idx) => {
    const start = trees.notes.filter((f) => ownerNote.get(f) === idx)
    const reach = closure(start)
    const ids = new Set<string>()
    const calls = new Set<string>()
    for (const f of reach) {
      fileNotes.set(f, (fileNotes.get(f) ?? 0) + 1)
      const i = info.get(f)
      if (!i) continue
      for (const u of i.uses) ids.add(`${u.package}${u.subpath ? `/${u.subpath}` : ''}#${u.name}`)
      for (const c of i.workerCalls) calls.add(c)
    }
    noteUses.push(ids)
    callsOfNote.push(calls)
    if (ids.size > 0) notesUsing++
    for (const id of ids) acc.get(id)?.notes.add(idx)
  })
  // Examples: the first five note slugs, else the first files.
  const symbols: UsedSymbol[] = [...acc.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([id, a]) => {
      const idxs = [...a.notes].sort((x, y) => x - y)
      const ex = idxs.slice(0, 5).map((i) => slugs[i] as string)
      return {
        id,
        package: a.package,
        subpath: a.subpath,
        name: a.name,
        notes: idxs.length,
        noteIndexes: idxs,
        files: a.files,
        labFiles: a.lab,
        siteFiles: a.site,
        scriptFiles: a.scripts,
        typeOnly: a.typeOnly,
        examples: ex,
      }
    })

  const bySub = new Map<string, { notes: Set<number>; symbols: Set<string> }>()
  for (const s of symbols) {
    const spec = `${s.package}${s.subpath ? `/${s.subpath}` : ''}`
    const e = bySub.get(spec) ?? { notes: new Set<number>(), symbols: new Set<string>() }
    for (const n of s.noteIndexes) e.notes.add(n)
    e.symbols.add(s.name)
    bySub.set(spec, e)
  }
  const subpaths = [...bySub.entries()]
    .map(([specifier, e]) => ({ specifier, notes: e.notes.size, symbols: e.symbols.size }))
    .sort((a, b) => b.notes - a.notes || (a.specifier < b.specifier ? -1 : 1))

  const callMap = new Map<string, Set<number>>()
  callsOfNote.forEach((cs, idx) => {
    for (const c of cs) (callMap.get(c) ?? callMap.set(c, new Set()).get(c))?.add(idx)
  })
  const workerCalls = [...callMap.entries()]
    .map(([call, ns]) => {
      const idx = [...ns].sort((x, y) => x - y)
      return { call, notes: ns.size, noteIndexes: idx, examples: idx.slice(0, 3).map((i) => slugs[i] as string) }
    })
    .sort((a, b) => b.notes - a.notes || (a.call < b.call ? -1 : 1))

  const dyn = new Map<string, string[]>()
  for (const [f, i] of info) for (const d of i.dynamic) (dyn.get(d) ?? dyn.set(d, []).get(d))?.push(relative(ML, f))
  const dynamicImports = [...dyn.entries()].map(([specifier, files]) => ({ specifier, files: files.slice(0, 5) }))

  // The code the notes wrote themselves: `.ts` files of the notes tree that several lines long, with what they export.
  const localFiles = trees.notes
    .filter((f) => f.endsWith('.ts'))
    .map((f) => {
      const text = readFileSync(f, 'utf8')
      const names = [...text.matchAll(/^export\s+(?:async\s+)?(?:function|const|class)\s+([A-Za-z_$][\w$]*)/gm)].map(
        (m) => m[1] as string,
      )
      return {
        file: relative(join(ML, 'content', 'notes'), f),
        lines: text.split('\n').length,
        notes: fileNotes.get(f) ?? 0,
        aifnImports: info.get(f)?.uses.length ?? 0,
        exports: names,
      }
    })
    .filter((x) => x.lines >= 60)
    .sort((a, b) => b.lines - a.lines)

  const data: UsageData = {
    source: { commit: git(ML, 'rev-parse', '--short', 'HEAD'), aifnVersion: aifnVersion() },
    totals: {
      notes: noteDirs.length,
      notesUsingAifn: notesUsing,
      noteFiles: trees.notes.length,
      noteFilesWithAifn: trees.notes.filter((f) => (info.get(f)?.uses.length ?? 0) > 0).length,
      labFiles: trees.lab.length,
      labFilesWithAifn: trees.lab.filter((f) => (info.get(f)?.uses.length ?? 0) > 0).length,
      siteFilesWithAifn: trees.site.filter((f) => (info.get(f)?.uses.length ?? 0) > 0).length,
      scriptFilesWithAifn: trees.scripts.filter((f) => (info.get(f)?.uses.length ?? 0) > 0).length,
      symbols: symbols.length,
      subpaths: subpaths.length,
    },
    notes: slugs,
    symbols,
    subpaths,
    workerCalls,
    dynamicImports,
    localFiles,
  }
  return data
}

function aifnVersion(): string {
  try {
    const p = JSON.parse(readFileSync(join(ML, 'package.json'), 'utf8')) as { dependencies?: Record<string, string> }
    return /v([\d.]+)\//.exec(p.dependencies?.['aifn-compute'] ?? '')?.[1] ?? 'unknown'
  } catch {
    return 'unknown'
  }
}

if (import.meta.main) {
  const data = scanUsage()
  // The committed file keeps the note indexes of the symbols few notes use (the rest are in the counts).
  const slim = data.symbols.map((s) => ({
    id: s.id,
    notes: s.notes,
    files: s.files,
    labFiles: s.labFiles,
    siteFiles: s.siteFiles,
    typeOnly: s.typeOnly,
    examples: s.examples.slice(0, 3),
    ...(s.notes <= MAX_LISTED ? { noteIndexes: s.noteIndexes } : {}),
  }))
  writeJson('usage.json', {
    ...data,
    workerCalls: data.workerCalls.map(({ noteIndexes: _n, ...rest }) => rest),
    symbols: slim,
  } as unknown as Record<string, unknown>)
  console.log(
    `usage: ${data.totals['notes']} notes, ${data.totals['notesUsingAifn']} use aifn, ${data.symbols.length} symbols, ${data.subpaths.length} subpaths`,
  )
}
