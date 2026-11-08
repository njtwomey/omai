/**
 * Enumerate what the AIFN engine exports, per package and entry point, and write `docs/aifn/data/aifn-surface.json`.
 *
 *   node scripts/aifn-usage/scan-aifn.ts
 *
 * The entry points are the import paths the packages allow: `aifn-compute` and `aifn-methods` map `./*` onto
 * `src/*\/index.ts` (any depth), `aifn-render` lists its subpaths in `package.json`. The names of an entry point are its
 * export symbols as the TypeScript checker resolves them (`export *` followed), read from the source, never built. Each
 * declaration is recorded once at its home, the deepest entry point whose folder holds the declaring file, with every
 * entry point that exports it.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { Project, ts } from 'ts-morph'
import { AIFN, type AifnSurface, type AifnSymbol, git, writeJson } from './lib.ts'

interface Entry {
  readonly specifier: string
  readonly file: string
}

function dirsWithIndex(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir).sort()) {
    if (name === 'node_modules' || name === 'dist') continue
    const p = join(dir, name)
    if (statSync(p).isDirectory()) {
      if (existsSync(join(p, 'index.ts'))) out.push(p)
      dirsWithIndex(p, out)
    }
  }
  return out
}

function entriesOf(pkg: 'compute' | 'methods' | 'render'): Entry[] {
  const dir = join(AIFN, 'packages', pkg)
  const name = `aifn-${pkg}`
  const out: Entry[] = []
  const src = join(dir, 'src')
  if (pkg === 'render') {
    const exportsMap = (
      JSON.parse(readFileSync(join(dir, 'package.json'), 'utf8')) as { exports: Record<string, string> }
    ).exports
    for (const [key, target] of Object.entries(exportsMap)) {
      if (target.endsWith('.css')) continue
      const sub = key === '.' ? '' : key.slice(2)
      if (!key.includes('*')) {
        out.push({ specifier: sub ? `${name}/${sub}` : name, file: join(dir, target) })
        continue
      }
      const base = join(dir, target.slice(0, target.indexOf('*')))
      const ext = target.slice(target.lastIndexOf('*') + 1)
      for (const f of readdirSync(base).sort()) {
        if (f.endsWith(ext) && f !== 'index.ts') {
          out.push({ specifier: `${name}/${sub.replace('*', f.slice(0, -ext.length))}`, file: join(base, f) })
        }
      }
    }
    return out
  }
  if (pkg === 'compute') out.push({ specifier: name, file: join(src, 'index.ts') })
  for (const d of dirsWithIndex(src)) out.push({ specifier: `${name}/${relative(src, d)}`, file: join(d, 'index.ts') })
  return out
}

function kindOf(decl: ts.Node): string {
  if (ts.isFunctionDeclaration(decl)) return 'function'
  if (ts.isClassDeclaration(decl)) return 'class'
  if (ts.isInterfaceDeclaration(decl)) return 'interface'
  if (ts.isTypeAliasDeclaration(decl)) return 'type'
  if (ts.isEnumDeclaration(decl)) return 'enum'
  if (ts.isVariableDeclaration(decl)) {
    const init = decl.initializer
    if (init && (ts.isArrowFunction(init) || ts.isFunctionExpression(init))) return 'function'
    return 'const'
  }
  if (ts.isModuleDeclaration(decl) || ts.isSourceFile(decl) || ts.isNamespaceExport(decl)) return 'namespace'
  return 'other'
}

/** Read the AIFN source and return its surface (with summaries). */
export function scanAifn(): AifnSurface {
  const entries = (['compute', 'methods', 'render'] as const).flatMap(entriesOf)
  const project = new Project({
    skipAddingFilesFromTsConfig: true,
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ESNext,
      moduleResolution: ts.ModuleResolutionKind.Bundler,
      allowImportingTsExtensions: true,
      jsx: ts.JsxEmit.ReactJSX,
      skipLibCheck: true,
      noEmit: true,
      customConditions: ['source'],
    },
  })
  for (const e of entries) project.addSourceFileAtPath(e.file)
  project.resolveSourceFileDependencies()
  const checker = project.getTypeChecker()
  const entryDirs = entries
    .filter((e) => e.file.endsWith('/index.ts'))
    .map((e) => ({ dir: e.file.slice(0, -'index.ts'.length), specifier: e.specifier }))
    .sort((a, b) => b.dir.length - a.dir.length)
  const rootOf = (file: string): string => {
    const pkg = /\/packages\/(\w+)\//.exec(file)?.[1] ?? 'compute'
    const home = entryDirs.find((d) => file.startsWith(d.dir) && d.specifier.startsWith(`aifn-${pkg}`))
    if (home) return home.specifier
    // A file that sits below no entry folder (render's `src/ui/*.tsx`, `src/lib/*.ts`): the entry that names the file.
    const own = entries.find((e) => e.file === file)
    return own?.specifier ?? `aifn-${pkg}`
  }

  type Declared = Omit<AifnSymbol, 'alsoIn'>
  const symbols = new Map<string, { sym: Declared; entries: Set<string> }>()
  const surfaceEntries: { specifier: string; file: string; names: string[] }[] = []
  for (const e of entries) {
    const sf = project.getSourceFileOrThrow(e.file)
    const names: string[] = []
    for (const s of sf.getExportSymbols()) {
      const name = s.getName()
      if (name === 'default') continue
      names.push(name)
      const target = s.isAlias() ? (s.getAliasedSymbol() ?? s) : s
      const decl = target.getDeclarations()[0]
      if (!decl) continue
      const file = decl.getSourceFile().getFilePath()
      if (!file.startsWith(AIFN)) continue
      const line = decl.getStartLineNumber()
      const key = `${file}#${target.getName()}`
      let rec = symbols.get(key)
      if (!rec) {
        const doc = ts.displayPartsToString(target.compilerSymbol.getDocumentationComment(checker.compilerObject))
        const summary =
          doc
            .split(/\n\s*\n/)[0]
            ?.replace(/\s+/g, ' ')
            .trim()
            .slice(0, 110) ?? ''
        rec = {
          sym: {
            id: '',
            name,
            kind: kindOf(decl.compilerNode),
            source: `${relative(join(AIFN, 'packages'), file).replace('/src/', '/')}:${line}`,
            summary,
          },
          entries: new Set(),
        }
        symbols.set(key, rec)
      }
      rec.entries.add(e.specifier)
    }
    surfaceEntries.push({ specifier: e.specifier, file: relative(AIFN, e.file), names: names.sort() })
  }
  const list: (Declared & { entries: string[] })[] = []
  for (const [key, rec] of symbols) {
    const file = key.slice(0, key.indexOf('#'))
    const homeEntry = rootOf(file)
    const entriesSorted = [...rec.entries].sort()
    // The home is the entry point that holds the declaring file; a name only re-exported elsewhere keeps its first entry.
    const home = rec.entries.has(homeEntry) ? homeEntry : (entriesSorted[0] as string)
    list.push({ ...rec.sym, id: `${home}#${rec.sym.name}`, entries: entriesSorted })
  }
  list.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
  const seen = new Set<string>()
  for (const s of list) {
    if (seen.has(s.id)) console.warn(`duplicate id ${s.id}`)
    seen.add(s.id)
  }
  const out: AifnSurface = {
    commit: git(AIFN, 'rev-parse', '--short', 'HEAD'),
    entries: surfaceEntries
      .map((e) => ({ specifier: e.specifier, file: e.file, count: e.names.length }))
      .sort((a, b) => (a.specifier < b.specifier ? -1 : 1)),
    symbols: list.map((s) => {
      const { entries: all, ...rest } = s
      const home = s.id.slice(0, s.id.indexOf('#'))
      const also = all.filter((x) => x !== home)
      return { ...rest, ...(also.length > 0 ? { alsoIn: also } : {}) }
    }),
  }
  return out
}

if (import.meta.main) {
  const out = scanAifn()
  const slim = (p: string): Record<string, unknown>[] =>
    out.symbols.filter((s) => s.id.startsWith(`aifn-${p}`)).map(({ summary: _s, ...rest }) => rest)
  for (const p of ['compute', 'methods', 'render']) {
    writeJson(`aifn-surface-${p}.json`, {
      commit: out.commit,
      entries: out.entries.filter((e) => e.specifier.startsWith(`aifn-${p}`)),
      symbols: slim(p),
    })
  }
  console.log(`aifn surface: ${out.entries.length} entry points, ${out.symbols.length} declarations`)
}
