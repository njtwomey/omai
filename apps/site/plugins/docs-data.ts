/**
 * The docs catalogue precompiled to static files, so no page waits for an extraction (`make docs-catalogue`, run by
 * `make dev` and `make site`; the dev server's plugin reruns it in the background after an engine edit).
 *
 * Into `apps/site/.generated/docs/` (gitignored) it writes, each named by a hash of its content:
 * - `index.<hash>.json`: the tree (folders with summaries; files with their opening sentence and their exports' names
 *   and kinds), a few hundred kilobytes where the full catalogue is several megabytes;
 * - `search.<hash>.json`: every export's one-sentence summary, fetched when someone searches;
 * - `files/<hash>.json`: one per source file, its full comment, its exports' `SymbolDoc`s and the type names they use,
 *   each resolved through the file's imports to the module that defines it (`refs`);
 * - `types/<hash>.json`: one per module, the declaration and summary of each type it exports, for the hover on a type
 *   name (fetched on the first hover; the index only names the file);
 * - `manifest.json` last (by rename), naming the current index, so a reader never sees half a version.
 *
 * Files of the current and the previous version are kept (a page open on the previous index can still load its files);
 * older ones are removed.
 */
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { join, posix, relative } from 'node:path'
import type { Catalogue } from '../../../packages/meta/src/schema/index.ts'
import type {
  WireFile,
  WireFileData,
  WireFolder,
  WireIndex,
  WireManifest,
  WireRefs,
  WireSearch,
  WireType,
} from '../src/docs/wire.ts'

/** The repository root. */
export const ROOT: string = join(import.meta.dirname, '..', '..', '..')
/** Where the files are written. */
export const OUT: string = join(ROOT, 'apps', 'site', '.generated', 'docs')
const MANIFEST = join(OUT, 'manifest.json')
const PREVIOUS = join(OUT, 'manifest.previous.json')
const SRC = 'packages/engine/src/'

/** What the catalogue is read from: a change to any of these makes the files stale. */
export const SOURCES: readonly string[] = [
  join(ROOT, 'packages', 'engine', 'src'),
  join(ROOT, 'packages', 'engine', 'package.json'),
  join(ROOT, 'packages', 'meta', 'src', 'extract'),
  join(ROOT, 'packages', 'meta', 'src', 'schema'),
  join(import.meta.dirname, 'docs-data.ts'),
  join(ROOT, 'apps', 'site', 'src', 'docs', 'wire.ts'),
]

function newest(path: string): number {
  if (!existsSync(path)) return 0
  const s = statSync(path)
  if (!s.isDirectory()) return s.mtimeMs
  // A directory's own time changes when an entry is added or removed.
  let t = s.mtimeMs
  for (const name of readdirSync(path)) {
    if (name === 'node_modules' || name.startsWith('.')) continue
    t = Math.max(t, newest(join(path, name)))
  }
  return t
}

/** The current manifest, or `null` when nothing has been written. */
export function readManifest(): WireManifest | null {
  try {
    return JSON.parse(readFileSync(MANIFEST, 'utf8')) as WireManifest
  } catch {
    return null
  }
}

/** Whether the files are missing or older than a source (a walk of a few thousand files: tens of milliseconds). */
export function isStale(): boolean {
  if (readManifest() === null) return true
  const built = statSync(MANIFEST).mtimeMs
  return SOURCES.some((s) => newest(s) > built)
}

function firstSentence(text: string): string {
  const end = text.search(/[.:](\s|$)/)
  const s = end < 0 ? text : text.slice(0, end + 1)
  return s.length > 200 ? `${s.slice(0, 197)}…` : s
}

type Symbol = Catalogue['symbols'][number]

const TYPE_KINDS = new Set(['interface', 'type', 'class', 'enum'])
const SIGNATURE_LINES = 30

/** A declaration as a hover shows it: at most {@link SIGNATURE_LINES} lines. */
function shortSignature(signature: string): string {
  const lines = signature.split('\n')
  return lines.length <= SIGNATURE_LINES ? signature : `${lines.slice(0, SIGNATURE_LINES - 1).join('\n')}\n  …`
}

/** The named imports of a source text: local name to the specifier and the imported name. */
function importsOf(text: string): Map<string, { readonly from: string; readonly name: string }> {
  const out = new Map<string, { readonly from: string; readonly name: string }>()
  for (const m of text.matchAll(/^import\s+(?:type\s+)?\{([^}]*)\}\s*from\s*['"]([^'"]+)['"]/gm)) {
    const from = m[2] as string
    for (const part of (m[1] as string).split(',')) {
      const [name, local = name] = part
        .trim()
        .replace(/^type\s+/, '')
        .split(/\s+as\s+/)
      if (name !== undefined && name !== '' && local !== undefined) out.set(local, { from, name })
    }
  }
  return out
}

/** The code of a symbol's docs (signatures and parameter types, its members' too), where type names appear. */
function codeOf(s: Symbol): string {
  return [
    s.signature,
    ...s.params.map((p) => p.type),
    ...s.members.flatMap((m) => [m.signature, ...m.params.map((p) => p.type)]),
  ].join('\n')
}

/**
 * Resolve the type names in each file's signatures: a type of the file itself, else the import of that name (a module
 * file or a barrel, followed to the type of that name in the barrel's folder, or a package specifier), else a type of
 * that name in the same module. Anything else (a global such as `Float64Array`, a private type) is left out.
 */
function typeRefs(c: Catalogue, symbolsOf: ReadonlyMap<string, readonly Symbol[]>): Map<string, WireRefs> {
  const byName = new Map<string, Symbol[]>()
  for (const s of c.symbols) if (TYPE_KINDS.has(s.kind)) byName.set(s.name, [...(byName.get(s.name) ?? []), s])
  const out = new Map<string, WireRefs>()
  for (const f of c.files) {
    const symbols = symbolsOf.get(f.path) ?? []
    let imports: Map<string, { readonly from: string; readonly name: string }>
    try {
      imports = importsOf(readFileSync(join(ROOT, f.path), 'utf8'))
    } catch {
      imports = new Map()
    }
    const words = new Set(
      symbols
        .map(codeOf)
        .join('\n')
        .match(/\b[A-Z][\w$]*\b/g) ?? [],
    )
    const refs: Record<string, readonly [string, string]> = {}
    for (const word of words) {
      const own = symbols.find((s) => s.name === word && TYPE_KINDS.has(s.kind))
      let target: Symbol | undefined = own
      const imp = own === undefined ? imports.get(word) : undefined
      if (imp !== undefined) {
        const named = byName.get(imp.name) ?? []
        if (imp.from.startsWith('.')) {
          const file = posix.normalize(posix.join(posix.dirname(f.path), imp.from))
          const dir = file.endsWith('/index.ts') ? file.slice(0, -'index.ts'.length) : undefined
          target =
            named.find((s) => s.source.file === file) ??
            (dir === undefined ? undefined : named.find((s) => s.source.file.startsWith(dir)))
        } else target = named.find((s) => s.module === imp.from)
        target ??= named[0]
      }
      if (target === undefined && imp === undefined) target = byName.get(word)?.find((s) => s.module === f.module)
      if (target !== undefined) refs[word] = [target.module, target.name]
    }
    out.set(f.path, refs)
  }
  return out
}

/** The summary up to its first full stop (a colon does not end it), at most 240 characters. */
function leadSentence(text: string): string {
  const end = text.search(/\.(\s|$)/)
  const s = (end < 0 ? text : text.slice(0, end + 1)).replace(/\s+/g, ' ')
  return s.length > 240 ? `${s.slice(0, 237)}…` : s
}

function hashed(text: string): string {
  return createHash('sha256').update(text).digest('hex').slice(0, 16)
}

/** Sizes in bytes of what {@link writeDocsData} wrote. */
export interface DocsDataReport {
  readonly manifest: WireManifest
  readonly index: number
  readonly search: number
  readonly files: number
  readonly largestFile: number
  readonly types: number
}

/**
 * Write the catalogue as the docs' static files.
 *
 * @param c - The catalogue (`extractCatalogue({ root, packages: ['engine'] })`).
 * @returns The manifest and the sizes written.
 */
export function writeDocsData(c: Catalogue): DocsDataReport {
  mkdirSync(join(OUT, 'files'), { recursive: true })
  mkdirSync(join(OUT, 'types'), { recursive: true })
  const written = new Set<string>()
  const write = (name: string, text: string): void => {
    written.add(name)
    const path = join(OUT, name)
    // Content-addressed: an existing file already holds this text.
    if (!existsSync(path)) writeFileSync(path, text)
  }
  const symbolsOf = new Map<string, Symbol[]>()
  for (const s of c.symbols) {
    const list = symbolsOf.get(s.source.file) ?? []
    list.push(s)
    symbolsOf.set(s.source.file, list)
  }
  const refs = typeRefs(c, symbolsOf)
  let largestFile = 0
  let fileBytes = 0
  const files: WireFile[] = c.files.map((f) => {
    const symbols = symbolsOf.get(f.path) ?? []
    const data: WireFileData = { description: f.description, symbols, refs: refs.get(f.path) ?? {} }
    const text = JSON.stringify(data)
    const name = `files/${hashed(text)}.json`
    write(name, text)
    largestFile = Math.max(largestFile, text.length)
    fileBytes += text.length
    return {
      path: f.path.slice(SRC.length),
      module: f.module,
      summary: f.summary,
      data: name,
      symbols: symbols.map((s) => [s.name, s.kind] as const),
    }
  })
  const search: WireSearch = {
    symbols: c.files.map((f) => (symbolsOf.get(f.path) ?? []).map((s) => firstSentence(s.summary))),
  }
  const searchText = JSON.stringify(search)
  const searchName = `search.${hashed(searchText)}.json`
  write(searchName, searchText)
  const typesOf = new Map<string, Record<string, WireType>>()
  for (const s of c.symbols) {
    if (!TYPE_KINDS.has(s.kind)) continue
    const types = typesOf.get(s.module) ?? {}
    types[s.name] = {
      file: s.source.file.slice(SRC.length),
      kind: s.kind,
      signature: shortSignature(s.signature),
      summary: leadSentence(s.summary),
    }
    typesOf.set(s.module, types)
  }
  let typeBytes = 0
  // Paths relative to packages/engine/src, so the tree mirrors the directories: lib/<domain>/..., methods/<area>/...
  // Every folder with its summary; an entry point's own folder also names its module and its types file.
  const folders: WireFolder[] = c.folders.map((f) => {
    const path = f.path.slice(SRC.length)
    if (!f.entry) return { path, summary: f.summary }
    const folder = { path, module: f.module, summary: f.summary }
    const types = typesOf.get(f.module)
    if (types === undefined) return folder
    const text = JSON.stringify(types)
    const name = `types/${hashed(text)}.json`
    write(name, text)
    typeBytes += text.length
    return { ...folder, types: name }
  })
  const body = { search: searchName, packages: c.packages, folders, files }
  const version = hashed(JSON.stringify(body))
  const index: WireIndex = { version, ...body }
  const indexText = JSON.stringify(index)
  const indexName = `index.${version}.json`
  write(indexName, indexText)

  const manifest: WireManifest = { version, index: indexName, files: [...written].sort() }
  const previous = readManifest()
  if (previous !== null && previous.version !== version) writeFileSync(PREVIOUS, JSON.stringify(previous))
  const tmp = `${MANIFEST}.${process.pid}.tmp`
  writeFileSync(tmp, JSON.stringify(manifest))
  renameSync(tmp, MANIFEST)
  prune(manifest)
  return {
    manifest,
    index: indexText.length,
    search: searchText.length,
    files: fileBytes,
    largestFile,
    types: typeBytes,
  }
}

/** Remove the files neither the current nor the previous version names. */
function prune(manifest: WireManifest): void {
  const keep = new Set(manifest.files)
  try {
    for (const f of (JSON.parse(readFileSync(PREVIOUS, 'utf8')) as WireManifest).files) keep.add(f)
  } catch {
    // No previous version.
  }
  for (const dir of [OUT, join(OUT, 'files'), join(OUT, 'types')])
    for (const name of readdirSync(dir)) {
      const rel = relative(OUT, join(dir, name))
      if (!name.endsWith('.json') || name.startsWith('manifest') || keep.has(rel)) continue
      rmSync(join(dir, name), { force: true })
    }
}
