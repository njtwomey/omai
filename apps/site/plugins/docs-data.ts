/**
 * The docs catalogue precompiled to static files, so no page waits for an extraction (`make docs-catalogue`, run by
 * `make dev` and `make site`; the dev server's plugin reruns it in the background after an engine edit).
 *
 * Into `apps/site/.generated/docs/` (gitignored) it writes, each named by a hash of its content:
 * - `index.<hash>.json`: the tree (folders with summaries; files with their opening sentence and their exports' names
 *   and kinds), a few hundred kilobytes where the full catalogue is several megabytes;
 * - `search.<hash>.json`: every export's one-sentence summary, fetched when someone searches;
 * - `files/<hash>.json`: one per source file, its full comment and its exports' `SymbolDoc`s;
 * - `manifest.json` last (by rename), naming the current index, so a reader never sees half a version.
 *
 * Files of the current and the previous version are kept (a page open on the previous index can still load its files);
 * older ones are removed.
 */
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { join, relative } from 'node:path'
import type { Catalogue } from '../../../packages/meta/src/schema/index.ts'
import type { WireFile, WireFileData, WireIndex, WireManifest, WireSearch } from '../src/docs/wire.ts'

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
}

/**
 * Write the catalogue as the docs' static files.
 *
 * @param c - The catalogue (`extractCatalogue({ root, packages: ['engine'] })`).
 * @returns The manifest and the sizes written.
 */
export function writeDocsData(c: Catalogue): DocsDataReport {
  mkdirSync(join(OUT, 'files'), { recursive: true })
  const written = new Set<string>()
  const write = (name: string, text: string): void => {
    written.add(name)
    const path = join(OUT, name)
    // Content-addressed: an existing file already holds this text.
    if (!existsSync(path)) writeFileSync(path, text)
  }
  const symbolsOf = new Map<string, Catalogue['symbols'][number][]>()
  for (const s of c.symbols) {
    const list = symbolsOf.get(s.source.file) ?? []
    list.push(s)
    symbolsOf.set(s.source.file, list)
  }
  let largestFile = 0
  let fileBytes = 0
  const files: WireFile[] = c.files.map((f) => {
    const symbols = symbolsOf.get(f.path) ?? []
    const data: WireFileData = { description: f.description, symbols }
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
  const body = {
    search: searchName,
    packages: c.packages,
    // Paths relative to packages/engine/src, so the tree mirrors the directories: lib/<domain>/..., methods/<area>/...
    folders: c.modules.map((m) => ({
      path: m.layer === 'core' ? `lib/${m.path}` : m.path,
      module: m.id,
      summary: m.summary,
    })),
    files,
  }
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
  return { manifest, index: indexText.length, search: searchText.length, files: fileBytes, largestFile }
}

/** Remove the files neither the current nor the previous version names. */
function prune(manifest: WireManifest): void {
  const keep = new Set(manifest.files)
  try {
    for (const f of (JSON.parse(readFileSync(PREVIOUS, 'utf8')) as WireManifest).files) keep.add(f)
  } catch {
    // No previous version.
  }
  for (const dir of [OUT, join(OUT, 'files')])
    for (const name of readdirSync(dir)) {
      const rel = relative(OUT, join(dir, name))
      if (!name.endsWith('.json') || name.startsWith('manifest') || keep.has(rel)) continue
      rmSync(join(dir, name), { force: true })
    }
}
