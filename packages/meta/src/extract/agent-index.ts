/**
 * The agent index: one extraction of the `@index` tags in the engine library and several renderings of it. The
 * collector reads the tags (files and symbols) with the folder summaries; the renderers write `llms.txt` (a hand-written
 * usage note, then one line per domain), one Markdown file per domain, and a JSON file with every entry for tools. All
 * output is a pure function of the source: sorted, with no timestamps, so it is stable across machines.
 */
import { readFileSync } from 'node:fs'
import { basename, join } from 'node:path'
import { INDEX_KINDS, type IndexEntry, type IndexProblem } from '../schema/agent-index.ts'
import { readProject } from './project.ts'
import { readSymbols } from './symbols.ts'

/** Where the library lives in the package, and where the generated files go (relative to the package root). */
export const AGENT_INDEX_LAYOUT = {
  package: 'engine',
  lib: 'src/lib',
  header: 'llms.header.md',
  llms: 'llms.txt',
  dir: 'index',
  json: 'index/index.json',
} as const

/** What collecting the index finds. */
export interface CollectedIndex {
  /** The package name, such as `mdml-engine`. */
  readonly packageName: string
  readonly entries: readonly IndexEntry[]
  /** Folder summaries by path relative to `lib/` (`numerics`, `numerics/linalg`). */
  readonly summaries: Readonly<Record<string, string>>
  /** Errors and warnings about the tags, with file:line (repository-relative paths). */
  readonly problems: readonly IndexProblem[]
}

/** The pieces the renderers need: the collected index and the hand-written usage note. */
export interface RenderInput {
  readonly packageName: string
  readonly entries: readonly IndexEntry[]
  readonly summaries: Readonly<Record<string, string>>
  /** The hand-written opening of `llms.txt` (`llms.header.md`). */
  readonly header: string
}

const KIND_ORDER = new Map<string, number>(INDEX_KINDS.map((k, i) => [k, i]))

/** Code-unit comparison: the same order on every machine (no locale). */
function compare(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0
}

/** The order of entries within a folder: files first, then by kind, name and line. */
function compareEntries(a: IndexEntry, b: IndexEntry): number {
  if (a.level !== b.level) return a.level === 'file' ? -1 : 1
  const k = (KIND_ORDER.get(a.kind) ?? 0) - (KIND_ORDER.get(b.kind) ?? 0)
  return k !== 0 ? k : compare(a.name, b.name) || a.line - b.line
}

/** The order of entries over the whole index: by domain, then folder, then {@link compareEntries}. */
function compareAll(a: IndexEntry, b: IndexEntry): number {
  return compare(a.domain, b.domain) || compare(a.folder, b.folder) || compareEntries(a, b)
}

/**
 * Read the `@index` tags of the engine library: every tagged file (its opening paragraph) and symbol, with the folder
 * summaries and the problems found. A key (import path plus name, per level) that two entries share is a problem.
 *
 * @param root - The repository root.
 * @returns The entries sorted for output, the summaries and the problems.
 */
export function collectAgentIndex(root: string): CollectedIndex {
  const project = readProject(root, [AGENT_INDEX_LAYOUT.package])
  const { files, symbols, folders, indexProblems } = readSymbols(project)
  const pkgDir = `packages/${AGENT_INDEX_LAYOUT.package}/`
  const libDir = `${pkgDir}${AGENT_INDEX_LAYOUT.lib}/`
  const packageName = project.packages[0]?.name ?? 'mdml-engine'
  const entries: IndexEntry[] = []
  const place = (file: string): { domain: string; folder: string; path: string } | undefined => {
    if (!file.startsWith(libDir)) return undefined
    const rel = file.slice(libDir.length)
    const slash = rel.lastIndexOf('/')
    const folder = slash < 0 ? '' : rel.slice(0, slash)
    return { domain: rel.split('/')[0] as string, folder, path: file.slice(pkgDir.length) }
  }
  for (const s of symbols) {
    const where = place(s.source.file)
    if (s.index === undefined || where === undefined) continue
    entries.push({
      kind: s.index.kind,
      name: s.name,
      level: 'symbol',
      importPath: s.module,
      path: where.path,
      line: s.source.line,
      description: s.index.description,
      domain: where.domain,
      folder: where.folder,
    })
  }
  for (const f of files) {
    const where = place(f.path)
    if (f.index === undefined || where === undefined) continue
    entries.push({
      kind: f.index.kind,
      name: basename(f.path),
      level: 'file',
      importPath: f.module,
      path: where.path,
      line: 1,
      description: f.index.description,
      domain: where.domain,
      folder: where.folder,
    })
  }
  entries.sort(compareAll)
  const problems: IndexProblem[] = indexProblems.filter((p) => p.file.startsWith(libDir))
  const seen = new Map<string, IndexEntry>()
  for (const e of entries) {
    const key = `${e.level} ${e.importPath}#${e.name}`
    const first = seen.get(key)
    if (first === undefined) seen.set(key, e)
    else
      problems.push({
        severity: 'error',
        file: `${pkgDir}${e.path}`,
        line: e.line,
        message: `${e.level} entry ${e.importPath}#${e.name} is also tagged at ${first.path}:${first.line}`,
      })
  }
  const summaries: Record<string, string> = {}
  for (const f of folders) {
    if (f.path.startsWith(libDir)) summaries[f.path.slice(libDir.length)] = f.summary
  }
  problems.sort((a, b) => compare(a.file, b.file) || a.line - b.line)
  return { packageName, entries, summaries, problems }
}

/** The longest `llms.txt` line for a domain, before whole sentences stop being added. */
const DOMAIN_LINE = 260

/** The last place at most `limit` characters in where a text can be cut: after a comma, semicolon or colon outside parentheses and code. */
function cutPoint(text: string, limit: number): number {
  let depth = 0
  let ticks = 0
  let best = -1
  for (let i = 0; i < Math.min(text.length, limit); i++) {
    const c = text[i]
    if (c === '`') ticks++
    else if (ticks % 2 === 1) continue
    else if (c === '(') depth++
    else if (c === ')') depth = Math.max(0, depth - 1)
    else if (depth === 0 && (c === ',' || c === ';' || c === ':')) best = i
  }
  return best
}

/**
 * The opening of a folder summary that fits on one `llms.txt` line: whole sentences while they fit in
 * {@link DOMAIN_LINE} characters. When the first sentence alone is longer it is cut after the last comma, semicolon or
 * colon that fits (outside parentheses and code) and ends with `...`.
 *
 * @param summary - A folder summary.
 * @returns The opening, on one line.
 * @example
 * ```ts
 * import { openingSentences } from 'mdml-meta/extract'
 * openingSentences('One. Two.') // => 'One. Two.'
 * ```
 */
export function openingSentences(summary: string): string {
  const sentences = summary.split(/(?<=[.!?])\s+(?=[A-Z`(])/)
  let out = sentences[0] ?? ''
  if (out.length > DOMAIN_LINE) {
    const at = cutPoint(out, DOMAIN_LINE)
    const cut = at > 0 ? out.slice(0, at) : out.slice(0, DOMAIN_LINE).replace(/\s+\S*$/, '')
    return `${cut} ...`
  }
  for (const next of sentences.slice(1)) {
    if (out.length + 1 + next.length > DOMAIN_LINE) break
    out += ` ${next}`
  }
  return out
}

/** The one-line form of an entry: kind, name, description, `path:line` and the import path. */
function entryLine(e: IndexEntry): string {
  const what = e.level === 'file' ? `\`${e.name}\` (file)` : `\`${e.name}\``
  return `- ${e.kind} ${what}: ${e.description} \`${e.path}:${e.line}\` \`${e.importPath}\``
}

/** The entries of one domain as Markdown: the domain summary, then its folders as nested sections. */
function renderDomain(
  domain: string,
  entries: readonly IndexEntry[],
  summaries: Readonly<Record<string, string>>,
): string {
  const byFolder = new Map<string, IndexEntry[]>()
  for (const e of entries) byFolder.set(e.folder, [...(byFolder.get(e.folder) ?? []), e])
  const folders = [...byFolder.keys()].sort(compare)
  const out: string[] = [
    `<!-- Generated by scripts/agent-index.ts from the @index tags in packages/engine/src/lib/${domain}. Do not edit. -->`,
    '',
    `# ${domain}`,
    '',
  ]
  const summary = summaries[domain]
  if (summary !== undefined && summary !== '') out.push(summary, '')
  for (const folder of folders) {
    const mine = (byFolder.get(folder) ?? []).sort(compareEntries)
    if (folder !== domain) {
      // Depth below the domain among the folders that have entries: a folder nested in another listed folder is deeper.
      const depth = folders.filter((f) => f !== domain && folder.startsWith(`${f}/`)).length
      out.push(`${'#'.repeat(2 + depth)} ${folder}`, '')
      const s = summaries[folder]
      if (s !== undefined && s !== '') out.push(s, '')
    }
    let previous = ''
    for (const e of mine) {
      const group = `${e.level}:${e.kind}`
      if (previous !== '' && previous !== group) out.push('')
      out.push(entryLine(e))
      previous = group
    }
    out.push('')
  }
  return `${out.join('\n').replace(/\n+$/, '')}\n`
}

/** The JSON file: every entry on a line of its own, so a change to one entry is a change to one line. */
function renderJson(packageName: string, entries: readonly IndexEntry[]): string {
  const rows = entries.map((e) =>
    JSON.stringify({
      kind: e.kind,
      name: e.name,
      level: e.level,
      importPath: e.importPath,
      path: e.path,
      line: e.line,
      description: e.description,
      domain: e.domain,
      folder: e.folder,
    }),
  )
  const head = `{\n  "version": 1,\n  "package": ${JSON.stringify(packageName)},\n  "kinds": ${JSON.stringify(INDEX_KINDS)},\n`
  return `${head}  "entries": [\n${rows.map((r) => `    ${r}`).join(',\n')}\n  ]\n}\n`
}

/**
 * Render the index outputs: `llms.txt`, `index/<domain>.md` for every domain with entries, and `index/index.json`.
 *
 * @param input - The entries, folder summaries and the hand-written usage note.
 * @returns The files by path relative to the package root. Deterministic: the same input gives the same text.
 * @example
 * ```ts
 * import { renderAgentIndex } from 'mdml-meta/extract'
 * const out = renderAgentIndex({ packageName: 'x', entries: [], summaries: {}, header: '# x\n' })
 * [...out.keys()] // => ['index/index.json', 'llms.txt']
 * ```
 */
export function renderAgentIndex(input: RenderInput): Map<string, string> {
  const out = new Map<string, string>()
  const entries = [...input.entries].sort(compareAll)
  const byDomain = new Map<string, IndexEntry[]>()
  for (const e of entries) byDomain.set(e.domain, [...(byDomain.get(e.domain) ?? []), e])
  const domains = [
    ...new Set([...Object.keys(input.summaries).filter((p) => !p.includes('/')), ...byDomain.keys()]),
  ].sort(compare)
  const lines = ['## Domains', '']
  for (const d of domains) {
    const mine = byDomain.get(d)
    const summary = input.summaries[d] ?? ''
    if (mine === undefined) lines.push(`- ${d}: ${openingSentences(summary)} (no entries yet)`)
    else {
      lines.push(
        `- [${d}](${AGENT_INDEX_LAYOUT.dir}/${d}.md): ${openingSentences(summary)} (${mine.length} ${mine.length === 1 ? 'entry' : 'entries'})`,
      )
      out.set(`${AGENT_INDEX_LAYOUT.dir}/${d}.md`, renderDomain(d, mine, input.summaries))
    }
  }
  lines.push('', '## For tools', '', `- [${AGENT_INDEX_LAYOUT.json}](${AGENT_INDEX_LAYOUT.json}): every entry as JSON.`)
  out.set(AGENT_INDEX_LAYOUT.llms, `${input.header.replace(/\n+$/, '')}\n\n${lines.join('\n')}\n`)
  out.set(AGENT_INDEX_LAYOUT.json, renderJson(input.packageName, entries))
  return new Map([...out].sort((a, b) => compare(a[0], b[0])))
}

/**
 * Collect the index of a repository and render it.
 *
 * @param root - The repository root.
 * @returns The collected index (with its problems) and the rendered files by path relative to the engine package.
 */
export function buildAgentIndex(root: string): { collected: CollectedIndex; files: Map<string, string> } {
  const collected = collectAgentIndex(root)
  const header = readFileSync(join(root, 'packages', AGENT_INDEX_LAYOUT.package, AGENT_INDEX_LAYOUT.header), 'utf8')
  const files = renderAgentIndex({
    packageName: collected.packageName,
    entries: collected.entries,
    summaries: collected.summaries,
    header,
  })
  return { collected, files }
}

/**
 * The generated files that are out of date: those whose content differs from what is on disk, those missing, and files
 * on disk that are no longer generated. This is what `scripts/agent-index.ts --check` reports.
 *
 * @param files - The files the index renders, by path relative to the package.
 * @param existing - The generated files now on disk (`llms.txt` and everything under `index/`), by the same paths.
 * @returns The stale paths, sorted; an orphan carries ` (no longer generated)`.
 * @example
 * ```ts
 * import { staleAgentIndex } from 'mdml-meta/extract'
 * staleAgentIndex(new Map([['a', '1'], ['b', '2']]), new Map([['a', '1'], ['b', 'old'], ['c', 'x']]))
 * // => ['b', 'c (no longer generated)']
 * ```
 */
export function staleAgentIndex(files: ReadonlyMap<string, string>, existing: ReadonlyMap<string, string>): string[] {
  const stale: string[] = []
  for (const [path, content] of files) if (existing.get(path) !== content) stale.push(path)
  for (const path of existing.keys()) if (!files.has(path)) stale.push(`${path} (no longer generated)`)
  return stale.sort(compare)
}
