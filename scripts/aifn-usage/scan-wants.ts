/**
 * Read what the notes want next from their non-code content and write `docs/aifn/data/wants.json`.
 *
 *   node scripts/aifn-usage/scan-wants.ts
 *
 * Sources, all read-only: the frontmatter of every note (stubs are the notes not yet written), `.scratch/implementations/
 * GAPS.md` (the engine needs of the unwritten notes, with a signature sketch, a size and the notes that use each),
 * the engine requests and API sketches of `.scratch`, comments that say what the engine lacks, and the numerics the
 * notes wrote for themselves (`.ts` files of `content/notes`). Each sketched name is looked up in the AIFN surface and
 * in omai by exact name, which is a hint to check by hand, not a verdict.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { git, ML, nearestNames, writeJson } from './lib.ts'
import { scanAifn } from './scan-aifn.ts'
import { scanOmai } from './scan-omai.ts'
import { scanUsage } from './scan-usage.ts'

const NOTES = join(ML, 'content', 'notes')

function walk(dir: string, accept: (name: string) => boolean, out: string[] = []): string[] {
  if (!existsSync(dir)) return out
  for (const name of readdirSync(dir).sort()) {
    if (name === 'node_modules' || name === 'dist' || name === '.git') continue
    const p = join(dir, name)
    if (statSync(p).isDirectory()) walk(p, accept, out)
    else if (accept(name)) out.push(p)
  }
  return out
}

const NOT_NAMES = new Set(
  'Float64Array Float32Array Int32Array Uint8Array Array Promise Record Map Set Generator Partial Readonly Omit Pick Parameters ReturnType number string boolean void Object Math'.split(
    ' ',
  ),
)

/** Identifiers followed by `(`, in a signature or a code line: the names a sketch calls for. */
function calledNames(text: string): string[] {
  const out = new Set<string>()
  for (const m of text.matchAll(/\b([A-Za-z][A-Za-z0-9_]{2,})\s*[(<]/g)) {
    const n = m[1] as string
    if (!NOT_NAMES.has(n) && !/^[A-Z]/.test(n)) out.add(n)
  }
  return [...out]
}

interface Need {
  section: string
  area: string
  name: string
  size: string
  signature: string
  usedBy: string[]
  stubs: number
  written: number
  looksFor: string[]
  inAifnNow: string[]
  inOmai: string[]
  /** The closest omai names by shared words, a hint to check by hand. */
  nearOmai: string[]
}

async function main(): Promise<void> {
  const aifn = scanAifn()
  const omai = await scanOmai()
  const aifnNames = new Set(aifn.symbols.map((s) => s.name))
  const omaiNames = new Set(omai.map((s) => s.name))

  // Notes and their status.
  const notes = walk(NOTES, (n) => n === 'index.mdx').map((f) => {
    const text = readFileSync(f, 'utf8')
    const fm = /^---\n([\s\S]*?)\n---/.exec(text)?.[1] ?? ''
    const get = (k: string): string =>
      new RegExp(`^${k}:\\s*(.*)$`, 'm').exec(fm)?.[1]?.replace(/^['"]|['"]$/g, '') ?? ''
    const dir = relative(NOTES, f).replace(/\/index\.mdx$/, '')
    const parts = dir.split('/')
    return {
      slug: parts[parts.length - 1] as string,
      category: parts.slice(0, -1).join('/'),
      status: get('status'),
      title: get('title'),
      kind: get('kind'),
    }
  })
  const bySlug = new Map(notes.map((n) => [n.slug, n]))
  const statusCount: Record<string, number> = {}
  for (const n of notes) statusCount[n.status] = (statusCount[n.status] ?? 0) + 1
  const stubsByCategory = new Map<string, string[]>()
  for (const n of notes.filter((x) => x.status === 'stub')) {
    const key = n.category.split('/').slice(0, 3).join('/')
    stubsByCategory.set(key, [...(stubsByCategory.get(key) ?? []), n.slug])
  }
  const stubCategories = [...stubsByCategory.entries()]
    .map(([category, slugs]) => ({ category, stubs: slugs.length, examples: slugs.slice(0, 4) }))
    .sort((a, b) => b.stubs - a.stubs || (a.category < b.category ? -1 : 1))

  // GAPS.md: the needs of the notes not yet written.
  const needs: Need[] = []
  const gapsFile = join(ML, '.scratch', 'implementations', 'GAPS.md')
  if (existsSync(gapsFile)) {
    let section = ''
    let area = ''
    for (const line of readFileSync(gapsFile, 'utf8').split('\n')) {
      if (line.startsWith('## ')) section = line.slice(3, 4)
      else if (line.startsWith('### ')) area = line.slice(4).trim()
      else if (line.startsWith('| ') && !line.startsWith('| Name') && !line.startsWith('| ---')) {
        const cells = line.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, '|'))
        const [, name = '', size = '', signature = '', , usedBy = ''] = cells
        const users = [...usedBy.matchAll(/([a-z0-9-]+) \((\w+)\)/g)].map((m) => ({
          slug: m[1] as string,
          status: m[2] as string,
        }))
        const looksFor = [...new Set([name, ...calledNames(signature)])]
        needs.push({
          section,
          area,
          name,
          size,
          signature: signature.replace(/`/g, '').slice(0, 200),
          usedBy: users.map((u) => u.slug),
          stubs: users.filter((u) => (bySlug.get(u.slug)?.status ?? u.status) === 'stub').length,
          written: users.filter((u) => (bySlug.get(u.slug)?.status ?? u.status) !== 'stub').length,
          looksFor,
          inAifnNow: looksFor.filter((n) => aifnNames.has(n)),
          inOmai: looksFor.filter((n) => omaiNames.has(n)),
          nearOmai: nearestNames(name, omai, 0.5, 2).map(
            (x) => `${x.item.module.replace('omai-engine/', '')}#${x.item.name}`,
          ),
        })
      }
    }
  }

  // Engine requests and API sketches written by the note authors.
  const requestFiles = walk(join(ML, '.scratch'), (n) =>
    /^(engine-request.*|engine-api-sketches|engine-linear-paths)\.md$/.test(n),
  )
  const requests = requestFiles.map((f) => {
    const text = readFileSync(f, 'utf8')
    const names = new Set<string>()
    for (const m of text.matchAll(/```[a-z]*\n([\s\S]*?)```/g))
      for (const n of calledNames(m[1] as string)) names.add(n)
    for (const m of text.matchAll(/\*\*([a-z][A-Za-z0-9]+)\*\*\(/g)) names.add(m[1] as string)
    const list = [...names].sort()
    return {
      file: relative(ML, f),
      lines: text.split('\n').length,
      names: list.length,
      inAifnNow: list.filter((n) => aifnNames.has(n)).length,
      inOmai: list.filter((n) => omaiNames.has(n)).length,
      missingInOmai: list.filter((n) => !omaiNames.has(n)).slice(0, 80),
    }
  })

  // Comments that say what the engine lacks.
  const PHRASE =
    /candidate for the engine|engine (lacks|needs|is missing|does not|has no)|not (yet )?in the engine|TODO[^\n]{0,20}engine|needs? engine support|missing from the engine/i
  const comments: { file: string; line: number; text: string }[] = []
  for (const dir of ['content/notes', 'lab/src', 'site/src', 'scripts']) {
    for (const f of walk(join(ML, dir), (n) => /\.(tsx?|mdx)$/.test(n))) {
      readFileSync(f, 'utf8')
        .split('\n')
        .forEach((l, i) => {
          if (PHRASE.test(l)) comments.push({ file: relative(ML, f), line: i + 1, text: l.trim().slice(0, 180) })
        })
    }
  }

  // The numerics the notes wrote for themselves.
  const usage = scanUsage()
  const local = usage.localFiles
    .filter((x) => x.lines >= 120 && x.notes > 0)
    .map((x) => ({
      file: x.file,
      lines: x.lines,
      notes: x.notes,
      exports: x.exports.slice(0, 12),
      sameNameInOmai: x.exports.filter((n) => omaiNames.has(n)).length,
      sameNameInAifn: x.exports.filter((n) => aifnNames.has(n)).length,
    }))

  writeJson('wants.json', {
    sources: { ml: git(ML, 'rev-parse', '--short', 'HEAD'), aifn: aifn.commit },
    noteStatus: statusCount,
    stubCategories,
    needs,
    requests,
    comments,
    localCode: local,
  })
  console.log(
    `wants: ${notes.length} notes (${JSON.stringify(statusCount)}), ${needs.length} needs, ${requests.length} request files, ${comments.length} comments, ${local.length} local files`,
  )
}

await main()
