/**
 * Match the public API of the reference libraries against omai's public surface and write `docs/completeness/data/` (`match-<lib>.json` per library, `summary.json`).
 *
 *   node scripts/completeness-survey/match.ts                  match everything, write match.json and summary.json
 *   node scripts/completeness-survey/match.ts --review <lib> [area-prefix]   print the entries with no automatic match, with
 *                                                      the nearest omai names, for writing verdicts
 *   node scripts/completeness-survey/match.ts --recheck <lib>   the same for entries judged absent or partial that have
 *                                                      candidates: a second look before a gap is reported
 *
 * Inputs: `docs/completeness/data/ref-<lib>.json` (from `enumerate.py`, introspection), `scripts/completeness-survey/memory.json`
 * (libraries that are not installed, listed from memory) and `scripts/completeness-survey/verdicts.json` (hand-kept rules).
 *
 * A reference name is matched in this order: (1) a verdict rule that names it; (2) an omai export whose name is the
 * same after dropping case and underscores (`LogisticRegression` and `logisticRegression`, `lr_scheduler` and
 * `lrScheduler`; `rgb2hsv` is also tried as `rgbToHsv`). Anything else is "none" and is reported with candidate omai names (shared word pieces) so a person can
 * decide. Candidates are never counted as matches. Only names are compared, not behaviour.
 */
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { scanOmai } from '../aifn-usage/scan-omai.ts'

const ROOT = resolve(import.meta.dirname, '..', '..')
const DATA = join(ROOT, 'docs', 'completeness', 'data')
const HERE = import.meta.dirname

export type Status = 'full' | 'partial' | 'absent' | 'skip'

interface RefEntry {
  readonly lib: string
  readonly area: string
  readonly name: string
  readonly kind: string
  readonly doc: string
  readonly source: 'introspection' | 'memory'
}

/** A hand-written rule. `name` is a regular expression on the reference name; `area` a prefix of the area. */
interface Verdict {
  readonly lib: string
  readonly name: string
  readonly area?: string | undefined
  readonly status: Status
  /** The nearest omai thing: `omai-engine/linear#linearModel` or free text for a concept. */
  readonly omai?: string | undefined
  readonly note?: string | undefined
  /** Proposed home (domain and file) for absent and partial entries. */
  readonly home?: string | undefined
  readonly size?: 'S' | 'M' | 'L' | undefined
  readonly priority?: 'high' | 'medium' | 'low' | undefined
  /** Group label in the family file ("add: ..." entries that belong together share it). */
  readonly group?: string | undefined
}

interface Result {
  readonly lib: string
  readonly area: string
  readonly name: string
  readonly kind: string
  readonly doc: string
  readonly source: string
  readonly status: Status
  readonly how: 'name' | 'verdict' | 'none'
  readonly omai?: string | undefined
  readonly note?: string | undefined
  readonly home?: string | undefined
  readonly size?: string | undefined
  readonly priority?: string | undefined
  readonly group?: string | undefined
}

const norm = (s: string): string => s.toLowerCase().replace(/[^a-z0-9]/g, '')

const STOP = new Set([
  'regressor',
  'classifier',
  'regression',
  'classification',
  'model',
  'models',
  'base',
  'estimator',
  'cv',
  'layer',
  'with',
  'and',
  'the',
  'of',
  'for',
  'from',
  'to',
  'in',
  'on',
  'by',
  'make',
  'get',
  'fit',
  'score',
  'is',
  'check',
  'plot',
  'test',
  'tests',
  'sample',
  'loss',
  'scorer',
  'transform',
  'transformer',
  'mixin',
  'dist',
  'distribution',
  'function',
  'class',
  'init',
  'utils',
  'util',
  'create',
  'compute',
  'calc',
  'calculate',
  'estimate',
  'conv',
]) // fmt: skip

function tokens(s: string): string[] {
  return s
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .split(/[^A-Za-z0-9]+/)
    .map((t) => t.toLowerCase())
    .filter(Boolean)
}

function loadRefs(): RefEntry[] {
  const out: RefEntry[] = []
  for (const f of readdirSync(DATA).filter((x) => x.startsWith('ref-') && x.endsWith('.json'))) {
    const d = JSON.parse(readFileSync(join(DATA, f), 'utf8')) as {
      library: string
      entries: { area: string; name: string; kind: string; doc: string }[]
    }
    for (const e of d.entries) out.push({ lib: d.library, ...e, source: 'introspection' })
  }
  const memPath = join(HERE, 'memory.json')
  if (existsSync(memPath)) {
    const mem = JSON.parse(readFileSync(memPath, 'utf8')) as {
      libraries: Record<string, Record<string, string>>
    }
    for (const [lib, areas] of Object.entries(mem.libraries)) {
      for (const [area, names] of Object.entries(areas)) {
        for (const name of names.split(/\s+/).filter(Boolean)) {
          out.push({ lib, area, name, kind: 'api', doc: '', source: 'memory' })
        }
      }
    }
  }
  // An aggregate module (`statsmodels.api`, `networkx.algorithms`) repeats names that a specific module also lists:
  // keep the occurrence with the longest area.
  const best = new Map<string, RefEntry>()
  for (const e of out) {
    const k = `${e.lib}|${e.name}`
    const cur = best.get(k)
    if (!cur || e.area.length > cur.area.length) best.set(k, e)
  }
  return out.filter((e) => best.get(`${e.lib}|${e.name}`) === e)
}

/**
 * Verdict files: `scripts/completeness-survey/verdicts/<lib>.txt`, one rule per line, fields separated by ` | `:
 *
 *   status | name regex | omai (nearest thing) | note | home | size | priority | group
 *
 * `status` is F (covered), P (partial), A (absent) or S (skip: out of scope or not an API a learner uses). Fields are
 * separated by a pipe with a space on each side (so `a|b` inside a regex is fine; an empty field is `| |`). Trailing
 * fields may be left out. A line `@lib name` sets the library and `@area prefix` restricts the following rules to
 * areas that start with the prefix (`@area` alone clears it). The first rule that matches wins. `#` starts a comment.
 */
function loadVerdicts(): Verdict[] {
  const dir = join(HERE, 'verdicts')
  if (!existsSync(dir)) return []
  const out: Verdict[] = []
  const map: Record<string, Status> = { F: 'full', P: 'partial', A: 'absent', S: 'skip' }
  for (const f of readdirSync(dir)
    .filter((x) => x.endsWith('.txt'))
    .sort()) {
    let lib = f.replace(/\.txt$/, '')
    let area: string | undefined
    for (const raw of readFileSync(join(dir, f), 'utf8').split('\n')) {
      const line = raw.trim()
      if (!line || line.startsWith('#')) continue
      if (line.startsWith('@lib ')) {
        lib = line.slice(5).trim()
        area = undefined
        continue
      }
      if (line.startsWith('@area')) {
        area = line.slice(5).trim() || undefined
        continue
      }
      const [st, name, omai, note, home, size, priority, group, ...extra] = `${line} `
        .split(/(?<=\s)\|(?=\s)/)
        .map((x) => x.trim())
      const status = map[st ?? '']
      if (
        !status ||
        !name ||
        extra.some(Boolean) ||
        (size && !'SML'.includes(size)) ||
        (priority && !['high', 'medium', 'low'].includes(priority))
      ) {
        throw new Error(`${f}: bad verdict line: ${line}`)
      }
      out.push({
        lib,
        name,
        area,
        status,
        omai: omai || undefined,
        note: note || undefined,
        home: home || undefined,
        size: (size as Verdict['size']) || undefined,
        priority: (priority as Verdict['priority']) || undefined,
        group: group || undefined,
      })
    }
  }
  return out
}

async function main(): Promise<void> {
  const refs = loadRefs()
  const verdicts = loadVerdicts().map((v) => ({ v, re: new RegExp(`^(?:${v.name})$`, 'i') }))
  const omai = (await scanOmai()).filter((s) => s.module.startsWith('omai-engine'))
  const byNorm = new Map<string, string[]>()
  const tokIndex: { id: string; norm: string; toks: Set<string> }[] = []
  for (const s of omai) {
    const id = `${s.module}#${s.name}`
    const n = norm(s.name)
    byNorm.set(n, [...(byNorm.get(n) ?? []), id])
    tokIndex.push({ id, norm: n, toks: new Set(tokens(s.name)) })
  }
  const homeRank = (id: string): number => (id.includes('/learn') || id.includes('/linear') ? 0 : 1)

  const recheck = process.argv.indexOf('--recheck')
  const review = recheck >= 0 ? recheck : process.argv.indexOf('--review')
  const reviewLib = review >= 0 ? process.argv[review + 1] : undefined
  const reviewArea = review >= 0 ? process.argv[review + 2] : undefined

  const results: Result[] = []
  for (const r of refs) {
    const base = { lib: r.lib, area: r.area, name: r.name, kind: r.kind, doc: r.doc, source: r.source }
    const rule = verdicts.find(
      ({ v, re }) => v.lib === r.lib && re.test(r.name) && (!v.area || r.area.startsWith(v.area)),
    )
    if (rule) {
      const v = rule.v
      results.push({
        ...base,
        status: v.status,
        how: 'verdict',
        omai: v.omai,
        note: v.note,
        home: v.home,
        size: v.size,
        priority: v.priority,
        group: v.group,
      })
      continue
    }
    const last = r.name.split('.').pop() ?? r.name
    const hits = byNorm.get(norm(last)) ?? byNorm.get(norm(last.replace(/(?<=[a-z])2(?=[a-z])/, 'to')))
    if (hits && norm(r.name).length >= 3) {
      const pick = [...hits].sort((a, b) => homeRank(a) - homeRank(b))
      results.push({ ...base, status: 'full', how: 'name', omai: pick.slice(0, 2).join(' | ') })
      continue
    }
    results.push({ ...base, status: 'absent', how: 'none' })
  }

  if (reviewLib) {
    let n = 0
    for (const r of results) {
      if (r.lib !== reviewLib) continue
      if (recheck >= 0 ? !(r.status === 'absent' || r.status === 'partial') : r.how !== 'none') continue
      if (reviewArea && !r.area.startsWith(reviewArea)) continue
      const toks = tokens(r.name).filter((t) => !STOP.has(t) && t.length >= 3)
      const rn = norm(r.name)
      const cands: string[] = []
      if (toks.length > 0) {
        for (const t of tokIndex) {
          const all = toks.every((k) => t.toks.has(k) || (k.length >= 5 && t.norm.includes(k)))
          if (all || (rn.length >= 6 && (t.norm.includes(rn) || (t.norm.length >= 6 && rn.includes(t.norm))))) {
            cands.push(t.id.replace('omai-engine/', ''))
          }
          if (cands.length >= 4) break
        }
      }
      if (recheck >= 0 && cands.length === 0) continue
      console.log(`${r.area.replace(/^[a-z_]+\./, '')}\t${r.name}\t${cands.join(' ; ')}`)
      n++
    }
    console.error(`${n} unmatched`)
    return
  }

  // write
  const libs = [...new Set(results.map((r) => r.lib))]
  for (const lib of libs) {
    const lines = results.filter((r) => r.lib === lib).map((r) => JSON.stringify(compact(r)))
    writeFileSync(join(DATA, `match-${lib}.json`), `[\n${lines.join(',\n')}\n]\n`)
  }
  const summary: Record<string, Record<string, Record<Status, number>>> = {}
  for (const r of results) {
    const lib = summary[r.lib] ?? {}
    summary[r.lib] = lib
    const a = lib[r.area] ?? { full: 0, partial: 0, absent: 0, skip: 0 }
    lib[r.area] = a
    a[r.status]++
  }
  const sources: Record<string, string> = {}
  for (const r of results) sources[r.lib] = r.source
  writeFileSync(
    join(DATA, 'summary.json'),
    `${JSON.stringify({ omaiExports: omai.length, sources, summary }, null, 1)}\n`,
  )
  const total: Record<Status, number> = { full: 0, partial: 0, absent: 0, skip: 0 }
  for (const r of results) total[r.status]++
  console.log(`omai exports: ${omai.length}; reference entries: ${results.length}`, total)
  const unreviewed = results.filter((r) => r.how === 'none').length
  console.log(`${unreviewed} absent without a verdict (not yet reviewed)`)
}

function compact(r: Result): Record<string, string | undefined> {
  const o: Record<string, string | undefined> = {
    lib: r.lib,
    area: r.area,
    name: r.name,
    status: r.status,
    how: r.how,
    omai: r.omai,
    note: r.note,
    home: r.home,
    size: r.size,
    priority: r.priority,
    group: r.group,
  }
  for (const k of Object.keys(o)) if (o[k] === undefined) delete o[k]
  return o
}

await main()
