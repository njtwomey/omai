/**
 * The EP digest: the proof that a behaviour-preserving refactoring of the EP capability (the compiler, the message-passing
 * engine and rules, the atoms and methods) changed nothing. It computes, for every atom of the registry, every benchmark model
 * and every EP program the library can run without the pocs, the compiled factor graph (hash), the verdict, the beliefs and
 * evidence, the check rows and the variational run, and the registry of operators and knob spaces
 * (`scripts/ep-digest-core.ts`), and compares them to the committed baseline `docs/probprog/ep-digest.json`.
 *
 *   make ep-digest                      compute and write `.cache/ep-digest/` (digest.json and the full graphs)
 *   make ep-digest-check                compare to the baseline; exit 1 with what moved (which item, which field, old and new)
 *   make ep-digest-check ARGS="--exact"             fail unless graphs, verdicts, counts and numbers (12 digits) are identical
 *   make ep-digest-check ARGS="--only gaussian"     one atom, model or prefix (a name, or `atom/gaussian-mean`)
 *   make ep-digest-check ARGS="--accept"            rewrite the baseline: only for a deliberate behaviour change, in a commit of its own
 *
 * Levels reached by a comparison, from the strictest: `bits` (every number identical at full precision), `exact` (graphs,
 * verdicts, sweep, restart and improper counts identical, numbers equal to 12 significant digits), `tolerant` (beliefs within
 * 1e-9 relative, evidence within 1e-12, check statistics within 1e-9), or a failure. `--bits` and `--exact` fail below
 * their level; the default fails only below `tolerant`.
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import {
  compact,
  compare,
  computeDigest,
  type Digest,
  defaultTolerances,
  diffPaths,
  digestText,
  type GraphStore,
  type J,
  sha256,
} from './ep-digest-core.ts'
import { programItems } from './ep-digest-programs.ts'

const ROOT = join(import.meta.dirname, '..')
const BASELINE = join(ROOT, 'docs/probprog/ep-digest.json')
const args = process.argv.slice(2)
const flag = (name: string): boolean => args.includes(`--${name}`)
const option = (name: string): string | undefined => {
  const i = args.indexOf(`--${name}`)
  return i >= 0 ? args[i + 1] : undefined
}
const only = option('only')?.split(',')
const cache = resolve(ROOT, option('cache') ?? '.cache/ep-digest')
const baselinePath = resolve(ROOT, option('baseline') ?? BASELINE)
const tolerances = {
  belief: Number(option('tol-belief') ?? defaultTolerances.belief),
  floor: defaultTolerances.floor,
  evidence: Number(option('tol-evidence') ?? defaultTolerances.evidence),
  check: Number(option('tol-check') ?? defaultTolerances.check),
}

const out = (s: string): void => void process.stdout.write(`${s}\n`)
const fileOf = (key: string): string => `${key.replaceAll('/', '__')}.json`

/** The graphs of an item (its variants) as one JSON text: `{"ep":..., "rows":..., "vmp":...}`. */
function itemGraphs(store: GraphStore): Map<string, string> {
  const byItem = new Map<string, string[]>()
  for (const [k, text] of store) {
    const dot = k.lastIndexOf('.')
    const item = k.slice(0, dot)
    const list = byItem.get(item) ?? []
    list.push(`${JSON.stringify(k.slice(dot + 1))}:${text}`)
    byItem.set(item, list)
  }
  return new Map([...byItem].map(([k, v]) => [k, `{${v.sort().join(',')}}\n`]))
}

function writeGraphs(dir: string, store: GraphStore, replaceAll: boolean): void {
  if (replaceAll) rmSync(dir, { recursive: true, force: true })
  mkdirSync(dir, { recursive: true })
  for (const [item, text] of itemGraphs(store)) writeFileSync(join(dir, fileOf(item)), text)
}

const t0 = Date.now()
const progress = (() => {
  let n = 0
  return (key: string, ms: number): void => {
    n += 1
    out(`  ${String(n).padStart(3)}  ${key.padEnd(46)} ${(ms / 1000).toFixed(2)} s`)
  }
})()

out(`ep-digest: computing${only === undefined ? ' everything' : ` ${only.join(', ')}`} (serial, in this process)`)
const { digest, graphs } = await computeDigest({
  ...(only === undefined ? {} : { only }),
  ...(flag('coverage') ? { coverage: true } : {}),
  progress,
  extra: programItems(),
})
const kept = compact(digest)
const text = digestText(kept)
out(
  `ep-digest: ${Object.keys(kept.items).length} items in ${((Date.now() - t0) / 1000).toFixed(1)} s, digest sha256 ${sha256(text).slice(0, 16)}`,
)

if (flag('accept')) {
  if (only !== undefined) {
    out('ep-digest: --accept rewrites the whole baseline; run it without --only')
    process.exit(2)
  }
  writeFileSync(baselinePath, text)
  writeGraphs(join(cache, 'graphs'), graphs, true)
  out(
    `ep-digest: baseline written to ${baselinePath} (${(text.length / 1e6).toFixed(2)} MB); commit it on its own, with the reason`,
  )
  process.exit(0)
}

if (!flag('check')) {
  mkdirSync(cache, { recursive: true })
  writeFileSync(join(cache, 'digest.json'), text)
  if (only === undefined) writeGraphs(join(cache, 'graphs'), graphs, true)
  else writeGraphs(join(cache, 'graphs'), graphs, false)
  out(`ep-digest: wrote ${join(cache, 'digest.json')} and the graphs under ${join(cache, 'graphs')}`)
  if (existsSync(baselinePath) && only === undefined)
    out(
      `ep-digest: ${readFileSync(baselinePath, 'utf8') === text ? 'identical to the committed baseline' : 'differs from the committed baseline (run make ep-digest-check)'}`,
    )
  process.exit(0)
}

// ----- check -----
if (!existsSync(baselinePath)) {
  out(`ep-digest-check: no baseline at ${baselinePath}; run make ep-digest-check ARGS="--accept" on the unchanged code`)
  process.exit(2)
}
const base = JSON.parse(readFileSync(baselinePath, 'utf8')) as Digest
const { level, moves, compared } = compare(base, kept, tolerances, only, only === undefined || flag('coverage'))
const nowGraphs = join(cache, 'graphs-now')
writeGraphs(nowGraphs, graphs, only === undefined)

const worst = moves.filter((m) => m.level === 'exact' || m.level === 'number')
const noted = moves.filter((m) => m.level === 'rounded' || m.level === 'bits')
const want = flag('bits') ? 'bits' : flag('exact') ? 'exact' : 'tolerant'
const order = ['none', 'tolerant', 'exact', 'bits']
const ok = order.indexOf(level) >= order.indexOf(want)

const show = (list: typeof moves, limit: number): void => {
  const byItem = new Map<string, typeof moves>()
  for (const m of list) byItem.set(m.item, [...(byItem.get(m.item) ?? []), m])
  let shown = 0
  for (const [item, ms] of byItem) {
    if (shown >= limit) return void out(`  ... ${byItem.size - shown} more items (use --only to narrow)`)
    shown += 1
    out(`  ${item}`)
    for (const m of ms.slice(0, 8))
      out(`    ${m.path}: ${m.was} -> ${m.now}${m.by === undefined ? '' : `  (by ${m.by.toExponential(2)})`}`)
    if (ms.length > 8) out(`    ... ${ms.length - 8} more fields`)
    for (const m of ms) {
      if (!/graphs\.\w+\.hash$|\.graph\.hash$/.test(m.path)) continue
      const variant = m.path.endsWith('.graph.hash') ? 'vmp' : (m.path.split('.').slice(-2, -1)[0] as string)
      const oldFile = join(cache, 'graphs', fileOf(item))
      const newFile = join(nowGraphs, fileOf(item))
      if (!existsSync(oldFile) || !existsSync(newFile)) {
        out(
          `    (graph ${variant}: no cached baseline graph; run make ep-digest on the baseline commit to cache it, then check again)`,
        )
        continue
      }
      const a = (JSON.parse(readFileSync(oldFile, 'utf8')) as Record<string, J>)[variant] ?? null
      const b = (JSON.parse(readFileSync(newFile, 'utf8')) as Record<string, J>)[variant] ?? null
      out(`    graph ${variant} changed:`)
      for (const line of diffPaths(a, b, 12)) out(`      ${line}`)
    }
  }
}

if (worst.length > 0) {
  out(`ep-digest-check: ${worst.length} moves in ${new Set(worst.map((m) => m.item)).size} of ${compared} items`)
  show(worst, flag('all') ? Number.POSITIVE_INFINITY : 25)
}
if (noted.length > 0 && (worst.length === 0 || flag('verbose'))) {
  out(
    `ep-digest-check: ${noted.length} numbers differ within the tolerance or in the last digits only, in ${new Set(noted.map((m) => m.item)).size} items`,
  )
  show(noted, flag('verbose') ? 25 : 5)
}
out(
  `ep-digest-check: compared ${compared} items; level reached: ${level.toUpperCase()}${level === 'bits' ? ' (every number identical at full precision)' : level === 'exact' ? ' (graphs, verdicts, counts and numbers to 12 digits identical)' : level === 'tolerant' ? ' (numbers within the tolerance, the rest identical)' : ''}`,
)
out(
  `ep-digest-check: ${ok ? 'PASS' : 'FAIL'} (wanted at least ${want.toUpperCase()}; tolerances: beliefs ${tolerances.belief} relative, evidence ${tolerances.evidence}, check statistics ${tolerances.check})`,
)
void readdirSync
process.exit(ok ? 0 : 1)
