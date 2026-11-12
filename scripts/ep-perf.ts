/**
 * The performance harness of the EP code: where the time goes, as numbers that can be compared later. It measures, deterministically
 * seeded, warmed up and as the median of several runs with the spread reported:
 *
 *   grid       the shared grid integrator (`message-passing/numerics/_grid.ts`: `integratePair`, `integrateAxes`, `integrateTriple`,
 *              `integrateLine`) on a typical integrand per profile, microseconds per call and nanoseconds per integrand call, and the
 *              kernels under it (`tiltGrid`, `findWindow`, `shiftedSum`);
 *   operators  one message call (and the evidence call) per operator and per typical cavity set: the contexts are recorded from real EP runs
 *              of the atoms, models and programs (the typical cavities of a graph that runs, with the shapes, ports and parameters it uses),
 *              then replayed under a timer, grouped by folder and sorted;
 *   items      EP wall time, sweeps, updates and updates per second of a fixed set of atoms, models and programs (`ITEMS`), and the share of
 *              the time per operator across each run, from timing the registry's operators from here (the library is never edited);
 *   calibration  a fixed pure-arithmetic workload timed before and after everything, so a baseline taken on another or a busier machine is
 *              compared by ratio.
 *
 *   make ep-perf                       measure everything, write docs/probprog/ep-perf.json (the baseline) and ep-perf.md
 *   make ep-perf-check                 re-measure and fail when anything is slower than the baseline by more than a ratio (default 1.5,
 *                                      after dividing by the machine factor from the calibration)
 *   make ep-perf-check ARGS="--only item/atom/gaussian,grid/"   limit to keys (prefixes); `--ratio 2` changes the ratio
 *   make ep-perf ARGS="--survey"       time every atom, model and program's EP run once (to choose ITEMS) and print them, writes nothing
 *
 * A heavy, periodic tool, not part of `make verify` or `make check`. Wall-clock time is read with `performance.now()` here, which is
 * fine in a script outside `packages/` (`scripts/banned.ts` covers published source only).
 */
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import { join } from 'node:path'
import type { FactorOperator, OperatorContext } from '../packages/engine/src/lib/message-passing/index.ts'
import { defaultOperators } from '../packages/engine/src/lib/message-passing/index.ts'
import { shiftedSum } from '../packages/engine/src/lib/message-passing/numerics/_moments.ts'
import { findWindow, tiltGrid } from '../packages/engine/src/lib/message-passing/numerics/_windows.ts'
import { covarianceOperatorList } from '../packages/engine/src/lib/message-passing/operators/covariance/list.ts'
import { discreteOperatorList } from '../packages/engine/src/lib/message-passing/operators/discrete/list.ts'
import { gaussianOperatorList } from '../packages/engine/src/lib/message-passing/operators/gaussian/list.ts'
import { liftOperatorList } from '../packages/engine/src/lib/message-passing/operators/lifts/list.ts'
import { likelihoodOperatorList } from '../packages/engine/src/lib/message-passing/operators/likelihoods/list.ts'
import { linkOperatorList } from '../packages/engine/src/lib/message-passing/operators/links/list.ts'
import { scaleOperatorList } from '../packages/engine/src/lib/message-passing/operators/scales/list.ts'
import { atomBound, type EpAtom, epAtoms } from '../packages/engine/src/lib/probprog/ep/index.ts'
import { epGridMethod, epSpecMethod } from '../packages/engine/src/lib/probprog/inference/index.ts'
import type { BoundModel } from '../packages/engine/src/lib/probprog/model/index.ts'
import { benchmarkModels } from '../packages/engine/src/lib/probprog/models/index.ts'
import type { Posterior } from '../packages/engine/src/lib/probprog/posterior.ts'
import { linspace, sin, toFlat } from '../packages/engine/src/lib/tensor/index.ts'
import type { Config } from '../packages/engine/src/lib/trace/index.ts'
import { session } from '../packages/engine/src/lib/trace/index.ts'
import { routeOf } from './ep-digest-core.ts'
import { programCases } from './ep-digest-programs.ts'
import {
  calibrationOf,
  chosenKey,
  compareTimes,
  formatMicros,
  machineFactor,
  median,
  percent,
  type Stat,
  summarise,
  table,
} from './ep-perf-stats.ts'
import { gridCases } from './perf/grid-cases.ts'
import { syntheticCalls } from './perf/synthetic-calls.ts'
import { bench, calibrate, now } from './perf/timing.ts'

const ROOT = join(import.meta.dirname, '..')


// ----- the items of the end-to-end measurement -----

/**
 * The end-to-end set: about 25 atoms, models and programs, chosen to cover each operator family (the Gaussian and linear operators,
 * the links, the discrete and the gate, the plates and rows, the likelihood sites over one, two and three scalars, the scale
 * operators, the covariance and Wishart families) and the heaviest runs `--survey` found. EP runs only (the digest's references,
 * graph compilations and variational runs are not timed).
 */
const ITEMS: readonly string[] = [
  // The heaviest runs of the survey (seconds on the machine of the baseline: 17, 8, 6, 6, 5, 4.5, 4, 3.5, 3, 2, 2, 1.7)
  'program/circular-rows',
  'atom/softmax-two-groups',
  'atom/gp-hyperparameters',
  'atom/chain-persistence-grid',
  'program/changepoint-extended',
  'atom/student-t-full-learned',
  'atom/gate-poisson-negative-binomial',
  'program/zero-inflated',
  'program/sparse-mixture',
  'model/radon',
  'program/kalman',
  'program/cox',
  // One or more per operator family
  'atom/clutter-mixture-learned',
  'atom/poisson-log-dispersion-learned',
  'atom/ordinal-probit-cuts-learned',
  'atom/softmax-regression',
  'atom/probit-learned',
  'atom/weibull-shape-learned',
  'atom/mvn-wishart-learned',
  'atom/bivariate-correlation-learned',
  'atom/random-intercepts-slopes-learned',
  'atom/stick-breaking',
  'atom/gaussian-learned',
  'model/eight-schools-centred',
  'model/stochastic-volatility',
  'model/negative-binomial-regression',
]

// ----- command line -----

const args = process.argv.slice(2)
const flag = (name: string): boolean => args.includes(`--${name}`)
const option = (name: string): string | undefined => {
  const i = args.indexOf(`--${name}`)
  return i >= 0 ? args[i + 1] : undefined
}
const only = option('only')?.split(',')
// `--out <dir>` writes the baseline and the report there instead of docs/probprog (a trial run of a part, or a check against another baseline).
const OUT = option('out') ?? join(ROOT, 'docs/probprog')
const BASELINE = join(OUT, 'ep-perf.json')
const REPORT = join(OUT, 'ep-perf.md')
const ratio = Number(option('ratio') ?? 1.5)
const out = (s: string): void => void process.stdout.write(`${s}\n`)

// ----- the grid integrator -----

interface GridRow {
  readonly key: string
  readonly profile: string
  readonly describe: string
  /** Nodes of the final grid (0 when the entry point does not return them). */
  readonly nodes: number
  /** How many times the integrand was called in one integral, over every pass. */
  readonly evals: number
  /** Microseconds per integral. */
  readonly us: Stat
  /** Nanoseconds per integrand call. */
  readonly nsPerEval: number
}

function measureGrid(): GridRow[] {
  const rows: GridRow[] = []
  for (const c of gridCases()) {
    const key = `grid/${c.name}`
    if (!chosenKey(only, key)) continue
    const first = c.run()
    const us = summarise(bench(c.run, 9, 20))
    rows.push({
      key,
      profile: c.name,
      describe: c.describe,
      nodes: first.nodes,
      evals: first.evals,
      us,
      nsPerEval: (us.median * 1000) / first.evals,
    })
  }
  return rows
}

interface KernelRow {
  readonly key: string
  readonly describe: string
  readonly size: number
  readonly us: Stat
  readonly nsPerNode: number
}

function measureKernels(): KernelRow[] {
  const rows: KernelRow[] = []
  const add = (name: string, describe: string, size: number, fn: () => unknown): void => {
    const key = `kernel/${name}`
    if (!chosenKey(only, key)) return
    const us = summarise(bench(fn, 9, 20))
    rows.push({ key, describe, size, us, nsPerNode: (us.median * 1000) / size })
  }
  const logw = toFlat(sin(linspace(0, 50, 4320)))
  add('shiftedSum', 'shift, exponentiate and sum 4320 log weights (the size of a pair grid)', logw.length, () =>
    shiftedSum(logw),
  )
  // A narrow peak far from the start: the window finder scans, widens and narrows (801 nodes a scan).
  const peak = (x: Float64Array): Float64Array => Float64Array.from(x, (v) => -0.5 * ((v - 3) / 0.05) ** 2)
  add('findWindow', 'window of a narrow peak at 3 from a start at 0 (801-node scans)', 801, () =>
    findWindow(peak, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, 0),
  )
  add('tiltGrid', 'zoomed uniform grid of 64 nodes over a window of width 12, smooth integrand', 64, () =>
    tiltGrid({ lo: -6, hi: 6 }, 64, (t) => -0.5 * (t - 1) ** 2 - Math.log1p(Math.exp(-t))),
  )
  return rows
}

// ----- the operators: instrumentation, capture, replay -----

type Hook = 'ep' | 'tilted' | 'logEvidence'
const HOOKS: readonly Hook[] = ['ep', 'tilted', 'logEvidence']

/** The folder each operator lives in. */
const FOLDERS: Map<string, string> = new Map(
  (
    [
      ['gaussian', gaussianOperatorList],
      ['links', linkOperatorList],
      ['discrete', discreteOperatorList],
      ['lifts', liftOperatorList],
      ['likelihoods', likelihoodOperatorList],
      ['scales', scaleOperatorList],
      ['covariance', covarianceOperatorList],
    ] as const
  ).flatMap(([folder, list]) => list.map((op): [string, string] => [op.name, folder])),
)
FOLDERS.set('gate', 'discrete')

/** What recorded calls of one operator, hook and signature add up to, and one context of them to replay. */
interface Captured {
  readonly op: string
  readonly hook: Hook
  readonly sig: string
  calls: number
  ms: number
  ctx: OperatorContext | undefined
  readonly from: string
}

/** The short names of the families in a call signature. */
const KIND_CODE: Readonly<Record<string, string>> = {
  gaussian: 'N',
  'gaussian-mv': 'Nv',
  gamma: 'G',
  beta: 'B',
  bernoulli: 'Be',
  discrete: 'D',
  dirichlet: 'Di',
  'normal-gamma': 'NG',
  wishart: 'W',
  'normal-wishart': 'NW',
}

/** What a signature says of a call: the port, the argument kinds (and sizes, `v` for an observed one) and the shape of the parameters. */
function signatureOf(c: OperatorContext): string {
  const codes = c.args.map((a) =>
    a.value !== undefined ? 'v' : `${KIND_CODE[a.family.kind] ?? a.family.kind}${a.family.dim > 1 ? a.family.dim : ''}`,
  )
  // Runs of one code are written `code*n` (a chain has fifty Gaussian arguments).
  const runs: string[] = []
  for (let k = 0; k < codes.length; ) {
    let n = 1
    while (codes[k + n] === codes[k]) n++
    runs.push(n > 3 ? `${codes[k]}*${n}` : (codes[k] as string).repeat(n))
    k += n
  }
  const argText = runs.join(' ')
  const shape = (v: unknown): string =>
    Array.isArray(v)
      ? `[${v.length}]`
      : typeof v === 'string'
        ? v.slice(0, 12)
        : typeof v === 'object' && v !== null
          ? `{${Object.keys(v).sort().join(',')}}`
          : typeof v === 'number'
            ? ''
            : String(v)
  const params = Object.keys(c.params)
    .sort()
    .map((k) => `${k}${shape(c.params[k])}`)
    .join(',')
  return `to${c.to} ${argText}${params === '' ? '' : ` {${params}}`}`
}

/** What one operator's calls add up to in a run. */
interface OpTime {
  self: number
  incl: number
  calls: number
}

interface Instrument {
  /** Time in operators by operator name (self: without the operators called inside it; incl: with them). */
  readonly times: Map<string, OpTime>
  readonly captured: Map<string, Captured>
  readonly restore: () => void
}

/**
 * Wrap the message, tilted and evidence rules of the registry's operators with timers, from here only (the operator objects are plain
 * values the engine reads at each call; `restore` puts the originals back). Self time is the call's time less that of the operators
 * called from inside it (a plate calls its row's operator). The gate is made fresh by each `defaultOperators()`, so it is not wrapped:
 * its own overhead lands in the engine's remainder, its branches are timed as themselves.
 */
function instrument(from: string, capture: boolean): Instrument {
  const times = new Map<string, OpTime>()
  const captured = new Map<string, Captured>()
  const undo: (() => void)[] = []
  const frames: { child: number }[] = []
  for (const [name, op] of Object.entries(defaultOperators())) {
    if (name === 'gate') continue
    const rec = op as unknown as Record<string, unknown>
    for (const hook of HOOKS) {
      const original = rec[hook]
      if (typeof original !== 'function') continue
      const f = original as (c: OperatorContext) => unknown
      rec[hook] = (c: OperatorContext): unknown => {
        const frame = { child: 0 }
        frames.push(frame)
        const t0 = now()
        let ok = false
        try {
          const r = f.call(op, c)
          ok = true
          return r
        } finally {
          const dt = now() - t0
          frames.pop()
          const parent = frames[frames.length - 1]
          if (parent !== undefined) parent.child += dt
          const t = times.get(name) ?? { self: 0, incl: 0, calls: 0 }
          times.set(name, t)
          t.self += dt - frame.child
          t.incl += dt
          t.calls += 1
          if (capture && ok) {
            const sig = signatureOf(c)
            const key = `${name}|${hook}|${sig}`
            const cap = captured.get(key) ?? { op: name, hook, sig, calls: 0, ms: 0, ctx: undefined, from }
            captured.set(key, cap)
            cap.calls += 1
            cap.ms += dt
            // Keep a copy of the call at the first and at every power of two: later ones are the converged cavities.
            if ((cap.calls & (cap.calls - 1)) === 0)
              cap.ctx = {
                ...c,
                ...(c.memo === undefined ? {} : { memo: new Map() }),
                args: c.args.map((a) => ({ ...a, eta: a.eta === undefined ? undefined : [...a.eta] })),
              }
          }
        }
      }
      undo.push(() => {
        rec[hook] = original
      })
    }
  }
  return { times, captured, restore: () => undo.forEach((u) => void u()) }
}

interface OpRow {
  readonly key: string
  readonly op: string
  readonly folder: string
  readonly hook: Hook
  readonly sig: string
  /** The item the context was recorded from, and how often a run of it made this call. */
  readonly from: string
  readonly calls: number
  readonly us: Stat
}

/** The signatures replayed per operator and hook: the most called; evidence only the most called. */
const SIGNATURES = 4

function measureOperators(
  captured: ReadonlyMap<string, Captured>,
  wanted: ReadonlySet<string> | undefined,
  filter: readonly string[] | undefined,
): OpRow[] {
  const ops = defaultOperators()
  const byGroup = new Map<string, Captured[]>()
  for (const c of captured.values()) {
    if (c.ctx === undefined) continue
    const g = `${c.op}|${c.hook}`
    byGroup.set(g, [...(byGroup.get(g) ?? []), c])
  }
  const rows: OpRow[] = []
  for (const [group, list] of [...byGroup].sort((a, b) => a[0].localeCompare(b[0]))) {
    const limit = group.endsWith('|logEvidence') ? 1 : SIGNATURES
    const real = list.filter((c) => c.from !== 'synthetic')
    const chosen = [
      ...real.sort((a, b) => b.calls - a.calls || a.sig.localeCompare(b.sig)).slice(0, limit),
      ...list.filter((c) => c.from === 'synthetic'),
    ]
    for (const c of chosen) {
      const key = `op/${c.op}/${c.hook}/${c.sig}`
      if (
        wanted !== undefined ? !wanted.has(key) : filter !== undefined && filter.length > 0 && !chosenKey(filter, key)
      )
        continue
      const op = ops[c.op] as FactorOperator
      const hook = (op as unknown as Record<string, (x: OperatorContext) => unknown>)[c.hook] as (
        x: OperatorContext,
      ) => unknown
      const ctx = c.ctx as OperatorContext
      const call = (): unknown => {
        try {
          ctx.memo?.clear()
          return hook.call(op, ctx)
        } catch {
          return 0
        }
      }
      const us = summarise(bench(call, 7, 4))
      rows.push({
        key,
        op: c.op,
        folder: FOLDERS.get(c.op) ?? '?',
        hook: c.hook,
        sig: c.sig,
        from: c.from,
        calls: c.calls,
        us,
      })
    }
  }
  return rows
}

// ----- the items: end to end -----

interface Ran {
  readonly ms: number
  readonly sweeps: number | null
  readonly updates: number | null
  readonly status: string
}

type Runner = () => Promise<Ran>

/** Called by a runner right after its EP session ends, before the posterior is read (the evidence rules called there are not part of the run's time). */
let afterSession: (() => void) | undefined

interface ItemDef {
  readonly key: string
  readonly kind: 'atom' | 'model' | 'program'
  /** Builds what the run needs (the bound model, the method) outside the timer; undefined when EP does not run it. */
  readonly prepare: () => Promise<{ readonly route: string; readonly run: Runner } | undefined>
}

const sessionRun =
  (method: ReturnType<typeof epSpecMethod> | ReturnType<typeof epGridMethod>, steps: number, config: Config): Runner =>
  async () => {
    const t0 = now()
    const s = session(method as never as ReturnType<typeof epSpecMethod>, {
      seed: 1,
      steps,
      mode: 'eager',
      config,
      logs: 'off',
    })
    const ms = now() - t0
    afterSession?.()
    const post = method.posterior?.(s.state as never, s.config as never) as Posterior | undefined
    const d = (post?.diagnostics ?? {}) as Record<string, unknown>
    return {
      ms,
      sweeps: typeof d['sweeps'] === 'number' ? d['sweeps'] : s.t,
      updates: typeof d['updates'] === 'number' ? d['updates'] : null,
      status: String(s.status ?? ''),
    }
  }

function boundRun(bound: BoundModel, gridOver: readonly string[] | undefined, config: Config, maxSweeps: number) {
  const { route } = routeOf(bound, gridOver)
  if (route === 'none') return undefined
  const method =
    route === 'ep' ? epSpecMethod(bound) : epGridMethod(bound, gridOver === undefined ? {} : { over: gridOver })
  return { route, run: sessionRun(method, route === 'ep' ? maxSweeps + 2 : 1000, config) }
}

function allItems(): ItemDef[] {
  const items: ItemDef[] = []
  for (const a of epAtoms as readonly EpAtom[]) {
    items.push({
      key: `atom/${a.name}`,
      kind: 'atom',
      prepare: async () => {
        const problem = a.problem(a.defaults)
        const bound = atomBound(problem)
        const maxSweeps = typeof a.ep?.['maxSweeps'] === 'number' ? a.ep['maxSweeps'] : 200
        return boundRun(
          bound,
          problem.hyperGrid?.over,
          { tol: 1e-9, ...(problem.hyperGrid?.knobs ?? {}), ...(a.ep ?? {}) },
          maxSweeps,
        )
      },
    })
  }
  for (const m of benchmarkModels()) {
    if (m.definition === undefined) continue
    items.push({
      key: `model/${m.name}`,
      kind: 'model',
      prepare: async () => boundRun(m.bound(), undefined, { tol: 1e-9 }, 200),
    })
  }
  for (const c of programCases) {
    items.push({
      key: `program/${c.key}`,
      kind: 'program',
      prepare: async () => ({
        route: 'ep',
        run: async () => {
          const t0 = now()
          const inferred = await c.load()
          const ms = now() - t0
          afterSession?.()
          const post = epSpecMethod(inferred.bound).posterior?.(
            inferred.run.state as never,
            inferred.run.config as never,
          ) as Posterior | undefined
          const d = (post?.diagnostics ?? {}) as Record<string, unknown>
          return {
            ms,
            sweeps: typeof d['sweeps'] === 'number' ? d['sweeps'] : inferred.run.t,
            updates: typeof d['updates'] === 'number' ? d['updates'] : null,
            status: String(inferred.run.status ?? ''),
          }
        },
      }),
    })
  }
  return items
}

interface ItemRow {
  readonly key: string
  readonly kind: string
  readonly route: string
  readonly status: string
  readonly sweeps: number | null
  readonly updates: number | null
  /** Seconds of the EP run. */
  readonly seconds: Stat
  readonly updatesPerSecond: number | null
}

async function measureItems(keys: readonly string[], runs: number): Promise<ItemRow[]> {
  const defs = new Map(allItems().map((d) => [d.key, d]))
  const rows: ItemRow[] = []
  for (const key of keys) {
    const def = defs.get(key)
    if (def === undefined) {
      out(`  ${key}: no such item`)
      continue
    }
    const prepared = await def.prepare()
    if (prepared === undefined) {
      out(`  ${key}: EP does not run it`)
      continue
    }
    await prepared.run() // warm-up: the first run compiles and optimises the code it runs
    const ran: Ran[] = []
    for (let i = 0; i < runs; i++) ran.push(await prepared.run())
    const seconds = summarise(ran.map((r) => r.ms / 1000))
    const last = ran[ran.length - 1] as Ran
    rows.push({
      key,
      kind: def.kind,
      route: prepared.route,
      status: last.status,
      sweeps: last.sweeps,
      updates: last.updates,
      seconds,
      updatesPerSecond: last.updates === null ? null : last.updates / seconds.median,
    })
    out(
      `  ${key.padEnd(44)} ${seconds.median.toFixed(3).padStart(8)} s  (spread ${percent(seconds.spread)})  sweeps ${last.sweeps}  updates ${last.updates}`,
    )
  }
  return rows
}

interface ShareRow {
  readonly item: string
  /** Seconds of the instrumented run. */
  readonly wall: number
  /** Self time by operator, in seconds, and calls. */
  readonly byOperator: readonly {
    readonly op: string
    readonly self: number
    readonly incl: number
    readonly calls: number
  }[]
}

/** One instrumented run of an item: where the time goes across the operators. */
async function measureShare(
  key: string,
): Promise<{ readonly share: ShareRow; readonly instrument: Instrument } | undefined> {
  const def = allItems().find((d) => d.key === key)
  const prepared = await def?.prepare()
  if (def === undefined || prepared === undefined) return undefined
  const inst = instrument(key, false)
  let times = new Map<string, OpTime>()
  afterSession = () => {
    times = new Map([...inst.times].map(([k, t]) => [k, { ...t }]))
  }
  try {
    const r = await prepared.run()
    const byOperator = [...times]
      .map(([op, t]) => ({ op, self: t.self / 1000, incl: t.incl / 1000, calls: t.calls }))
      .sort((a, b) => b.self - a.self)
    return { share: { item: key, wall: r.ms / 1000, byOperator }, instrument: inst }
  } finally {
    afterSession = undefined
    inst.restore()
  }
}

/** One instrumented run of each item that records the typical calls of every operator it reaches. */
async function capturePass(keys: readonly string[]): Promise<Map<string, Captured>> {
  const defs = new Map(allItems().map((d) => [d.key, d]))
  const all = new Map<string, Captured>()
  for (const key of keys) {
    const prepared = await defs.get(key)?.prepare()
    if (prepared === undefined) continue
    const inst = instrument(key, true)
    try {
      await prepared.run()
    } catch (e) {
      out(`  capture ${key}: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      inst.restore()
    }
    for (const [k, c] of inst.captured) {
      const have = all.get(k)
      if (have === undefined) all.set(k, c)
      else {
        have.calls += c.calls
        have.ms += c.ms
        if (c.calls >= have.calls - c.calls) have.ctx = c.ctx ?? have.ctx
      }
    }
  }
  return all
}

// ----- the report -----

interface Report {
  readonly format: 1
  readonly date: string
  readonly commit: string
  readonly machine: {
    readonly cpu: string
    readonly cores: number
    readonly memoryGb: number
    readonly platform: string
    readonly node: string
  }
  /** The one-minute load average before and after the measurements. */
  readonly load: { readonly before: number; readonly after: number }
  /** The calibration workload in milliseconds: before, after, their mean and the drift (after / before). */
  readonly calibration: {
    readonly startMs: number
    readonly endMs: number
    readonly ms: number
    readonly drift: number
  }
  readonly settings: { readonly itemRuns: number; readonly microSamples: number; readonly items: readonly string[] }
  readonly grid: readonly GridRow[]
  readonly kernels: readonly KernelRow[]
  readonly operators: readonly OpRow[]
  /** Operators of the registry no run reached (no replay). */
  readonly unreached: readonly string[]
  readonly items: readonly ItemRow[]
  readonly share: readonly ShareRow[]
}

function machine(): Report['machine'] {
  const cpus = os.cpus()
  return {
    cpu: cpus[0]?.model ?? 'unknown',
    cores: cpus.length,
    memoryGb: Math.round(os.totalmem() / 2 ** 30),
    platform: `${os.platform()} ${os.arch()} ${os.release()}`,
    node: process.version,
  }
}

function commit(): string {
  try {
    return execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: ROOT }).toString().trim()
  } catch {
    return 'unknown'
  }
}

const load1 = (): number => Number((os.loadavg()[0] as number).toFixed(2))

function markdown(r: Report): string {
  const lines: string[] = []
  const p = (s = ''): void => void lines.push(s)
  p('# EP performance baseline')
  p()
  p('<!-- Generated by `make ep-perf` from `docs/probprog/ep-perf.json` (scripts/ep-perf.ts). Do not edit by hand. -->')
  p()
  p(
    `Measured ${r.date.slice(0, 10)} at commit \`${r.commit}\` on ${r.machine.cpu} (${r.machine.cores} cores, ${r.machine.memoryGb} GB, ${r.machine.platform}), node ${r.machine.node}. ` +
      `Medians of ${r.settings.microSamples} timings for the micro rows and ${r.settings.itemRuns} runs for the items, after a warm-up, spread = (max - min) / median.`,
  )
  p()
  p(
    `**Calibration** (a fixed pure-arithmetic workload, ms): ${r.calibration.startMs.toFixed(1)} at the start, ${r.calibration.endMs.toFixed(1)} at the end, mean ${r.calibration.ms.toFixed(1)} (drift ${r.calibration.drift.toFixed(2)}x). ` +
      `**Load** (1-minute average): ${r.load.before} before, ${r.load.after} after. ` +
      'A baseline taken under load is compared by the ratio of calibration figures (`make ep-perf-check` does so); absolute numbers here carry that load.',
  )
  p()
  p('## The shared grid integrator')
  p()
  p(
    'One integral through the library entry point on a typical integrand per profile (`scripts/perf/grid-cases.ts`). "Integrand calls" counts every pass (the coarse passes that find the window and the fine one); the nodes are those of the final grid.',
  )
  p()
  p(
    table(
      [
        'profile',
        'integrand',
        'final nodes',
        'integrand calls',
        'time per integral',
        'spread',
        'ns per integrand call',
      ],
      r.grid.map((g) => [
        g.profile,
        g.describe,
        g.nodes === 0 ? 'n/a' : String(g.nodes),
        String(g.evals),
        formatMicros(g.us.median),
        percent(g.us.spread),
        g.nsPerEval.toFixed(0),
      ]),
    ),
  )
  p()
  p('The kernels under the integrators:')
  p()
  p(
    table(
      ['kernel', 'what', 'size', 'time per call', 'spread', 'ns per node'],
      r.kernels.map((k) => [
        k.key.replace('kernel/', ''),
        k.describe,
        String(k.size),
        formatMicros(k.us.median),
        percent(k.us.spread),
        k.nsPerNode.toFixed(1),
      ]),
    ),
  )
  p()
  p('## Operators: time per message call')
  p()
  p(
    'The typical calls of each operator, recorded from EP runs of the atoms, models and programs and replayed under a timer (a fresh memo per call, as the engine gives). One row per operator, hook and call signature (`toN` the target port, then the argument families: `N` Gaussian, `Nv` vector Gaussian, `G` gamma, `B` beta, `Be` Bernoulli, `D` discrete, `Di` Dirichlet, `NG` normal-gamma, `W` Wishart, `NW` normal-Wishart, each with its dimension when above one, and `v` an observed value; then the shape of the parameters). `ep` is the message rule (or `tilted` for an operator that only has the tilted moments), `logEvidence` the evidence rule.',
  )
  p()
  const epRows = r.operators.filter((o) => o.hook !== 'logEvidence').sort((a, b) => b.us.median - a.us.median)
  p('### The 30 slowest message calls')
  p()
  p(
    table(
      ['operator', 'folder', 'hook', 'call', 'time per call', 'spread', 'calls in its run', 'recorded from'],
      epRows
        .slice(0, 30)
        .map((o) => [
          o.op,
          o.folder,
          o.hook,
          o.sig,
          formatMicros(o.us.median),
          percent(o.us.spread),
          String(o.calls),
          o.from,
        ]),
    ),
  )
  p()
  const folders = [...new Set(r.operators.map((o) => o.folder))].sort()
  for (const folder of folders) {
    p(`### ${folder}`)
    p()
    p(
      table(
        ['operator', 'hook', 'call', 'time per call', 'spread'],
        r.operators
          .filter((o) => o.folder === folder)
          .sort((a, b) => b.us.median - a.us.median)
          .map((o) => [o.op, o.hook, o.sig, formatMicros(o.us.median), percent(o.us.spread)]),
      ),
    )
    p()
  }
  if (r.unreached.length > 0) {
    p(`Operators no measured run reached (not replayed): ${r.unreached.join(', ')}.`)
    p()
  }
  p('## End to end: the EP runs')
  p()
  p(
    'The EP session alone (the digest also compiles three graphs, runs a reference and a variational run per atom; those are not timed here). Updates per second is updates over the median wall time.',
  )
  p()
  p(
    table(
      ['item', 'route', 'wall time', 'spread', 'sweeps', 'updates', 'updates per second'],
      [...r.items]
        .sort((a, b) => b.seconds.median - a.seconds.median)
        .map((i) => [
          i.key,
          i.route,
          formatMicros(i.seconds.median * 1e6),
          percent(i.seconds.spread),
          String(i.sweeps),
          String(i.updates),
          i.updatesPerSecond === null ? 'n/a' : i.updatesPerSecond.toFixed(0),
        ]),
    ),
  )
  p()
  p('## Where the time goes: the share per operator')
  p()
  p(
    'One instrumented run of each item (the registry\'s message, tilted and evidence rules timed from the harness: about 0.1 microseconds a call of overhead). Self time excludes the operators an operator calls (a plate calls its row); the gate is made fresh by each registry call and is not wrapped, so its own overhead is part of "engine and the rest" and its branches are timed as themselves.',
  )
  p()
  const total = new Map<string, { self: number; calls: number }>()
  let wall = 0
  let inOps = 0
  for (const s of r.share) {
    wall += s.wall
    for (const o of s.byOperator) {
      inOps += o.self
      const t = total.get(o.op) ?? { self: 0, calls: 0 }
      total.set(o.op, { self: t.self + o.self, calls: t.calls + o.calls })
    }
  }
  p(
    `Across the ${r.share.length} items (${wall.toFixed(2)} s of wall time): ${percent(inOps / wall)} in operators, ${percent(1 - inOps / wall)} in the engine and the rest (cavities, projections, scheduling, graph and tape bookkeeping).`,
  )
  p()
  p(
    table(
      ['operator', 'folder', 'self time', 'share of wall', 'share of operator time', 'calls', 'us per call'],
      [...total]
        .sort((a, b) => b[1].self - a[1].self)
        .slice(0, 30)
        .map(([op, t]) => [
          op,
          FOLDERS.get(op) ?? '?',
          `${t.self.toFixed(3)} s`,
          percent(t.self / wall),
          percent(t.self / inOps),
          String(t.calls),
          ((t.self * 1e6) / t.calls).toFixed(1),
        ]),
    ),
  )
  p()
  p('Per item, the three operators with the most self time:')
  p()
  p(
    table(
      ['item', 'wall (instrumented)', 'in operators', 'top operators (share of the item)'],
      r.share.map((s) => {
        const ops = s.byOperator.reduce((a, o) => a + o.self, 0)
        return [
          s.item,
          `${s.wall.toFixed(3)} s`,
          percent(ops / s.wall),
          s.byOperator
            .slice(0, 3)
            .map((o) => `${o.op} ${percent(o.self / s.wall)}`)
            .join(', '),
        ]
      }),
    ),
  )
  p()
  return `${lines.join('\n')}\n`
}

// ----- main -----

async function main(): Promise<void> {
  const check = flag('check')
  const defs = allItems()
  if (flag('survey')) {
    out(`ep-perf survey: the EP run of every item once (${defs.length} items), load ${load1()}`)
    const rows: { key: string; s: number; sweeps: number | null; updates: number | null }[] = []
    for (const d of defs) {
      if (!chosenKey(only, d.key)) continue
      try {
        const prepared = await d.prepare()
        if (prepared === undefined) continue
        const r = await prepared.run()
        rows.push({ key: d.key, s: r.ms / 1000, sweeps: r.sweeps, updates: r.updates })
        out(`  ${d.key.padEnd(46)} ${(r.ms / 1000).toFixed(3).padStart(8)} s  sweeps ${r.sweeps}  updates ${r.updates}`)
      } catch (e) {
        out(`  ${d.key}: ${e instanceof Error ? e.message : String(e)}`)
      }
    }
    out('slowest first:')
    for (const r of rows.sort((a, b) => b.s - a.s))
      out(`${r.s.toFixed(3).padStart(9)} s  ${r.key}  (sweeps ${r.sweeps}, updates ${r.updates})`)
    return
  }

  const baseline: Report | undefined =
    check && existsSync(BASELINE) ? (JSON.parse(readFileSync(BASELINE, 'utf8')) as Report) : undefined
  if (check && baseline === undefined) {
    out(`ep-perf-check: no baseline at ${BASELINE}; run make ep-perf first`)
    process.exit(2)
  }
  const itemKeys = (baseline?.settings.items ?? ITEMS).filter((k) => chosenKey(only, `item/${k}`))
  const itemRuns = 3
  const startLoad = load1()
  out(`ep-perf: load ${startLoad}; calibrating`)
  const calStart = calibrate()
  out(`  calibration ${median(calStart).toFixed(1)} ms`)

  const grid = measureGrid()
  const kernels = measureKernels()
  for (const g of grid)
    out(
      `  ${g.key.padEnd(14)} ${formatMicros(g.us.median).padStart(10)}  ${g.nsPerEval.toFixed(0)} ns per integrand call`,
    )

  // Operators: recorded from every item in a full run; in a check only the signatures the baseline measured, from the items they came from.
  const opOnly = only?.filter((o) => o.startsWith('op'))
  const wantedOps =
    baseline === undefined ? undefined : new Set(baseline.operators.map((o) => o.key).filter((k) => chosenKey(only, k)))
  const sources: string[] = flag('no-capture')
    ? []
    : baseline !== undefined
      ? [...new Set(baseline.operators.filter((o) => wantedOps?.has(o.key)).map((o) => o.from))]
      : only === undefined || opOnly?.length
        ? defs.map((d) => d.key)
        : defs.filter((d) => chosenKey(only, d.key)).map((d) => d.key)
  let operators: OpRow[] = []
  let unreached: string[] = []
  if (
    sources.length > 0 ||
    flag('no-capture') ||
    baseline?.operators.some((o) => o.from === 'synthetic' && wantedOps?.has(o.key))
  ) {
    out(`ep-perf: recording typical operator calls from ${sources.length} runs`)
    const captured = await capturePass(sources.filter((k) => k !== 'synthetic'))
    for (const call of syntheticCalls()) {
      const sig = signatureOf(call.ctx)
      captured.set(`${call.op}|${call.hook}|${sig}`, {
        op: call.op,
        hook: call.hook,
        sig,
        calls: 0,
        ms: 0,
        ctx: call.ctx,
        from: 'synthetic',
      })
    }
    operators = measureOperators(captured, wantedOps, opOnly)
    const reached = new Set(operators.map((o) => o.op))
    unreached = Object.keys(defaultOperators()).filter((n) => !reached.has(n))
    out(
      `  ${operators.length} operator calls timed; ${unreached.length} operators not reached: ${unreached.join(', ')}`,
    )
    if (flag('verbose'))
      for (const o of operators)
        out(
          `    ${o.key.padEnd(70)} ${formatMicros(o.us.median).padStart(10)} (spread ${percent(o.us.spread)}, ${o.calls} calls)`,
        )
  }

  out(`ep-perf: end to end, ${itemKeys.length} items, ${itemRuns} runs each`)
  const items = await measureItems(itemKeys, itemRuns)

  const share: ShareRow[] = []
  if (!check) {
    out('ep-perf: time share per operator (one instrumented run each)')
    for (const key of itemKeys) {
      const r = await measureShare(key)
      if (r !== undefined) share.push(r.share)
    }
  }

  const calEnd = calibrate()
  const cal = calibrationOf(calStart, calEnd)
  const endLoad = load1()
  out(
    `ep-perf: calibration ${median(calStart).toFixed(1)} ms before, ${median(calEnd).toFixed(1)} ms after; load ${startLoad} -> ${endLoad}`,
  )

  const report: Report = {
    format: 1,
    date: new Date().toISOString(),
    commit: commit(),
    machine: machine(),
    load: { before: startLoad, after: endLoad },
    calibration: { startMs: median(calStart), endMs: median(calEnd), ms: cal.ms, drift: cal.drift },
    settings: { itemRuns, microSamples: 9, items: itemKeys },
    grid,
    kernels,
    operators,
    unreached,
    items,
    share,
  }

  if (!check) {
    if (only !== undefined && option('out') === undefined) {
      out('ep-perf: --only measures a part and writes nothing (the baseline is the whole set; --out <dir> writes a trial)')
      return
    }
    writeFileSync(BASELINE, `${JSON.stringify(report, null, 1)}\n`)
    writeFileSync(REPORT, markdown(report))
    out(`ep-perf: wrote ${BASELINE} and ${REPORT}`)
    return
  }

  const base = baseline as Report
  const factor = machineFactor(cal.ms, base.calibration.ms)
  const baseTimes = new Map<string, number>([
    ...base.grid.map((g): [string, number] => [g.key, g.us.median]),
    ...base.kernels.map((k): [string, number] => [k.key, k.us.median]),
    ...base.operators.map((o): [string, number] => [o.key, o.us.median]),
    ...base.items.map((i): [string, number] => [`item/${i.key}`, i.seconds.median]),
  ])
  const nowTimes = new Map<string, number>([
    ...grid.map((g): [string, number] => [g.key, g.us.median]),
    ...kernels.map((k): [string, number] => [k.key, k.us.median]),
    ...operators.map((o): [string, number] => [o.key, o.us.median]),
    ...items.map((i): [string, number] => [`item/${i.key}`, i.seconds.median]),
  ])
  const compared = compareTimes(baseTimes, nowTimes, factor, ratio)
  out(
    `ep-perf-check: machine factor ${factor.toFixed(2)} (calibration ${cal.ms.toFixed(1)} ms now, ${base.calibration.ms.toFixed(1)} ms in the baseline; load ${endLoad} now, ${base.load.before} then)`,
  )
  for (const c of compared.slice(0, 15))
    out(
      `  ${c.slower ? 'SLOWER' : 'ok    '} ${c.key.padEnd(60)} ratio ${c.ratio.toFixed(2)}  (was ${c.base.toPrecision(4)}, now ${c.now.toPrecision(4)})`,
    )
  const slower = compared.filter((c) => c.slower)
  out(
    `ep-perf-check: ${compared.length} compared, ${slower.length} slower than ${ratio}x: ${slower.length === 0 ? 'PASS' : 'FAIL'}`,
  )
  process.exit(slower.length === 0 ? 0 : 1)
}

await main()
