/**
 * The reader of variational tapes for the `elboRun` record: `elboRunFromTape` turns the tape of a stochastic variational
 * run (`svi/elbo`, `svi/smoothed`, `svi/gradNorm`, `svi/gradVariance`, `svi/diag/*`, `svi/params/*`, `svi/snapshot`) or of
 * Pathfinder (`pathfinder/elbo`, `pathfinder/iterate`, `pathfinder/path/*`, `pathfinder/paretoK`, `pathfinder/ess`), finished
 * or still running, and any anytime snapshots, into the record. It reads only those documented paths, under the prefix the
 * tape has, and thins every series to a number of points a chart can draw.
 */

import {
  type ElboRun,
  elboRun,
  type GuideParameter,
  type NamedSeries,
  type QuantityBelief,
  trailingMean,
} from './elbo-run.ts'
import type { SnapshotAt, TapeEventLike, TapeLike } from './model-run-tape.ts'
import { type MarginalLike, posteriorComparison } from './posterior-comparison.ts'

/** Options of {@link elboRunFromTape}. */
export interface ElboRunOptions {
  /** The tape's path prefix, `svi/` or `pathfinder/` (default: found from the tape's first variational path). */
  readonly prefix?: string
  /** The method's name for the record (default the prefix without its slash). */
  readonly method?: string
  /** The window of the trailing mean of the ELBO; `0` for none (default the tape's own smoothed series, else a twentieth of the steps). */
  readonly smoothing?: number
  /** The most points kept of each series, thinned evenly with the last always kept (default 400). */
  readonly maxPoints?: number
  /** The most guide parameter elements kept (default 12), and diagnostics (default 6). */
  readonly maxParameters?: number
  readonly maxDiagnostics?: number
  /** The marginals of the truth by quantity (`distributionOracle`, `weightedOracle`, a grid's): the final posterior is laid over them. */
  readonly truth?: Readonly<Record<string, MarginalLike>>
  /** What the truth is called in the final overlay (default `truth`). */
  readonly truthName?: string
}

const num = (x: unknown): number | undefined => (typeof x === 'number' && Number.isFinite(x) ? x : undefined)
const isObject = (x: unknown): x is Readonly<Record<string, unknown>> =>
  typeof x === 'object' && x !== null && !Array.isArray(x)

interface Acc {
  steps: number[]
  values: number[]
}

/** Append a value at a step: a second value at the same step replaces the first. */
function put(acc: Acc, step: number, value: number): void {
  if (acc.steps.at(-1) === step) acc.values[acc.values.length - 1] = value
  else {
    acc.steps.push(step)
    acc.values.push(value)
  }
}

/** The indices of at most `cap` evenly spaced elements of `n`, the last always among them. */
function thinIndices(n: number, cap: number): number[] {
  if (n <= cap) return Array.from({ length: n }, (_, i) => i)
  return Array.from({ length: cap }, (_, i) => Math.round((i * (n - 1)) / (cap - 1)))
}

function thinned(acc: Acc, cap: number): { steps: number[]; values: number[] } {
  const keep = thinIndices(acc.steps.length, cap)
  return { steps: keep.map((i) => acc.steps[i] as number), values: keep.map((i) => acc.values[i] as number) }
}

const SUFFIX = /(?:_?auto_)(loc|scale)$/

/** The role and site of a guide parameter by its name: `mu_auto_loc` is the location of `mu`; `auto_scale` of a joint guide is a scale with no site. */
function roleOf(name: string): { role: GuideParameter['role']; site?: string } {
  const m = SUFFIX.exec(name)
  if (m === null) return { role: 'other' }
  const site = name.slice(0, m.index)
  return { role: m[1] as 'loc' | 'scale', ...(site === '' ? {} : { site }) }
}

/**
 * The record of a variational run, read from a tape. The ELBO comes from the metric `elbo` (Pathfinder's best so far),
 * smoothed by the tape's `smoothed` or a trailing mean; the gradient from `gradNorm` (or `naturalGradNorm`) and
 * `gradVariance`; the estimator's reports from `diag/*` and the other scalars of the method; the guide's parameters from
 * the tensors `params/*`, one trajectory per element on the unconstrained scale, the location and scale of each site;
 * Pathfinder's paths from `path/*`, its k-hat from `paretoK`; the beliefs from `snapshot` messages and the anytime
 * snapshots given (the posterior summary's mean and sd per quantity). With the truth's marginals, the last belief is laid
 * over each as the final posterior. A tape with no `end` event makes a record with status `running`, so the same call serves
 * a run in progress.
 *
 * @param source - The tape and/or the anytime snapshots (`{ step, snapshot }`, as `runAnytime` hands them out).
 * @param options - The prefix, smoothing, limits and the truth.
 * @returns The record.
 * @index function Variational-run record (ELBO, gradient, guide parameters, beliefs) from an svi or pathfinder tape or snapshots.
 * @example
 * ```ts
 * import { elboRunFromTape } from 'omai-engine/exposition'
 * const metric = (step: number, name: string, value: number) => ({ type: 'metric', step, name, value })
 * const events = [metric(1, 'svi/elbo', -9), metric(2, 'svi/elbo', -5), metric(2, 'svi/gradNorm', 3)]
 * const r = elboRunFromTape({ run: { events } })
 * [r.status, r.method, r.elbo.values, r.gradNorm?.values] // => ['running', 'svi', [-9, -5], [3]]
 * ```
 */
export function elboRunFromTape(
  source: { readonly run?: TapeLike; readonly snapshots?: readonly SnapshotAt[] },
  options: ElboRunOptions = {},
): ElboRun {
  const events: readonly TapeEventLike[] = source.run?.events ?? []
  const prefix =
    options.prefix ??
    (() => {
      for (const e of events) {
        const m =
          e.type === 'metric' || e.type === 'tensor' || e.type === 'message'
            ? /^(svi|pathfinder)\//.exec(e.name ?? e.kind ?? '')
            : null
        if (m !== null) return `${m[1]}/`
      }
      return 'svi/'
    })()
  const method = options.method ?? prefix.replace(/\/$/, '')
  const cap = Math.max(2, options.maxPoints ?? 400)
  const scalars = new Map<string, Acc>()
  const tensors = new Map<string, Acc[]>()
  const beliefs = new Map<
    string,
    { steps: number[]; mean: number[]; sd: number[]; lower: (number | undefined)[]; upper: (number | undefined)[] }
  >()
  let status: ElboRun['status'] = 'running'
  const rest = (name: string | undefined): string | undefined =>
    name?.startsWith(prefix) ? name.slice(prefix.length) : undefined
  const ingestSnapshot = (step: number, snap: unknown): void => {
    if (!isObject(snap) || !Array.isArray(snap['summary'])) return
    for (const row of snap['summary'] as unknown[]) {
      if (!isObject(row) || typeof row['name'] !== 'string') continue
      const mean = num(row['mean'])
      if (mean === undefined) continue
      const b = beliefs.get(row['name']) ?? { steps: [], mean: [], sd: [], lower: [], upper: [] }
      const sd = num(row['sd']) ?? 0
      if (b.steps.at(-1) === step) {
        b.mean[b.mean.length - 1] = mean
        b.sd[b.sd.length - 1] = sd
        b.lower[b.lower.length - 1] = num(row['lower'])
        b.upper[b.upper.length - 1] = num(row['upper'])
      } else {
        b.steps.push(step)
        b.mean.push(mean)
        b.sd.push(sd)
        b.lower.push(num(row['lower']))
        b.upper.push(num(row['upper']))
      }
      beliefs.set(row['name'], b)
    }
  }
  for (const e of events) {
    const step = e.step ?? 0
    if (e.type === 'end' && e.status !== undefined) status = e.status as ElboRun['status']
    else if (e.type === 'metric') {
      const name = rest(e.name)
      const v = num(e.value)
      if (name === undefined || v === undefined) continue
      const acc = scalars.get(name) ?? { steps: [], values: [] }
      put(acc, step, v)
      scalars.set(name, acc)
    } else if (e.type === 'tensor') {
      const name = rest(e.name)
      if (name === undefined || !name.startsWith('params/')) continue
      const list = e.values ?? (e.summary?.mean === undefined ? [] : [e.summary.mean])
      const rows = tensors.get(name) ?? []
      for (const [j, v] of list.entries()) {
        if (!Number.isFinite(v)) continue
        const acc = rows[j] ?? { steps: [], values: [] }
        put(acc, step, v)
        rows[j] = acc
      }
      tensors.set(name, rows)
    } else if (e.type === 'message' && rest(e.kind) === 'snapshot') ingestSnapshot(step, e.payload)
  }
  for (const s of source.snapshots ?? []) ingestSnapshot(s.step, s.snapshot)

  const series = (name: string): { steps: number[]; values: number[] } | undefined => {
    const a = scalars.get(name)
    return a === undefined ? undefined : thinned(a, cap)
  }
  const elbo = scalars.get('elbo') ?? { steps: [], values: [] }
  const loss = scalars.get('loss')
  // A tape that logs only the loss has the bound as its negative.
  const bound: Acc =
    elbo.steps.length === 0 && loss !== undefined ? { steps: loss.steps, values: loss.values.map((v) => -v) } : elbo
  const window =
    options.smoothing ??
    (scalars.has('smoothed') ? undefined : bound.steps.length >= 20 ? Math.ceil(bound.steps.length / 20) : 0)
  const own = window === undefined ? series('smoothed') : undefined
  const boundOut = thinned(bound, cap)
  const smoothed =
    own !== undefined
      ? own
      : window !== undefined && window >= 2 && bound.steps.length >= 2
        ? (() => {
            const full = trailingMean(bound.values, window)
            const keep = thinIndices(bound.steps.length, cap)
            return { steps: keep.map((i) => bound.steps[i] as number), values: keep.map((i) => full[i] as number) }
          })()
        : undefined
  const gradNorm = series('gradNorm') ?? series('naturalGradNorm')
  const gradVariance = series('gradVariance')
  const known = new Set([
    'elbo',
    'loss',
    'smoothed',
    'gradNorm',
    'naturalGradNorm',
    'gradVariance',
    'iterate',
    'paretoK',
    'ess',
  ])
  const diagnostics: NamedSeries[] = [...scalars.entries()]
    .filter(([n]) => !known.has(n) && !n.startsWith('path/'))
    .slice(0, options.maxDiagnostics ?? 6)
    .map(([n, a]) => ({ name: n.replace(/^diag\//, ''), ...thinned(a, cap) }))
  const iterate = series('iterate')
  if (iterate !== undefined) diagnostics.unshift({ name: 'iterate', label: 'furthest L-BFGS iteration', ...iterate })
  const paths: NamedSeries[] = [...scalars.entries()]
    .filter(([n]) => n.startsWith('path/'))
    .map(([n, a]) => ({ name: `path ${n.slice('path/'.length)}`, ...thinned(a, cap) }))
  const maxParameters = options.maxParameters ?? 12
  const parameters: GuideParameter[] = []
  for (const [path, rows] of tensors) {
    const base = path.slice('params/'.length)
    for (const [j, acc] of rows.entries()) {
      if (acc === undefined || parameters.length >= maxParameters) continue
      parameters.push({
        name: rows.length === 1 ? base : `${base}[${j}]`,
        ...roleOf(base),
        ...thinned(acc, cap),
      })
    }
  }
  const beliefOut: QuantityBelief[] = [...beliefs.entries()].map(([name, b]) => {
    const keep = thinIndices(b.steps.length, cap)
    const pick = <T>(v: readonly T[]): T[] => keep.map((i) => v[i] as T)
    const lower = pick(b.lower)
    const upper = pick(b.upper)
    return {
      name,
      steps: pick(b.steps),
      values: pick(b.mean),
      sd: pick(b.sd),
      ...(lower.every((v) => v !== undefined) ? { lower: lower as number[] } : {}),
      ...(upper.every((v) => v !== undefined) ? { upper: upper as number[] } : {}),
    }
  })
  let final: ElboRun['final']
  if (options.truth !== undefined) {
    const named = beliefOut.filter((b) => options.truth?.[b.name] !== undefined)
    if (named.length > 0) {
      const normal = Object.fromEntries(
        named.map((b) => [b.name, { mean: b.values.at(-1) as number, sd: b.sd.at(-1) as number }]),
      )
      const truth = Object.fromEntries(named.map((b) => [b.name, options.truth?.[b.name] as MarginalLike]))
      const c = posteriorComparison([{ name: method, normal }], {
        truth,
        quantities: named.map((b) => b.name),
        oracle: options.truthName ?? 'truth',
        gridPoints: 160,
      })
      final = { oracle: c.oracle, method, quantities: c.quantities }
    }
  }
  const paretoK = scalars.get('paretoK')?.values.at(-1)
  const ess = scalars.get('ess')?.values.at(-1)
  return elboRun({
    method,
    status,
    elbo: boundOut,
    ...(smoothed === undefined ? {} : { smoothed }),
    ...(smoothed !== undefined && window !== undefined && own === undefined ? { smoothing: window } : {}),
    ...(gradNorm === undefined ? {} : { gradNorm }),
    ...(gradVariance === undefined ? {} : { gradVariance }),
    diagnostics,
    paths,
    parameters,
    beliefs: beliefOut,
    ...(paretoK === undefined ? {} : { paretoK }),
    ...(ess === undefined ? {} : { ess }),
    ...(final === undefined ? {} : { final }),
  })
}
