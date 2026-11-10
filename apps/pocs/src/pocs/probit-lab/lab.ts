/**
 * The lab's computing, free of any thread: one inference method on the probit model of a problem, run for a time budget
 * through the library's `runAnytimeAsync`, recorded as frames (a snapshot of the run every fraction of the budget), and
 * finished into plain data: the draws with the time each was available, the predictive probability over the plane at every
 * frame, the coefficients, the diagnostics and the held-out scores. A worker runs this (`worker.ts`); the tests and a
 * page without Workers run it on the main thread. The library has no clock, so the caller gives one.
 */

import type { Posterior } from 'omai-engine/probprog'
import { posteriorChains } from 'omai-engine/probprog'
import { inferenceMethodFor, modelRegistry, runAnytimeAsync } from 'omai-engine/probprog/inference'
import {
  heldOutScores,
  polynomialFeatureMap,
  probitClassifier,
  probitData,
  probitPredictive,
  probitPredictiveGrid,
} from 'omai-engine/probprog/models'
import { key } from 'omai-engine/random'
import { ess, rhat } from 'omai-engine/sampling'
import { tensor, toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import { type LabMethod, methodById } from './methods.ts'
import type {
  CoefficientSummary,
  Ending,
  LabFrame,
  LabRequest,
  LabStatus,
  MethodResult,
  RunCurve,
  Scores,
} from './types.ts'

/** What the lab's run needs from its host: a clock, a way to hand over the thread, a stop flag and a place to say how it is going. */
export interface LabHost {
  readonly now: () => number
  readonly yieldToPage: () => Promise<void>
  readonly stopped: () => boolean
  readonly status: (status: LabStatus) => void
}

/** The most frames kept of a run. */
export const MAX_FRAMES = 80
/** The most draws kept of a sampler's pool, over all its chains. */
export const MAX_POOL = 600
/** The draws a frame's predictive grid is made from. */
const GRID_DRAWS = 60
/** The draws the held-out scores are made from. */
const SCORE_DRAWS = 400
/** The most points of a curve. */
const MAX_CURVE = 300

/** `max` indices spread evenly over `count` items, the first and the last included. */
export function evenly(count: number, max: number): number[] {
  if (count <= max) return Array.from({ length: count }, (_, i) => i)
  return Array.from({ length: max }, (_, i) => Math.round((i * (count - 1)) / (max - 1)))
}

const rowsOf = (flat: ArrayLike<number>, p: number): number[][] =>
  Array.from({ length: Math.floor(flat.length / p) }, (_, i) =>
    Array.from({ length: p }, (_, j) => flat[i * p + j] ?? Number.NaN),
  )

function meanSd(rows: readonly (readonly number[])[], p: number): { mean: Float64Array; sd: Float64Array } {
  const mean = new Float64Array(p).fill(Number.NaN)
  const sd = new Float64Array(p).fill(Number.NaN)
  const n = rows.length
  if (n === 0) return { mean, sd }
  for (let j = 0; j < p; j++) {
    let m = 0
    for (const r of rows) m += (r[j] ?? 0) / n
    let v = 0
    for (const r of rows) v += ((r[j] ?? 0) - m) ** 2
    mean[j] = m
    sd[j] = n > 1 ? Math.sqrt(v / (n - 1)) : 0
  }
  return { mean, sd }
}

const finite = (v: ArrayLike<number>): boolean => Array.from(v).every(Number.isFinite)

interface Raw {
  readonly step: number
  readonly elapsedMs: number
  readonly posterior: Posterior | undefined
}

/**
 * Runs one method of the lab on a problem for the time budget and returns everything the page draws.
 *
 * @param request - The problem, the method, the budget, the seed and the size of the predictive grid.
 * @param host - The clock, the hand-over, the stop flag and the status line.
 * @returns The method's result.
 */
export async function runLab(request: LabRequest, host: LabHost): Promise<MethodResult> {
  const spec = methodById(request.method)
  const { problem } = request
  const { settings } = problem
  const map = polynomialFeatureMap(problem.train.x, settings.degree)
  const design = map.transform(problem.train.x)
  const heldOut = map.transform(problem.test.x)
  const p = map.size
  const names = Array.from({ length: p }, (_, j) => `w[${j}]`)
  const registry = modelRegistry({
    probit: {
      definition: probitClassifier(settings.priorScale * map.weightScale),
      data: probitData(design, problem.train.y),
    },
  })
  const method = inferenceMethodFor(registry, spec.kind, 'probit')
  const config = { ...spec.config }
  const s = session(method, { config, seed: request.seed, steps: 1_000_000_000, mode: 'live' })
  const posteriorOf = (state: unknown): Posterior | undefined => {
    try {
      const post = method.posterior?.(state as never, s.config as never)
      return post !== undefined && post.sites.includes('w') ? post : undefined
    } catch {
      return undefined
    }
  }
  const iterate = spec.organisation === 'iterate'
  const raw: Raw[] = []
  const t0 = host.now()
  const run = await runAnytimeAsync(s, {
    budget: { ms: request.budgetMs },
    snapshot: {
      everyMs: Math.max(20, request.budgetMs / MAX_FRAMES),
      ...(spec.everySteps === undefined ? {} : { everySteps: spec.everySteps }),
    },
    clock: host.now,
    sliceMs: 40,
    yieldToPage: host.yieldToPage,
    shouldStop: () => {
      host.status({ type: 'status', phase: 'running', step: s.t, elapsedMs: host.now() - t0 })
      return host.stopped()
    },
    onSnapshot: (x) => {
      raw.push({ step: x.t, elapsedMs: x.elapsedMs, posterior: iterate ? posteriorOf(x.state) : undefined })
    },
  })
  host.status({ type: 'status', phase: 'finishing', step: s.t, elapsedMs: host.now() - t0 })
  let ending: Ending =
    run.reason === 'cancelled'
      ? 'stopped'
      : run.reason === 'budget'
        ? 'budget'
        : (s.status ?? 'completed') === 'stopped'
          ? 'completed'
          : ((s.status ?? 'completed') as Ending)
  let message: string | undefined
  const finalPosterior = iterate ? undefined : posteriorOf(s.state)
  const base = {
    id: spec.id,
    label: spec.label,
    organisation: spec.organisation,
    steps: run.t,
    elapsedMs: run.elapsedMs,
    names,
    gridSize: request.gridSize,
  }
  const curve = curveOf(spec, s.run().events as readonly CurveEvent[])
  const gridOf = (rows: readonly (readonly number[])[]): Float32Array => {
    const n = request.gridSize
    if (rows.length === 0) return new Float32Array(n * n).fill(Number.NaN)
    return Float32Array.from(probitPredictiveGrid(rows, map, problem.extent, n).probabilities)
  }
  const scores = (rows: readonly (readonly number[])[]): { train: Scores; test: Scores } => {
    if (rows.length === 0) {
      const none = { accuracy: Number.NaN, logPredictive: Number.NaN }
      return { train: none, test: none }
    }
    return {
      train: heldOutScores(probitPredictive(rows, design), problem.train.y),
      test: heldOutScores(probitPredictive(rows, heldOut), problem.test.y),
    }
  }

  if (!iterate) {
    const chains = finalPosterior === undefined ? undefined : posteriorChains(finalPosterior, { sites: ['w'] })
    const series = names.map((n) => chains?.draws[n])
    const numChains = series[0]?.length ?? 0
    const perChain = series[0]?.[0]?.length ?? 0
    const keep = evenly(perChain, Math.max(2, Math.floor(MAX_POOL / Math.max(1, numChains))))
    const total = keep.length * numChains
    const values = new Float64Array(total * p)
    const at = new Float64Array(total)
    const until = new Float64Array(total).fill(Number.POSITIVE_INFINITY)
    const chain = new Int32Array(total)
    const order = new Int32Array(total)
    // The step at which each draw was made: the draws are the last `perChain` steps of the run, one per step.
    const stepOf = (i: number): number =>
      run.t >= perChain ? run.t - perChain + i + 1 : (run.t * (i + 1)) / Math.max(1, perChain)
    const marks = [
      { step: 0, elapsedMs: 0 },
      ...raw.map((r) => ({ step: r.step, elapsedMs: r.elapsedMs })),
      { step: run.t, elapsedMs: run.elapsedMs },
    ]
    const elapsedAt = (step: number): number => {
      for (let k = 1; k < marks.length; k++) {
        const hi = marks[k]
        const lo = marks[k - 1]
        if (hi !== undefined && lo !== undefined && step <= hi.step)
          return hi.step === lo.step
            ? hi.elapsedMs
            : lo.elapsedMs + ((hi.elapsedMs - lo.elapsedMs) * (step - lo.step)) / (hi.step - lo.step)
      }
      return run.elapsedMs
    }
    let row = 0
    for (let c = 0; c < numChains; c++)
      for (const i of keep) {
        for (let j = 0; j < p; j++) values[row * p + j] = series[j]?.[c]?.[i] ?? Number.NaN
        at[row] = elapsedAt(stepOf(i))
        chain[row] = c
        order[row] = i
        row++
      }
    const pool = { p, values, at, until, chain, order }
    const visibleAt = (t: number): number[] =>
      Array.from({ length: total }, (_, i) => i).filter((i) => (at[i] ?? 0) <= t)
    const rowOf = (i: number): number[] => Array.from(values.subarray(i * p, (i + 1) * p))
    const frames: LabFrame[] = []
    const grids: number[] = []
    for (const k of evenly(raw.length, MAX_FRAMES)) {
      const f = raw[k] as Raw
      const seen = visibleAt(f.elapsedMs)
      const { mean, sd } = meanSd(seen.map(rowOf), p)
      frames.push({ step: f.step, elapsedMs: f.elapsedMs, mean, sd })
      grids.push(...gridOf(evenly(seen.length, GRID_DRAWS).map((q) => rowOf(seen[q] as number))))
    }
    const all = Array.from({ length: total }, (_, i) => rowOf(i))
    // R-hat and ESS of at most 400 draws per chain (the effective sample size costs time that grows as the square of the draws).
    const stride = Math.max(1, Math.ceil(perChain / 400))
    const thinned = (j: number): number[][] => (series[j] ?? []).map((row) => row.filter((_, i) => i % stride === 0))
    const moments = meanSd(all, p)
    const coefficients: CoefficientSummary[] = names.map((_, j) => {
      const column = all.map((x) => x[j] ?? 0).sort((a, b) => a - b)
      const x = thinned(j)
      const usable = numChains >= 2 && (x[0]?.length ?? 0) >= 8
      return {
        mean: moments.mean[j] ?? Number.NaN,
        sd: moments.sd[j] ?? Number.NaN,
        lower: column[Math.floor(0.05 * column.length)] ?? Number.NaN,
        upper: column[Math.min(column.length - 1, Math.floor(0.95 * column.length))] ?? Number.NaN,
        rhat: usable ? rhat(tensor(x)) : undefined,
        ess: usable ? ess(tensor(x)) * stride : undefined,
      }
    })
    const rhats = coefficients.flatMap((c) => (c.rhat === undefined ? [] : [c.rhat]))
    const esses = coefficients.flatMap((c) => (c.ess === undefined ? [] : [c.ess]))
    const divergent = finalPosterior?.diagnostics['divergences']
    if (total === 0) message = 'no draws yet: the budget ended during warm-up'
    return {
      ...base,
      gaussian: false,
      ending,
      message,
      frames,
      grids: Float32Array.from(grids),
      pool,
      curve,
      coefficients,
      ...scores(all.length > SCORE_DRAWS ? evenly(all.length, SCORE_DRAWS).map((i) => all[i] as number[]) : all),
      maxRhat: rhats.length === 0 ? undefined : Math.max(...rhats),
      minEss: esses.length === 0 ? undefined : Math.min(...esses),
      divergences: typeof divergent === 'number' ? divergent : undefined,
      draws: total === 0 ? 0 : perChain * numChains,
    }
  }

  // An approximation is redrawn at each snapshot; a fit that diverged keeps the snapshots before it did.
  const usable = raw.filter((r) => {
    const post = r.posterior
    if (post === undefined) return false
    try {
      return finite(toFlat(post.mean('w')))
    } catch {
      return false
    }
  })
  if (usable.length < raw.length && usable.length > 0 && s.status === 'diverged') {
    ending = 'diverged'
    message = `the ELBO or a parameter left the finite numbers at step ${run.t}; the last finite snapshot (step ${usable.at(-1)?.step}) is shown`
  } else if (usable.length === 0) {
    ending = 'failed'
    message = 'no usable snapshot: the run ended before the first one'
  }
  const kept = evenly(usable.length, MAX_FRAMES).map((k) => usable[k] as Raw)
  const total = kept.length * GRID_DRAWS
  const values = new Float64Array(total * p)
  const at = new Float64Array(total)
  const until = new Float64Array(total)
  const frames: LabFrame[] = []
  const grids: number[] = []
  let lastDraws: number[][] = []
  for (const [k, r] of kept.entries()) {
    const post = r.posterior as Posterior
    const draws = rowsOf(toFlat(post.sample(key(1000 + k), GRID_DRAWS)['w'] as never), p)
    lastDraws = draws
    frames.push({
      step: r.step,
      elapsedMs: r.elapsedMs,
      mean: Float64Array.from(toFlat(post.mean('w'))),
      sd: Float64Array.from(toFlat(post.stddev('w'))),
    })
    grids.push(...gridOf(draws))
    for (const [i, d] of draws.entries()) {
      const row = k * GRID_DRAWS + i
      values.set(d, row * p)
      at[row] = r.elapsedMs
      until[row] = kept[k + 1]?.elapsedMs ?? Number.POSITIVE_INFINITY
    }
  }
  const last = kept.at(-1)?.posterior
  const coefficients: CoefficientSummary[] = names.map((_, j) => {
    const q = last?.quantiles('w', [0.05, 0.95])
    const flat = q === undefined ? [] : Array.from(toFlat(q))
    return {
      mean: frames.at(-1)?.mean[j] ?? Number.NaN,
      sd: frames.at(-1)?.sd[j] ?? Number.NaN,
      lower: flat[j] ?? Number.NaN,
      upper: flat[p + j] ?? Number.NaN,
      rhat: undefined,
      ess: undefined,
    }
  })
  const scoreDraws = last === undefined ? [] : rowsOf(toFlat(last.sample(key(7), SCORE_DRAWS)['w'] as never), p)
  void lastDraws
  return {
    ...base,
    gaussian: last?.form === 'variational',
    ending,
    message,
    frames,
    grids: Float32Array.from(grids),
    pool: {
      p,
      values,
      at,
      until,
      chain: new Int32Array(total),
      order: Int32Array.from({ length: total }, (_, i) => i),
    },
    curve,
    coefficients,
    ...scores(scoreDraws),
    maxRhat: undefined,
    minEss: undefined,
    divergences: undefined,
    draws: undefined,
  }
}

interface CurveEvent {
  readonly type?: string
  readonly name?: string
  readonly step?: number
  readonly value?: number
}

function curveOf(spec: LabMethod, events: readonly CurveEvent[]): RunCurve | undefined {
  if (spec.curve === undefined) return undefined
  const mine = events.filter(
    (e) => e.name === spec.curve?.path && typeof e.value === 'number' && Number.isFinite(e.value),
  )
  if (mine.length === 0) return undefined
  const pick = evenly(mine.length, MAX_CURVE).map((i) => mine[i] as CurveEvent)
  return {
    label: spec.curve.label,
    direction: spec.curve.direction,
    steps: Float64Array.from(pick, (e) => e.step ?? 0),
    values: Float64Array.from(pick, (e) => e.value ?? 0),
  }
}
