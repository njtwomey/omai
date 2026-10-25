/**
 * Reliable evaluation of reinforcement learning results across runs and tasks, following rliable (Agarwal et al.): the
 * interquartile mean, aggregate mean and median, optimality gap and probability of improvement, score normalisation,
 * stratified bootstrap confidence intervals, performance profiles, and learning curves aligned and aggregated with
 * pointwise intervals.
 */

import {
  type Estimate,
  linspaceNumbers,
  maxOf,
  meanOf,
  minOf,
  quantileOf,
  varianceOf,
  WhiteboxError,
} from '../base/index.ts'
import { interp } from '../numerics/interpolate/index.ts'
import { foldIn, type Key, key as makeKey, randint } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'

/** Scores of several runs on several tasks: `scores[run][task]` (rliable's `[num_runs, num_tasks]` matrices). */
export type ScoreMatrix = readonly (readonly number[])[]

/** Scores as a matrix, or a flat list of runs on a single task. */
export type Scores = ScoreMatrix | readonly number[]

/** The aggregate metrics of rliable (Agarwal et al. 2021). */
export type AggregateMetric = 'mean' | 'median' | 'iqm' | 'optimalityGap'

/** Options shared by the bootstrap estimators. */
export interface BootstrapOptions {
  /** The key of the resampling (default `key(seed)`). */
  readonly key?: Key
  readonly seed?: number
  /** Bootstrap replications (default 2000; rliable defaults to 50000). */
  readonly reps?: number
  /** Coverage of the percentile intervals (default 0.95). */
  readonly level?: number
  /** Also resample tasks (rliable's `task_bootstrap`), not only runs within each task. */
  readonly taskBootstrap?: boolean
}

/** A bootstrap estimate: the statistic on the data with a percentile interval and the bootstrap replicates. */
export interface BootstrapEstimate extends Estimate {
  readonly interval: readonly [number, number]
  readonly level: number
  /** The statistic on each bootstrap resample. */
  readonly samples: readonly number[]
}

/** A pointwise series with intervals (performance profiles and aggregated learning curves). */
export interface IntervalSeries {
  /** The x values (thresholds or steps). */
  readonly x: readonly number[]
  readonly value: readonly number[]
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  readonly level: number
}

function asMatrix(scores: Scores, name: string): ScoreMatrix {
  if (scores.length === 0) throw new WhiteboxError('value', `${name}: no scores`)
  const m: ScoreMatrix =
    typeof scores[0] === 'number' ? (scores as readonly number[]).map((x) => [x]) : (scores as ScoreMatrix)
  const nTasks = m[0]?.length ?? 0
  if (nTasks === 0) throw new WhiteboxError('value', `${name}: no tasks`)
  for (const row of m) {
    if (row.length !== nTasks)
      throw new WhiteboxError('value', `${name}: every run needs a score for each of ${nTasks} tasks`)
  }
  return m
}

const flat = (m: ScoreMatrix): number[] => m.flat()

const taskMeans = (m: ScoreMatrix): number[] => (m[0] ?? []).map((_, t) => meanOf(m.map((row) => row[t] ?? Number.NaN)))

/**
 * The interquartile mean (IQM) of all scores across runs and tasks: the mean of the middle 50% after discarding the
 * lowest and highest quarter (`scipy.stats.trim_mean(scores, 0.25, axis=None)`), rliable's recommended aggregate,
 * robust to outlier runs yet more efficient than the median.
 *
 * @param scores - `scores[run][task]`, or the runs of one task.
 * @returns The IQM.
 * @throws WhiteboxError with code `value` for empty or ragged scores.
 * @concept rl-evaluation
 * @reference Agarwal, Schwarzer, Castro, Courville and Bellemare (2021), Deep reinforcement learning at the edge of the
 *   statistical precipice.
 * @example
 * ```ts
 * import { interquartileMean } from 'whitebox-engine/rl'
 * interquartileMean([1, 2, 3, 4, 5, 6, 7, 100]) // => 4.5
 * ```
 */
export function interquartileMean(scores: Scores): number {
  const xs = Float64Array.from(flat(asMatrix(scores, 'interquartileMean'))).sort()
  // scipy's trim_mean cuts int(0.25 n) values from each end.
  const cut = Math.floor(0.25 * xs.length)
  return meanOf(xs.subarray(cut, xs.length - cut))
}

/**
 * The mean over tasks of the mean score of each task (rliable's `aggregate_mean`).
 *
 * @param scores - `scores[run][task]`, or the runs of one task.
 * @returns The mean.
 * @concept rl-evaluation
 * @example
 * ```ts
 * import { aggregateMean } from 'whitebox-engine/rl'
 * aggregateMean([[1, 10], [3, 20]]) // => 8.5
 * ```
 */
export function aggregateMean(scores: Scores): number {
  return meanOf(taskMeans(asMatrix(scores, 'aggregateMean')))
}

/**
 * The median over tasks of the mean score of each task (rliable's `aggregate_median`).
 *
 * @param scores - `scores[run][task]`, or the runs of one task.
 * @returns The median.
 * @concept rl-evaluation
 * @example
 * ```ts
 * import { aggregateMedian } from 'whitebox-engine/rl'
 * aggregateMedian([[1, 10, 4], [3, 20, 6]]) // => 5
 * ```
 */
export function aggregateMedian(scores: Scores): number {
  return quantileOf(taskMeans(asMatrix(scores, 'aggregateMedian')), 0.5)
}

/**
 * The optimality gap: how far scores fall short of a target `gamma` on average, with scores above the target counting
 * as the target (`gamma - mean(min(scores, gamma))`, that is `mean(max(gamma - x, 0))`; lower is better).
 *
 * @param scores - `scores[run][task]`, or the runs of one task (normalised so that `gamma` is the target).
 * @param gamma - The target score (default 1).
 * @returns The optimality gap.
 * @concept rl-evaluation
 * @example
 * ```ts
 * import { optimalityGap } from 'whitebox-engine/rl'
 * optimalityGap([0.5, 1.5, 0.75, 1]) // => 0.1875
 * ```
 */
export function optimalityGap(scores: Scores, gamma = 1): number {
  return meanOf(flat(asMatrix(scores, 'optimalityGap')).map((x) => Math.max(gamma - x, 0)))
}

/** The Mann-Whitney U statistic of `x` over `y` divided by `|x| |y|`: P(X > Y) with ties counted as 1/2. */
function improvement(x: readonly number[], y: readonly number[]): number {
  let u = 0
  for (const a of x) for (const b of y) u += a > b ? 1 : a === b ? 0.5 : 0
  return u / (x.length * y.length)
}

/**
 * The probability of improvement of algorithm X over algorithm Y: for each task, the probability that a random run of
 * X scores higher than a random run of Y (the Mann-Whitney U statistic over `n m`, ties counted as 1/2), averaged over
 * tasks (rliable's `probability_of_improvement`). Runs may differ in number between the two.
 *
 * @param x - Scores of X, `[runs][tasks]`.
 * @param y - Scores of Y on the same tasks.
 * @returns P(X > Y) in `[0, 1]`.
 * @throws WhiteboxError with code `value` when the task counts differ.
 * @concept rl-evaluation
 * @example
 * ```ts
 * import { probabilityOfImprovement } from 'whitebox-engine/rl'
 * probabilityOfImprovement([3, 4, 5], [1, 4, 2]) // => 0.8333333333333334
 * ```
 */
export function probabilityOfImprovement(x: Scores, y: Scores): number {
  const mx = asMatrix(x, 'probabilityOfImprovement')
  const my = asMatrix(y, 'probabilityOfImprovement')
  const nTasks = mx[0]?.length ?? 0
  if ((my[0]?.length ?? 0) !== nTasks)
    throw new WhiteboxError('value', 'probabilityOfImprovement: the task counts differ')
  const per: number[] = []
  for (let t = 0; t < nTasks; t++) {
    per.push(
      improvement(
        mx.map((r) => r[t] ?? 0),
        my.map((r) => r[t] ?? 0),
      ),
    )
  }
  return meanOf(per)
}

/**
 * Normalise scores per task to `(x - low) / (high - low)`: human-normalised scores with `low` the random score and
 * `high` the human score (Atari), or min-max normalisation with reference bounds per task (Procgen).
 *
 * @param scores - `scores[run][task]`.
 * @param low - The score that maps to 0 on each task.
 * @param high - The score that maps to 1 on each task.
 * @returns The normalised scores.
 * @throws WhiteboxError with code `value` when the bounds do not match the tasks or coincide.
 * @concept rl-evaluation
 * @example
 * ```ts
 * import { normalizeScores } from 'whitebox-engine/rl'
 * normalizeScores([[5, 0], [10, 50]], [0, 0], [10, 100]) // => [[0.5, 0], [1, 0.5]]
 * ```
 */
export function normalizeScores(scores: ScoreMatrix, low: readonly number[], high: readonly number[]): number[][] {
  const m = asMatrix(scores, 'normalizeScores')
  const nTasks = m[0]?.length ?? 0
  if (low.length !== nTasks || high.length !== nTasks) {
    throw new WhiteboxError('value', `normalizeScores: need ${nTasks} lower and upper bounds`)
  }
  for (let t = 0; t < nTasks; t++) {
    if ((high[t] ?? 0) === (low[t] ?? 0)) throw new WhiteboxError('value', `normalizeScores: task ${t} has high = low`)
  }
  return m.map((row) => row.map((x, t) => (x - (low[t] ?? 0)) / ((high[t] ?? 1) - (low[t] ?? 0))))
}

/**
 * Min-max normalise the scores of several algorithms together: on each task, the lowest score of any run of any
 * algorithm maps to 0 and the highest to 1, so the algorithms stay comparable across tasks.
 *
 * @param byAlgorithm - Score matrices by algorithm name, on the same tasks.
 * @returns The normalised matrices, by the same names.
 * @concept rl-evaluation
 * @example
 * ```ts
 * import { minMaxNormalizedScores } from 'whitebox-engine/rl'
 * minMaxNormalizedScores({ a: [[0, 5]], b: [[10, 15]] }) // => { a: [[0, 0]], b: [[1, 1]] }
 * ```
 */
export function minMaxNormalizedScores(byAlgorithm: Readonly<Record<string, ScoreMatrix>>): Record<string, number[][]> {
  const mats = Object.values(byAlgorithm).map((m) => asMatrix(m, 'minMaxNormalizedScores'))
  const nTasks = mats[0]?.[0]?.length ?? 0
  const low: number[] = []
  const high: number[] = []
  for (let t = 0; t < nTasks; t++) {
    const column = mats.flatMap((m) => m.map((r) => r[t] ?? 0))
    low.push(minOf(column))
    high.push(maxOf(column))
  }
  const out: Record<string, number[][]> = {}
  for (const [name, m] of Object.entries(byAlgorithm)) {
    out[name] = m.map((row) => row.map((x, t) => (x - (low[t] ?? 0)) / ((high[t] ?? 1) - (low[t] ?? 0) || 1)))
  }
  return out
}

/**
 * The percentile interval of bootstrap replicates: the `(1 - level) / 2` and `(1 + level) / 2` quantiles (linear
 * interpolation, as `numpy.percentile` in the arch bootstrap that rliable uses).
 *
 * @param samples - The replicates.
 * @param level - The coverage (default 0.95).
 * @returns The lower and upper ends.
 * @concept bootstrap
 * @example
 * ```ts
 * import { percentileInterval } from 'whitebox-engine/rl'
 * percentileInterval([0, 1, 2, 3, 4, 5, 6, 7, 8], 0.5) // => [2, 6]
 * ```
 */
export function percentileInterval(samples: ArrayLike<number>, level = 0.95): [number, number] {
  const tail = (1 - level) / 2
  return [quantileOf(samples, tail), quantileOf(samples, 1 - tail)]
}

const bootKey = (options: BootstrapOptions): Key => options.key ?? makeKey(options.seed ?? 0)

/**
 * One stratified resample of a score matrix: optionally resample tasks, then, within each task, resample runs with
 * replacement. Returns, for each output cell, the source cell.
 */
function resampleIndices(
  k: Key,
  nRuns: number,
  nTasks: number,
  taskBootstrap: boolean,
): { readonly runs: ArrayLike<number>; readonly tasks: ArrayLike<number> } {
  const tasks = taskBootstrap
    ? randint(foldIn(k, 1), [nTasks], 0, nTasks).data
    : Int32Array.from({ length: nTasks }, (_, t) => t)
  return { runs: randint(foldIn(k, 0), [nRuns * nTasks], 0, nRuns).data, tasks }
}

function applyIndices(
  m: ScoreMatrix,
  idx: { readonly runs: ArrayLike<number>; readonly tasks: ArrayLike<number> },
): number[][] {
  const nRuns = m.length
  const nTasks = idx.tasks.length
  const out: number[][] = Array.from({ length: nRuns }, () => new Array<number>(nTasks).fill(0))
  for (let t = 0; t < nTasks; t++) {
    const src = idx.tasks[t] ?? t
    for (let i = 0; i < nRuns; i++) {
      const row = out[i] as number[]
      row[t] = m[idx.runs[t * nRuns + i] ?? i]?.[src] ?? 0
    }
  }
  return out
}

/** Replicates of a vector statistic over independent stratified resamples of several score matrices. */
function replicate(
  mats: readonly ScoreMatrix[],
  statistic: (resampled: readonly ScoreMatrix[]) => readonly number[],
  options: BootstrapOptions,
): number[][] {
  const reps = options.reps ?? 2000
  if (!(Number.isInteger(reps) && reps >= 1))
    throw new WhiteboxError('value', 'bootstrap: reps must be a positive integer')
  const k = bootKey(options)
  const out: number[][] = []
  for (let r = 0; r < reps; r++) {
    const kr = foldIn(k, r)
    const resampled = mats.map((m, j) =>
      applyIndices(m, resampleIndices(foldIn(kr, j), m.length, m[0]?.length ?? 0, options.taskBootstrap === true)),
    )
    out.push([...statistic(resampled)])
  }
  return out
}

function bootstrapEstimate(value: number, samples: number[], level: number): BootstrapEstimate {
  return {
    value,
    standardError: Math.sqrt(varianceOf(samples, 1)),
    interval: percentileInterval(samples, level),
    level,
    method: 'stratified bootstrap (percentile)',
    samples,
  }
}

/**
 * A stratified bootstrap confidence interval for any statistic of score matrices (rliable's `StratifiedBootstrap`):
 * each replicate resamples runs with replacement within each task (and, with `taskBootstrap`, the tasks too), so the
 * per-task structure is kept; several matrices (two algorithms for a probability of improvement) are resampled
 * independently. Replicate `r` uses `foldIn(key, r)`, so intervals are reproducible from the key.
 *
 * @param scores - One score matrix, or several (passed to the statistic in order).
 * @param statistic - The statistic of the (resampled) matrices.
 * @param options - Key, replications, level and task resampling.
 * @returns The statistic with its percentile interval, standard error and replicates.
 * @throws WhiteboxError with code `value` for empty or ragged scores.
 * @concept bootstrap
 * @example
 * ```ts
 * import { interquartileMean, stratifiedBootstrap } from 'whitebox-engine/rl'
 * const ci = stratifiedBootstrap([[1, 2], [3, 4], [5, 6], [7, 8]], interquartileMean, { seed: 0, reps: 200 })
 * ci.value // => 4.5
 * ci.interval[0] <= 4.5 && 4.5 <= ci.interval[1] // => true
 * ```
 */
export function stratifiedBootstrap(
  scores: ScoreMatrix | readonly ScoreMatrix[],
  statistic: (...scores: ScoreMatrix[]) => number,
  options: BootstrapOptions = {},
): BootstrapEstimate {
  const many = Array.isArray(scores[0]?.[0])
  const mats = (many ? (scores as readonly ScoreMatrix[]) : [scores as ScoreMatrix]).map((m) =>
    asMatrix(m, 'stratifiedBootstrap'),
  )
  const samples = replicate(mats, (r) => [statistic(...r)], options).map((s) => s[0] ?? Number.NaN)
  return bootstrapEstimate(statistic(...mats), samples, options.level ?? 0.95)
}

const METRICS: Readonly<Record<AggregateMetric, (m: ScoreMatrix, gamma: number) => number>> = {
  mean: (m) => aggregateMean(m),
  median: (m) => aggregateMedian(m),
  iqm: (m) => interquartileMean(m),
  optimalityGap: (m, gamma) => optimalityGap(m, gamma),
}

/**
 * The aggregate metrics of rliable with stratified bootstrap intervals: mean and median of task means, interquartile
 * mean and optimality gap, all from the same resamples (rliable's `get_interval_estimates`).
 *
 * @param scores - `scores[run][task]`, or the runs of one task.
 * @param options - The metrics to compute (default all four), the optimality-gap target `gamma` (default 1) and the
 *   bootstrap options.
 * @returns An estimate per metric.
 * @concept rl-evaluation
 * @example
 * ```ts
 * import { aggregateScores } from 'whitebox-engine/rl'
 * const a = aggregateScores([[2, 9], [4, 11], [6, 13]], { seed: 1, reps: 100 })
 * a.mean?.value // => 7.5
 * Object.keys(a) // => ['mean', 'median', 'iqm', 'optimalityGap']
 * ```
 */
export function aggregateScores(
  scores: Scores,
  options: BootstrapOptions & { readonly metrics?: readonly AggregateMetric[]; readonly gamma?: number } = {},
): Partial<Record<AggregateMetric, BootstrapEstimate>> {
  const m = asMatrix(scores, 'aggregateScores')
  const names = options.metrics ?? (['mean', 'median', 'iqm', 'optimalityGap'] as const)
  const gamma = options.gamma ?? 1
  const level = options.level ?? 0.95
  const reps = replicate([m], ([r]) => names.map((n) => METRICS[n](r ?? m, gamma)), options)
  const out: Partial<Record<AggregateMetric, BootstrapEstimate>> = {}
  for (const [j, n] of names.entries()) {
    out[n] = bootstrapEstimate(
      METRICS[n](m, gamma),
      reps.map((s) => s[j] ?? Number.NaN),
      level,
    )
  }
  return out
}

/**
 * The probability of improvement of X over Y with a stratified bootstrap interval, the two algorithms' runs resampled
 * independently (rliable's `StratifiedIndependentBootstrap`).
 *
 * @param x - Scores of X, `[runs][tasks]`.
 * @param y - Scores of Y on the same tasks.
 * @param options - Bootstrap options.
 * @returns P(X > Y) with its interval.
 * @concept rl-evaluation
 * @example
 * ```ts
 * import { improvementEstimate } from 'whitebox-engine/rl'
 * improvementEstimate([5, 6, 7], [1, 2, 3], { seed: 0, reps: 50 }).interval // => [1, 1]
 * ```
 */
export function improvementEstimate(x: Scores, y: Scores, options: BootstrapOptions = {}): BootstrapEstimate {
  return stratifiedBootstrap(
    [asMatrix(x, 'improvementEstimate'), asMatrix(y, 'improvementEstimate')],
    probabilityOfImprovement,
    options,
  )
}

/**
 * Performance profiles (Dolan and Moré 2002, as rliable's `create_performance_profile`): for each threshold `tau`, the
 * fraction of runs, over all runs and tasks, scoring above `tau` (the run-score distribution, default), or with
 * `kind: 'average'` the fraction of tasks whose mean score is above `tau`; with pointwise stratified bootstrap
 * intervals.
 *
 * @param scores - `scores[run][task]`.
 * @param taus - The thresholds.
 * @param options - `kind` and the bootstrap options (`reps: 0` skips the intervals).
 * @returns The profile with its intervals.
 * @concept rl-evaluation
 * @example
 * ```ts
 * import { performanceProfile } from 'whitebox-engine/rl'
 * performanceProfile([[0.1, 0.5], [0.7, 0.9]], [0, 0.5, 1], { reps: 0 }).value // => [1, 0.5, 0]
 * ```
 */
export function performanceProfile(
  scores: Scores,
  taus: readonly number[],
  options: BootstrapOptions & { readonly kind?: 'run' | 'average' } = {},
): IntervalSeries {
  const m = asMatrix(scores, 'performanceProfile')
  const average = options.kind === 'average'
  const profile = (s: ScoreMatrix): number[] => {
    const xs = average ? taskMeans(s) : flat(s)
    return taus.map((tau) => {
      let above = 0
      for (const x of xs) if (x > tau) above++
      return above / xs.length
    })
  }
  const value = profile(m)
  const level = options.level ?? 0.95
  if (options.reps === 0) return { x: [...taus], value, lower: value, upper: value, level }
  const reps = replicate([m], ([r]) => profile(r ?? m), options)
  const lower: number[] = []
  const upper: number[] = []
  for (let j = 0; j < taus.length; j++) {
    const [lo, hi] = percentileInterval(
      reps.map((s) => s[j] ?? 0),
      level,
    )
    lower.push(lo)
    upper.push(hi)
  }
  return { x: [...taus], value, lower, upper, level }
}

/** A learning curve: values per step (steps 0, 1, 2, ...), or explicit steps and values. */
export type LearningCurve =
  | readonly number[]
  | { readonly steps: readonly number[]; readonly values: readonly number[] }

/**
 * Put learning curves on a common step grid: curves that already share their steps are kept as they are; otherwise
 * every curve is linearly interpolated (`numpy.interp`) onto `grid`, or onto `points` evenly spaced steps over the range
 * that all curves cover (no extrapolation).
 *
 * @param curves - The curves.
 * @param options - An explicit grid, or the number of grid points (default the shortest curve's length).
 * @returns The grid and the values of every curve on it.
 * @throws WhiteboxError with code `value` when there are no curves, a curve is empty, or the curves do not overlap.
 * @concept rl-evaluation
 * @example
 * ```ts
 * import { alignCurves } from 'whitebox-engine/rl'
 * alignCurves([{ steps: [0, 10], values: [0, 1] }, { steps: [0, 5, 10], values: [0, 2, 2] }], { points: 3 }).values
 * // => [[0, 0.5, 1], [0, 2, 2]]
 * ```
 */
export function alignCurves(
  curves: readonly LearningCurve[],
  options: { readonly grid?: readonly number[]; readonly points?: number } = {},
): { readonly steps: number[]; readonly values: number[][] } {
  if (curves.length === 0) throw new WhiteboxError('value', 'alignCurves: no curves')
  const parts = curves.map((c) => {
    const values = 'steps' in c ? c.values : (c as readonly number[])
    const steps = 'steps' in c ? c.steps : values.map((_, i) => i)
    if (values.length === 0 || steps.length !== values.length) {
      throw new WhiteboxError('value', 'alignCurves: every curve needs as many steps as values, and at least one')
    }
    return { steps, values }
  })
  const first = parts[0]?.steps ?? []
  const same = parts.every((p) => p.steps.length === first.length && p.steps.every((s, i) => s === first[i]))
  if (options.grid === undefined && same) return { steps: [...first], values: parts.map((p) => [...p.values]) }
  let grid = options.grid
  if (grid === undefined) {
    const lo = maxOf(parts.map((p) => p.steps[0] ?? 0))
    const hi = minOf(parts.map((p) => p.steps[p.steps.length - 1] ?? 0))
    if (!(hi >= lo)) throw new WhiteboxError('value', 'alignCurves: the curves cover no common range of steps')
    const n = options.points ?? minOf(parts.map((p) => p.steps.length))
    grid = lo === hi ? [lo] : linspaceNumbers(lo, hi, n)
  }
  const g = [...grid]
  return { steps: g, values: parts.map((p) => toFlat(interp(g, [...p.steps], [...p.values]))) }
}

/**
 * Aggregate learning curves across runs (and tasks) step by step, with pointwise stratified bootstrap intervals
 * (rliable's sample-efficiency curves): curves are aligned on a common grid ({@link alignCurves}), each step's scores
 * form a `[runs][tasks]` matrix, and each bootstrap replicate resamples runs within tasks once for all steps.
 *
 * @param curves - One curve per run.
 * @param options - The aggregate (`'iqm'` default, `'mean'`, `'median'` or `'optimalityGap'`), the task of each
 *   curve (default all one task; every task needs the same number of runs), the grid and the bootstrap options (`reps:
 *   0` skips the intervals).
 * @returns The aggregate curve with its intervals.
 * @throws WhiteboxError with code `value` when tasks have different numbers of runs.
 * @concept rl-evaluation
 * @example
 * ```ts
 * import { aggregateCurves } from 'whitebox-engine/rl'
 * aggregateCurves([[0, 1, 2], [0, 3, 4]], { aggregate: 'mean', reps: 0 }).value // => [0, 2, 3]
 * ```
 */
export function aggregateCurves(
  curves: readonly LearningCurve[],
  options: BootstrapOptions & {
    readonly aggregate?: AggregateMetric
    readonly tasks?: readonly number[]
    readonly grid?: readonly number[]
    readonly points?: number
    readonly gamma?: number
  } = {},
): IntervalSeries {
  const { steps, values } = alignCurves(curves, {
    ...(options.grid === undefined ? {} : { grid: options.grid }),
    ...(options.points === undefined ? {} : { points: options.points }),
  })
  const tasks = options.tasks ?? values.map(() => 0)
  if (tasks.length !== values.length) throw new WhiteboxError('value', 'aggregateCurves: give a task for every curve')
  const ids = [...new Set(tasks)].sort((a, b) => a - b)
  const byTask = ids.map((id) => values.filter((_, i) => tasks[i] === id))
  const nRuns = byTask[0]?.length ?? 0
  if (byTask.some((b) => b.length !== nRuns)) {
    throw new WhiteboxError('value', 'aggregateCurves: every task needs the same number of runs')
  }
  // cube[step] is the [runs][tasks] matrix of that step.
  const cube: number[][][] = steps.map((_, j) =>
    Array.from({ length: nRuns }, (_, i) => byTask.map((b) => b[i]?.[j] ?? Number.NaN)),
  )
  const metric = METRICS[options.aggregate ?? 'iqm']
  const gamma = options.gamma ?? 1
  const value = cube.map((m) => metric(m, gamma))
  const level = options.level ?? 0.95
  if (options.reps === 0) return { x: steps, value, lower: value, upper: value, level }
  const reps = options.reps ?? 2000
  const k = bootKey(options)
  const draws: number[][] = []
  for (let r = 0; r < reps; r++) {
    const idx = resampleIndices(foldIn(foldIn(k, r), 0), nRuns, ids.length, options.taskBootstrap === true)
    draws.push(cube.map((m) => metric(applyIndices(m, idx), gamma)))
  }
  const lower: number[] = []
  const upper: number[] = []
  for (let j = 0; j < steps.length; j++) {
    const [lo, hi] = percentileInterval(
      draws.map((d) => d[j] ?? 0),
      level,
    )
    lower.push(lo)
    upper.push(hi)
  }
  return { x: steps, value, lower, upper, level }
}
