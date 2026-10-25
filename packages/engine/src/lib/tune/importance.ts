/**
 * Hyperparameter importance from a finished study, which says which parameters mattered for the objective: rank
 * correlation, permutation importance on a Gaussian-process surrogate, functional ANOVA (fANOVA, the share of the
 * surrogate's variance each parameter explains) and PED-ANOVA, which compares the best trials with the rest.
 */

import { meanOf, varianceOf, WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey, permutation, scrambledSobol } from '../random/index.ts'
import { pearson, spearman } from '../stats/index.ts'
import { type Leaf, leafCoordinate, leavesOf, nativeOf } from './_leaves.ts'
import { type ParzenEstimator, parzenEstimator, parzenLogPdf } from './parzen.ts'
import { completeTrials, lossOf, type Study, type Trial } from './study.ts'
import { fitGpSurrogate, type GpSurrogate } from './surrogate.ts'
import { splitTrials } from './tpe.ts'

/** Parameter importances by parameter path, largest first. */
export type Importances = Readonly<Record<string, number>>

function sorted(entries: readonly (readonly [string, number])[]): Importances {
  return Object.fromEntries([...entries].sort((a, b) => b[1] - a[1]))
}

function normalised(entries: readonly (readonly [string, number])[]): Importances {
  const total = entries.reduce((s, [, v]) => s + v, 0)
  return sorted(entries.map(([k, v]) => [k, total > 0 ? v / total : 1 / entries.length]))
}

interface Data {
  readonly leaves: Leaf[]
  readonly trials: Trial[]
  readonly natives: number[][]
  readonly losses: number[]
}

function dataOf(study: Study, objective: number): Data {
  const d = study.directions[objective]
  if (d === undefined) throw new WhiteboxError('value', `importance: no objective ${objective}`)
  const trials = completeTrials(study).filter((t) => t.values.every(Number.isFinite))
  return {
    leaves: leavesOf(study.space),
    trials,
    natives: trials.map((t) => nativeOf(study.space, t.params)),
    losses: trials.map((t) => lossOf(d, t.values[objective] ?? 0)),
  }
}

/**
 * Importance by rank correlation: |Spearman's rho| (or |Pearson's r|) between each parameter's unit-cube coordinate
 * (option index for options) and the objective over the complete trials where it is active, normalised to sum to 1.
 * Cheap, and blind to non-monotone effects.
 *
 * @param study - The study.
 * @param options - The correlation (default `'spearman'`) and objective index.
 * @returns Importances by parameter, summing to 1.
 * @concept hyperparameter-importance
 * @example
 * ```ts
 * import { createStudy, correlationImportance } from 'whitebox-engine/tune'
 * import { dict, real } from 'whitebox-engine/space'
 * const trials = [0, 1, 2, 3, 4, 5].map((i) => ({ number: i, params: { a: i / 5, b: ((i * 7) % 6) / 5 }, state: 'complete', values: [i], intermediate: [], source: 'x' }) as const)
 * Object.keys(correlationImportance(createStudy(dict({ a: real(0, 1), b: real(0, 1) }), { trials })))[0] // => 'a'
 * ```
 */
export function correlationImportance(
  study: Study,
  options: { readonly method?: 'spearman' | 'pearson'; readonly objective?: number } = {},
): Importances {
  const { leaves, natives, losses } = dataOf(study, options.objective ?? 0)
  const test = options.method === 'pearson' ? pearson : spearman
  return normalised(
    leaves.map((l, j) => {
      const rows = natives.flatMap((r, i) => (Number.isNaN(r[j] ?? Number.NaN) ? [] : [i]))
      const x = rows.map((i) =>
        l.kind === 'categorical' ? (natives[i]?.[j] ?? 0) : leafCoordinate(l, natives[i]?.[j] ?? 0),
      )
      const y = rows.map((i) => losses[i] as number)
      if (rows.length < 3 || varianceOf(x) === 0 || varianceOf(y) === 0) return [l.path, 0] as const
      const r = test(x, y).statistic
      return [l.path, Number.isFinite(r) ? Math.abs(r) : 0] as const
    }),
  )
}

function surrogateOf(data: Data): { gp: GpSurrogate; columns: number[][] } {
  const columns = data.natives.map((r) =>
    data.leaves.map((l, j) => (l.kind === 'categorical' ? (r[j] ?? -1) : leafCoordinate(l, r[j] ?? Number.NaN))),
  )
  const gp = fitGpSurrogate(columns, data.losses, {
    categorical: data.leaves.map((l) => l.kind === 'categorical'),
    sizes: data.leaves.map((l) => (l.kind === 'categorical' ? l.n : 1)),
  })
  return { gp, columns }
}

function needTrials(data: Data, n: number, name: string): void {
  if (data.trials.length < n) throw new WhiteboxError('value', `${name}: need at least ${n} complete trials`)
}

/**
 * Permutation importance on a surrogate: fit a Gaussian process to the complete trials, then for each parameter
 * shuffle its column (`repeats` times) and measure how much the surrogate's squared error on the trials grows,
 * normalised to sum to 1 (Breiman's permutation importance, with the surrogate as the model).
 *
 * @param study - The study.
 * @param options - Repeats (default 5), key and objective index.
 * @returns Importances by parameter, summing to 1.
 * @throws WhiteboxError with code `value` with fewer than 3 complete trials.
 * @concept hyperparameter-importance
 * @example
 * ```ts
 * import { createStudy, permutationImportance } from 'whitebox-engine/tune'
 * import { dict, real } from 'whitebox-engine/space'
 * const trials = Array.from({ length: 12 }, (_, i) => {
 *   const a = (i % 4) / 3
 *   const b = Math.floor(i / 4) / 2
 *   return { number: i, params: { a, b }, state: 'complete', values: [5 * a + 0.1 * b], intermediate: [], source: 'x' } as const
 * })
 * Object.keys(permutationImportance(createStudy(dict({ a: real(0, 1), b: real(0, 1) }), { trials })))[0] // => 'a'
 * ```
 */
export function permutationImportance(
  study: Study,
  options: { readonly repeats?: number; readonly key?: Key; readonly objective?: number } = {},
): Importances {
  const data = dataOf(study, options.objective ?? 0)
  needTrials(data, 3, 'permutationImportance')
  const { gp, columns } = surrogateOf(data)
  const error = (rows: readonly (readonly number[])[]): number => {
    const mean = gp.predict(rows).mean
    return meanOf(mean.map((m, i) => (m - (data.losses[i] as number)) ** 2))
  }
  const base = error(columns)
  const repeats = options.repeats ?? 5
  const key = options.key ?? makeKey(0)
  return normalised(
    data.leaves.map((l, j) => {
      let total = 0
      for (let r = 0; r < repeats; r++) {
        const p = permutation(foldIn(foldIn(key, j), r), columns.length).data
        total +=
          error(columns.map((row, i) => row.map((v, c) => (c === j ? (columns[p[i] as number]?.[j] ?? v) : v)))) - base
      }
      return [l.path, Math.max(0, total / repeats)] as const
    }),
  )
}

/**
 * Functional ANOVA importance (Hutter, Hoos and Leyton-Brown 2014) of a Gaussian-process surrogate: the fraction of the
 * surrogate's variance over the unit cube explained by each parameter's main effect, Var_xj(E_x-j[f]) / Var(f),
 * estimated on `nSamples` scrambled Sobol' points for the other parameters and a grid of `gridSize` values (every
 * option for options) of the parameter; normalised to sum to 1, as Optuna reports them. Optuna fits random forests;
 * this decomposes a GP surrogate instead.
 *
 * @param study - The study.
 * @param options - Monte Carlo points (default 128), grid size (default 16), key and objective index.
 * @returns Importances by parameter, summing to 1.
 * @throws WhiteboxError with code `value` with fewer than 3 complete trials.
 * @concept hyperparameter-importance
 * @reference Hutter, Hoos and Leyton-Brown (2014), An efficient approach for assessing hyperparameter importance.
 * @example
 * ```ts
 * import { createStudy, fanovaImportance } from 'whitebox-engine/tune'
 * import { dict, real } from 'whitebox-engine/space'
 * const trials = Array.from({ length: 16 }, (_, i) => {
 *   const a = (i % 4) / 3
 *   const b = Math.floor(i / 4) / 3
 *   return { number: i, params: { a, b }, state: 'complete', values: [Math.sin(3 * a) + 0.05 * b], intermediate: [], source: 'x' } as const
 * })
 * (fanovaImportance(createStudy(dict({ a: real(0, 1), b: real(0, 1) }), { trials }))['a'] ?? 0) > 0.9 // => true
 * ```
 */
export function fanovaImportance(
  study: Study,
  options: {
    readonly nSamples?: number
    readonly gridSize?: number
    readonly key?: Key
    readonly objective?: number
  } = {},
): Importances {
  const data = dataOf(study, options.objective ?? 0)
  needTrials(data, 3, 'fanovaImportance')
  const { gp } = surrogateOf(data)
  const d = data.leaves.length
  const m = options.nSamples ?? 128
  const g = options.gridSize ?? 16
  const u = scrambledSobol(options.key ?? makeKey(0), m, Math.max(1, d)).data
  const base = Array.from({ length: m }, (_, i) =>
    data.leaves.map((l, j) => {
      const v = u[i * Math.max(1, d) + j] ?? 0.5
      return l.kind === 'categorical' ? Math.min(l.n - 1, Math.floor(v * l.n)) : v
    }),
  )
  const all: number[] = []
  const effects = data.leaves.map((l, j) => {
    const ticks =
      l.kind === 'categorical'
        ? Array.from({ length: l.n }, (_, i) => i)
        : Array.from({ length: g }, (_, i) => (i + 0.5) / g)
    const marginal = ticks.map((t) => {
      const mean = gp.predict(base.map((row) => row.map((v, c) => (c === j ? t : v)))).mean
      all.push(...mean)
      return meanOf(mean)
    })
    return [l.path, varianceOf(marginal)] as const
  })
  const total = varianceOf(all)
  return normalised(effects.map(([k, v]) => [k, total > 0 ? v / total : 0]))
}

/** Optuna's histogram of a leaf's values on its PED-ANOVA grid: counts per cell. */
function gridCounts(leaf: Leaf, values: readonly number[], nSteps: number): number[] {
  if (leaf.kind === 'categorical') {
    const counts = new Array<number>(leaf.n).fill(0)
    for (const v of values) counts[v] = (counts[v] ?? 0) + 1
    return counts
  }
  let steps = nSteps
  if (leaf.kind === 'int' && leaf.log)
    steps = Math.min(Math.ceil(Math.log(leaf.high - leaf.low + 1) / Math.log(2)) + 1, steps)
  else if (leaf.step !== undefined) steps = Math.min(Math.round((leaf.high - leaf.low) / leaf.step) + 1, steps)
  const low = leaf.log ? Math.log(leaf.low) : leaf.low
  const high = leaf.log ? Math.log(leaf.high) : leaf.high
  const width = (high - low) / (steps - 1)
  const counts = new Array<number>(steps).fill(0)
  for (const raw of values) {
    const v = leaf.log ? Math.log(raw) : raw
    const i = Math.min(steps - 1, Math.max(0, Math.ceil((v - low) / width - 0.5)))
    counts[i] = (counts[i] ?? 0) + 1
  }
  return counts
}

/** Optuna's Scott-rule Parzen estimator on the grid cells of a leaf (PED-ANOVA). */
function scottEstimator(leaf: Leaf, counts: readonly number[], priorWeight: number): ParzenEstimator {
  const observations = counts.flatMap((c, i) => (c > 0 ? [i] : []))
  const weights = observations.map((i) => counts[i] as number)
  if (leaf.kind === 'categorical')
    return parzenEstimator(
      [leaf],
      observations.map((i) => [i]),
      { priorWeight, predeterminedWeights: weights },
    )
  const size = counts.length
  const cell: Leaf = { ...leaf, kind: 'int', low: 0, high: size - 1, log: false, step: 1 }
  const n = observations.length
  const total = weights.reduce((a, b) => a + b, 0)
  const all = n === 0 ? [1] : [...weights, priorWeight]
  const sum = all.reduce((a, b) => a + b, 0)
  let sigma = 0
  if (n > 0) {
    const mean = observations.reduce((a, o, k) => a + o * (weights[k] as number), 0) / total
    sigma = Math.sqrt(
      observations.reduce((a, o, k) => a + (o - mean) ** 2 * (weights[k] as number), 0) / Math.max(1, total - 1),
    )
    const cum: number[] = []
    for (const w of weights) cum.push((cum.at(-1) ?? 0) + w)
    const q1 = cum.findIndex((c) => c >= Math.floor(total / 4))
    const q3target = Math.floor((total * 3) / 4)
    const q3raw = cum.findIndex((c) => c > q3target)
    const q3 = q3raw < 0 ? n : q3raw
    const iqr = (observations[Math.min(n - 1, q3)] as number) - (observations[q1 < 0 ? n - 1 : q1] as number)
    sigma = 1.059 * Math.min(iqr / 1.34, sigma) * total ** -0.2
  }
  const s = Math.max(sigma, 0.5 / 1.64)
  return {
    leaves: [cell],
    weights: Float64Array.from(all, (w) => w / sum),
    kernels: [
      {
        kind: 'numerical',
        mus: Float64Array.from([...observations, (size - 1) / 2]),
        sigmas: Float64Array.from([...observations.map(() => s), size]),
        low: -0.5,
        high: size - 0.5,
        step: 1,
        log: false,
      },
    ],
  }
}

function topTrials(study: Study, data: Data, quantile: number): Set<Trial> {
  if (quantile >= 1) return new Set(data.trials)
  if (study.directions.length > 1) {
    const n = Math.ceil(quantile * data.trials.length)
    return new Set(splitTrials(study, data.trials, n).below)
  }
  const losses = data.losses
  const cutoff = [...losses].sort((a, b) => a - b)[Math.ceil(quantile * losses.length) - 1] as number
  return new Set(data.trials.filter((_, i) => (losses[i] as number) <= cutoff))
}

/**
 * PED-ANOVA importance (Watanabe et al. 2023; Optuna's `PedAnovaImportanceEvaluator`): for each parameter, the Pearson
 * divergence between the Parzen density of its values among the top `targetQuantile` trials and among the top
 * `regionQuantile` (all by default), on a grid of 50 cells (Scott's bandwidth), weighted by the share of trials where
 * the parameter is active and by (target / region)^2. Deterministic, and closed form: no surrogate.
 *
 * @param study - The study.
 * @param options - Target quantile (default 0.1), region quantile (default 1), whether to compare with the region's
 *   density (default true) or the uniform one, and objective index.
 * @returns Importances by parameter (not normalised), largest first.
 * @concept hyperparameter-importance
 * @reference Watanabe, Bansal and Hutter (2023), PED-ANOVA: efficiently quantifying hyperparameter importance in
 *   arbitrary subspaces.
 * @example
 * ```ts
 * import { createStudy, pedAnovaImportance } from 'whitebox-engine/tune'
 * import { dict, real } from 'whitebox-engine/space'
 * const trials = Array.from({ length: 40 }, (_, i) => {
 *   const a = (i * 0.618) % 1
 *   const b = (i * 0.414) % 1
 *   return { number: i, params: { a, b }, state: 'complete', values: [a], intermediate: [], source: 'x' } as const
 * })
 * Object.keys(pedAnovaImportance(createStudy(dict({ a: real(0, 1), b: real(0, 1) }), { trials })))[0] // => 'a'
 * ```
 */
export function pedAnovaImportance(
  study: Study,
  options: {
    readonly targetQuantile?: number
    readonly regionQuantile?: number
    readonly evaluateOnLocal?: boolean
    readonly objective?: number
  } = {},
): Importances {
  const target = options.targetQuantile ?? 0.1
  const region = options.regionQuantile ?? 1
  if (!(target > 0 && target < region && region <= 1))
    throw new WhiteboxError('value', 'pedAnovaImportance: need 0 < targetQuantile < regionQuantile <= 1')
  const objective = options.objective ?? 0
  const data = dataOf(study, objective)
  if (data.trials.length <= 1) return sorted(data.leaves.map((l) => [l.path, 0]))
  const top = topTrials(study, data, target)
  const reg = topTrials(study, data, region)
  if (top.size === 0) return sorted(data.leaves.map((l) => [l.path, 0]))
  const quantile = top.size / reg.size
  const local = options.evaluateOnLocal ?? true
  const index = new Map(data.trials.map((t, i) => [t, i]))
  return sorted(
    data.leaves.map((leaf, j) => {
      const active = (set: Set<Trial>): Trial[] =>
        [...set].filter((t) => !Number.isNaN(data.natives[index.get(t) ?? 0]?.[j] ?? Number.NaN))
      const regionActive = active(reg)
      const targetActive = active(top)
      // Optuna drops regimes with fewer than 2 trials, and parameters with a single value.
      if (regionActive.length < 2 || targetActive.length === 0 || !(leaf.low < leaf.high))
        return [leaf.path, 0] as const
      const values = (ts: readonly Trial[]): number[] => ts.map((t) => data.natives[index.get(t) ?? 0]?.[j] ?? 0)
      const alpha = targetActive.length / top.size
      const beta = regionActive.length / reg.size
      const countsTop = gridCounts(leaf, values(targetActive), 50)
      const size = countsTop.length
      const grid = Array.from({ length: size }, (_, i) => [i])
      const pdf = (pe: ParzenEstimator): number[] => parzenLogPdf(pe, grid).map((v) => Math.exp(v) + 1e-12)
      const pTop = pdf(scottEstimator(leaf, countsTop, 1))
      const pLocal = local
        ? pdf(scottEstimator(leaf, gridCounts(leaf, values(regionActive), 50), 1))
        : grid.map(() => 1 / size)
      const divergence = pLocal.reduce((s, p, i) => s + p * ((pTop[i] as number) / p - 1) ** 2, 0)
      return [leaf.path, (alpha ** 2 / beta) * divergence * quantile ** 2] as const
    }),
  )
}
