/**
 * Global inspection of fitted models (sklearn.inspection and beyond): permutation, conditional permutation and
 * drop-column importance, partial dependence with ICE curves (centred and derivative), accumulated local effects of
 * first and second order, Friedman's H-statistic and the functional ANOVA decomposition.
 */

import { MdmlError, meanOf, quantileOf, varianceOf } from '../base/index.ts'
import { keyFor, runOptionsFor } from '../learn/_shared.ts'
import { type Estimator, type FitOptions, type Fitted, type LearnData, requireTargets } from '../learn/estimator.ts'
import { type Scoring, scorersOf } from '../learn/scoring.ts'
import {
  columnOf,
  fromView,
  type MatrixView,
  matrixView,
  sampleWeightsOf,
  takeColumns,
  takeRows,
} from '../learn/validation.ts'
import { foldIn, permutation, subsample } from '../random/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'

/** Options of {@link permutationImportance}. */
export interface PermutationImportanceOptions extends FitOptions {
  /** The scorer (default the estimator's `score`). */
  readonly scoring?: Scoring
  /** Repetitions per feature (default 5). */
  readonly nRepeats?: number
  /** Explicit row permutations, one per repetition (shared by all features); default drawn from the key. */
  readonly permutations?: readonly (readonly number[])[]
  /** Features to permute together, as groups of column indices (default each column alone). */
  readonly features?: readonly (readonly number[])[]
}

/** The result of {@link permutationImportance}. */
export interface PermutationImportanceResult {
  /** The score on the unpermuted data. */
  readonly baseline: number
  /** importances[feature][repeat]: the drop in score when the feature is permuted. */
  readonly importances: readonly (readonly number[])[]
  readonly mean: readonly number[]
  /** The population standard deviation over repetitions. */
  readonly std: readonly number[]
}

/**
 * Permutation feature importance (Breiman 2001; sklearn.inspection.permutation_importance): the drop in a model's
 * score when one feature's values are shuffled across rows, breaking its link to the target while keeping its
 * distribution. Model-agnostic; computed on held-out data it measures what the model relies on. A traced algorithm
 * with one step per feature.
 *
 * @param fitted - A fitted estimator.
 * @param data - Evaluation data (features and targets).
 * @param options - scoring, nRepeats, permutations, features (column groups), key and tracing.
 * @returns The baseline score and the importances.
 * @concept feature-importance
 * @example
 * ```ts
 * import { permutationImportance } from 'mdml-engine/explain'
 * import { tensor } from 'mdml-engine/tensor'
 * const firstColumn = { estimator: 'f', kind: 'regressor', params: {}, predict: (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[i * 2])) }
 * const x = [[1, 9], [2, 8], [3, 7], [4, 6]]
 * const r = permutationImportance(firstColumn, { x, y: [1, 2, 3, 4] }, { scoring: 'neg_mean_squared_error', permutations: [[3, 2, 1, 0]] })
 * r.mean // => [5, 0]
 * ```
 */
export function permutationImportance(
  fitted: Fitted,
  data: LearnData,
  options: PermutationImportanceOptions = {},
): PermutationImportanceResult {
  const m = matrixView(data.x, 'permutationImportance', { allowNaN: true })
  requireTargets(data, 'permutationImportance')
  const scorer = Object.values(scorersOf(options.scoring))[0]
  if (scorer === undefined) throw new MdmlError('value', 'permutationImportance: no scorer')
  const key = keyFor(options, 'permutationImportance')
  const nRepeats = options.permutations?.length ?? options.nRepeats ?? 5
  const perms =
    options.permutations ?? Array.from({ length: nRepeats }, (_, r) => toFlat(permutation(foldIn(key, r), m.n)))
  for (const p of perms) {
    if (p.length !== m.n)
      throw new MdmlError('shape', `permutationImportance: a permutation of ${p.length} rows for ${m.n}`)
  }
  const groups = options.features ?? Array.from({ length: m.d }, (_, j) => [j])
  const evalData = { ...data, x: fromView(m) }
  const baseline = scorer.score(fitted, evalData)
  const algorithm: Algorithm<readonly (readonly number[])[]> = {
    name: 'permutation-importance',
    config: { estimator: fitted.estimator, features: groups.length, nRepeats },
    schema: channels({
      metrics: { importance: { description: 'the mean drop in score of the feature just permuted' } },
    }),
    init: () => [],
    step: (done) => {
      const cols = groups[done.length] ?? []
      const drops = perms.map((perm) => {
        const shuffled = Float64Array.from(m.data)
        for (let i = 0; i < m.n; i++) {
          for (const j of cols) shuffled[i * m.d + j] = m.data[(perm[i] ?? i) * m.d + j] ?? 0
        }
        return baseline - scorer.score(fitted, { ...data, x: fromStorage(shuffled, [m.n, m.d], 'float64') })
      })
      return [...done, drops]
    },
    done: (s) => (s.length >= groups.length ? 'completed' : undefined),
    observe: (s) => ({ metrics: s.length === 0 ? {} : { importance: meanOf(s[s.length - 1] ?? []) } }),
  }
  const { state } = runAlgorithm(algorithm, {
    ...runOptionsFor(options, 'permutationImportance'),
    maxSteps: groups.length,
  })
  return {
    baseline,
    importances: state,
    mean: state.map((d) => meanOf(d)),
    std: state.map((d) => Math.sqrt(varianceOf(d, 0))),
  }
}

/** The response {@link partialDependence} averages. */
export type ResponseKind = 'auto' | 'predict' | 'predictProba' | 'decisionFunction'

function responseFn(
  fitted: Fitted,
  kind: ResponseKind,
): (x: Tensor) => { readonly values: Float64Array; readonly k: number } {
  const pick = (): ((x: TensorLike) => Tensor) => {
    if (kind === 'predict' || (kind === 'auto' && fitted.kind !== 'classifier')) {
      if (fitted.predict === undefined)
        throw new MdmlError('value', `partialDependence: ${fitted.estimator} cannot predict`)
      return fitted.predict
    }
    const f = kind === 'decisionFunction' ? fitted.decisionFunction : (fitted.predictProba ?? fitted.decisionFunction)
    if (f === undefined)
      throw new MdmlError(
        'value',
        `partialDependence: ${fitted.estimator} has no ${kind === 'auto' ? 'predictProba' : kind}`,
      )
    return f
  }
  const f = pick()
  return (x) => {
    const out = f(x)
    let k = out.shape.length === 2 ? (out.shape[1] ?? 1) : 1
    let values = Float64Array.from(toFlat(out))
    // Binary classifiers report the positive class only (sklearn).
    if (k === 2 && fitted.kind === 'classifier') {
      values = Float64Array.from({ length: values.length / 2 }, (_, i) => values[2 * i + 1] ?? 0)
      k = 1
    }
    return { values, k }
  }
}

/** scipy.stats.mstats.mquantiles with its default plotting positions (alphap = betap = 0.4), as sklearn's PD grid. */
function mquantile(sorted: Float64Array, p: number): number {
  const n = sorted.length
  const m = 0.4 + p * (1 - 0.8)
  const aleph = n * p + m
  const k = Math.floor(Math.min(Math.max(aleph, 1), n - 1))
  const gamma = Math.min(Math.max(aleph - k, 0), 1)
  return (1 - gamma) * (sorted[k - 1] ?? 0) + gamma * (sorted[k] ?? 0)
}

/** Options of {@link partialDependence}. */
export interface PartialDependenceOptions {
  /** Grid points per feature (default 100); features with fewer distinct values use them all. */
  readonly gridResolution?: number
  /** The grid's range as quantiles (default [0.05, 0.95]). */
  readonly percentiles?: readonly [number, number]
  /** Fixed grid values per feature, instead of the automatic grid. */
  readonly values?: readonly (readonly number[])[]
  /** Use the distinct values of these features (by position in `features`) as their grid. */
  readonly categorical?: readonly boolean[]
  /** The response (default `'auto'`: probabilities for classifiers, predictions otherwise). */
  readonly response?: ResponseKind
  /** Sample weights for the average. */
  readonly weights?: TensorLike
}

/** The result of {@link partialDependence}. */
export interface PartialDependenceResult {
  /** The grid values of each feature. */
  readonly values: readonly (readonly number[])[]
  /** average[output][gridIndex] over the Cartesian grid (row-major: the last feature fastest). */
  readonly average: readonly (readonly number[])[]
  /** individual[output][sample][gridIndex]: the ICE curves. */
  readonly individual: readonly (readonly (readonly number[])[])[]
}

/**
 * Partial dependence and individual conditional expectation (Friedman 2001; Goldstein et al. 2015;
 * sklearn.inspection.partial_dependence with method 'brute'): set the chosen features to each grid value in every row,
 * predict, and average over the rows (ICE keeps each row's curve). The grid spans the 5th to 95th percentiles (scipy's
 * mquantiles) or a feature's distinct values when it has few.
 *
 * @param fitted - A fitted estimator.
 * @param x - The background data `[n, d]`.
 * @param features - One or two (or more) column indices.
 * @param options - gridResolution, percentiles, values, categorical, response and weights.
 * @returns The grid, the averaged curves and the individual curves.
 * @concept partial-dependence
 * @example
 * ```ts
 * import { partialDependence } from 'mdml-engine/explain'
 * import { tensor } from 'mdml-engine/tensor'
 * const sumModel = { estimator: 's', kind: 'regressor', params: {}, predict: (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[2 * i] + x.data[2 * i + 1])) }
 * const r = partialDependence(sumModel, [[0, 1], [1, 3]], [0], { values: [[0, 10]] })
 * r.average // => [[2, 12]]
 * ```
 */
export function partialDependence(
  fitted: Fitted,
  x: TensorLike,
  features: readonly number[],
  options: PartialDependenceOptions = {},
): PartialDependenceResult {
  const m = matrixView(x, 'partialDependence', { allowNaN: true })
  if (features.length === 0) throw new MdmlError('value', 'partialDependence: no features')
  for (const f of features) {
    if (!(Number.isInteger(f) && f >= 0 && f < m.d))
      throw new MdmlError('value', `partialDependence: feature ${f} out of range`)
  }
  const resolution = options.gridResolution ?? 100
  const [pLo, pHi] = options.percentiles ?? [0.05, 0.95]
  const values = features.map((f, a) => {
    const given = options.values?.[a]
    if (given !== undefined) return [...given]
    const col = columnOf(m, f)
    const uniq = Array.from(new Set(col)).sort((u, v) => u - v)
    if (options.categorical?.[a] === true || uniq.length < resolution) return uniq
    const sorted = Float64Array.from(col).sort()
    const lo = mquantile(sorted, pLo)
    const hi = mquantile(sorted, pHi)
    if (Math.abs(hi - lo) <= 1e-8 + 1e-5 * Math.abs(lo)) {
      throw new MdmlError('value', `partialDependence: percentiles too close for feature ${f}`)
    }
    return Array.from({ length: resolution }, (_, i) => lo + ((hi - lo) * i) / (resolution - 1))
  })
  const gridSize = values.reduce((p, v) => p * v.length, 1)
  const response = responseFn(fitted, options.response ?? 'auto')
  const w = options.weights === undefined ? undefined : sampleWeightsOf(options.weights, m.n, 'partialDependence')
  let nOut = 1
  const individual: number[][][] = []
  for (let g = 0; g < gridSize; g++) {
    let r = g
    const point = new Array<number>(features.length)
    for (let a = features.length - 1; a >= 0; a--) {
      const len = values[a]?.length ?? 1
      point[a] = values[a]?.[r % len] ?? 0
      r = Math.floor(r / len)
    }
    const evalX = Float64Array.from(m.data)
    for (let i = 0; i < m.n; i++)
      features.forEach((f, a) => {
        evalX[i * m.d + f] = point[a] ?? 0
      })
    const out = response(fromStorage(evalX, [m.n, m.d], 'float64'))
    nOut = out.k
    for (let o = 0; o < out.k; o++) {
      const rows = individual[o] ?? Array.from({ length: m.n }, () => [] as number[])
      individual[o] = rows
      for (let i = 0; i < m.n; i++) rows[i]?.push(out.values[i * out.k + o] ?? 0)
    }
  }
  const average = Array.from({ length: nOut }, (_, o) =>
    Array.from({ length: gridSize }, (_, g) => {
      let s = 0
      let sw = 0
      for (let i = 0; i < m.n; i++) {
        const wi = w?.[i] ?? 1
        s += wi * (individual[o]?.[i]?.[g] ?? 0)
        sw += wi
      }
      return s / sw
    }),
  )
  return { values, average, individual }
}

/** Options of {@link accumulatedLocalEffects}. */
export interface AleOptions {
  /** Quantile bins (default 10); repeated quantiles merge. */
  readonly bins?: number
  readonly response?: ResponseKind
}

/** The result of {@link accumulatedLocalEffects}. */
export interface AleResult {
  /** The bin edges (quantiles of the feature). */
  readonly edges: readonly number[]
  /** The centred ALE at each edge, per output: effects[output][edge]. */
  readonly effects: readonly (readonly number[])[]
  /** Samples per bin. */
  readonly counts: readonly number[]
}

/**
 * First-order accumulated local effects (Apley and Zhu 2020): split a feature at its quantiles, average within each
 * bin the change in prediction from the bin's lower to upper edge (other features held at each row's values),
 * accumulate, and centre (as R's ALEPlot). Unlike partial dependence, it never evaluates the model far from the data,
 * so it stays faithful with correlated features.
 *
 * @param fitted - A fitted estimator.
 * @param x - The data `[n, d]`.
 * @param feature - The column.
 * @param options - bins (default 10) and response.
 * @returns The edges, centred effects and bin counts.
 * @example
 * ```ts
 * import { accumulatedLocalEffects } from 'mdml-engine/explain'
 * import { tensor } from 'mdml-engine/tensor'
 * const twice = { estimator: 't', kind: 'regressor', params: {}, predict: (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => 2 * x.data[i])) }
 * const r = accumulatedLocalEffects(twice, [[0], [1], [2], [3], [4]], 0, { bins: 2 })
 * // Steps of 4 per bin, centred on the data: -3.6, 0.4, 4.4.
 * r.effects[0].map((v) => Math.round(v * 10) / 10) // => [-3.6, 0.4, 4.4]
 * ```
 */
export function accumulatedLocalEffects(
  fitted: Fitted,
  x: TensorLike,
  feature: number,
  options: AleOptions = {},
): AleResult {
  const m = matrixView(x, 'accumulatedLocalEffects')
  if (!(Number.isInteger(feature) && feature >= 0 && feature < m.d))
    throw new MdmlError('value', 'accumulatedLocalEffects: feature out of range')
  const K = options.bins ?? 10
  const col = columnOf(m, feature)
  const sorted = Float64Array.from(col).sort()
  const q = (p: number): number => {
    const h = (sorted.length - 1) * p
    const lo = Math.floor(h)
    return (sorted[lo] ?? 0) + (h - lo) * ((sorted[Math.min(lo + 1, sorted.length - 1)] ?? 0) - (sorted[lo] ?? 0))
  }
  const edges = Array.from(new Set(Array.from({ length: K + 1 }, (_, k) => q(k / K)))).sort((a, b) => a - b)
  if (edges.length < 2) throw new MdmlError('value', 'accumulatedLocalEffects: the feature is constant')
  const nb = edges.length - 1
  const binOf = (v: number): number => {
    let b = 0
    while (b < nb - 1 && v > (edges[b + 1] ?? 0)) b++
    return b
  }
  const bins = Array.from(col, binOf)
  const counts = new Array<number>(nb).fill(0)
  for (const b of bins) counts[b] = (counts[b] ?? 0) + 1
  const response = responseFn(fitted, options.response ?? 'auto')
  const at = (side: 0 | 1): { readonly values: Float64Array; readonly k: number } => {
    const evalX = Float64Array.from(m.data)
    for (let i = 0; i < m.n; i++) evalX[i * m.d + feature] = edges[(bins[i] ?? 0) + side] ?? 0
    return response(fromStorage(evalX, [m.n, m.d], 'float64'))
  }
  const lo = at(0)
  const hi = at(1)
  const k = lo.k
  const effects = Array.from({ length: k }, (_, o) => {
    const delta = new Array<number>(nb).fill(0)
    for (let i = 0; i < m.n; i++) {
      const b = bins[i] ?? 0
      delta[b] = (delta[b] ?? 0) + ((hi.values[i * k + o] ?? 0) - (lo.values[i * k + o] ?? 0)) / (counts[b] ?? 1)
    }
    const acc = [0]
    for (let b = 0; b < nb; b++) acc.push((acc[b] ?? 0) + (delta[b] ?? 0))
    let centre = 0
    for (let b = 0; b < nb; b++) centre += (((acc[b] ?? 0) + (acc[b + 1] ?? 0)) / 2) * (counts[b] ?? 0)
    centre /= m.n
    return acc.map((v) => v - centre)
  })
  return { edges, effects, counts }
}

/** Mean predictions with the features in `cols` set to row i's values, averaged over every row of the background. */
function pdAtRows(
  response: (x: Tensor) => { readonly values: Float64Array; readonly k: number },
  m: MatrixView,
  cols: readonly number[],
): Float64Array {
  const out = new Float64Array(m.n)
  for (let i = 0; i < m.n; i++) {
    const evalX = Float64Array.from(m.data)
    for (let r = 0; r < m.n; r++) for (const c of cols) evalX[r * m.d + c] = m.data[i * m.d + c] ?? 0
    const p = response(fromStorage(evalX, [m.n, m.d], 'float64'))
    let s = 0
    for (let r = 0; r < m.n; r++) s += p.values[r * p.k] ?? 0
    out[i] = s / m.n
  }
  const mu = meanOf(out)
  return out.map((v) => v - mu)
}

/** Options of {@link hStatistic}. */
export interface HStatisticOptions {
  /** Use at most this many rows (a random subsample; the cost is quadratic in rows). */
  readonly maxSamples?: number
  readonly response?: ResponseKind
  readonly key?: FitOptions['key']
}

/**
 * Friedman and Popescu's (2008) H-statistic of interaction strength: for a pair of features, the share of the joint
 * partial dependence's variance not explained by the two one-way partial dependences; for one feature (the other
 * argument omitted), the share of the prediction's variance not explained by its own and its complement's partial
 * dependences. 0 means no interaction; partial dependences are evaluated at the data points (centred).
 *
 * @param fitted - A fitted estimator.
 * @param x - The data `[n, d]`.
 * @param j - A feature.
 * @param k - A second feature (omit for j's total interaction).
 * @param options - maxSamples (default 200), response and key.
 * @returns H squared, in [0, 1] up to estimation noise.
 * @example
 * ```ts
 * import { hStatistic } from 'mdml-engine/explain'
 * import { tensor } from 'mdml-engine/tensor'
 * const additive = { estimator: 'a', kind: 'regressor', params: {}, predict: (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[2 * i] + x.data[2 * i + 1])) }
 * hStatistic(additive, [[0, 1], [1, 0], [2, 2], [3, 1]], 0, 1) < 1e-20 // => true
 * ```
 */
export function hStatistic(
  fitted: Fitted,
  x: TensorLike,
  j: number,
  k?: number,
  options: HStatisticOptions = {},
): number {
  let m = matrixView(x, 'hStatistic')
  const max = options.maxSamples ?? 200
  if (m.n > max) m = takeRows(m, toFlat(subsample(options.key ?? keyFor({}, 'hStatistic'), m.n, max)))
  const response = responseFn(fitted, options.response ?? 'auto')
  const pdJ = pdAtRows(response, m, [j])
  if (k !== undefined) {
    const pdK = pdAtRows(response, m, [k])
    const pdJK = pdAtRows(response, m, [j, k])
    let num = 0
    let den = 0
    for (let i = 0; i < m.n; i++) {
      num += ((pdJK[i] ?? 0) - (pdJ[i] ?? 0) - (pdK[i] ?? 0)) ** 2
      den += (pdJK[i] ?? 0) ** 2
    }
    return den === 0 ? 0 : num / den
  }
  const others = Array.from({ length: m.d }, (_, c) => c).filter((c) => c !== j)
  const pdRest = pdAtRows(response, m, others)
  const f = response(fromView(m))
  const fm = meanOf(Array.from({ length: m.n }, (_, i) => f.values[i * f.k] ?? 0))
  let num = 0
  let den = 0
  for (let i = 0; i < m.n; i++) {
    const fi = (f.values[i * f.k] ?? 0) - fm
    num += (fi - (pdJ[i] ?? 0) - (pdRest[i] ?? 0)) ** 2
    den += fi * fi
  }
  return den === 0 ? 0 : num / den
}

/** A component of a functional ANOVA decomposition. */
export interface AnovaComponent {
  /** The features of the component. */
  readonly features: readonly number[]
  /** Its values at the background rows. */
  readonly values: readonly number[]
  /** Its variance, and its share of the prediction's variance. */
  readonly variance: number
  readonly share: number
}

/**
 * The functional ANOVA decomposition of a model (Hoeffding 1948; Hooker 2004) up to second order on background data:
 * f = f0 + sum_j f_j(x_j) + sum_{j<k} f_jk(x_j, x_k) + ..., each component a partial dependence minus its lower-order
 * components, with variance shares (Sobol-type indices). Exact for a background that is a product grid (independent
 * features); otherwise the partial-dependence approximation.
 *
 * @param fitted - A fitted estimator.
 * @param x - The background `[n, d]` (cost is quadratic in rows).
 * @param options - maxOrder (1 or 2, default 2), features (default all) and response.
 * @returns The constant, the components and the share of variance left to higher orders.
 * @example
 * ```ts
 * import { functionalAnova } from 'mdml-engine/explain'
 * import { tensor } from 'mdml-engine/tensor'
 * const grid = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
 * const f = { estimator: 'f', kind: 'regressor', params: {}, predict: (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[2 * i] + x.data[2 * i] * x.data[2 * i + 1])) }
 * functionalAnova(f, grid).components.map((c) => c.share) // => [0.5, 0, 0.5]
 * ```
 */
export function functionalAnova(
  fitted: Fitted,
  x: TensorLike,
  options: { readonly maxOrder?: 1 | 2; readonly features?: readonly number[]; readonly response?: ResponseKind } = {},
): { readonly constant: number; readonly components: readonly AnovaComponent[]; readonly residualShare: number } {
  const m = matrixView(x, 'functionalAnova')
  const response = responseFn(fitted, options.response ?? 'auto')
  const feats = options.features ?? Array.from({ length: m.d }, (_, j) => j)
  const f = response(fromView(m))
  const fv = Array.from({ length: m.n }, (_, i) => f.values[i * f.k] ?? 0)
  const constant = meanOf(fv)
  const total = varianceOf(fv, 0)
  const main = new Map(feats.map((j) => [j, Array.from(pdAtRows(response, m, [j]))]))
  const components: AnovaComponent[] = feats.map((j) => {
    const v = main.get(j) ?? []
    const variance = varianceOf(v, 0)
    return { features: [j], values: v, variance, share: total === 0 ? 0 : variance / total }
  })
  if ((options.maxOrder ?? 2) >= 2) {
    for (let a = 0; a < feats.length; a++) {
      for (let b = a + 1; b < feats.length; b++) {
        const j = feats[a] ?? 0
        const k = feats[b] ?? 0
        const joint = pdAtRows(response, m, [j, k])
        const v = Array.from(joint, (t, i) => t - (main.get(j)?.[i] ?? 0) - (main.get(k)?.[i] ?? 0))
        const variance = varianceOf(v, 0)
        components.push({ features: [j, k], values: v, variance, share: total === 0 ? 0 : variance / total })
      }
    }
  }
  const explained = components.reduce((s, c) => s + c.share, 0)
  return { constant, components, residualShare: total === 0 ? 0 : 1 - explained }
}

/**
 * Centred ICE curves (Goldstein et al. 2015's c-ICE; sklearn's `centered=True`): each curve minus its value at one grid
 * point (default the first), so curves that differ only by an offset coincide and the spread left is heterogeneity of
 * the effect (an interaction).
 *
 * @param individual - ICE curves, individual[row][gridIndex].
 * @param at - The grid index anchored at zero (default 0).
 * @returns The centred curves.
 * @example
 * ```ts
 * import { centredIce } from 'mdml-engine/explain'
 * centredIce([[1, 2, 4], [5, 6, 8]]) // => [[0, 1, 3], [0, 1, 3]]
 * ```
 */
export function centredIce(individual: readonly (readonly number[])[], at = 0): number[][] {
  return individual.map((curve) => {
    if (!(Number.isInteger(at) && at >= 0 && at < curve.length))
      throw new MdmlError('value', `centredIce: grid index ${at} out of range`)
    const c = curve[at] ?? 0
    return curve.map((v) => v - c)
  })
}

/**
 * Derivative ICE curves (Goldstein et al. 2015's d-ICE): each curve's slope along the grid by second-order central
 * differences inside and one-sided differences at the ends (numpy.gradient on an uneven grid), so a slope that varies
 * across rows at the same grid value reveals an interaction.
 *
 * @param grid - The grid values (increasing, at least two).
 * @param individual - ICE curves, individual[row][gridIndex].
 * @returns The derivative curves.
 * @example
 * ```ts
 * import { iceDerivative } from 'mdml-engine/explain'
 * iceDerivative([0, 1, 2], [[0, 1, 4]]) // => [[1, 2, 3]]
 * ```
 */
export function iceDerivative(grid: readonly number[], individual: readonly (readonly number[])[]): number[][] {
  const g = grid.length
  if (g < 2) throw new MdmlError('value', 'iceDerivative: needs at least two grid points')
  return individual.map((curve) => {
    if (curve.length !== g) throw new MdmlError('shape', 'iceDerivative: a curve does not match the grid')
    return curve.map((_, i) => {
      if (i === 0) return ((curve[1] ?? 0) - (curve[0] ?? 0)) / ((grid[1] ?? 0) - (grid[0] ?? 0))
      if (i === g - 1) return ((curve[i] ?? 0) - (curve[i - 1] ?? 0)) / ((grid[i] ?? 0) - (grid[i - 1] ?? 0))
      const h1 = (grid[i] ?? 0) - (grid[i - 1] ?? 0)
      const h2 = (grid[i + 1] ?? 0) - (grid[i] ?? 0)
      return (
        (h1 * h1 * (curve[i + 1] ?? 0) - h2 * h2 * (curve[i - 1] ?? 0) + (h2 * h2 - h1 * h1) * (curve[i] ?? 0)) /
        (h1 * h2 * (h1 + h2))
      )
    })
  })
}

/** Quantile edges of a column (numpy's linear quantiles, repeats merged), and each row's bin. */
function quantileBins(col: Float64Array, bins: number, who: string): { edges: number[]; bin: number[] } {
  const sorted = Float64Array.from(col).sort()
  const q = (p: number): number => {
    const h = (sorted.length - 1) * p
    const lo = Math.floor(h)
    return (sorted[lo] ?? 0) + (h - lo) * ((sorted[Math.min(lo + 1, sorted.length - 1)] ?? 0) - (sorted[lo] ?? 0))
  }
  const edges = Array.from(new Set(Array.from({ length: bins + 1 }, (_, k) => q(k / bins)))).sort((a, b) => a - b)
  if (edges.length < 2) throw new MdmlError('value', `${who}: the feature is constant`)
  const nb = edges.length - 1
  const bin = Array.from(col, (v) => {
    let b = 0
    while (b < nb - 1 && v > (edges[b + 1] ?? 0)) b++
    return b
  })
  return { edges, bin }
}

/** The result of {@link accumulatedLocalEffects2D}. */
export interface Ale2DResult {
  /** The bin edges of the two features. */
  readonly edges: readonly [readonly number[], readonly number[]]
  /** The pure second-order effect at each pair of edges, effects[i][j] (first feature's edge i, second's j). */
  readonly effects: readonly (readonly number[])[]
  /** Rows per cell, counts[i][j]. */
  readonly counts: readonly (readonly number[])[]
}

/**
 * Second-order accumulated local effects (Apley and Zhu 2020, as R's ALEPlot): over the grid of both features'
 * quantile bins, average in each cell the second difference f(hi, hi) - f(lo, hi) - f(hi, lo) + f(lo, lo) of the rows
 * in it (empty cells take the value of the nearest non-empty cell, by bin index), accumulate over both axes, then
 * remove each feature's own first-order effect and centre, leaving the pure interaction of the pair.
 *
 * @param fitted - A fitted estimator.
 * @param x - The data `[n, d]`.
 * @param features - The two columns.
 * @param options - bins per feature (default 10) and response.
 * @returns The edges, the interaction surface and the cell counts.
 * @example
 * ```ts
 * import { accumulatedLocalEffects2D } from 'mdml-engine/explain'
 * import { tensor } from 'mdml-engine/tensor'
 * const additive = { estimator: 'a', kind: 'regressor', params: {}, predict: (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[2 * i] + 3 * x.data[2 * i + 1])) }
 * const x = [[0, 0], [1, 2], [2, 1], [3, 3], [1, 0], [2, 3]]
 * // No interaction: the surface is flat.
 * accumulatedLocalEffects2D(additive, x, [0, 1], { bins: 2 }).effects.flat().every((v) => Math.abs(v) < 1e-12) // => true
 * ```
 */
export function accumulatedLocalEffects2D(
  fitted: Fitted,
  x: TensorLike,
  features: readonly [number, number],
  options: AleOptions = {},
): Ale2DResult {
  const m = matrixView(x, 'accumulatedLocalEffects2D')
  const [j, k] = features
  for (const f of features)
    if (!(Number.isInteger(f) && f >= 0 && f < m.d))
      throw new MdmlError('value', 'accumulatedLocalEffects2D: feature out of range')
  if (j === k) throw new MdmlError('value', 'accumulatedLocalEffects2D: the two features must differ')
  const K = options.bins ?? 10
  const a = quantileBins(columnOf(m, j), K, 'accumulatedLocalEffects2D')
  const b = quantileBins(columnOf(m, k), K, 'accumulatedLocalEffects2D')
  const K1 = a.edges.length - 1
  const K2 = b.edges.length - 1
  const response = responseFn(fitted, options.response ?? 'auto')
  const at = (s1: 0 | 1, s2: 0 | 1): Float64Array => {
    const evalX = Float64Array.from(m.data)
    for (let i = 0; i < m.n; i++) {
      evalX[i * m.d + j] = a.edges[(a.bin[i] ?? 0) + s1] ?? 0
      evalX[i * m.d + k] = b.edges[(b.bin[i] ?? 0) + s2] ?? 0
    }
    const r = response(fromStorage(evalX, [m.n, m.d], 'float64'))
    return Float64Array.from({ length: m.n }, (_, i) => r.values[i * r.k] ?? 0)
  }
  const f11 = at(1, 1)
  const f01 = at(0, 1)
  const f10 = at(1, 0)
  const f00 = at(0, 0)
  const counts = Array.from({ length: K1 }, () => new Array<number>(K2).fill(0))
  const delta = Array.from({ length: K1 }, () => new Array<number>(K2).fill(0))
  for (let i = 0; i < m.n; i++) {
    const p = a.bin[i] ?? 0
    const q = b.bin[i] ?? 0
    ;(counts[p] as number[])[q] = (counts[p]?.[q] ?? 0) + 1
    ;(delta[p] as number[])[q] = (delta[p]?.[q] ?? 0) + (f11[i] ?? 0) - (f01[i] ?? 0) - (f10[i] ?? 0) + (f00[i] ?? 0)
  }
  const filled: [number, number][] = []
  for (let p = 0; p < K1; p++)
    for (let q = 0; q < K2; q++) {
      const c = counts[p]?.[q] ?? 0
      if (c > 0) {
        ;(delta[p] as number[])[q] = (delta[p]?.[q] ?? 0) / c
        filled.push([p, q])
      }
    }
  // Empty cells copy their nearest non-empty cell (distance in bin indices scaled to [0, 1] per axis).
  for (let p = 0; p < K1; p++)
    for (let q = 0; q < K2; q++) {
      if ((counts[p]?.[q] ?? 0) > 0) continue
      let best = Number.POSITIVE_INFINITY
      let value = 0
      for (const [u, v] of filled) {
        const dist = ((u - p) / Math.max(1, K1)) ** 2 + ((v - q) / Math.max(1, K2)) ** 2
        if (dist < best) {
          best = dist
          value = delta[u]?.[v] ?? 0
        }
      }
      ;(delta[p] as number[])[q] = value
    }
  // Accumulate over both axes onto the (K1 + 1) x (K2 + 1) edge grid.
  const fJ = Array.from({ length: K1 + 1 }, () => new Array<number>(K2 + 1).fill(0))
  for (let p = 1; p <= K1; p++)
    for (let q = 1; q <= K2; q++)
      (fJ[p] as number[])[q] =
        (delta[p - 1]?.[q - 1] ?? 0) + (fJ[p - 1]?.[q] ?? 0) + (fJ[p]?.[q - 1] ?? 0) - (fJ[p - 1]?.[q - 1] ?? 0)
  const rowCounts = counts.map((r) => r.reduce((s, v) => s + v, 0))
  const colCounts = Array.from({ length: K2 }, (_, q) => counts.reduce((s, r) => s + (r[q] ?? 0), 0))
  // Remove the first-order effects the accumulated surface still holds (ALEPlot's fJ1 and fJ2).
  const fJ1 = [0]
  for (let p = 0; p < K1; p++) {
    let s = 0
    for (let q = 0; q < K2; q++) {
      const d0 = (fJ[p + 1]?.[q] ?? 0) - (fJ[p]?.[q] ?? 0)
      const d1 = (fJ[p + 1]?.[q + 1] ?? 0) - (fJ[p]?.[q + 1] ?? 0)
      s += ((counts[p]?.[q] ?? 0) * (d0 + d1)) / 2
    }
    fJ1.push((fJ1[p] ?? 0) + ((rowCounts[p] ?? 0) > 0 ? s / (rowCounts[p] ?? 1) : 0))
  }
  const fJ2 = [0]
  for (let q = 0; q < K2; q++) {
    let s = 0
    for (let p = 0; p < K1; p++) {
      const d0 = (fJ[p]?.[q + 1] ?? 0) - (fJ[p]?.[q] ?? 0)
      const d1 = (fJ[p + 1]?.[q + 1] ?? 0) - (fJ[p + 1]?.[q] ?? 0)
      s += ((counts[p]?.[q] ?? 0) * (d0 + d1)) / 2
    }
    fJ2.push((fJ2[q] ?? 0) + ((colCounts[q] ?? 0) > 0 ? s / (colCounts[q] ?? 1) : 0))
  }
  const pure = fJ.map((row, p) => row.map((v, q) => v - (fJ1[p] ?? 0) - (fJ2[q] ?? 0)))
  let centre = 0
  for (let p = 0; p < K1; p++)
    for (let q = 0; q < K2; q++)
      centre +=
        ((counts[p]?.[q] ?? 0) *
          ((pure[p]?.[q] ?? 0) + (pure[p]?.[q + 1] ?? 0) + (pure[p + 1]?.[q] ?? 0) + (pure[p + 1]?.[q + 1] ?? 0))) /
        4
  centre /= m.n
  return { edges: [a.edges, b.edges], effects: pure.map((row) => row.map((v) => v - centre)), counts }
}

/** Options of {@link dropColumnImportance}. */
export interface DropColumnOptions extends FitOptions {
  /** The scorer (default the estimator's `score`). */
  readonly scoring?: Scoring
  /** Data to score on (default the training data). */
  readonly test?: LearnData
  /** Features to drop together, as groups of column indices (default each column alone). */
  readonly features?: readonly (readonly number[])[]
}

/** The result of {@link dropColumnImportance}. */
export interface DropColumnResult {
  /** The score of the model fitted on every column. */
  readonly baseline: number
  /** The score of each refit without the feature (group). */
  readonly scores: readonly number[]
  /** baseline - score: how much the model loses without the feature. */
  readonly importances: readonly number[]
}

/**
 * Drop-column importance (Parr et al. 2018): refit the estimator without each feature (or group) and measure the drop
 * in score against the model fitted on every column. Unlike permutation importance it asks what the data hold that no
 * other feature replaces, so correlated features share little importance; it costs one fit per feature.
 *
 * @param estimator - An unfitted estimator.
 * @param train - The training data (features and targets).
 * @param options - scoring, test data, features (column groups) and fit options.
 * @returns The baseline score, the score of each refit and the importances.
 * @example
 * ```ts
 * import { dropColumnImportance } from 'mdml-engine/explain'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { tensor } from 'mdml-engine/tensor'
 * // Predicts the first column (0 when it is dropped).
 * const firstOrZero = defineEstimator({ name: 'first', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: (x) => tensor(x.map((row) => row[0] ?? 0)) }) }, {})
 * dropColumnImportance(firstOrZero, { x: [[1], [2]], y: [1, 2] }, { scoring: 'neg_mean_squared_error' }).importances // => [2.5]
 * ```
 */
export function dropColumnImportance(
  estimator: Estimator,
  train: LearnData,
  options: DropColumnOptions = {},
): DropColumnResult {
  const m = matrixView(train.x, 'dropColumnImportance', { allowNaN: true })
  requireTargets(train, 'dropColumnImportance')
  const scorer = Object.values(scorersOf(options.scoring))[0]
  if (scorer === undefined) throw new MdmlError('value', 'dropColumnImportance: no scorer')
  const test = options.test ?? train
  const t = matrixView(test.x, 'dropColumnImportance', { allowNaN: true, nFeatures: m.d })
  const fitOptions: FitOptions = { ...options, explain: false }
  const baseline = scorer.score(estimator.fit({ ...train, x: rowsMatrix(m) }, fitOptions), {
    ...test,
    x: rowsMatrix(t),
  })
  const groups = options.features ?? Array.from({ length: m.d }, (_, j) => [j])
  const scores = groups.map((g) => {
    const keep = Array.from({ length: m.d }, (_, c) => c).filter((c) => !g.includes(c))
    const fitted = estimator.fit({ ...train, x: rowsMatrix(takeColumns(m, keep)) }, fitOptions)
    return scorer.score(fitted, { ...test, x: rowsMatrix(takeColumns(t, keep)) })
  })
  return { baseline, scores, importances: scores.map((s) => baseline - s) }
}

// A view as nested rows (so a model of no columns still sees one empty row per sample).
function rowsMatrix(m: MatrixView): number[][] {
  return Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
}

/** Options of {@link conditionalPermutationImportance}. */
export interface ConditionalPermutationOptions extends FitOptions {
  readonly scoring?: Scoring
  /** Repetitions per feature (default 5). */
  readonly nRepeats?: number
  /** The features each feature is conditioned on, conditionOn[j] (default those correlated with it, see `threshold`). */
  readonly conditionOn?: readonly (readonly number[])[]
  /** The |Pearson correlation| above which a feature is conditioned on (default 0.3; at most two, the strongest). */
  readonly threshold?: number
  /** Quantile bins per conditioning feature (default 4). */
  readonly bins?: number
}

/** The result of {@link conditionalPermutationImportance}. */
export interface ConditionalPermutationResult extends PermutationImportanceResult {
  /** The features each feature was conditioned on. */
  readonly conditionOn: readonly (readonly number[])[]
}

function pearsonOf(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const ma = meanOf(a)
  const mb = meanOf(b)
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < a.length; i++) {
    const u = (a[i] ?? 0) - ma
    const v = (b[i] ?? 0) - mb
    sab += u * v
    saa += u * u
    sbb += v * v
  }
  return saa === 0 || sbb === 0 ? 0 : sab / Math.sqrt(saa * sbb)
}

/**
 * Conditional permutation importance (Strobl et al. 2008): permute a feature only within strata of the features it is
 * correlated with (quantile bins of each), so the permuted values stay plausible given the rest of the row and a
 * feature that only duplicates another's information scores near zero. Without conditioning features it is the
 * marginal permutation importance.
 *
 * @param fitted - A fitted estimator.
 * @param data - Evaluation data (features and targets).
 * @param options - scoring, nRepeats, conditionOn, threshold, bins and key.
 * @returns The baseline, the importances per repeat, their mean and spread, and the conditioning features.
 * @example
 * ```ts
 * import { conditionalPermutationImportance } from 'mdml-engine/explain'
 * import { tensor } from 'mdml-engine/tensor'
 * const firstColumn = { estimator: 'f', kind: 'regressor', params: {}, predict: (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[i * 2])) }
 * // The second column copies the first: conditioned on it, permuting the first changes nothing.
 * const x = [[1, 1], [2, 2], [3, 3], [4, 4]]
 * conditionalPermutationImportance(firstColumn, { x, y: [1, 2, 3, 4] }, { scoring: 'neg_mean_squared_error', bins: 4 }).mean // => [0, 0]
 * ```
 */
export function conditionalPermutationImportance(
  fitted: Fitted,
  data: LearnData,
  options: ConditionalPermutationOptions = {},
): ConditionalPermutationResult {
  const m = matrixView(data.x, 'conditionalPermutationImportance')
  requireTargets(data, 'conditionalPermutationImportance')
  const scorer = Object.values(scorersOf(options.scoring))[0]
  if (scorer === undefined) throw new MdmlError('value', 'conditionalPermutationImportance: no scorer')
  const key = keyFor(options, 'conditionalPermutationImportance')
  const nRepeats = options.nRepeats ?? 5
  const bins = options.bins ?? 4
  const threshold = options.threshold ?? 0.3
  const cols = Array.from({ length: m.d }, (_, j) => columnOf(m, j))
  const conditionOn =
    options.conditionOn ??
    cols.map((cj, j) =>
      cols
        .map((ck, k) => ({ k, r: k === j ? 0 : Math.abs(pearsonOf(cj, ck)) }))
        .filter((e) => e.r > threshold)
        .sort((u, v) => v.r - u.r || u.k - v.k)
        .slice(0, 2)
        .map((e) => e.k),
    )
  const binsOf = cols.map((c) => {
    const sorted = Float64Array.from(c).sort()
    const edges = Array.from({ length: bins - 1 }, (_, b) => quantileOf(sorted, (b + 1) / bins))
    return Array.from(c, (v) => edges.filter((e) => v > e).length)
  })
  const evalData = { ...data, x: fromView(m) }
  const baseline = scorer.score(fitted, evalData)
  const importances = cols.map((_, j) => {
    const z = conditionOn[j] ?? []
    const strata = new Map<string, number[]>()
    for (let i = 0; i < m.n; i++) {
      const s = z.map((k) => binsOf[k]?.[i] ?? 0).join(',')
      const list = strata.get(s) ?? []
      list.push(i)
      strata.set(s, list)
    }
    return Array.from({ length: nRepeats }, (_, r) => {
      const shuffled = Float64Array.from(m.data)
      let t = 0
      for (const rows of strata.values()) {
        const perm = toFlat(permutation(foldIn(foldIn(key, j), r * 7919 + t++), rows.length))
        rows.forEach((i, a) => {
          shuffled[i * m.d + j] = m.data[(rows[perm[a] ?? a] ?? i) * m.d + j] ?? 0
        })
      }
      return baseline - scorer.score(fitted, { ...data, x: fromStorage(shuffled, [m.n, m.d], 'float64') })
    })
  })
  return {
    baseline,
    importances,
    mean: importances.map((d) => meanOf(d)),
    std: importances.map((d) => Math.sqrt(varianceOf(d, 0))),
    conditionOn,
  }
}
