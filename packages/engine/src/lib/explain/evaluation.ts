/**
 * Evaluating explanations, in Quantus's categories: faithfulness (deletion and insertion curves, faithfulness
 * correlation, monotonicity, ROAR's remove-and-retrain), robustness (maximum and average sensitivity, the local
 * Lipschitz estimate), localisation against a ground-truth mask (pointing game, relevance mass and rank), complexity
 * (sparseness, entropy), randomisation (similarity between explanations, for sanity checks), axioms (completeness)
 * and the agreement between explainers.
 */

import { MdmlError } from '../base/index.ts'
import { keyFor } from '../learn/_shared.ts'
import type { Estimator, LearnData } from '../learn/estimator.ts'
import { type Scoring, scorersOf } from '../learn/scoring.ts'
import { matrixView } from '../learn/validation.ts'
import { foldIn, type Key, permutation, uniform } from '../random/index.ts'
import { fromStorage, type TensorLike, toFlat } from '../tensor/index.ts'
import { baselineRows, targetOutputs } from './_outputs.ts'
import type { Target } from './gradients.ts'
import type { PredictFunction } from './shapley.ts'

/** Attributions, one row of values per explained row. */
export type AttributionRows = readonly (readonly number[])[]

function rowsOf(x: TensorLike, who: string): { n: number; d: number; data: Float64Array } {
  return matrixView(x, who)
}

function checkAttributions(a: AttributionRows, n: number, d: number, who: string): void {
  if (a.length !== n || a.some((r) => r.length !== d))
    throw new MdmlError('shape', `${who}: attributions must have one row of ${d} values per input`)
}

function order(values: readonly number[], descending: boolean): number[] {
  return values
    .map((v, j) => ({ v, j }))
    .sort((p, q) => (descending ? q.v - p.v : p.v - q.v) || p.j - q.j)
    .map((e) => e.j)
}

function trapezoid(xs: readonly number[], ys: readonly number[]): number {
  let s = 0
  for (let i = 1; i < xs.length; i++) s += (((ys[i] ?? 0) + (ys[i - 1] ?? 0)) / 2) * ((xs[i] ?? 0) - (xs[i - 1] ?? 0))
  return s
}

/** Options of {@link deletionCurve}. */
export interface DeletionOptions {
  readonly target?: Target
  /** What removed features become (default 0). */
  readonly baseline?: TensorLike | number
  /** Points of the curve (default 11: 0%, 10%, ..., 100% of the features). */
  readonly steps?: number
  /** `deletion` (default: remove the most relevant first) or `insertion` (start from the baseline and add them). */
  readonly mode?: 'deletion' | 'insertion'
}

/** A deletion or insertion curve per row and its area. */
export interface DeletionCurve {
  /** The share of features removed (or inserted) at each point. */
  readonly fractions: readonly number[]
  /** curves[row][point]: the model's target output. */
  readonly curves: readonly (readonly number[])[]
  /** The mean curve over rows. */
  readonly mean: readonly number[]
  /** The area under each row's curve (over fractions 0 to 1). */
  readonly auc: readonly number[]
}

/**
 * Deletion and insertion curves (Petsiuk, Das and Saenko 2018; Samek et al. 2017's region perturbation): remove the
 * features in order of decreasing attribution (replacing them by the baseline) and follow the target output; a
 * faithful explanation makes it fall fast (small area). Insertion starts from the baseline and adds them in the same
 * order; a faithful explanation makes the output rise fast (large area).
 *
 * @param predict - The model.
 * @param x - Inputs `[n, d]`.
 * @param attributions - One attribution per feature and row.
 * @param options - target, baseline, steps and mode.
 * @returns The curves, their mean and their areas.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { tensor } from 'mdml-engine/tensor'
 * const sum = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[2 * i] + x.data[2 * i + 1]))
 * explain.deletionCurve(sum, [[3, 1]], [[3, 1]], { steps: 3 }).curves // => [[4, 1, 0]]
 * ```
 */
export function deletionCurve(
  predict: PredictFunction,
  x: TensorLike,
  attributions: AttributionRows,
  options: DeletionOptions = {},
): DeletionCurve {
  const m = rowsOf(x, 'deletionCurve')
  checkAttributions(attributions, m.n, m.d, 'deletionCurve')
  const steps = options.steps ?? 11
  if (!(Number.isInteger(steps) && steps >= 2)) throw new MdmlError('value', 'deletionCurve: steps must be >= 2')
  const base = baselineRows(options.baseline, m.n, m.d, 'deletionCurve')
  const fractions = Array.from({ length: steps }, (_, s) => s / (steps - 1))
  const insertion = options.mode === 'insertion'
  const ranked = attributions.map((a) => order(a, true))
  const data = new Float64Array(steps * m.n * m.d)
  fractions.forEach((f, s) => {
    const k = Math.round(f * m.d)
    for (let i = 0; i < m.n; i++) {
      const top = new Set((ranked[i] ?? []).slice(0, k))
      for (let j = 0; j < m.d; j++) {
        const keep = insertion ? top.has(j) : !top.has(j)
        data[(s * m.n + i) * m.d + j] = keep ? (m.data[i * m.d + j] ?? 0) : (base[i * m.d + j] ?? 0)
      }
    }
  })
  const f = targetOutputs(predict, data, steps * m.n, m.d, options.target, (r) => r % m.n)
  const curves = Array.from({ length: m.n }, (_, i) => fractions.map((_, s) => f[s * m.n + i] ?? 0))
  const mean = fractions.map((_, s) => curves.reduce((acc, c) => acc + (c[s] ?? 0), 0) / m.n)
  return { fractions, curves, mean, auc: curves.map((c) => trapezoid(fractions, c)) }
}

/** Options of {@link faithfulnessCorrelation}. */
export interface FaithfulnessOptions {
  readonly target?: Target
  readonly baseline?: TensorLike | number
  /** Features removed per run (default a quarter, at least one). */
  readonly subsetSize?: number
  /** Random subsets per row (default 50). */
  readonly nRuns?: number
  readonly key?: Key
}

function pearson(a: readonly number[], b: readonly number[]): number {
  const n = a.length
  const ma = a.reduce((s, v) => s + v, 0) / n
  const mb = b.reduce((s, v) => s + v, 0) / n
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < n; i++) {
    sab += ((a[i] ?? 0) - ma) * ((b[i] ?? 0) - mb)
    saa += ((a[i] ?? 0) - ma) ** 2
    sbb += ((b[i] ?? 0) - mb) ** 2
  }
  return saa === 0 || sbb === 0 ? 0 : sab / Math.sqrt(saa * sbb)
}

/**
 * Faithfulness correlation (Bhatt, Weller and Moura 2020; Quantus FaithfulnessCorrelation): over random subsets of
 * features, the Pearson correlation between the subset's summed attribution and the fall in the target output when
 * the subset is set to the baseline. 1 means the attributions predict the effect of removing features exactly.
 *
 * @param predict - The model.
 * @param x - Inputs `[n, d]`.
 * @param attributions - One attribution per feature and row.
 * @param options - target, baseline, subsetSize, nRuns and key.
 * @returns One correlation per row.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { tensor } from 'mdml-engine/tensor'
 * const linear = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => 2 * x.data[3 * i] - x.data[3 * i + 1] + 3 * x.data[3 * i + 2]))
 * // The exact attributions w x of a linear model are perfectly faithful.
 * explain.faithfulnessCorrelation(linear, [[1, 2, 1]], [[2, -2, 3]], { subsetSize: 1, nRuns: 20 })[0].toFixed(6) // => '1.000000'
 * ```
 */
export function faithfulnessCorrelation(
  predict: PredictFunction,
  x: TensorLike,
  attributions: AttributionRows,
  options: FaithfulnessOptions = {},
): number[] {
  const m = rowsOf(x, 'faithfulnessCorrelation')
  checkAttributions(attributions, m.n, m.d, 'faithfulnessCorrelation')
  const size = Math.min(m.d, Math.max(1, options.subsetSize ?? Math.round(m.d / 4)))
  const runs = options.nRuns ?? 50
  const key = options.key ?? keyFor({}, 'faithfulnessCorrelation')
  const base = baselineRows(options.baseline, m.n, m.d, 'faithfulnessCorrelation')
  const fx = targetOutputs(predict, Float64Array.from(m.data), m.n, m.d, options.target)
  return Array.from({ length: m.n }, (_, i) => {
    const sums: number[] = []
    const data = new Float64Array(runs * m.d)
    for (let r = 0; r < runs; r++) {
      const subset = toFlat(permutation(foldIn(foldIn(key, i), r), m.d)).slice(0, size)
      const s = new Set(subset)
      sums.push(subset.reduce((acc, j) => acc + (attributions[i]?.[j] ?? 0), 0))
      for (let j = 0; j < m.d; j++) data[r * m.d + j] = s.has(j) ? (base[i * m.d + j] ?? 0) : (m.data[i * m.d + j] ?? 0)
    }
    const t = typeof options.target === 'object' ? (options.target[i] ?? 0) : (options.target ?? 0)
    const f = targetOutputs(predict, data, runs, m.d, t)
    return pearson(
      sums,
      Array.from(f, (v) => (fx[i] ?? 0) - v),
    )
  })
}

/**
 * Monotonicity (Arya et al. 2019; Quantus Monotonicity): starting from the baseline, add the features in order of
 * increasing attribution; the share of steps at which the target output does not fall. 1 when every added feature
 * helps at least as much as the ones before claimed.
 *
 * @param predict - The model.
 * @param x - Inputs `[n, d]`.
 * @param attributions - One attribution per feature and row.
 * @param options - target and baseline.
 * @returns One share per row.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { tensor } from 'mdml-engine/tensor'
 * const sum = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[2 * i] + x.data[2 * i + 1]))
 * explain.monotonicity(sum, [[3, 1]], [[3, 1]]) // => [1]
 * ```
 */
export function monotonicity(
  predict: PredictFunction,
  x: TensorLike,
  attributions: AttributionRows,
  options: { readonly target?: Target; readonly baseline?: TensorLike | number } = {},
): number[] {
  const m = rowsOf(x, 'monotonicity')
  checkAttributions(attributions, m.n, m.d, 'monotonicity')
  const base = baselineRows(options.baseline, m.n, m.d, 'monotonicity')
  return Array.from({ length: m.n }, (_, i) => {
    const ranked = order(attributions[i] ?? [], false)
    const data = new Float64Array((m.d + 1) * m.d)
    const cur = Float64Array.from(base.subarray(i * m.d, (i + 1) * m.d))
    data.set(cur, 0)
    ranked.forEach((j, s) => {
      cur[j] = m.data[i * m.d + j] ?? 0
      data.set(cur, (s + 1) * m.d)
    })
    const t = typeof options.target === 'object' ? (options.target[i] ?? 0) : (options.target ?? 0)
    const f = targetOutputs(predict, data, m.d + 1, m.d, t)
    let ok = 0
    for (let s = 1; s <= m.d; s++) if ((f[s] ?? 0) >= (f[s - 1] ?? 0) - 1e-12) ok++
    return ok / m.d
  })
}

/** Options of {@link removeAndRetrain}. */
export interface RoarOptions {
  /** Shares of each row's features removed (default 0, 0.1, ..., 0.9). */
  readonly fractions?: readonly number[]
  readonly scoring?: Scoring
  /** Remove the least relevant features instead (KAR, keep-and-retrain's complement; default false). */
  readonly leastFirst?: boolean
}

/**
 * Remove and retrain (ROAR; Hooker, Erhan, Kindermans and Kim 2019): for each share, replace each row's most
 * relevant features (by its own attributions) by the feature's training mean, in the training and the test data,
 * retrain and score; a faithful explanation makes the score fall faster than a random ranking. Retraining keeps the
 * fall from being an artefact of out-of-distribution inputs.
 *
 * @param estimator - The estimator to retrain.
 * @param train - Training data and the attributions of its rows.
 * @param test - Test data and the attributions of its rows.
 * @param trainAttributions - Attributions of the training rows.
 * @param testAttributions - Attributions of the test rows.
 * @param options - fractions, scoring and leastFirst.
 * @returns The shares and the score after retraining at each.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { tensor } from 'mdml-engine/tensor'
 * // A model that predicts the first column: removing it first ruins the score at once.
 * const first = defineEstimator({ name: 'first', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: (x) => tensor(x.map((r) => r[0])) }) }, {})
 * const data = { x: [[1, 0], [2, 0], [3, 0]], y: [1, 2, 3] }
 * const a = [[1, 0], [1, 0], [1, 0]]
 * explain.removeAndRetrain(first, data, data, a, a, { fractions: [0, 0.5], scoring: 'neg_mean_squared_error' }).scores.map((s) => Math.round(s * 1e9) / 1e9 + 0) // => [0, -0.666666667]
 * ```
 */
export function removeAndRetrain(
  estimator: Estimator,
  train: LearnData,
  test: LearnData,
  trainAttributions: AttributionRows,
  testAttributions: AttributionRows,
  options: RoarOptions = {},
): { readonly fractions: readonly number[]; readonly scores: readonly number[] } {
  const tr = rowsOf(train.x, 'removeAndRetrain')
  const te = rowsOf(test.x, 'removeAndRetrain')
  checkAttributions(trainAttributions, tr.n, tr.d, 'removeAndRetrain')
  checkAttributions(testAttributions, te.n, te.d, 'removeAndRetrain')
  const scorer = Object.values(scorersOf(options.scoring))[0]
  if (scorer === undefined) throw new MdmlError('value', 'removeAndRetrain: no scorer')
  const mean = Array.from({ length: tr.d }, (_, j) => {
    let s = 0
    for (let i = 0; i < tr.n; i++) s += tr.data[i * tr.d + j] ?? 0
    return s / tr.n
  })
  const fractions = options.fractions ?? Array.from({ length: 10 }, (_, i) => i / 10)
  const masked = (m: typeof tr, a: AttributionRows, f: number): number[][] => {
    const k = Math.round(f * m.d)
    return Array.from({ length: m.n }, (_, i) => {
      const drop = new Set(order(a[i] ?? [], options.leastFirst !== true).slice(0, k))
      return Array.from({ length: m.d }, (_, j) => (drop.has(j) ? (mean[j] ?? 0) : (m.data[i * m.d + j] ?? 0)))
    })
  }
  const scores = fractions.map((f) => {
    const fitted = estimator.fit({ ...train, x: masked(tr, trainAttributions, f) }, { explain: false })
    return scorer.score(fitted, { ...test, x: masked(te, testAttributions, f) })
  })
  return { fractions, scores }
}

/** An explainer to probe for robustness: inputs `[n, d]` to attributions `[n][d]`. */
export type Explainer = (x: TensorLike) => AttributionRows

/** Options of the sensitivity measures. */
export interface SensitivityOptions {
  /** The half-width of the L-infinity ball sampled around each input (default 0.1). */
  readonly radius?: number
  /** Perturbed inputs per row (default 10). */
  readonly nSamples?: number
  readonly key?: Key
}

function norm(v: readonly number[]): number {
  return Math.sqrt(v.reduce((s, a) => s + a * a, 0))
}

function perturbations(
  explainer: Explainer,
  x: TensorLike,
  options: SensitivityOptions,
  who: string,
): {
  readonly m: { n: number; d: number; data: Float64Array }
  readonly base: AttributionRows
  readonly samples: { readonly z: number[]; readonly e: readonly number[] }[][]
} {
  const m = rowsOf(x, who)
  const base = explainer(fromStorage(Float64Array.from(m.data), [m.n, m.d], 'float64'))
  checkAttributions(base, m.n, m.d, who)
  const ns = options.nSamples ?? 10
  const r = options.radius ?? 0.1
  const key = options.key ?? keyFor({}, who)
  const samples = Array.from({ length: m.n }, (_, i) => {
    const u = toFlat(uniform(foldIn(key, i), [ns, m.d]))
    const z = Array.from({ length: ns }, (_, s) =>
      Array.from({ length: m.d }, (_, j) => (m.data[i * m.d + j] ?? 0) + (2 * (u[s * m.d + j] ?? 0) - 1) * r),
    )
    const e = explainer(fromStorage(Float64Array.from(z.flat()), [ns, m.d], 'float64'))
    return z.map((zz, s) => ({ z: zz, e: e[s] ?? [] }))
  })
  return { m, base, samples }
}

/**
 * Maximum and average sensitivity (Yeh et al. 2019; Quantus MaxSensitivity and AvgSensitivity): how much an
 * explanation changes under small perturbations of the input, the largest and the mean of |e(x + delta) - e(x)| /
 * |e(x)| over inputs drawn uniformly from an L-infinity ball. Lower is more robust.
 *
 * @param explainer - The explainer.
 * @param x - Inputs `[n, d]`.
 * @param options - radius, nSamples and key.
 * @returns Per row, the maximum and the average relative change.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { toArray } from 'mdml-engine/tensor'
 * // A constant explanation never changes.
 * explain.sensitivity((x) => toArray(x).map(() => [1, 2]), [[0, 0]], { nSamples: 4 }) // => { max: [0], average: [0] }
 * ```
 */
export function sensitivity(
  explainer: Explainer,
  x: TensorLike,
  options: SensitivityOptions = {},
): { readonly max: readonly number[]; readonly average: readonly number[] } {
  const { base, samples } = perturbations(explainer, x, options, 'sensitivity')
  const max: number[] = []
  const average: number[] = []
  samples.forEach((rows, i) => {
    const e0 = base[i] ?? []
    const n0 = norm(e0) || 1
    const ratios = rows.map((s) => norm(s.e.map((v, j) => v - (e0[j] ?? 0))) / n0)
    max.push(Math.max(0, ...ratios))
    average.push(ratios.reduce((a, b) => a + b, 0) / Math.max(1, ratios.length))
  })
  return { max, average }
}

/**
 * The local Lipschitz estimate (Alvarez-Melis and Jaakkola 2018; Quantus LocalLipschitzEstimate): the largest
 * |e(x') - e(x)| / |x' - x| over inputs drawn from an L-infinity ball around x. Lower is more stable.
 *
 * @param explainer - The explainer.
 * @param x - Inputs `[n, d]`.
 * @param options - radius, nSamples and key.
 * @returns One estimate per row.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { toArray } from 'mdml-engine/tensor'
 * // The explanation 2x changes twice as fast as its input.
 * const twice = (x) => toArray(x).map((r) => r.map((v) => 2 * v))
 * explain.localLipschitz(twice, [[0, 0]], { nSamples: 5 }).map((v) => Math.round(v * 1e9) / 1e9) // => [2]
 * ```
 */
export function localLipschitz(explainer: Explainer, x: TensorLike, options: SensitivityOptions = {}): number[] {
  const { m, base, samples } = perturbations(explainer, x, options, 'localLipschitz')
  return samples.map((rows, i) => {
    const e0 = base[i] ?? []
    const x0 = Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
    return Math.max(
      0,
      ...rows.map((s) => {
        const dz = norm(s.z.map((v, j) => v - (x0[j] ?? 0)))
        return dz === 0 ? 0 : norm(s.e.map((v, j) => v - (e0[j] ?? 0))) / dz
      }),
    )
  })
}

/**
 * Localisation against a ground-truth mask (Zhang et al. 2018's pointing game; Arras, Osman and Samek 2022's relevance
 * mass and rank accuracy; Quantus): whether the most relevant feature lies in the mask, the share of positive
 * relevance that falls inside it, and the share of the top-|mask| features that do.
 *
 * @param attribution - One attribution per feature.
 * @param mask - Whether each feature belongs to the ground truth.
 * @returns hit, mass and rank.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * explain.localisation([0.25, 0.5, 0.25, -0.2], [false, true, true, false]) // => { hit: true, mass: 0.75, rank: 0.5 }
 * ```
 */
export function localisation(
  attribution: readonly number[],
  mask: readonly boolean[],
): { readonly hit: boolean; readonly mass: number; readonly rank: number } {
  if (mask.length !== attribution.length) throw new MdmlError('shape', 'localisation: one mask entry per feature')
  const ranked = order(attribution, true)
  const k = mask.filter(Boolean).length
  const pos = attribution.reduce((s, v) => s + Math.max(0, v), 0)
  const inside = attribution.reduce((s, v, j) => s + (mask[j] === true ? Math.max(0, v) : 0), 0)
  return {
    hit: mask[ranked[0] ?? 0] === true,
    mass: pos === 0 ? 0 : inside / pos,
    rank: k === 0 ? 0 : ranked.slice(0, k).filter((j) => mask[j] === true).length / k,
  }
}

/**
 * Complexity of an explanation (Chalasani et al. 2020's sparseness, Bhatt et al. 2020's complexity; Quantus): the
 * Gini index of the absolute attributions (0 when all are equal, near 1 when one feature holds everything) and the
 * entropy of their distribution (nats; low is simple).
 *
 * @param attribution - One attribution per feature.
 * @returns sparseness and entropy.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * explain.complexity([1, 1, 1, 1]).sparseness // => 0
 * explain.complexity([0, 0, 0, 5]).entropy // => 0
 * ```
 */
export function complexity(attribution: readonly number[]): { readonly sparseness: number; readonly entropy: number } {
  const a = attribution.map(Math.abs).sort((p, q) => p - q)
  const n = a.length
  const total = a.reduce((s, v) => s + v, 0)
  if (n === 0 || total === 0) return { sparseness: 0, entropy: 0 }
  let g = 0
  a.forEach((v, i) => {
    g += (2 * (i + 1) - n - 1) * v
  })
  let h = 0
  for (const v of a) {
    const p = v / total
    if (p > 0) h -= p * Math.log(p)
  }
  return { sparseness: g / (n * total), entropy: h }
}

function ranks(v: readonly number[]): number[] {
  const idx = v.map((x, i) => ({ x, i })).sort((p, q) => p.x - q.x)
  const out = new Array<number>(v.length).fill(0)
  let i = 0
  while (i < idx.length) {
    let j = i
    while (j + 1 < idx.length && idx[j + 1]?.x === idx[i]?.x) j++
    for (let t = i; t <= j; t++) out[idx[t]?.i ?? 0] = (i + j) / 2 + 1
    i = j + 1
  }
  return out
}

/** How two explanations of the same row are compared. */
export type SimilarityMetric = 'spearman' | 'pearson' | 'topk' | 'sign'

/**
 * The similarity of two explanations of one input (Adebayo et al. 2018's sanity checks compare the explanations of a
 * trained and a randomised model this way; Krishna et al. 2022 compare explainers): Spearman's correlation of their
 * absolute values (default), Pearson's correlation of the values, the overlap of their top-k features by magnitude,
 * or the share of those features on which their signs agree.
 *
 * @param a - One explanation.
 * @param b - Another of the same features.
 * @param options - metric and k (default a quarter of the features, at least one).
 * @returns The similarity, 1 for identical rankings.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * explain.explanationSimilarity([3, -2, 1], [6, -4, 2]) // => 1
 * explain.explanationSimilarity([3, 0, 0, 1], [0, 3, 0, 1], { metric: 'topk', k: 2 }) // => 0.5
 * ```
 */
export function explanationSimilarity(
  a: readonly number[],
  b: readonly number[],
  options: { readonly metric?: SimilarityMetric; readonly k?: number } = {},
): number {
  if (a.length !== b.length) throw new MdmlError('shape', 'explanationSimilarity: explanations differ in features')
  const metric = options.metric ?? 'spearman'
  if (metric === 'pearson') return pearson(a, b)
  if (metric === 'spearman') return pearson(ranks(a.map(Math.abs)), ranks(b.map(Math.abs)))
  const k = Math.min(a.length, Math.max(1, options.k ?? Math.round(a.length / 4)))
  const topA = order(a.map(Math.abs), true).slice(0, k)
  const topB = new Set(order(b.map(Math.abs), true).slice(0, k))
  if (metric === 'topk') return topA.filter((j) => topB.has(j)).length / k
  const shared = topA.filter((j) => topB.has(j))
  return shared.length === 0 ? 0 : shared.filter((j) => Math.sign(a[j] ?? 0) === Math.sign(b[j] ?? 0)).length / k
}

/**
 * Agreement between explainers (Krishna et al. 2022, the disagreement problem): for every pair of explainers, the
 * mean over rows of {@link explanationSimilarity} of their explanations.
 *
 * @param explanations - explanations[explainer][row][feature].
 * @param options - metric and k.
 * @returns The symmetric matrix of mean similarities (1 on the diagonal).
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * explain.explainerAgreement([[[1, 2, 3]], [[2, 4, 6]], [[3, 2, 1]]]) // => [[1, 1, -1], [1, 1, -1], [-1, -1, 1]]
 * ```
 */
export function explainerAgreement(
  explanations: readonly AttributionRows[],
  options: { readonly metric?: SimilarityMetric; readonly k?: number } = {},
): number[][] {
  const e = explanations.length
  const n = explanations[0]?.length ?? 0
  for (const x of explanations)
    if (x.length !== n) throw new MdmlError('shape', 'explainerAgreement: explainers differ in rows')
  return Array.from({ length: e }, (_, a) =>
    Array.from({ length: e }, (_, b) => {
      if (a === b) return 1
      let s = 0
      for (let i = 0; i < n; i++)
        s += explanationSimilarity(explanations[a]?.[i] ?? [], explanations[b]?.[i] ?? [], options)
      return n === 0 ? 0 : s / n
    }),
  )
}

/**
 * The completeness gap (Sundararajan et al. 2017's completeness, Shapley efficiency; Quantus Completeness): how far
 * each row's attributions fall from summing to the change in output from the baseline, |sum a - (f(x) - f(b))|.
 *
 * @param attributions - One attribution per feature and row.
 * @param outputs - f(x) per row.
 * @param baselineOutputs - f(baseline) per row (or one for all).
 * @returns One gap per row.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * explain.completenessGap([[1, 2], [1, 1]], [4, 4], [1]) // => [0, 1]
 * ```
 */
export function completenessGap(
  attributions: AttributionRows,
  outputs: readonly number[],
  baselineOutputs: readonly number[],
): number[] {
  if (outputs.length !== attributions.length) throw new MdmlError('shape', 'completenessGap: one output per row')
  return attributions.map((a, i) => {
    const b = baselineOutputs.length === 1 ? (baselineOutputs[0] ?? 0) : (baselineOutputs[i] ?? 0)
    return Math.abs(a.reduce((s, v) => s + v, 0) - ((outputs[i] ?? 0) - b))
  })
}
