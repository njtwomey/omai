/**
 * Filter feature selection, scikit-learn style: univariate scores (ANOVA F, correlation and regression F tests,
 * chi-squared, and mutual information by nearest-neighbour estimators) and the selectors built on them (variance
 * threshold, k best, percentile, p-value control of false positive, false discovery or familywise error rates) plus
 * `selectFromModel`, which keeps the features a fitted model deems important.
 */

import { type Json, quantileOf, WhiteboxError } from '../base/index.ts'
import { ksgMutualInformation, mixedMutualInformation, plugInMutualInformation } from '../information/index.ts'
import { chdtrc, fdtrc } from '../numerics/special/index.ts'
import { dict, int, real, type Space } from '../space/index.ts'
import { fOneway } from '../stats/tests/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { inputNames } from './_shared.ts'
import { defineEstimator, type Estimator, type FitOptions, type Fitted, type LearnData } from './estimator.ts'
import { columnOf, encodeLabels, fromView, matrixView, takeColumns, targetVector } from './validation.ts'

/** Scores and p-values of each feature from a univariate test. */
export interface UnivariateScores {
  readonly scores: readonly number[]
  /** p-values (absent for scores without a test, such as mutual information). */
  readonly pValues?: readonly number[]
}

/** A univariate scoring function of features and targets. */
export type ScoreFunction = (x: TensorLike, y: TensorLike) => UnivariateScores

/**
 * The ANOVA F statistic of each feature across the classes (sklearn.feature_selection.f_classif).
 *
 * @param x - Features `[n, d]`.
 * @param y - Class labels `[n]`.
 * @returns F statistics and p-values.
 * @concept feature-selection
 * @example
 * ```ts
 * import { fClassif } from 'whitebox-engine/learn'
 * fClassif([[1], [2], [3], [4], [5], [6]], [0, 0, 0, 1, 1, 1]).scores // => [13.5]
 * ```
 */
export function fClassif(x: TensorLike, y: TensorLike): UnivariateScores {
  const m = matrixView(x, 'fClassif')
  const { classes, codes } = encodeLabels(y, 'fClassif', m.n)
  const scores: number[] = []
  const pValues: number[] = []
  for (let j = 0; j < m.d; j++) {
    const col = columnOf(m, j)
    const groups = classes.map((_, c) => Array.from(col).filter((_, i) => codes[i] === c))
    const r = fOneway(groups)
    scores.push(r.statistic)
    pValues.push(r.pValue)
  }
  return { scores, pValues }
}

/**
 * Pearson's correlation of each feature with the target (sklearn.feature_selection.r_regression).
 *
 * @param x - Features `[n, d]`.
 * @param y - Targets `[n]`.
 * @param options - Centre the data (default true).
 * @returns One correlation per feature.
 * @example
 * ```ts
 * import { rRegression } from 'whitebox-engine/learn'
 * rRegression([[1, 3], [2, 2], [3, 1]], [1, 2, 3]).map((r) => Math.round(r * 1e12) / 1e12) // => [1, -1]
 * ```
 */
export function rRegression(x: TensorLike, y: TensorLike, options: { readonly center?: boolean } = {}): number[] {
  const m = matrixView(x, 'rRegression')
  const t = targetVector(y, 'rRegression', m.n)
  const center = options.center ?? true
  const yMean = center ? t.reduce((a, b) => a + b, 0) / m.n : 0
  const yc = t.map((v) => v - yMean)
  const yNorm = Math.sqrt(yc.reduce((a, b) => a + b * b, 0))
  return Array.from({ length: m.d }, (_, j) => {
    const col = columnOf(m, j)
    const mu = center ? col.reduce((a, b) => a + b, 0) / m.n : 0
    let num = 0
    let ss = 0
    for (let i = 0; i < m.n; i++) {
      const v = (col[i] ?? 0) - mu
      num += v * (yc[i] ?? 0)
      ss += v * v
    }
    return num / (Math.sqrt(ss) * yNorm)
  })
}

/**
 * The univariate linear-regression F test of each feature (sklearn.feature_selection.f_regression): F = r^2 / (1 - r^2)
 * (n - 2), with p-values from F(1, n - 2).
 *
 * @param x - Features `[n, d]`.
 * @param y - Targets `[n]`.
 * @returns F statistics and p-values.
 * @example
 * ```ts
 * import { fRegression } from 'whitebox-engine/learn'
 * Math.round(fRegression([[1], [2], [3], [4]], [1, 2, 2, 4]).scores[0] * 1000) / 1000 // => 11.571
 * ```
 */
export function fRegression(x: TensorLike, y: TensorLike): UnivariateScores {
  const r = rRegression(x, y)
  const n = matrixView(x, 'fRegression').n
  const dof = n - 2
  const scores = r.map((c) => ((c * c) / (1 - c * c)) * dof)
  const pValues = toFlat(fdtrc(1, dof, fromStorage(Float64Array.from(scores), [scores.length], 'float64')))
  return { scores, pValues }
}

/**
 * The chi-squared statistic between each non-negative feature (counts, frequencies) and the classes
 * (sklearn.feature_selection.chi2): observed class totals of the feature against those expected from the class
 * frequencies.
 *
 * @param x - Non-negative features `[n, d]`.
 * @param y - Class labels `[n]`.
 * @returns Chi-squared statistics and p-values (k - 1 degrees of freedom).
 * @throws WhiteboxError with code `domain` for negative features.
 * @example
 * ```ts
 * import { chi2 } from 'whitebox-engine/learn'
 * chi2([[1, 0], [1, 0], [0, 1], [0, 1]], [0, 0, 1, 1]).scores // => [2, 2]
 * ```
 */
export function chi2(x: TensorLike, y: TensorLike): UnivariateScores {
  const m = matrixView(x, 'chi2')
  if (m.data.some((v) => v < 0)) throw new WhiteboxError('domain', 'chi2: features must be non-negative')
  const { classes, codes } = encodeLabels(y, 'chi2', m.n)
  const k = classes.length
  const classProb = classes.map((_, c) => codes.filter((v) => v === c).length / m.n)
  const scores: number[] = []
  for (let j = 0; j < m.d; j++) {
    const observed = new Float64Array(k)
    let total = 0
    for (let i = 0; i < m.n; i++) {
      const v = m.data[i * m.d + j] ?? 0
      observed[codes[i] ?? 0] = (observed[codes[i] ?? 0] ?? 0) + v
      total += v
    }
    let s = 0
    for (let c = 0; c < k; c++) {
      const e = (classProb[c] ?? 0) * total
      s += e === 0 ? Number.NaN : ((observed[c] ?? 0) - e) ** 2 / e
    }
    scores.push(s)
  }
  const pValues = toFlat(chdtrc(k - 1, fromStorage(Float64Array.from(scores), [scores.length], 'float64')))
  return { scores, pValues }
}

/** Options of the mutual-information scores. */
export interface MutualInfoOptions {
  /** Which features are discrete (default none). */
  readonly discreteFeatures?: boolean | readonly boolean[]
  /** Neighbours of the k-NN estimators (default 3). */
  readonly nNeighbors?: number
}

function isDiscrete(options: MutualInfoOptions, j: number): boolean {
  const d = options.discreteFeatures
  return d === true || (Array.isArray(d) && d[j] === true)
}

function unitScale(v: Float64Array): Float64Array {
  const mu = v.reduce((a, b) => a + b, 0) / v.length
  // sklearn's preprocessing.scale: the population standard deviation.
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / v.length)
  return sd > 0 ? v.map((t) => t / sd) : v
}

/**
 * Mutual information between each feature and a class label (sklearn.feature_selection.mutual_info_classif, in nats):
 * Ross's k-NN estimator for continuous features (scaled to unit variance), the plug-in estimate for discrete ones;
 * negative estimates are clipped to 0. Unlike sklearn, no jitter is added, so the result is deterministic.
 *
 * @param x - Features `[n, d]`.
 * @param y - Class labels `[n]`.
 * @param options - discreteFeatures and nNeighbors.
 * @returns One estimate per feature.
 * @example
 * ```ts
 * import { mutualInfoClassif } from 'whitebox-engine/learn'
 * const x = [[0.1], [0.2], [0.3], [0.4], [5.1], [5.2], [5.3], [5.4]]
 * mutualInfoClassif(x, [0, 0, 0, 0, 1, 1, 1, 1])[0] > 0.5 // => true
 * ```
 */
export function mutualInfoClassif(x: TensorLike, y: TensorLike, options: MutualInfoOptions = {}): number[] {
  const m = matrixView(x, 'mutualInfoClassif')
  const labels = Array.from(encodeLabels(y, 'mutualInfoClassif', m.n).codes)
  return Array.from({ length: m.d }, (_, j) => {
    const col = columnOf(m, j)
    const mi = isDiscrete(options, j)
      ? plugInMutualInformation(Array.from(col), labels)
      : mixedMutualInformation(unitScale(col), labels, { k: options.nNeighbors ?? 3 })
    return Math.max(0, mi)
  })
}

/**
 * Mutual information between each feature and a continuous target (sklearn.feature_selection.mutual_info_regression,
 * in nats): the Kraskov estimator (features and target scaled to unit variance), Ross's estimator for discrete
 * features; clipped at 0, and deterministic (no jitter).
 *
 * @param x - Features `[n, d]`.
 * @param y - Targets `[n]`.
 * @param options - discreteFeatures and nNeighbors.
 * @returns One estimate per feature.
 * @example
 * ```ts
 * import { mutualInfoRegression } from 'whitebox-engine/learn'
 * const x = Array.from({ length: 40 }, (_, i) => [Math.sin(i), (i * 7) % 5])
 * const y = x.map(([a]) => 2 * a)
 * mutualInfoRegression(x, y)[0] > mutualInfoRegression(x, y)[1] // => true
 * ```
 */
export function mutualInfoRegression(x: TensorLike, y: TensorLike, options: MutualInfoOptions = {}): number[] {
  const m = matrixView(x, 'mutualInfoRegression')
  const t = unitScale(targetVector(y, 'mutualInfoRegression', m.n))
  const k = options.nNeighbors ?? 3
  return Array.from({ length: m.d }, (_, j) => {
    const col = columnOf(m, j)
    const mi = isDiscrete(options, j)
      ? mixedMutualInformation(t, Array.from(col), { k })
      : ksgMutualInformation(unitScale(col), t, { k })
    return Math.max(0, mi)
  })
}

/** The verbs of a fitted feature selector. */
export interface SelectorVerbs {
  /** Which input features are kept. */
  readonly support: readonly boolean[]
  readonly transform: (x: TensorLike) => Tensor
  /** Put the kept columns back in place, zeros elsewhere (sklearn's). */
  readonly inverseTransform: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
  readonly featureNamesOut: (input?: readonly string[]) => string[]
}

/**
 * The verbs of a selector from its support mask (shared by every selector).
 *
 * @param support - Which features are kept.
 * @param who - The selector (for errors).
 * @returns The verbs.
 * @example
 * ```ts
 * import { selectorVerbs } from 'whitebox-engine/learn'
 * import { toArray } from 'whitebox-engine/tensor'
 * const s = selectorVerbs([true, false, true], 'demo')
 * toArray(s.transform([[1, 2, 3]])) // => [[1, 3]]
 * toArray(s.inverseTransform([[1, 3]])) // => [[1, 0, 3]]
 * ```
 */
export function selectorVerbs(support: readonly boolean[], who: string): SelectorVerbs {
  const kept = support.flatMap((s, j) => (s ? [j] : []))
  const d = support.length
  return {
    support,
    nFeaturesIn: d,
    transform: (x) => fromView(takeColumns(matrixView(x, who, { allowNaN: true, nFeatures: d }), kept)),
    inverseTransform: (x) => {
      const m = matrixView(x, who, { allowNaN: true, nFeatures: kept.length })
      const out = new Float64Array(m.n * d)
      for (let i = 0; i < m.n; i++) {
        kept.forEach((j, c) => {
          out[i * d + j] = m.data[i * kept.length + c] ?? 0
        })
      }
      return fromStorage(out, [m.n, d], 'float64')
    },
    featureNamesOut: (input) => {
      const names = inputNames(d, input, who)
      return kept.map((j) => names[j] ?? '')
    },
  }
}

/** A fitted univariate or threshold selector: its support and the statistics it ranked by. */
export type SelectorParams = {
  readonly support: boolean[]
  readonly scores: number[]
  readonly pValues: number[] | null
}

/** A fitted feature selector. */
export type FittedSelector<P extends SelectorParams = SelectorParams> = Fitted<P> & SelectorVerbs

function selector(
  name: string,
  hyperSpace: (() => Space) | undefined,
  select: (data: LearnData, options: FitOptions) => SelectorParams,
): Estimator<Record<string, never>, FittedSelector> {
  return defineEstimator(
    {
      name,
      kind: 'transformer',
      ...(hyperSpace === undefined ? {} : { hyperSpace }),
      fit: (data, _h, options) => select(data, options),
      methods: (p) => selectorVerbs(p.support, name),
    },
    {},
  )
}

/**
 * Drop features whose (population) variance is at most a threshold (sklearn.feature_selection.VarianceThreshold);
 * with threshold 0 the peak-to-peak range decides, so constant features go however they are scaled.
 *
 * @param threshold - The variance threshold (default 0).
 * @returns The estimator.
 * @example
 * ```ts
 * import { varianceThreshold } from 'whitebox-engine/learn'
 * varianceThreshold().fit({ x: [[1, 0], [1, 2], [1, 4]] }).support // => [false, true]
 * ```
 */
export function varianceThreshold(threshold = 0): Estimator<Record<string, never>, FittedSelector> {
  return selector('variance-threshold', undefined, (data) => {
    const m = matrixView(data.x, 'varianceThreshold', { allowNaN: true })
    const scores = Array.from({ length: m.d }, (_, j) => {
      const col = Array.from(columnOf(m, j)).filter((v) => !Number.isNaN(v))
      const mu = col.reduce((a, b) => a + b, 0) / col.length
      const v = col.reduce((a, b) => a + (b - mu) ** 2, 0) / col.length
      const ptp =
        col.reduce((a, b) => Math.max(a, b), Number.NEGATIVE_INFINITY) -
        col.reduce((a, b) => Math.min(a, b), Number.POSITIVE_INFINITY)
      return threshold === 0 ? Math.min(v, ptp) : v
    })
    const support = scores.map((s) => s > threshold)
    if (!support.some(Boolean))
      throw new WhiteboxError('value', `varianceThreshold: no feature has variance above ${threshold}`)
    return { support, scores, pValues: null }
  })
}

function univariate(
  data: LearnData,
  scoreFunc: ScoreFunction,
  who: string,
): { scores: number[]; pValues: number[] | null } {
  if (data.y === undefined) throw new WhiteboxError('value', `${who}: needs targets y`)
  const r = scoreFunc(data.x, data.y)
  // NaN scores rank lowest (sklearn's _clean_nans).
  return {
    scores: r.scores.map((s) => (Number.isNaN(s) ? -Number.MAX_VALUE : s)),
    pValues: r.pValues === undefined ? null : [...r.pValues],
  }
}

/**
 * Keep the k best-scoring features (sklearn.feature_selection.SelectKBest); ties go to the later feature, as sklearn's
 * stable sort does.
 *
 * @param options - The score function (default {@link fClassif}) and k (default 10, or `'all'`).
 * @returns The estimator.
 * @example
 * ```ts
 * import { selectKBest, fRegression } from 'whitebox-engine/learn'
 * const x = [[1, 5, 0], [2, 3, 1], [3, 4, 0], [4, 1, 1]]
 * selectKBest({ scoreFunc: fRegression, k: 1 }).fit({ x, y: [1, 2, 3, 4] }).support // => [true, false, false]
 * ```
 */
export function selectKBest(
  options: { readonly scoreFunc?: ScoreFunction; readonly k?: number | 'all' } = {},
): Estimator<Record<string, never>, FittedSelector> {
  const scoreFunc = options.scoreFunc ?? fClassif
  return selector(
    'select-k-best',
    () => dict({ k: int(1, 100) }),
    (data) => {
      const { scores, pValues } = univariate(data, scoreFunc, 'selectKBest')
      const k = options.k === 'all' ? scores.length : Math.min(options.k ?? 10, scores.length)
      const order = scores.map((s, j) => ({ s, j })).sort((a, b) => a.s - b.s || a.j - b.j)
      const support = scores.map(() => false)
      for (const { j } of order.slice(order.length - k)) support[j] = true
      return { support, scores, pValues }
    },
  )
}

/**
 * Keep the features scoring in the top percentile (sklearn.feature_selection.SelectPercentile), breaking ties at the
 * threshold in feature order up to the percentile's count.
 *
 * @param options - The score function (default {@link fClassif}) and the percentile (default 10).
 * @returns The estimator.
 * @example
 * ```ts
 * import { selectPercentile, fRegression } from 'whitebox-engine/learn'
 * const x = [[1, 5, 0, 2], [2, 3, 1, 2], [3, 4, 0, 1], [4, 1, 1, 0]]
 * selectPercentile({ scoreFunc: fRegression, percentile: 50 }).fit({ x, y: [1, 2, 3, 4] }).support // => [true, false, false, true]
 * ```
 */
export function selectPercentile(
  options: { readonly scoreFunc?: ScoreFunction; readonly percentile?: number } = {},
): Estimator<Record<string, never>, FittedSelector> {
  const scoreFunc = options.scoreFunc ?? fClassif
  const percentile = options.percentile ?? 10
  if (!(percentile >= 0 && percentile <= 100))
    throw new WhiteboxError('value', 'selectPercentile: percentile must be in [0, 100]')
  return selector(
    'select-percentile',
    () => dict({ percentile: real(1, 100) }),
    (data) => {
      const { scores, pValues } = univariate(data, scoreFunc, 'selectPercentile')
      if (percentile === 100) return { support: scores.map(() => true), scores, pValues }
      if (percentile === 0) return { support: scores.map(() => false), scores, pValues }
      const threshold = quantileOf(scores, 1 - percentile / 100)
      const support = scores.map((s) => s > threshold)
      const ties = scores.flatMap((s, j) => (s === threshold ? [j] : []))
      const maxFeats = Math.floor((scores.length * percentile) / 100)
      const room = maxFeats - support.filter(Boolean).length
      for (const j of ties.slice(0, Math.max(0, room))) support[j] = true
      return { support, scores, pValues }
    },
  )
}

/** The error-rate controls of {@link selectByPValue}. */
export type PValueControl = 'fpr' | 'fdr' | 'fwe'

/**
 * Keep features by their p-values (sklearn's SelectFpr, SelectFdr and SelectFwe): below alpha (false-positive rate),
 * by the Benjamini-Hochberg step-up procedure (false-discovery rate), or below alpha / d (Bonferroni, family-wise).
 *
 * @param control - Which error rate to control.
 * @param options - The score function (default {@link fClassif}) and alpha (default 0.05).
 * @returns The estimator.
 * @example
 * ```ts
 * import { selectByPValue, fRegression } from 'whitebox-engine/learn'
 * const x = Array.from({ length: 20 }, (_, i) => [i, (i * 7) % 5])
 * selectByPValue('fwe', { scoreFunc: fRegression }).fit({ x, y: x.map(([a]) => a + 0.1 * (a % 3)) }).support // => [true, false]
 * ```
 */
export function selectByPValue(
  control: PValueControl,
  options: { readonly scoreFunc?: ScoreFunction; readonly alpha?: number } = {},
): Estimator<Record<string, never>, FittedSelector> {
  const scoreFunc = options.scoreFunc ?? fClassif
  const alpha = options.alpha ?? 0.05
  return selector(
    `select-${control}`,
    () => dict({ alpha: real(1e-4, 0.5, { log: true }) }),
    (data) => {
      const { scores, pValues } = univariate(data, scoreFunc, `select-${control}`)
      if (pValues === null) throw new WhiteboxError('value', `select-${control}: the score function gives no p-values`)
      const d = pValues.length
      let support: boolean[]
      if (control === 'fpr') support = pValues.map((p) => p < alpha)
      else if (control === 'fwe') support = pValues.map((p) => p < alpha / d)
      else {
        const sorted = [...pValues].sort((a, b) => a - b)
        let cut = Number.NEGATIVE_INFINITY
        sorted.forEach((p, i) => {
          if (p <= (alpha * (i + 1)) / d) cut = p
        })
        support = pValues.map((p) => p <= cut)
      }
      return { support, scores, pValues }
    },
  )
}

/** Options of {@link selectFromModel}. */
export interface SelectFromModelOptions {
  /** The fitting estimator (its fitted `featureImportances` are ranked). */
  readonly estimator: Estimator
  /** Keep importances at or above this: a number, `'mean'`, `'median'` or a scaled one such as `'1.25*mean'`. */
  readonly threshold?: number | string
  /** Keep at most this many features (the most important). */
  readonly maxFeatures?: number
}

/** Parameters of a fitted {@link selectFromModel}. */
export type FromModelParams = SelectorParams & { readonly threshold: number; readonly estimator: Json }

function resolveThreshold(threshold: number | string, importances: readonly number[]): number {
  if (typeof threshold === 'number') return threshold
  const m = /^\s*(?:([0-9.eE+-]+)\s*\*\s*)?(mean|median)\s*$/.exec(threshold)
  if (m === null) throw new WhiteboxError('value', `selectFromModel: unknown threshold ${threshold}`)
  const scale = m[1] === undefined ? 1 : Number(m[1])
  const base =
    m[2] === 'mean' ? importances.reduce((a, b) => a + b, 0) / importances.length : quantileOf(importances, 0.5)
  return scale * base
}

/**
 * Keep the features an estimator deems important (sklearn.feature_selection.SelectFromModel): fit it, read its
 * `featureImportances` (absolute coefficients or impurity decreases), and keep those at or above the threshold
 * (default the mean), at most `maxFeatures` of them.
 *
 * @param options - estimator, threshold and maxFeatures.
 * @returns The estimator.
 * @throws WhiteboxError with code `value` when the fitted estimator reports no importances.
 * @example
 * ```ts
 * import { defineEstimator, selectFromModel } from 'whitebox-engine/learn'
 * const fixed = defineEstimator({ name: 'fixed', kind: 'regressor', fit: () => ({}), methods: () => ({ featureImportances: [0.1, 2, 0.5] }) }, {})
 * selectFromModel({ estimator: fixed }).fit({ x: [[0, 0, 0]], y: [0] }).support // => [false, true, false]
 * ```
 */
export function selectFromModel(
  options: SelectFromModelOptions,
): Estimator<SelectFromModelOptions, Fitted<FromModelParams> & SelectorVerbs> {
  return defineEstimator(
    {
      name: 'select-from-model',
      kind: 'transformer',
      fit: (data, h, fitOptions) => {
        const fitted = h.estimator.fit(data, fitOptions)
        const imp = fitted.featureImportances
        if (imp === undefined)
          throw new WhiteboxError('value', `selectFromModel: ${fitted.estimator} reports no featureImportances`)
        const threshold = resolveThreshold(h.threshold ?? 'mean', imp)
        let support = imp.map((v) => v >= threshold)
        if (h.maxFeatures !== undefined) {
          const top = new Set(
            imp
              .map((v, j) => ({ v, j }))
              .sort((a, b) => b.v - a.v || a.j - b.j)
              .slice(0, h.maxFeatures)
              .map((e) => e.j),
          )
          support = support.map((s, j) => s && top.has(j))
        }
        return { support, scores: [...imp], pValues: null, threshold, estimator: fitted.params }
      },
      methods: (p) => selectorVerbs(p.support, 'selectFromModel'),
    },
    options,
  )
}
