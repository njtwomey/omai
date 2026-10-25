/**
 * Conformal prediction, which turns any model's point predictions into prediction intervals or label sets with
 * guaranteed coverage under exchangeability, MAPIE style. Split conformal regression and classification (including APS
 * and RAPS scores), conformalised quantile regression, jackknife+ and CV+, Mondrian group-conditional intervals,
 * adaptive conformal inference for shifting data, and coverage checks.
 */

import { WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, permutation, uniform } from '../random/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { innerFitOptions, keyFor, runOptionsFor } from './_shared.ts'
import {
  type Estimator,
  type FitOptions,
  type Fitted,
  type LearnData,
  requireTargets,
  requireVerb,
} from './estimator.ts'
import { type CvOption, checkCv, splitInputOf } from './model-selection/cross-validation.ts'
import { splitIndices } from './model-selection/splitters.ts'
import { encodeLabels, fractionOption, matrixView, rowsOf, takeData, targetVector } from './validation.ts'

const EPSILON = 1e-8

/**
 * The split-conformal quantile of calibration scores: the k-th smallest with k = ceil((1 - alpha)(n + 1)), or
 * +infinity when k > n (too few calibration points for the level). With exchangeable scores, a new score falls at
 * or below it with probability at least 1 - alpha (Vovk et al. 2005; Lei et al. 2018).
 *
 * @param scores - Calibration conformity scores (NaN ignored).
 * @param alpha - The miscoverage level in (0, 1).
 * @returns The quantile.
 * @concept conformal-prediction
 * @example
 * ```ts
 * import { conformalQuantile } from 'whitebox-engine/learn'
 * conformalQuantile([5, 1, 4, 2, 3, 9, 7, 8, 6], 0.2) // => 8
 * conformalQuantile([1, 2], 0.1) // => Infinity
 * ```
 */
export function conformalQuantile(scores: ArrayLike<number>, alpha: number): number {
  fractionOption(alpha, 'alpha', 'conformalQuantile')
  const s = Array.from(scores)
    .filter((v) => !Number.isNaN(v))
    .sort((a, b) => a - b)
  if (s.length === 0) throw new WhiteboxError('value', 'conformalQuantile: no calibration scores')
  const k = Math.ceil((1 - alpha) * (s.length + 1))
  return k > s.length ? Number.POSITIVE_INFINITY : (s[k - 1] ?? Number.POSITIVE_INFINITY)
}

/** Prediction intervals: the point prediction and the bounds, each `[n]`. */
export interface Intervals {
  readonly prediction: Tensor
  readonly lower: Tensor
  readonly upper: Tensor
}

/** A fitted conformal regressor: its quantile(s) and the interval map. */
export interface ConformalRegressor {
  readonly alpha: number
  /** The conformal quantile of the scores (the half-width of absolute-residual intervals). */
  readonly quantile: number
  readonly predictInterval: (x: TensorLike) => Intervals
}

/** The conformity score of {@link splitConformalRegressor}. */
export type RegressionScore = 'absolute' | 'gamma' | 'normalized'

/** Options of {@link splitConformalRegressor}. */
export interface SplitConformalOptions {
  /** Miscoverage (default 0.1: 90% intervals). */
  readonly alpha?: number
  /**
   * |y - yhat| (default), the signed relative residual (y - yhat) / yhat with each tail at alpha / 2 (gamma, for
   * positive targets, as MAPIE), or |y - yhat| / sigma(x) (normalised).
   */
  readonly score?: RegressionScore
  /** The scale sigma(x) for normalised scores (a fitted model of |residual|, or any function of x). */
  readonly scale?: Fitted | ((x: TensorLike) => Tensor)
}

function predictOf(f: Fitted | ((x: TensorLike) => Tensor), who: string): (x: TensorLike) => Float64Array {
  if (typeof f === 'function') return (x) => Float64Array.from(toFlat(asTensor(f(x))))
  const p = requireVerb(f, 'predict', who)
  return (x) => Float64Array.from(toFlat(p(x)))
}

const vec = (v: ArrayLike<number>): Tensor => fromStorage(Float64Array.from(v), [v.length], 'float64')

/**
 * Split (inductive) conformal regression (Papadopoulos et al. 2002; MAPIE SplitConformalRegressor with a prefit
 * model): calibrate the scores of a fitted regressor on held-out data, then predict yhat -/+ q (scaled for gamma and
 * normalised scores). Marginal coverage is at least 1 - alpha for exchangeable data, whatever the model.
 *
 * @param fitted - A fitted regressor (its `predict` is used).
 * @param calibration - Held-out calibration data.
 * @param options - alpha, score and scale.
 * @returns The conformal regressor.
 * @example
 * ```ts
 * import { splitConformalRegressor } from 'whitebox-engine/learn'
 * import { tensor, toFlat } from 'whitebox-engine/tensor'
 * const zero = { estimator: 'zero', kind: 'regressor', params: {}, predict: (x) => tensor(x.map(() => 0)) }
 * const c = splitConformalRegressor(zero, { x: [[0], [0], [0], [0]], y: [1, -2, 3, -4] }, { alpha: 0.2 })
 * toFlat(c.predictInterval([[0]]).upper) // => [4]
 * ```
 */
export function splitConformalRegressor(
  fitted: Fitted,
  calibration: LearnData,
  options: SplitConformalOptions = {},
): ConformalRegressor {
  const alpha = fractionOption(options.alpha ?? 0.1, 'alpha', 'splitConformalRegressor')
  const score = options.score ?? 'absolute'
  const predict = predictOf(fitted, 'splitConformalRegressor')
  const n = rowsOf(calibration.x)
  const y = targetVector(requireTargets(calibration, 'splitConformalRegressor'), 'splitConformalRegressor', n)
  const scaleOf = (x: TensorLike, yhat: Float64Array): Float64Array => {
    if (score === 'absolute') return new Float64Array(yhat.length).fill(1)
    if (options.scale === undefined)
      throw new WhiteboxError('value', 'splitConformalRegressor: normalised scores need a scale')
    return predictOf(options.scale, 'splitConformalRegressor')(x)
  }
  const yhat = predict(calibration.x)
  if (score === 'gamma') {
    // MAPIE's gamma score is signed and asymmetric: (y - yhat) / yhat, each tail at alpha / 2.
    if (Array.from(y).some((v) => v <= 0) || Array.from(yhat).some((v) => v <= 0)) {
      throw new WhiteboxError('domain', 'splitConformalRegressor: gamma scores need positive targets and predictions')
    }
    const signed = Array.from(y, (v, i) => (v - (yhat[i] ?? 1)) / (yhat[i] ?? 1)).sort((a, b) => a - b)
    const k = Math.ceil((1 - alpha / 2) * (signed.length + 1))
    const qHi = k > signed.length ? Number.POSITIVE_INFINITY : (signed[k - 1] ?? Number.POSITIVE_INFINITY)
    const qLo = k > signed.length ? Number.NEGATIVE_INFINITY : (signed[signed.length - k] ?? Number.NEGATIVE_INFINITY)
    return {
      alpha,
      quantile: qHi,
      predictInterval: (x) => {
        const p = predict(x)
        return { prediction: vec(p), lower: vec(p.map((v) => v * (1 + qLo))), upper: vec(p.map((v) => v * (1 + qHi))) }
      },
    }
  }
  const sc = scaleOf(calibration.x, yhat)
  const scores = Array.from(y, (v, i) => Math.abs(v - (yhat[i] ?? 0)) / (sc[i] ?? 1))
  const q = conformalQuantile(scores, alpha)
  return {
    alpha,
    quantile: q,
    predictInterval: (x) => {
      const p = predict(x)
      const s = scaleOf(x, p)
      return {
        prediction: vec(p),
        lower: vec(p.map((v, i) => v - q * (s[i] ?? 1))),
        upper: vec(p.map((v, i) => v + q * (s[i] ?? 1))),
      }
    },
  }
}

/** Options of {@link conformalizedQuantileRegressor}. */
export interface CqrOptions {
  readonly alpha?: number
  /** One correction for both bounds (default false: each bound corrected at level alpha / 2, as MAPIE). */
  readonly symmetric?: boolean
}

/**
 * Conformalized quantile regression (Romano, Patterson and Candes 2019; MAPIE ConformalizedQuantileRegressor):
 * widen (or narrow) fitted lower and upper quantile models by conformal quantiles of q_lo(x) - y and y - q_hi(x), so
 * the intervals adapt to heteroscedasticity and still cover with probability 1 - alpha.
 *
 * @param models - Fitted lower and upper quantile regressors (and optionally a median for the point prediction).
 * @param calibration - Held-out calibration data.
 * @param options - alpha (default 0.1) and symmetric.
 * @returns The conformal regressor (its `quantile` is the lower correction).
 * @example
 * ```ts
 * import { conformalizedQuantileRegressor } from 'whitebox-engine/learn'
 * import { tensor, toFlat } from 'whitebox-engine/tensor'
 * const constant = (c) => ({ estimator: 'c', kind: 'regressor', params: {}, predict: (x) => tensor(x.map(() => c)) })
 * const c = conformalizedQuantileRegressor({ lower: constant(-1), upper: constant(1) }, { x: [[0], [0], [0]], y: [0, 2, -3] }, { alpha: 0.5, symmetric: true })
 * toFlat(c.predictInterval([[0]]).upper) // => [2]
 * ```
 */
export function conformalizedQuantileRegressor(
  models: { readonly lower: Fitted; readonly upper: Fitted; readonly median?: Fitted },
  calibration: LearnData,
  options: CqrOptions = {},
): ConformalRegressor & { readonly upperQuantile: number } {
  const alpha = fractionOption(options.alpha ?? 0.1, 'alpha', 'conformalizedQuantileRegressor')
  const lo = predictOf(models.lower, 'conformalizedQuantileRegressor')
  const hi = predictOf(models.upper, 'conformalizedQuantileRegressor')
  const mid = models.median === undefined ? undefined : predictOf(models.median, 'conformalizedQuantileRegressor')
  const y = targetVector(requireTargets(calibration, 'cqr'), 'cqr', rowsOf(calibration.x))
  const l = lo(calibration.x)
  const u = hi(calibration.x)
  const eLo = Array.from(y, (v, i) => (l[i] ?? 0) - v)
  const eHi = Array.from(y, (v, i) => v - (u[i] ?? 0))
  const symmetric = options.symmetric ?? false
  const qLo = symmetric
    ? conformalQuantile(
        eLo.map((v, i) => Math.max(v, eHi[i] ?? 0)),
        alpha,
      )
    : conformalQuantile(eLo, alpha / 2)
  const qHi = symmetric ? qLo : conformalQuantile(eHi, alpha / 2)
  return {
    alpha,
    quantile: qLo,
    upperQuantile: qHi,
    predictInterval: (x) => {
      const a = lo(x)
      const b = hi(x)
      return {
        prediction: vec(mid === undefined ? a.map((v, i) => (v + (b[i] ?? 0)) / 2) : mid(x)),
        lower: vec(a.map((v) => v - qLo)),
        upper: vec(b.map((v) => v + qHi)),
      }
    },
  }
}

/** Options of {@link crossConformalRegressor}. */
export interface CrossConformalOptions extends FitOptions {
  readonly alpha?: number
  /** The splitter (leave-one-out gives the jackknife+; default 5 folds gives CV+). */
  readonly cv?: CvOption
  /** `'plus'` (Barber et al.), `'minmax'` (jackknife-minmax) or `'base'` (the full model -/+ the residual quantile). */
  readonly method?: 'plus' | 'minmax' | 'base'
}

/**
 * The jackknife+ and CV+ (Barber, Candes, Ramdas and Tibshirani 2021; MAPIE CrossConformalRegressor): fit one model
 * per fold, score each sample by its out-of-fold residual R_i, and bound a new point by the floor(alpha (n + 1))-th
 * smallest of mu_-k(i)(x) - R_i and the ceil((1 - alpha)(n + 1))-th smallest of mu_-k(i)(x) + R_i. Coverage is at
 * least 1 - 2 alpha without a held-out set (about 1 - alpha in practice). One step per fold, traced.
 *
 * @param estimator - The regressor.
 * @param data - The training data.
 * @param options - alpha (default 0.1), cv, method, key and tracing.
 * @returns The conformal regressor (prediction: the mean of the fold models; its `quantile` is the residual quantile).
 * @example
 * ```ts
 * import { crossConformalRegressor, defineEstimator, rowsOf } from 'whitebox-engine/learn'
 * import { leaveOneOut } from 'whitebox-engine/learn/model-selection'
 * import { full, item, mean, toFlat } from 'whitebox-engine/tensor'
 * const meanModel = defineEstimator(
 *   { name: 'mean', kind: 'regressor', fit: (d) => ({ m: item(mean(d.y)) }), methods: (p) => ({ predict: (x) => full([rowsOf(x)], p.m) }) },
 *   {},
 * )
 * const y = [0, 1, 2, 3, 4, 5, 6, 7, 8]
 * const c = crossConformalRegressor(meanModel, { x: y.map(() => [0]), y }, { cv: leaveOneOut(), alpha: 0.2 })
 * toFlat(c.predictInterval([[0]]).lower).length // => 1
 * ```
 */
export function crossConformalRegressor(
  estimator: Estimator,
  data: LearnData,
  options: CrossConformalOptions = {},
): ConformalRegressor {
  const alpha = fractionOption(options.alpha ?? 0.1, 'alpha', 'crossConformalRegressor')
  const method = options.method ?? 'plus'
  const n = rowsOf(data.x)
  const y = targetVector(requireTargets(data, 'crossConformalRegressor'), 'crossConformalRegressor', n)
  const splitter = checkCv(options.cv, estimator)
  if (!splitter.partition)
    throw new WhiteboxError('value', 'crossConformalRegressor: the splitter must partition the samples')
  const key = keyFor(options, 'crossConformalRegressor')
  const folds = splitIndices(splitter, splitInputOf(data), foldIn(key, 0))
  const foldOf = new Int32Array(n)
  const residuals = new Float64Array(n)
  const models: Fitted[] = []
  const algorithm: Algorithm<number> = {
    name: 'cross-conformal',
    config: { estimator: estimator.name, folds: folds.length, method },
    init: () => 0,
    step: (f, ctx) => {
      const fold = folds[f]
      if (fold === undefined) return f + 1
      const fitted = estimator.fit(takeData(data, fold.train), innerFitOptions(options, foldIn(ctx.key, 0), ctx.run))
      const p = predictOf(fitted, 'crossConformalRegressor')(takeData(data, fold.test).x)
      fold.test.forEach((i, r) => {
        foldOf[i] = f
        residuals[i] = Math.abs((y[i] ?? 0) - (p[r] ?? 0))
      })
      models.push(fitted)
      return f + 1
    },
    done: (f) => (f >= folds.length ? 'completed' : undefined),
  }
  runAlgorithm(algorithm, { ...runOptionsFor(options, 'crossConformalRegressor'), maxSteps: folds.length })
  const q = conformalQuantile(residuals, alpha)
  const full =
    method === 'base' ? estimator.fit(data, innerFitOptions(options, foldIn(key, 1), options.parent)) : undefined
  return {
    alpha,
    quantile: q,
    predictInterval: (x) => {
      const m = rowsOf(x)
      const preds = models.map((f) => predictOf(f, 'crossConformalRegressor')(x))
      const centre = Float64Array.from(
        { length: m },
        (_, j) => preds.reduce((s, p) => s + (p[j] ?? 0), 0) / preds.length,
      )
      if (full !== undefined) {
        const p = predictOf(full, 'crossConformalRegressor')(x)
        return { prediction: vec(p), lower: vec(p.map((v) => v - q)), upper: vec(p.map((v) => v + q)) }
      }
      const lower = new Float64Array(m)
      const upper = new Float64Array(m)
      const kLo = Math.floor(alpha * (n + 1))
      const kHi = Math.ceil((1 - alpha) * (n + 1))
      for (let j = 0; j < m; j++) {
        if (method === 'minmax') {
          let mn = Number.POSITIVE_INFINITY
          let mx = Number.NEGATIVE_INFINITY
          for (const p of preds) {
            mn = Math.min(mn, p[j] ?? 0)
            mx = Math.max(mx, p[j] ?? 0)
          }
          lower[j] = mn - q
          upper[j] = mx + q
          continue
        }
        const lo = Array.from({ length: n }, (_, i) => (preds[foldOf[i] ?? 0]?.[j] ?? 0) - (residuals[i] ?? 0)).sort(
          (a, b) => a - b,
        )
        const hi = Array.from({ length: n }, (_, i) => (preds[foldOf[i] ?? 0]?.[j] ?? 0) + (residuals[i] ?? 0)).sort(
          (a, b) => a - b,
        )
        lower[j] = kLo < 1 ? Number.NEGATIVE_INFINITY : (lo[kLo - 1] ?? Number.NEGATIVE_INFINITY)
        upper[j] = kHi > n ? Number.POSITIVE_INFINITY : (hi[kHi - 1] ?? Number.POSITIVE_INFINITY)
      }
      return { prediction: vec(centre), lower: vec(lower), upper: vec(upper) }
    },
  }
}

/**
 * Mondrian (group-conditional) split conformal regression (Vovk et al. 2005; crepes' Mondrian categories): one
 * conformal quantile per group, so coverage holds within each group, not only on average.
 *
 * @param fitted - A fitted regressor.
 * @param calibration - Calibration data.
 * @param groups - The group of each calibration row.
 * @param options - alpha (default 0.1).
 * @returns The quantile of each group and an interval map that takes the test rows' groups.
 * @example
 * ```ts
 * import { mondrianConformalRegressor } from 'whitebox-engine/learn'
 * import { tensor, toFlat } from 'whitebox-engine/tensor'
 * const zero = { estimator: 'zero', kind: 'regressor', params: {}, predict: (x) => tensor(x.map(() => 0)) }
 * const c = mondrianConformalRegressor(zero, { x: [[0], [0], [0], [0]], y: [1, 1, 5, 5] }, [0, 0, 1, 1], { alpha: 0.4 })
 * toFlat(c.predictInterval([[0], [0]], [0, 1]).upper) // => [1, 5]
 * ```
 */
export function mondrianConformalRegressor(
  fitted: Fitted,
  calibration: LearnData,
  groups: ArrayLike<number>,
  options: { readonly alpha?: number } = {},
): {
  readonly quantiles: ReadonlyMap<number, number>
  readonly predictInterval: (x: TensorLike, groups: ArrayLike<number>) => Intervals
} {
  const alpha = fractionOption(options.alpha ?? 0.1, 'alpha', 'mondrianConformalRegressor')
  const predict = predictOf(fitted, 'mondrianConformalRegressor')
  const n = rowsOf(calibration.x)
  if (groups.length !== n) throw new WhiteboxError('shape', 'mondrianConformalRegressor: one group per calibration row')
  const y = targetVector(requireTargets(calibration, 'mondrian'), 'mondrian', n)
  const p = predict(calibration.x)
  const byGroup = new Map<number, number[]>()
  for (let i = 0; i < n; i++) {
    const g = groups[i] ?? 0
    const list = byGroup.get(g) ?? []
    list.push(Math.abs((y[i] ?? 0) - (p[i] ?? 0)))
    byGroup.set(g, list)
  }
  const quantiles = new Map(Array.from(byGroup, ([g, s]) => [g, conformalQuantile(s, alpha)] as const))
  return {
    quantiles,
    predictInterval: (x, g) => {
      const q = predict(x)
      const half = Array.from(q, (_, i) => quantiles.get(g[i] ?? Number.NaN) ?? Number.POSITIVE_INFINITY)
      return {
        prediction: vec(q),
        lower: vec(q.map((v, i) => v - (half[i] ?? 0))),
        upper: vec(q.map((v, i) => v + (half[i] ?? 0))),
      }
    },
  }
}

/** The conformity score of {@link splitConformalClassifier}. */
export type ClassificationScore = 'lac' | 'aps' | 'raps'

/** Options of {@link splitConformalClassifier}. */
export interface ConformalClassifierOptions {
  readonly alpha?: number
  /** LAC (1 - p of the true class, default), APS (cumulative mass up to it) or RAPS (APS plus a rank penalty). */
  readonly score?: ClassificationScore
  /** Randomise the APS/RAPS calibration scores, subtracting U p(true label) (default false; MAPIE always does). */
  readonly randomized?: boolean
  /** APS/RAPS sets: keep the label that crosses the threshold (`'last'`, default, conservative) or drop it at random. */
  readonly include?: 'last' | 'randomized'
  /** Uniform draws for the randomised calibration scores (default drawn from the key): to reproduce a reference. */
  readonly uniforms?: ArrayLike<number>
  /** The RAPS penalty per label beyond kReg (default 0.01) and the penalty-free set size (default 1). */
  readonly lambda?: number
  readonly kReg?: number
  /** Label-conditional (Mondrian by class) quantiles (default false). */
  readonly classConditional?: boolean
  readonly key?: Key
}

/** A fitted conformal classifier. */
export interface ConformalClassifier {
  readonly alpha: number
  /** The conformal quantile (or one per class, for class-conditional calibration). */
  readonly quantile: number | readonly number[]
  readonly classes: readonly number[]
  /** Membership `[n, k]` (1 when a class is in the set), columns in the order of `classes`. */
  readonly predictSet: (x: TensorLike) => Tensor
  /** The prediction sets as lists of labels. */
  readonly predictLabels: (x: TensorLike) => number[][]
}

function sortedDesc(p: ArrayLike<number>): number[] {
  // Descending by probability; ties by the later class first, as numpy's fliplr(argsort).
  return Array.from({ length: p.length }, (_, c) => c)
    .sort((a, b) => (p[a] ?? 0) - (p[b] ?? 0) || a - b)
    .reverse()
}

/**
 * Split conformal classification (Sadinle, Lei and Wasserman 2019; Romano, Sesia and Candes 2020; Angelopoulos et
 * al. 2021; MAPIE SplitConformalClassifier with a prefit model): calibrate the scores of the true labels, then return
 * every label whose score is within the conformal quantile. LAC gives the smallest sets on average; APS adapts the
 * set size to the difficulty; RAPS penalises long tails of unlikely labels. Sets cover the true label with
 * probability at least 1 - alpha.
 *
 * @param fitted - A fitted classifier with `predictProba`.
 * @param calibration - Held-out calibration data.
 * @param options - alpha (default 0.1), score, randomized, include, uniforms, lambda, kReg, classConditional and key.
 * @returns The conformal classifier.
 * @example
 * ```ts
 * import { splitConformalClassifier } from 'whitebox-engine/learn'
 * import { tensor } from 'whitebox-engine/tensor'
 * const fixed = { estimator: 'fixed', kind: 'classifier', params: {}, classes: [0, 1], predictProba: (x) => tensor(x.map((r) => [r[0], 1 - r[0]])) }
 * const x = [[0.9], [0.8], [0.3], [0.6], [0.2], [0.1], [0.7], [0.4], [0.95]]
 * const c = splitConformalClassifier(fixed, { x, y: [0, 0, 1, 0, 1, 1, 0, 1, 0] }, { alpha: 0.2 })
 * // A confident row gets one label; an ambiguous one gets none at this level (an abstention).
 * c.predictLabels([[0.99], [0.5]]) // => [[0], []]
 * ```
 */
export function splitConformalClassifier(
  fitted: Fitted,
  calibration: LearnData,
  options: ConformalClassifierOptions = {},
): ConformalClassifier {
  const alpha = fractionOption(options.alpha ?? 0.1, 'alpha', 'splitConformalClassifier')
  const score = options.score ?? 'lac'
  const proba = requireVerb(fitted, 'predictProba', 'splitConformalClassifier')
  const y = requireTargets(calibration, 'splitConformalClassifier')
  const classes = fitted.classes ?? encodeLabels(y, 'splitConformalClassifier').classes
  const index = new Map(classes.map((c, i) => [c, i]))
  const labels = Array.from(targetVector(y, 'splitConformalClassifier'), (v) => {
    const c = index.get(v)
    if (c === undefined)
      throw new WhiteboxError('value', `splitConformalClassifier: calibration label ${v} is not a class`)
    return c
  })
  const k = classes.length
  const lambda = score === 'raps' ? (options.lambda ?? 0.01) : 0
  const kReg = options.kReg ?? 1
  const randomized = options.randomized ?? false
  const p = matrixView(proba(calibration.x), 'splitConformalClassifier', { nFeatures: k })
  const u =
    options.uniforms !== undefined
      ? Float64Array.from(options.uniforms)
      : Float64Array.from(toFlat(uniform(options.key ?? foldIn(keyFor({}, 'splitConformalClassifier'), 0), [p.n])))
  // The score of class c for a row: LAC 1 - p_c; APS/RAPS the (penalised) mass of the labels ranked up to c.
  const scoreOf = (row: Float64Array, c: number, ui: number | undefined): number => {
    if (score === 'lac') return 1 - (row[c] ?? 0)
    const order = sortedDesc(row)
    let cum = 0
    for (let r = 0; r < order.length; r++) {
      const label = order[r] ?? 0
      cum += row[label] ?? 0
      if (label === c) {
        const penalty = lambda * Math.max(0, r + 1 - kReg)
        return cum + penalty - (ui === undefined ? 0 : ui * (row[c] ?? 0))
      }
    }
    return Number.POSITIVE_INFINITY
  }
  const rows = Array.from({ length: p.n }, (_, i) => p.data.subarray(i * k, (i + 1) * k))
  const calScores = rows.map((row, i) => scoreOf(row, labels[i] ?? 0, randomized ? (u[i] ?? 0) : undefined))
  const quantile: number | number[] =
    options.classConditional === true
      ? classes.map((_, c) => {
          const s = calScores.filter((_, i) => labels[i] === c)
          return s.length === 0 ? Number.POSITIVE_INFINITY : conformalQuantile(s, alpha)
        })
      : conformalQuantile(calScores, alpha)
  const threshold = (c: number): number =>
    typeof quantile === 'number' ? quantile : (quantile[c] ?? Number.POSITIVE_INFINITY)
  const predictSet = (x: TensorLike): Tensor => {
    const m = matrixView(proba(x), 'splitConformalClassifier', { nFeatures: k })
    const out = new Float64Array(m.n * k)
    const draws =
      options.include === 'randomized'
        ? Float64Array.from(toFlat(uniform(foldIn(options.key ?? keyFor({}, 'splitConformalClassifier'), 1), [m.n])))
        : undefined
    for (let i = 0; i < m.n; i++) {
      const row = m.data.subarray(i * k, (i + 1) * k)
      if (score === 'lac' || options.classConditional === true) {
        for (let c = 0; c < k; c++) out[i * k + c] = scoreOf(row, c, undefined) - threshold(c) <= EPSILON ? 1 : 0
        continue
      }
      // APS/RAPS: take labels in decreasing probability until the (penalised) mass reaches the threshold, then keep
      // every label at least as likely as the last one (or drop the last at random for randomised sets).
      const order = sortedDesc(row)
      const q = threshold(0)
      let cum = 0
      let last = order.length - 1
      for (let r = 0; r < order.length; r++) {
        cum += row[order[r] ?? 0] ?? 0
        if (cum + lambda * Math.max(0, r + 1 - kReg) - q >= -EPSILON) {
          last = r
          break
        }
      }
      const pLast = row[order[last] ?? 0] ?? 0
      for (let c = 0; c < k; c++) out[i * k + c] = (row[c] ?? 0) - pLast >= -EPSILON ? 1 : 0
      if (draws !== undefined) {
        let mass = 0
        for (let r = 0; r <= last; r++) mass += row[order[r] ?? 0] ?? 0
        const v = (mass + lambda * Math.max(0, last + 1 - kReg) - q) / Math.max(pLast, EPSILON)
        if (v - (draws[i] ?? 0) > EPSILON) out[i * k + (order[last] ?? 0)] = 0
      }
    }
    return fromStorage(out, [m.n, k], 'float64')
  }
  return {
    alpha,
    quantile,
    classes,
    predictSet,
    predictLabels: (x) => {
      const s = toFlat(predictSet(x))
      return Array.from({ length: s.length / k }, (_, i) => classes.filter((_, c) => (s[i * k + c] ?? 0) === 1))
    },
  }
}

/** Options of {@link tuneRaps}. */
export interface TuneRapsOptions {
  readonly alpha?: number
  /** The penalties tried (default Angelopoulos et al.'s 0.001, 0.01, 0.1, 0.2, 0.5). */
  readonly lambdas?: readonly number[]
  /** The share of the calibration rows used for tuning (default 0.2). */
  readonly fraction?: number
  readonly key?: Key
}

/**
 * Choose RAPS's regularisation (Angelopoulos, Bates, Malik and Jordan 2021; MAPIE's RAPS): on a held-out share of the
 * calibration data, kReg is one more than the conformal quantile of the true labels' ranks, and lambda the penalty
 * whose RAPS sets are smallest on average; the rest of the data is left to calibrate with them.
 *
 * @param fitted - A fitted classifier with `predictProba`.
 * @param calibration - Calibration data.
 * @param options - alpha, lambdas, fraction and key.
 * @returns kReg, lambda, and the row indices left for calibration.
 * @example
 * ```ts
 * import { tuneRaps } from 'whitebox-engine/learn'
 * import { asTensor, tensor, toArray } from 'whitebox-engine/tensor'
 * const fixed = { estimator: 'f', kind: 'classifier', params: {}, classes: [0, 1, 2], predictProba: (x) => tensor(toArray(asTensor(x)).map((r) => [r[0], (1 - r[0]) * 0.7, (1 - r[0]) * 0.3])) }
 * const x = Array.from({ length: 50 }, (_, i) => [(i % 10) / 10])
 * const t = tuneRaps(fixed, { x, y: x.map((r, i) => (r[0] > 0.4 ? 0 : i % 2 === 0 ? 1 : 2)) }, { alpha: 0.2 })
 * t.kReg >= 1 // => true
 * ```
 */
export function tuneRaps(
  fitted: Fitted,
  calibration: LearnData,
  options: TuneRapsOptions = {},
): { readonly kReg: number; readonly lambda: number; readonly calibrationRows: readonly number[] } {
  const alpha = fractionOption(options.alpha ?? 0.1, 'alpha', 'tuneRaps')
  const n = rowsOf(calibration.x)
  const nTune = Math.max(1, Math.round((options.fraction ?? 0.2) * n))
  if (nTune >= n) throw new WhiteboxError('value', 'tuneRaps: no calibration rows left after tuning')
  const order = toFlat(permutation(options.key ?? keyFor({}, 'tuneRaps'), n))
  const tuneRows = order.slice(0, nTune)
  const rest = order.slice(nTune).sort((a, b) => a - b)
  const tune = takeData(calibration, tuneRows)
  const proba = requireVerb(fitted, 'predictProba', 'tuneRaps')
  const classes = fitted.classes ?? encodeLabels(requireTargets(calibration, 'tuneRaps'), 'tuneRaps').classes
  const k = classes.length
  const p = matrixView(proba(tune.x), 'tuneRaps', { nFeatures: k })
  const labels = Array.from(targetVector(requireTargets(tune, 'tuneRaps'), 'tuneRaps'), (v) => classes.indexOf(v))
  const ranks = labels.map((c, i) => sortedDesc(p.data.subarray(i * k, (i + 1) * k)).indexOf(c))
  const kReg = Math.min(k, conformalQuantile(ranks, alpha) + 1)
  let best = { lambda: options.lambdas?.[0] ?? 0.001, size: Number.POSITIVE_INFINITY }
  for (const lambda of options.lambdas ?? [0.001, 0.01, 0.1, 0.2, 0.5]) {
    const c = splitConformalClassifier(fitted, tune, { alpha, score: 'raps', lambda, kReg })
    const sets = toFlat(c.predictSet(tune.x))
    const size = sets.reduce((a, b) => a + b, 0) / nTune
    if (size < best.size - 1e-12) best = { lambda, size }
  }
  return { kReg: Number.isFinite(kReg) ? kReg : k, lambda: best.lambda, calibrationRows: rest }
}

/** Options of {@link adaptiveConformalInference}. */
export interface AciOptions {
  /** The target miscoverage (default 0.1). */
  readonly alpha?: number
  /** The step size of the level update (default 0.005). */
  readonly gamma?: number
  /** Scores used for each quantile: all past ones (default) or the last `window`. */
  readonly window?: number
  /** Scores available before the stream starts (warm start; default none). */
  readonly initialScores?: ArrayLike<number>
  readonly tracer?: FitOptions['tracer']
  readonly policy?: FitOptions['policy']
  readonly parent?: FitOptions['parent']
}

/** The trajectory of {@link adaptiveConformalInference}. */
export interface AciResult {
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  /** The working level alpha_t at each time. */
  readonly alphas: readonly number[]
  /** 1 when y_t fell outside its interval. */
  readonly errors: readonly number[]
  /** The realised coverage, 1 - mean(errors). */
  readonly coverage: number
}

/**
 * Adaptive conformal inference for time series and shifting distributions (Gibbs and Candes 2021): at each time,
 * predict yhat_t -/+ the (1 - alpha_t) quantile of past absolute residuals, observe y_t, and update
 * alpha_{t+1} = alpha_t + gamma (alpha - err_t). Without any exchangeability, the long-run error rate stays within
 * (max(alpha_1, 1 - alpha_1) + gamma) / (gamma T) of alpha. A traced algorithm with one step per time.
 *
 * @param predictions - Point forecasts yhat_t, in time order.
 * @param targets - The realised values y_t.
 * @param options - alpha, gamma, window, initialScores and tracing.
 * @returns The intervals, working levels, errors and coverage.
 * @example
 * ```ts
 * import { adaptiveConformalInference } from 'whitebox-engine/learn'
 * const y = Array.from({ length: 200 }, (_, t) => Math.sin(t) * (1 + t / 50))
 * const r = adaptiveConformalInference(y.map(() => 0), y, { alpha: 0.1, gamma: 0.05 })
 * Math.abs(r.coverage - 0.9) < 0.05 // => true
 * ```
 */
export function adaptiveConformalInference(
  predictions: ArrayLike<number>,
  targets: ArrayLike<number>,
  options: AciOptions = {},
): AciResult {
  const alpha = fractionOption(options.alpha ?? 0.1, 'alpha', 'adaptiveConformalInference')
  const gamma = options.gamma ?? 0.005
  if (!(gamma > 0)) throw new WhiteboxError('value', 'adaptiveConformalInference: gamma must be positive')
  const T = predictions.length
  if (targets.length !== T) throw new WhiteboxError('shape', 'adaptiveConformalInference: one target per prediction')
  type State = {
    readonly t: number
    readonly alphaT: number
    readonly scores: readonly number[]
    readonly out: AciResult
  }
  const quantileAt = (scores: readonly number[], level: number): number => {
    if (level <= 0) return Number.POSITIVE_INFINITY
    if (level >= 1 || scores.length === 0) return level >= 1 ? 0 : Number.POSITIVE_INFINITY
    const s = [...scores].sort((a, b) => a - b)
    const kk = Math.ceil((1 - level) * (s.length + 1))
    return kk > s.length ? Number.POSITIVE_INFINITY : (s[kk - 1] ?? 0)
  }
  const algorithm: Algorithm<State> = {
    name: 'adaptive-conformal-inference',
    config: { alpha, gamma, window: options.window ?? null, steps: T },
    schema: channels({
      metrics: {
        alpha: { description: 'the working level alpha_t' },
        coverage: { description: 'the running coverage', direction: 'higher' },
      },
    }),
    init: () => ({
      t: 0,
      alphaT: alpha,
      scores: Array.from(options.initialScores ?? []),
      out: { lower: [], upper: [], alphas: [], errors: [], coverage: Number.NaN },
    }),
    step: (s) => {
      const recent =
        options.window === undefined ? s.scores : s.scores.slice(Math.max(0, s.scores.length - options.window))
      const q = quantileAt(recent, s.alphaT)
      const yhat = predictions[s.t] ?? 0
      const y = targets[s.t] ?? 0
      // A level at or above 1 asks for the empty set (always an error), at or below 0 for the whole line.
      const err = s.alphaT >= 1 || Math.abs(y - yhat) > q ? 1 : 0
      const errors = [...s.out.errors, err]
      return {
        t: s.t + 1,
        alphaT: s.alphaT + gamma * (alpha - err),
        scores: [...s.scores, Math.abs(y - yhat)],
        out: {
          lower: [...s.out.lower, yhat - q],
          upper: [...s.out.upper, yhat + q],
          alphas: [...s.out.alphas, s.alphaT],
          errors,
          coverage: 1 - errors.reduce((a, b) => a + b, 0) / errors.length,
        },
      }
    },
    done: (s) => (s.t >= T ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { alpha: s.alphaT, ...(Number.isNaN(s.out.coverage) ? {} : { coverage: s.out.coverage }) },
    }),
  }
  const run = {
    ...(options.parent === undefined ? {} : { parent: options.parent }),
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  }
  return runAlgorithm(algorithm, { ...run, maxSteps: T }).state.out
}

/**
 * The empirical coverage and mean width of intervals on labelled data.
 *
 * @param intervals - The intervals.
 * @param y - The true values.
 * @returns Coverage (fraction inside) and mean width.
 * @example
 * ```ts
 * import { intervalCoverage } from 'whitebox-engine/learn'
 * import { tensor } from 'whitebox-engine/tensor'
 * intervalCoverage({ prediction: tensor([0, 0]), lower: tensor([-1, -1]), upper: tensor([1, 1]) }, [0.5, 2]) // => { coverage: 0.5, width: 2 }
 * ```
 */
export function intervalCoverage(
  intervals: Intervals,
  y: TensorLike,
): { readonly coverage: number; readonly width: number } {
  const lo = toFlat(intervals.lower)
  const hi = toFlat(intervals.upper)
  const t = targetVector(y, 'intervalCoverage', lo.length)
  let inside = 0
  let width = 0
  t.forEach((v, i) => {
    if (v >= (lo[i] ?? 0) && v <= (hi[i] ?? 0)) inside++
    width += (hi[i] ?? 0) - (lo[i] ?? 0)
  })
  return { coverage: inside / t.length, width: width / t.length }
}
