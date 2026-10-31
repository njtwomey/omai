/**
 * The nuisance-model slot of the causal estimators: a propensity model (the chance of treatment given the covariates)
 * and outcome models (the mean outcome given the covariates, per arm), each any estimator of the library (a logistic or
 * linear model by default, a forest, a boosted ensemble, nearest neighbours), the stratified folds that cross-fitting
 * needs, and the propensity diagnostics: overlap, the effective sample size of the weights and standardised mean
 * differences before and after weighting.
 */

import { MdmlError } from '../base/index.ts'
import type { Estimator, Fitted } from '../learn/estimator.ts'
import { type Fold, splitIndices, stratifiedKFold } from '../learn/model-selection/splitters.ts'
import { linearModel } from '../linear/model.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import { fromStorage, type TensorLike, toFlat } from '../tensor/index.ts'
import { mean, type Prepared, pick, pickRows, prepare, variance } from './_dense.ts'
import type { CausalData } from './data.ts'

/** Any estimator of the library used as a nuisance model: it is fitted on `{ x, y }` and asked for `predict` or `predictProba`. */
export type Learner = Estimator

/** The nuisance models of an estimator. */
export interface NuisanceOptions {
  /** The propensity model (default a logistic regression, `linearModel({ family: 'binomial' })`). */
  readonly propensity?: Learner
  /** The outcome model fitted per arm (default ordinary least squares). */
  readonly outcome?: Learner
  /** Whether the outcome is binary, so a classifier's probability of 1 is its mean (default false). */
  readonly binaryOutcome?: boolean
  /** Propensities are clipped to `[clip, 1 - clip]` (default 0.01). */
  readonly clip?: number
  /** The key for stochastic learners and the folds. */
  readonly key?: Key
}

/**
 * The default propensity model: an unpenalised logistic regression.
 *
 * @returns The estimator (it takes any solver of the GLM slot through `linearModel`).
 * @example
 * ```ts
 * import { defaultPropensity } from 'mdml-engine/causal'
 * defaultPropensity().name // => 'linear-model'
 * ```
 */
export function defaultPropensity(): Learner {
  return linearModel({ family: 'binomial', solver: 'newton' })
}

/**
 * The default outcome model: ordinary least squares.
 *
 * @returns The estimator.
 * @example
 * ```ts
 * import { defaultOutcome } from 'mdml-engine/causal'
 * defaultOutcome().kind // => 'regressor'
 * ```
 */
export function defaultOutcome(): Learner {
  return linearModel({ family: 'gaussian', solver: 'auto' })
}

/** The probability of class 1 (or the mean, for a regressor) of a fitted model at the rows of `x`. */
export function probabilityOf(fitted: Fitted, x: TensorLike): Float64Array {
  if (fitted.predictProba !== undefined) {
    const p = fitted.predictProba(x)
    const k = p.shape[1] ?? 1
    const flat = toFlat(p)
    const classes = fitted.classes ?? [0, 1]
    const col = classes.indexOf(1)
    const n = p.shape[0] ?? 0
    if (col < 0) return new Float64Array(n)
    return Float64Array.from({ length: n }, (_, i) => flat[i * k + col] ?? 0)
  }
  if (fitted.predict === undefined)
    throw new MdmlError('value', `${fitted.estimator}: cannot predict, so it cannot be a nuisance model`)
  return Float64Array.from(toFlat(fitted.predict(x)))
}

/** The mean prediction of a fitted outcome model: its probability of 1 for a binary outcome, else its `predict`. */
export function expectationOf(fitted: Fitted, x: TensorLike, binary: boolean): Float64Array {
  if (binary) return probabilityOf(fitted, x)
  if (fitted.predict === undefined)
    throw new MdmlError('value', `${fitted.estimator}: cannot predict, so it cannot be a nuisance model`)
  return Float64Array.from(toFlat(fitted.predict(x)))
}

const matrixOf = (v: Float64Array, d: number): TensorLike => fromStorage(v, [v.length / d, d], 'float64')

const clipTo = (v: Float64Array, c: number): Float64Array => v.map((p) => Math.min(1 - c, Math.max(c, p)))

/**
 * Fit the propensity model on the training rows and predict (clipped) at the test rows.
 *
 * @param p - The prepared data.
 * @param train - Rows to fit on.
 * @param test - Rows to predict at.
 * @param options - The propensity learner, clipping and key.
 * @param fold - A number folded into the key.
 * @returns The propensity at each test row.
 */
export function fitPropensity(
  p: Prepared,
  train: ArrayLike<number>,
  test: ArrayLike<number>,
  options: NuisanceOptions = {},
  fold = 0,
): Float64Array {
  return propensityModel(p, train, options, fold)(matrixOf(pickRows(p.x, p.d, test), p.d))
}

/**
 * Fit the propensity model on the training rows and return it as a function of covariates (clipped scores).
 *
 * @param p - The prepared data.
 * @param train - Rows to fit on.
 * @param options - The propensity learner, clipping and key.
 * @param fold - A number folded into the key.
 * @returns A function from a covariate matrix to propensity scores.
 */
export function propensityModel(
  p: Prepared,
  train: ArrayLike<number>,
  options: NuisanceOptions = {},
  fold = 0,
): (x: TensorLike) => Float64Array {
  const key = foldIn(options.key ?? makeKey(0), fold)
  const prop = options.propensity ?? defaultPropensity()
  const fitted = prop.fit(
    { x: matrixOf(pickRows(p.x, p.d, train), p.d), y: pick(p.t, train) },
    { key: foldIn(key, 0), explain: false },
  )
  return (x) => clipTo(probabilityOf(fitted, x), options.clip ?? 0.01)
}

/** Nuisance predictions at the test rows of a split. */
export interface Nuisances {
  /** The (clipped) propensity at each test row. */
  readonly e: Float64Array
  /** The outcome model of the controls at each test row. */
  readonly mu0: Float64Array
  /** The outcome model of the treated at each test row. */
  readonly mu1: Float64Array
  /** The pooled outcome model `E[Y | X]` (without the treatment) at each test row, when asked for. */
  readonly m?: Float64Array
}

/**
 * Fit the propensity and per-arm outcome models on the training rows and predict at the test rows (the same rows for no
 * cross-fitting).
 *
 * @param p - The prepared data.
 * @param train - Rows to fit on.
 * @param test - Rows to predict at.
 * @param options - The learners, clipping and key, and `pooled` to also fit `E[Y | X]` on all training rows.
 * @param fold - A number folded into the key so each fold's stochastic learners differ.
 * @returns The propensity and outcome predictions at the test rows.
 */
export function fitNuisances(
  p: Prepared,
  train: ArrayLike<number>,
  test: ArrayLike<number>,
  options: NuisanceOptions & { readonly pooled?: boolean } = {},
  fold = 0,
): Nuisances {
  const key = foldIn(options.key ?? makeKey(0), fold)
  const outc = options.outcome ?? defaultOutcome()
  const xTrain = pickRows(p.x, p.d, train)
  const xTest = matrixOf(pickRows(p.x, p.d, test), p.d)
  const e = fitPropensity(p, train, test, options, fold)
  const arm = (a: number): Float64Array => {
    const rows = Array.from(train).filter((r) => p.t[r] === a)
    if (rows.length < 2)
      throw new MdmlError('value', `the ${a === 1 ? 'treated' : 'control'} arm has fewer than two training units`)
    const fitted = outc.fit(
      { x: matrixOf(pickRows(p.x, p.d, rows), p.d), y: pick(p.y, rows) },
      { key: foldIn(key, 1 + a), explain: false },
    )
    return expectationOf(fitted, xTest, options.binaryOutcome ?? false)
  }
  const m =
    options.pooled === true
      ? expectationOf(
          outc.fit({ x: matrixOf(xTrain, p.d), y: pick(p.y, train) }, { key: foldIn(key, 3), explain: false }),
          xTest,
          options.binaryOutcome ?? false,
        )
      : undefined
  return { e, mu0: arm(0), mu1: arm(1), ...(m === undefined ? {} : { m }) }
}

/** The rows `0..n-1` as an array. */
export function allRows(n: number): Int32Array {
  return Int32Array.from({ length: n }, (_, i) => i)
}

/**
 * Stratified folds of the units on their treatment, so every fold has both arms: the folds cross-fitting uses.
 *
 * @param treatment - The 0/1 treatment.
 * @param folds - The number of folds (default 5).
 * @param key - The key (the folds are not shuffled; the key is accepted for the splitter contract).
 * @returns The folds, each with training and test indices.
 * @example
 * ```ts
 * import { crossFitFolds } from 'mdml-engine/causal'
 * crossFitFolds([0, 1, 0, 1, 0, 1, 0, 1], 2).map((f) => f.test.length) // => [4, 4]
 * ```
 */
export function crossFitFolds(treatment: ArrayLike<number>, folds = 5, key: Key = makeKey(0)): Fold[] {
  return splitIndices(stratifiedKFold(folds, { shuffle: true }), { n: treatment.length, y: treatment }, key)
}

/**
 * The fitted propensity scores of every unit, from one model fitted on all of them.
 *
 * @param data - Covariates, treatment and outcome (the outcome is ignored).
 * @param options - The propensity learner, clipping and key.
 * @returns One propensity per unit.
 * @example
 * ```ts
 * import { propensityScores } from 'mdml-engine/causal'
 * const x = [[-2], [-1], [-0.5], [0], [0.3], [0.5], [1], [2]]
 * const e = propensityScores({ x, treatment: [0, 0, 1, 0, 1, 0, 1, 1], outcome: [0, 0, 0, 0, 0, 0, 0, 0] })
 * e.length // => 8
 * ```
 */
export function propensityScores(data: CausalData, options: NuisanceOptions = {}): Float64Array {
  const p = prepare(data.x, data.treatment, data.outcome, 'propensityScores')
  const rows = allRows(p.n)
  return fitPropensity(p, rows, rows, options)
}

/** The overlap of a propensity model: how far the scores of the two arms are from 0 and 1 and from each other. */
export interface Overlap {
  /** Smallest and largest score of the treated. */
  readonly treated: readonly [number, number]
  /** Smallest and largest score of the controls. */
  readonly control: readonly [number, number]
  /** The interval on which both arms have units: `[max of the minima, min of the maxima]`. */
  readonly common: readonly [number, number]
  /** The share of units whose score lies in the common support. */
  readonly share: number
  /** The shares of treated and of control units with a score outside `[threshold, 1 - threshold]`. */
  readonly extreme: readonly [number, number]
}

/**
 * Describe the overlap of the propensity scores: the range of each arm, the common support and the units outside it.
 *
 * @param scores - Propensity scores.
 * @param treatment - The 0/1 treatment.
 * @param threshold - What counts as extreme (default 0.05).
 * @returns The overlap summary.
 * @example
 * ```ts
 * import { overlap } from 'mdml-engine/causal'
 * overlap([0.2, 0.5, 0.4, 0.7], [0, 0, 1, 1]).common // => [0.4, 0.5]
 * ```
 */
export function overlap(scores: ArrayLike<number>, treatment: ArrayLike<number>, threshold = 0.05): Overlap {
  const range = (a: number): [number, number] => {
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (let i = 0; i < scores.length; i++)
      if (treatment[i] === a) {
        lo = Math.min(lo, scores[i] ?? 0)
        hi = Math.max(hi, scores[i] ?? 0)
      }
    return [lo, hi]
  }
  const treated = range(1)
  const control = range(0)
  const common: [number, number] = [Math.max(treated[0], control[0]), Math.min(treated[1], control[1])]
  let inside = 0
  const ext = [0, 0]
  const counts = [0, 0]
  for (let i = 0; i < scores.length; i++) {
    const s = scores[i] ?? 0
    if (s >= common[0] && s <= common[1]) inside++
    const a = treatment[i] === 1 ? 1 : 0
    counts[a] = (counts[a] ?? 0) + 1
    if (s < threshold || s > 1 - threshold) ext[a] = (ext[a] ?? 0) + 1
  }
  return {
    treated,
    control,
    common,
    share: inside / Math.max(1, scores.length),
    extreme: [(ext[1] ?? 0) / Math.max(1, counts[1] ?? 0), (ext[0] ?? 0) / Math.max(1, counts[0] ?? 0)],
  }
}

/**
 * The Kish effective sample size of a set of weights, `(sum w)^2 / sum w^2`: how many equally weighted units would carry
 * the same information.
 *
 * @param weights - Non-negative weights.
 * @returns The effective sample size.
 * @example
 * ```ts
 * import { effectiveSampleSize } from 'mdml-engine/causal'
 * [effectiveSampleSize([1, 1, 1, 1]), effectiveSampleSize([4, 0, 0, 0])] // => [4, 1]
 * ```
 */
export function effectiveSampleSize(weights: ArrayLike<number>): number {
  let s = 0
  let s2 = 0
  for (let i = 0; i < weights.length; i++) {
    s += weights[i] ?? 0
    s2 += (weights[i] ?? 0) ** 2
  }
  return s2 === 0 ? 0 : (s * s) / s2
}

/** One row of a balance table. */
export interface BalanceRow {
  readonly covariate: number
  /** Standardised mean difference (treated minus control over the pooled standard deviation) before weighting. */
  readonly before: number
  /** The same after weighting. */
  readonly after: number
}

/**
 * Standardised mean differences of every covariate before and after weighting: `(mean_treated - mean_control) / s`,
 * with `s` the square root of the average of the two arms' unweighted variances (so the denominator does not move with
 * the weights), and weighted means after.
 *
 * @param x - Covariates `[n, d]`.
 * @param treatment - The 0/1 treatment.
 * @param weights - Unit weights (default all one).
 * @returns One row per covariate.
 * @example
 * ```ts
 * import { balanceTable } from 'mdml-engine/causal'
 * balanceTable([[0], [1], [2], [3]], [0, 0, 1, 1])[0].before // => 2.8284271247461903
 * ```
 */
export function balanceTable(x: TensorLike, treatment: TensorLike, weights?: ArrayLike<number>): BalanceRow[] {
  const p = prepare(x, treatment, treatment, 'balanceTable')
  const rows: BalanceRow[] = []
  for (let j = 0; j < p.d; j++) {
    const g = [0, 1].map((a) => {
      const idx = Array.from({ length: p.n }, (_, i) => i).filter((i) => p.t[i] === a)
      const v = idx.map((i) => p.x[i * p.d + j] ?? 0)
      const w = idx.map((i) => weights?.[i] ?? 1)
      const sw = w.reduce((s, b) => s + b, 0)
      return { m: mean(v), v: variance(v, 1), wm: v.reduce((s, b, k) => s + b * (w[k] ?? 0), 0) / sw }
    })
    const s = Math.sqrt(((g[0]?.v ?? 0) + (g[1]?.v ?? 0)) / 2)
    rows.push({
      covariate: j,
      before: s === 0 ? 0 : ((g[1]?.m ?? 0) - (g[0]?.m ?? 0)) / s,
      after: s === 0 ? 0 : ((g[1]?.wm ?? 0) - (g[0]?.wm ?? 0)) / s,
    })
  }
  return rows
}
