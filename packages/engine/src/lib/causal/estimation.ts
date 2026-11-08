/**
 * Estimators of one average treatment effect from observational data with a binary treatment, each against the same
 * `CausalData`: regression adjustment (a linear model with the treatment, Lin's interacted model, or an outcome model
 * of any learner per arm), inverse probability weighting (Horvitz-Thompson or Hajek, stabilised weights, trimming) on
 * a propensity model of any learner, stratification on the propensity score, the doubly robust AIPW estimator,
 * targeted maximum likelihood (TMLE, its fluctuation a smooth problem for the solver slot), and double machine
 * learning with cross-fitting as an algorithm that takes one fold per step. Effects come with standard errors from the
 * influence values of the estimator, or from the bootstrap where no influence is available.
 */

import { type Estimate, estimate, maxOf, minOf, OmaiError } from '../base/index.ts'
import { analyticObjective, type SmoothProblem } from '../optim/index.ts'
import { bootstrapIndices, foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { covariance, expit, logit, mean, type Prepared, pick, prepare, takePrepared, variance, wls } from './_dense.ts'
import { newtonSolver, solveSmooth } from './_solve.ts'
import { type CausalData, type EffectEstimate, effectFromInfluence, effectFromSe } from './data.ts'
import {
  allRows,
  crossFitFolds,
  effectiveSampleSize,
  fitNuisances,
  fitPropensity,
  type NuisanceOptions,
  type Nuisances,
} from './nuisance.ts'

/** The averages an estimator can target: all units, the treated, or the controls. */
export type AverageEstimand = 'ate' | 'att' | 'atc'

/** Options every estimator here accepts. */
export interface EffectOptions extends NuisanceOptions {
  /** The coverage of the interval (default 0.95). */
  readonly level?: number
}

const causalOf = (p: Prepared): CausalData => ({
  x: fromStorage(p.x, [p.n, p.d], 'float64'),
  treatment: fromStorage(p.t, [p.n], 'float64'),
  outcome: fromStorage(p.y, [p.n], 'float64'),
})

/**
 * Inverse probability weights of the units for an estimand: the treated get `1/e` and the controls `1/(1-e)` for the
 * ATE, the treated 1 and the controls the odds `e/(1-e)` for the ATT, and the mirror for the ATC. Stabilised weights
 * multiply by the marginal probability of the unit's arm (the ATE) or by the inverse marginal odds (the others), so
 * they average about one.
 *
 * @param treatment - The 0/1 treatment.
 * @param scores - Propensity scores.
 * @param options - `estimand` (default `'ate'`) and `stabilised` (default false).
 * @returns One weight per unit.
 * @example
 * ```ts
 * import { ipwWeights } from 'omai-engine/causal'
 * Array.from(ipwWeights([1, 0], [0.5, 0.25])) // => [2, 1.3333333333333333]
 * ```
 */
export function ipwWeights(
  treatment: ArrayLike<number>,
  scores: ArrayLike<number>,
  options: { readonly estimand?: AverageEstimand; readonly stabilised?: boolean } = {},
): Float64Array {
  const estimand = options.estimand ?? 'ate'
  const n = treatment.length
  const pbar = mean(treatment)
  return Float64Array.from({ length: n }, (_, i) => {
    const e = scores[i] ?? 0.5
    const treated = treatment[i] === 1
    let w: number
    if (estimand === 'ate') w = treated ? 1 / e : 1 / (1 - e)
    else if (estimand === 'att') w = treated ? 1 : e / (1 - e)
    else w = treated ? (1 - e) / e : 1
    if (options.stabilised !== true) return w
    if (estimand === 'ate') return w * (treated ? pbar : 1 - pbar)
    if (estimand === 'att') return treated ? w : (w * (1 - pbar)) / pbar
    return treated ? (w * pbar) / (1 - pbar) : w
  })
}

/** An effect with the weights that produced it. */
export interface WeightedEffect extends EffectEstimate {
  /** The weight of every unit (zero for trimmed units). */
  readonly weights: Float64Array
  /** The propensity score of every unit. */
  readonly propensity: Float64Array
  /** The Kish effective sample size of the weights. */
  readonly ess: number
  /** The number of units kept after trimming. */
  readonly kept: number
}

/** Options of {@link ipw}. */
export interface IpwOptions extends EffectOptions {
  readonly estimand?: AverageEstimand
  /** Divide each arm's weighted sum by the sum of its weights (Hajek, the default) rather than by n (Horvitz-Thompson). */
  readonly normalised?: boolean
  /** Return stabilised weights (the Hajek estimate does not change). */
  readonly stabilised?: boolean
  /** Drop units whose propensity is outside `[trim, 1 - trim]` (default 0: none). */
  readonly trim?: number
  /** Known propensity scores, used instead of fitting a model. */
  readonly scores?: ArrayLike<number>
}

function ipwCore(
  t: ArrayLike<number>,
  y: ArrayLike<number>,
  w: ArrayLike<number>,
  estimand: AverageEstimand,
  normalised: boolean,
): { readonly value: number; readonly influence: Float64Array } {
  const n = t.length
  const influence = new Float64Array(n)
  if (normalised) {
    let s1 = 0
    let s0 = 0
    let a1 = 0
    let a0 = 0
    for (let i = 0; i < n; i++) {
      const wi = w[i] ?? 0
      if (t[i] === 1) {
        s1 += wi
        a1 += wi * (y[i] ?? 0)
      } else {
        s0 += wi
        a0 += wi * (y[i] ?? 0)
      }
    }
    const m1 = a1 / s1
    const m0 = a0 / s0
    for (let i = 0; i < n; i++)
      influence[i] =
        t[i] === 1 ? (n * (w[i] ?? 0) * ((y[i] ?? 0) - m1)) / s1 : (-n * (w[i] ?? 0) * ((y[i] ?? 0) - m0)) / s0
    return { value: m1 - m0, influence }
  }
  let total = 0
  let big = 0
  for (let i = 0; i < n; i++) {
    total += (t[i] === 1 ? 1 : -1) * (w[i] ?? 0) * (y[i] ?? 0)
    big += estimand === 'ate' ? 1 : (t[i] === 1) === (estimand === 'att') ? 1 : 0
  }
  const value = total / big
  for (let i = 0; i < n; i++) {
    const g = estimand === 'ate' ? 1 : (t[i] === 1) === (estimand === 'att') ? 1 : 0
    influence[i] = (n / big) * ((t[i] === 1 ? 1 : -1) * (w[i] ?? 0) * (y[i] ?? 0) - value * g)
  }
  return { value, influence }
}

/**
 * Inverse probability weighting: weight each unit by the inverse of the probability of the treatment it received so the
 * weighted arms resemble the target population, then take the difference of weighted outcome means. The propensity is
 * fitted by any learner (a logistic regression by default). The standard error treats the scores as known, so it is
 * conservative when they are estimated.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - The propensity learner, estimand, Hajek or Horvitz-Thompson, trimming and stabilised weights.
 * @returns The effect with the weights, the propensity scores and the effective sample size.
 * @example
 * ```ts
 * import { ipw } from 'omai-engine/causal'
 * const x = [[0], [0], [0], [0], [1], [1], [1], [1]]
 * const r = ipw({ x, treatment: [1, 0, 0, 0, 1, 1, 1, 0], outcome: [3, 1, 1, 1, 5, 5, 5, 3] }, { scores: [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75] })
 * r.estimate.value // => 2
 * ```
 */
export function ipw(data: CausalData, options: IpwOptions = {}): WeightedEffect {
  const p = prepare(data.x, data.treatment, data.outcome, 'ipw')
  const estimand = options.estimand ?? 'ate'
  const scores = propensityOf(p, options)
  const trim = options.trim ?? 0
  const keep = allRows(p.n).filter((i) => (scores[i] ?? 0) >= trim && (scores[i] ?? 0) <= 1 - trim)
  const t = pick(p.t, keep)
  const y = pick(p.y, keep)
  const e = pick(scores, keep)
  const w = ipwWeights(t, e, { estimand })
  const core = ipwCore(t, y, w, estimand, options.normalised ?? true)
  const effect = effectFromInfluence(estimand, core.value, core.influence, {
    ...(options.level === undefined ? {} : { level: options.level }),
    method: `inverse probability weighting (${options.normalised === false ? 'Horvitz-Thompson' : 'Hajek'})`,
  })
  const full = new Float64Array(p.n)
  const shown = options.stabilised === true ? ipwWeights(t, e, { estimand, stabilised: true }) : w
  keep.forEach((row, k) => {
    full[row] = shown[k] ?? 0
  })
  return { ...effect, weights: full, propensity: scores, ess: effectiveSampleSize(w), kept: keep.length }
}

function propensityOf(p: Prepared, options: NuisanceOptions & { readonly scores?: ArrayLike<number> }): Float64Array {
  if (options.scores !== undefined) return Float64Array.from(options.scores)
  const rows = allRows(p.n)
  return fitPropensity(p, rows, rows, options)
}

/**
 * The estimate of a statistic over bootstrap resamples of the units: the percentile spread as a standard error with a
 * normal interval. Use it where an estimator has no influence values.
 *
 * @param data - The causal data.
 * @param statistic - An estimator of one number from causal data.
 * @param options - `resamples` (default 100), `key`, `level`.
 * @returns The point estimate on the full data with the bootstrap standard error and interval.
 * @example
 * ```ts
 * import { bootstrapEffect } from 'omai-engine/causal'
 * import { toFlat } from 'omai-engine/tensor'
 * const d = { x: [[0], [1], [2], [3]], treatment: [0, 1, 0, 1], outcome: [1, 3, 2, 4] }
 * const diff = (c: typeof d): number => {
 *   const t = toFlat(c.treatment), y = toFlat(c.outcome)
 *   let a = 0, b = 0, n1 = 0, n0 = 0
 *   t.forEach((v, i) => (v === 1 ? ((a += y[i]), n1++) : ((b += y[i]), n0++)))
 *   return a / n1 - b / n0
 * }
 * bootstrapEffect(d, diff, { resamples: 20 }).value // => 2
 * ```
 */
export function bootstrapEffect(
  data: CausalData,
  statistic: (data: CausalData) => number,
  options: { readonly resamples?: number; readonly key?: Key; readonly level?: number } = {},
): Estimate {
  const p = prepare(data.x, data.treatment, data.outcome, 'bootstrapEffect')
  const key = options.key ?? makeKey(0)
  const b = options.resamples ?? 100
  const values: number[] = []
  for (let r = 0; r < b; r++) {
    const rows = toFlat(bootstrapIndices(foldIn(key, r), p.n))
    try {
      const v = statistic(causalOf(takePrepared(p, rows)))
      if (Number.isFinite(v)) values.push(v)
    } catch (error) {
      // a resample without one of the arms has no estimate; it is left out
      if (!(error instanceof OmaiError)) throw error
    }
  }
  const value = statistic(causalOf(p))
  const est = estimate(value, Math.sqrt(variance(values, 1)), options.level ?? 0.95)
  return { ...est, method: `bootstrap (${values.length} resamples)` }
}

/** Options of {@link regressionAdjustment}. */
export interface RegressionAdjustmentOptions extends EffectOptions {
  /**
   * `'linear'`: ordinary least squares of the outcome on the treatment and covariates (a constant effect); `'interacted'`:
   * Lin's model with the covariates centred and interacted with the treatment, the coefficient on the treatment the
   * average effect of the estimand; `'g-computation'`: the learner is fitted in each arm and its predictions averaged
   * (any learner; the standard error is the bootstrap's).
   */
  readonly mode?: 'linear' | 'interacted' | 'g-computation'
  readonly estimand?: AverageEstimand
  /** Bootstrap resamples for `'g-computation'` (default 100). */
  readonly bootstrap?: number
}

/**
 * Regression adjustment: model the outcome given the treatment and covariates and read the effect off the model. The
 * `'linear'` mode is the coefficient of the treatment in a least-squares fit (HC1 standard error), `'interacted'`
 * adds the covariate-by-treatment interactions (Lin 2013), and `'g-computation'` fits the outcome learner in each arm
 * and averages the difference of its predictions over the units of the estimand.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - The mode, estimand, outcome learner and bootstrap size.
 * @returns The effect.
 * @example
 * ```ts
 * import { regressionAdjustment } from 'omai-engine/causal'
 * const x = [[0], [1], [2], [3], [4], [5]]
 * const r = regressionAdjustment({ x, treatment: [0, 1, 0, 1, 0, 1], outcome: [0, 3, 2, 5, 4, 7] }, { mode: 'linear' })
 * r.estimate.value // => 2
 * ```
 */
export function regressionAdjustment(data: CausalData, options: RegressionAdjustmentOptions = {}): EffectEstimate {
  const mode = options.mode ?? 'linear'
  const estimand = options.estimand ?? 'ate'
  const p = prepare(data.x, data.treatment, data.outcome, 'regressionAdjustment')
  const level = options.level ?? 0.95
  if (mode === 'g-computation') {
    const run = (q: Prepared): number => {
      const rows = allRows(q.n)
      const nu = fitNuisances(q, rows, rows, options)
      let s = 0
      let c = 0
      for (let i = 0; i < q.n; i++) {
        const inTarget = estimand === 'ate' || (estimand === 'att') === (q.t[i] === 1)
        if (!inTarget) continue
        s += (nu.mu1[i] ?? 0) - (nu.mu0[i] ?? 0)
        c++
      }
      return s / c
    }
    const value = run(p)
    const est = bootstrapEffect(data, (c) => run(prepare(c.x, c.treatment, c.outcome, 'regressionAdjustment')), {
      resamples: options.bootstrap ?? 100,
      key: options.key ?? makeKey(0),
      level,
    })
    return {
      ...effectFromSe(estimand, value, est.standardError ?? Number.NaN, p.n, { level, method: 'g-computation' }),
      n: p.n,
    }
  }
  let design: Float64Array
  let width: number
  const at = 1
  if (mode === 'linear') {
    // columns: intercept, treatment, covariates
    width = p.d + 2
    design = new Float64Array(p.n * width)
    for (let i = 0; i < p.n; i++) {
      design[i * width] = 1
      design[i * width + 1] = p.t[i] ?? 0
      for (let j = 0; j < p.d; j++) design[i * width + 2 + j] = p.x[i * p.d + j] ?? 0
    }
  } else {
    // Lin: [1, t, xc, t * xc] with the covariates centred at the estimand's mean.
    const centre = new Float64Array(p.d)
    let count = 0
    for (let i = 0; i < p.n; i++) {
      if (estimand !== 'ate' && (estimand === 'att') !== (p.t[i] === 1)) continue
      count++
      for (let j = 0; j < p.d; j++) centre[j] = (centre[j] ?? 0) + (p.x[i * p.d + j] ?? 0)
    }
    for (let j = 0; j < p.d; j++) centre[j] = (centre[j] ?? 0) / count
    width = 2 + 2 * p.d
    design = new Float64Array(p.n * width)
    for (let i = 0; i < p.n; i++) {
      design[i * width] = 1
      design[i * width + 1] = p.t[i] ?? 0
      for (let j = 0; j < p.d; j++) {
        const c = (p.x[i * p.d + j] ?? 0) - (centre[j] ?? 0)
        design[i * width + 2 + j] = c
        design[i * width + 2 + p.d + j] = (p.t[i] ?? 0) * c
      }
    }
  }
  const fit = wls(design, p.n, width, p.y)
  const cov = covariance(fit, design, 'HC1')
  const se = Math.sqrt(cov[at * width + at] ?? Number.NaN)
  return effectFromSe(mode === 'linear' ? 'ate' : estimand, fit.beta[at] ?? Number.NaN, se, p.n, {
    level,
    method: mode === 'linear' ? 'linear regression adjustment (HC1)' : "Lin's interacted regression adjustment (HC1)",
  })
}

/** A stratum of {@link stratification}. */
export interface Stratum {
  readonly size: number
  readonly treated: number
  readonly propensity: readonly [number, number]
  /** The difference of the arms' mean outcomes (NaN when an arm is missing, and the stratum is then left out). */
  readonly effect: number
}

/** Options of {@link stratification}. */
export interface StratificationOptions extends EffectOptions {
  /** Known propensity scores, used instead of fitting a model. */
  readonly scores?: ArrayLike<number>
  /** The number of strata, equal-sized in the propensity score (default 5). */
  readonly strata?: number
  readonly estimand?: 'ate' | 'att'
}

/**
 * Stratification on the propensity score: sort the units by score, cut them into equal strata, take the difference of
 * arm means in each, and average the strata by size (the ATE) or by treated count (the ATT). Strata missing an arm are
 * left out.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - The number of strata, estimand and propensity learner.
 * @returns The effect and the strata.
 * @example
 * ```ts
 * import { stratification } from 'omai-engine/causal'
 * const x = [[0], [0], [0], [0], [1], [1], [1], [1]]
 * const r = stratification(
 *   { x, treatment: [0, 1, 0, 1, 0, 1, 0, 1], outcome: [0, 2, 0, 2, 5, 7, 5, 7] },
 *   { strata: 2, scores: [0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8] },
 * )
 * r.estimate.value // => 2
 * ```
 */
export function stratification(
  data: CausalData,
  options: StratificationOptions = {},
): EffectEstimate & { readonly strata: readonly Stratum[] } {
  const p = prepare(data.x, data.treatment, data.outcome, 'stratification')
  const k = options.strata ?? 5
  const estimand = options.estimand ?? 'ate'
  const e = propensityOf(p, options)
  const order = Array.from(allRows(p.n)).sort((a, b) => (e[a] ?? 0) - (e[b] ?? 0))
  const strata: Stratum[] = []
  const used: { w: number; effect: number; v: number }[] = []
  let totalW = 0
  for (let s = 0; s < k; s++) {
    const rows = order.slice(Math.floor((s * p.n) / k), Math.floor(((s + 1) * p.n) / k))
    const y1 = rows.filter((r) => p.t[r] === 1).map((r) => p.y[r] ?? 0)
    const y0 = rows.filter((r) => p.t[r] === 0).map((r) => p.y[r] ?? 0)
    const effect = y1.length > 0 && y0.length > 0 ? mean(y1) - mean(y0) : Number.NaN
    strata.push({
      size: rows.length,
      treated: y1.length,
      propensity: [e[rows[0] ?? 0] ?? 0, e[rows[rows.length - 1] ?? 0] ?? 0],
      effect,
    })
    if (Number.isNaN(effect)) continue
    const w = estimand === 'ate' ? rows.length : y1.length
    totalW += w
    used.push({
      w,
      effect,
      v: (y1.length > 1 ? variance(y1, 1) / y1.length : 0) + (y0.length > 1 ? variance(y0, 1) / y0.length : 0),
    })
  }
  if (used.length === 0) throw new OmaiError('value', 'stratification: no stratum has both arms')
  const value = used.reduce((a, u) => a + (u.w / totalW) * u.effect, 0)
  const se = Math.sqrt(used.reduce((a, u) => a + (u.w / totalW) ** 2 * u.v, 0))
  const eff = effectFromSe(estimand, value, se, p.n, {
    ...(options.level === undefined ? {} : { level: options.level }),
    method: `stratification on the propensity score (${k} strata)`,
  })
  return { ...eff, strata }
}

/** Options of {@link aipw}. */
export interface AipwOptions extends EffectOptions {
  readonly estimand?: AverageEstimand
  /** Cross-fit the nuisance models over this many folds (default 0: fit and predict on all units). */
  readonly folds?: number
  /** Drop units whose propensity is outside `[trim, 1 - trim]`. */
  readonly trim?: number
}

/**
 * The nuisance predictions at every unit: fitted on all units, or cross-fitted over `folds` stratified folds.
 *
 * @param p - The prepared data.
 * @param options - The learners, `folds`, `pooled` and key.
 * @returns The propensity and outcome predictions per unit.
 * @example
 * ```ts
 * import { nuisancesFor } from 'omai-engine/causal'
 * const p = { n: 8, d: 1, x: Float64Array.from([-2, -1, -0.5, 0, 0.3, 0.5, 1, 2]), t: Float64Array.from([0, 0, 1, 0, 1, 0, 1, 1]), y: Float64Array.from([0, 1, 2, 1, 3, 2, 4, 5]) }
 * nuisancesFor(p, {}).e.length // => 8
 * ```
 */
export function nuisancesFor(
  p: Prepared,
  options: NuisanceOptions & { readonly folds?: number; readonly pooled?: boolean },
): Nuisances {
  const folds = options.folds ?? 0
  if (folds < 2) return fitNuisances(p, allRows(p.n), allRows(p.n), options)
  const e = new Float64Array(p.n)
  const mu0 = new Float64Array(p.n)
  const mu1 = new Float64Array(p.n)
  const m = options.pooled === true ? new Float64Array(p.n) : undefined
  crossFitFolds(p.t, folds, options.key ?? makeKey(0)).forEach((f, i) => {
    const nu = fitNuisances(p, f.train, f.test, options, i)
    f.test.forEach((row, k) => {
      e[row] = nu.e[k] ?? 0
      mu0[row] = nu.mu0[k] ?? 0
      mu1[row] = nu.mu1[k] ?? 0
      if (m !== undefined) m[row] = nu.m?.[k] ?? 0
    })
  })
  return { e, mu0, mu1, ...(m === undefined ? {} : { m }) }
}

/**
 * The AIPW estimate and its influence values from given nuisance predictions.
 *
 * @param p - Treatment and outcome of each unit.
 * @param nu - Propensity and outcome predictions per unit.
 * @param estimand - The ATE, ATT or ATC.
 * @returns The value and the per-unit influence values.
 * @example
 * ```ts
 * import { aipwFrom } from 'omai-engine/causal'
 * const nu = { e: Float64Array.of(0.5, 0.5), mu0: Float64Array.of(1, 1), mu1: Float64Array.of(3, 3) }
 * aipwFrom({ n: 2, t: Float64Array.of(1, 0), y: Float64Array.of(3, 1) }, nu, 'ate').value // => 2
 * ```
 */
export function aipwFrom(
  p: Pick<Prepared, 'n' | 't' | 'y'>,
  nu: Nuisances,
  estimand: AverageEstimand,
): { readonly value: number; readonly influence: Float64Array } {
  const n = p.n
  const influence = new Float64Array(n)
  if (estimand === 'ate') {
    const gamma = Float64Array.from({ length: n }, (_, i) => {
      const e = nu.e[i] ?? 0.5
      const m1 = nu.mu1[i] ?? 0
      const m0 = nu.mu0[i] ?? 0
      return (
        m1 - m0 + ((p.t[i] ?? 0) * ((p.y[i] ?? 0) - m1)) / e - ((1 - (p.t[i] ?? 0)) * ((p.y[i] ?? 0) - m0)) / (1 - e)
      )
    })
    const value = mean(gamma)
    for (let i = 0; i < n; i++) influence[i] = (gamma[i] ?? 0) - value
    return { value, influence }
  }
  // ATT: the treated are the target; the ATC is the mirror image.
  const flip = estimand === 'atc'
  const t = flip ? Float64Array.from(p.t, (v) => 1 - v) : p.t
  const e = flip ? Float64Array.from(nu.e, (v) => 1 - v) : nu.e
  const mu0 = flip ? nu.mu1 : nu.mu0
  const n1 = t.reduce((a, b) => a + b, 0)
  let num = 0
  for (let i = 0; i < n; i++) {
    const ei = e[i] ?? 0.5
    const r = (p.y[i] ?? 0) - (mu0[i] ?? 0)
    num += (t[i] ?? 0) * r - (1 - (t[i] ?? 0)) * (ei / (1 - ei)) * r
  }
  const value = num / n1
  const pbar = n1 / n
  for (let i = 0; i < n; i++) {
    const ei = e[i] ?? 0.5
    const r = (p.y[i] ?? 0) - (mu0[i] ?? 0)
    influence[i] = ((t[i] ?? 0) * (r - value) - (1 - (t[i] ?? 0)) * (ei / (1 - ei)) * r) / pbar
  }
  return flip ? { value: -value, influence: influence.map((v) => -v) } : { value, influence }
}

/**
 * Augmented inverse probability weighting, the doubly robust estimator: an outcome model's prediction of each unit's
 * effect corrected by the inverse-probability-weighted residuals of the observed outcomes. It is consistent when either
 * the propensity model or the outcome models are right, and with cross-fitting (`folds`) it tolerates flexible learners.
 * The standard error is from the efficient influence function.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - Propensity and outcome learners, estimand, folds and trimming.
 * @returns The effect.
 * @index function Augmented inverse-probability-weighted treatment effect.
 * @example
 * ```ts
 * import { aipw } from 'omai-engine/causal'
 * const x = [[0], [1], [2], [3], [4], [5], [6], [7]]
 * const r = aipw({ x, treatment: [0, 1, 0, 1, 1, 0, 1, 0], outcome: [0, 3, 2, 5, 6, 5, 8, 7] })
 * r.estimand // => 'ate'
 * ```
 */
export function aipw(data: CausalData, options: AipwOptions = {}): EffectEstimate {
  const p = prepare(data.x, data.treatment, data.outcome, 'aipw')
  const estimand = options.estimand ?? 'ate'
  const nu = nuisancesFor(p, options)
  const trim = options.trim ?? 0
  const keep = allRows(p.n).filter((i) => (nu.e[i] ?? 0) >= trim && (nu.e[i] ?? 0) <= 1 - trim)
  const sub = {
    n: keep.length,
    t: pick(p.t, keep),
    y: pick(p.y, keep),
  }
  const r = aipwFrom(sub, { e: pick(nu.e, keep), mu0: pick(nu.mu0, keep), mu1: pick(nu.mu1, keep) }, estimand)
  return effectFromInfluence(estimand, r.value, r.influence, {
    ...(options.level === undefined ? {} : { level: options.level }),
    method: `augmented inverse probability weighting${(options.folds ?? 0) >= 2 ? ` (${options.folds}-fold cross-fitting)` : ''}`,
  })
}

/** Options of {@link tmleFluctuationProblem}. */
export interface Fluctuation {
  /** The clever covariate per unit. */
  readonly h: ArrayLike<number>
  /** The initial prediction of each unit's observed arm on the logit scale. */
  readonly offset: ArrayLike<number>
  /** The outcome scaled into `[0, 1]`. */
  readonly y: ArrayLike<number>
}

/**
 * The logistic fluctuation of targeted maximum likelihood as a smooth problem for the solver slot: the negative
 * quasi-log-likelihood of `Y` in `expit(offset + epsilon H)` as a function of the one parameter `epsilon`, with its
 * gradient `sum H (Q* - Y)` and Hessian `sum H^2 Q* (1 - Q*)` (averaged over the units).
 *
 * @param f - The clever covariate, the offset and the scaled outcomes.
 * @returns The objective and the start `epsilon = 0`.
 * @example
 * ```ts
 * import { tmleFluctuationProblem } from 'omai-engine/causal'
 * tmleFluctuationProblem({ h: [1, -1], offset: [0, 0], y: [1, 0] }).x0 // => [0]
 * ```
 */
export function tmleFluctuationProblem(f: Fluctuation): SmoothProblem {
  const n = f.h.length
  const q = (eps: number, i: number): number => expit((f.offset[i] ?? 0) + eps * (f.h[i] ?? 0))
  return {
    objective: analyticObjective({
      value: (x) => {
        let s = 0
        for (let i = 0; i < n; i++) {
          const eta = (f.offset[i] ?? 0) + (x[0] ?? 0) * (f.h[i] ?? 0)
          s += Math.log1p(Math.exp(eta)) - (f.y[i] ?? 0) * eta
        }
        return s / n
      },
      gradient: (x) => {
        let g = 0
        for (let i = 0; i < n; i++) g += (f.h[i] ?? 0) * (q(x[0] ?? 0, i) - (f.y[i] ?? 0))
        return [g / n]
      },
      hessian: (x) => {
        let h = 0
        for (let i = 0; i < n; i++) {
          const m = q(x[0] ?? 0, i)
          h += (f.h[i] ?? 0) ** 2 * m * (1 - m)
        }
        return [h / n]
      },
    }),
    x0: [0],
  }
}

/** Options of {@link tmle}. */
export interface TmleOptions extends EffectOptions {
  /** The fluctuation: `'logistic'` on the outcome scaled to `[0, 1]` (default) or `'linear'` (a least-squares step). */
  readonly fluctuation?: 'logistic' | 'linear'
  /** A solver of the slot for the logistic fluctuation (default Newton). */
  readonly solver?: SpaceValue
  /** Cross-fitting folds for the initial models (default 0: none). */
  readonly folds?: number
}

/** A TMLE estimate. */
export interface TmleResult extends EffectEstimate {
  /** The fluctuation parameter. */
  readonly epsilon: number
  /** The plug-in (g-computation) estimate before the targeting step. */
  readonly initial: number
}

/**
 * Targeted maximum likelihood estimation of the ATE (van der Laan and Rose): start from outcome models and a
 * propensity model, then fluctuate the outcome predictions along the clever covariate `T/e - (1-T)/(1-e)` by the one
 * parameter that solves the efficient score equation, and average the targeted predictions. The result is doubly
 * robust and its standard error is the efficient influence function's. The logistic fluctuation is a smooth problem for
 * the solver slot (Newton by default; any solver of `smoothSolvers` reaches the same epsilon).
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - Learners, fluctuation, solver and folds.
 * @returns The effect with the fluctuation parameter and the untargeted estimate.
 * @example
 * ```ts
 * import { tmle } from 'omai-engine/causal'
 * const x = [[0], [1], [2], [3], [4], [5], [6], [7]]
 * tmle({ x, treatment: [0, 1, 0, 1, 1, 0, 1, 0], outcome: [0, 3, 2, 5, 6, 5, 8, 7] }, { fluctuation: 'linear' }).estimand // => 'ate'
 * ```
 */
export function tmle(data: CausalData, options: TmleOptions = {}): TmleResult {
  const p = prepare(data.x, data.treatment, data.outcome, 'tmle')
  const nu = nuisancesFor(p, options)
  const lo = minOf(p.y)
  const hi = maxOf(p.y)
  const span = hi - lo || 1
  const linear = options.fluctuation === 'linear'
  const scale = (v: number): number => (linear ? v : (v - lo) / span)
  const squeeze = (v: number): number => (linear ? v : Math.min(1 - 1e-4, Math.max(1e-4, v)))
  const y = Float64Array.from(p.y, scale)
  const q0 = Float64Array.from(nu.mu0, (v) => squeeze(scale(v)))
  const q1 = Float64Array.from(nu.mu1, (v) => squeeze(scale(v)))
  const h1 = Float64Array.from(nu.e, (e) => 1 / e)
  const h0 = Float64Array.from(nu.e, (e) => -1 / (1 - e))
  const h = Float64Array.from({ length: p.n }, (_, i) => (p.t[i] === 1 ? (h1[i] ?? 0) : (h0[i] ?? 0)))
  const qa = Float64Array.from({ length: p.n }, (_, i) => (p.t[i] === 1 ? (q1[i] ?? 0) : (q0[i] ?? 0)))
  let eps: number
  let q1s: Float64Array
  let q0s: Float64Array
  if (linear) {
    let num = 0
    let den = 0
    for (let i = 0; i < p.n; i++) {
      num += (h[i] ?? 0) * ((y[i] ?? 0) - (qa[i] ?? 0))
      den += (h[i] ?? 0) ** 2
    }
    eps = num / den
    q1s = q1.map((v, i) => v + eps * (h1[i] ?? 0))
    q0s = q0.map((v, i) => v + eps * (h0[i] ?? 0))
  } else {
    const offset = qa.map(logit)
    const solved = solveSmooth(
      tmleFluctuationProblem({ h, offset, y }),
      options.solver ?? newtonSolver(1e-10),
      options.key ?? makeKey(0),
      200,
      'tmle',
    )
    eps = solved.x[0] ?? 0
    q1s = q1.map((v, i) => expit(logit(v) + eps * (h1[i] ?? 0)))
    q0s = q0.map((v, i) => expit(logit(v) + eps * (h0[i] ?? 0)))
  }
  const factor = linear ? 1 : span
  const psi = mean(q1s.map((v, i) => v - (q0s[i] ?? 0)))
  const influence = Float64Array.from({ length: p.n }, (_, i) => {
    const qaStar = p.t[i] === 1 ? (q1s[i] ?? 0) : (q0s[i] ?? 0)
    return factor * ((h[i] ?? 0) * ((y[i] ?? 0) - qaStar) + (q1s[i] ?? 0) - (q0s[i] ?? 0) - psi)
  })
  const effect = effectFromInfluence('ate', factor * psi, influence, {
    ...(options.level === undefined ? {} : { level: options.level }),
    method: `targeted maximum likelihood (${linear ? 'linear' : 'logistic'} fluctuation)`,
  })
  const initial = mean(q1.map((v, i) => v - (q0[i] ?? 0))) * factor
  return { ...effect, epsilon: eps, initial }
}

/** Options of {@link doubleMachineLearningAlgorithm}. */
export interface DmlOptions extends EffectOptions {
  /** `'plr'`: the partially linear model (Robinson's residual-on-residual regression, a constant effect); `'irm'`: the interactive model with the AIPW score (the ATE). */
  readonly model?: 'plr' | 'irm'
  /** The number of cross-fitting folds (default 5). */
  readonly folds?: number
}

/** The state of {@link doubleMachineLearningAlgorithm}: the cross-fitted scores filled in one fold per step. */
export type DmlState = {
  /** The folds done. */
  readonly fold: number
  /** Outcome residuals (PLR) or the AIPW score (IRM) per unit; zero until the unit's fold is done. */
  readonly score: number[]
  /** Treatment residuals per unit (PLR). */
  readonly treatmentResidual: number[]
  /** The units whose fold is done. */
  readonly filled: number[]
  readonly estimate: number
  readonly standardError: number
}

function dmlSummary(
  model: 'plr' | 'irm',
  s: Pick<DmlState, 'score' | 'treatmentResidual' | 'filled'>,
): { readonly estimate: number; readonly standardError: number; readonly influence: Float64Array } {
  const rows = s.filled
  const n = rows.length
  if (n === 0) return { estimate: 0, standardError: Number.NaN, influence: new Float64Array(0) }
  if (model === 'irm') {
    const g = rows.map((r) => s.score[r] ?? 0)
    const est = mean(g)
    const infl = Float64Array.from(g, (v) => v - est)
    return { estimate: est, standardError: Math.sqrt(variance(infl, 1) / n), influence: infl }
  }
  let num = 0
  let den = 0
  for (const r of rows) {
    num += (s.score[r] ?? 0) * (s.treatmentResidual[r] ?? 0)
    den += (s.treatmentResidual[r] ?? 0) ** 2
  }
  const theta = num / den
  const j = den / n
  const infl = Float64Array.from(
    rows,
    (r) => (((s.score[r] ?? 0) - theta * (s.treatmentResidual[r] ?? 0)) * (s.treatmentResidual[r] ?? 0)) / j,
  )
  return { estimate: theta, standardError: Math.sqrt(mean(infl.map((v) => v * v)) / n), influence: infl }
}

/**
 * Double (debiased) machine learning as an algorithm with one cross-fitting fold per step: on each fold the nuisance
 * learners are fitted on the other folds and the held-out units get their orthogonal score (the residual-on-residual
 * product of the partially linear model, or the AIPW score of the interactive model); the running estimate after each
 * fold and its standard error are observed, converging as the folds fill in. The state serialises.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - The model, folds, learners and key.
 * @returns The algorithm; it observes `fold`, `estimate` and `standardError`.
 * @example
 * ```ts
 * import { doubleMachineLearningAlgorithm } from 'omai-engine/causal'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = Array.from({ length: 40 }, (_, i) => [Math.sin(i), Math.cos(i)])
 * const treatment = x.map((_, i) => (i % 3 === 0 ? 1 : 0))
 * const outcome = x.map((r, i) => (r[0] ?? 0) + 2 * (treatment[i] ?? 0) + 0.01 * i)
 * runAlgorithm(doubleMachineLearningAlgorithm({ x, treatment, outcome }, { folds: 4 }), { maxSteps: 10 }).steps // => 4
 * ```
 */
export function doubleMachineLearningAlgorithm(data: CausalData, options: DmlOptions = {}): Algorithm<DmlState> {
  const p = prepare(data.x, data.treatment, data.outcome, 'doubleMachineLearning')
  const model = options.model ?? 'plr'
  const k = options.folds ?? 5
  const key = options.key ?? makeKey(0)
  const folds = crossFitFolds(p.t, k, key)
  return {
    name: 'double-machine-learning',
    config: { model, folds: k, n: p.n },
    schema: channels({
      metrics: {
        fold: { description: 'cross-fitting folds done' },
        estimate: { description: 'the effect estimated from the folds done' },
        standardError: { description: 'its standard error', direction: 'lower' },
      },
    }),
    init: () => ({
      fold: 0,
      score: new Array<number>(p.n).fill(0),
      treatmentResidual: new Array<number>(p.n).fill(0),
      filled: [],
      estimate: 0,
      standardError: Number.NaN,
    }),
    step: (s) => {
      const f = folds[s.fold]
      if (f === undefined) return s
      const nu = fitNuisances(p, f.train, f.test, { ...options, pooled: model === 'plr' }, s.fold)
      const score = [...s.score]
      const tr = [...s.treatmentResidual]
      f.test.forEach((row, i) => {
        if (model === 'plr') {
          score[row] = (p.y[row] ?? 0) - (nu.m?.[i] ?? 0)
          tr[row] = (p.t[row] ?? 0) - (nu.e[i] ?? 0)
        } else {
          const e = nu.e[i] ?? 0.5
          const m1 = nu.mu1[i] ?? 0
          const m0 = nu.mu0[i] ?? 0
          const t = p.t[row] ?? 0
          const y = p.y[row] ?? 0
          score[row] = m1 - m0 + (t * (y - m1)) / e - ((1 - t) * (y - m0)) / (1 - e)
        }
      })
      const filled = [...s.filled, ...f.test]
      const sum = dmlSummary(model, { score, treatmentResidual: tr, filled })
      return {
        fold: s.fold + 1,
        score,
        treatmentResidual: tr,
        filled,
        estimate: sum.estimate,
        standardError: sum.standardError,
      }
    },
    done: (s) => (s.fold >= folds.length ? 'converged' : undefined),
    observe: (s) => ({ metrics: { fold: s.fold, estimate: s.estimate, standardError: s.standardError } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as DmlState,
  }
}

/**
 * Double machine learning, run to the end: cross-fitted nuisance learners and an orthogonal score, so the effect's
 * estimate is root-n consistent although the learners converge more slowly.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - The model (`'plr'` or `'irm'`), folds, learners and key.
 * @returns The effect with its influence values.
 * @index function Double machine learning treatment effect with cross-fitting.
 * @example
 * ```ts
 * import { doubleMachineLearning } from 'omai-engine/causal'
 * const x = Array.from({ length: 60 }, (_, i) => [Math.sin(i), Math.cos(3 * i)])
 * const treatment = x.map((r, i) => ((r[0] ?? 0) + Math.sin(7 * i) > 0 ? 1 : 0))
 * const outcome = x.map((r, i) => (r[0] ?? 0) + 2 * (treatment[i] ?? 0) + 0.1 * Math.cos(5 * i))
 * doubleMachineLearning({ x, treatment, outcome }, { folds: 3 }).estimand // => 'ate'
 * ```
 */
export function doubleMachineLearning(data: CausalData, options: DmlOptions = {}): EffectEstimate {
  const alg = doubleMachineLearningAlgorithm(data, options)
  const s = runAlgorithm(alg, { key: options.key ?? makeKey(0), maxSteps: options.folds ?? 5 }).state
  const model = options.model ?? 'plr'
  const sum = dmlSummary(model, s)
  return effectFromInfluence('ate', sum.estimate, sum.influence, {
    ...(options.level === undefined ? {} : { level: options.level }),
    method: `double machine learning (${model === 'plr' ? 'partially linear' : 'interactive'}, ${options.folds ?? 5}-fold)`,
  })
}
