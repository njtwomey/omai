/**
 * Sensitivity analysis: how strong an unobserved confounder would have to be to explain an estimated effect away. The
 * E-value (VanderWeele and Ding) for risk ratios, odds and hazard ratios and standardised mean differences; Rosenbaum
 * bounds on the significance of a matched-pairs effect as the hidden bias Gamma grows (the Wilcoxon signed-rank form for
 * measured outcomes, the McNemar form for binary ones) with the critical Gamma; omitted-variable bias for a regression
 * effect after Cinelli and Hazlett (the bias and adjusted estimate for a confounder of given partial R-squared, the
 * robustness value, a contour sweep and benchmarks from observed covariates); and refutation tests that rerun an
 * estimator on a placebo treatment, with a random common cause added, and on random subsets.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey, normal, permutation } from '../random/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import { betaInc, mean, normCdf, prepare, tPpf, variance, wls } from './_dense.ts'
import type { CausalData } from './data.ts'
import type { MatchingResult } from './matching.ts'

/** The scales an effect can be given on for the E-value. */
export type EffectScale = 'rr' | 'or-rare' | 'or-common' | 'hr-rare' | 'hr-common' | 'smd'

/** An E-value: the minimum strength of association, on the risk-ratio scale, a confounder would need. */
export interface EValue {
  /** For the point estimate: with both the treatment and the outcome, the confounder would need this risk ratio to move the estimate to the null. */
  readonly point: number
  /** For the confidence limit closest to the null (1 when the interval includes the null). */
  readonly interval: number
  /** The estimate and limits on the risk-ratio scale. */
  readonly riskRatio: { readonly estimate: number; readonly lower?: number; readonly upper?: number }
}

const toRiskRatio = (v: number, scale: EffectScale): number => {
  switch (scale) {
    case 'rr':
    case 'or-rare':
    case 'hr-rare':
      return v
    case 'or-common':
      return Math.sqrt(v)
    case 'hr-common':
      return (1 - 0.5 ** Math.sqrt(v)) / (1 - 0.5 ** Math.sqrt(1 / v))
    case 'smd':
      return Math.exp(0.91 * v)
  }
}

const eOf = (rr: number): number => {
  const r = rr < 1 ? 1 / rr : rr
  return r + Math.sqrt(r * (r - 1))
}

/**
 * The E-value of an observed association (VanderWeele and Ding 2017): the smallest risk ratio, with both the treatment
 * and the outcome, an unmeasured confounder would need to fully explain the estimate (and, for the interval, its
 * confidence limit nearest the null). Odds and hazard ratios of common outcomes and standardised mean differences are
 * first converted to approximate risk ratios.
 *
 * @param estimate - The point estimate on the given scale.
 * @param options - `scale` (default `'rr'`) and the interval limits `lower` and `upper` on the same scale.
 * @returns The E-value for the estimate and for the interval.
 * @example
 * ```ts
 * import { eValue } from 'omai-engine/causal'
 * Math.round(eValue(3.9, { lower: 1.8, upper: 8.7 }).point * 1000) / 1000 // => 7.263
 * ```
 */
export function eValue(
  estimate: number,
  options: { readonly scale?: EffectScale; readonly lower?: number; readonly upper?: number } = {},
): EValue {
  const scale = options.scale ?? 'rr'
  const rr = toRiskRatio(estimate, scale)
  const lo = options.lower === undefined ? undefined : toRiskRatio(options.lower, scale)
  const hi = options.upper === undefined ? undefined : toRiskRatio(options.upper, scale)
  const crossesNull = lo !== undefined && hi !== undefined && lo <= 1 && hi >= 1
  let interval = 1
  if (!crossesNull) {
    const nearest = rr >= 1 ? lo : hi
    if (nearest !== undefined) interval = eOf(nearest)
    else if (lo === undefined && hi === undefined) interval = Number.NaN
  }
  const point = eOf(rr)
  return {
    point,
    interval,
    riskRatio: { estimate: rr, ...(lo === undefined ? {} : { lower: lo }), ...(hi === undefined ? {} : { upper: hi }) },
  }
}

/** One row of the Rosenbaum sensitivity table. */
export interface RosenbaumRow {
  readonly gamma: number
  /** The largest p-value consistent with a hidden bias of at most `gamma` (the effect is significant at alpha if below). */
  readonly pUpper: number
  /** The smallest such p-value. */
  readonly pLower: number
}

/** The result of {@link rosenbaumBounds}. */
export interface RosenbaumResult {
  readonly rows: readonly RosenbaumRow[]
  /** The smallest `gamma` at which the upper p-value exceeds `alpha` (the finding stops being significant); Infinity if never in the grid's reach. */
  readonly criticalGamma: number
  readonly alpha: number
  readonly test: 'signed-rank' | 'mcnemar'
}

/**
 * Rosenbaum bounds for matched pairs with measured outcomes: Wilcoxon's signed-rank statistic of the within-pair
 * differences, whose null distribution under hidden bias `gamma` (the odds of treatment of two matched units may differ
 * by a factor up to `gamma`) is bounded by sign probabilities `gamma / (1 + gamma)` and `1 / (1 + gamma)`. P-values use the normal
 * approximation. The critical Gamma is the hidden bias at which the finding stops being significant.
 *
 * @param treated - The treated unit's outcome in each pair.
 * @param control - The matched control's outcome in each pair.
 * @param options - The `gammas` to tabulate (default 1 to 3 by 0.25) and `alpha` (default 0.05).
 * @returns The table of bounds and the critical Gamma.
 * @example
 * ```ts
 * import { rosenbaumBounds } from 'omai-engine/causal'
 * const treated = [5, 6, 7, 8, 9, 10, 11, 12]
 * const control = [4, 4.5, 6.1, 7, 7.6, 9, 9.8, 10.5]
 * rosenbaumBounds(treated, control).rows[0].pUpper < 0.05 // => true
 * ```
 */
export function rosenbaumBounds(
  treated: ArrayLike<number>,
  control: ArrayLike<number>,
  options: { readonly gammas?: readonly number[]; readonly alpha?: number } = {},
): RosenbaumResult {
  if (treated.length !== control.length) throw new OmaiError('shape', 'rosenbaumBounds: one control per treated unit')
  const alpha = options.alpha ?? 0.05
  const diffs = Array.from(treated, (v, i) => v - (control[i] ?? 0)).filter((d) => d !== 0)
  const n = diffs.length
  if (n < 2) throw new OmaiError('value', 'rosenbaumBounds: fewer than two pairs differ')
  // Mid-ranks of the absolute differences.
  const order = diffs.map((_, i) => i).sort((a, b) => Math.abs(diffs[a] ?? 0) - Math.abs(diffs[b] ?? 0))
  const ranks = new Float64Array(n)
  for (let i = 0; i < n; ) {
    let j = i
    while (j + 1 < n && Math.abs(diffs[order[j + 1] ?? 0] ?? 0) === Math.abs(diffs[order[i] ?? 0] ?? 0)) j++
    for (let k = i; k <= j; k++) ranks[order[k] ?? 0] = (i + j) / 2 + 1
    i = j + 1
  }
  const tPlus = diffs.reduce((s, d, i) => s + (d > 0 ? (ranks[i] ?? 0) : 0), 0)
  const sumRanks = ranks.reduce((a, b) => a + b, 0)
  const sumSq = ranks.reduce((a, b) => a + b * b, 0)
  const pValue = (gamma: number, upper: boolean): number => {
    const q = upper ? gamma / (1 + gamma) : 1 / (1 + gamma)
    const mu = q * sumRanks
    const sd = Math.sqrt(q * (1 - q) * sumSq)
    return 1 - normCdf((tPlus - mu) / sd)
  }
  const gammas = options.gammas ?? Array.from({ length: 9 }, (_, i) => 1 + 0.25 * i)
  const rows = gammas.map((gamma) => ({ gamma, pUpper: pValue(gamma, true), pLower: pValue(gamma, false) }))
  let critical = Number.POSITIVE_INFINITY
  if (pValue(1, true) <= alpha) {
    let [lo, hi] = [1, 2]
    while (pValue(hi, true) <= alpha && hi < 1e6) hi *= 2
    if (hi < 1e6) {
      for (let it = 0; it < 80; it++) {
        const mid = (lo + hi) / 2
        if (pValue(mid, true) <= alpha) lo = mid
        else hi = mid
      }
      critical = (lo + hi) / 2
    }
  } else critical = 1
  return { rows, criticalGamma: critical, alpha, test: 'signed-rank' }
}

/**
 * Rosenbaum bounds for matched pairs with a binary outcome: among the discordant pairs (one member responded) the
 * number in which the treated unit responded is binomial with probability between `1 / (1 + gamma)` and `gamma / (1 +
 * gamma)` under hidden bias `gamma`, giving exact bounds on the one-sided McNemar p-value.
 *
 * @param treated - The treated unit's 0/1 outcome in each pair.
 * @param control - The control's 0/1 outcome in each pair.
 * @param options - The `gammas` to tabulate and `alpha`.
 * @returns The table of exact bounds and the critical Gamma.
 * @example
 * ```ts
 * import { rosenbaumBoundsBinary } from 'omai-engine/causal'
 * const treated = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
 * const control = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
 * rosenbaumBoundsBinary(treated, control).rows[0].pUpper.toFixed(4) // => '0.0195'
 * ```
 */
export function rosenbaumBoundsBinary(
  treated: ArrayLike<number>,
  control: ArrayLike<number>,
  options: { readonly gammas?: readonly number[]; readonly alpha?: number } = {},
): RosenbaumResult {
  if (treated.length !== control.length)
    throw new OmaiError('shape', 'rosenbaumBoundsBinary: one control per treated unit')
  const alpha = options.alpha ?? 0.05
  let b = 0
  let c = 0
  for (let i = 0; i < treated.length; i++) {
    if (treated[i] === 1 && control[i] === 0) b++
    else if (treated[i] === 0 && control[i] === 1) c++
  }
  const m = b + c
  if (m === 0) throw new OmaiError('value', 'rosenbaumBoundsBinary: no discordant pairs')
  // P(Bin(m, p) >= b) = I_p(b, m - b + 1).
  const tail = (p: number): number => (b === 0 ? 1 : betaInc(b, m - b + 1, p))
  const gammas = options.gammas ?? Array.from({ length: 9 }, (_, i) => 1 + 0.25 * i)
  const rows = gammas.map((gamma) => ({ gamma, pUpper: tail(gamma / (1 + gamma)), pLower: tail(1 / (1 + gamma)) }))
  let critical = Number.POSITIVE_INFINITY
  if (tail(0.5) <= alpha) {
    let [lo, hi] = [1, 2]
    while (tail(hi / (1 + hi)) <= alpha && hi < 1e6) hi *= 2
    if (hi < 1e6) {
      for (let it = 0; it < 80; it++) {
        const mid = (lo + hi) / 2
        if (tail(mid / (1 + mid)) <= alpha) lo = mid
        else hi = mid
      }
      critical = (lo + hi) / 2
    }
  } else critical = 1
  return { rows, criticalGamma: critical, alpha, test: 'mcnemar' }
}

/**
 * The outcomes of the pairs a one-to-`k` matching made: for each matched unit of the target arm, its outcome and the
 * mean outcome of its matches, oriented as treated and control.
 *
 * @param data - The data the matching was run on.
 * @param result - The matching result.
 * @returns Parallel arrays of treated and control outcomes.
 * @example
 * ```ts
 * import { matchedPairs, matching } from 'omai-engine/causal'
 * const data = { x: [[0], [1], [2], [3], [0.1], [1.1], [2.1], [3.1]], treatment: [1, 1, 1, 1, 0, 0, 0, 0], outcome: [3, 4, 5, 6, 1, 2, 3, 4] }
 * matchedPairs(data, matching(data, { on: 'covariates' })).treated // => [3, 4, 5, 6]
 * ```
 */
export function matchedPairs(
  data: CausalData,
  result: MatchingResult,
): { readonly treated: number[]; readonly control: number[] } {
  const p = prepare(data.x, data.treatment, data.outcome, 'matchedPairs')
  const treated: number[] = []
  const control: number[] = []
  for (const u of result.pairs) {
    const own = p.y[u.unit] ?? 0
    const others = mean(u.matches.map((j) => p.y[j] ?? 0))
    if (p.t[u.unit] === 1) {
      treated.push(own)
      control.push(others)
    } else {
      treated.push(others)
      control.push(own)
    }
  }
  return { treated, control }
}

/** A regression estimate of a treatment effect, as the omitted-variable-bias formulas need it. */
export interface RegressionEffect {
  readonly estimate: number
  readonly standardError: number
  /** Residual degrees of freedom. */
  readonly df: number
}

/**
 * The partial R-squared of the treatment with the outcome given the other regressors: `t^2 / (t^2 + df)`.
 *
 * @param reg - The treatment's coefficient, standard error and residual degrees of freedom.
 * @returns The partial R-squared.
 * @example
 * ```ts
 * import { partialR2 } from 'omai-engine/causal'
 * partialR2({ estimate: 2, standardError: 1, df: 8 }).toFixed(3) // => '0.333'
 * ```
 */
export function partialR2(reg: RegressionEffect): number {
  const t = reg.estimate / reg.standardError
  return (t * t) / (t * t + reg.df)
}

/**
 * The partial Cohen's f of the treatment: `|t| / sqrt(df)`.
 *
 * @param reg - The treatment's coefficient, standard error and residual degrees of freedom.
 * @returns Cohen's f.
 * @example
 * ```ts
 * import { partialCohenF } from 'omai-engine/causal'
 * partialCohenF({ estimate: 2, standardError: 1, df: 8 }).toFixed(3) // => '0.707'
 * ```
 */
export function partialCohenF(reg: RegressionEffect): number {
  return Math.abs(reg.estimate / reg.standardError) / Math.sqrt(reg.df)
}

/**
 * The robustness value (Cinelli and Hazlett 2020): the strength of association, as a partial R-squared with both the
 * treatment and the outcome, an unobserved confounder would need to reduce the estimate by the share `q` (to zero when
 * `q` is 1) or, with `alpha`, to make it statistically insignificant at that level. Larger means a more robust effect.
 *
 * @param reg - The treatment's coefficient, standard error and residual degrees of freedom.
 * @param options - The share `q` to explain away (default 1) and `alpha` (default 1: the point estimate only).
 * @returns The robustness value, in `[0, 1]`.
 * @example
 * ```ts
 * import { robustnessValue } from 'omai-engine/causal'
 * robustnessValue({ estimate: 0.09731582, standardError: 0.02325654, df: 783 }).toFixed(4) // => '0.1388'
 * ```
 */
export function robustnessValue(
  reg: RegressionEffect,
  options: { readonly q?: number; readonly alpha?: number } = {},
): number {
  const q = options.q ?? 1
  const alpha = options.alpha ?? 1
  let f = q * partialCohenF(reg)
  if (alpha < 1) f -= Math.abs(tPpf(reg.df - 1, alpha / 2)) / Math.sqrt(reg.df - 1)
  if (f < 0) return 0
  return 0.5 * (Math.sqrt(f ** 4 + 4 * f * f) - f * f)
}

/** The estimate after adjusting for a confounder of given strength. */
export interface OvbResult {
  /** The bias the confounder would cause. */
  readonly bias: number
  /** The estimate with the bias removed (towards zero for a positive estimate). */
  readonly adjustedEstimate: number
  readonly adjustedStandardError: number
  readonly adjustedT: number
}

/**
 * Omitted-variable bias for a regression effect (Cinelli and Hazlett): a confounder with partial R-squared `r2dz` with
 * the treatment and `r2yz` with the outcome (given the treatment and covariates) biases the coefficient by
 * `se sqrt(df) sqrt(r2yz r2dz / (1 - r2dz))`; the adjusted estimate removes it, with the standard error recomputed.
 *
 * @param reg - The treatment's coefficient, standard error and residual degrees of freedom.
 * @param r2dz - The confounder's partial R-squared with the treatment given the covariates.
 * @param r2yz - Its partial R-squared with the outcome given the treatment and covariates.
 * @returns The bias, adjusted estimate and adjusted standard error.
 * @example
 * ```ts
 * import { omittedVariableBias } from 'omai-engine/causal'
 * omittedVariableBias({ estimate: 0.1, standardError: 0.02, df: 800 }, 0.05, 0.05).adjustedEstimate.toFixed(4) // => '0.0710'
 * ```
 */
export function omittedVariableBias(reg: RegressionEffect, r2dz: number, r2yz: number): OvbResult {
  if (!(r2dz >= 0 && r2dz < 1) || !(r2yz >= 0 && r2yz <= 1))
    throw new OmaiError('value', 'omittedVariableBias: partial R-squared outside [0, 1)')
  const bias = reg.standardError * Math.sqrt(reg.df) * Math.sqrt((r2yz * r2dz) / (1 - r2dz))
  const adjusted = Math.sign(reg.estimate) * (Math.abs(reg.estimate) - bias)
  const se = reg.standardError * Math.sqrt((1 - r2yz) / (1 - r2dz)) * Math.sqrt(reg.df / (reg.df - 1))
  return { bias, adjustedEstimate: adjusted, adjustedStandardError: se, adjustedT: adjusted / se }
}

/** The adjusted estimate over a grid of confounder strengths. */
export interface OvbContour {
  readonly r2dz: readonly number[]
  readonly r2yz: readonly number[]
  /** Adjusted estimates, `r2yz` rows by `r2dz` columns. */
  readonly estimate: readonly (readonly number[])[]
  readonly tStatistic: readonly (readonly number[])[]
}

/**
 * The contour data of a sensitivity plot: the adjusted estimate (and its t statistic) for every pair of confounder
 * strengths on a grid, where the zero contour is the combinations that would explain the effect away.
 *
 * @param reg - The treatment's coefficient, standard error and residual degrees of freedom.
 * @param r2dz - The partial R-squared with the treatment to try.
 * @param r2yz - The partial R-squared with the outcome to try.
 * @returns The grid of adjusted estimates and t statistics.
 * @example
 * ```ts
 * import { ovbContour } from 'omai-engine/causal'
 * ovbContour({ estimate: 0.1, standardError: 0.02, df: 800 }, [0, 0.1], [0, 0.1]).estimate[0] // => [0.1, 0.1]
 * ```
 */
export function ovbContour(reg: RegressionEffect, r2dz: readonly number[], r2yz: readonly number[]): OvbContour {
  const cells = r2yz.map((y) => r2dz.map((d) => omittedVariableBias(reg, d, y)))
  return {
    r2dz,
    r2yz,
    estimate: cells.map((row) => row.map((c) => c.adjustedEstimate)),
    tStatistic: cells.map((row) => row.map((c) => c.adjustedT)),
  }
}

/** The treatment regression and the partial R-squared of every covariate, for benchmarking a hidden confounder. */
export interface SensitivityModel {
  readonly effect: RegressionEffect
  /** For each covariate: its partial R-squared with the treatment given the others. */
  readonly r2dxj: readonly number[]
  /** For each covariate: its partial R-squared with the outcome given the treatment and the other covariates. */
  readonly r2yxj: readonly number[]
}

function olsT(
  a: Float64Array,
  n: number,
  p: number,
  y: ArrayLike<number>,
  at: number,
): { beta: number; se: number; df: number } {
  const fit = wls(a, n, p, y)
  const df = n - p
  return { beta: fit.beta[at] ?? Number.NaN, se: Math.sqrt((fit.bread[at * p + at] ?? Number.NaN) * fit.sigma2), df }
}

/**
 * Fit the regressions a sensitivity analysis starts from: the outcome on the treatment and covariates (the effect, its
 * error and degrees of freedom), and for every covariate its partial R-squared with the treatment and with the outcome,
 * which benchmark how strong an unobserved confounder is plausible.
 *
 * @param data - Covariates, treatment and outcome.
 * @returns The effect and the covariates' partial R-squared.
 * @example
 * ```ts
 * import { sensitivityModel } from 'omai-engine/causal'
 * const x = [[0], [1], [2], [3], [4], [5], [6], [7]]
 * sensitivityModel({ x, treatment: [0, 1, 0, 1, 1, 0, 1, 0], outcome: [0, 3.1, 2, 5.2, 6, 5, 8.3, 7.1] }).effect.df // => 5
 * ```
 */
export function sensitivityModel(data: CausalData): SensitivityModel {
  const p = prepare(data.x, data.treatment, data.outcome, 'sensitivityModel')
  // Outcome: [1, t, x].
  const w = p.d + 2
  const a = new Float64Array(p.n * w)
  for (let i = 0; i < p.n; i++) {
    a[i * w] = 1
    a[i * w + 1] = p.t[i] ?? 0
    for (let j = 0; j < p.d; j++) a[i * w + 2 + j] = p.x[i * p.d + j] ?? 0
  }
  const out = olsT(a, p.n, w, p.y, 1)
  // Treatment: [1, x].
  const wd = p.d + 1
  const b = new Float64Array(p.n * wd)
  for (let i = 0; i < p.n; i++) {
    b[i * wd] = 1
    for (let j = 0; j < p.d; j++) b[i * wd + 1 + j] = p.x[i * p.d + j] ?? 0
  }
  const treat = Array.from({ length: p.d }, (_, j) => olsT(b, p.n, wd, p.t, 1 + j))
  const r2 = (r: { beta: number; se: number; df: number }): number =>
    (r.beta / r.se) ** 2 / ((r.beta / r.se) ** 2 + r.df)
  return {
    effect: { estimate: out.beta, standardError: out.se, df: out.df },
    r2dxj: treat.map(r2),
    r2yxj: Array.from({ length: p.d }, (_, j) => r2(olsT(a, p.n, w, p.y, 2 + j))),
  }
}

/**
 * The strength of a hidden confounder that is `kd` times as strongly related to the treatment, and `ky` times as strongly
 * related to the outcome, as an observed covariate (Cinelli and Hazlett's benchmarking bounds): the partial R-squared
 * values to put into {@link omittedVariableBias}.
 *
 * @param r2dxj - The covariate's partial R-squared with the treatment given the others.
 * @param r2yxj - Its partial R-squared with the outcome given the treatment and the others.
 * @param kd - How many times as strong with the treatment (default 1).
 * @param ky - How many times as strong with the outcome (default `kd`).
 * @returns The confounder's `r2dz` and `r2yz`.
 * @example
 * ```ts
 * import { benchmarkStrength } from 'omai-engine/causal'
 * benchmarkStrength(0.1, 0.05, 1, 1).r2dz // => 0.1111111111111111
 * ```
 */
export function benchmarkStrength(
  r2dxj: number,
  r2yxj: number,
  kd = 1,
  ky: number = kd,
): { readonly r2dz: number; readonly r2yz: number } {
  const r2dz = (kd * r2dxj) / (1 - r2dxj)
  const r2zxj = (kd * r2dxj * r2dxj) / ((1 - kd * r2dxj) * (1 - r2dxj))
  if (kd * r2dxj >= 1 || r2zxj >= 1)
    throw new OmaiError(
      'value',
      'benchmarkStrength: the confounder strength is impossible (R-squared of Z with the covariate reaches 1)',
    )
  const r2yz = ((Math.sqrt(ky) + Math.sqrt(r2zxj)) / Math.sqrt(1 - r2zxj)) ** 2 * (r2yxj / (1 - r2yxj))
  return { r2dz, r2yz: Math.min(1, r2yz) }
}

/** The tests of {@link refute}. */
export type RefutationTest = 'placebo-treatment' | 'random-common-cause' | 'data-subset'

/** The result of a refutation test. */
export interface Refutation {
  readonly test: RefutationTest
  /** The estimate on the original data. */
  readonly original: number
  /** The estimates over the trials. */
  readonly estimates: readonly number[]
  readonly mean: number
  readonly sd: number
  /**
   * For the placebo test, the share of placebo estimates at least as large as the original in absolute value (small means
   * the effect is distinguishable from chance); for the others, the two-sided z-test p-value that the original is a draw
   * from the trials' distribution.
   */
  readonly pValue: number
  /** Whether the estimator passed: the placebo effect vanishes, or the estimate does not move. */
  readonly passed: boolean
}

/**
 * Refute an estimator: rerun it on data that should not change its answer, or should make it vanish. A placebo test
 * permutes the treatment (the estimate should fall to zero); a random common cause adds a covariate of pure noise (the
 * estimate should not move); a data-subset test reruns on random subsets (the estimate should be stable).
 *
 * @param data - Covariates, treatment and outcome.
 * @param estimator - Any estimator of one number from causal data.
 * @param options - The `test`, `trials` (default 20), `fraction` of the data for the subset test (default 0.8) and `key`.
 * @returns The estimates over the trials, their mean and spread, and whether the test passed.
 * @example
 * ```ts
 * import { refute } from 'omai-engine/causal'
 * import { toFlat } from 'omai-engine/tensor'
 * const d = { x: [[0], [1], [2], [3], [4], [5], [6], [7]], treatment: [0, 1, 0, 1, 1, 0, 1, 0], outcome: [0, 3, 2, 5, 6, 5, 8, 7] }
 * const slope = (c: typeof d): number => toFlat(c.outcome as never).length
 * refute(d, slope, { test: 'data-subset', trials: 3 }).test // => 'data-subset'
 * ```
 */
export function refute(
  data: CausalData,
  estimator: (data: CausalData) => number,
  options: {
    readonly test?: RefutationTest
    readonly trials?: number
    readonly fraction?: number
    readonly key?: Key
  } = {},
): Refutation {
  const test = options.test ?? 'placebo-treatment'
  const trials = options.trials ?? 20
  const key = options.key ?? makeKey(0)
  const p = prepare(data.x, data.treatment, data.outcome, 'refute')
  const asData = (x: Float64Array, d: number, t: ArrayLike<number>, y: ArrayLike<number>, n: number): CausalData => ({
    x: fromStorage(x, [n, d], 'float64'),
    treatment: fromStorage(Float64Array.from(t), [n], 'float64'),
    outcome: fromStorage(Float64Array.from(y), [n], 'float64'),
  })
  const original = estimator(asData(p.x, p.d, p.t, p.y, p.n))
  const estimates: number[] = []
  for (let r = 0; r < trials; r++) {
    const k = foldIn(key, r)
    try {
      if (test === 'placebo-treatment') {
        const perm = toFlat(permutation(k, p.n))
        estimates.push(
          estimator(
            asData(
              p.x,
              p.d,
              perm.map((i) => p.t[i] ?? 0),
              p.y,
              p.n,
            ),
          ),
        )
      } else if (test === 'random-common-cause') {
        const noise = toFlat(normal(k, [p.n]))
        const w = p.d + 1
        const x = new Float64Array(p.n * w)
        for (let i = 0; i < p.n; i++) {
          for (let j = 0; j < p.d; j++) x[i * w + j] = p.x[i * p.d + j] ?? 0
          x[i * w + p.d] = noise[i] ?? 0
        }
        estimates.push(estimator(asData(x, w, p.t, p.y, p.n)))
      } else {
        const rows = toFlat(permutation(k, p.n)).slice(0, Math.max(4, Math.round((options.fraction ?? 0.8) * p.n)))
        const x = new Float64Array(rows.length * p.d)
        rows.forEach((row, i) => {
          for (let j = 0; j < p.d; j++) x[i * p.d + j] = p.x[row * p.d + j] ?? 0
        })
        estimates.push(
          estimator(
            asData(
              x,
              p.d,
              rows.map((row) => p.t[row] ?? 0),
              rows.map((row) => p.y[row] ?? 0),
              rows.length,
            ),
          ),
        )
      }
    } catch (error) {
      if (!(error instanceof OmaiError)) throw error
    }
  }
  const m = mean(estimates)
  const sd = Math.sqrt(variance(estimates, 1))
  let pValue: number
  let passed: boolean
  if (test === 'placebo-treatment') {
    pValue = (1 + estimates.filter((e) => Math.abs(e) >= Math.abs(original)).length) / (1 + estimates.length)
    passed = Math.abs(m) <= (2 * sd) / Math.sqrt(Math.max(1, estimates.length)) + 0.1 * Math.abs(original)
  } else {
    const z = sd === 0 ? (original === m ? 0 : Number.POSITIVE_INFINITY) : (original - m) / sd
    pValue = 2 * (1 - normCdf(Math.abs(z)))
    passed = pValue > 0.05
  }
  return { test, original, estimates, mean: m, sd, pValue, passed }
}
