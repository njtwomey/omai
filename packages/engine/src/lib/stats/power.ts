/**
 * Statistical power and sample-size calculations, statsmodels style: the probability that a test detects an effect of a
 * given size, and the sample size needed to reach a target power, for t tests, the two-sample proportion test, one-way
 * ANOVA and the chi-square goodness-of-fit test, plus power by simulation for any test.
 */

import { type Alternative, type Estimate, OmaiError } from '../base/index.ts'
import { chi2, fDistribution, noncentralChi2, noncentralF, noncentralT } from '../dists/index.ts'
import { brentq } from '../numerics/roots/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import { item } from '../tensor/index.ts'
import { normCdf, normPpf, normSf, tPpf } from './_util.ts'
import { proportionConfint } from './tests/proportions.ts'

/** The design of a t test: one sample, paired (the one-sample test of differences) or two independent samples. */
export type TTestKind = 'one-sample' | 'paired' | 'two-sample'

/** Options of {@link powerTTest}. */
export interface TTestPowerOptions {
  /** Cohen's d. */
  readonly effectSize: number
  /** Observations (in the first sample for two samples). */
  readonly nobs: number
  readonly alpha?: number
  readonly alternative?: Alternative
  readonly kind?: TTestKind
  /** nobs2 / nobs1 for two samples (default 1). */
  readonly ratio?: number
}

function checkAlpha(alpha: number): void {
  if (!(alpha > 0 && alpha < 1)) throw new OmaiError('value', 'power: alpha must lie in (0, 1)')
}

/**
 * The power of a t test (statsmodels TTestPower and TTestIndPower): the probability that the test rejects when the
 * standardised effect is d, from the noncentral t distribution with noncentrality d sqrt(n_eff).
 *
 * @param options - effectSize, nobs, alpha (default 0.05), alternative, kind (default `'one-sample'`) and ratio.
 * @returns The power.
 * @example
 * ```ts
 * import { powerTTest } from 'omai-engine/stats'
 * powerTTest({ effectSize: 0.5, nobs: 64, kind: 'two-sample' }) // => 0.801460
 * ```
 */
export function powerTTest(options: TTestPowerOptions): number {
  const alpha = options.alpha ?? 0.05
  checkAlpha(alpha)
  const alternative = options.alternative ?? 'two-sided'
  const kind = options.kind ?? 'one-sample'
  let n: number
  let df: number
  if (kind === 'two-sample') {
    const n2 = options.nobs * (options.ratio ?? 1)
    n = (options.nobs * n2) / (options.nobs + n2)
    df = options.nobs + n2 - 2
  } else {
    n = options.nobs
    df = options.nobs - 1
  }
  const nct = noncentralT(df, options.effectSize * Math.sqrt(n))
  const a = alternative === 'two-sided' ? alpha / 2 : alpha
  let power = 0
  if (alternative !== 'less') power += item(nct.sf(tPpf(df, 1 - a)))
  if (alternative !== 'greater') power += item(nct.cdf(tPpf(df, a)))
  return power
}

/** Solve power(n) = target for n > lo by Brent's method (statsmodels' solve_power for nobs). */
function solveN(power: (n: number) => number, target: number, lo: number): number {
  if (!(target > 0 && target < 1)) throw new OmaiError('value', 'sampleSize: power must lie in (0, 1)')
  let hi = Math.max(lo * 2, 10)
  while (power(hi) < target && hi < 1e9) hi *= 2
  if (power(hi) < target) throw new OmaiError('convergence', 'sampleSize: the power cannot be reached')
  return brentq((n) => power(n) - target, lo, hi, { xtol: 1e-10, rtol: 1e-12 }).root
}

/**
 * The sample size giving a t test a target power (statsmodels solve_power for nobs): real-valued, so round up.
 *
 * @param options - effectSize, power, alpha, alternative, kind and ratio.
 * @returns The required nobs (of the first sample for two samples).
 * @example
 * ```ts
 * import { sampleSizeTTest } from 'omai-engine/stats'
 * Math.ceil(sampleSizeTTest({ effectSize: 0.5, power: 0.8, kind: 'two-sample' })) // => 64
 * ```
 */
export function sampleSizeTTest(options: Omit<TTestPowerOptions, 'nobs'> & { readonly power: number }): number {
  return solveN((n) => powerTTest({ ...options, nobs: n }), options.power, 2 + 1e-9)
}

/** Options of {@link powerProportions}. */
export interface ProportionPowerOptions {
  /** Cohen's h (`cohensH`). */
  readonly effectSize: number
  /** Observations in the first sample. */
  readonly nobs: number
  /** nobs2 / nobs1 (default 1). */
  readonly ratio?: number
  readonly alpha?: number
  readonly alternative?: Alternative
}

/**
 * The power of the two-sample z test of proportions by the arcsine transform (statsmodels NormalIndPower with Cohen's
 * h): normal tails at h sqrt(n1 n2 / (n1 + n2)).
 *
 * @param options - effectSize (h), nobs, ratio, alpha and alternative.
 * @returns The power.
 * @example
 * ```ts
 * import { cohensH, powerProportions } from 'omai-engine/stats'
 * powerProportions({ effectSize: cohensH(0.6, 0.4).value, nobs: 100 }) // => 0.812641
 * ```
 */
export function powerProportions(options: ProportionPowerOptions): number {
  const alpha = options.alpha ?? 0.05
  checkAlpha(alpha)
  const alternative = options.alternative ?? 'two-sided'
  const r = options.ratio ?? 1
  const shift = options.effectSize * Math.sqrt((options.nobs * r) / (1 + r))
  if (alternative === 'greater') return normSf(normPpf(1 - alpha) - shift)
  if (alternative === 'less') return normCdf(normPpf(alpha) - shift)
  const crit = normPpf(1 - alpha / 2)
  return normSf(crit - shift) + normCdf(-crit - shift)
}

/**
 * The sample size giving the two-sample proportion test a target power (statsmodels NormalIndPower.solve_power).
 *
 * @param options - effectSize (h), power, ratio, alpha and alternative.
 * @returns The required size of the first sample.
 * @example
 * ```ts
 * import { cohensH, sampleSizeProportions } from 'omai-engine/stats'
 * Math.ceil(sampleSizeProportions({ effectSize: cohensH(0.6, 0.4).value, power: 0.8 })) // => 97
 * ```
 */
export function sampleSizeProportions(
  options: Omit<ProportionPowerOptions, 'nobs'> & { readonly power: number },
): number {
  return solveN((n) => powerProportions({ ...options, nobs: n }), options.power, 1e-6)
}

/** Options of {@link powerAnova}. */
export interface AnovaPowerOptions {
  /** Cohen's f. */
  readonly effectSize: number
  /** Total observations over all groups. */
  readonly nobs: number
  readonly kGroups: number
  readonly alpha?: number
}

/**
 * The power of the one-way ANOVA F test (statsmodels FTestAnovaPower): the noncentral F survival function at the
 * critical F, with noncentrality f^2 N on (k - 1, N - k) degrees of freedom.
 *
 * @param options - effectSize (f), nobs (total), kGroups and alpha.
 * @returns The power.
 * @example
 * ```ts
 * import { powerAnova } from 'omai-engine/stats'
 * powerAnova({ effectSize: 0.25, nobs: 159, kGroups: 3 }) // => 0.804887
 * ```
 */
export function powerAnova(options: AnovaPowerOptions): number {
  const alpha = options.alpha ?? 0.05
  checkAlpha(alpha)
  const d1 = options.kGroups - 1
  const d2 = options.nobs - options.kGroups
  if (!(d1 >= 1 && d2 > 0)) throw new OmaiError('value', 'powerAnova: needs kGroups >= 2 and nobs > kGroups')
  const crit = item(fDistribution(d1, d2).isf(alpha))
  return item(noncentralF(d1, d2, options.effectSize ** 2 * options.nobs).sf(crit))
}

/**
 * The total sample size giving a one-way ANOVA a target power (statsmodels FTestAnovaPower.solve_power).
 *
 * @param options - effectSize, power, kGroups and alpha.
 * @returns The total nobs.
 * @example
 * ```ts
 * import { sampleSizeAnova } from 'omai-engine/stats'
 * Math.ceil(sampleSizeAnova({ effectSize: 0.25, power: 0.8, kGroups: 3 })) // => 158
 * ```
 */
export function sampleSizeAnova(options: Omit<AnovaPowerOptions, 'nobs'> & { readonly power: number }): number {
  return solveN((n) => powerAnova({ ...options, nobs: n }), options.power, options.kGroups + 1e-6)
}

/** Options of {@link powerChisquare}. */
export interface ChisquarePowerOptions {
  /** Cohen's w. */
  readonly effectSize: number
  readonly nobs: number
  /** Number of cells. */
  readonly nBins: number
  /** Degrees of freedom lost to estimated parameters (default 0). */
  readonly ddof?: number
  readonly alpha?: number
}

/**
 * The power of the chi-square goodness-of-fit test (statsmodels GofChisquarePower): the noncentral chi-square survival
 * function at the critical value, noncentrality w^2 n on nBins - 1 - ddof degrees of freedom.
 *
 * @param options - effectSize (w), nobs, nBins, ddof and alpha.
 * @returns The power.
 * @example
 * ```ts
 * import { powerChisquare } from 'omai-engine/stats'
 * powerChisquare({ effectSize: 0.3, nobs: 100, nBins: 4 }) // => 0.711254
 * ```
 */
export function powerChisquare(options: ChisquarePowerOptions): number {
  const alpha = options.alpha ?? 0.05
  checkAlpha(alpha)
  const df = options.nBins - 1 - (options.ddof ?? 0)
  if (!(df >= 1)) throw new OmaiError('value', 'powerChisquare: needs at least one degree of freedom')
  const crit = item(chi2(df).isf(alpha))
  return item(noncentralChi2(df, options.effectSize ** 2 * options.nobs).sf(crit))
}

/**
 * The sample size giving the chi-square goodness-of-fit test a target power (statsmodels GofChisquarePower.solve_power).
 *
 * @param options - effectSize, power, nBins, ddof and alpha.
 * @returns The required nobs.
 * @example
 * ```ts
 * import { sampleSizeChisquare } from 'omai-engine/stats'
 * Math.ceil(sampleSizeChisquare({ effectSize: 0.3, power: 0.8, nBins: 4 })) // => 122
 * ```
 */
export function sampleSizeChisquare(options: Omit<ChisquarePowerOptions, 'nobs'> & { readonly power: number }): number {
  return solveN((n) => powerChisquare({ ...options, nobs: n }), options.power, 1e-6)
}

/**
 * Power by simulation: the rejection rate of any test on datasets simulated under an alternative, with its Wilson
 * interval; each simulation receives its own key.
 *
 * @param pValue - The p-value of the test on a dataset simulated from the key.
 * @param options - nSimulations (default 1000), alpha (default 0.05), level, key or seed.
 * @returns The estimated power with its interval.
 * @example
 * ```ts
 * import { simulatePower } from 'omai-engine/stats'
 * import { ttest1samp } from 'omai-engine/stats/tests'
 * import { normal as gaussian } from 'omai-engine/random'
 * const p = simulatePower((k) => ttest1samp(Array.from(gaussian(k, [30]).data, (v) => v + 1)).pValue, { nSimulations: 200 })
 * p.value > 0.95 // => true
 * ```
 */
export function simulatePower(
  pValue: (key: Key) => number,
  options: {
    readonly nSimulations?: number
    readonly alpha?: number
    readonly level?: number
    readonly key?: Key
    readonly seed?: number
  } = {},
): Estimate {
  const alpha = options.alpha ?? 0.05
  checkAlpha(alpha)
  const m = options.nSimulations ?? 1000
  const root = options.key ?? makeKey(options.seed ?? 0)
  let rejections = 0
  for (let i = 0; i < m; i++) if (pValue(foldIn(root, i)) < alpha) rejections++
  const ci = proportionConfint(rejections, m, { method: 'wilson', level: options.level ?? 0.95 })
  return { ...ci, method: `simulation of ${m} datasets, Wilson interval` }
}
