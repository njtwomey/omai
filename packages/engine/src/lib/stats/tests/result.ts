/**
 * The common shape of every hypothesis test: the `TestResult` record (statistic, p-value, degrees of freedom, effect
 * size, interval, assumption checks) and the helpers that build, format and read it. Also the tests of equal variances
 * (Levene, Brown-Forsythe, Bartlett, Fligner-Killeen) and the normality, equal-variance and expected-count checks that
 * parametric tests attach to their results.
 *
 * @index function testResult builder and options shared by hypothesis tests; TestResult is in base.
 */

import {
  type Alternative,
  type ConfidenceLevel,
  defaultAlternative,
  type Estimate,
  meanOf,
  OmaiError,
  resolveConfidenceLevel,
  type TestResult,
  varianceOf,
} from '../../base/index.ts'
import { asTensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { rankKernel } from '../_ranks.ts'
import { chi2Sf, fSf, groupsOf, normPpf } from '../_util.ts'
import { median } from '../quantiles.ts'
import { shapiroWilk } from './_shapiro.ts'

/** Options shared by tests: the alternative hypothesis and the confidence level of reported intervals. */
export interface TestOptions {
  /** Default `'two-sided'`. */
  readonly alternative?: Alternative
  /** Default 0.95. */
  readonly level?: ConfidenceLevel
}

/**
 * Check an alternative, defaulting to two-sided.
 *
 * @param alternative - The alternative, or undefined.
 * @returns The alternative.
 * @throws OmaiError with code `value` for anything else.
 * @example
 * ```ts
 * import { resolveAlternative } from 'omai-engine/stats/tests'
 * resolveAlternative() // => 'two-sided'
 * resolveAlternative('less') // => 'less'
 * ```
 */
export function resolveAlternative(alternative: Alternative = defaultAlternative): Alternative {
  if (alternative !== 'two-sided' && alternative !== 'less' && alternative !== 'greater')
    throw new OmaiError('value', `unknown alternative ${String(alternative)}`)
  return alternative
}

/**
 * The p-value of a statistic from its null distribution's lower and upper tail probabilities: `'less'` takes the CDF,
 * `'greater'` the survival function, `'two-sided'` twice the smaller tail, capped at one (scipy's convention).
 *
 * @param alternative - The alternative.
 * @param cdf - P(T <= t) under the null.
 * @param sf - P(T >= t) under the null.
 * @returns The p-value.
 * @example
 * ```ts
 * import { pValueFromTails } from 'omai-engine/stats/tests'
 * pValueFromTails('two-sided', 0.01, 0.99) // => 0.02
 * pValueFromTails('greater', 0.01, 0.99) // => 0.99
 * ```
 */
export function pValueFromTails(alternative: Alternative, cdf: number, sf: number): number {
  if (alternative === 'less') return Math.min(1, Math.max(0, cdf))
  if (alternative === 'greater') return Math.min(1, Math.max(0, sf))
  return Math.min(1, Math.max(0, 2 * Math.min(cdf, sf)))
}

/** The fields of a test result before cleaning (undefined optional fields are dropped). */
export interface TestResultInput {
  readonly test: string
  readonly statistic: number
  readonly pValue: number
  readonly alternative?: Alternative
  readonly df?: number | readonly number[] | undefined
  readonly effectSize?: (Estimate & { readonly name: string }) | undefined
  readonly estimate?: Estimate | undefined
  readonly assumptions?: Readonly<Record<string, boolean | number | string>> | undefined
  readonly method?: string | undefined
}

/**
 * Build a {@link TestResult}, dropping undefined optional fields and defaulting the alternative to two-sided. A NaN
 * statistic gives a NaN p-value, and a NaN statistic or p-value marks the result `status: 'undefined'`.
 *
 * @param input - The fields.
 * @returns The result.
 * @example
 * ```ts
 * import { testResult } from 'omai-engine/stats/tests'
 * testResult({ test: 'example', statistic: 2, pValue: 0.05 }).alternative // => 'two-sided'
 * testResult({ test: 'example', statistic: Number.NaN, pValue: 0.05 }).status // => 'undefined'
 * ```
 */
export function testResult(input: TestResultInput): TestResult {
  return {
    test: input.test,
    statistic: input.statistic,
    // A p-value of an undefined statistic is undefined, never a number.
    pValue: Number.isNaN(input.statistic) ? Number.NaN : input.pValue,
    ...(Number.isNaN(input.statistic) || Number.isNaN(input.pValue) ? { status: 'undefined' as const } : {}),
    alternative: input.alternative ?? 'two-sided',
    ...(input.df === undefined ? {} : { df: input.df }),
    ...(input.effectSize === undefined ? {} : { effectSize: input.effectSize }),
    ...(input.estimate === undefined ? {} : { estimate: input.estimate }),
    ...(input.assumptions === undefined ? {} : { assumptions: input.assumptions }),
    ...(input.method === undefined ? {} : { method: input.method }),
  }
}

/**
 * An estimate with an interval, from its value and bounds.
 *
 * @param value - The point estimate.
 * @param interval - The lower and upper bounds.
 * @param level - The confidence level.
 * @param extra - Standard error and method, if known.
 * @returns The estimate.
 * @example
 * ```ts
 * import { intervalEstimate } from 'omai-engine/stats/tests'
 * intervalEstimate(1, [0, 2], 0.95).interval // => [0, 2]
 * ```
 */
export function intervalEstimate(
  value: number,
  interval: readonly [number, number],
  level: number,
  extra: { readonly standardError?: number | undefined; readonly method?: string | undefined } = {},
): Estimate {
  return {
    value,
    interval,
    level,
    ...(extra.standardError === undefined ? {} : { standardError: extra.standardError }),
    ...(extra.method === undefined ? {} : { method: extra.method }),
  }
}

/**
 * A confidence interval for a parameter estimated by an asymptotically normal or t statistic: value -+ crit se, with
 * the one-sided intervals of the alternatives (unbounded on one side).
 *
 * @param value - The estimate.
 * @param se - Its standard error.
 * @param alternative - The alternative.
 * @param level - The confidence level.
 * @param quantile - The reference quantile function (default the standard normal's).
 * @returns The bounds.
 * @example
 * ```ts
 * import { waldInterval } from 'omai-engine/stats/tests'
 * waldInterval(0, 1, 'greater', 0.95)[1] // => Infinity
 * ```
 */
export function waldInterval(
  value: number,
  se: number,
  alternative: Alternative,
  level: number,
  quantile: (p: number) => number = normPpf,
): [number, number] {
  if (alternative === 'two-sided') {
    const c = quantile(0.5 + level / 2)
    return [value - c * se, value + c * se]
  }
  const c = quantile(level)
  return alternative === 'less'
    ? [Number.NEGATIVE_INFINITY, value + c * se]
    : [value - c * se, Number.POSITIVE_INFINITY]
}

/**
 * Format a test for reports: name, statistic, degrees of freedom, p-value, and the effect size or estimate with its
 * interval.
 *
 * @param result - The test result.
 * @param digits - Significant digits (default 4).
 * @returns One line.
 * @example
 * ```ts
 * import { formatTest } from 'omai-engine/stats/tests'
 * formatTest({ test: "Student's t test", statistic: 2.5, pValue: 0.0312, df: 10, alternative: 'two-sided' })
 * // => "Student's t test: statistic 2.5 (df 10), p = 0.0312, two-sided"
 * ```
 */
export function formatTest(result: TestResult, digits = 4): string {
  const f = (x: number): string => (Number.isFinite(x) ? String(Number(x.toPrecision(digits))) : String(x))
  const df =
    result.df === undefined ? '' : ` (df ${typeof result.df === 'number' ? f(result.df) : result.df.map(f).join(', ')})`
  const p = Number.isNaN(result.pValue)
    ? ''
    : `, p = ${result.pValue < 1e-4 ? result.pValue.toExponential(2) : f(result.pValue)}`
  let tail = ''
  const e = result.effectSize ?? result.estimate
  if (e !== undefined) {
    const name = result.effectSize?.name ?? 'estimate'
    const ci =
      e.interval === undefined
        ? ''
        : ` (${Math.round((e.level ?? 0.95) * 100)}% CI ${f(e.interval[0])} to ${f(e.interval[1])})`
    tail = `, ${name} ${f(e.value)}${ci}`
  }
  return `${result.test}: statistic ${f(result.statistic)}${df}${p}, ${result.alternative}${tail}`
}

/**
 * Whether a test rejects its null hypothesis at a significance level.
 *
 * @param result - The test result.
 * @param alpha - The level (default 0.05).
 * @returns True when p < alpha.
 * @example
 * ```ts
 * import { rejects } from 'omai-engine/stats/tests'
 * rejects({ test: 't', statistic: 3, pValue: 0.01, alternative: 'two-sided' }) // => true
 * ```
 */
export function rejects(result: TestResult, alpha = 0.05): boolean {
  return result.pValue < alpha
}

/** A centre for Levene's and Fligner's tests. */
export type SpreadCenter = 'median' | 'mean'

/**
 * Levene's test of equal variances (scipy.stats.levene): a one-way ANOVA of the absolute deviations from each group's
 * centre; with the median (default) it is the Brown-Forsythe test.
 *
 * @param groups - Two or more samples.
 * @param options - center (default `'median'`).
 * @returns The F statistic and its p-value.
 * @example
 * ```ts
 * import { levene } from 'omai-engine/stats/tests'
 * levene([[1, 2, 3, 4], [2, 4, 6, 8]]).pValue < 0.5 // => true
 * ```
 */
export function levene(groups: readonly TensorLike[], options: { readonly center?: SpreadCenter } = {}): TestResult {
  const gs = groupsOf('levene', groups, 2)
  const k = gs.length
  const z = gs.map((g) => {
    const c = options.center === 'mean' ? meanOf(g) : median(g)
    return Float64Array.from(g, (v) => Math.abs(v - c))
  })
  const N = gs.reduce((s, g) => s + g.length, 0)
  const zbar = z.map(meanOf)
  let grand = 0
  for (let i = 0; i < k; i++) grand += (zbar[i] ?? 0) * (z[i]?.length ?? 0)
  grand /= N
  let between = 0
  let within = 0
  for (let i = 0; i < k; i++) {
    const zi = z[i] ?? new Float64Array()
    between += zi.length * ((zbar[i] ?? 0) - grand) ** 2
    for (const v of zi) within += (v - (zbar[i] ?? 0)) ** 2
  }
  const W = ((N - k) / (k - 1)) * (between / within)
  return testResult({
    test: options.center === 'mean' ? "Levene's test" : 'Brown-Forsythe test',
    statistic: W,
    pValue: fSf(k - 1, N - k, W),
    df: [k - 1, N - k],
    method: `deviations from the ${options.center ?? 'median'}`,
  })
}

/**
 * Bartlett's test of equal variances for normal samples (scipy.stats.bartlett).
 *
 * @param groups - Two or more samples (each of size at least 2).
 * @returns The chi-square statistic and its p-value (df k - 1).
 * @example
 * ```ts
 * import { bartlett } from 'omai-engine/stats/tests'
 * bartlett([[1, 2, 3, 4], [1, 2, 3, 4]]).statistic // => 0
 * ```
 */
export function bartlett(groups: readonly TensorLike[]): TestResult {
  const gs = groupsOf('bartlett', groups, 2)
  const k = gs.length
  const ni = gs.map((g) => g.length)
  const ssq = gs.map((g) => varianceOf(g, 1))
  const N = ni.reduce((s, v) => s + v, 0)
  let pooled = 0
  let logs = 0
  let inv = 0
  for (let i = 0; i < k; i++) {
    const d = (ni[i] ?? 0) - 1
    pooled += d * (ssq[i] ?? 0)
    logs += d * Math.log(ssq[i] ?? 0)
    inv += 1 / d
  }
  pooled /= N - k
  const numer = (N - k) * Math.log(pooled) - logs
  const denom = 1 + (1 / (3 * (k - 1))) * (inv - 1 / (N - k))
  const T = numer / denom
  return testResult({ test: "Bartlett's test", statistic: T, pValue: chi2Sf(k - 1, T), df: k - 1 })
}

/**
 * The Fligner-Killeen test of equal variances (scipy.stats.fligner): normal scores of the ranked absolute deviations
 * from each group's median, robust to non-normality.
 *
 * @param groups - Two or more samples.
 * @param options - center (default `'median'`).
 * @returns The chi-square statistic and its p-value (df k - 1).
 * @example
 * ```ts
 * import { fligner } from 'omai-engine/stats/tests'
 * fligner([[1, 2, 3, 4, 5], [1, 3, 5, 7, 9]]).df // => 1
 * ```
 */
export function fligner(groups: readonly TensorLike[], options: { readonly center?: SpreadCenter } = {}): TestResult {
  const gs = groupsOf('fligner', groups, 1)
  const k = gs.length
  const all: number[] = []
  for (const g of gs) {
    const c = options.center === 'mean' ? meanOf(g) : median(g)
    for (const v of g) all.push(Math.abs(v - c))
  }
  const N = all.length
  const ranks = rankKernel(all)
  const a = Float64Array.from(ranks, (r) => normPpf(r / (2 * (N + 1)) + 0.5))
  const abar = meanOf(a)
  const varsq = varianceOf(a, 1)
  let s = 0
  let offset = 0
  for (const g of gs) {
    const part = a.subarray(offset, offset + g.length)
    s += g.length * (meanOf(part) - abar) ** 2
    offset += g.length
  }
  // Every deviation tied: the scores are one value, their variance is rounding noise and the ratio is not a statistic.
  if (ranks.every((r) => r === ranks[0]))
    return testResult({ test: 'Fligner-Killeen test', statistic: Number.NaN, pValue: Number.NaN, df: k - 1 })
  const statistic = s / varsq
  return testResult({ test: 'Fligner-Killeen test', statistic, pValue: chi2Sf(k - 1, statistic), df: k - 1 })
}

/**
 * The normality assumption of a parametric test: Shapiro-Wilk on every sample of size at least 3.
 *
 * @param samples - The samples.
 * @param alpha - The level below which normality is rejected (default 0.05).
 * @returns Whether every sample passes and the smallest Shapiro-Wilk p-value.
 * @example
 * ```ts
 * import { checkNormality } from 'omai-engine/stats/tests'
 * checkNormality([[2.1, 1.9, 2.4, 2.0, 1.7, 2.2]]).normal // => true
 * ```
 */
export function checkNormality(
  samples: readonly TensorLike[],
  alpha = 0.05,
): { readonly normal: boolean; readonly minP: number } {
  let minP = 1
  for (const s of groupsOf('checkNormality', samples, 1, 1)) {
    if (s.length < 3) continue
    let constant = true
    for (const v of s) if (v !== s[0]) constant = false
    if (constant) continue
    minP = Math.min(minP, shapiroWilk(s).p)
  }
  return { normal: minP >= alpha, minP }
}

/**
 * The equal-variance assumption of a pooled test: the Brown-Forsythe (median-centred Levene) test.
 *
 * @param groups - Two or more samples.
 * @param alpha - The level below which equality is rejected (default 0.05).
 * @returns Whether the variances look equal and the test's p-value (NaN when the test is undefined, as for constant
 * samples, whose spreads are all 0: `equal` is then true).
 * @example
 * ```ts
 * import { checkEqualVariances } from 'omai-engine/stats/tests'
 * checkEqualVariances([[1, 2, 3, 4], [2, 3, 4, 5]]).equal // => true
 * ```
 */
export function checkEqualVariances(
  groups: readonly TensorLike[],
  alpha = 0.05,
): { readonly equal: boolean; readonly pValue: number } {
  const p = levene(groups).pValue
  return { equal: !(p < alpha), pValue: p }
}

/**
 * The expected-count assumption of chi-square approximations (Cochran's rule): every expected count at least
 * `minimum` (default 5). Takes tensors (migration, D-009: `expected` was `ArrayLike<number>`; a table of any rank is read
 * in row-major order).
 *
 * @param expected - The expected counts: a vector or a table (a tensor, a typed array or a nested array).
 * @param minimum - The smallest acceptable expected count (default 5).
 * @returns Whether the rule holds and the smallest expected count.
 * @example
 * ```ts
 * import { checkExpectedCounts } from 'omai-engine/stats/tests'
 * checkExpectedCounts([6, 7, 4.5]).adequate // => false
 * ```
 */
export function checkExpectedCounts(
  expected: TensorLike,
  minimum = 5,
): { readonly adequate: boolean; readonly minExpected: number } {
  const counts = expected instanceof Float64Array ? expected : toFlat(asTensor(expected))
  let m = Number.POSITIVE_INFINITY
  for (let i = 0; i < counts.length; i++) m = Math.min(m, counts[i] ?? 0)
  return { adequate: m >= minimum, minExpected: m }
}

/**
 * Resolve the shared test options: the alternative (default two-sided) and the confidence level (default 0.95).
 *
 * @param options - The options.
 * @returns Both, checked.
 * @example
 * ```ts
 * import { resolveTestOptions } from 'omai-engine/stats/tests'
 * resolveTestOptions({}) // => { alternative: 'two-sided', level: 0.95 }
 * ```
 */
export function resolveTestOptions(options: TestOptions = {}): {
  readonly alternative: Alternative
  readonly level: number
} {
  return { alternative: resolveAlternative(options.alternative), level: resolveConfidenceLevel(options.level) }
}
