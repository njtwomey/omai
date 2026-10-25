/**
 * Bayesian hypothesis testing: Bayes factors for the t test (the JZS prior), the binomial test and Pearson's
 * correlation (as pingouin), a Bayesian A/B comparison of two proportions with conjugate Beta posteriors and expected
 * losses, and the highest-density interval and region of practical equivalence (ROPE) for summarising posterior draws.
 */

import { type TestResult, WhiteboxError } from '../../base/index.ts'
import { betaBinomial, beta as betaDist, binomial } from '../../dists/index.ts'
import { betainc } from '../../numerics/special/index.ts'
import { beta as betaDraws, foldIn, type Key, key as makeKey } from '../../random/index.ts'
import { item, type TensorLike, toFlat } from '../../tensor/index.ts'
import { integrate, logGamma, sample, sortedCopy } from '../_util.ts'
import { quantile } from '../quantiles.ts'
import { testResult } from './result.ts'

// Generalised hypergeometric series pFq(a; b; z) for |z| < 1, summed in logs until the terms are negligible.
// TODO: replace with numerics/special's hypergeometric functions once they exist.
function hypergeometricSeries(a: readonly number[], b: readonly number[], z: number): number {
  if (!(Math.abs(z) < 1)) throw new WhiteboxError('domain', 'hypergeometric series: |z| must be below 1')
  let term = 1
  let sum = 1
  for (let k = 0; k < 100000; k++) {
    let ratio = z / (k + 1)
    for (const v of a) ratio *= v + k
    for (const v of b) ratio /= v + k
    term *= ratio
    sum += term
    if (Math.abs(term) < 1e-17 * Math.abs(sum)) break
  }
  return sum
}

/** Options of {@link bayesFactorTTest}. */
export interface BayesFactorTOptions {
  /** Size of the second sample, for an independent two-sample test. */
  readonly ny?: number
  readonly paired?: boolean
  /** Scale of the Cauchy prior on the standardised effect (default 0.707, "medium"). */
  readonly r?: number
}

/**
 * The JZS Bayes factor BF10 of a t test (Rouder et al. 2009; pingouin.bayesfactor_ttest): the evidence for a Cauchy(0,
 * r) prior on the standardised effect against the point null, by numerical integration over the prior's g scale.
 * Bayesian tests report evidence, not p-values: `pValue` is NaN and `statistic` is BF10.
 *
 * @param t - The t statistic.
 * @param nx - Size of the (first) sample.
 * @param options - ny, paired and r.
 * @returns BF10 (with BF01 and the posterior probability of H1 under equal prior odds in `assumptions`).
 * @reference J. N. Rouder, P. L. Speckman, D. Sun, R. D. Morey and G. Iverson, Bayesian t tests for accepting and
 * rejecting the null hypothesis, Psychonomic Bulletin and Review 16 (2009).
 * @example
 * ```ts
 * import { bayesFactorTTest } from 'whitebox-engine/stats/tests'
 * bayesFactorTTest(3.5, 20).statistic > 10 // => true
 * ```
 */
export function bayesFactorTTest(t: number, nx: number, options: BayesFactorTOptions = {}): TestResult {
  // biome-ignore lint/suspicious/noApproximativeNumericConstant: pingouin's default scale is exactly 0.707, not 1/sqrt(2).
  const r = options.r ?? 0.707
  const one = options.ny === undefined || options.ny === 1 || options.paired === true
  const n = one ? nx : (nx * (options.ny ?? 1)) / (nx + (options.ny ?? 1))
  const df = one ? nx - 1 : nx + (options.ny ?? 1) - 2
  const f = (g: number): number =>
    g <= 0
      ? 0
      : (1 + n * g * r * r) ** -0.5 *
        (1 + (t * t) / ((1 + n * g * r * r) * df)) ** (-(df + 1) / 2) *
        (2 * Math.PI) ** -0.5 *
        g ** -1.5 *
        Math.exp(-1 / (2 * g))
  const integral = integrate(f, 0, Number.POSITIVE_INFINITY)
  const bf10 = 1 / ((1 + (t * t) / df) ** (-(df + 1) / 2) / integral)
  return bayesResult('JZS Bayes factor t test', bf10, `Cauchy prior, r = ${r}`)
}

function bayesResult(test: string, bf10: number, method: string): TestResult {
  return testResult({
    test,
    statistic: bf10,
    pValue: Number.NaN,
    method,
    assumptions: { bf01: 1 / bf10, posteriorProbabilityH1: bf10 / (1 + bf10) },
  })
}

/**
 * The Bayes factor of a binomial test (pingouin.bayesfactor_binom): a Beta(a, b) prior on the success probability
 * against the point null p, BF10 = BetaBinomial(k; n, a, b) / Binomial(k; n, p).
 *
 * @param k - Successes.
 * @param n - Trials.
 * @param options - p (default 0.5) and the prior's a and b (default 1, uniform).
 * @returns BF10 as the statistic.
 * @example
 * ```ts
 * import { bayesFactorBinomial } from 'whitebox-engine/stats/tests'
 * bayesFactorBinomial(115, 200).statistic // => 0.835328
 * ```
 */
export function bayesFactorBinomial(
  k: number,
  n: number,
  options: { readonly p?: number; readonly a?: number; readonly b?: number } = {},
): TestResult {
  const p = options.p ?? 0.5
  if (!(p > 0 && p < 1)) throw new WhiteboxError('value', 'bayesFactorBinomial: p must lie in (0, 1)')
  if (!(Number.isInteger(k) && Number.isInteger(n) && k >= 0 && k <= n))
    throw new WhiteboxError('value', 'bayesFactorBinomial: k and n must be integers with k <= n')
  const a = options.a ?? 1
  const b = options.b ?? 1
  const bf10 = Math.exp(item(betaBinomial(n, a, b).logProb(k)) - item(binomial(n, p).logProb(k)))
  return bayesResult('Bayes factor binomial test', bf10, `Beta(${a}, ${b}) prior`)
}

/**
 * The Bayes factor of a Pearson correlation (pingouin.bayesfactor_pearson): Ly, Verhagen and Wagenmakers' (2016)
 * exact BF10 under a stretched Beta(1/kappa, 1/kappa) prior on rho (uniform for kappa 1), two- or one-sided, or
 * Wetzels and Wagenmakers' (2012) integral approximation.
 *
 * @param r - The sample correlation.
 * @param n - The sample size.
 * @param options - alternative, method (`'ly'`, default, or `'wetzels'`) and kappa (default 1).
 * @returns BF10 as the statistic.
 * @example
 * ```ts
 * import { bayesFactorPearson } from 'whitebox-engine/stats/tests'
 * bayesFactorPearson(0.6, 20).statistic > 10 // => true
 * ```
 */
export function bayesFactorPearson(
  r: number,
  n: number,
  options: {
    readonly alternative?: 'two-sided' | 'greater' | 'less'
    readonly method?: 'ly' | 'wetzels'
    readonly kappa?: number
  } = {},
): TestResult {
  if (!(r >= -1 && r <= 1)) throw new WhiteboxError('value', 'bayesFactorPearson: r must lie in [-1, 1]')
  const alternative = options.alternative ?? 'two-sided'
  const method = alternative === 'two-sided' ? (options.method ?? 'ly') : 'ly'
  let bf10: number
  if (method === 'wetzels') {
    const f = (g: number): number =>
      g <= 0
        ? 0
        : Math.exp(
            ((n - 2) / 2) * Math.log(1 + g) +
              (-(n - 1) / 2) * Math.log(1 + (1 - r * r) * g) -
              1.5 * Math.log(g) -
              n / (2 * g),
          )
    bf10 = (Math.sqrt(n / 2) / Math.sqrt(Math.PI)) * integrate(f, 0, Number.POSITIVE_INFINITY)
  } else {
    const k = options.kappa ?? 1
    const lbeta = 2 * logGamma(1 / k) - logGamma(2 / k)
    const hyper = Math.log(hypergeometricSeries([(n - 1) / 2, (n - 1) / 2], [(n + 2 / k) / 2], r * r))
    const logBf =
      (1 - 2 / k) * Math.LN2 +
      0.5 * Math.log(Math.PI) -
      lbeta +
      logGamma((n + 2 / k - 1) / 2) -
      logGamma((n + 2 / k) / 2) +
      hyper
    bf10 = Math.exp(logBf)
    if (alternative !== 'two-sided') {
      const h3 = hypergeometricSeries([1, n / 2, n / 2], [1.5, (2 + k * (n + 1)) / (2 * k)], r * r)
      const logTerm = 2 * (logGamma(n / 2) - logGamma((n - 1) / 2)) - lbeta
      const C = ((2 ** ((3 * k - 2) / k) * k * r) / (2 + (n - 1) * k)) * Math.exp(logTerm) * h3
      bf10 = Math.max(0, alternative === 'greater' ? bf10 + C : bf10 - C)
    }
  }
  return bayesResult(
    'Bayes factor Pearson correlation',
    bf10,
    method === 'ly' ? `Ly et al., kappa ${options.kappa ?? 1}` : 'Wetzels',
  )
}

/** Options of {@link bayesianAbTest}. */
export interface BayesianAbOptions {
  /** The Beta prior of both rates (default [1, 1]). */
  readonly prior?: readonly [number, number]
  /** Credible level of the interval of the difference (default 0.95). */
  readonly level?: number
  /** Draws for the interval of the difference (default 20000). */
  readonly nDraws?: number
  readonly key?: Key
  readonly seed?: number
}

/**
 * A Bayesian A/B comparison of two proportions with conjugate Beta posteriors: P(p_B > p_A) and the expected losses
 * E[max(p_A - p_B, 0)] (of choosing B) and E[max(p_B - p_A, 0)] (of choosing A) by numerical integration, and the
 * posterior mean of p_B - p_A with an equal-tailed credible interval from keyed draws. `statistic` is P(p_B > p_A);
 * `pValue` is NaN.
 *
 * @param kA - Successes in A.
 * @param nA - Trials in A.
 * @param kB - Successes in B.
 * @param nB - Trials in B.
 * @param options - prior, level, nDraws, key or seed.
 * @returns The probability that B is better, with the losses in `assumptions` and the difference in `estimate`.
 * @example
 * ```ts
 * import { bayesianAbTest } from 'whitebox-engine/stats/tests'
 * bayesianAbTest(20, 100, 35, 100).statistic > 0.99 // => true
 * ```
 */
export function bayesianAbTest(
  kA: number,
  nA: number,
  kB: number,
  nB: number,
  options: BayesianAbOptions = {},
): TestResult {
  const [a0, b0] = options.prior ?? [1, 1]
  const aA = a0 + kA
  const bA = b0 + nA - kA
  const aB = a0 + kB
  const bB = b0 + nB - kB
  const pdfA = betaDist(aA, bA)
  const pdfB = betaDist(aB, bB)
  const dens = (d: ReturnType<typeof betaDist>, x: number): number => (x <= 0 || x >= 1 ? 0 : item(d.prob(x)))
  const I = (a: number, b: number, x: number): number => item(betainc(a, b, Math.min(1, Math.max(0, x))))
  // P(B > A) = int f_A(x) (1 - F_B(x)) dx.
  const pBetter = integrate((x) => dens(pdfA, x) * (1 - I(aB, bB, x)), 0, 1)
  // E[(A - B)+] = int f_B(y) [E(A; A > y) - y P(A > y)] dy, with E(A; A > y) = mean_A (1 - I_y(aA + 1, bA)).
  const meanA = aA / (aA + bA)
  const meanB = aB / (aB + bB)
  const lossB = integrate((y) => dens(pdfB, y) * (meanA * (1 - I(aA + 1, bA, y)) - y * (1 - I(aA, bA, y))), 0, 1)
  const lossA = integrate((x) => dens(pdfA, x) * (meanB * (1 - I(aB + 1, bB, x)) - x * (1 - I(aB, bB, x))), 0, 1)
  const root = options.key ?? makeKey(options.seed ?? 0)
  const m = options.nDraws ?? 20000
  const da = toFlat(betaDraws(foldIn(root, 0), [m], aA, bA))
  const db = toFlat(betaDraws(foldIn(root, 1), [m], aB, bB))
  const diff = Float64Array.from(db, (v, i) => v - (da[i] ?? 0))
  const level = options.level ?? 0.95
  return testResult({
    test: 'Bayesian A/B test of proportions',
    statistic: pBetter,
    pValue: Number.NaN,
    alternative: 'greater',
    estimate: {
      value: meanB - meanA,
      interval: [quantile(diff, (1 - level) / 2), quantile(diff, (1 + level) / 2)],
      level,
      method: 'posterior mean and equal-tailed credible interval',
    },
    assumptions: { expectedLossChoosingB: lossB, expectedLossChoosingA: lossA },
    method: `Beta(${a0}, ${b0}) priors`,
  })
}

/**
 * The highest-density interval of posterior draws: the narrowest interval holding a fraction `level` of them. TODO:
 * alias the sampling domain's hdi once it exists.
 *
 * @param draws - Posterior draws.
 * @param level - The probability mass (default 0.95).
 * @returns The interval.
 * @example
 * ```ts
 * import { highestDensityInterval } from 'whitebox-engine/stats/tests'
 * highestDensityInterval([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], 0.8) // => [1, 8]
 * ```
 */
export function highestDensityInterval(draws: TensorLike, level = 0.95): [number, number] {
  const s = sortedCopy(sample('highestDensityInterval', draws))
  const n = s.length
  const k = Math.max(1, Math.floor(level * n))
  let best = 0
  let width = Number.POSITIVE_INFINITY
  for (let i = 0; i + k - 1 < n; i++) {
    const w = (s[i + k - 1] ?? 0) - (s[i] ?? 0)
    if (w < width) {
      width = w
      best = i
    }
  }
  return [s[best] ?? Number.NaN, s[best + k - 1] ?? Number.NaN]
}

/**
 * The region of practical equivalence (Kruschke 2018): the share of the posterior's highest-density interval that lies
 * inside [low, high]. The null value is accepted for practical purposes when the whole HDI lies inside the ROPE,
 * rejected when it lies wholly outside, and otherwise undecided. `statistic` is the share; `pValue` is NaN.
 *
 * @param draws - Posterior draws of the parameter.
 * @param options - low and high bounds of the ROPE and the HDI level (default 0.95).
 * @returns The share inside, with the decision in `assumptions` and the HDI in `estimate`.
 * @example
 * ```ts
 * import { rope } from 'whitebox-engine/stats/tests'
 * rope([0.01, -0.02, 0.03, 0.0, -0.01, 0.02], { low: -0.1, high: 0.1 }).assumptions?.decision // => 'accept'
 * ```
 */
export function rope(
  draws: TensorLike,
  options: { readonly low: number; readonly high: number; readonly level?: number },
): TestResult {
  if (!(options.low < options.high)) throw new WhiteboxError('value', 'rope: low must be below high')
  const level = options.level ?? 0.95
  const s = sortedCopy(sample('rope', draws))
  const [lo, hi] = highestDensityInterval(s, level)
  const inHdi = s.filter((v) => v >= lo && v <= hi)
  const inside = inHdi.filter((v) => v >= options.low && v <= options.high).length / inHdi.length
  const decision =
    lo >= options.low && hi <= options.high ? 'accept' : hi < options.low || lo > options.high ? 'reject' : 'undecided'
  let mean = 0
  for (const v of s) mean += v / s.length
  return testResult({
    test: 'region of practical equivalence',
    statistic: inside,
    pValue: Number.NaN,
    estimate: { value: mean, interval: [lo, hi], level, method: 'highest-density interval' },
    assumptions: { decision },
  })
}
