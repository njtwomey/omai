/**
 * Goodness-of-fit tests of whether a sample follows a hypothesised distribution, scipy.stats style: chi-square and
 * Cressie-Read power-divergence tests on counts, the Kolmogorov-Smirnov, Anderson-Darling and Cramer-von Mises tests
 * against a CDF, and the normality tests of Shapiro-Wilk, Jarque-Bera and Lilliefors.
 */

import { meanOf, OmaiError, type TestResult, varianceOf } from '../../base/index.ts'
import type { Distribution } from '../../dists/index.ts'
import { kolmogorov, logNdtr, ndtr, smirnov } from '../../numerics/special/index.ts'
import { exponential as expNoise, foldIn, type Key, key as makeKey, normal as normalNoise } from '../../random/index.ts'
import { item, type TensorLike, toFlat } from '../../tensor/index.ts'
import { chi2Sf, rootAbove, sample, sortedCopy, vec } from '../_util.ts'
import { kurtosis, skewness } from '../descriptive.ts'
import { cvmCdf } from './_cvm.ts'
import { kolmogn } from './_kolmogorov.ts'
import { shapiroWilk } from './_shapiro.ts'
import { resolveAlternative, type TestOptions, testResult } from './result.ts'

/** A hypothesised continuous distribution: a CDF of numbers, or any distribution from `dists`. */
export type CdfLike = ((x: number) => number) | Distribution

function cdfValues(cdf: CdfLike, x: Float64Array): number[] {
  if (typeof cdf === 'function') return Array.from(x, cdf)
  return toFlat(cdf.cdf(vec(x)))
}

/** The Cressie-Read power of a power-divergence statistic, by value or by scipy's name. */
export type DivergenceLambda =
  | number
  | 'pearson'
  | 'log-likelihood'
  | 'freeman-tukey'
  | 'mod-log-likelihood'
  | 'neyman'
  | 'cressie-read'

const LAMBDAS: Readonly<Record<string, number>> = {
  pearson: 1,
  'log-likelihood': 0,
  'freeman-tukey': -0.5,
  'mod-log-likelihood': -1,
  neyman: -2,
  'cressie-read': 2 / 3,
}

/**
 * The Cressie-Read power-divergence statistic 2 / (lambda (lambda + 1)) sum o ((o / e)^lambda - 1), with its limits
 * 2 sum o log(o / e) (lambda 0, the G statistic) and 2 sum e log(e / o) (lambda -1).
 *
 * @param observed - Observed counts.
 * @param expected - Expected counts.
 * @param lambda - The power (default 1, Pearson's chi-square).
 * @returns The statistic.
 * @example
 * ```ts
 * import { powerDivergenceStatistic } from 'omai-engine/stats/tests'
 * powerDivergenceStatistic([10, 20], [15, 15]) // => 3.333333
 * ```
 */
export function powerDivergenceStatistic(
  observed: ArrayLike<number>,
  expected: ArrayLike<number>,
  lambda: DivergenceLambda = 1,
): number {
  const l = typeof lambda === 'number' ? lambda : LAMBDAS[lambda]
  if (l === undefined) throw new OmaiError('value', `powerDivergence: unknown lambda ${String(lambda)}`)
  let s = 0
  for (let i = 0; i < observed.length; i++) {
    const o = observed[i] ?? 0
    const e = expected[i] ?? 0
    if (l === 1) s += (o - e) ** 2 / e
    else if (l === 0) s += o === 0 ? 0 : 2 * o * Math.log(o / e)
    else if (l === -1) s += e === 0 ? 0 : 2 * e * Math.log(e / o)
    else s += (2 / (l * (l + 1))) * o * ((o / e) ** l - 1)
  }
  return s
}

/** Options of {@link powerDivergence} and {@link chisquare}. */
export interface GoodnessOfFitOptions {
  /** Expected counts (default uniform with the observed total). */
  readonly expected?: TensorLike
  /** Degrees of freedom lost to estimated parameters (default 0). */
  readonly ddof?: number
}

/**
 * A power-divergence goodness-of-fit test (scipy.stats.power_divergence): any Cressie-Read statistic of observed
 * against expected counts, referred to chi-square with k - 1 - ddof degrees of freedom.
 *
 * @param observed - Observed counts.
 * @param options - expected, ddof and lambda (default 1).
 * @returns The statistic and its p-value.
 * @example
 * ```ts
 * import { powerDivergence } from 'omai-engine/stats/tests'
 * powerDivergence([16, 18, 16, 14, 12, 12], { lambda: 'log-likelihood' }).statistic // => 2.006573
 * ```
 */
export function powerDivergence(
  observed: TensorLike,
  options: GoodnessOfFitOptions & { readonly lambda?: DivergenceLambda } = {},
): TestResult {
  const o = sample('powerDivergence', observed)
  const k = o.length
  const total = o.reduce((s, v) => s + v, 0)
  const e =
    options.expected === undefined ? new Float64Array(k).fill(total / k) : sample('powerDivergence', options.expected)
  if (e.length !== k) throw new OmaiError('shape', 'powerDivergence: expected must match observed')
  if (!(o.every((v) => v >= 0) && e.every((v) => v >= 0)))
    throw new OmaiError('value', 'powerDivergence: counts must be non-negative')
  const te = e.reduce((s, v) => s + v, 0)
  if (Math.abs(te - total) > 1e-8 * Math.max(Math.abs(total), Math.abs(te)))
    throw new OmaiError('value', 'powerDivergence: observed and expected totals differ')
  const df = k - 1 - (options.ddof ?? 0)
  const statistic = powerDivergenceStatistic(o, e, options.lambda ?? 1)
  return testResult({
    test: 'power-divergence goodness-of-fit test',
    statistic,
    pValue: chi2Sf(df, statistic),
    df,
    assumptions: { minExpected: e.reduce((m, v) => Math.min(m, v), Number.POSITIVE_INFINITY) },
  })
}

/**
 * Pearson's chi-square goodness-of-fit test (scipy.stats.chisquare).
 *
 * @param observed - Observed counts.
 * @param options - expected (default uniform) and ddof.
 * @returns The statistic and its p-value.
 * @example
 * ```ts
 * import { chisquare } from 'omai-engine/stats/tests'
 * chisquare([16, 18, 16, 14, 12, 12]).statistic // => 2
 * ```
 */
export function chisquare(observed: TensorLike, options: GoodnessOfFitOptions = {}): TestResult {
  return { ...powerDivergence(observed, { ...options, lambda: 1 }), test: 'chi-square goodness-of-fit test' }
}

/** How {@link kstest} computes its p-value (scipy's `method`). */
export type KsMethod = 'auto' | 'exact' | 'asymp' | 'approx'

/**
 * The one-sample Kolmogorov-Smirnov test (scipy.stats.ks_1samp): the largest distance between the empirical CDF and a
 * hypothesised CDF, with the exact two-sided distribution (Simard and L'Ecuyer's choice of methods), the asymptotic
 * Kolmogorov distribution, or Smirnov's one-sided distribution.
 *
 * @param x - The sample.
 * @param cdf - The hypothesised CDF (a function or a distribution).
 * @param options - alternative and method (default `'auto'`, exact).
 * @returns D (D+ or D- for one-sided alternatives) and its p-value.
 * @example
 * ```ts
 * import { kstest } from 'omai-engine/stats/tests'
 * import { normal } from 'omai-engine/dists'
 * kstest([-1.2, -0.3, 0.1, 0.6, 1.4], normal()).statistic // => 0.182089
 * ```
 */
export function kstest(
  x: TensorLike,
  cdf: CdfLike,
  options: TestOptions & { readonly method?: KsMethod } = {},
): TestResult {
  const alternative = resolveAlternative(options.alternative)
  const s = sortedCopy(sample('kstest', x, 1, true))
  const n = s.length
  const F = cdfValues(cdf, s)
  let dplus = Number.NEGATIVE_INFINITY
  let dminus = Number.NEGATIVE_INFINITY
  for (let i = 0; i < n; i++) {
    dplus = Math.max(dplus, (i + 1) / n - (F[i] ?? 0))
    dminus = Math.max(dminus, (F[i] ?? 0) - i / n)
  }
  const smirnovSf = (d: number): number => item(smirnov(n, d))
  if (alternative !== 'two-sided') {
    const d = alternative === 'greater' ? dplus : dminus
    return testResult({ test: 'Kolmogorov-Smirnov test', statistic: d, pValue: Math.min(1, smirnovSf(d)), alternative })
  }
  const D = Math.max(dplus, dminus)
  const method = options.method ?? 'auto'
  let p: number
  if (method === 'asymp') p = item(kolmogorov(D * Math.sqrt(n)))
  else if (method === 'approx') p = 2 * smirnovSf(D)
  else p = kolmogn(n, D, true)
  return testResult({
    test: 'Kolmogorov-Smirnov test',
    statistic: D,
    pValue: Math.min(1, Math.max(0, p)),
    alternative,
    method: method === 'auto' ? 'exact' : method,
  })
}

/** A distribution family for {@link andersonDarling}. */
export type AndersonDist = 'norm' | 'expon' | 'logistic' | 'gumbel_r' | 'gumbel_l'

const AVALS: Readonly<Record<AndersonDist, readonly number[]>> = {
  norm: [0.561, 0.631, 0.752, 0.873, 1.035],
  expon: [0.916, 1.062, 1.321, 1.591, 1.959],
  logistic: [0.426, 0.563, 0.66, 0.769, 0.906, 1.01],
  gumbel_r: [0.474, 0.637, 0.757, 0.877, 1.038],
  gumbel_l: [0.474, 0.637, 0.757, 0.877, 1.038],
}
const SIG: Readonly<Record<AndersonDist, readonly number[]>> = {
  norm: [15, 10, 5, 2.5, 1],
  expon: [15, 10, 5, 2.5, 1],
  logistic: [25, 10, 5, 2.5, 1, 0.5],
  gumbel_r: [25, 10, 5, 2.5, 1],
  gumbel_l: [25, 10, 5, 2.5, 1],
}

// numpy.around to 3 decimals (round half to even).
function round3(x: number): number {
  const y = x * 1000
  const r = Math.round(y)
  return (Math.abs(y % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r) / 1000
}

// Gumbel (right-skewed) maximum likelihood, as scipy.stats.gumbel_r.fit solves it.
function gumbelFit(x: Float64Array): { loc: number; scale: number } {
  const m = meanOf(x)
  const weighted = (s: number): number => {
    let top = Number.NEGATIVE_INFINITY
    for (const v of x) top = Math.max(top, -v / s)
    let num = 0
    let den = 0
    for (const v of x) {
      const w = Math.exp(-v / s - top)
      num += v * w
      den += w
    }
    return num / den
  }
  const sd = Math.sqrt(varianceOf(x, 1))
  const scale = rootAbove((s) => m - weighted(s) - s, sd * 1e-3, sd)
  let top = Number.NEGATIVE_INFINITY
  for (const v of x) top = Math.max(top, -v / scale)
  let lse = 0
  for (const v of x) lse += Math.exp(-v / scale - top)
  return { loc: -scale * (top + Math.log(lse) - Math.log(x.length)), scale }
}

// Logistic maximum likelihood (the score equations scipy.stats.anderson solves), by Newton's method.
function logisticFit(x: Float64Array): { loc: number; scale: number } {
  let a = meanOf(x)
  let b = Math.sqrt(varianceOf(x, 1))
  const n = x.length
  for (let it = 0; it < 200; it++) {
    const f = (aa: number, bb: number): [number, number] => {
      let f1 = 0
      let f2 = 0
      for (const v of x) {
        const t = (v - aa) / bb
        const e = Math.exp(t)
        f1 += 1 / (1 + e)
        f2 += (t * (1 - e)) / (1 + e)
      }
      return [f1 - 0.5 * n, f2 + n]
    }
    const [f1, f2] = f(a, b)
    const h = 1e-6
    const [f1a, f2a] = f(a + h * b, b)
    const [f1b, f2b] = f(a, b * (1 + h))
    const j11 = (f1a - f1) / (h * b)
    const j21 = (f2a - f2) / (h * b)
    const j12 = (f1b - f1) / (h * b)
    const j22 = (f2b - f2) / (h * b)
    const det = j11 * j22 - j12 * j21
    const da = (f1 * j22 - f2 * j12) / det
    const db = (j11 * f2 - j21 * f1) / det
    a -= da
    b -= db
    if (Math.abs(da) < 1e-14 * (1 + Math.abs(a)) && Math.abs(db) < 1e-14 * b) break
  }
  return { loc: a, scale: b }
}

/**
 * The Anderson-Darling test that a sample comes from a family with estimated parameters (scipy.stats.anderson): A^2
 * = -n - sum (2i - 1) / n (log F(x_(i)) + log S(x_(n+1-i))), with Stephens' critical values and the p-value interpolated
 * between them (scipy's `method='interpolate'`, so it is capped at the table's ends).
 *
 * @param x - The sample.
 * @param options - dist (default `'norm'`).
 * @returns A^2 and its interpolated p-value; `assumptions` holds the critical values at the table's levels.
 * @example
 * ```ts
 * import { andersonDarling } from 'omai-engine/stats/tests'
 * andersonDarling([2.1, 1.9, 2.4, 2.0, 1.7, 2.2, 2.3, 1.8]).pValue // => 0.15
 * ```
 */
export function andersonDarling(x: TensorLike, options: { readonly dist?: AndersonDist } = {}): TestResult {
  const dist = options.dist ?? 'norm'
  const a = sample('andersonDarling', x, 3)
  const y = sortedCopy(a)
  const N = y.length
  let logcdf: number[]
  let logsf: number[]
  let critical: number[]
  const avals = AVALS[dist]
  if (dist === 'norm') {
    const m = meanOf(a)
    const s = Math.sqrt(varianceOf(a, 1))
    const w = Array.from(y, (v) => (v - m) / s)
    logcdf = toFlat(logNdtr(w))
    logsf = toFlat(logNdtr(w.map((v) => -v)))
    critical = avals.map((v) => round3(v / (1 + 0.75 / N + 2.25 / N / N)))
  } else if (dist === 'expon') {
    const m = meanOf(a)
    const w = Array.from(y, (v) => v / m)
    logcdf = w.map((v) => Math.log(-Math.expm1(-v)))
    logsf = w.map((v) => -v)
    critical = avals.map((v) => round3(v / (1 + 0.6 / N)))
  } else if (dist === 'logistic') {
    const { loc, scale } = logisticFit(a)
    const w = Array.from(y, (v) => (v - loc) / scale)
    logcdf = w.map((v) => -Math.log1p(Math.exp(-v)))
    logsf = w.map((v) => -Math.log1p(Math.exp(v)))
    critical = avals.map((v) => round3(v / (1 + 0.25 / N)))
  } else {
    const left = dist === 'gumbel_l'
    const fit = gumbelFit(left ? a.map((v) => -v) : a)
    const loc = left ? -fit.loc : fit.loc
    const w = Array.from(y, (v) => (v - loc) / fit.scale)
    // gumbel_r: F = exp(-exp(-w)); gumbel_l: F = 1 - exp(-exp(w)).
    logcdf = w.map((v) => (left ? Math.log(-Math.expm1(-Math.exp(v))) : -Math.exp(-v)))
    logsf = w.map((v) => (left ? -Math.exp(v) : Math.log(-Math.expm1(-Math.exp(-v)))))
    critical = avals.map((v) => round3(v / (1 + 0.2 / Math.sqrt(N))))
  }
  let s = 0
  for (let i = 1; i <= N; i++) s += ((2 * i - 1) / N) * ((logcdf[i - 1] ?? 0) + (logsf[N - i] ?? 0))
  const A2 = -N - s
  const sig = SIG[dist].map((v) => v / 100)
  // numpy.interp(A2, critical, sig): clamped linear interpolation.
  let p: number
  if (A2 <= (critical[0] ?? 0)) p = sig[0] ?? 1
  else if (A2 >= (critical[critical.length - 1] ?? 0)) p = sig[sig.length - 1] ?? 0
  else {
    let j = 0
    while (A2 > (critical[j + 1] ?? 0)) j++
    const c0 = critical[j] ?? 0
    const c1 = critical[j + 1] ?? 0
    p = (sig[j] ?? 0) + ((A2 - c0) / (c1 - c0)) * ((sig[j + 1] ?? 0) - (sig[j] ?? 0))
  }
  const assumptions: Record<string, number> = {}
  SIG[dist].forEach((level, i) => {
    assumptions[`critical${level}`] = critical[i] ?? Number.NaN
  })
  return testResult({
    test: 'Anderson-Darling test',
    statistic: A2,
    pValue: p,
    alternative: 'greater',
    assumptions,
    method: `${dist}, p-value interpolated in Stephens' table`,
  })
}

/**
 * The Shapiro-Wilk test of normality (scipy.stats.shapiro), by Royston's algorithm AS R94 (valid for 3 <= n <= 5000).
 *
 * @param x - The sample.
 * @returns W and its p-value.
 * @index function Shapiro-Wilk normality test.
 * @example
 * ```ts
 * import { shapiro } from 'omai-engine/stats/tests'
 * shapiro([2.1, 1.9, 2.4, 2.0, 1.7, 2.2, 2.3, 1.8]).statistic // => 0.974858
 * ```
 */
export function shapiro(x: TensorLike): TestResult {
  const a = sample('shapiro', x, 3)
  const { w, p } = shapiroWilk(a)
  return testResult({
    test: 'Shapiro-Wilk test',
    statistic: w,
    pValue: p,
    alternative: 'less',
    assumptions: { validSampleSize: a.length <= 5000 },
  })
}

/**
 * The Jarque-Bera test of normality (scipy.stats.jarque_bera): n / 6 (S^2 + K^2 / 4) from the sample skewness and
 * excess kurtosis, against chi-square with 2 degrees of freedom.
 *
 * @param x - The sample.
 * @returns JB and its asymptotic p-value.
 * @example
 * ```ts
 * import { jarqueBera } from 'omai-engine/stats/tests'
 * jarqueBera([1, 2, 3, 4, 5]).statistic // => 0.352083
 * ```
 */
export function jarqueBera(x: TensorLike): TestResult {
  const a = sample('jarqueBera', x, 2)
  const S = skewness(a)
  const Kt = kurtosis(a)
  const jb = (a.length / 6) * (S * S + (Kt * Kt) / 4)
  return testResult({ test: 'Jarque-Bera test', statistic: jb, pValue: chi2Sf(2, jb), df: 2, alternative: 'greater' })
}

/** Options of {@link lilliefors}. */
export interface LillieforsOptions {
  /** The family with estimated parameters (default `'norm'`). */
  readonly dist?: 'norm' | 'exp'
  /** Simulated samples for the p-value when the Dallal-Wilkinson approximation does not apply (default 10000). */
  readonly nResamples?: number
  /** Key (or seed) of the simulation (default seed 0). */
  readonly key?: Key
  readonly seed?: number
}

function lillieforsStatistic(x: Float64Array, dist: 'norm' | 'exp'): number {
  const n = x.length
  const s = sortedCopy(x)
  let F: number[]
  if (dist === 'norm') {
    const m = meanOf(x)
    const sd = Math.sqrt(varianceOf(x, 1))
    F = toFlat(ndtr(Array.from(s, (v) => (v - m) / sd)))
  } else {
    const m = meanOf(x)
    F = Array.from(s, (v) => -Math.expm1(-v / m))
  }
  let d = 0
  for (let i = 0; i < n; i++) d = Math.max(d, (i + 1) / n - (F[i] ?? 0), (F[i] ?? 0) - i / n)
  return d
}

/**
 * The Lilliefors test (statsmodels lilliefors): the Kolmogorov-Smirnov statistic against a normal (or exponential)
 * distribution with estimated parameters. For the normal family, p-values below 0.1 use Dallal and Wilkinson's (1986)
 * analytic approximation (statsmodels' `'approx'`); otherwise the p-value is simulated from the null with a key.
 *
 * @param x - The sample (n >= 4 for the normal, 3 for the exponential).
 * @param options - dist, nResamples and key or seed.
 * @returns D and its p-value.
 * @example
 * ```ts
 * import { lilliefors } from 'omai-engine/stats/tests'
 * lilliefors([2.1, 1.9, 2.4, 2.0, 1.7, 2.2, 2.3, 1.8]).statistic // => 0.104854
 * ```
 */
export function lilliefors(x: TensorLike, options: LillieforsOptions = {}): TestResult {
  const dist = options.dist ?? 'norm'
  const a = sample('lilliefors', x, dist === 'norm' ? 4 : 3)
  const n = a.length
  const d = lillieforsStatistic(a, dist)
  let p = Number.NaN
  let method = 'simulation'
  if (dist === 'norm') {
    let dd = d
    let nn = n
    if (nn > 100) {
      dd *= (nn / 100) ** 0.49
      nn = 100
    }
    const approx = Math.exp(
      -7.01256 * dd * dd * (nn + 2.78019) +
        2.99587 * dd * Math.sqrt(nn + 2.78019) -
        0.122119 +
        0.974598 / Math.sqrt(nn) +
        1.67997 / nn,
    )
    if (approx <= 0.1) {
      p = approx
      method = 'Dallal-Wilkinson approximation'
    }
  }
  if (Number.isNaN(p)) {
    const reps = options.nResamples ?? 10000
    const root = options.key ?? makeKey(options.seed ?? 0)
    let count = 0
    for (let r = 0; r < reps; r++) {
      const k = foldIn(root, r)
      const sim = dist === 'norm' ? normalNoise(k, [n]).data : expNoise(k, [n]).data
      if (lillieforsStatistic(Float64Array.from(sim), dist) >= d) count++
    }
    p = (count + 1) / (reps + 1)
  }
  return testResult({ test: 'Lilliefors test', statistic: d, pValue: p, alternative: 'greater', method })
}

/**
 * The one-sample Cramer-von Mises test (scipy.stats.cramervonmises): W^2 = 1 / (12 n) + sum ((2i - 1) / (2n) -
 * F(x_(i)))^2, with Csorgo and Faraway's finite-sample distribution.
 *
 * @param x - The sample.
 * @param cdf - The hypothesised CDF.
 * @returns W^2 and its p-value.
 * @example
 * ```ts
 * import { cramervonmises } from 'omai-engine/stats/tests'
 * cramervonmises([0.1, 0.4, 0.5, 0.8, 0.95], (u) => u).statistic // => 0.039167
 * ```
 */
export function cramervonmises(x: TensorLike, cdf: CdfLike): TestResult {
  const s = sortedCopy(sample('cramervonmises', x, 2, true))
  const n = s.length
  const F = cdfValues(cdf, s)
  let w = 1 / (12 * n)
  for (let i = 0; i < n; i++) w += ((2 * (i + 1) - 1) / (2 * n) - (F[i] ?? 0)) ** 2
  return testResult({
    test: 'Cramer-von Mises test',
    statistic: w,
    pValue: Math.max(0, 1 - cvmCdf(w, n)),
    alternative: 'greater',
  })
}
