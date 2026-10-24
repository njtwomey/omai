import { type Estimate, logSumExpOf, type TestResult, WhiteboxError } from '../../base/index.ts'
import { hypergeometric } from '../../dists/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import { arange, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { chi2Sf, logGamma, mat, normPpf, rootAbove, rowsOf } from '../_util.ts'
import { cramersV } from '../effect-size.ts'
import { permutationTest, type ResamplingRunOptions } from '../resampling.ts'
import { type DivergenceLambda, powerDivergenceStatistic } from './distribution.ts'
import { binomTest } from './proportions.ts'
import {
  checkExpectedCounts,
  intervalEstimate,
  resolveAlternative,
  resolveTestOptions,
  type TestOptions,
  testResult,
} from './result.ts'

function table(name: string, x: TensorLike): { r: number; c: number; data: Float64Array } {
  const t = rowsOf(name, x)
  for (const v of t.data) if (!(v >= 0)) throw new WhiteboxError('value', `${name}: counts must be non-negative`)
  return { r: t.rows, c: t.cols, data: t.data }
}

function expected(r: number, c: number, data: Float64Array): Float64Array {
  const rows = new Float64Array(r)
  const cols = new Float64Array(c)
  let n = 0
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) {
      const v = data[i * c + j] ?? 0
      rows[i] = (rows[i] ?? 0) + v
      cols[j] = (cols[j] ?? 0) + v
      n += v
    }
  const e = new Float64Array(r * c)
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) e[i * c + j] = ((rows[i] ?? 0) * (cols[j] ?? 0)) / n
  return e
}

/**
 * The expected counts of a contingency table under independence, row sum times column sum over the total
 * (scipy.stats.contingency.expected_freq).
 *
 * @param x - An r x c table of counts.
 * @returns The expected table.
 * @example
 * ```ts
 * import { expectedFrequencies } from 'whitebox-engine/stats/tests'
 * expectedFrequencies([[10, 10], [20, 20]]) // => [[10, 10], [20, 20]]
 * ```
 */
export function expectedFrequencies(x: TensorLike): Tensor {
  const { r, c, data } = table('expectedFrequencies', x)
  return mat(expected(r, c, data), r, c)
}

/** Options of {@link chi2Contingency}. */
export interface ContingencyOptions {
  /** Yates' continuity correction when there is one degree of freedom (default true, as scipy). */
  readonly correction?: boolean
  /** The Cressie-Read power (default 1, Pearson's chi-square; 0 or `'log-likelihood'` is the G test). */
  readonly lambda?: DivergenceLambda
}

/**
 * The chi-square test of independence for a contingency table (scipy.stats.chi2_contingency), with Yates' correction for
 * 2 x 2 tables and any Cressie-Read power divergence; Cramer's V is the effect size and Cochran's expected-count rule is
 * checked.
 *
 * @param x - An r x c table of counts.
 * @param options - correction and lambda.
 * @returns The statistic, p-value (df (r - 1)(c - 1)) and effect size.
 * @example
 * ```ts
 * import { chi2Contingency } from 'whitebox-engine/stats/tests'
 * chi2Contingency([[10, 20], [20, 10]], { correction: false }).statistic // => 6.666667
 * ```
 */
export function chi2Contingency(x: TensorLike, options: ContingencyOptions = {}): TestResult {
  const { r, c, data } = table('chi2Contingency', x)
  const e = expected(r, c, data)
  const df = (r - 1) * (c - 1)
  const check = checkExpectedCounts(e)
  if (df === 0) return testResult({ test: 'chi-square test of independence', statistic: 0, pValue: 1, df: 0 })
  let observed = data
  if (df === 1 && options.correction !== false) {
    observed = Float64Array.from(data, (o, i) => {
      const diff = (e[i] ?? 0) - o
      return o + Math.sign(diff) * Math.min(0.5, Math.abs(diff))
    })
  }
  const lambda = options.lambda ?? 1
  const statistic = powerDivergenceStatistic(observed, e, lambda)
  const g = lambda === 0 || lambda === 'log-likelihood'
  return testResult({
    test: g ? 'G test of independence' : 'chi-square test of independence',
    statistic,
    pValue: chi2Sf(df, statistic),
    df,
    effectSize: cramersV(x),
    assumptions: { minExpected: check.minExpected, expectedCountsAdequate: check.adequate },
    method: df === 1 && options.correction !== false ? "Yates' continuity correction" : undefined,
  })
}

/**
 * The G test (log-likelihood ratio test) of independence: {@link chi2Contingency} with lambda 0, by default without
 * Yates' correction.
 *
 * @param x - An r x c table of counts.
 * @param options - correction (default false).
 * @returns The G statistic and its p-value.
 * @example
 * ```ts
 * import { gTest } from 'whitebox-engine/stats/tests'
 * gTest([[10, 20], [20, 10]]).statistic // => 6.795961
 * ```
 */
export function gTest(x: TensorLike, options: { readonly correction?: boolean } = {}): TestResult {
  return chi2Contingency(x, { correction: options.correction ?? false, lambda: 'log-likelihood' })
}

const lchoose = (n: number, k: number): number => logGamma(n + 1) - logGamma(k + 1) - logGamma(n - k + 1)

// The log pmf of Fisher's noncentral hypergeometric distribution over its support, at log odds ratio t.
function fisherNoncentral(M: number, n: number, N: number): { lo: number; hi: number; logp: (t: number) => number[] } {
  const lo = Math.max(0, N - (M - n))
  const hi = Math.min(n, N)
  const ks = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
  const base = ks.map((k) => lchoose(n, k) + lchoose(M - n, N - k))
  return {
    lo,
    hi,
    logp: (t) => {
      const w = base.map((b, i) => b + (ks[i] ?? 0) * t)
      const z = logSumExpOf(w)
      return w.map((v) => v - z)
    },
  }
}

function tailSum(logp: readonly number[], from: number, to: number): number {
  let s = 0
  for (let i = from; i <= to; i++) s += Math.exp(logp[i] ?? Number.NEGATIVE_INFINITY)
  return s
}

/** Options of {@link fisherExact}. */
export interface FisherOptions extends ResamplingRunOptions {
  readonly alternative?: TestOptions['alternative']
  /** For r x c tables: random tables with the observed margins (default 9999). */
  readonly nResamples?: number
}

/**
 * Fisher's exact test (scipy.stats.fisher_exact). For a 2 x 2 table: the hypergeometric probability of tables at least
 * as extreme, with the sample odds ratio as the statistic; two-sided p-values sum the tables no more likely than the
 * observed one. For larger tables (Freeman-Halton): a permutation test of the table probability with the margins
 * fixed, exact when feasible and otherwise from keyed random tables.
 *
 * @param x - A table of counts.
 * @param options - alternative (2 x 2 only), nResamples, key or seed.
 * @returns The odds ratio (2 x 2) or the table probability, and the p-value.
 * @example
 * ```ts
 * import { fisherExact } from 'whitebox-engine/stats/tests'
 * fisherExact([[8, 2], [1, 5]]).pValue // => 0.034965
 * ```
 */
export function fisherExact(x: TensorLike, options: FisherOptions = {}): TestResult {
  const { r, c, data } = table('fisherExact', x)
  if (r === 2 && c === 2) {
    const alternative = resolveAlternative(options.alternative)
    const [a = 0, b = 0, cc = 0, d = 0] = data
    const or = b > 0 && cc > 0 ? (a * d) / (b * cc) : Number.POSITIVE_INFINITY
    const n1 = a + b
    const n2 = cc + d
    const n = a + cc
    if (n1 === 0 || n2 === 0 || n === 0 || b + d === 0)
      return testResult({ test: "Fisher's exact test", statistic: Number.NaN, pValue: 1, alternative })
    const lo = Math.max(0, n - n2)
    const hi = Math.min(n1, n)
    const pmf = toFlat(hypergeometric(n1 + n2, n1, n).prob(arange(lo, hi + 1)))
    const at = (k: number): number => pmf[k - lo] ?? 0
    let p = 0
    if (alternative === 'less') for (let k = lo; k <= a; k++) p += at(k)
    else if (alternative === 'greater') for (let k = a; k <= hi; k++) p += at(k)
    else {
      const threshold = at(a) * (1 + 1e-14)
      for (let k = lo; k <= hi; k++) if (at(k) <= threshold) p += at(k)
    }
    return testResult({
      test: "Fisher's exact test",
      statistic: or,
      pValue: Math.min(1, p),
      alternative,
      estimate: oddsRatio(x, { kind: 'conditional' }),
    })
  }
  // r x c: untabulate and permute the column labels against the row labels.
  const rowsLabel: number[] = []
  const colsLabel: number[] = []
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++)
      for (let k = 0; k < (data[i * c + j] ?? 0); k++) {
        rowsLabel.push(i)
        colsLabel.push(j)
      }
  // log P(table | margins) = sum log r_i! + sum log c_j! - log n! - sum log n_ij!, with log k! tabulated.
  const total = rowsLabel.length
  const logFactorial = toFlat(lgamma(arange(1, total + 2)))
  const lf = (k: number): number => logFactorial[k] ?? 0
  let margins = -lf(total)
  for (let i = 0; i < r; i++) {
    let s = 0
    for (let j = 0; j < c; j++) s += data[i * c + j] ?? 0
    margins += lf(s)
  }
  for (let j = 0; j < c; j++) {
    let s = 0
    for (let i = 0; i < r; i++) s += data[i * c + j] ?? 0
    margins += lf(s)
  }
  const logProbability = (rl: Float64Array, cl: Float64Array): number => {
    const t = new Float64Array(r * c)
    for (let i = 0; i < rl.length; i++) {
      const at = (rl[i] ?? 0) * c + (cl[i] ?? 0)
      t[at] = (t[at] ?? 0) + 1
    }
    let s = margins
    for (const v of t) s -= lf(v)
    return s
  }
  const { alternative: _a, nResamples, ...run } = options
  const res = permutationTest([rowsLabel, colsLabel], logProbability, {
    ...run,
    permutationType: 'pairings',
    alternative: 'less',
    nResamples: nResamples ?? 9999,
  })
  return testResult({
    test: "Fisher's exact test (Freeman-Halton)",
    statistic: Math.exp(res.statistic),
    pValue: res.pValue,
    method: res.method,
  })
}

/**
 * McNemar's test of marginal homogeneity for a paired 2 x 2 table (statsmodels mcnemar): exact (the binomial test of
 * the smaller discordant count, default) or chi-square with an optional continuity correction.
 *
 * @param x - The 2 x 2 table of paired outcomes.
 * @param options - exact (default true) and correction (default true).
 * @returns The statistic and p-value.
 * @example
 * ```ts
 * import { mcnemar } from 'whitebox-engine/stats/tests'
 * mcnemar([[10, 5], [15, 20]], { exact: false }).statistic // => 4.05
 * ```
 */
export function mcnemar(
  x: TensorLike,
  options: { readonly exact?: boolean; readonly correction?: boolean } = {},
): TestResult {
  const { r, c, data } = table('mcnemar', x)
  if (r !== 2 || c !== 2) throw new WhiteboxError('shape', 'mcnemar: needs a 2 x 2 table')
  const n1 = data[1] ?? 0
  const n2 = data[2] ?? 0
  if (options.exact !== false) {
    const statistic = Math.min(n1, n2)
    const n = n1 + n2
    const p = n === 0 ? 1 : Math.min(1, 2 * binomTest(statistic, n, 0.5, { alternative: 'less' }).pValue)
    return testResult({ test: "McNemar's test", statistic, pValue: p, method: 'exact binomial' })
  }
  const corr = options.correction === false ? 0 : 1
  const statistic = (Math.abs(n1 - n2) - corr) ** 2 / (n1 + n2)
  return testResult({ test: "McNemar's test", statistic, pValue: chi2Sf(1, statistic), df: 1, method: 'chi-square' })
}

/**
 * Cochran's Q test that k paired binary outcomes have the same success rate (statsmodels cochrans_q).
 *
 * @param x - An n x k matrix of 0/1 outcomes (subjects in rows).
 * @returns Q and its p-value (df k - 1).
 * @example
 * ```ts
 * import { cochranQ } from 'whitebox-engine/stats/tests'
 * cochranQ([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0]]).statistic // => 3.5
 * ```
 */
export function cochranQ(x: TensorLike): TestResult {
  const { rows: n, cols: k, data } = rowsOf('cochranQ', x)
  let top = Number.NEGATIVE_INFINITY
  for (const v of data) if (v > top) top = v
  const rowS = new Float64Array(n)
  const colS = new Float64Array(k)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < k; j++)
      if ((data[i * k + j] ?? 0) === top) {
        rowS[i] = (rowS[i] ?? 0) + 1
        colS[j] = (colS[j] ?? 0) + 1
      }
  let total = 0
  let colSq = 0
  let rowSq = 0
  for (const v of colS) {
    total += v
    colSq += v * v
  }
  for (const v of rowS) rowSq += v * v
  const Q = ((k - 1) * (k * colSq - total * total)) / (k * total - rowSq)
  return testResult({ test: "Cochran's Q test", statistic: Q, pValue: chi2Sf(k - 1, Q), df: k - 1 })
}

/**
 * The odds ratio of a 2 x 2 table (scipy.stats.contingency.odds_ratio): the conditional maximum-likelihood estimate
 * under Fisher's noncentral hypergeometric distribution (default) with its exact interval, or the sample odds ratio ad
 * / bc with Woolf's logit interval.
 *
 * @param x - The 2 x 2 table.
 * @param options - kind (`'conditional'` or `'sample'`), level and alternative.
 * @returns The odds ratio with its interval.
 * @example
 * ```ts
 * import { oddsRatio } from 'whitebox-engine/stats/tests'
 * oddsRatio([[10, 5], [5, 10]], { kind: 'sample' }).value // => 4
 * ```
 */
export function oddsRatio(
  x: TensorLike,
  options: TestOptions & { readonly kind?: 'conditional' | 'sample' } = {},
): Estimate {
  const { alternative, level } = resolveTestOptions(options)
  const { r, c, data } = table('oddsRatio', x)
  if (r !== 2 || c !== 2) throw new WhiteboxError('shape', 'oddsRatio: needs a 2 x 2 table')
  const [a = 0, b = 0, cc = 0, d = 0] = data
  if (options.kind === 'sample') {
    const or = (a * d) / (b * cc)
    const se = Math.sqrt(1 / a + 1 / b + 1 / cc + 1 / d)
    const lor = Math.log(or)
    let interval: [number, number]
    if (alternative === 'two-sided') {
      const z = normPpf(0.5 + level / 2)
      interval = [Math.exp(lor - z * se), Math.exp(lor + z * se)]
    } else {
      const z = normPpf(level)
      interval =
        alternative === 'less' ? [0, Math.exp(lor + z * se)] : [Math.exp(lor - z * se), Number.POSITIVE_INFINITY]
    }
    return intervalEstimate(or, interval, level, { standardError: se, method: 'sample odds ratio, Woolf interval' })
  }
  const M = a + b + cc + d
  const n = a + b
  const N = a + cc
  const { lo, hi, logp } = fisherNoncentral(M, n, N)
  const xi = a - lo
  const mean = (t: number): number => {
    const lp = logp(t)
    let s = 0
    for (let i = 0; i < lp.length; i++) s += (lo + i) * Math.exp(lp[i] ?? Number.NEGATIVE_INFINITY)
    return s
  }
  // Monotone root finding on the log odds ratio, bracketed outwards from zero.
  const solve = (f: (t: number) => number): number => {
    const f0 = f(0)
    if (f0 === 0) return 0
    const f1 = f(1)
    const dir = Math.sign(f1) !== Math.sign(f0) || Math.abs(f1) < Math.abs(f0) ? 1 : -1
    return dir * rootAbove((s) => f(dir * s), 0, 1)
  }
  const estimate = a === lo ? 0 : a === hi ? Number.POSITIVE_INFINITY : Math.exp(solve((t) => mean(t) - a))
  const cdfAt = (t: number): number => tailSum(logp(t), 0, xi)
  const sfAt = (t: number): number => tailSum(logp(t), xi, hi - lo)
  const upper = (alpha: number): number =>
    a === hi || b * cc === 0 ? Number.POSITIVE_INFINITY : Math.exp(solve((t) => cdfAt(t) - alpha))
  const lower = (alpha: number): number => (a === lo || a * d === 0 ? 0 : Math.exp(solve((t) => sfAt(t) - alpha)))
  let interval: [number, number]
  if (alternative === 'two-sided') interval = [lower((1 - level) / 2), upper((1 - level) / 2)]
  else if (alternative === 'less') interval = [0, upper(1 - level)]
  else interval = [lower(1 - level), Number.POSITIVE_INFINITY]
  return intervalEstimate(estimate, interval, level, { method: 'conditional maximum likelihood' })
}

/**
 * The relative risk (risk ratio) of an exposed and a control group (scipy.stats.contingency.relative_risk), with Katz's
 * log interval.
 *
 * @param exposedCases - Cases among the exposed.
 * @param exposedTotal - Size of the exposed group.
 * @param controlCases - Cases among the controls.
 * @param controlTotal - Size of the control group.
 * @param options - level.
 * @returns The relative risk with its interval.
 * @example
 * ```ts
 * import { relativeRisk } from 'whitebox-engine/stats/tests'
 * relativeRisk(20, 100, 10, 100).value // => 2
 * ```
 */
export function relativeRisk(
  exposedCases: number,
  exposedTotal: number,
  controlCases: number,
  controlTotal: number,
  options: { readonly level?: number } = {},
): Estimate {
  const { level } = resolveTestOptions({ level: options.level ?? 0.95 })
  const a = exposedCases
  const c = controlCases
  if (a === 0 && c === 0) return { value: Number.NaN, interval: [Number.NaN, Number.NaN], level }
  if (a === 0) return { value: 0, interval: [0, Number.POSITIVE_INFINITY], level }
  if (c === 0) return { value: Number.POSITIVE_INFINITY, interval: [0, Number.POSITIVE_INFINITY], level }
  const rr = a / exposedTotal / (c / controlTotal)
  const se = Math.sqrt(1 / a - 1 / exposedTotal + 1 / c - 1 / controlTotal)
  const z = normPpf(0.5 + level / 2)
  return intervalEstimate(rr, [rr * Math.exp(-z * se), rr * Math.exp(z * se)], level, {
    standardError: se,
    method: 'Katz log interval',
  })
}
