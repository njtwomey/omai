/**
 * Tests of means, scipy.stats and pingouin style: one-sample, independent (Student or Welch) and paired t tests, z
 * tests, one-way ANOVA and Welch's ANOVA, Tukey's honestly significant difference for all pairwise comparisons, and
 * repeated-measures ANOVA with Mauchly's sphericity test and its corrections. Results report effect sizes and
 * assumption checks alongside the p-values.
 */

import { type Alternative, meanOf, type TestResult, varianceOf, WhiteboxError } from '../../base/index.ts'
import { studentisedRange } from '../../dists/index.ts'
import { item, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { covarianceOfColumns, eigvalsSym, logdetSpd, matmul, orthonormalContrasts, transpose } from '../_linear.ts'
import {
  chi2Sf,
  fSf,
  groupsOf,
  mat,
  normCdf,
  normPpf,
  normSf,
  paired,
  rowsOf,
  sample,
  tCdf,
  tPpf,
  tSf,
} from '../_util.ts'
import { cohensD, etaSquared } from '../effect-size.ts'
import {
  checkEqualVariances,
  checkNormality,
  intervalEstimate,
  pValueFromTails,
  resolveTestOptions,
  type TestOptions,
  testResult,
  waldInterval,
} from './result.ts'

function tTest(
  t: number,
  df: number,
  alternative: Alternative,
  estimateValue: number,
  se: number,
  level: number,
): { pValue: number; interval: [number, number] } {
  return {
    pValue: pValueFromTails(alternative, tCdf(df, t), tSf(df, t)),
    interval: waldInterval(estimateValue, se, alternative, level, (p) => tPpf(df, p)),
  }
}

/**
 * The one-sample t test of a mean (scipy.stats.ttest_1samp), with the confidence interval of the mean and Cohen's d.
 *
 * @param x - The sample (n >= 2).
 * @param popmean - The hypothesised mean (default 0).
 * @param options - alternative and level.
 * @returns t, its p-value (df n - 1), the mean with its interval, Cohen's d and the normality check.
 * @example
 * ```ts
 * import { ttest1samp } from 'whitebox-engine/stats/tests'
 * ttest1samp([1, 2, 3, 4, 5], 2).statistic // => 1.414214
 * ```
 */
export function ttest1samp(x: TensorLike, popmean = 0, options: TestOptions = {}): TestResult {
  const { alternative, level } = resolveTestOptions(options)
  const a = sample('ttest1samp', x, 2)
  const n = a.length
  const m = meanOf(a)
  const se = Math.sqrt(varianceOf(a, 1) / n)
  const t = (m - popmean) / se
  const { pValue, interval } = tTest(t, n - 1, alternative, m, se, level)
  return testResult({
    test: 'one-sample t test',
    statistic: t,
    pValue,
    alternative,
    df: n - 1,
    estimate: intervalEstimate(m, interval, level, { standardError: se, method: 't interval' }),
    effectSize: cohensD(a, popmean, { level }),
    assumptions: { normal: checkNormality([a]).normal },
  })
}

/** Options of {@link ttestInd}. */
export interface TtestIndOptions extends TestOptions {
  /** Pool the variances (Student, default true) or not (Welch). */
  readonly equalVar?: boolean
}

/**
 * The independent two-sample t test (scipy.stats.ttest_ind): Student's pooled test or, with `equalVar: false`, Welch's
 * test with Satterthwaite's degrees of freedom; reports the mean difference with its interval, Cohen's d, and the
 * normality and equal-variance checks.
 *
 * @param x - First sample.
 * @param y - Second sample.
 * @param options - equalVar (default true), alternative and level.
 * @returns The t statistic, p-value, degrees of freedom, estimate and effect size.
 * @example
 * ```ts
 * import { ttestInd } from 'whitebox-engine/stats/tests'
 * ttestInd([1, 2, 3, 4], [3, 4, 5, 6]).statistic // => -2.19089
 * ttestInd([1, 2, 3, 4], [3, 4, 5, 6]).df // => 6
 * ```
 */
export function ttestInd(x: TensorLike, y: TensorLike, options: TtestIndOptions = {}): TestResult {
  const { alternative, level } = resolveTestOptions(options)
  const a = sample('ttestInd', x, 2)
  const b = sample('ttestInd', y, 2)
  const nx = a.length
  const ny = b.length
  const vx = varianceOf(a, 1)
  const vy = varianceOf(b, 1)
  const diff = meanOf(a) - meanOf(b)
  const welch = options.equalVar === false
  let se: number
  let df: number
  if (welch) {
    const ax = vx / nx
    const ay = vy / ny
    se = Math.sqrt(ax + ay)
    df = (ax + ay) ** 2 / (ax ** 2 / (nx - 1) + ay ** 2 / (ny - 1))
  } else {
    df = nx + ny - 2
    se = Math.sqrt((((nx - 1) * vx + (ny - 1) * vy) / df) * (1 / nx + 1 / ny))
  }
  const t = diff / se
  const { pValue, interval } = tTest(t, df, alternative, diff, se, level)
  return testResult({
    test: welch ? "Welch's t test" : "Student's t test",
    statistic: t,
    pValue,
    alternative,
    df,
    estimate: intervalEstimate(diff, interval, level, { standardError: se, method: 'difference of means' }),
    effectSize: cohensD(a, b, { level }),
    assumptions: { normal: checkNormality([a, b]).normal, equalVariances: checkEqualVariances([a, b]).equal },
  })
}

/**
 * The paired t test (scipy.stats.ttest_rel): the one-sample t test of the differences x - y.
 *
 * @param x - First measurements.
 * @param y - Second measurements (same length).
 * @param options - alternative and level.
 * @returns t, p-value (df n - 1), the mean difference with its interval and Cohen's d_av.
 * @example
 * ```ts
 * import { ttestRel } from 'whitebox-engine/stats/tests'
 * ttestRel([2, 4, 6, 9], [1, 2, 4, 5]).statistic // => 3.576237
 * ```
 */
export function ttestRel(x: TensorLike, y: TensorLike, options: TestOptions = {}): TestResult {
  const { alternative, level } = resolveTestOptions(options)
  const [a, b] = paired('ttestRel', x, y, 2)
  const d = Float64Array.from(a, (v, i) => v - (b[i] ?? 0))
  const n = d.length
  const m = meanOf(d)
  const se = Math.sqrt(varianceOf(d, 1) / n)
  const t = m / se
  const { pValue, interval } = tTest(t, n - 1, alternative, m, se, level)
  return testResult({
    test: 'paired t test',
    statistic: t,
    pValue,
    alternative,
    df: n - 1,
    estimate: intervalEstimate(m, interval, level, { standardError: se, method: 'mean difference' }),
    effectSize: cohensD(a, b, { paired: true, level }),
    assumptions: { normal: checkNormality([d]).normal },
  })
}

/** Options of {@link ztest}. */
export interface ZtestOptions extends TestOptions {
  /** The hypothesised mean or mean difference (default 0). */
  readonly value?: number
  /** For two samples: pool the variances (default `'pooled'`, statsmodels) or not (`'unequal'`). */
  readonly usevar?: 'pooled' | 'unequal'
  /** Known population standard deviation(s); default the sample estimates (ddof 1). */
  readonly sigma?: number | readonly [number, number]
}

/**
 * The z test of a mean or a difference of means (statsmodels.stats.weightstats.ztest): like the t test but referred
 * to the standard normal, with the sample standard deviation (ddof 1) or a known sigma.
 *
 * @param x - First sample.
 * @param y - Second sample, or undefined for the one-sample test.
 * @param options - value, usevar, sigma, alternative and level.
 * @returns z, its p-value and the estimate with its normal interval.
 * @example
 * ```ts
 * import { ztest } from 'whitebox-engine/stats/tests'
 * ztest([1, 2, 3, 4, 5], undefined, { value: 2, sigma: 1 }).statistic // => 2.236068
 * ```
 */
export function ztest(x: TensorLike, y?: TensorLike, options: ZtestOptions = {}): TestResult {
  const { alternative, level } = resolveTestOptions(options)
  const value = options.value ?? 0
  const a = sample('ztest', x, 2)
  let est: number
  let se: number
  const sig = options.sigma
  if (y === undefined) {
    est = meanOf(a)
    const s = typeof sig === 'number' ? sig : Math.sqrt(varianceOf(a, 1))
    se = s / Math.sqrt(a.length)
  } else {
    const b = sample('ztest', y, 2)
    est = meanOf(a) - meanOf(b)
    const nx = a.length
    const ny = b.length
    if (sig !== undefined) {
      const [s1, s2] = typeof sig === 'number' ? [sig, sig] : sig
      se = Math.sqrt((s1 * s1) / nx + (s2 * s2) / ny)
    } else if (options.usevar === 'unequal') se = Math.sqrt(varianceOf(a, 1) / nx + varianceOf(b, 1) / ny)
    else {
      const pooled = ((nx - 1) * varianceOf(a, 1) + (ny - 1) * varianceOf(b, 1)) / (nx + ny - 2)
      se = Math.sqrt(pooled * (1 / nx + 1 / ny))
    }
  }
  const z = (est - value) / se
  const pValue = pValueFromTails(alternative, normCdf(z), normSf(z))
  return testResult({
    test: y === undefined ? 'one-sample z test' : 'two-sample z test',
    statistic: z,
    pValue,
    alternative,
    estimate: intervalEstimate(est, waldInterval(est, se, alternative, level, normPpf), level, {
      standardError: se,
      method: 'normal interval',
    }),
  })
}

/**
 * One-way analysis of variance (scipy.stats.f_oneway): F = MS_between / MS_within on (k - 1, N - k) degrees of
 * freedom, with eta squared and its interval, and the normality and equal-variance checks.
 *
 * @param groups - Two or more samples.
 * @returns The F statistic, p-value, degrees of freedom and effect size.
 * @example
 * ```ts
 * import { fOneway } from 'whitebox-engine/stats/tests'
 * fOneway([[1, 2, 3], [4, 5, 6]]).statistic // => 13.5
 * ```
 */
export function fOneway(groups: readonly TensorLike[]): TestResult {
  const gs = groupsOf('fOneway', groups, 1)
  const all = gs.flatMap((g) => Array.from(g))
  const grand = meanOf(all)
  let ssb = 0
  let ssw = 0
  for (const g of gs) {
    const m = meanOf(g)
    ssb += g.length * (m - grand) ** 2
    for (const v of g) ssw += (v - m) ** 2
  }
  const k = gs.length
  const N = all.length
  const F = ssb / (k - 1) / (ssw / (N - k))
  return testResult({
    test: 'one-way ANOVA',
    statistic: F,
    pValue: fSf(k - 1, N - k, F),
    df: [k - 1, N - k],
    effectSize: etaSquared(gs),
    assumptions: { normal: checkNormality(gs).normal, equalVariances: checkEqualVariances(gs).equal },
  })
}

/**
 * Welch's one-way ANOVA for unequal variances (pingouin.welch_anova): the weighted F statistic with Welch's (1951)
 * denominator degrees of freedom; the effect size is eta squared.
 *
 * @param groups - Two or more samples (each of size at least 2).
 * @returns The F statistic, p-value and degrees of freedom [k - 1, df2].
 * @example
 * ```ts
 * import { welchAnova } from 'whitebox-engine/stats/tests'
 * welchAnova([[1, 2, 3], [4, 5, 6]]).statistic // => 13.5
 * ```
 */
export function welchAnova(groups: readonly TensorLike[]): TestResult {
  const gs = groupsOf('welchAnova', groups, 2)
  const r = gs.length
  const ni = gs.map((g) => g.length)
  const mi = gs.map(meanOf)
  const vi = gs.map((g) => varianceOf(g, 1))
  const w = ni.map((n, i) => n / (vi[i] ?? 1))
  const sw = w.reduce((s, v) => s + v, 0)
  let adj = 0
  for (let i = 0; i < r; i++) adj += (w[i] ?? 0) * (mi[i] ?? 0)
  adj /= sw
  let ssAdj = 0
  let lamb = 0
  for (let i = 0; i < r; i++) {
    ssAdj += (w[i] ?? 0) * ((mi[i] ?? 0) - adj) ** 2
    lamb += (1 - (w[i] ?? 0) / sw) ** 2 / ((ni[i] ?? 2) - 1)
  }
  lamb = (3 * lamb) / (r * r - 1)
  const F = ssAdj / (r - 1) / (1 + (2 * lamb * (r - 2)) / 3)
  const df2 = 1 / lamb
  return testResult({
    test: "Welch's ANOVA",
    statistic: F,
    pValue: fSf(r - 1, df2, F),
    df: [r - 1, df2],
    effectSize: etaSquared(gs),
    assumptions: { normal: checkNormality(gs).normal },
  })
}

/** One pairwise comparison of {@link tukeyHsd}. */
export interface TukeyComparison extends TestResult {
  readonly i: number
  readonly j: number
}

/** The result of {@link tukeyHsd}. */
export interface TukeyHsd {
  /** Every pair i < j: the mean difference, its simultaneous interval and adjusted p-value. */
  readonly comparisons: readonly TukeyComparison[]
  /** Mean differences mean_i - mean_j, k x k. */
  readonly statistic: Tensor
  /** Adjusted p-values, k x k (1 on the diagonal). */
  readonly pValue: Tensor
  readonly low: Tensor
  readonly high: Tensor
}

/**
 * Tukey's honestly significant difference test (scipy.stats.tukey_hsd): all pairwise mean differences with p-values
 * and simultaneous intervals from the studentised range distribution (Tukey-Kramer for unequal sizes).
 *
 * @param groups - Two or more samples.
 * @param options - level (default 0.95).
 * @returns The pairwise comparisons and their matrices.
 * @example
 * ```ts
 * import { tukeyHsd } from 'whitebox-engine/stats/tests'
 * tukeyHsd([[1, 2, 3], [4, 5, 6], [1, 2, 4]]).comparisons.length // => 3
 * ```
 */
export function tukeyHsd(groups: readonly TensorLike[], options: { readonly level?: number } = {}): TukeyHsd {
  const level = resolveTestOptions({ level: options.level ?? 0.95 }).level
  const gs = groupsOf('tukeyHsd', groups, 1)
  const k = gs.length
  const N = gs.reduce((s, g) => s + g.length, 0)
  const means = gs.map(meanOf)
  let ssw = 0
  for (let i = 0; i < k; i++) for (const v of gs[i] ?? []) ssw += (v - (means[i] ?? 0)) ** 2
  const df = N - k
  const mse = ssw / df
  const dist = studentisedRange(k, df)
  const crit = item(dist.ppf(level))
  const stat = new Float64Array(k * k)
  const pv = new Float64Array(k * k).fill(1)
  const low = new Float64Array(k * k)
  const high = new Float64Array(k * k)
  const comparisons: TukeyComparison[] = []
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++) {
      const diff = (means[i] ?? 0) - (means[j] ?? 0)
      const se = Math.sqrt((mse / 2) * (1 / (gs[i]?.length ?? 1) + 1 / (gs[j]?.length ?? 1)))
      stat[i * k + j] = diff
      low[i * k + j] = diff - crit * se
      high[i * k + j] = diff + crit * se
      if (i === j) continue
      const q = Math.abs(diff) / se
      const p = item(dist.sf(q))
      pv[i * k + j] = p
      if (i < j)
        comparisons.push({
          ...testResult({
            test: "Tukey's HSD",
            statistic: q,
            pValue: p,
            df: [k, df],
            estimate: intervalEstimate(diff, [diff - crit * se, diff + crit * se], level, {
              standardError: se,
              method: 'simultaneous studentised range interval',
            }),
          }),
          i,
          j,
        })
    }
  return { comparisons, statistic: mat(stat, k, k), pValue: mat(pv, k, k), low: mat(low, k, k), high: mat(high, k, k) }
}

/** Options of {@link rmAnova}. */
export interface RmAnovaOptions {
  /** Effect size: generalised eta squared (`'ng2'`, default, pingouin) or partial eta squared (`'np2'`). */
  readonly effsize?: 'ng2' | 'np2'
}

/**
 * One-way repeated-measures ANOVA (pingouin.rm_anova) on a subjects x conditions matrix: F for conditions against the
 * subject-by-condition residual, Mauchly's test of sphericity and the Greenhouse-Geisser corrected p-value (reported in
 * `assumptions` as `pGreenhouseGeisser`, with epsilon and Mauchly's W and p).
 *
 * @param data - An n x k matrix: one row per subject, one column per condition (k >= 2).
 * @param options - effsize.
 * @returns F, the uncorrected p-value, degrees of freedom and effect size.
 * @example
 * ```ts
 * import { rmAnova } from 'whitebox-engine/stats/tests'
 * rmAnova([[1, 2, 4], [2, 3, 4], [1, 3, 5], [2, 2, 6]]).df // => [2, 6]
 * ```
 */
export function rmAnova(data: TensorLike, options: RmAnovaOptions = {}): TestResult {
  const { rows: n, cols: k, data: x } = rowsOf('rmAnova', data)
  if (n < 2 || k < 2) throw new WhiteboxError('value', 'rmAnova: needs at least 2 subjects and 2 conditions')
  const grand = meanOf(x)
  const condMeans = new Float64Array(k)
  const subjMeans = new Float64Array(n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < k; j++) {
      const v = x[i * k + j] ?? 0
      condMeans[j] = (condMeans[j] ?? 0) + v / n
      subjMeans[i] = (subjMeans[i] ?? 0) + v / k
    }
  let ssWith = 0
  for (const m of condMeans) ssWith += n * (m - grand) ** 2
  let ssResall = 0
  for (let j = 0; j < k; j++) for (let i = 0; i < n; i++) ssResall += ((x[i * k + j] ?? 0) - (condMeans[j] ?? 0)) ** 2
  let ssSubj = 0
  for (const m of subjMeans) ssSubj += k * (m - grand) ** 2
  const ssResWith = ssResall - ssSubj
  const df1 = k - 1
  const df2 = df1 * (n - 1)
  const F = ssWith / df1 / (ssResWith / df2)
  const pUnc = fSf(df1, df2, F)
  const ef = options.effsize === 'np2' ? ssWith / (ssWith + ssResWith) : ssWith / (ssWith + ssResall)
  // Sphericity on the contrast covariance M = C^T S C.
  const S = covarianceOfColumns(x, n, k)
  const C = orthonormalContrasts(k)
  const d = k - 1
  const M = matmul(matmul(transpose(C, k, d), S, d, k, k), C, d, k, d)
  let tr = 0
  for (let i = 0; i < d; i++) tr += M[i * d + i] ?? 0
  let tr2 = 0
  for (const e of eigvalsSym(M, d)) tr2 += e * e
  const eps = d <= 1 ? 1 : Math.min(1, (tr * tr) / (d * tr2))
  let W = Number.NaN
  let pSpher = 1
  if (d > 1) {
    const logdet = logdetSpd(M, d)
    const logW = Number.isNaN(logdet) ? Number.NEGATIVE_INFINITY : logdet - d * Math.log(tr / d)
    W = Math.exp(logW)
    const dfr = n - 1
    const ddof = (d * (d + 1)) / 2 - 1
    const f = 1 - (2 * d * d + d + 2) / (6 * d * dfr)
    const w2 = ((d + 2) * (d - 1) * (d - 2) * (2 * d ** 3 + 6 * d * d + 3 * k + 2)) / (288 * (dfr * d * f) ** 2)
    const chi = -dfr * f * logW
    const p1 = chi2Sf(ddof, chi)
    const p2 = chi2Sf(ddof + 4, chi)
    pSpher = p1 + w2 * (p2 - p1)
  }
  const pGG = fSf(Math.max(df1 * eps, 1), Math.max(df2 * eps, 1), F)
  return testResult({
    test: 'repeated-measures ANOVA',
    statistic: F,
    pValue: pUnc,
    df: [df1, df2],
    effectSize: {
      name: options.effsize === 'np2' ? 'partial eta squared' : 'generalised eta squared',
      value: ef,
    },
    assumptions: {
      sphericity: pSpher > 0.05,
      mauchlyW: W,
      mauchlyP: pSpher,
      epsilon: eps,
      pGreenhouseGeisser: pGG,
    },
  })
}
