/**
 * Tests on multivariate data: Hotelling's T-squared test of a mean vector, one-way MANOVA with Wilks' lambda, Pillai's
 * trace, the Hotelling-Lawley trace and Roy's root, Box's M test of equal covariance matrices, and Mardia's tests of
 * multivariate normality.
 */

import { OmaiError, type TestResult } from '../../base/index.ts'
import { dpotrf, dsyev, dtrsv } from '../../numerics/linalg/index.ts'
import type { TensorLike } from '../../tensor/index.ts'
import { covarianceOfColumns, logdetSpd, pinvSym } from '../_linear.ts'
import { chi2Sf, fSf, normSf, type Rows, rowsOf } from '../_util.ts'
import { testResult } from './result.ts'

function colMeans(m: Rows): Float64Array {
  const out = new Float64Array(m.cols)
  for (let i = 0; i < m.rows; i++)
    for (let j = 0; j < m.cols; j++) out[j] = (out[j] ?? 0) + (m.data[i * m.cols + j] ?? 0) / m.rows
  return out
}

function quadForm(d: ArrayLike<number>, A: ArrayLike<number>, k: number): number {
  let s = 0
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) s += (d[i] ?? 0) * (A[i * k + j] ?? 0) * (d[j] ?? 0)
  return s
}

/** Options of {@link hotellingT2}. */
export interface HotellingOptions {
  /** For one sample, the hypothesised mean vector (default zero). */
  readonly mu?: readonly number[]
  /** Treat x and y as paired rows (the one-sample test of the differences). */
  readonly paired?: boolean
}

/**
 * Hotelling's T-squared test of a mean vector (pingouin.multivariate_ttest): one sample against mu, paired samples, or
 * two independent samples with the pooled covariance; referred to F with (k, n - k) degrees of freedom.
 *
 * @param x - First sample `[N, K]` (N >= 5).
 * @param y - Second sample `[M, K]`, or undefined for one sample.
 * @param options - mu (one sample) and paired.
 * @returns T^2 as the statistic (F in `assumptions.F`) and the p-value.
 * @example
 * ```ts
 * import { hotellingT2 } from 'omai-engine/stats/tests'
 * const x = [[1, 2], [2, 3], [3, 3], [4, 5], [5, 6], [2, 1]]
 * hotellingT2(x).df // => [2, 4]
 * ```
 */
export function hotellingT2(x: TensorLike, y?: TensorLike, options: HotellingOptions = {}): TestResult {
  let a = rowsOf('hotellingT2', x)
  const k = a.cols
  let n: number
  let t2: number
  if (y === undefined || options.paired === true) {
    if (y !== undefined) {
      const b = rowsOf('hotellingT2', y)
      if (b.rows !== a.rows || b.cols !== k) throw new OmaiError('shape', 'hotellingT2: paired samples differ in shape')
      a = { rows: a.rows, cols: k, data: Float64Array.from(a.data, (v, i) => v - (b.data[i] ?? 0)) }
    }
    if (a.rows < 5) throw new OmaiError('value', 'hotellingT2: needs at least five observations')
    n = a.rows
    const mu = options.mu ?? new Array<number>(k).fill(0)
    const m = colMeans(a)
    const diff = Float64Array.from(m, (v, j) => v - (mu[j] ?? 0))
    t2 = n * quadForm(diff, pinvSym(covarianceOfColumns(a.data, a.rows, k), k), k)
  } else {
    const b = rowsOf('hotellingT2', y)
    if (b.cols !== k) throw new OmaiError('shape', 'hotellingT2: samples differ in dimension')
    const nx = a.rows
    const ny = b.rows
    n = nx + ny - 1
    const cx = covarianceOfColumns(a.data, nx, k)
    const cy = covarianceOfColumns(b.data, ny, k)
    const pooled = Float64Array.from(
      cx,
      (v, i) => ((1 / nx + 1 / ny) * ((nx - 1) * v + (ny - 1) * (cy[i] ?? 0))) / (n - 1),
    )
    const mx = colMeans(a)
    const my = colMeans(b)
    t2 = quadForm(
      Float64Array.from(mx, (v, j) => v - (my[j] ?? 0)),
      pinvSym(pooled, k),
      k,
    )
  }
  const F = (t2 * (n - k)) / (k * (n - 1))
  return testResult({
    test: "Hotelling's T-squared test",
    statistic: t2,
    pValue: fSf(k, n - k, F),
    df: [k, n - k],
    alternative: 'greater',
    assumptions: { F },
  })
}

/** One of the four MANOVA statistics. */
export interface ManovaStatistic {
  readonly value: number
  readonly F: number
  readonly df: readonly [number, number]
  readonly pValue: number
}

/** The result of {@link manova}. */
export interface ManovaResult {
  readonly wilks: ManovaStatistic
  readonly pillai: ManovaStatistic
  readonly hotellingLawley: ManovaStatistic
  readonly roy: ManovaStatistic
  /** Wilks' test as the headline TestResult. */
  readonly test: TestResult
}

/**
 * One-way multivariate analysis of variance (statsmodels MANOVA for a single factor): Wilks' lambda, Pillai's trace,
 * the Hotelling-Lawley trace and Roy's greatest root from the eigenvalues of E^-1 H, each with statsmodels' F
 * approximation.
 *
 * @param groups - Two or more samples `[N_i, P]`.
 * @returns The four statistics and Wilks' test.
 * @example
 * ```ts
 * import { manova } from 'omai-engine/stats/tests'
 * const a = [[1, 2], [2, 2], [2, 3], [3, 4]]
 * const b = [[4, 5], [5, 5], [5, 7], [6, 6]]
 * manova([a, b]).wilks.df // => [2, 5]
 * ```
 */
export function manova(groups: readonly TensorLike[]): ManovaResult {
  if (groups.length < 2) throw new OmaiError('value', 'manova: needs at least two groups')
  const gs = groups.map((g) => rowsOf('manova', g))
  const p = gs[0]?.cols ?? 0
  for (const g of gs) if (g.cols !== p) throw new OmaiError('shape', 'manova: groups differ in dimension')
  const N = gs.reduce((s, g) => s + g.rows, 0)
  const grand = new Float64Array(p)
  for (const g of gs)
    for (let i = 0; i < g.rows; i++)
      for (let j = 0; j < p; j++) grand[j] = (grand[j] ?? 0) + (g.data[i * p + j] ?? 0) / N
  const E = new Float64Array(p * p)
  const H = new Float64Array(p * p)
  for (const g of gs) {
    const m = colMeans(g)
    for (let a = 0; a < p; a++)
      for (let b = 0; b < p; b++)
        H[a * p + b] = (H[a * p + b] ?? 0) + g.rows * ((m[a] ?? 0) - (grand[a] ?? 0)) * ((m[b] ?? 0) - (grand[b] ?? 0))
    for (let i = 0; i < g.rows; i++)
      for (let a = 0; a < p; a++)
        for (let b = 0; b < p; b++)
          E[a * p + b] =
            (E[a * p + b] ?? 0) + ((g.data[i * p + a] ?? 0) - (m[a] ?? 0)) * ((g.data[i * p + b] ?? 0) - (m[b] ?? 0))
  }
  // Eigenvalues of E^-1 H through the symmetric L^-1 H L^-T, with E = L L^T.
  const { factor: U, info } = dpotrf(E, p)
  if (info !== 0) throw new OmaiError('singular', 'manova: the within-group SSCP matrix is singular')
  const W = new Float64Array(p * p)
  for (let j = 0; j < p; j++) {
    const col = dtrsv(
      U,
      Float64Array.from({ length: p }, (_, i) => H[i * p + j] ?? 0),
      p,
      { transpose: true },
    )
    for (let i = 0; i < p; i++) W[i * p + j] = col[i] ?? 0
  }
  const M = new Float64Array(p * p)
  for (let i = 0; i < p; i++) {
    const row = dtrsv(
      U,
      Float64Array.from({ length: p }, (_, j) => W[i * p + j] ?? 0),
      p,
      { transpose: true },
    )
    for (let j = 0; j < p; j++) M[i * p + j] = row[j] ?? 0
  }
  const sym = Float64Array.from(M, (v, k) => 0.5 * (v + (M[(k % p) * p + Math.floor(k / p)] ?? 0)))
  const lambdas = Array.from(dsyev(sym, p).w, (v) => Math.max(0, v))
  const thetas = lambdas.map((l) => l / (1 + l))
  const q = gs.length - 1
  const v = N - gs.length
  const m = (Math.abs(p - q) - 1) / 2
  const nn = (v - p - 1) / 2
  const s = Math.min(p, q)
  const stat = (value: number, F: number, d1: number, d2: number): ManovaStatistic => ({
    value,
    F,
    df: [d1, d2],
    pValue: fSf(d1, d2, F),
  })
  const wilksValue = thetas.reduce((acc, t) => acc * (1 - t), 1)
  const r = v - (p - q + 1) / 2
  const u = (p * q - 2) / 4
  const t = p * p + q * q - 5 > 0 ? Math.sqrt((p * p * q * q - 4) / (p * p + q * q - 5)) : 1
  const wd1 = p * q
  const wd2 = r * t - 2 * u
  const lmd = wilksValue ** (1 / t)
  const wilks = stat(wilksValue, (((1 - lmd) / lmd) * wd2) / wd1, wd1, wd2)
  const V = thetas.reduce((acc, x) => acc + x, 0)
  const pd1 = s * (2 * m + s + 1)
  const pd2 = s * (2 * nn + s + 1)
  const pillai = stat(V, ((pd2 / pd1) * V) / (s - V), pd1, pd2)
  const Uv = lambdas.reduce((acc, x) => acc + x, 0)
  let hl: ManovaStatistic
  if (nn > 0) {
    const b = ((p + 2 * nn) * (q + 2 * nn)) / 2 / (2 * nn + 1) / (nn - 1)
    const d1 = p * q
    const d2 = 4 + (p * q + 2) / (b - 1)
    const c = (d2 - 2) / 2 / nn
    hl = stat(Uv, ((d2 / d1) * Uv) / c, d1, d2)
  } else {
    const d1 = s * (2 * m + s + 1)
    const d2 = s * (s * nn + 1)
    hl = stat(Uv, ((d2 / d1 / s) * Uv) / s, d1, d2)
  }
  const sigma = lambdas.reduce((acc, x) => Math.max(acc, x), 0)
  const rr = Math.max(p, q)
  const roy = stat(sigma, ((v - rr + q) / rr) * sigma, rr, v - rr + q)
  return {
    wilks,
    pillai,
    hotellingLawley: hl,
    roy,
    test: testResult({
      test: "MANOVA (Wilks' lambda)",
      statistic: wilks.value,
      pValue: wilks.pValue,
      df: [...wilks.df],
      alternative: 'less',
      assumptions: { F: wilks.F, pillai: pillai.pValue, hotellingLawley: hl.pValue, roy: roy.pValue },
    }),
  }
}

/**
 * Box's M test of equal covariance matrices (pingouin.box_m): -2 (1 - c) log M with Box's chi-square approximation.
 *
 * @param groups - Two or more samples `[N_i, P]`.
 * @returns The chi-square statistic, its p-value and degrees of freedom.
 * @example
 * ```ts
 * import { boxM } from 'omai-engine/stats/tests'
 * const a = [[1, 2], [2, 1], [3, 4], [4, 2], [2, 3]]
 * const b = [[1, 1], [2, 5], [6, 2], [3, 3], [5, 6]]
 * boxM([a, b]).df // => 3
 * ```
 */
export function boxM(groups: readonly TensorLike[]): TestResult {
  if (groups.length < 2) throw new OmaiError('value', 'boxM: needs at least two groups')
  const gs = groups.map((g) => rowsOf('boxM', g))
  const p = gs[0]?.cols ?? 0
  const k = gs.length
  const v = gs.map((g) => g.rows - 1)
  const covs = gs.map((g) => covarianceOfColumns(g.data, g.rows, p))
  const vs = v.reduce((a, b) => a + b, 0)
  const S = new Float64Array(p * p)
  covs.forEach((c, i) => {
    for (let j = 0; j < p * p; j++) S[j] = (S[j] ?? 0) + ((v[i] ?? 0) * (c[j] ?? 0)) / vs
  })
  const logS = logdetSpd(S, p)
  let logM = 0
  covs.forEach((c, i) => {
    logM += ((v[i] ?? 0) / 2) * (logdetSpd(c, p) - logS)
  })
  const c = ((2 * p * p + 3 * p - 1) / (6 * (p + 1) * (k - 1))) * (v.reduce((a, b) => a + 1 / b, 0) - 1 / vs)
  const u = -2 * (1 - c) * logM
  const df = 0.5 * p * (p + 1) * (k - 1)
  return testResult({ test: "Box's M test", statistic: u, pValue: chi2Sf(df, u), df, alternative: 'greater' })
}

/** The result of {@link mardia}. */
export interface MardiaResult {
  /** Multivariate skewness b1,p: n b1 / 6 against chi-square with p (p + 1)(p + 2) / 6 degrees of freedom. */
  readonly skewness: TestResult
  /** Multivariate kurtosis b2,p: standardised against N(p (p + 2), 8 p (p + 2) / n). */
  readonly kurtosis: TestResult
  /** Neither test rejects at 5%. */
  readonly normal: boolean
}

/**
 * Mardia's (1970) tests of multivariate normality from the multivariate skewness and kurtosis of the Mahalanobis
 * products d_ij = (x_i - m)^T S^-1 (x_j - m), with the biased covariance S.
 *
 * @param x - Observations `[N, P]`.
 * @returns The skewness and kurtosis tests.
 * @example
 * ```ts
 * import { mardia } from 'omai-engine/stats/tests'
 * const x = [[1, 2], [2, 1], [3, 4], [4, 2], [2, 3], [5, 5], [3, 1], [1, 4]]
 * mardia(x).skewness.df // => 4
 * ```
 */
export function mardia(x: TensorLike): MardiaResult {
  const m = rowsOf('mardia', x)
  const n = m.rows
  const p = m.cols
  if (n <= p) throw new OmaiError('value', 'mardia: needs more observations than dimensions')
  const mu = colMeans(m)
  const centred = Float64Array.from(m.data, (v, i) => v - (mu[i % p] ?? 0))
  const Sinv = pinvSym(covarianceOfColumns(m.data, n, p, 0), p)
  let b1 = 0
  let b2 = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const di = centred.subarray(i * p, (i + 1) * p)
      const dj = centred.subarray(j * p, (j + 1) * p)
      let s = 0
      for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) s += (di[a] ?? 0) * (Sinv[a * p + b] ?? 0) * (dj[b] ?? 0)
      b1 += s ** 3
      if (i === j) b2 += s * s
    }
  b1 /= n * n
  b2 /= n
  const df = (p * (p + 1) * (p + 2)) / 6
  const skewStat = (n * b1) / 6
  const z = (b2 - p * (p + 2)) / Math.sqrt((8 * p * (p + 2)) / n)
  const skewness = testResult({
    test: "Mardia's skewness test",
    statistic: skewStat,
    pValue: chi2Sf(df, skewStat),
    df,
    alternative: 'greater',
    estimate: { value: b1, method: 'b1,p' },
  })
  const kurtosis = testResult({
    test: "Mardia's kurtosis test",
    statistic: z,
    pValue: 2 * normSf(Math.abs(z)),
    estimate: { value: b2, method: 'b2,p' },
  })
  return { skewness, kurtosis, normal: skewness.pValue >= 0.05 && kurtosis.pValue >= 0.05 }
}
