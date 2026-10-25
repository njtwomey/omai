/**
 * Tests of whether two (or k) samples come from the same distribution: Kolmogorov-Smirnov, Cramer-von Mises, k-sample
 * Anderson-Darling and Epps-Singleton (scipy.stats), the energy and Wasserstein distances between empirical
 * distributions with permutation tests, and the classifier two-sample test, which asks whether any classifier can tell
 * the samples apart.
 */

import { type Alternative, MdmlError, maxOf, meanOf, minOf, type TestResult } from '../../base/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../../random/index.ts'
import type { TensorLike } from '../../tensor/index.ts'
import { matrixRankSym, ols, pinvSym } from '../_linear.ts'
import { chi2Sf, groupsOf, rowsOf, sample, sortedCopy } from '../_util.ts'
import { iqr } from '../quantiles.ts'
import { ranksOf } from '../ranks.ts'
import { permutationTest, type ResamplingRunOptions } from '../resampling.ts'
import { cvmCdfInf } from './_cvm.ts'
import { kolmogn, ks2Exact } from './_kolmogorov.ts'
import { binomTest } from './proportions.ts'
import { resolveAlternative, type TestOptions, testResult } from './result.ts'

/**
 * The two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp): the largest distance between the two empirical CDFs,
 * with the exact null distribution (counting lattice paths) for samples up to 10000 and otherwise the asymptotic one.
 *
 * @param x - First sample.
 * @param y - Second sample.
 * @param options - alternative (`'greater'` tests whether the CDF of x lies above that of y) and method.
 * @returns D (D+ or D-) and its p-value.
 * @example
 * ```ts
 * import { ks2samp } from 'mdml-engine/stats/tests'
 * ks2samp([1, 2, 3, 4], [5, 6, 7, 8]).pValue // => 0.028571
 * ```
 */
export function ks2samp(
  x: TensorLike,
  y: TensorLike,
  options: TestOptions & { readonly method?: 'auto' | 'exact' | 'asymp' } = {},
): TestResult {
  const alternative = resolveAlternative(options.alternative)
  const a = sortedCopy(sample('ks2samp', x))
  const b = sortedCopy(sample('ks2samp', y))
  const n1 = a.length
  const n2 = b.length
  // The ECDF difference just after each pooled value.
  const all = sortedCopy([...a, ...b])
  let i = 0
  let j = 0
  let maxS = Number.NEGATIVE_INFINITY
  let minS = Number.POSITIVE_INFINITY
  for (const v of all) {
    while (i < n1 && (a[i] ?? 0) <= v) i++
    while (j < n2 && (b[j] ?? 0) <= v) j++
    const d = i / n1 - j / n2
    maxS = Math.max(maxS, d)
    minS = Math.min(minS, d)
  }
  const dMinus = Math.min(1, Math.max(0, -minS))
  const d = alternative === 'less' ? dMinus : alternative === 'two-sided' ? Math.max(maxS, dMinus) : maxS
  let method = options.method ?? 'auto'
  if (method === 'auto') method = Math.max(n1, n2) <= 10000 ? 'exact' : 'asymp'
  let p: number
  let statistic = d
  if (method === 'exact') {
    const r = ks2Exact(n1, n2, d, alternative === 'two-sided')
    p = r.p
    statistic = r.d
  } else {
    const m = Math.max(n1, n2)
    const n = Math.min(n1, n2)
    const en = (m * n) / (m + n)
    if (alternative === 'two-sided') p = kolmogn(Math.round(en), d, true)
    else {
      const z = Math.sqrt(en) * d
      p = Math.exp(-2 * z * z - (2 * z * (m + 2 * n)) / Math.sqrt(m * n * (m + n)) / 3)
    }
  }
  return testResult({
    test: 'two-sample Kolmogorov-Smirnov test',
    statistic,
    pValue: Math.min(1, Math.max(0, p)),
    alternative,
    method,
  })
}

// The exact null distribution of the two-sample Cramer-von Mises U statistic, by dynamic programming over lattice
// paths (scipy's _pval_cvm_2samp_exact): P(U >= u).
function cvmExact(u: number, m: number, n: number): number {
  const g = gcdOf(m, n)
  const lcm = (m / g) * n
  const a = lcm / m
  const b = lcm / n
  const mn = m * n
  const zeta = Math.floor((lcm ** 2 * (m + n) * (6 * u - mn * (4 * mn - 1))) / (6 * mn ** 2) + 1e-9)
  // f[v] maps a running sum of (a v - b w)^2 to its path count, for paths to (w, v).
  let f: Map<number, number>[] = Array.from({ length: m + 1 }, () => new Map<number, number>())
  for (let w = 0; w <= n; w++) {
    const next: Map<number, number>[] = []
    let left = new Map<number, number>()
    for (let v = 0; v <= m; v++) {
      const merged = new Map<number, number>(left)
      const up =
        w === 0 ? (v === 0 ? new Map([[0, 1]]) : new Map<number, number>()) : (f[v] ?? new Map<number, number>())
      for (const [k, c] of up) merged.set(k, (merged.get(k) ?? 0) + c)
      const add = (a * v - b * w) ** 2
      const shifted = new Map<number, number>()
      for (const [k, c] of merged) shifted.set(k + add, c)
      next.push(shifted)
      left = shifted
    }
    f = next
  }
  let total = 0
  let tail = 0
  for (const [k, c] of f[m] ?? new Map<number, number>()) {
    total += c
    if (k >= zeta) tail += c
  }
  return tail / total
}

function gcdOf(a: number, b: number): number {
  let x = a
  let y = b
  while (y) [x, y] = [y, x % y]
  return x
}

/**
 * The two-sample Cramer-von Mises test (scipy.stats.cramervonmises_2samp): T = U / (n m N) - (4 n m - 1) / (6 N) from
 * the ranks, with the exact distribution when both samples have at most 20 observations and otherwise Anderson's
 * asymptotic approximation.
 *
 * @param x - First sample (at least 2).
 * @param y - Second sample (at least 2).
 * @param options - method (`'auto'`, `'exact'` or `'asymptotic'`).
 * @returns T and its p-value.
 * @example
 * ```ts
 * import { cramervonmises2samp } from 'mdml-engine/stats/tests'
 * cramervonmises2samp([1, 2, 3, 4], [5, 6, 7, 8]).pValue // => 0.028571
 * ```
 */
export function cramervonmises2samp(
  x: TensorLike,
  y: TensorLike,
  options: { readonly method?: 'auto' | 'exact' | 'asymptotic' } = {},
): TestResult {
  const a = sortedCopy(sample('cramervonmises2samp', x, 2))
  const b = sortedCopy(sample('cramervonmises2samp', y, 2))
  const nx = a.length
  const ny = b.length
  const r = ranksOf(Float64Array.from([...a, ...b]))
  let u = 0
  for (let i = 0; i < nx; i++) u += nx * ((r[i] ?? 0) - (i + 1)) ** 2
  for (let j = 0; j < ny; j++) u += ny * ((r[nx + j] ?? 0) - (j + 1)) ** 2
  const k = nx * ny
  const N = nx + ny
  const t = u / (k * N) - (4 * k - 1) / (6 * N)
  let method = options.method ?? 'auto'
  if (method === 'auto') method = Math.max(nx, ny) > 20 ? 'asymptotic' : 'exact'
  let p: number
  if (method === 'exact') p = cvmExact(u, nx, ny)
  else {
    const et = (1 + 1 / N) / 6
    const vt = ((N + 1) * (4 * k * N - 3 * (nx * nx + ny * ny) - 2 * k)) / (45 * N * N * 4 * k)
    const tn = 1 / 6 + (t - et) / Math.sqrt(45 * vt)
    p = tn >= 0.003 ? Math.max(0, 1 - cvmCdfInf(tn)) : 1
  }
  return testResult({
    test: 'two-sample Cramer-von Mises test',
    statistic: t,
    pValue: p,
    alternative: 'greater',
    method,
  })
}

/**
 * The k-sample Anderson-Darling test (scipy.stats.anderson_ksamp, midrank variant for ties): the standardised
 * statistic and its p-value interpolated from Scholz and Stephens' critical values (a quadratic in log p), capped to
 * [0.001, 0.25] as scipy caps it.
 *
 * @param samples - Two or more samples.
 * @returns The standardised A^2_akN and its p-value.
 * @example
 * ```ts
 * import { andersonKsamp } from 'mdml-engine/stats/tests'
 * andersonKsamp([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).pValue // => 0.004356
 * ```
 */
export function andersonKsamp(samples: readonly TensorLike[]): TestResult {
  const gs = groupsOf('andersonKsamp', samples, 1)
  const k = gs.length
  const Z = sortedCopy(gs.flatMap((g) => Array.from(g)))
  const N = Z.length
  const Zstar = Array.from(new Set(Z)).sort((p, q) => p - q)
  if (Zstar.length < 2) throw new MdmlError('value', 'andersonKsamp: needs more than one distinct observation')
  const n = gs.map((g) => g.length)
  const left = (s: ArrayLike<number>, v: number): number => {
    let lo = 0
    let hi = s.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if ((s[mid] ?? 0) < v) lo = mid + 1
      else hi = mid
    }
    return lo
  }
  const right = (s: ArrayLike<number>, v: number): number => {
    let lo = 0
    let hi = s.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if ((s[mid] ?? 0) <= v) lo = mid + 1
      else hi = mid
    }
    return lo
  }
  let A2 = 0
  const lj = Zstar.map((v) => right(Z, v) - left(Z, v))
  const Bj = Zstar.map((v, i) => left(Z, v) + (lj[i] ?? 0) / 2)
  gs.forEach((g, i) => {
    const s = sortedCopy(g)
    let inner = 0
    Zstar.forEach((v, j) => {
      const fij = right(s, v) - left(s, v)
      const Mij = right(s, v) - fij / 2
      const B = Bj[j] ?? 0
      const l = lj[j] ?? 0
      inner += ((l / N) * (N * Mij - B * (n[i] ?? 1)) ** 2) / (B * (N - B) - (N * l) / 4)
    })
    A2 += inner / (n[i] ?? 1)
  })
  A2 *= (N - 1) / N
  const H = n.reduce((s, v) => s + 1 / v, 0)
  let hs = 0
  let g = 0
  const cs: number[] = []
  for (let j = N - 1; j > 1; j--) {
    hs += 1 / j
    cs.push(hs)
  }
  const h = (cs.at(-1) ?? 0) + 1
  for (let i = 0; i < cs.length; i++) g += (cs[i] ?? 0) / (i + 2)
  const a = (4 * g - 6) * (k - 1) + (10 - 6 * g) * H
  const b = (2 * g - 4) * k * k + 8 * h * k + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6
  const c = (6 * h + 2 * g - 2) * k * k + (4 * h - 4 * g + 6) * k + (2 * h - 6) * H + 4 * h
  const d = (2 * h + 6) * k * k - 4 * h * k
  const sigmasq = (a * N ** 3 + b * N * N + c * N + d) / ((N - 1) * (N - 2) * (N - 3))
  const m = k - 1
  const stat = (A2 - m) / Math.sqrt(sigmasq)
  const b0 = [0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085]
  const b1 = [-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615]
  const b2 = [-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154]
  const critical = b0.map((v, i) => v + (b1[i] ?? 0) / Math.sqrt(m) + (b2[i] ?? 0) / m)
  const sig = [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
  let p: number
  if (stat < minOf(critical)) p = 0.25
  else if (stat > maxOf(critical)) p = 0.001
  else {
    // numpy.polyfit(critical, log(sig), 2) by least squares, then evaluated at the statistic.
    const X = critical.flatMap((v) => [v * v, v, 1])
    const fit = ols(X, critical.length, 3, sig.map(Math.log))
    p = Math.exp((fit.beta[0] ?? 0) * stat * stat + (fit.beta[1] ?? 0) * stat + (fit.beta[2] ?? 0))
  }
  return testResult({
    test: 'k-sample Anderson-Darling test',
    statistic: stat,
    pValue: p,
    alternative: 'greater',
    method: 'midrank, interpolated p-value',
  })
}

// scipy's _cdf_distance: the L^p distance between two (weighted) empirical CDFs.
function cdfDistance(p: 1 | 2, u: Float64Array, v: Float64Array, uw?: Float64Array, vw?: Float64Array): number {
  const order = (x: Float64Array): number[] =>
    Array.from({ length: x.length }, (_, i) => i).sort((i, j) => (x[i] ?? 0) - (x[j] ?? 0))
  const uo = order(u)
  const vo = order(v)
  const us = Float64Array.from(uo, (i) => u[i] ?? 0)
  const vs = Float64Array.from(vo, (i) => v[i] ?? 0)
  const cum = (o: number[], w?: Float64Array): Float64Array => {
    const c = new Float64Array(o.length + 1)
    for (let i = 0; i < o.length; i++) c[i + 1] = (c[i] ?? 0) + (w === undefined ? 1 : (w[o[i] ?? 0] ?? 0))
    return c
  }
  const uc = cum(uo, uw)
  const vc = cum(vo, vw)
  const all = sortedCopy([...u, ...v])
  const searchRight = (s: Float64Array, x: number): number => {
    let lo = 0
    let hi = s.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if ((s[mid] ?? 0) <= x) lo = mid + 1
      else hi = mid
    }
    return lo
  }
  let total = 0
  for (let i = 0; i + 1 < all.length; i++) {
    const delta = (all[i + 1] ?? 0) - (all[i] ?? 0)
    if (delta === 0) continue
    const fu = (uc[searchRight(us, all[i] ?? 0)] ?? 0) / (uc[us.length] ?? 1)
    const fv = (vc[searchRight(vs, all[i] ?? 0)] ?? 0) / (vc[vs.length] ?? 1)
    total += p === 1 ? Math.abs(fu - fv) * delta : (fu - fv) ** 2 * delta
  }
  return p === 1 ? total : Math.sqrt(total)
}

/** Options of the distribution distances. */
export interface DistanceOptions {
  readonly uWeights?: TensorLike
  readonly vWeights?: TensorLike
}

/**
 * The energy distance between two univariate distributions (scipy.stats.energy_distance): sqrt(2) times the L^2
 * distance between their (weighted) empirical CDFs, sqrt(2 E|X - Y| - E|X - X'| - E|Y - Y'|).
 *
 * @param u - First sample.
 * @param v - Second sample.
 * @param options - Weights.
 * @returns The distance.
 * @example
 * ```ts
 * import { energyDistance } from 'mdml-engine/stats/tests'
 * energyDistance([0], [2]) // => 2
 * ```
 */
export function energyDistance(u: TensorLike, v: TensorLike, options: DistanceOptions = {}): number {
  const uw = options.uWeights === undefined ? undefined : sample('energyDistance', options.uWeights)
  const vw = options.vWeights === undefined ? undefined : sample('energyDistance', options.vWeights)
  return Math.SQRT2 * cdfDistance(2, sample('energyDistance', u), sample('energyDistance', v), uw, vw)
}

/**
 * The first Wasserstein (earth mover's) distance between two univariate distributions (scipy.stats.wasserstein_distance):
 * the L^1 distance between their (weighted) empirical CDFs.
 *
 * @param u - First sample.
 * @param v - Second sample.
 * @param options - Weights.
 * @returns The distance.
 * @example
 * ```ts
 * import { wassersteinDistance } from 'mdml-engine/stats/tests'
 * wassersteinDistance([0, 1, 3], [5, 6, 8]) // => 5
 * ```
 */
export function wassersteinDistance(u: TensorLike, v: TensorLike, options: DistanceOptions = {}): number {
  const uw = options.uWeights === undefined ? undefined : sample('wassersteinDistance', options.uWeights)
  const vw = options.vWeights === undefined ? undefined : sample('wassersteinDistance', options.vWeights)
  return cdfDistance(1, sample('wassersteinDistance', u), sample('wassersteinDistance', v), uw, vw)
}

/** Options of the permutation tests of distribution distances. */
export interface DistanceTestOptions extends ResamplingRunOptions {
  readonly nResamples?: number
}

function distanceTest(
  name: string,
  distance: (a: Float64Array, b: Float64Array) => number,
  x: TensorLike,
  y: TensorLike,
  options: DistanceTestOptions,
): TestResult {
  const { nResamples, ...run } = options
  const r = permutationTest([x, y], distance, { ...run, alternative: 'greater', nResamples: nResamples ?? 9999 })
  return testResult({ test: name, statistic: r.statistic, pValue: r.pValue, alternative: 'greater', method: r.method })
}

/**
 * The permutation test of equal distributions by the energy distance (Szekely and Rizzo's energy test).
 *
 * @param x - First sample.
 * @param y - Second sample.
 * @param options - nResamples (default 9999), key or seed, tracer.
 * @returns The energy distance and its permutation p-value.
 * @example
 * ```ts
 * import { energyTest } from 'mdml-engine/stats/tests'
 * energyTest([1, 2, 3, 4], [5, 6, 7, 8]).pValue // => 0.028571
 * ```
 */
export function energyTest(x: TensorLike, y: TensorLike, options: DistanceTestOptions = {}): TestResult {
  return distanceTest('energy distance test', (a, b) => energyDistance(a, b), x, y, options)
}

/**
 * The permutation test of equal distributions by the first Wasserstein distance.
 *
 * @param x - First sample.
 * @param y - Second sample.
 * @param options - nResamples, key or seed, tracer.
 * @returns The Wasserstein distance and its permutation p-value.
 * @example
 * ```ts
 * import { wassersteinTest } from 'mdml-engine/stats/tests'
 * wassersteinTest([1, 2, 3, 4], [5, 6, 7, 8]).pValue // => 0.028571
 * ```
 */
export function wassersteinTest(x: TensorLike, y: TensorLike, options: DistanceTestOptions = {}): TestResult {
  return distanceTest('Wasserstein distance test', (a, b) => wassersteinDistance(a, b), x, y, options)
}

/**
 * The Epps-Singleton test of equal distributions (scipy.stats.epps_singleton_2samp), comparing empirical characteristic
 * functions at the points t / sigma (sigma the pooled semi-IQR), with the small-sample correction for samples under 25.
 *
 * @param x - First sample (at least 5).
 * @param y - Second sample (at least 5).
 * @param options - t, the evaluation points (default [0.4, 0.8]).
 * @returns W and its chi-square p-value.
 * @example
 * ```ts
 * import { eppsSingleton } from 'mdml-engine/stats/tests'
 * eppsSingleton([1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]).pValue < 0.05 // => true
 * ```
 */
export function eppsSingleton(
  x: TensorLike,
  y: TensorLike,
  options: { readonly t?: readonly number[] } = {},
): TestResult {
  const a = sample('eppsSingleton', x, 5)
  const b = sample('eppsSingleton', y, 5)
  const t = options.t ?? [0.4, 0.8]
  for (const v of t) if (!(v > 0)) throw new MdmlError('value', 'eppsSingleton: t must be positive')
  const nx = a.length
  const ny = b.length
  const n = nx + ny
  const sigma = iqr(Float64Array.from([...a, ...b])) / 2
  const ts = t.map((v) => v / sigma)
  const q = 2 * ts.length
  const features = (s: Float64Array): Float64Array[] =>
    Array.from(s, (v) => Float64Array.from([...ts.map((u) => Math.cos(u * v)), ...ts.map((u) => Math.sin(u * v))]))
  const gx = features(a)
  const gy = features(b)
  const meanOfRows = (rows: Float64Array[]): Float64Array =>
    Float64Array.from({ length: q }, (_, j) => meanOf(rows.map((r) => r[j] ?? 0)))
  const covOf = (rows: Float64Array[]): Float64Array => {
    const m = meanOfRows(rows)
    const c = new Float64Array(q * q)
    for (const r of rows)
      for (let i = 0; i < q; i++)
        for (let j = 0; j < q; j++)
          c[i * q + j] = (c[i * q + j] ?? 0) + ((r[i] ?? 0) - (m[i] ?? 0)) * ((r[j] ?? 0) - (m[j] ?? 0))
    return c.map((v) => v / rows.length)
  }
  const cx = covOf(gx)
  const cy = covOf(gy)
  const est = Float64Array.from(cx, (v, i) => (n / nx) * v + (n / ny) * (cy[i] ?? 0))
  const inv = pinvSym(est, q)
  const r = matrixRankSym(inv, q)
  const mx = meanOfRows(gx)
  const my = meanOfRows(gy)
  const diff = Float64Array.from(mx, (v, i) => v - (my[i] ?? 0))
  let w = 0
  for (let i = 0; i < q; i++) for (let j = 0; j < q; j++) w += (diff[i] ?? 0) * (inv[i * q + j] ?? 0) * (diff[j] ?? 0)
  w *= n
  if (Math.max(nx, ny) < 25) w *= 1 / (1 + n ** -0.45 + 10.1 * (nx ** -1.7 + ny ** -1.7))
  return testResult({ test: 'Epps-Singleton test', statistic: w, pValue: chi2Sf(r, w), df: r, alternative: 'greater' })
}

/**
 * A classifier: fitted on training points and labels, it predicts labels (0 or 1, or probabilities thresholded at 1/2)
 * for test points.
 */
export type Classifier = (
  trainX: readonly (readonly number[])[],
  trainY: readonly number[],
  testX: readonly (readonly number[])[],
) => ArrayLike<number>

/** Options of {@link classifierTwoSampleTest}. */
export interface ClassifierTestOptions {
  /** The fraction of the pooled data held out for testing (default 0.5). */
  readonly testFraction?: number
  readonly key?: Key
  readonly seed?: number
}

/**
 * The classifier two-sample test (Lopez-Paz and Oquab 2017): train any classifier to tell the samples apart on a random
 * half and test its accuracy on the other; under the null of equal distributions the accuracy is Binomial(n_test,
 * 1/2) / n_test, so the p-value is that binomial's upper tail.
 *
 * @param x - First sample `[N]` or `[N, D]`.
 * @param y - Second sample `[M]` or `[M, D]`.
 * @param classifier - The classifier.
 * @param options - testFraction and key or seed.
 * @returns The held-out accuracy and its p-value.
 * @example
 * ```ts
 * import { classifierTwoSampleTest } from 'mdml-engine/stats/tests'
 * const threshold = (_x: readonly (readonly number[])[], _y: readonly number[], t: readonly (readonly number[])[]) =>
 *   t.map((p) => ((p[0] ?? 0) > 5 ? 1 : 0))
 * classifierTwoSampleTest([1, 2, 3, 4, 2, 3], [7, 8, 9, 8, 7, 9], threshold, { seed: 0 }).statistic // => 1
 * ```
 */
export function classifierTwoSampleTest(
  x: TensorLike,
  y: TensorLike,
  classifier: Classifier,
  options: ClassifierTestOptions = {},
): TestResult {
  const a = rowsOf('classifierTwoSampleTest', x)
  const b = rowsOf('classifierTwoSampleTest', y)
  if (a.cols !== b.cols) throw new MdmlError('shape', 'classifierTwoSampleTest: samples differ in dimension')
  const points: number[][] = []
  const labels: number[] = []
  for (let i = 0; i < a.rows; i++) {
    points.push(Array.from(a.data.subarray(i * a.cols, (i + 1) * a.cols)))
    labels.push(0)
  }
  for (let i = 0; i < b.rows; i++) {
    points.push(Array.from(b.data.subarray(i * b.cols, (i + 1) * b.cols)))
    labels.push(1)
  }
  const N = points.length
  const order = permutation(foldIn(options.key ?? makeKey(options.seed ?? 0), 0), N).data
  const nTest = Math.max(1, Math.round(N * (options.testFraction ?? 0.5)))
  const test = Array.from(order.subarray(0, nTest))
  const train = Array.from(order.subarray(nTest))
  const predictions = classifier(
    train.map((i) => points[i] ?? []),
    train.map((i) => labels[i] ?? 0),
    test.map((i) => points[i] ?? []),
  )
  let correct = 0
  test.forEach((i, j) => {
    if (((predictions[j] ?? 0) >= 0.5 ? 1 : 0) === labels[i]) correct++
  })
  const alternative: Alternative = 'greater'
  return testResult({
    test: 'classifier two-sample test',
    statistic: correct / nTest,
    pValue: binomTest(correct, nTest, 0.5, { alternative }).pValue,
    alternative,
    assumptions: { nTest },
  })
}
