/**
 * Tests of independence and conditional independence that detect nonlinear dependence: the distance covariance test,
 * Hoeffding's D, mutual information with a permutation test, and the conditional tests used by causal discovery
 * (Fisher's z on partial correlations, the stratified G test for discrete data, and the kernel conditional independence
 * test, as causal-learn).
 */

import { MdmlError, type TestResult } from '../../base/index.ts'
import { mutualInformation as tableMutualInformation } from '../../information/index.ts'
import { dsyev } from '../../numerics/linalg/index.ts'
import { gammainc } from '../../numerics/special/index.ts'
import { fromStorage, item, type TensorLike, toFlat } from '../../tensor/index.ts'
import { matmul, pinvSym } from '../_linear.ts'
import { rankKernel } from '../_ranks.ts'
import { chi2Sf, normSf, paired, type Rows, rowsOf } from '../_util.ts'
import { centredDistances, corrcoef, distanceCorrelationOf } from '../correlation.ts'
import { mean, std } from '../descriptive.ts'
import { histogram2d } from '../histograms.ts'
import { permutationTest, type ResamplingRunOptions } from '../resampling.ts'
import { centreGram, gaussianKernel, type Kernel } from './kernel.ts'
import { testResult } from './result.ts'

/** Options of the permutation-based independence tests. */
export interface IndependenceOptions extends ResamplingRunOptions {
  /** Permutations (default 999). */
  readonly nResamples?: number
}

/**
 * The distance covariance test of independence (Szekely, Rizzo and Bakirov 2007; pingouin.distance_corr): the
 * distance correlation with a permutation p-value; detects any dependence, linear or not, between vectors of any
 * dimensions.
 *
 * @param x - First variable(s) `[N]` or `[N, P]`.
 * @param y - Second variable(s) `[N]` or `[N, Q]`.
 * @param options - nResamples (default 999), key or seed, tracer.
 * @returns The distance correlation and its p-value.
 * @example
 * ```ts
 * import { distanceCovarianceTest } from 'mdml-engine/stats/tests'
 * const x = [-3, -2, -1, 0, 1, 2, 3]
 * distanceCovarianceTest(x, x.map((v) => v * v), { seed: 1 }).statistic // => 0.50532
 * ```
 */
export function distanceCovarianceTest(x: TensorLike, y: TensorLike, options: IndependenceOptions = {}): TestResult {
  const a = rowsOf('distanceCovarianceTest', x)
  const b = rowsOf('distanceCovarianceTest', y)
  if (a.rows !== b.rows) throw new MdmlError('shape', 'distanceCovarianceTest: x and y differ in length')
  const n = a.rows
  const A = centredDistances(a)
  const B = centredDistances(b)
  const index = Array.from({ length: n }, (_, i) => i)
  const { nResamples, ...run } = options
  const r = permutationTest([index], (perm) => distanceCorrelationOf(A, B, n, perm), {
    ...run,
    permutationType: 'pairings',
    alternative: 'greater',
    nResamples: nResamples ?? 999,
  })
  return testResult({
    test: 'distance covariance test',
    statistic: r.statistic,
    pValue: r.pValue,
    alternative: 'greater',
    method: r.method,
    effectSize: { name: 'distance correlation', value: r.statistic },
  })
}

/** Hoeffding's D from midranks and bivariate ranks (Hollander and Wolfe's formula with Hmisc's tie handling). */
function hoeffdingStatistic(x: Float64Array, y: Float64Array): number {
  const n = x.length
  const R = rankKernel(x)
  const S = rankKernel(y)
  let D1 = 0
  let D2 = 0
  let D3 = 0
  for (let i = 0; i < n; i++) {
    let q = 1
    for (let j = 0; j < n; j++) {
      if (i === j) continue
      const lx = (x[j] ?? 0) < (x[i] ?? 0)
      const ex = (x[j] ?? 0) === (x[i] ?? 0)
      const ly = (y[j] ?? 0) < (y[i] ?? 0)
      const ey = (y[j] ?? 0) === (y[i] ?? 0)
      if (lx && ly) q += 1
      else if (ex && ly) q += 0.5
      else if (lx && ey) q += 0.5
      else if (ex && ey) q += 0.25
    }
    const r = R[i] ?? 0
    const s = S[i] ?? 0
    D1 += (q - 1) * (q - 2)
    D2 += (r - 1) * (r - 2) * (s - 1) * (s - 2)
    D3 += (r - 2) * (s - 2) * (q - 1)
  }
  return (30 * ((n - 2) * (n - 3) * D1 + D2 - 2 * (n - 2) * D3)) / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))
}

/**
 * Hoeffding's D test of independence (Hoeffding 1948; the statistic of R's Hmisc::hoeffd): a rank statistic that is
 * zero in the population exactly under independence, here with a permutation p-value (one-sided, large D).
 *
 * @param x - First variable `[N]` (N >= 5).
 * @param y - Second variable `[N]`.
 * @param options - nResamples, key or seed.
 * @returns D (on Hmisc's scale, 30 times Hoeffding's) and its p-value.
 * @example
 * ```ts
 * import { hoeffdingD } from 'mdml-engine/stats/tests'
 * hoeffdingD([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]).statistic // => 1
 * ```
 */
export function hoeffdingD(x: TensorLike, y: TensorLike, options: IndependenceOptions = {}): TestResult {
  const [a, b] = paired('hoeffdingD', x, y, 5)
  const { nResamples, ...run } = options
  const r = permutationTest([b], (yy) => hoeffdingStatistic(a, yy), {
    ...run,
    permutationType: 'pairings',
    alternative: 'greater',
    nResamples: nResamples ?? 999,
  })
  return testResult({
    test: "Hoeffding's D test",
    statistic: r.statistic,
    pValue: r.pValue,
    alternative: 'greater',
    method: r.method,
  })
}

/**
 * The plug-in mutual information of two variables in nats: from their joint frequencies when discrete, otherwise from
 * an equal-width two-dimensional histogram, computed by `information`'s `mutualInformation` of that table (its
 * sample estimators, such as KSG, live there too).
 *
 * @param x - First variable `[N]`.
 * @param y - Second variable `[N]`.
 * @param options - discrete (treat values as categories) or bins (default Sturges' ceil(log2 N + 1)).
 * @returns The mutual information, at least 0.
 * @throws MdmlError with code `value` when the data contains NaN: it is not a category in `discrete` mode, and the histogram rejects it otherwise.
 * @example
 * ```ts
 * import { mutualInformation } from 'mdml-engine/stats/tests'
 * mutualInformation([0, 0, 1, 1], [0, 0, 1, 1], { discrete: true }) // => 0.693147
 * ```
 */
export function mutualInformation(
  x: TensorLike,
  y: TensorLike,
  options: { readonly discrete?: boolean; readonly bins?: number } = {},
): number {
  const [a, b] = paired('mutualInformation', x, y, 2, true)
  return plugInMI(a, b, options)
}

function plugInMI(
  a: Float64Array,
  b: Float64Array,
  options: { readonly discrete?: boolean; readonly bins?: number },
): number {
  const n = a.length
  let counts: Float64Array
  let r: number
  let c: number
  if (options.discrete === true) {
    const ux = Array.from(new Set(a)).sort((p, q) => p - q)
    const uy = Array.from(new Set(b)).sort((p, q) => p - q)
    const ix = new Map(ux.map((v, i) => [v, i]))
    const iy = new Map(uy.map((v, i) => [v, i]))
    r = ux.length
    c = uy.length
    counts = new Float64Array(r * c)
    for (let i = 0; i < n; i++) {
      const at = (ix.get(a[i] ?? 0) ?? 0) * c + (iy.get(b[i] ?? 0) ?? 0)
      counts[at] = (counts[at] ?? 0) + 1
    }
  } else {
    const bins = options.bins ?? Math.ceil(Math.log2(n) + 1)
    counts = Float64Array.from(toFlat(histogram2d(a, b, { bins }).counts))
    r = bins
    c = bins
  }
  // The table goes to the information domain, which owns mutual information.
  return Math.max(0, item(tableMutualInformation(fromStorage(counts, [r, c], 'float64'))))
}

/**
 * The permutation test of independence by the plug-in mutual information.
 *
 * @param x - First variable `[N]`.
 * @param y - Second variable `[N]`.
 * @param options - discrete, bins, nResamples, key or seed.
 * @returns The mutual information and its p-value.
 * @throws MdmlError with code `value` when the data contains NaN: it is not a category in `discrete` mode, and the histogram rejects it otherwise.
 * @example
 * ```ts
 * import { mutualInformationTest } from 'mdml-engine/stats/tests'
 * const x = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
 * mutualInformationTest(x, x, { discrete: true, seed: 0 }).pValue < 0.01 // => true
 * ```
 */
export function mutualInformationTest(
  x: TensorLike,
  y: TensorLike,
  options: IndependenceOptions & { readonly discrete?: boolean; readonly bins?: number } = {},
): TestResult {
  const [a, b] = paired('mutualInformationTest', x, y, 2, true)
  const { nResamples, discrete, bins, ...run } = options
  const settings = { ...(discrete === undefined ? {} : { discrete }), ...(bins === undefined ? {} : { bins }) }
  const r = permutationTest([b], (yy) => plugInMI(a, yy, settings), {
    ...run,
    permutationType: 'pairings',
    alternative: 'greater',
    nResamples: nResamples ?? 999,
  })
  return testResult({
    test: 'mutual information test',
    statistic: r.statistic,
    pValue: r.pValue,
    alternative: 'greater',
    method: r.method,
  })
}

function columnsOf(name: string, data: TensorLike, indices: readonly number[]): Float64Array[] {
  const m = rowsOf(name, data)
  return indices.map((j) => {
    if (!(Number.isInteger(j) && j >= 0 && j < m.cols))
      throw new MdmlError('value', `${name}: column ${j} is out of range`)
    return Float64Array.from({ length: m.rows }, (_, i) => m.data[i * m.cols + j] ?? 0)
  })
}

/**
 * Fisher's z test of conditional independence by partial correlation (causal-learn's fisherz, the test of the PC
 * algorithm for Gaussian data): r from the inverse of the correlation submatrix, z = atanh(r) sqrt(n - |Z| - 3).
 *
 * @param data - Observations `[N, P]`.
 * @param x - Column of the first variable.
 * @param y - Column of the second variable.
 * @param z - Columns of the conditioning set.
 * @returns The partial correlation as the statistic and the two-sided p-value.
 * @example
 * ```ts
 * import { partialCorrelationTest } from 'mdml-engine/stats/tests'
 * const data = [[1, 2, 1], [2, 1, 1], [3, 4, 2], [4, 3, 2], [5, 6, 3], [6, 5, 3.5]]
 * partialCorrelationTest(data, 0, 1, []).statistic // => 0.828571
 * ```
 */
export function partialCorrelationTest(data: TensorLike, x: number, y: number, z: readonly number[] = []): TestResult {
  const cols = columnsOf('partialCorrelationTest', data, [x, y, ...z])
  const n = cols[0]?.length ?? 0
  const k = cols.length
  const R = toFlat(corrcoef(cols.map((c) => Array.from(c))))
  const inv = pinvSym(R, k)
  let r = -(inv[1] ?? 0) / Math.sqrt(Math.abs((inv[0] ?? 0) * (inv[k + 1] ?? 0)))
  if (Math.abs(r) >= 1) r = (1 - Number.EPSILON) * Math.sign(r)
  const Z = 0.5 * Math.log((1 + r) / (1 - r))
  const stat = Math.sqrt(n - z.length - 3) * Math.abs(Z)
  return testResult({
    test: "Fisher's z conditional independence test",
    statistic: r,
    pValue: 2 * normSf(stat),
    method: `conditioning on ${z.length} variables`,
  })
}

/**
 * The G (or chi-square) test of conditional independence for discrete data (causal-learn's gsq and chisq): the
 * statistic summed over the strata of the conditioning set, with degrees of freedom reduced for empty rows and columns
 * of each stratum.
 *
 * @param data - Discrete observations `[N, P]` (any values; each column's distinct values are its categories).
 * @param x - Column of the first variable.
 * @param y - Column of the second variable.
 * @param z - Columns of the conditioning set.
 * @param options - statistic (`'g'`, default, or `'chi2'`).
 * @returns The statistic, its p-value and degrees of freedom.
 * @throws MdmlError with code `value` when a column contains NaN: a NaN is not a category.
 * @example
 * ```ts
 * import { gTestConditional } from 'mdml-engine/stats/tests'
 * gTestConditional([[0, 0], [0, 0], [1, 1], [1, 1], [0, 1], [1, 0]], 0, 1).df // => 1
 * ```
 */
export function gTestConditional(
  data: TensorLike,
  x: number,
  y: number,
  z: readonly number[] = [],
  options: { readonly statistic?: 'g' | 'chi2' } = {},
): TestResult {
  const m = rowsOf('gTestConditional', data)
  // Each column's distinct values (over all the data) are its categories, as causal-learn encodes them.
  const code = (j: number): { codes: Int32Array; card: number } => {
    const col = Float64Array.from({ length: m.rows }, (_, i) => m.data[i * m.cols + j] ?? 0)
    if (col.some(Number.isNaN))
      throw new MdmlError('value', `gTestConditional: column ${j} contains NaN, which is not a category`)
    const u = Array.from(new Set(col)).sort((p, q) => p - q)
    const at = new Map(u.map((v, i) => [v, i]))
    return { codes: Int32Array.from(col, (v) => at.get(v) ?? 0), card: u.length }
  }
  const cx = code(x)
  const cy = code(y)
  const cz = z.map(code)
  const strata = new Map<string, number[]>()
  for (let i = 0; i < m.rows; i++) {
    const keyOf = cz.map((c) => c.codes[i]).join(',')
    const list = strata.get(keyOf) ?? []
    list.push(i)
    strata.set(keyOf, list)
  }
  let stat = 0
  let df = 0
  const g = options.statistic !== 'chi2'
  for (const rows of strata.values()) {
    const t = new Float64Array(cx.card * cy.card)
    for (const i of rows) {
      const at = (cx.codes[i] ?? 0) * cy.card + (cy.codes[i] ?? 0)
      t[at] = (t[at] ?? 0) + 1
    }
    const rs = new Float64Array(cx.card)
    const cs = new Float64Array(cy.card)
    for (let a = 0; a < cx.card; a++)
      for (let b = 0; b < cy.card; b++) {
        rs[a] = (rs[a] ?? 0) + (t[a * cy.card + b] ?? 0)
        cs[b] = (cs[b] ?? 0) + (t[a * cy.card + b] ?? 0)
      }
    const total = rows.length
    for (let a = 0; a < cx.card; a++)
      for (let b = 0; b < cy.card; b++) {
        const e = ((rs[a] ?? 0) * (cs[b] ?? 0)) / total
        const o = t[a * cy.card + b] ?? 0
        if (g) stat += e === 0 || o === 0 ? 0 : 2 * o * Math.log(o / e)
        else stat += e === 0 ? 0 : (o - e) ** 2 / e
      }
    const zeroRows = Array.from(rs).filter((v) => v === 0).length
    const zeroCols = Array.from(cs).filter((v) => v === 0).length
    df += (cx.card - 1 - zeroRows) * (cy.card - 1 - zeroCols)
  }
  return testResult({
    test: g ? 'G test of conditional independence' : 'chi-square test of conditional independence',
    statistic: stat,
    pValue: df === 0 ? 1 : chi2Sf(df, stat),
    df,
    alternative: 'greater',
  })
}

/** Options of {@link kernelCITest}. */
export interface KernelCIOptions {
  /** The regularisation of the kernel ridge regression on z (default 1e-3, as the original implementation). */
  readonly epsilon?: number
  /** Gaussian kernel widths for (x, z), y and z on standardised data; default the empirical rule of KCI. */
  readonly widths?: { readonly x?: number; readonly y?: number; readonly z?: number }
}

/**
 * The kernel conditional independence test (KCI; Zhang, Peters, Janzing and Scholkopf 2011, as causal-learn's
 * KCI_CInd): Gaussian kernels on standardised (x, z / 2), y and z, the kernel matrices of x and y residualised on z by
 * kernel ridge regression, the V-statistic tr(K_x|z K_y|z), and the gamma approximation of its null distribution from
 * the eigen-decomposition of the residual kernels.
 *
 * @param data - Observations `[N, P]`.
 * @param x - Column of the first variable.
 * @param y - Column of the second variable.
 * @param z - Columns of the conditioning set (at least one).
 * @param options - epsilon and kernel widths.
 * @returns The statistic and its p-value.
 * @reference K. Zhang, J. Peters, D. Janzing and B. Scholkopf, Kernel-based conditional independence test and
 * application in causal discovery, UAI (2011).
 * @example
 * ```ts
 * import { kernelCITest } from 'mdml-engine/stats/tests'
 * const data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((z) => [z + Math.sin(z), z * 0.5 + Math.cos(3 * z), z])
 * kernelCITest(data, 0, 1, [2]).pValue > 0 // => true
 * ```
 */
export function kernelCITest(
  data: TensorLike,
  x: number,
  y: number,
  z: readonly number[],
  options: KernelCIOptions = {},
): TestResult {
  if (z.length === 0) throw new MdmlError('value', 'kernelCITest: needs a conditioning set (use hsicTest without one)')
  const [cx = new Float64Array(), cy = new Float64Array(), ...czs] = columnsOf('kernelCITest', data, [x, y, ...z])
  const n = cx.length
  const zscore = (c: Float64Array): Float64Array => {
    const m = mean(c)
    const s = std(c, { ddof: 1 })
    return Float64Array.from(c, (v) => (s > 0 ? (v - m) / s : 0))
  }
  const zx = zscore(cx)
  const zy = zscore(cy)
  const zz = czs.map(zscore)
  const dz = zz.length
  const xRows: Rows = {
    rows: n,
    cols: 1 + dz,
    data: Float64Array.from({ length: n * (1 + dz) }, (_, k) => {
      const i = Math.floor(k / (1 + dz))
      const j = k % (1 + dz)
      return j === 0 ? (zx[i] ?? 0) : 0.5 * (zz[j - 1]?.[i] ?? 0)
    }),
  }
  const yRows: Rows = { rows: n, cols: 1, data: zy }
  const zRows: Rows = {
    rows: n,
    cols: dz,
    data: Float64Array.from({ length: n * dz }, (_, k) => zz[k % dz]?.[Math.floor(k / dz)] ?? 0),
  }
  // causal-learn's empirical widths: 1.2, 0.7 or 0.4 by sample size, with theta = 1 / (width^2 dim_z).
  const empirical = n < 200 ? 1.2 : n < 1200 ? 0.7 : 0.4
  const thetaOf = (w: number | undefined): number => (w === undefined ? 1 / (empirical * empirical) / dz : 1 / (w * w))
  const kernelOf = (theta: number): Kernel => gaussianKernel(1 / Math.sqrt(theta))
  const gramOf = (m: Rows, k: Kernel): Float64Array => {
    const g = new Float64Array(n * n)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        g[i * n + j] = k(m.data.subarray(i * m.cols, (i + 1) * m.cols), m.data.subarray(j * m.cols, (j + 1) * m.cols))
    return g
  }
  const Kx = centreGram(gramOf(xRows, kernelOf(thetaOf(options.widths?.x))), n)
  const Ky = centreGram(gramOf(yRows, kernelOf(thetaOf(options.widths?.y))), n)
  const Kz = centreGram(gramOf(zRows, kernelOf(thetaOf(options.widths?.z))), n)
  const eps = options.epsilon ?? 1e-3
  const shifted = Float64Array.from(Kz, (v, k) => v + (Math.floor(k / n) === k % n ? eps : 0))
  const Rz = Float64Array.from(pinvSym(shifted, n), (v) => eps * v)
  const KxR = matmul(matmul(Rz, Kx, n, n, n), Rz, n, n, n)
  const KyR = matmul(matmul(Rz, Ky, n, n, n), Rz, n, n, n)
  let stat = 0
  for (let k = 0; k < n * n; k++) stat += (KxR[k] ?? 0) * (KyR[k] ?? 0)
  // uu^T from the leading eigenvectors of both residual kernels, scaled by the square roots of their eigenvalues.
  const factors = (K: Float64Array): Float64Array[] => {
    const sym = Float64Array.from(K, (v, k) => 0.5 * (v + (K[(k % n) * n + Math.floor(k / n)] ?? 0)))
    const { w, v } = dsyev(sym, n)
    let top = Number.NEGATIVE_INFINITY
    for (const e of w) top = Math.max(top, e)
    const out: Float64Array[] = []
    for (let c = n - 1; c >= 0; c--) {
      const e = w[c] ?? 0
      if (!(e > top * 1e-5)) continue
      const s = Math.sqrt(e)
      out.push(Float64Array.from({ length: n }, (_, i) => (v[i * n + c] ?? 0) * s))
    }
    return out
  }
  const vx = factors(KxR)
  const vy = factors(KyR)
  // trace(UU) and trace(UU UU) through the T x T matrix uu uu^T (equal traces either way).
  const UU = new Float64Array(n * n)
  for (const a of vx)
    for (const b of vy)
      for (let i = 0; i < n; i++) {
        const ui = (a[i] ?? 0) * (b[i] ?? 0)
        if (ui === 0) continue
        for (let j = 0; j < n; j++) UU[i * n + j] = (UU[i * n + j] ?? 0) + ui * (a[j] ?? 0) * (b[j] ?? 0)
      }
  let tr = 0
  for (let i = 0; i < n; i++) tr += UU[i * n + i] ?? 0
  let tr2 = 0
  for (let k = 0; k < n * n; k++) tr2 += (UU[k] ?? 0) * (UU[(k % n) * n + Math.floor(k / n)] ?? 0)
  const meanAppr = tr
  const varAppr = 2 * tr2
  const kAppr = (meanAppr * meanAppr) / varAppr
  const theta = varAppr / meanAppr
  const pValue = 1 - item(gammainc(kAppr, stat / theta))
  return testResult({
    test: 'kernel conditional independence test (KCI)',
    statistic: stat,
    pValue,
    alternative: 'greater',
    method: 'gamma approximation',
  })
}
