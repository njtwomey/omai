/**
 * Classical regression diagnostics (statsmodels' `OLSInfluence`, `variance_inflation_factor`, `het_breuschpagan` and
 * `het_white`): the hat matrix and leverage, internally and externally studentised residuals, Cook's distance, DFFITS,
 * DFBETAS and COVRATIO of a (weighted) least-squares fit, variance inflation factors and Belsley-Kuh-Welsch condition
 * indices for collinearity, and the Breusch-Pagan and White tests for heteroscedasticity. Everything is computed from
 * one SVD of the weighted design (intercept column last), so rank-deficient designs are handled through the
 * pseudo-inverse. With sample weights the residuals and leverages are those of the whitened problem sqrt(w) y on
 * sqrt(w) X, as statsmodels reports for weighted fits.
 *
 * @index function Leverage, influence, VIF and heteroscedasticity diagnostics.
 */

import { OmaiError, type TestResult } from '../base/index.ts'
import { type LearnData, matrixView, sampleWeightsOf } from '../learn/index.ts'
import { chdtrc, stdtr } from '../numerics/special/index.ts'
import { durbinWatson, testResult } from '../stats/tests/index.ts'
import { fromStorage, item, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Design, designOf, withIntercept } from './design.ts'
import { svdOf } from './solvers.ts'

/** Options shared by the diagnostics. */
export interface DiagnosticsOptions {
  /** Whether the model has an intercept (a column of ones appended last; default true). */
  readonly fitIntercept?: boolean
}

/** The influence measures of a least-squares fit, one entry per observation unless noted. */
export interface Influence {
  readonly n: number
  /** The rank p of the (augmented) design. */
  readonly rank: number
  /** Residual degrees of freedom n - p. */
  readonly dfResid: number
  /** The residual standard error sqrt(RSS / (n - p)). */
  readonly sigma: number
  /** Residuals of the whitened fit (the ordinary residuals when the weights are equal to one). */
  readonly residuals: number[]
  /** The diagonal of the hat matrix, in [0, 1], summing to the rank. */
  readonly leverage: number[]
  /** Residuals divided by sigma sqrt(1 - h). */
  readonly internal: number[]
  /** Leave-one-out standard error of the regression without observation i. */
  readonly sigmaLoo: number[]
  /** Externally studentised (jackknife) residuals e_i / (sigma_(i) sqrt(1 - h_i)), t-distributed with n - p - 1 df. */
  readonly external: number[]
  /** Cook's distance r_i^2 h_i / (p (1 - h_i)) with r the internal residuals. */
  readonly cooksDistance: number[]
  /** DFFITS: external residual times sqrt(h / (1 - h)). */
  readonly dffits: number[]
  /** The internal version of DFFITS (statsmodels' `dffits_internal`). */
  readonly dffitsInternal: number[]
  /** The change of the coefficients when observation i is deleted, `[n, p]` (intercept last). */
  readonly dfbeta: Tensor
  /** DFBETA scaled by sigma_(i) sqrt((X'X)^-1_jj), `[n, p]`. */
  readonly dfbetas: Tensor
  /** COVRATIO: the ratio of the coefficient covariance determinants without and with observation i. */
  readonly covRatio: number[]
}

interface Core {
  readonly n: number
  readonly p: number
  readonly rank: number
  /** The whitened augmented design `[n, p]`, row-major. */
  readonly a: Float64Array
  readonly yw: Float64Array
  readonly u: ArrayLike<number>
  readonly s: ArrayLike<number>
  readonly vh: ArrayLike<number>
  readonly k: number
  readonly keep: readonly boolean[]
}

function core(design: Design, fitIntercept: boolean): Core {
  const { n } = design.x
  const { a: raw, p } = withIntercept(design, fitIntercept)
  const a = new Float64Array(n * p)
  const yw = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const r = Math.sqrt(design.w[i] ?? 0)
    yw[i] = r * (design.y[i] ?? 0)
    for (let j = 0; j < p; j++) a[i * p + j] = r * (raw[i * p + j] ?? 0)
  }
  const { u, s, vh } = svdOf(a, n, p)
  const k = s.length
  const tol = Math.max(n, p) * Number.EPSILON * (s[0] ?? 0)
  const keep = Array.from(s, (v) => v > tol)
  const rank = keep.filter(Boolean).length
  return { n, p, rank, a, yw, u, s, vh, k, keep }
}

/** The leverages (squared row norms of the active left singular vectors). */
function hatDiagonal(c: Core): number[] {
  const h = new Array<number>(c.n).fill(0)
  for (let i = 0; i < c.n; i++) {
    let t = 0
    for (let q = 0; q < c.k; q++) if (c.keep[q] === true) t += (c.u[i * c.k + q] ?? 0) ** 2
    h[i] = Math.min(1, t)
  }
  return h
}

function influenceOf(data: LearnData, options: DiagnosticsOptions, who: string): Influence {
  const design = designOf(data, who)
  const c = core(design, options.fitIntercept ?? true)
  const { n, p, rank, k } = c
  const dfResid = n - rank
  if (dfResid < 1) throw new OmaiError('value', `${who}: needs more observations than the design's rank`)
  // The projection of y on the column space, and the residuals.
  const uty = new Float64Array(k)
  for (let q = 0; q < k; q++) {
    if (c.keep[q] !== true) continue
    let t = 0
    for (let i = 0; i < n; i++) t += (c.u[i * k + q] ?? 0) * (c.yw[i] ?? 0)
    uty[q] = t
  }
  const residuals = new Array<number>(n).fill(0)
  let rss = 0
  for (let i = 0; i < n; i++) {
    let fit = 0
    for (let q = 0; q < k; q++) fit += (c.u[i * k + q] ?? 0) * (uty[q] ?? 0)
    residuals[i] = (c.yw[i] ?? 0) - fit
    rss += (residuals[i] ?? 0) ** 2
  }
  const h = hatDiagonal(c)
  const sigma2 = rss / dfResid
  const sigma = Math.sqrt(sigma2)
  const internal = residuals.map((e, i) => e / Math.sqrt(sigma2 * (1 - (h[i] ?? 0))))
  const sigmaLoo = residuals.map((e, i) =>
    dfResid > 1 ? Math.sqrt(Math.max(0, (rss - (e * e) / (1 - (h[i] ?? 0))) / (dfResid - 1))) : Number.NaN,
  )
  const external = residuals.map((e, i) => e / ((sigmaLoo[i] ?? 0) * Math.sqrt(1 - (h[i] ?? 0))))
  const cooks = internal.map((r, i) => (r * r * (h[i] ?? 0)) / (rank * (1 - (h[i] ?? 0))))
  const dffits = external.map((t, i) => t * Math.sqrt((h[i] ?? 0) / (1 - (h[i] ?? 0))))
  const dffitsInternal = internal.map((r, i) => r * Math.sqrt((h[i] ?? 0) / (1 - (h[i] ?? 0))))
  // (X'X)^+ = V S^-2 V'; dfbeta_i = (X'X)^+ x_i e_i / (1 - h_i) with (X'X)^+ x_i = V S^-1 u_i.
  const diag = new Float64Array(p)
  for (let j = 0; j < p; j++)
    for (let q = 0; q < k; q++)
      if (c.keep[q] === true) diag[j] = (diag[j] ?? 0) + ((c.vh[q * p + j] ?? 0) / (c.s[q] ?? 1)) ** 2
  const dfbeta = new Float64Array(n * p)
  const dfbetas = new Float64Array(n * p)
  for (let i = 0; i < n; i++) {
    const f = (residuals[i] ?? 0) / (1 - (h[i] ?? 0))
    for (let j = 0; j < p; j++) {
      let t = 0
      for (let q = 0; q < k; q++)
        if (c.keep[q] === true) t += ((c.vh[q * p + j] ?? 0) * (c.u[i * k + q] ?? 0)) / (c.s[q] ?? 1)
      dfbeta[i * p + j] = t * f
      dfbetas[i * p + j] = (t * f) / ((sigmaLoo[i] ?? 0) * Math.sqrt(diag[j] ?? 1))
    }
  }
  const covRatio = sigmaLoo.map((sl, i) => (sl / sigma) ** (2 * rank) / (1 - (h[i] ?? 0)))
  return {
    n,
    rank,
    dfResid,
    sigma,
    residuals,
    leverage: h,
    internal,
    sigmaLoo,
    external,
    cooksDistance: cooks,
    dffits,
    dffitsInternal,
    dfbeta: fromStorage(dfbeta, [n, p], 'float64'),
    dfbetas: fromStorage(dfbetas, [n, p], 'float64'),
    covRatio,
  }
}

/**
 * All the influence measures of a least-squares fit at once (statsmodels' `OLSInfluence`): leverage, internal and
 * external studentised residuals, Cook's distance, DFFITS, DFBETA(S) and COVRATIO.
 *
 * @param data - Features, targets and optional sample weights.
 * @param options - Whether the model has an intercept (default true).
 * @returns The measures, each over the observations.
 * @throws OmaiError `value` when there are no more observations than the design's rank.
 * @example
 * ```ts
 * import { olsInfluence } from 'omai-engine/linear'
 * const infl = olsInfluence({ x: [[1], [2], [3], [4], [5]], y: [1, 2, 3, 4, 10] })
 * infl.leverage.map((h) => Math.round(h * 1000) / 1000) // => [0.6, 0.3, 0.2, 0.3, 0.6]
 * infl.cooksDistance.indexOf(Math.max(...infl.cooksDistance)) // => 4
 * ```
 */
export function olsInfluence(data: LearnData, options: DiagnosticsOptions = {}): Influence {
  return influenceOf(data, options, 'olsInfluence')
}

/**
 * The hat matrix H = X (X'X)^+ X' of the (weighted) design, which maps responses to fitted values.
 *
 * @param x - Features `[n, d]`.
 * @param options - Whether the model has an intercept (default true), and sample weights.
 * @returns H `[n, n]`, symmetric and idempotent with trace equal to the rank.
 * @example
 * ```ts
 * import { hatMatrix } from 'omai-engine/linear'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(hatMatrix([[1], [2], [3]]))[0] // => [0.8333333333333334, 0.3333333333333333, -0.16666666666666666]
 * ```
 */
export function hatMatrix(x: TensorLike, options: DiagnosticsOptions & { readonly weights?: TensorLike } = {}): Tensor {
  const design = featuresOnly(x, options.weights, 'hatMatrix')
  const c = core(design, options.fitIntercept ?? true)
  const { n, k } = c
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let t = 0
      for (let q = 0; q < k; q++) if (c.keep[q] === true) t += (c.u[i * k + q] ?? 0) * (c.u[j * k + q] ?? 0)
      out[i * n + j] = t
    }
  return fromStorage(out, [n, n], 'float64')
}

/** A design with no targets (zeros), for the diagnostics that need the features only. */
function featuresOnly(x: TensorLike, weights: TensorLike | undefined, who: string): Design {
  const m = matrixView(x, who)
  return { x: m, y: new Float64Array(m.n), w: sampleWeightsOf(weights, m.n, who) }
}

/**
 * The leverage h_ii of each observation, without forming the hat matrix.
 *
 * @param x - Features `[n, d]`.
 * @param options - Whether the model has an intercept (default true), and sample weights.
 * @returns h, summing to the rank of the design.
 * @example
 * ```ts
 * import { leverage } from 'omai-engine/linear'
 * leverage([[1], [2], [3]]).map((h) => Math.round(h * 1e4) / 1e4) // => [0.8333, 0.3333, 0.8333]
 * ```
 */
export function leverage(
  x: TensorLike,
  options: DiagnosticsOptions & { readonly weights?: TensorLike } = {},
): number[] {
  const design = featuresOnly(x, options.weights, 'leverage')
  return hatDiagonal(core(design, options.fitIntercept ?? true))
}

/**
 * Internally and externally studentised residuals of a least-squares fit.
 *
 * @param data - Features, targets and optional weights.
 * @param options - Whether the model has an intercept (default true).
 * @returns The raw residuals, the internal and the external studentised residuals, and the leverages.
 * @example
 * ```ts
 * import { studentizedResiduals } from 'omai-engine/linear'
 * const r = studentizedResiduals({ x: [[1], [2], [3], [4], [5]], y: [2, 4, 6, 8, 15] })
 * Math.abs(r.external[4] ?? 0) > Math.abs(r.internal[4] ?? 0) // => true
 * ```
 */
export function studentizedResiduals(
  data: LearnData,
  options: DiagnosticsOptions = {},
): {
  readonly residuals: number[]
  readonly internal: number[]
  readonly external: number[]
  readonly leverage: number[]
} {
  const { residuals, internal, external, leverage: lev } = influenceOf(data, options, 'studentizedResiduals')
  return { residuals, internal, external, leverage: lev }
}

/**
 * Cook's distance of each observation: the scaled change of all fitted values when it is left out.
 *
 * @param data - Features, targets and optional weights.
 * @param options - Whether the model has an intercept (default true).
 * @returns D_i = r_i^2 h_i / (p (1 - h_i)).
 * @example
 * ```ts
 * import { cooksDistance } from 'omai-engine/linear'
 * const d = cooksDistance({ x: [[1], [2], [3], [4], [5]], y: [1, 2, 3, 4, 10] })
 * (d[4] ?? 0) > (d[0] ?? 0) // => true
 * ```
 */
export function cooksDistance(data: LearnData, options: DiagnosticsOptions = {}): number[] {
  return influenceOf(data, options, 'cooksDistance').cooksDistance
}

/**
 * DFFITS: the externally studentised residual times sqrt(h / (1 - h)), the scaled change of observation i's own fitted
 * value when it is left out.
 *
 * @param data - Features, targets and optional weights.
 * @param options - Whether the model has an intercept (default true).
 * @returns DFFITS per observation.
 * @example
 * ```ts
 * import { dffits } from 'omai-engine/linear'
 * Math.abs(dffits({ x: [[1], [2], [3], [4], [5]], y: [1, 2, 3, 4, 10] })[4] ?? 0) > 1 // => true
 * ```
 */
export function dffits(data: LearnData, options: DiagnosticsOptions = {}): number[] {
  return influenceOf(data, options, 'dffits').dffits
}

/**
 * DFBETAS: the change of each coefficient when observation i is deleted, scaled by sigma_(i) sqrt((X'X)^-1_jj).
 *
 * @param data - Features, targets and optional weights.
 * @param options - Whether the model has an intercept (default true; its column is last).
 * @returns `[n, p]`.
 * @example
 * ```ts
 * import { dfbetas } from 'omai-engine/linear'
 * dfbetas({ x: [[1], [2], [3], [4], [5]], y: [1, 2, 3, 4, 10] }).shape // => [5, 2]
 * ```
 */
export function dfbetas(data: LearnData, options: DiagnosticsOptions = {}): Tensor {
  return influenceOf(data, options, 'dfbetas').dfbetas
}

/**
 * COVRATIO: how much leaving out observation i changes the determinant of the coefficient covariance (1 is neutral;
 * far from 1 flags influence on the precision of the estimates).
 *
 * @param data - Features, targets and optional weights.
 * @param options - Whether the model has an intercept (default true).
 * @returns (sigma_(i)^2 / sigma^2)^p / (1 - h_i) per observation.
 * @example
 * ```ts
 * import { covRatio } from 'omai-engine/linear'
 * covRatio({ x: [[1], [2], [3], [4], [5]], y: [2.1, 3.9, 6.0, 8.1, 9.9] }).every((v) => v > 0) // => true
 * ```
 */
export function covRatio(data: LearnData, options: DiagnosticsOptions = {}): number[] {
  return influenceOf(data, options, 'covRatio').covRatio
}

/** Residual tools shared by the tests: the OLS fit of a target on columns (intercept appended), its R^2 and rank. */
function auxiliaryFit(
  cols: Float64Array,
  n: number,
  d: number,
  target: ArrayLike<number>,
): { readonly rSquared: number; readonly ess: number; readonly rank: number } {
  const p = d + 1
  const a = new Float64Array(n * p)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) a[i * p + j] = cols[i * d + j] ?? 0
    a[i * p + d] = 1
  }
  const { u, s } = svdOf(a, n, p)
  const k = s.length
  const tol = Math.max(n, p) * Number.EPSILON * (s[0] ?? 0)
  let rank = 0
  const proj = new Float64Array(n)
  for (let q = 0; q < k; q++) {
    if ((s[q] ?? 0) <= tol) continue
    rank++
    let t = 0
    for (let i = 0; i < n; i++) t += (u[i * k + q] ?? 0) * (target[i] ?? 0)
    for (let i = 0; i < n; i++) proj[i] = (proj[i] ?? 0) + t * (u[i * k + q] ?? 0)
  }
  let mean = 0
  for (let i = 0; i < n; i++) mean += (target[i] ?? 0) / n
  let ess = 0
  let tss = 0
  for (let i = 0; i < n; i++) {
    ess += ((proj[i] ?? 0) - mean) ** 2
    tss += ((target[i] ?? 0) - mean) ** 2
  }
  return { rSquared: tss > 0 ? ess / tss : 0, ess, rank }
}

function olsResiduals(data: LearnData, who: string): { readonly residuals: number[]; readonly design: Design } {
  const design = designOf(data, who)
  const c = core(design, true)
  const k = c.k
  const out = Array.from(c.yw)
  for (let q = 0; q < k; q++) {
    if (c.keep[q] !== true) continue
    let t = 0
    for (let i = 0; i < c.n; i++) t += (c.u[i * k + q] ?? 0) * (c.yw[i] ?? 0)
    for (let i = 0; i < c.n; i++) out[i] = (out[i] ?? 0) - t * (c.u[i * k + q] ?? 0)
  }
  return { residuals: out, design }
}

/** Options of {@link breuschPagan}. */
export interface BreuschPaganOptions {
  /** Koenker's studentised statistic n R^2 (default true, robust to non-normal errors); false is the classical score ESS / 2. */
  readonly studentize?: boolean
  /** The variance regressors Z `[n, q]` (an intercept is added; default the model's features). */
  readonly auxiliary?: TensorLike
}

/**
 * The Breusch-Pagan Lagrange multiplier test for heteroscedasticity (statsmodels' `het_breuschpagan`): regress the
 * squared residuals on the variance regressors; under constant variance n R^2 is chi-square with rank - 1 degrees of
 * freedom. Fits the model with an intercept; weights are applied to the residuals only.
 *
 * @param data - Features, targets and optional weights.
 * @param options - Studentisation and the auxiliary regressors.
 * @returns The test result.
 * @example
 * ```ts
 * import { breuschPagan } from 'omai-engine/linear'
 * const x = [[1], [2], [3], [4], [5], [6], [7], [8]]
 * const y = [1.1, 1.9, 3.2, 3.9, 5.1, 5.9, 7.1, 7.9]
 * breuschPagan({ x, y }).pValue > 0.05 // => true
 * ```
 */
export function breuschPagan(data: LearnData, options: BreuschPaganOptions = {}): TestResult {
  const { residuals, design } = olsResiduals(data, 'breuschPagan')
  const n = residuals.length
  let rss = 0
  for (const e of residuals) rss += e * e
  const sigma2 = rss / n
  const g = residuals.map((e) => (e * e) / sigma2)
  const z = options.auxiliary === undefined ? design.x : matrixView(options.auxiliary, 'breuschPagan')
  const fit = auxiliaryFit(z.data, n, z.d, g)
  const df = fit.rank - 1
  const studentize = options.studentize ?? true
  const statistic = studentize ? n * fit.rSquared : fit.ess / 2
  return testResult({
    test: 'Breusch-Pagan test',
    statistic,
    pValue: item(chdtrc(df, statistic)),
    df,
    method: studentize ? 'Koenker (studentised) Lagrange multiplier' : 'Lagrange multiplier',
  })
}

/**
 * White's test for heteroscedasticity (statsmodels' `het_white`): the squared residuals regressed on the features,
 * their squares and cross products; n R^2 is chi-square with rank - 1 degrees of freedom under constant variance.
 *
 * @param data - Features, targets and optional weights.
 * @returns The test result.
 * @example
 * ```ts
 * import { whiteTest } from 'omai-engine/linear'
 * const x = [[1], [2], [3], [4], [5], [6], [7], [8]]
 * const y = [1.1, 1.9, 3.2, 3.9, 5.1, 5.9, 7.1, 7.9]
 * whiteTest({ x, y }).df // => 2
 * ```
 */
export function whiteTest(data: LearnData): TestResult {
  const { residuals, design } = olsResiduals(data, 'whiteTest')
  const { n, d } = design.x
  const q = d + (d * (d + 1)) / 2
  const z = new Float64Array(n * q)
  for (let i = 0; i < n; i++) {
    let c = 0
    for (let j = 0; j < d; j++) z[i * q + c++] = design.x.data[i * d + j] ?? 0
    for (let j = 0; j < d; j++)
      for (let l = j; l < d; l++) z[i * q + c++] = (design.x.data[i * d + j] ?? 0) * (design.x.data[i * d + l] ?? 0)
  }
  const fit = auxiliaryFit(
    z,
    n,
    q,
    residuals.map((e) => e * e),
  )
  const df = fit.rank - 1
  const statistic = n * fit.rSquared
  return testResult({
    test: 'White test',
    statistic,
    pValue: item(chdtrc(df, statistic)),
    df,
    method: 'Lagrange multiplier',
  })
}

/**
 * Variance inflation factors VIF_j = 1 / (1 - R_j^2), where R_j^2 is the share of column j explained by the other
 * columns (and an intercept), as the diagonal of the inverse correlation matrix. Values above about 5 to 10 flag
 * collinearity.
 *
 * @param x - Features `[n, d]`.
 * @param options - Whether to centre the columns (default true; false uses the uncentred R^2 as statsmodels does without a constant).
 * @returns One factor per column (infinite for a column that is a linear combination of the others).
 * @example
 * ```ts
 * import { varianceInflationFactors } from 'omai-engine/linear'
 * const v = varianceInflationFactors([[1, 2.1], [2, 3.9], [3, 6.2], [4, 7.8], [5, 10.1]])
 * (v[0] ?? 0) > 50 // => true
 * ```
 */
export function varianceInflationFactors(x: TensorLike, options: DiagnosticsOptions = {}): number[] {
  const m = matrixView(x, 'varianceInflationFactors')
  const { n, d } = m
  const centre = options.fitIntercept ?? true
  const cols = new Float64Array(n * d)
  for (let j = 0; j < d; j++) {
    let mean = 0
    if (centre) for (let i = 0; i < n; i++) mean += (m.data[i * d + j] ?? 0) / n
    let norm = 0
    for (let i = 0; i < n; i++) norm += ((m.data[i * d + j] ?? 0) - mean) ** 2
    norm = Math.sqrt(norm)
    for (let i = 0; i < n; i++) cols[i * d + j] = norm > 0 ? ((m.data[i * d + j] ?? 0) - mean) / norm : 0
  }
  // diag((R)^-1) = sum_q V_jq^2 / s_q^2 for the unit-norm columns.
  const { s, vh } = svdOf(cols, n, d)
  const tol = Math.max(n, d) * Number.EPSILON * (s[0] ?? 0)
  return Array.from({ length: d }, (_, j) => {
    let t = 0
    for (let q = 0; q < s.length; q++) {
      if ((s[q] ?? 0) <= tol) return Number.POSITIVE_INFINITY
      t += ((vh[q * d + j] ?? 0) / (s[q] ?? 1)) ** 2
    }
    return t
  })
}

/** Belsley-Kuh-Welsch collinearity diagnostics. */
export interface ConditionIndices {
  /** Singular values of the column-equilibrated design, descending. */
  readonly singularValues: number[]
  /** mu_max / mu_k for each singular value. */
  readonly conditionIndices: number[]
  /** The largest condition index. */
  readonly conditionNumber: number
  /** Variance-decomposition proportions `[p, p]`: entry (k, j) is the share of coefficient j's variance on singular value k. */
  readonly varianceProportions: Tensor
}

/**
 * Belsley-Kuh-Welsch condition indices and variance-decomposition proportions of the design with its columns scaled
 * to unit length (intercept column last). A condition index above about 30 together with two or more variance
 * proportions above 0.5 on the same row marks a near dependency.
 *
 * @param x - Features `[n, d]`.
 * @param options - Whether the model has an intercept (default true).
 * @returns The singular values, indices, condition number and `[p, p]` proportions (columns sum to one).
 * @example
 * ```ts
 * import { conditionIndices } from 'omai-engine/linear'
 * conditionIndices([[1, 2], [2, 4.001], [3, 6], [4, 8.001]]).conditionNumber > 50 // => true
 * ```
 */
export function conditionIndices(x: TensorLike, options: DiagnosticsOptions = {}): ConditionIndices {
  const design = featuresOnly(x, undefined, 'conditionIndices')
  const { a, p } = withIntercept(design, options.fitIntercept ?? true)
  const { n } = design.x
  const scaled = new Float64Array(n * p)
  for (let j = 0; j < p; j++) {
    let norm = 0
    for (let i = 0; i < n; i++) norm += (a[i * p + j] ?? 0) ** 2
    norm = Math.sqrt(norm) || 1
    for (let i = 0; i < n; i++) scaled[i * p + j] = (a[i * p + j] ?? 0) / norm
  }
  const { s, vh } = svdOf(scaled, n, p)
  const k = s.length
  const top = s[0] ?? 0
  const indices = Array.from(s, (mu) => (mu > 0 ? top / mu : Number.POSITIVE_INFINITY))
  const phi = new Float64Array(k * p)
  const totals = new Float64Array(p)
  for (let q = 0; q < k; q++)
    for (let j = 0; j < p; j++) {
      const v = (vh[q * p + j] ?? 0) ** 2 / (s[q] ?? 1) ** 2
      phi[q * p + j] = v
      totals[j] = (totals[j] ?? 0) + v
    }
  const proportions = new Float64Array(k * p)
  for (let q = 0; q < k; q++)
    for (let j = 0; j < p; j++) proportions[q * p + j] = (phi[q * p + j] ?? 0) / (totals[j] || 1)
  return {
    singularValues: Array.from(s),
    conditionIndices: indices,
    conditionNumber: indices[k - 1] ?? 1,
    varianceProportions: fromStorage(proportions, [k, p], 'float64'),
  }
}

/** A diagnostics report: the influence measures, collinearity and the residual tests. */
export interface RegressionDiagnostics {
  readonly influence: Influence
  /** VIFs of the features. */
  readonly vif: number[]
  readonly collinearity: ConditionIndices
  readonly durbinWatson: TestResult
  readonly breuschPagan: TestResult
  readonly white: TestResult
  /** Observations over Cook's 4/n, DFFITS' 2 sqrt(p/n) or whose externally studentised residual is significant at 5%. */
  readonly influentialPoints: number[]
}

/**
 * Every diagnostic of a least-squares fit in one report, with the observations the usual cut-offs flag.
 *
 * @param data - Features, targets and optional weights.
 * @param options - Whether the model has an intercept (default true).
 * @returns The {@link RegressionDiagnostics}.
 * @example
 * ```ts
 * import { regressionDiagnostics } from 'omai-engine/linear'
 * const r = regressionDiagnostics({ x: [[1], [2], [3], [4], [5], [6]], y: [2, 4, 6, 8, 10, 30] })
 * r.influentialPoints.includes(5) // => true
 * ```
 */
export function regressionDiagnostics(data: LearnData, options: DiagnosticsOptions = {}): RegressionDiagnostics {
  const influence = influenceOf(data, options, 'regressionDiagnostics')
  const { n, rank, dfResid } = influence
  const flagged: number[] = []
  for (let i = 0; i < n; i++) {
    const t = Math.abs(influence.external[i] ?? 0)
    const pOutlier = dfResid > 1 ? 2 * (1 - item(stdtr(dfResid - 1, t))) : 1
    if (
      (influence.cooksDistance[i] ?? 0) > 4 / n ||
      Math.abs(influence.dffits[i] ?? 0) > 2 * Math.sqrt(rank / n) ||
      pOutlier < 0.05
    )
      flagged.push(i)
  }
  return {
    influence,
    vif: varianceInflationFactors(data.x, options),
    collinearity: conditionIndices(data.x, options),
    durbinWatson: durbinWatson(influence.residuals),
    breuschPagan: breuschPagan(data),
    white: whiteTest(data),
    influentialPoints: flagged,
  }
}
