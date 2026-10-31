/**
 * Inference on a fitted generalised additive model, from the Bayesian posterior covariance `scale (X'WX + S)^(-1)` of the
 * coefficients: the summary table (estimate, effective degrees of freedom, reference degrees of freedom, statistic and
 * p-value per term, with Wood's 2013 test for smooth terms and the Wald test for parametric ones), pointwise and
 * simultaneous credible bands for each partial effect, the linear predictor and the response, and posterior simulation of
 * the coefficients and the fitted curves. Everything starts from a {@link GamFitContext}, the fitted pieces of a
 * problem at a state, so it applies to a model whichever algorithm produced the state.
 */

import { MdmlError } from '../base/index.ts'
import { dgemm, dgemv, dgeqrf, dpotrf, dpotrs, dsyev } from '../numerics/linalg/index.ts'
import { betainc, gammaincc, ndtr, ndtri } from '../numerics/special/index.ts'
import { type Key, normal } from '../random/index.ts'
import { fromStorage, item, type TensorLike, toFlat } from '../tensor/index.ts'
import { hessianSolve, type PenalisedSolution, posteriorCovariance } from './penalised.ts'
import { type GamProblem, type GamState, gamSolution } from './pirls.ts'
import { evaluateDesign } from './terms.ts'

/** The fitted pieces inference works from: the coefficients, covariances and degrees of freedom of a fit. */
export interface GamFitContext {
  readonly problem: GamProblem
  readonly beta: Float64Array
  /** The linear predictor at the data. */
  readonly eta: Float64Array
  /** The fitted means. */
  readonly mu: Float64Array
  readonly scale: number
  /** Whether the scale was estimated (so tests use `t` and `F` references instead of the normal and chi-square). */
  readonly scaleEstimated: boolean
  readonly logLambda: readonly number[]
  /** `X'WX` at the fit (Fisher weights), `[p, p]` row-major. */
  readonly information: Float64Array
  /** The Bayesian covariance `scale (X'WX + S)^(-1)`, `[p, p]` row-major. */
  readonly vp: Float64Array
  /** The frequentist covariance `scale (X'WX + S)^(-1) X'WX (X'WX + S)^(-1)`. */
  readonly vf: Float64Array
  /** The effective degrees of freedom of each coefficient (the diagonal of `(X'WX + S)^(-1) X'WX`). */
  readonly edf: Float64Array
  /** `2 edf - diag(F F)`: the degrees of freedom the reference distribution of a smooth's test is matched to. */
  readonly edf1: Float64Array
  readonly edfTotal: number
  /** `n - edfTotal`, the degrees of freedom of the scale estimate. */
  readonly residualDf: number
  readonly deviance: number
  readonly solution: PenalisedSolution
}

/**
 * The fitted pieces of a problem at a state: covariances of the coefficients and the degrees of freedom per coefficient.
 *
 * @param problem - The problem.
 * @param state - A state of the fit (from {@link fitGam} or {@link gamAlgorithm}).
 * @returns The context.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 6 })] }
 * const problem = gamProblem(x, x.map((r) => Math.sin(5 * (r[0] ?? 0))), hyper)
 * const ctx = fitContext(problem, fitGam(problem, hyper).state)
 * ctx.vp.length === ctx.problem.design.p ** 2 // => true
 * ```
 */
export function fitContext(problem: GamProblem, state: GamState): GamFitContext {
  const { g, solution, scale } = gamSolution(problem, state)
  const p = problem.design.p
  const vp = posteriorCovariance(solution, p, scale)
  // F = (X'WX + S)^(-1) X'WX column by column; edf = diag F; Vf = F Vp.
  const F = new Float64Array(p * p)
  const col = new Float64Array(p)
  for (let c = 0; c < p; c++) {
    for (let r = 0; r < p; r++) col[r] = g.xtwx[r * p + c] ?? 0
    const sol = hessianSolve(solution, col)
    for (let r = 0; r < p; r++) F[r * p + c] = sol[r] ?? 0
  }
  const edf = new Float64Array(p)
  const edf1 = new Float64Array(p)
  for (let i = 0; i < p; i++) {
    let ff = 0
    for (let k = 0; k < p; k++) ff += (F[i * p + k] ?? 0) * (F[k * p + i] ?? 0)
    edf[i] = F[i * p + i] ?? 0
    edf1[i] = 2 * (edf[i] ?? 0) - ff
  }
  const vf = dgemm(F, vp, p, p, p)
  for (let i = 0; i < p; i++)
    for (let j = 0; j < i; j++) vf[i * p + j] = vf[j * p + i] = ((vf[i * p + j] ?? 0) + (vf[j * p + i] ?? 0)) / 2
  let n = 0
  for (let i = 0; i < problem.prior.length; i++) if ((problem.prior[i] ?? 0) > 0) n++
  const edfTotal = edf.reduce((a, b) => a + b, 0)
  const mu = Float64Array.from(
    toFlat(problem.link.inverse(fromStorage(Float64Array.from(state.eta), [state.eta.length], 'float64'))),
  )
  return {
    problem,
    beta: Float64Array.from(state.beta),
    eta: Float64Array.from(state.eta),
    mu,
    scale,
    scaleEstimated: problem.knownScale === undefined,
    logLambda: state.logLambda,
    information: g.xtwx,
    vp,
    vf,
    edf,
    edf1,
    edfTotal,
    residualDf: n - edfTotal,
    deviance: state.deviance,
    solution,
  }
}

/** The survival function of the chi-square distribution with `df` degrees of freedom. */
function chiSquareSf(x: number, df: number): number {
  return x <= 0 ? 1 : item(gammaincc(df / 2, x / 2))
}

/** The survival function of the F distribution. */
function fSf(x: number, d1: number, d2: number): number {
  return x <= 0 ? 1 : item(betainc(d2 / 2, d1 / 2, d2 / (d2 + d1 * x)))
}

/**
 * The upper tail `P(sum_i w_i chi2_1 > x)` of a weighted sum of independent chi-square variables with one degree of
 * freedom (positive weights), by the moment-matching approximation of Liu, Tang and Zhang (2009): a shifted and scaled
 * chi-square that has the sum's mean, variance and skewness. With equal weights it is exact.
 *
 * @param x - The point.
 * @param weights - The positive weights.
 * @returns The probability.
 * @concept generalized-additive-models
 * @reference H. Liu, Y. Tang and H. H. Zhang, A new chi-square approximation to the distribution of non-negative
 *   definite quadratic forms in non-central normal variables (Computational Statistics and Data Analysis 53, 2009).
 * @example
 * ```ts
 * import { weightedChiSquareSf } from 'mdml-engine/gam'
 * Math.abs(weightedChiSquareSf(3.841458820694124, [1]) - 0.05) < 1e-9 // => true
 * ```
 */
export function weightedChiSquareSf(x: number, weights: readonly number[]): number {
  if (weights.length === 0) return x > 0 ? 0 : 1
  const c = [0, 1, 2, 3].map((k) => weights.reduce((a, w) => a + w ** k, 0))
  const [, c1, c2, c3] = c as [number, number, number, number]
  const mean = c1
  const sd = Math.sqrt(2 * c2)
  // For central variables s1^2 <= s2 (Cauchy-Schwarz), so the matching chi-square is central with 1 / s1^2 degrees of
  // freedom.
  const s1 = c3 / c2 ** 1.5
  const l = 1 / (s1 * s1)
  const t = ((x - mean) / sd) * Math.sqrt(2 * l) + l
  return chiSquareSf(t, l)
}

/**
 * The reference test of a smooth term (Wood, 2013): the statistic `T = f' V^- f` of the fitted effect `f` at the data
 * with the pseudo-inverse of its posterior covariance truncated at the fractional rank `refDf` (the eigenvalue beyond
 * the integer part enters with a fractional weight), and its p-value under the null that the effect is zero.
 *
 * @param X - The term's columns of the design `[n, w]`, row-major.
 * @param n - Rows.
 * @param w - Columns.
 * @param beta - The term's coefficients `[w]`.
 * @param V - Their posterior covariance `[w, w]`.
 * @param refDf - The reference degrees of freedom (the smooth's `edf1`, at most `w`).
 * @param residualDf - The degrees of freedom of the scale estimate, or `0` for a known scale.
 * @returns The statistic (`T`), the rank it was truncated at and the p-value (an F reference when the scale is estimated).
 * @concept generalized-additive-models
 * @reference S. N. Wood, On p-values for smooth components of an extended empirical Bayesian GAM (Biometrika 100, 2013).
 * @example
 * ```ts
 * import { referenceTest } from 'mdml-engine/gam'
 * // One coefficient of 3 standard errors, tested as a Wald statistic on one degree of freedom.
 * const r = referenceTest([1, 2, 3, 4], 4, 1, [3], [1], 1, 0)
 * [Math.round(r.statistic), r.rank] // => [9, 1]
 * ```
 */
export function referenceTest(
  X: ArrayLike<number>,
  n: number,
  w: number,
  beta: ArrayLike<number>,
  V: ArrayLike<number>,
  refDf: number,
  residualDf: number,
): { readonly statistic: number; readonly rank: number; readonly pValue: number } {
  const { r } = dgeqrf(X, n, w)
  const k0 = Math.min(n, w)
  // V' = R V R' and the effect's coordinates R beta.
  const rb = dgemv(r, k0, w, beta)
  const rv = dgemm(dgemm(r, V, k0, w, w), r, k0, w, k0, { transposeB: true })
  for (let i = 0; i < k0; i++)
    for (let j = 0; j < i; j++) rv[i * k0 + j] = rv[j * k0 + i] = ((rv[i * k0 + j] ?? 0) + (rv[j * k0 + i] ?? 0)) / 2
  const { w: values, v: vectors } = dsyev(rv, k0)
  // Descending eigenvalues.
  const lambda = (i: number): number => Math.max(values[k0 - 1 - i] ?? 0, 0)
  const vec = (i: number): Float64Array =>
    Float64Array.from({ length: k0 }, (_, row) => vectors[row * k0 + (k0 - 1 - i)] ?? 0)
  const z = (i: number): number => {
    const u = vec(i)
    let a = 0
    for (let row = 0; row < k0; row++) a += (u[row] ?? 0) * (rb[row] ?? 0)
    return a / Math.sqrt(lambda(i))
  }
  let k = Math.max(0, Math.floor(Math.min(refDf, k0)))
  let nu = Math.min(refDf, k0) - k
  const top = lambda(0)
  let rank = 0
  for (let i = 0; i < k0; i++) if (lambda(i) > top * Number.EPSILON ** 0.9) rank++
  if (rank < (nu > 0 ? k + 1 : k)) {
    k = rank
    nu = 0
  }
  let statistic = 0
  let weights: number[]
  if (k === 0 && nu > 0) {
    statistic = nu * z(0) ** 2
    weights = [nu]
  } else {
    for (let i = 0; i < (nu > 0 ? k - 1 : k); i++) statistic += z(i) ** 2
    weights = new Array<number>(nu > 0 ? k - 1 : k).fill(1)
    if (nu > 0) {
      const b = Math.sqrt(Math.max(0.5 * nu * (1 - nu), 0))
      const za = z(k - 1)
      const zb = z(k)
      statistic += za * za + 2 * b * za * zb + nu * zb * zb
      // The eigenvalues of [[1, b], [b, nu]].
      const tr = 1 + nu
      const disc = Math.sqrt(Math.max(((1 - nu) / 2) ** 2 + b * b, 0))
      weights.push(tr / 2 + disc, tr / 2 - disc)
    }
  }
  const total = k + nu
  const pValue =
    residualDf <= 0
      ? weightedChiSquareSf(statistic, weights)
      : (() => {
          // Welch-Satterthwaite: the weighted sum is matched to sigma chi2_l, the ratio to the scale estimate is an F.
          const c1 = weights.reduce((a, v) => a + v, 0)
          const c2 = weights.reduce((a, v) => a + v * v, 0)
          const sigma = c2 / c1
          const l = (c1 * c1) / c2
          return fSf(statistic / (sigma * l), l, residualDf)
        })()
  return { statistic, rank: total, pValue }
}

/** The Wald test of a block of unpenalised coefficients: `T = b' V^(-1) b` on `w` degrees of freedom. */
function waldTest(
  beta: ArrayLike<number>,
  V: ArrayLike<number>,
  w: number,
  residualDf: number,
): { readonly statistic: number; readonly pValue: number } {
  const f = dpotrf(V, w)
  if (f.info !== 0) return { statistic: Number.NaN, pValue: Number.NaN }
  const sol = dpotrs(f.factor, beta, w)
  let t = 0
  for (let i = 0; i < w; i++) t += (beta[i] ?? 0) * (sol[i] ?? 0)
  return { statistic: t, pValue: residualDf <= 0 ? chiSquareSf(t, w) : fSf(t / w, w, residualDf) }
}

/** A row of the table of parametric terms. */
export interface ParametricTest {
  readonly label: string
  /** The coefficients tested together. */
  readonly df: number
  /** The coefficient when the term has one. */
  readonly estimate: number | undefined
  readonly stdError: number | undefined
  /** `z` or `t` for one coefficient, `chi2` or `F` for several (`t` and `F` when the scale is estimated). */
  readonly statisticKind: 'z' | 't' | 'chi2' | 'F'
  /** The statistic (for several coefficients: the chi-square statistic divided by `df` when `F`). */
  readonly statistic: number
  readonly pValue: number
}

/** A row of the table of smooth terms. */
export interface SmoothTest {
  readonly label: string
  /** The basis dimension (columns of the term). */
  readonly k: number
  /** The effective degrees of freedom. */
  readonly edf: number
  /** The reference degrees of freedom the test statistic is matched to. */
  readonly refDf: number
  /** `chi2`, or `F` (the statistic divided by `refDf`) when the scale is estimated. */
  readonly statisticKind: 'chi2' | 'F'
  readonly statistic: number
  readonly pValue: number
}

/** The summary table of a fitted model (mgcv's `summary.gam`). */
export interface GamSummary {
  readonly family: string
  readonly link: string
  readonly n: number
  readonly intercept: ParametricTest
  /** The unpenalised linear and factor terms. */
  readonly parametric: readonly ParametricTest[]
  /** The penalised terms: smooths, tensors, random effects. */
  readonly smooth: readonly SmoothTest[]
  readonly scale: number
  readonly scaleEstimated: boolean
  readonly residualDf: number
  readonly edfTotal: number
  readonly deviance: number
  readonly nullDeviance: number
  /** `1 - deviance / nullDeviance`. */
  readonly devianceExplained: number
}

function parametricRow(ctx: GamFitContext, label: string, offset: number, w: number): ParametricTest {
  const p = ctx.problem.design.p
  const beta = Array.from({ length: w }, (_, j) => ctx.beta[offset + j] ?? 0)
  const V = new Float64Array(w * w)
  for (let i = 0; i < w; i++) for (let j = 0; j < w; j++) V[i * w + j] = ctx.vp[(offset + i) * p + offset + j] ?? 0
  const df = ctx.scaleEstimated ? ctx.residualDf : 0
  const wald = waldTest(beta, V, w, df)
  if (w === 1) {
    const est = beta[0] ?? 0
    const se = Math.sqrt(V[0] ?? 0)
    const stat = est / se
    const pValue = ctx.scaleEstimated
      ? item(betainc(ctx.residualDf / 2, 0.5, ctx.residualDf / (ctx.residualDf + stat * stat)))
      : 2 * item(ndtr(-Math.abs(stat)))
    return {
      label,
      df: 1,
      estimate: est,
      stdError: se,
      statisticKind: ctx.scaleEstimated ? 't' : 'z',
      statistic: stat,
      pValue,
    }
  }
  return {
    label,
    df: w,
    estimate: undefined,
    stdError: undefined,
    statisticKind: ctx.scaleEstimated ? 'F' : 'chi2',
    statistic: ctx.scaleEstimated ? wald.statistic / w : wald.statistic,
    pValue: wald.pValue,
  }
}

/**
 * The summary table of a fitted model: the intercept and each unpenalised term with its Wald test, each penalised term
 * with its effective and reference degrees of freedom, test statistic and p-value ({@link referenceTest}), the scale,
 * and the deviance explained. The test of a smooth asks whether its partial effect could be zero; a smooth with a null
 * space is tested including it.
 *
 * @param ctx - The fitted pieces ({@link fitContext}).
 * @returns The summary.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, gamSummary, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 60 }, (_, i) => [i / 59])
 * const y = x.map((r, i) => Math.sin(6 * (r[0] ?? 0)) + 0.3 * Math.sin(i * 12.9898))
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 8 })] }
 * const problem = gamProblem(x, y, hyper)
 * const table = gamSummary(fitContext(problem, fitGam(problem, hyper).state))
 * [table.smooth[0]?.label, (table.smooth[0]?.pValue ?? 1) < 0.001] // => ['s(x0)', true]
 * ```
 */
export function gamSummary(ctx: GamFitContext): GamSummary {
  const { design } = ctx.problem
  const parametric: ParametricTest[] = []
  const smooth: SmoothTest[] = []
  design.terms.forEach((t) => {
    if (t.penalties.length === 0) {
      parametric.push(parametricRow(ctx, t.label, t.offset, t.width))
      return
    }
    let edf = 0
    let edf1 = 0
    for (let j = 0; j < t.width; j++) {
      edf += ctx.edf[t.offset + j] ?? 0
      edf1 += ctx.edf1[t.offset + j] ?? 0
    }
    const X = new Float64Array(design.n * t.width)
    const V = new Float64Array(t.width * t.width)
    for (let i = 0; i < design.n; i++)
      for (let j = 0; j < t.width; j++) X[i * t.width + j] = design.X[i * design.p + t.offset + j] ?? 0
    for (let i = 0; i < t.width; i++)
      for (let j = 0; j < t.width; j++) V[i * t.width + j] = ctx.vp[(t.offset + i) * design.p + t.offset + j] ?? 0
    const beta = Array.from({ length: t.width }, (_, j) => ctx.beta[t.offset + j] ?? 0)
    const refDf = Math.min(t.width, Math.max(edf1, 1e-6))
    const test = referenceTest(X, design.n, t.width, beta, V, refDf, ctx.scaleEstimated ? ctx.residualDf : 0)
    smooth.push({
      label: t.label,
      k: t.width,
      edf,
      refDf: test.rank,
      statisticKind: ctx.scaleEstimated ? 'F' : 'chi2',
      statistic: ctx.scaleEstimated ? test.statistic / test.rank : test.statistic,
      pValue: test.pValue,
    })
  })
  let n = 0
  let sw = 0
  let swy = 0
  for (let i = 0; i < ctx.problem.y.length; i++) {
    const wi = ctx.problem.prior[i] ?? 0
    if (wi > 0) n++
    sw += wi
    swy += wi * (ctx.problem.y[i] ?? 0)
  }
  const ybar = swy / sw
  const nullMu = fromStorage(new Float64Array(ctx.problem.y.length).fill(ybar), [ctx.problem.y.length], 'float64')
  const yv = fromStorage(Float64Array.from(ctx.problem.y), [ctx.problem.y.length], 'float64')
  const pv = fromStorage(Float64Array.from(ctx.problem.prior), [ctx.problem.y.length], 'float64')
  const nullDeviance = toFlat(ctx.problem.family.deviance(yv, nullMu, pv))[0] ?? Number.NaN
  return {
    family: ctx.problem.family.name,
    link: ctx.problem.link.name,
    n,
    intercept: parametricRow(ctx, '(Intercept)', 0, 1),
    parametric,
    smooth,
    scale: ctx.scale,
    scaleEstimated: ctx.scaleEstimated,
    residualDf: ctx.residualDf,
    edfTotal: ctx.edfTotal,
    deviance: ctx.deviance,
    nullDeviance,
    devianceExplained: 1 - ctx.deviance / nullDeviance,
  }
}

/**
 * Draws from the Gaussian approximation to the posterior of the coefficients, `N(beta, Vp)`; the covariance may be
 * singular (a constrained or fully penalised direction), which the eigen-factor handles.
 *
 * @param ctx - The fitted pieces.
 * @param count - The number of draws.
 * @param key - The random key.
 * @returns The draws `[count, p]`, row-major.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, s, sampleCoefficients } from 'mdml-engine/gam'
 * import { key } from 'mdml-engine/random'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 5 })] }
 * const problem = gamProblem(x, x.map((r) => r[0] ?? 0), hyper)
 * const ctx = fitContext(problem, fitGam(problem, hyper).state)
 * sampleCoefficients(ctx, 7, key(0)).length === 7 * ctx.problem.design.p // => true
 * ```
 */
export function sampleCoefficients(ctx: GamFitContext, count: number, key: Key): Float64Array {
  const p = ctx.problem.design.p
  return drawFrom(ctx.beta, ctx.vp, p, count, key)
}

/** `count` draws from `N(mean, V)` with `V` `[d, d]`, through its eigen-factor. */
function drawFrom(mean: ArrayLike<number>, V: ArrayLike<number>, d: number, count: number, key: Key): Float64Array {
  const { w, v } = dsyev(V, d)
  const root = Float64Array.from(w, (x) => Math.sqrt(Math.max(x, 0)))
  const z = normal(key, [count, d]).data as Float64Array
  const out = new Float64Array(count * d)
  for (let s = 0; s < count; s++)
    for (let i = 0; i < d; i++) {
      let a = 0
      for (let j = 0; j < d; j++) a += (v[i * d + j] ?? 0) * (root[j] ?? 0) * (z[s * d + j] ?? 0)
      out[s * d + i] = (mean[i] ?? 0) + a
    }
  return out
}

/** Options of the credible bands. */
export interface BandOptions {
  /** The coverage (default 0.95). */
  readonly level?: number
  /** `link` (default) or `response` (the bounds are mapped through the inverse link). */
  readonly scale?: 'link' | 'response'
  /** Simultaneous bands by posterior simulation: the draws (default 5000) and their key. */
  readonly simultaneous?: { readonly key: Key; readonly draws?: number }
}

/** A credible band over a set of points. */
export interface Band {
  readonly fit: Float64Array
  /** The posterior standard error on the link scale. */
  readonly se: Float64Array
  readonly lower: Float64Array
  readonly upper: Float64Array
  readonly level: number
  /** The factor on `se`: the normal quantile, or the simulated quantile of the maximal standardised deviation. */
  readonly multiplier: number
  readonly simultaneous: boolean
}

function bandOf(
  ctx: GamFitContext,
  X: Float64Array,
  m: number,
  columns: readonly number[],
  options: BandOptions,
): Band {
  const p = ctx.problem.design.p
  const c = columns.length
  const Xc = new Float64Array(m * c)
  for (let i = 0; i < m; i++) for (let j = 0; j < c; j++) Xc[i * c + j] = X[i * p + (columns[j] ?? 0)] ?? 0
  const bc = Float64Array.from(columns, (j) => ctx.beta[j] ?? 0)
  const V = new Float64Array(c * c)
  for (let i = 0; i < c; i++)
    for (let j = 0; j < c; j++) V[i * c + j] = ctx.vp[(columns[i] ?? 0) * p + (columns[j] ?? 0)] ?? 0
  const fit = dgemv(Xc, m, c, bc)
  const xv = dgemm(Xc, V, m, c, c)
  const se = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    let a = 0
    for (let j = 0; j < c; j++) a += (xv[i * c + j] ?? 0) * (Xc[i * c + j] ?? 0)
    se[i] = Math.sqrt(Math.max(a, 0))
  }
  const level = options.level ?? 0.95
  if (!(level > 0 && level < 1)) throw new MdmlError('value', 'bands: level must lie in (0, 1)')
  let multiplier = item(ndtri(0.5 + level / 2))
  if (options.simultaneous !== undefined) {
    const draws = options.simultaneous.draws ?? 5000
    const delta = drawFrom(new Float64Array(c), V, c, draws, options.simultaneous.key)
    const maxima = new Float64Array(draws)
    for (let s = 0; s < draws; s++) {
      let best = 0
      for (let i = 0; i < m; i++) {
        if ((se[i] ?? 0) <= 0) continue
        let a = 0
        for (let j = 0; j < c; j++) a += (Xc[i * c + j] ?? 0) * (delta[s * c + j] ?? 0)
        best = Math.max(best, Math.abs(a) / (se[i] ?? 1))
      }
      maxima[s] = best
    }
    maxima.sort()
    multiplier = maxima[Math.min(draws - 1, Math.ceil(level * draws) - 1)] ?? multiplier
  }
  let lower = Float64Array.from(fit, (f, i) => f - multiplier * (se[i] ?? 0))
  let upper = Float64Array.from(fit, (f, i) => f + multiplier * (se[i] ?? 0))
  let centre = fit
  if (options.scale === 'response') {
    const through = (v: Float64Array): Float64Array =>
      Float64Array.from(toFlat(ctx.problem.link.inverse(fromStorage(v, [v.length], 'float64'))))
    const a = through(lower)
    const b = through(upper)
    lower = Float64Array.from(a, (v, i) => Math.min(v, b[i] ?? v))
    upper = Float64Array.from(a, (v, i) => Math.max(v, b[i] ?? v))
    centre = through(fit)
  }
  return { fit: centre, se, lower, upper, level, multiplier, simultaneous: options.simultaneous !== undefined }
}

/**
 * The credible band of one term's partial effect at new rows, pointwise or simultaneous.
 *
 * The pointwise band is `effect +- z se` with `se` from the Bayesian posterior covariance of the term's coefficients
 * (Nychka's across-the-function coverage: about 95% of the points of the true curve lie inside). The simultaneous
 * band replaces `z` by the quantile of `max_x |f*(x) - f(x)| / se(x)` over posterior draws `f*`, so that the whole
 * curve lies inside with the stated probability (Ruppert, Wand and Carroll; Marra and Wood 2012).
 *
 * @param ctx - The fitted pieces.
 * @param term - The index of the term.
 * @param x - New covariates `[m, d]`.
 * @param options - The level, and `simultaneous` for the simulation-based multiplier.
 * @returns The band of the partial effect (the intercept is not included).
 * @throws MdmlError `value` for an index with no term.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, s, termBand } from 'mdml-engine/gam'
 * const x = Array.from({ length: 50 }, (_, i) => [i / 49])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 6 })] }
 * const problem = gamProblem(x, x.map((r, i) => Math.sin(5 * (r[0] ?? 0)) + 0.2 * Math.sin(i * 12.9898)), hyper)
 * const band = termBand(fitContext(problem, fitGam(problem, hyper).state), 0, [[0.5]])
 * (band.lower[0] ?? 0) < (band.fit[0] ?? 0) && (band.fit[0] ?? 0) < (band.upper[0] ?? 0) // => true
 * ```
 */
export function termBand(ctx: GamFitContext, term: number, x: TensorLike, options: BandOptions = {}): Band {
  const t = ctx.problem.design.terms[term]
  if (t === undefined) throw new MdmlError('value', `termBand: no term ${term}`)
  const { X, n } = evaluateDesign(ctx.problem.design, x)
  const columns = Array.from({ length: t.width }, (_, j) => t.offset + j)
  return bandOf(ctx, X, n, columns, { ...options, scale: 'link' })
}

/**
 * The credible band of the linear predictor (or, on `scale: 'response'`, of the mean) at new rows, pointwise or
 * simultaneous; the band includes the intercept and every term.
 *
 * @param ctx - The fitted pieces.
 * @param x - New covariates `[m, d]`.
 * @param options - The level, the scale and `simultaneous`.
 * @returns The band.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, predictionBand, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 50 }, (_, i) => [i / 49])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 6 })] }
 * const problem = gamProblem(x, x.map((r) => 2 * (r[0] ?? 0)), hyper)
 * predictionBand(fitContext(problem, fitGam(problem, hyper).state), [[0.2], [0.8]]).fit.length // => 2
 * ```
 */
export function predictionBand(ctx: GamFitContext, x: TensorLike, options: BandOptions = {}): Band {
  const { X, n } = evaluateDesign(ctx.problem.design, x)
  return bandOf(
    ctx,
    X,
    n,
    Array.from({ length: ctx.problem.design.p }, (_, j) => j),
    options,
  )
}

/**
 * Posterior draws of a fitted curve at new rows: the partial effect of one term (`term` given), or the linear predictor
 * or the mean (`scale: 'response'`) of the whole model, for coefficients drawn from `N(beta, Vp)`.
 *
 * @param ctx - The fitted pieces.
 * @param x - New covariates `[m, d]`.
 * @param count - The number of draws.
 * @param key - The random key.
 * @param options - `term` for one partial effect, and the `scale`.
 * @returns The draws `[count, m]`, row-major.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitContext, fitGam, gamProblem, s, simulateCurves } from 'mdml-engine/gam'
 * import { key } from 'mdml-engine/random'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 5 })] }
 * const problem = gamProblem(x, x.map((r) => r[0] ?? 0), hyper)
 * const ctx = fitContext(problem, fitGam(problem, hyper).state)
 * simulateCurves(ctx, [[0.1], [0.5], [0.9]], 4, key(1), { term: 0 }).length // => 12
 * ```
 */
export function simulateCurves(
  ctx: GamFitContext,
  x: TensorLike,
  count: number,
  key: Key,
  options: { readonly term?: number; readonly scale?: 'link' | 'response' } = {},
): Float64Array {
  const { design } = ctx.problem
  const { X, n } = evaluateDesign(design, x)
  const p = design.p
  const draws = sampleCoefficients(ctx, count, key)
  const t = options.term === undefined ? undefined : design.terms[options.term]
  if (options.term !== undefined && t === undefined)
    throw new MdmlError('value', `simulateCurves: no term ${options.term}`)
  const out = new Float64Array(count * n)
  for (let s = 0; s < count; s++)
    for (let i = 0; i < n; i++) {
      let a = 0
      const from = t === undefined ? 0 : t.offset
      const to = t === undefined ? p : t.offset + t.width
      for (let j = from; j < to; j++) a += (X[i * p + j] ?? 0) * (draws[s * p + j] ?? 0)
      out[s * n + i] = a
    }
  if (options.scale === 'response' && t === undefined) {
    return Float64Array.from(toFlat(ctx.problem.link.inverse(fromStorage(out, [out.length], 'float64'))))
  }
  return out
}
