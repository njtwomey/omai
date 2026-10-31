/**
 * Instrumental variables: the k-class family of estimators of a linear structural equation with endogenous regressors
 * and excluded instruments (ordinary least squares at kappa 0, two-stage least squares at kappa 1, limited-information
 * maximum likelihood at the smallest eigenvalue of the variance-ratio problem, Fuller's modification), with classical
 * or heteroscedasticity-robust errors, optional weights, the first-stage F statistic and partial R-squared of each
 * endogenous regressor (the weak-instrument diagnostic), Sargan's test of the overidentifying restrictions, and the
 * Wald estimate for a single instrument.
 */

import { MdmlError } from '../base/index.ts'
import { matrixView } from '../learn/validation.ts'
import { dgemm, dpotrf, dsyev } from '../numerics/linalg/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { asTensor, toFlat } from '../tensor/index.ts'
import { chi2Sf, fSf, invert } from './_dense.ts'
import { type EffectEstimate, effectFromInfluence, effectFromSe } from './data.ts'

/** The data of an instrumental-variable regression: `y = X beta + u` with some columns of `X` correlated with `u`. */
export interface IvData {
  /** The outcome `[n]`. */
  readonly outcome: TensorLike
  /** The endogenous regressors `[n]` or `[n, k]` (the treatment). */
  readonly endogenous: TensorLike
  /** The excluded instruments `[n]` or `[n, m]`, with `m >= k`. */
  readonly instruments: TensorLike
  /** Included exogenous controls `[n, c]` (also instruments of themselves). */
  readonly exogenous?: TensorLike
}

/** Options of {@link instrumentalVariables}. */
export interface IvOptions {
  /** `'2sls'` (default), `'liml'`, `'fuller'` (LIML with Fuller's correction), `'ols'`, or an explicit k-class `kappa` below. */
  readonly estimator?: '2sls' | 'liml' | 'fuller' | 'ols'
  /** Fuller's alpha (default 1) for `estimator: 'fuller'`. */
  readonly fuller?: number
  /** An explicit k-class parameter overriding `estimator`. */
  readonly kappa?: number
  /** Include an intercept (default true). */
  readonly intercept?: boolean
  /** The covariance of the estimates (default `'nonrobust'`). */
  readonly covariance?: 'nonrobust' | 'HC0' | 'HC1'
  /** Observation weights (weighted two-stage least squares). */
  readonly weights?: ArrayLike<number>
  /** The coverage of the interval of the first endogenous coefficient (default 0.95). */
  readonly level?: number
}

/** The strength of the instruments for one endogenous regressor. */
export interface FirstStage {
  /** The F statistic of the excluded instruments in the regression of the regressor on all instruments and controls. */
  readonly fStatistic: number
  readonly pValue: number
  /** The share of the regressor's variance (after the controls) the excluded instruments explain. */
  readonly partialR2: number
  /** Coefficients of the excluded instruments. */
  readonly coefficients: readonly number[]
}

/** The result of {@link instrumentalVariables}. */
export interface IvResult {
  /** Coefficient names: `endogenous0..`, `exogenous0..`, then `intercept`. */
  readonly names: readonly string[]
  readonly coefficients: readonly number[]
  readonly standardErrors: readonly number[]
  /** The covariance of the coefficients, row-major. */
  readonly covariance: readonly number[]
  /** The first endogenous coefficient as a local average treatment effect. */
  readonly effect: EffectEstimate
  /** The k-class parameter used. */
  readonly kappa: number
  /** One first stage per endogenous regressor. */
  readonly firstStage: readonly FirstStage[]
  /** Sargan's test, when there are more instruments than endogenous regressors. */
  readonly sargan?: { readonly statistic: number; readonly pValue: number; readonly df: number }
  readonly residuals: Float64Array
  readonly n: number
}

interface Mat {
  readonly data: Float64Array
  readonly n: number
  readonly k: number
}

function asMatrix(x: TensorLike | undefined, n: number, who: string): Mat {
  if (x === undefined) return { data: new Float64Array(0), n, k: 0 }
  const t = asTensor(x)
  if (t.shape.length === 1) {
    const flat = Float64Array.from(toFlat(t))
    if (flat.length !== n) throw new MdmlError('shape', `${who}: ${flat.length} values for ${n} observations`)
    return { data: flat, n, k: 1 }
  }
  const v = matrixView(x, who)
  if (v.n !== n) throw new MdmlError('shape', `${who}: ${v.n} rows for ${n} observations`)
  return { data: v.data, n, k: v.d }
}

function hcat(parts: readonly Mat[], n: number): Mat {
  const k = parts.reduce((s, p) => s + p.k, 0)
  const out = new Float64Array(n * k)
  let c0 = 0
  for (const p of parts) {
    for (let i = 0; i < n; i++) for (let j = 0; j < p.k; j++) out[i * k + c0 + j] = p.data[i * p.k + j] ?? 0
    c0 += p.k
  }
  return { data: out, n, k }
}

/** The residuals of the columns of `a` regressed on the columns of `w` (all zero columns when `w` is empty: the columns themselves). */
function residualise(a: Mat, w: Mat): Mat {
  if (w.k === 0) return a
  const gram = dgemm(w.data, w.data, w.k, w.n, w.k, { transposeA: true })
  const rhs = dgemm(w.data, a.data, w.k, w.n, a.k, { transposeA: true })
  const coef = dgemm(invert(gram, w.k), rhs, w.k, w.k, a.k)
  const fit = dgemm(w.data, coef, w.n, w.k, a.k)
  return { data: Float64Array.from(a.data, (v, i) => v - (fit[i] ?? 0)), n: a.n, k: a.k }
}

const column = (a: Mat, j: number): Float64Array =>
  Float64Array.from({ length: a.n }, (_, i) => a.data[i * a.k + j] ?? 0)

const sumSquares = (a: Mat, j: number): number => {
  let s = 0
  for (let i = 0; i < a.n; i++) s += (a.data[i * a.k + j] ?? 0) ** 2
  return s
}

/** The smallest eigenvalue of `A^-1 B` for symmetric positive-definite `A` and symmetric `B` (order `p`). */
function smallestGeneralisedEigenvalue(a: Float64Array, b: Float64Array, p: number): number {
  const { factor, info } = dpotrf(a, p)
  if (info !== 0) throw new MdmlError('value', 'liml: the instruments leave no variation in the regressors')
  // C = U^-T B U^-1 for A = U^T U, symmetric with the same eigenvalues as A^-1 B.
  const left = new Float64Array(p * p)
  for (let col = 0; col < p; col++)
    for (let row = 0; row < p; row++) {
      let s = b[row * p + col] ?? 0
      for (let k = 0; k < row; k++) s -= (factor[k * p + row] ?? 0) * (left[k * p + col] ?? 0)
      left[row * p + col] = s / (factor[row * p + row] ?? 1)
    }
  const c = new Float64Array(p * p)
  for (let row = 0; row < p; row++)
    for (let col = 0; col < p; col++) {
      let s = left[row * p + col] ?? 0
      for (let k = 0; k < col; k++) s -= (c[row * p + k] ?? 0) * (factor[k * p + col] ?? 0)
      c[row * p + col] = s / (factor[col * p + col] ?? 1)
    }
  const sym = Float64Array.from(c, (_, i) => 0.5 * ((c[i] ?? 0) + (c[(i % p) * p + Math.floor(i / p)] ?? 0)))
  return dsyev(sym, p).w[0] ?? 1
}

/**
 * Estimate a linear structural equation with instruments by a k-class estimator: ordinary least squares, two-stage
 * least squares (the default), limited-information maximum likelihood (LIML, which is median-unbiased and has better
 * behaviour with weak or many instruments) or Fuller's modified LIML. Reports the first-stage F statistic and partial
 * R-squared of every endogenous regressor, Sargan's test when overidentified, and the first endogenous coefficient as
 * the effect of the treatment.
 *
 * @param data - Outcome, endogenous regressors, excluded instruments and included controls.
 * @param options - Estimator, intercept, covariance type and weights.
 * @returns Coefficients, errors, first stages and tests.
 * @example
 * ```ts
 * import { instrumentalVariables } from 'mdml-engine/causal'
 * const z = [0, 1, 2, 3, 4, 5, 6, 7]
 * const d = [0.1, 1.2, 1.9, 3.2, 3.9, 5.2, 5.9, 7.1]
 * const y = d.map((v) => 2 * v + 1)
 * instrumentalVariables({ outcome: y, endogenous: d, instruments: z }).coefficients[0] // => 2
 * ```
 */
export function instrumentalVariables(data: IvData, options: IvOptions = {}): IvResult {
  const yt = asTensor(data.outcome)
  const n = yt.shape[0] ?? 0
  const sw = options.weights === undefined ? undefined : Float64Array.from(options.weights, Math.sqrt)
  const scale = (m: Mat): Mat =>
    sw === undefined
      ? m
      : { ...m, data: Float64Array.from(m.data, (v, i) => v * (sw[Math.floor(i / Math.max(1, m.k))] ?? 0)) }
  const intercept = options.intercept ?? true
  const ones: Mat = intercept
    ? { data: Float64Array.from({ length: n }, () => 1), n, k: 1 }
    : { data: new Float64Array(0), n, k: 0 }
  const endog = scale(asMatrix(data.endogenous, n, 'instrumentalVariables'))
  const z = scale(asMatrix(data.instruments, n, 'instrumentalVariables'))
  const exog = scale(asMatrix(data.exogenous, n, 'instrumentalVariables'))
  const icpt = scale(ones)
  const yMat = scale({ data: Float64Array.from(toFlat(yt)), n, k: 1 })
  const k = endog.k
  if (z.k < k)
    throw new MdmlError(
      'value',
      `instrumentalVariables: ${k} endogenous regressors need at least ${k} instruments, got ${z.k}`,
    )
  const x = hcat([endog, exog, icpt], n)
  const included = hcat([exog, icpt], n)
  const w = hcat([included, z], n)
  const p = x.k
  // k-class parameter.
  let kappa: number
  if (options.kappa !== undefined) kappa = options.kappa
  else if (options.estimator === 'ols') kappa = 0
  else if (options.estimator === 'liml' || options.estimator === 'fuller') {
    const yy = hcat([yMat, endog], n)
    const mw = residualise(yy, w)
    const m1 = residualise(yy, included)
    const q = yy.k
    const a = dgemm(mw.data, mw.data, q, n, q, { transposeA: true })
    const b = dgemm(m1.data, m1.data, q, n, q, { transposeA: true })
    kappa = smallestGeneralisedEigenvalue(a, b, q)
    if (options.estimator === 'fuller') kappa -= (options.fuller ?? 1) / (n - w.k)
  } else kappa = 1
  // X~ = (1 - kappa) X + kappa P_W X.
  const mwx = residualise(x, w)
  const xt = { ...x, data: Float64Array.from(x.data, (v, i) => v - kappa * (mwx.data[i] ?? 0)) }
  const left = dgemm(xt.data, x.data, p, n, p, { transposeA: true })
  const rhs = dgemm(xt.data, yMat.data, p, n, 1, { transposeA: true })
  const bread = invert(left, p)
  const beta = dgemm(bread, rhs, p, p, 1)
  const fitted = dgemm(x.data, beta, n, p, 1)
  const residuals = Float64Array.from(yMat.data, (v, i) => v - (fitted[i] ?? 0))
  const kind = options.covariance ?? 'nonrobust'
  let cov: Float64Array
  if (kind === 'nonrobust') {
    const s2 = residuals.reduce((s, r) => s + r * r, 0) / (n - p)
    cov = Float64Array.from(bread, (v) => v * s2)
  } else {
    // Sandwich with X~: B (sum u^2 x~ x~') B'.
    const meat = new Float64Array(p * p)
    for (let i = 0; i < n; i++)
      for (let a = 0; a < p; a++)
        for (let b = 0; b < p; b++)
          meat[a * p + b] =
            (meat[a * p + b] ?? 0) + (residuals[i] ?? 0) ** 2 * (xt.data[i * p + a] ?? 0) * (xt.data[i * p + b] ?? 0)
    const half = dgemm(bread, meat, p, p, p)
    const sand = dgemm(half, bread, p, p, p, { transposeB: true })
    const factor = kind === 'HC1' ? n / (n - p) : 1
    cov = Float64Array.from(sand, (v) => v * factor)
  }
  const se = Array.from({ length: p }, (_, j) => Math.sqrt(cov[j * p + j] ?? Number.NaN))
  // First stages.
  const restricted = included
  const firstStage: FirstStage[] = Array.from({ length: k }, (_, j) => {
    const target: Mat = { data: column(endog, j), n, k: 1 }
    const full = residualise(target, w)
    const part = residualise(target, restricted)
    const rssU = sumSquares(full, 0)
    const rssR = sumSquares(part, 0)
    const m = z.k
    const df = n - w.k
    const f = (rssR - rssU) / m / (rssU / df)
    // The coefficients of the excluded instruments in the full first stage.
    const coefAll = dgemm(
      invert(dgemm(w.data, w.data, w.k, n, w.k, { transposeA: true }), w.k),
      dgemm(w.data, target.data, w.k, n, 1, { transposeA: true }),
      w.k,
      w.k,
      1,
    )
    return {
      fStatistic: f,
      pValue: fSf(m, df, f),
      partialR2: (rssR - rssU) / rssR,
      coefficients: Array.from(coefAll.subarray(included.k)),
    }
  })
  let sargan: IvResult['sargan']
  if (z.k > k) {
    const u: Mat = { data: residuals, n, k: 1 }
    const ru = residualise(u, w)
    const ssu = sumSquares(u, 0)
    const stat = (n * (ssu - sumSquares(ru, 0))) / ssu
    sargan = { statistic: stat, pValue: chi2Sf(z.k - k, stat), df: z.k - k }
  }
  const names = [
    ...Array.from({ length: k }, (_, j) => `endogenous${j}`),
    ...Array.from({ length: exog.k }, (_, j) => `exogenous${j}`),
    ...(intercept ? ['intercept'] : []),
  ]
  const effect = effectFromSe('late', beta[0] ?? Number.NaN, se[0] ?? Number.NaN, n, {
    ...(options.level === undefined ? {} : { level: options.level }),
    df: Math.max(1, n - p),
    method: `${options.estimator ?? '2sls'} instrumental variables${kind === 'nonrobust' ? '' : ` (${kind})`}`,
  })
  return {
    names,
    coefficients: Array.from(beta),
    standardErrors: se,
    covariance: Array.from(cov),
    effect,
    kappa,
    firstStage,
    ...(sargan === undefined ? {} : { sargan }),
    residuals,
    n,
  }
}

/**
 * Two-stage least squares, the k-class estimator with kappa 1: regress the endogenous regressors on the instruments and
 * controls, then the outcome on the fitted values.
 *
 * @param data - Outcome, endogenous regressors, instruments and controls.
 * @param options - Intercept, covariance type and weights.
 * @returns Coefficients, errors, first stages and tests.
 * @example
 * ```ts
 * import { twoStageLeastSquares } from 'mdml-engine/causal'
 * const r = twoStageLeastSquares({ outcome: [1, 3, 5, 7, 9, 11], endogenous: [0, 1, 2, 3, 4, 5], instruments: [0.1, 0.9, 2.1, 2.9, 4.1, 4.9] })
 * r.firstStage[0].fStatistic > 100 // => true
 * ```
 */
export function twoStageLeastSquares(
  data: IvData,
  options: Omit<IvOptions, 'estimator' | 'kappa' | 'fuller'> = {},
): IvResult {
  return instrumentalVariables(data, { ...options, estimator: '2sls' })
}

/**
 * Limited-information maximum likelihood (Anderson and Rubin): the k-class estimator at the smallest root of the
 * variance-ratio problem. It equals two-stage least squares when the equation is exactly identified, and is far less
 * biased than it when the instruments are weak. Fuller's modification (`fuller`) has finite moments.
 *
 * @param data - Outcome, endogenous regressors, instruments and controls.
 * @param options - Intercept, covariance type, weights and Fuller's alpha (`fuller`, 0 for plain LIML).
 * @returns Coefficients, errors, first stages and tests; `kappa` is the root.
 * @example
 * ```ts
 * import { liml } from 'mdml-engine/causal'
 * const r = liml({ outcome: [1, 3.4, 4.7, 7.3, 8.6, 11.2], endogenous: [0, 1, 2, 3, 4, 5], instruments: [0.1, 0.9, 2.2, 2.9, 4.1, 5.2] })
 * Math.round(r.kappa * 1e8) / 1e8 // => 1
 * ```
 */
export function liml(data: IvData, options: Omit<IvOptions, 'estimator' | 'kappa'> = {}): IvResult {
  return instrumentalVariables(data, {
    ...options,
    estimator: options.fuller !== undefined && options.fuller !== 0 ? 'fuller' : 'liml',
  })
}

/**
 * The first-stage F statistic of the excluded instruments for the first endogenous regressor: the weak-instrument
 * diagnostic (values below about 10 mean the two-stage estimate is unreliable).
 *
 * @param data - Outcome, endogenous regressors, instruments and controls.
 * @returns The F statistic.
 * @example
 * ```ts
 * import { firstStageF } from 'mdml-engine/causal'
 * firstStageF({ outcome: [1, 2, 3, 4, 5, 6], endogenous: [1, 2, 2, 4, 5, 7], instruments: [1, 2, 3, 4, 5, 6] }) > 50 // => true
 * ```
 */
export function firstStageF(data: IvData): number {
  return instrumentalVariables(data).firstStage[0]?.fStatistic ?? Number.NaN
}

/**
 * The Wald estimate with one instrument: the ratio of the instrument's covariance with the outcome to its covariance
 * with the treatment (for a binary instrument, the difference in mean outcomes over the difference in mean treatment).
 * The standard error is the influence-function one, valid with heteroscedasticity.
 *
 * @param outcome - The outcome `[n]`.
 * @param treatment - The treatment `[n]`.
 * @param instrument - The instrument `[n]`.
 * @returns The local average treatment effect with its influence values.
 * @example
 * ```ts
 * import { waldEstimate } from 'mdml-engine/causal'
 * waldEstimate([1, 1, 3, 3], [0, 0, 1, 1], [0, 0, 1, 1]).estimate.value // => 2
 * ```
 */
export function waldEstimate(outcome: TensorLike, treatment: TensorLike, instrument: TensorLike): EffectEstimate {
  const y = toFlat(asTensor(outcome))
  const d = toFlat(asTensor(treatment))
  const z = toFlat(asTensor(instrument))
  const n = y.length
  const mean = (v: readonly number[]): number => v.reduce((s, b) => s + b, 0) / n
  const [my, md, mz] = [mean(y), mean(d), mean(z)]
  let cyz = 0
  let cdz = 0
  for (let i = 0; i < n; i++) {
    cyz += ((z[i] ?? 0) - mz) * ((y[i] ?? 0) - my)
    cdz += ((z[i] ?? 0) - mz) * ((d[i] ?? 0) - md)
  }
  const beta = cyz / cdz
  const influence = Float64Array.from(
    y,
    (yi, i) => (n * ((z[i] ?? 0) - mz) * (yi - my - beta * ((d[i] ?? 0) - md))) / cdz,
  )
  return { ...effectFromInfluence('late', beta, influence, { method: 'Wald estimate' }), n }
}
