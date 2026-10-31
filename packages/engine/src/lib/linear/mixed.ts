/**
 * Linear mixed-effects models for grouped data, `y = X beta + Z b_g + e` with group effects `b_g ~ N(0, Psi)` and
 * `e ~ N(0, sigma^2 I)` (statsmodels' `MixedLM`, R's `lme4::lmer`). The fixed effects and the residual variance are
 * profiled out of the (restricted) likelihood, which then depends only on the relative covariance factor `L` with
 * `Psi = sigma^2 L L'`; everything is computed from per-group sufficient statistics, so a likelihood evaluation costs
 * a few small Cholesky factorisations. `mixedAlgorithm` maximises it by damped Newton steps; `mixedModel` is the
 * estimator on the iterative base, with the fixed-effects table, the group effects (BLUPs) and group-aware prediction.
 */

import { MdmlError } from '../base/index.ts'
import type { CoefficientTable, Exposition } from '../exposition/index.ts'
import {
  type IterativeEstimator,
  iterative,
  type LearnData,
  type MatrixView,
  matrixView,
  targetVector,
  withDefaultScore,
} from '../learn/index.ts'
import { dpotrf, dpotrs } from '../numerics/linalg/index.ts'
import { bool, dict, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { backtrack, newtonDirection } from './_newton.ts'
import { waldTable } from './inference.ts'

/** The structure of the random-effects covariance. */
export type MixedCovariance = 'unstructured' | 'diagonal'

/** Per-group sufficient statistics: all that the likelihood needs of a group's rows. */
interface GroupStats {
  readonly n: number
  readonly xtx: Float64Array
  readonly ztx: Float64Array
  readonly ztz: Float64Array
  readonly xty: Float64Array
  readonly zty: Float64Array
  readonly yty: number
}

/** A mixed-model problem: the fixed and random designs summarised per group. */
export interface MixedProblem {
  readonly groups: readonly GroupStats[]
  /** The distinct group labels, in the order of `groups`. */
  readonly labels: readonly number[]
  /** Fixed-effects columns (the intercept, when present, is last). */
  readonly p: number
  /** Random-effects columns. */
  readonly q: number
  readonly nObs: number
}

/** How a model reads its data: which column holds the group, which features get random slopes. */
export interface MixedLayout {
  /** The column of `x` holding the group label (default 0); the other columns are the fixed-effect features. */
  readonly groupColumn: number
  readonly fitIntercept: boolean
  readonly randomIntercept: boolean
  /** Feature indices (among the non-group columns) with a random slope. */
  readonly randomSlopes: readonly number[]
}

/** The rows of `x` split into group labels and features. */
function splitGroups(x: MatrixView, groupColumn: number, who: string): { labels: number[]; features: MatrixView } {
  if (!Number.isInteger(groupColumn) || groupColumn < 0 || groupColumn >= x.d)
    throw new MdmlError('value', `${who}: groupColumn ${groupColumn} is not a column of x`)
  const d = x.d - 1
  const features = new Float64Array(x.n * d)
  const labels: number[] = []
  for (let i = 0; i < x.n; i++) {
    labels.push(x.data[i * x.d + groupColumn] ?? 0)
    let c = 0
    for (let j = 0; j < x.d; j++) if (j !== groupColumn) features[i * d + c++] = x.data[i * x.d + j] ?? 0
  }
  return { labels, features: { n: x.n, d, data: features } }
}

/**
 * Summarise grouped data per group.
 *
 * @param data - `x` with a group column and the features, and `y`.
 * @param layout - Where the group column is, the intercepts, and the random slopes.
 * @param who - The caller, for errors.
 * @returns The problem.
 * @throws MdmlError `value` for sample weights (not supported), a missing random effect, or too few rows.
 * @example
 * ```ts
 * import { mixedProblemOf } from 'mdml-engine/linear'
 * const layout = { groupColumn: 0, fitIntercept: true, randomIntercept: true, randomSlopes: [] }
 * const p = mixedProblemOf({ x: [[1, 0], [1, 1], [2, 0], [2, 1]], y: [1, 2, 3, 4] }, layout, 'demo')
 * p.labels // => [1, 2]
 * ```
 */
export function mixedProblemOf(data: LearnData, layout: MixedLayout, who: string): MixedProblem {
  if (data.weights !== undefined) throw new MdmlError('value', `${who}: sample weights are not supported`)
  const x = matrixView(data.x, who)
  const y = targetVector(data.y ?? [], who, x.n)
  const { labels, features } = splitGroups(x, layout.groupColumn, who)
  const d = features.d
  for (const j of layout.randomSlopes)
    if (!Number.isInteger(j) || j < 0 || j >= d)
      throw new MdmlError('value', `${who}: random slope ${j} is not a feature`)
  const p = d + (layout.fitIntercept ? 1 : 0)
  const q = (layout.randomIntercept ? 1 : 0) + layout.randomSlopes.length
  if (q === 0) throw new MdmlError('value', `${who}: needs at least one random effect`)
  const distinct = [...new Set(labels)].sort((a, b) => a - b)
  if (x.n <= p) throw new MdmlError('value', `${who}: needs more rows than fixed effects`)
  const index = new Map(distinct.map((l, i) => [l, i]))
  const acc = distinct.map(() => ({
    n: 0,
    xtx: new Float64Array(p * p),
    ztx: new Float64Array(q * p),
    ztz: new Float64Array(q * q),
    xty: new Float64Array(p),
    zty: new Float64Array(q),
    yty: 0,
  }))
  const xr = new Float64Array(p)
  const zr = new Float64Array(q)
  for (let i = 0; i < x.n; i++) {
    const g = acc[index.get(labels[i] ?? 0) ?? 0]
    if (g === undefined) continue
    for (let j = 0; j < d; j++) xr[j] = features.data[i * d + j] ?? 0
    if (layout.fitIntercept) xr[d] = 1
    let c = 0
    if (layout.randomIntercept) zr[c++] = 1
    for (const j of layout.randomSlopes) zr[c++] = features.data[i * d + j] ?? 0
    const yi = y[i] ?? 0
    g.n++
    g.yty += yi * yi
    for (let a = 0; a < p; a++) {
      g.xty[a] = (g.xty[a] ?? 0) + (xr[a] ?? 0) * yi
      for (let b = 0; b < p; b++) g.xtx[a * p + b] = (g.xtx[a * p + b] ?? 0) + (xr[a] ?? 0) * (xr[b] ?? 0)
    }
    for (let a = 0; a < q; a++) {
      g.zty[a] = (g.zty[a] ?? 0) + (zr[a] ?? 0) * yi
      for (let b = 0; b < p; b++) g.ztx[a * p + b] = (g.ztx[a * p + b] ?? 0) + (zr[a] ?? 0) * (xr[b] ?? 0)
      for (let b = 0; b < q; b++) g.ztz[a * q + b] = (g.ztz[a * q + b] ?? 0) + (zr[a] ?? 0) * (zr[b] ?? 0)
    }
  }
  return { groups: acc, labels: distinct, p, q, nObs: x.n }
}

/** The relative covariance factor `L` (lower triangular, row-major `[q, q]`) of a parameter vector. */
export function covarianceFactor(theta: ArrayLike<number>, q: number, structure: MixedCovariance): Float64Array {
  const L = new Float64Array(q * q)
  let k = 0
  for (let i = 0; i < q; i++) {
    if (structure === 'diagonal') L[i * q + i] = theta[k++] ?? 0
    else for (let j = 0; j <= i; j++) L[i * q + j] = theta[k++] ?? 0
  }
  return L
}

/** The number of covariance parameters. */
function thetaLength(q: number, structure: MixedCovariance): number {
  return structure === 'diagonal' ? q : (q * (q + 1)) / 2
}

/** What the profiled likelihood gives at a covariance factor. */
export interface MixedProfile {
  /** Minus twice the (restricted) log-likelihood. */
  readonly deviance: number
  /** The generalised least squares fixed effects. */
  readonly beta: Float64Array
  /** The residual variance estimate. */
  readonly sigma2: number
  /** `X' M X` summed over groups, row-major `[p, p]`: the fixed effects' covariance is `sigma2` times its inverse. */
  readonly information: Float64Array
}

const LOG_2PI = Math.log(2 * Math.PI)

/** Solve `A x = b` for several right-hand sides with a Cholesky factor `u` of `A = u'u`. */
function choleskySolveColumns(u: Float64Array, b: Float64Array, n: number, cols: number): Float64Array {
  const out = new Float64Array(n * cols)
  for (let c = 0; c < cols; c++) {
    const rhs = Float64Array.from({ length: n }, (_, i) => b[i * cols + c] ?? 0)
    const x = dpotrs(u, rhs, n)
    for (let i = 0; i < n; i++) out[i * cols + c] = x[i] ?? 0
  }
  return out
}

/**
 * The profiled (restricted) deviance of a mixed model at a relative covariance factor, with the fixed effects and
 * residual variance that go with it. With `A_g = I + L' Z_g'Z_g L`, the quadratic forms of the marginal precision are
 * `u'Mv = u'v - (Z'u)' L A_g^-1 L' (Z'v)` and `log|V_g / sigma^2| = log|A_g|`.
 *
 * @param problem - The summarised data.
 * @param L - The relative covariance factor `[q, q]`.
 * @param reml - Restricted (default true) or ordinary maximum likelihood.
 * @returns The profile, or `undefined` when a matrix is not positive definite.
 * @example
 * ```ts
 * import { covarianceFactor, mixedProblemOf, mixedProfile } from 'mdml-engine/linear'
 * const layout = { groupColumn: 0, fitIntercept: true, randomIntercept: true, randomSlopes: [] }
 * const x = [[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]]
 * const p = mixedProblemOf({ x, y: [1, 2.1, 2.9, 3, 4.2, 4.8, 0.8, 2, 3.2] }, layout, 'demo')
 * mixedProfile(p, covarianceFactor([1], 1, 'unstructured'))?.beta.length // => 2
 * ```
 */
export function mixedProfile(problem: MixedProblem, L: Float64Array, reml = true): MixedProfile | undefined {
  const { p, q, nObs } = problem
  const xmx = new Float64Array(p * p)
  const xmy = new Float64Array(p)
  let ymy = 0
  let logDetA = 0
  for (const g of problem.groups) {
    // A = I + L' ZtZ L, T = L' ZtX, t = L' Zty.
    const ztzL = new Float64Array(q * q)
    for (let a = 0; a < q; a++)
      for (let b = 0; b < q; b++) {
        let s = 0
        for (let c = 0; c < q; c++) s += (g.ztz[a * q + c] ?? 0) * (L[c * q + b] ?? 0)
        ztzL[a * q + b] = s
      }
    const A = new Float64Array(q * q)
    for (let a = 0; a < q; a++)
      for (let b = 0; b < q; b++) {
        let s = a === b ? 1 : 0
        for (let c = 0; c < q; c++) s += (L[c * q + a] ?? 0) * (ztzL[c * q + b] ?? 0)
        A[a * q + b] = s
      }
    const T = new Float64Array(q * (p + 1))
    for (let a = 0; a < q; a++) {
      for (let b = 0; b < p; b++) {
        let s = 0
        for (let c = 0; c < q; c++) s += (L[c * q + a] ?? 0) * (g.ztx[c * p + b] ?? 0)
        T[a * (p + 1) + b] = s
      }
      let s = 0
      for (let c = 0; c < q; c++) s += (L[c * q + a] ?? 0) * (g.zty[c] ?? 0)
      T[a * (p + 1) + p] = s
    }
    const { factor, info } = dpotrf(A, q)
    if (info !== 0) return undefined
    for (let a = 0; a < q; a++) logDetA += 2 * Math.log(factor[a * q + a] ?? 1)
    const S = choleskySolveColumns(factor, T, q, p + 1)
    for (let a = 0; a < p; a++) {
      for (let b = 0; b < p; b++) {
        let s = 0
        for (let c = 0; c < q; c++) s += (T[c * (p + 1) + a] ?? 0) * (S[c * (p + 1) + b] ?? 0)
        xmx[a * p + b] = (xmx[a * p + b] ?? 0) + (g.xtx[a * p + b] ?? 0) - s
      }
      let s = 0
      for (let c = 0; c < q; c++) s += (T[c * (p + 1) + a] ?? 0) * (S[c * (p + 1) + p] ?? 0)
      xmy[a] = (xmy[a] ?? 0) + (g.xty[a] ?? 0) - s
    }
    let s = 0
    for (let c = 0; c < q; c++) s += (T[c * (p + 1) + p] ?? 0) * (S[c * (p + 1) + p] ?? 0)
    ymy += g.yty - s
  }
  const chol = dpotrf(xmx, p)
  if (chol.info !== 0) return undefined
  const beta = dpotrs(chol.factor, xmy, p)
  let rss = ymy
  for (let a = 0; a < p; a++) rss -= (beta[a] ?? 0) * (xmy[a] ?? 0)
  if (!(rss > 0)) return undefined
  let logDetX = 0
  for (let a = 0; a < p; a++) logDetX += 2 * Math.log(chol.factor[a * p + a] ?? 1)
  const dof = reml ? nObs - p : nObs
  const sigma2 = rss / dof
  const deviance = dof * (LOG_2PI + Math.log(sigma2) + 1) + logDetA + (reml ? logDetX : 0)
  return { deviance, beta, sigma2, information: xmx }
}

/** Options of {@link mixedAlgorithm}. */
export interface MixedOptions {
  readonly reml?: boolean
  readonly covariance?: MixedCovariance
  /** Stop when the largest gradient entry of the deviance per observation falls below this (default 1e-8). */
  readonly tol?: number
}

/** The state of {@link mixedAlgorithm}. */
export type MixedState = {
  /** The covariance factor parameters (the lower triangle of `L` by rows, or its diagonal). */
  readonly theta: number[]
  /** The deviance (minus twice the log-likelihood). */
  readonly deviance: number
  readonly gradient: number
  readonly evaluations: number
  readonly stalled: boolean
}

/**
 * Maximum (restricted) likelihood for a mixed model as a traced algorithm: damped Newton steps on the profiled
 * deviance over the relative covariance factor, with the gradient and Hessian by central differences. It logs
 * `deviance`, `gradient`, `sigma2` and the tensor `theta`.
 *
 * @param problem - The summarised data.
 * @param options - REML or ML, the covariance structure and the tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { mixedAlgorithm, mixedProblemOf } from 'mdml-engine/linear'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const layout = { groupColumn: 0, fitIntercept: true, randomIntercept: true, randomSlopes: [] }
 * const x = [[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]]
 * const p = mixedProblemOf({ x, y: [1, 2.1, 2.9, 3, 4.2, 4.8, 0.8, 2, 3.2] }, layout, 'demo')
 * runAlgorithm(mixedAlgorithm(p), { maxSteps: 50 }).status // => 'converged'
 * ```
 */
export function mixedAlgorithm(problem: MixedProblem, options: MixedOptions = {}): Algorithm<MixedState> {
  const reml = options.reml ?? true
  const structure = options.covariance ?? 'unstructured'
  const tol = options.tol ?? 1e-8
  const { q, nObs } = problem
  const m = thetaLength(q, structure)
  const f = (theta: ArrayLike<number>): number => {
    const prof = mixedProfile(problem, covarianceFactor(theta, q, structure), reml)
    return prof === undefined ? Number.POSITIVE_INFINITY : prof.deviance / nObs
  }
  const maxAbs = (v: ArrayLike<number>): number => {
    let s = 0
    for (let i = 0; i < v.length; i++) s = Math.max(s, Math.abs(v[i] ?? 0))
    return s
  }
  const gradientAt = (u: readonly number[]): Float64Array =>
    Float64Array.from(u, (v, j) => {
      const h = 1e-6 * Math.max(1, Math.abs(v))
      const up = [...u]
      const dn = [...u]
      up[j] = v + h
      dn[j] = v - h
      return (f(up) - f(dn)) / (2 * h)
    })
  const start = (): number[] => {
    const theta: number[] = []
    for (let i = 0; i < q; i++)
      if (structure === 'diagonal') theta.push(1)
      else for (let j = 0; j <= i; j++) theta.push(i === j ? 1 : 0)
    return theta
  }
  return {
    name: `mixed-${reml ? 'reml' : 'ml'}`,
    config: { reml, covariance: structure, tol },
    schema: channels({
      metrics: {
        deviance: { description: 'minus twice the (restricted) log-likelihood', direction: 'lower' },
        gradient: { description: 'the largest gradient entry of the deviance per observation', direction: 'lower' },
        sigma2: { description: 'the residual variance at the current covariance' },
      },
      tensors: { theta: { role: 'param', description: 'the relative covariance factor parameters', shape: [m] } },
    }),
    init: () => {
      const theta = start()
      return {
        theta,
        deviance: f(theta) * nObs,
        gradient: maxAbs(gradientAt(theta)),
        evaluations: 1 + 2 * m,
        stalled: false,
      }
    },
    step: (s) => {
      const u = s.theta
      const g = gradientAt(u)
      const H = new Float64Array(m * m)
      const f0 = f(u)
      const hs = u.map((v) => 1e-3 * Math.max(1, Math.abs(v)))
      const shifted = (changes: readonly [number, number][]): number => {
        const t = [...u]
        for (const [j, delta] of changes) t[j] = (t[j] ?? 0) + delta
        return f(t)
      }
      for (let i = 0; i < m; i++) {
        const hi = hs[i] ?? 1e-3
        H[i * m + i] = (shifted([[i, hi]]) - 2 * f0 + shifted([[i, -hi]])) / (hi * hi)
        for (let j = 0; j < i; j++) {
          const hj = hs[j] ?? 1e-3
          const v =
            (shifted([
              [i, hi],
              [j, hj],
            ]) -
              shifted([
                [i, hi],
                [j, -hj],
              ]) -
              shifted([
                [i, -hi],
                [j, hj],
              ]) +
              shifted([
                [i, -hi],
                [j, -hj],
              ])) /
            (4 * hi * hj)
          H[i * m + j] = v
          H[j * m + i] = v
        }
      }
      const dir = newtonDirection(H, g, m)
      const found = backtrack((t) => ({ objective: f(t) }), u, f0, g, dir)
      const evaluations = s.evaluations + 2 * m + 1 + 2 * m * m + 3 * m + (found?.evaluations ?? 50)
      if (found === undefined) return { ...s, evaluations, stalled: true }
      return {
        theta: found.u,
        deviance: found.result.objective * nObs,
        gradient: maxAbs(gradientAt(found.u)),
        evaluations,
        stalled: false,
      }
    },
    done: (s) => (s.gradient <= tol ? 'converged' : s.stalled ? 'stopped' : undefined),
    observe: (s) => ({
      metrics: {
        deviance: s.deviance,
        gradient: s.gradient,
        sigma2: mixedProfile(problem, covarianceFactor(s.theta, q, structure), reml)?.sigma2 ?? Number.NaN,
      },
      tensors: { theta: { role: 'param', value: s.theta, shape: [m] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MixedState,
  }
}

/** The hyperparameters of {@link mixedModel}. */
export interface MixedHyper {
  /** The column of `x` holding the group label (default 0). */
  readonly groupColumn: number
  /** Restricted maximum likelihood (default) or maximum likelihood. */
  readonly reml: boolean
  readonly fitIntercept: boolean
  readonly randomIntercept: boolean
  /** Feature indices (among the non-group columns) with a random slope. */
  readonly randomSlopes: readonly number[]
  readonly covariance: MixedCovariance
  readonly tol: number
  readonly maxIter: number
}

/** The fitted parameters of {@link mixedModel}. */
export type MixedParams = {
  /** Fixed-effect slopes of the non-group columns. */
  readonly coef: number[]
  readonly intercept: number
  /** The residual variance. */
  readonly sigma2: number
  /** The covariance `Psi` of the random effects `[q][q]`. */
  readonly randomCovariance: number[][]
  /** The names of the random effects (`intercept` and the slope features). */
  readonly randomNames: string[]
  /** The group labels seen in fit, ascending. */
  readonly groupLabels: number[]
  /** The best linear unbiased predictions of each group's effects `[groups][q]`. */
  readonly randomEffects: number[][]
  /** The covariance of the fixed effects (slopes, then the intercept) `[p][p]`. */
  readonly fixedCovariance: number[][]
  /** The (restricted) log-likelihood. */
  readonly logLikelihood: number
  readonly aic: number
  readonly bic: number
  readonly theta: number[]
  readonly reml: boolean
  readonly iterations: number
  readonly converged: boolean
}

/** What a fitted mixed model offers. */
export interface MixedVerbs {
  readonly nFeaturesIn: number
  /** Predictions: the fixed effects plus the group's effects for groups seen in fit (the group column of `x` is read). */
  readonly predict: (x: TensorLike) => Tensor
  /** Population-level predictions from the fixed effects alone. */
  readonly predictFixed: (x: TensorLike) => Tensor
  readonly featureImportances: number[]
  readonly score: (data: LearnData) => number
}

const MIXED_DEFAULTS: MixedHyper = {
  groupColumn: 0,
  reml: true,
  fitIntercept: true,
  randomIntercept: true,
  randomSlopes: [],
  covariance: 'unstructured',
  tol: 1e-8,
  maxIter: 200,
}

const layoutOf = (h: MixedHyper): MixedLayout => ({
  groupColumn: h.groupColumn,
  fitIntercept: h.fitIntercept,
  randomIntercept: h.randomIntercept,
  randomSlopes: h.randomSlopes,
})

/** The fit behind the parameters: the profile at the final covariance and the group effects. */
function paramsOf(problem: MixedProblem, h: MixedHyper, s: MixedState, featureNames: readonly string[]): MixedParams {
  const { p, q, nObs } = problem
  const L = covarianceFactor(s.theta, q, h.covariance)
  const prof = mixedProfile(problem, L, h.reml)
  if (prof === undefined) throw new MdmlError('value', 'mixedModel: the fit ended at a singular covariance')
  const d = p - (h.fitIntercept ? 1 : 0)
  const rows = (a: ArrayLike<number>, n: number, c: number): number[][] =>
    Array.from({ length: n }, (_, i) => Array.from({ length: c }, (__, j) => a[i * c + j] ?? 0))
  // Psi = sigma^2 L L'.
  const psi = new Float64Array(q * q)
  for (let a = 0; a < q; a++)
    for (let b = 0; b < q; b++) {
      let t = 0
      for (let c = 0; c < q; c++) t += (L[a * q + c] ?? 0) * (L[b * q + c] ?? 0)
      psi[a * q + b] = prof.sigma2 * t
    }
  // The covariance of the fixed effects: sigma^2 (X'MX)^-1.
  const chol = dpotrf(prof.information, p)
  const fixedCov = new Float64Array(p * p)
  for (let c = 0; c < p; c++) {
    const e = new Float64Array(p)
    e[c] = 1
    const col = dpotrs(chol.factor, e, p)
    for (let r = 0; r < p; r++) fixedCov[r * p + c] = prof.sigma2 * (col[r] ?? 0)
  }
  // BLUPs: b_g = L A_g^-1 L' (Z'y - Z'X beta).
  const effects: number[][] = []
  for (const g of problem.groups) {
    const r = new Float64Array(q)
    for (let a = 0; a < q; a++) {
      let t = g.zty[a] ?? 0
      for (let b = 0; b < p; b++) t -= (g.ztx[a * p + b] ?? 0) * (prof.beta[b] ?? 0)
      r[a] = t
    }
    const lr = new Float64Array(q)
    for (let a = 0; a < q; a++) for (let c = 0; c < q; c++) lr[a] = (lr[a] ?? 0) + (L[c * q + a] ?? 0) * (r[c] ?? 0)
    const A = new Float64Array(q * q)
    for (let a = 0; a < q; a++)
      for (let b = 0; b < q; b++) {
        let t = a === b ? 1 : 0
        for (let c = 0; c < q; c++) {
          let zl = 0
          for (let e = 0; e < q; e++) zl += (g.ztz[c * q + e] ?? 0) * (L[e * q + b] ?? 0)
          t += (L[c * q + a] ?? 0) * zl
        }
        A[a * q + b] = t
      }
    const fac = dpotrf(A, q)
    const w = dpotrs(fac.factor, lr, q)
    const b = new Array<number>(q).fill(0)
    for (let a = 0; a < q; a++) for (let c = 0; c < q; c++) b[a] = (b[a] ?? 0) + (L[a * q + c] ?? 0) * (w[c] ?? 0)
    effects.push(b)
  }
  const kParams = p + thetaLength(q, h.covariance) + 1
  const logLikelihood = -prof.deviance / 2
  const names = [...(h.randomIntercept ? ['intercept'] : []), ...h.randomSlopes.map((j) => featureNames[j] ?? `x${j}`)]
  return {
    coef: Array.from(prof.beta.slice(0, d)),
    intercept: h.fitIntercept ? (prof.beta[d] ?? 0) : 0,
    sigma2: prof.sigma2,
    randomCovariance: rows(psi, q, q),
    randomNames: names,
    groupLabels: [...problem.labels],
    randomEffects: effects,
    fixedCovariance: rows(fixedCov, p, p),
    logLikelihood,
    aic: -2 * logLikelihood + 2 * kParams,
    bic: -2 * logLikelihood + kParams * Math.log(nObs),
    theta: s.theta,
    reml: h.reml,
    iterations: s.evaluations,
    converged: s.gradient <= h.tol,
  }
}

function fixedTable(p: MixedParams, h: MixedHyper, names: readonly string[]): CoefficientTable {
  const all = [...names, ...(h.fitIntercept ? ['intercept'] : [])]
  const estimates = [...p.coef, ...(h.fitIntercept ? [p.intercept] : [])]
  const m = estimates.length
  const cov = Float64Array.from(
    { length: m * m },
    (_, idx) => p.fixedCovariance[Math.floor(idx / m)]?.[idx % m] ?? Number.NaN,
  )
  return waldTable(all, estimates, cov)
}

/**
 * The linear mixed-effects model estimator: `y = X beta + Z b_g + e` for data in groups. The group label is a column of
 * `x` (`groupColumn`, default the first) and the other columns are the fixed-effect features; every group gets a random
 * intercept and, for the features in `randomSlopes`, random slopes, with an unstructured (or diagonal) covariance.
 * Fitted by restricted maximum likelihood (or ML) with damped Newton steps (session: `solver/deviance`,
 * `solver/gradient`, `solver/sigma2`, `solver/theta`). `predict` adds the group effects for groups seen in fit;
 * `params` has the fixed effects and their covariance, `randomCovariance`, `sigma2` and the group effects; the
 * exposition holds the Wald table of the fixed effects.
 *
 * @param hyper - `groupColumn`, `reml`, `fitIntercept`, `randomIntercept`, `randomSlopes`, `covariance`, `tol`, `maxIter`.
 * @returns The estimator.
 * @reference N. Laird and J. Ware, Random-effects models for longitudinal data, Biometrics 38 (1982); D. Bates et al.,
 * Fitting linear mixed-effects models using lme4, Journal of Statistical Software 67 (2015).
 * @example
 * ```ts
 * import { mixedModel } from 'mdml-engine/linear'
 * const x = [[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [4, 0], [4, 1], [4, 2]]
 * const y = [1, 2.1, 2.9, 3, 4.2, 4.8, 0.8, 2, 3.2, 2.1, 2.9, 4.2]
 * const f = mixedModel().fit({ x, y })
 * f.params.groupLabels // => [1, 2, 3, 4]
 * f.params.randomEffects.length // => 4
 * ```
 */
export function mixedModel(
  hyper: Partial<MixedHyper> = {},
): IterativeEstimator<MixedHyper, MixedState, MixedParams, MixedVerbs> {
  const namesOf = (data: LearnData, h: MixedHyper): string[] => {
    const x = matrixView(data.x, 'mixedModel')
    return Array.from({ length: x.d - 1 }, (_, j) => `x${j}`).map((n, j) => (j >= h.groupColumn ? `x${j + 1}` : n))
  }
  return iterative<MixedHyper, MixedState, MixedParams, MixedVerbs>(
    {
      name: 'mixed-model',
      kind: 'regressor',
      defaults: MIXED_DEFAULTS,
      hyperSpace: (h) =>
        dict({ reml: bool({ default: h.reml }), tol: real(1e-12, 1e-3, { log: true, default: h.tol }) }),
      algorithm: (data, h) =>
        mixedAlgorithm(mixedProblemOf(data, layoutOf(h), 'mixedModel'), {
          reml: h.reml,
          covariance: h.covariance,
          tol: h.tol,
        }),
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => paramsOf(mixedProblemOf(data, layoutOf(h), 'mixedModel'), h, s, namesOf(data, h)),
      verbs: (p, h) => {
        const index = new Map(p.groupLabels.map((l, i) => [l, i]))
        const d = p.coef.length
        const fixed = (x: TensorLike): { n: number; values: Float64Array; labels: number[]; feats: Float64Array } => {
          const m = matrixView(x, 'mixedModel', { nFeatures: d + 1 })
          const { labels, features } = splitGroups(m, h.groupColumn, 'mixedModel')
          const values = new Float64Array(m.n)
          for (let i = 0; i < m.n; i++) {
            let t = p.intercept
            for (let j = 0; j < d; j++) t += (features.data[i * d + j] ?? 0) * (p.coef[j] ?? 0)
            values[i] = t
          }
          return { n: m.n, values, labels, feats: features.data }
        }
        const predictFixed = (x: TensorLike): Tensor => {
          const f = fixed(x)
          return fromStorage(f.values, [f.n], 'float64')
        }
        const predict = (x: TensorLike): Tensor => {
          const f = fixed(x)
          const out = Float64Array.from(f.values)
          for (let i = 0; i < f.n; i++) {
            const g = index.get(f.labels[i] ?? Number.NaN)
            const b = g === undefined ? undefined : p.randomEffects[g]
            if (b === undefined) continue
            let c = 0
            if (h.randomIntercept) out[i] = (out[i] ?? 0) + (b[c++] ?? 0)
            for (const j of h.randomSlopes) out[i] = (out[i] ?? 0) + (f.feats[i * d + j] ?? 0) * (b[c++] ?? 0)
          }
          return fromStorage(out, [f.n], 'float64')
        }
        const base = { nFeaturesIn: d + 1, predict, predictFixed, featureImportances: p.coef.map(Math.abs) }
        return { ...base, score: withDefaultScore('regressor', base).score ?? ((): number => Number.NaN) }
      },
      records: (s, data, h): readonly Exposition[] => {
        const names = namesOf(data, h)
        const p = paramsOf(mixedProblemOf(data, layoutOf(h), 'mixedModel'), h, s, names)
        return [fixedTable(p, h, names)]
      },
      logs: {
        'solver/deviance': Log(),
        'solver/gradient': Log(),
        'solver/sigma2': Log(),
        'solver/theta': Log({ detail: 'full' }),
      },
    },
    { ...MIXED_DEFAULTS, ...hyper },
  )
}
