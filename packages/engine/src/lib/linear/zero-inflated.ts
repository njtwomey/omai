/**
 * Models for counts with too many zeros. A zero-inflated model mixes a point mass at zero (probability `pi`, a logistic
 * regression) with a count distribution (Poisson or negative binomial with log mean `x beta`), so a zero can come from
 * either part; a hurdle model has `pi` as the probability of any zero and a zero-truncated count distribution for the
 * positive counts, so the two parts do not share zeros (statsmodels' `ZeroInflatedPoisson`, `ZeroInflatedNegativeBinomialP`
 * and `HurdleCountModel`, R's `pscl`). `countModelAlgorithm` maximises the likelihood by damped Newton steps (analytic
 * gradient, Hessian by central differences); `zeroInflated` and `hurdle` are the estimators on the iterative base, with
 * the counts part, the inflation part and Wald tables of both.
 */

import { MdmlError } from '../base/index.ts'
import type { CoefficientTable, Exposition } from '../exposition/index.ts'
import {
  type IterativeEstimator,
  iterative,
  type LearnData,
  matrixView,
  sampleWeightsOf,
  targetVector,
  withDefaultScore,
} from '../learn/index.ts'
import { dgemv, dpotrf, dpotrs } from '../numerics/linalg/index.ts'
import { digamma, lgamma } from '../numerics/special/index.ts'
import { dict, real } from '../space/index.ts'
import { fromStorage, item, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { backtrack, newtonDirection } from './_newton.ts'
import { waldTable } from './inference.ts'

/** The count distribution of the non-degenerate part. */
export type CountFamily = 'poisson' | 'negative-binomial'

/** Whether zeros come from a mixture (zero-inflated) or from a separate hurdle. */
export type CountModelKind = 'zero-inflated' | 'hurdle'

/** Which features drive the zero part: only a constant, all features, or the listed feature indices. */
export type Inflation = 'constant' | 'all' | readonly number[]

/** A count problem: features, counts, weights, and the features of the zero part. */
export interface CountProblem {
  readonly n: number
  /** Count-part design `[n, p]` with the intercept last. */
  readonly x: Float64Array
  readonly p: number
  /** Zero-part design `[n, q]` with the intercept last. */
  readonly z: Float64Array
  readonly q: number
  readonly y: Float64Array
  readonly w: Float64Array
  /** The feature indices (not counting the intercept) that enter the zero part. */
  readonly zeroFeatures: readonly number[]
}

/** Read a count problem. */
export function countProblemOf(data: LearnData, inflation: Inflation, who: string): CountProblem {
  const m = matrixView(data.x, who)
  const y = targetVector(data.y ?? [], who, m.n)
  for (const v of y)
    if (!(v >= 0) || !Number.isInteger(v)) throw new MdmlError('value', `${who}: counts must be non-negative integers`)
  const zeroFeatures =
    inflation === 'constant' ? [] : inflation === 'all' ? Array.from({ length: m.d }, (_, j) => j) : [...inflation]
  for (const j of zeroFeatures)
    if (!Number.isInteger(j) || j < 0 || j >= m.d)
      throw new MdmlError('value', `${who}: inflation feature ${j} is not a column`)
  const p = m.d + 1
  const q = zeroFeatures.length + 1
  const x = new Float64Array(m.n * p)
  const z = new Float64Array(m.n * q)
  for (let i = 0; i < m.n; i++) {
    for (let j = 0; j < m.d; j++) x[i * p + j] = m.data[i * m.d + j] ?? 0
    x[i * p + m.d] = 1
    zeroFeatures.forEach((j, k) => {
      z[i * q + k] = m.data[i * m.d + j] ?? 0
    })
    z[i * q + q - 1] = 1
  }
  return { n: m.n, x, p, z, q, y, w: sampleWeightsOf(data.weights, m.n, who), zeroFeatures }
}

/** A count density at one observation: its log, its zero probability, and their derivatives in the predictors. */
interface CountTerms {
  readonly logf: number
  readonly dEta: number
  readonly dLnAlpha: number
  readonly logf0: number
  readonly dEta0: number
  readonly dLnAlpha0: number
}

const lg = (x: number): number => item(lgamma(x))
const dg = (x: number): number => item(digamma(x))

function countTerms(family: CountFamily, y: number, eta: number, lnAlpha: number): CountTerms {
  const mu = Math.exp(Math.min(eta, 700))
  if (family === 'poisson')
    return { logf: y * eta - mu - lg(y + 1), dEta: y - mu, dLnAlpha: 0, logf0: -mu, dEta0: -mu, dLnAlpha0: 0 }
  // Negative binomial with variance mu + alpha mu^2: r = 1 / alpha.
  const r = Math.exp(-lnAlpha)
  const logRatio = Math.log(r / (r + mu))
  const logf = lg(y + r) - lg(r) - lg(y + 1) + r * logRatio + y * Math.log(mu / (r + mu))
  return {
    logf,
    dEta: (r * (y - mu)) / (r + mu),
    dLnAlpha: -r * (dg(y + r) - dg(r) + logRatio + (mu - y) / (r + mu)),
    logf0: r * logRatio,
    dEta0: (-r * mu) / (r + mu),
    dLnAlpha0: -r * (logRatio + mu / (r + mu)),
  }
}

const logistic = (v: number): number => (v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v)))

/** The negative log-likelihood of a count model and its gradient at parameters `[beta (p), gamma (q), lnAlpha?]`. */
export function countModelObjective(
  problem: CountProblem,
  kind: CountModelKind,
  family: CountFamily,
  theta: ArrayLike<number>,
): { readonly objective: number; readonly gradient: Float64Array } {
  const { n, p, q, y, w } = problem
  const beta = Array.from({ length: p }, (_, j) => theta[j] ?? 0)
  const gamma = Array.from({ length: q }, (_, j) => theta[p + j] ?? 0)
  const lnAlpha = family === 'negative-binomial' ? (theta[p + q] ?? 0) : 0
  const etaMu = dgemv(problem.x, n, p, beta)
  const etaZ = dgemv(problem.z, n, q, gamma)
  const dMu = new Float64Array(n)
  const dZ = new Float64Array(n)
  let dAlpha = 0
  let ll = 0
  for (let i = 0; i < n; i++) {
    const wi = w[i] ?? 0
    const yi = y[i] ?? 0
    const pi = logistic(etaZ[i] ?? 0)
    const t = countTerms(family, yi, etaMu[i] ?? 0, lnAlpha)
    let li: number
    let gMu: number
    let gZ: number
    let gA: number
    if (kind === 'zero-inflated') {
      if (yi === 0) {
        const f0 = Math.exp(t.logf0)
        const p0 = pi + (1 - pi) * f0
        li = Math.log(Math.max(p0, 1e-300))
        gZ = (pi * (1 - pi) * (1 - f0)) / p0
        gMu = ((1 - pi) * f0 * t.dEta0) / p0
        gA = ((1 - pi) * f0 * t.dLnAlpha0) / p0
      } else {
        li = Math.log(Math.max(1 - pi, 1e-300)) + t.logf
        gZ = -pi
        gMu = t.dEta
        gA = t.dLnAlpha
      }
    } else if (yi === 0) {
      li = Math.log(Math.max(pi, 1e-300))
      gZ = 1 - pi
      gMu = 0
      gA = 0
    } else {
      // Zero-truncated count: log f(y) - log(1 - f(0)).
      const f0 = Math.exp(t.logf0)
      const oneMinus = Math.max(-Math.expm1(t.logf0), 1e-300)
      li = Math.log(Math.max(1 - pi, 1e-300)) + t.logf - Math.log(oneMinus)
      gZ = -pi
      gMu = t.dEta + (f0 * t.dEta0) / oneMinus
      gA = t.dLnAlpha + (f0 * t.dLnAlpha0) / oneMinus
    }
    ll += wi * li
    dMu[i] = wi * gMu
    dZ[i] = wi * gZ
    dAlpha += wi * gA
  }
  const gx = dgemv(problem.x, n, p, dMu, true)
  const gz = dgemv(problem.z, n, q, dZ, true)
  const gradient = new Float64Array(theta.length)
  for (let j = 0; j < p; j++) gradient[j] = -(gx[j] ?? 0)
  for (let j = 0; j < q; j++) gradient[p + j] = -(gz[j] ?? 0)
  if (family === 'negative-binomial') gradient[p + q] = -dAlpha
  return { objective: -ll, gradient }
}

/** Options of {@link countModelAlgorithm}. */
export interface CountModelOptions {
  readonly kind?: CountModelKind
  readonly family?: CountFamily
  /** Stop when the largest gradient entry (per unit weight) falls below this (default 1e-8). */
  readonly tol?: number
}

/** The state of {@link countModelAlgorithm}. */
export type CountModelState = {
  /** The count coefficients (intercept last), the zero-part coefficients (intercept last), then ln alpha (negative binomial). */
  readonly theta: number[]
  /** The negative log-likelihood. */
  readonly objective: number
  readonly gradient: number
  readonly evaluations: number
  readonly stalled: boolean
}

/**
 * Zero-inflated or hurdle count regression as a traced algorithm: damped Newton steps on the negative log-likelihood
 * (analytic gradient, Hessian by central differences of it). It logs `objective`, `gradient` and the tensor
 * `parameters`.
 *
 * @param problem - The data.
 * @param options - Zero-inflated or hurdle, Poisson or negative binomial, and the tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { countModelAlgorithm, countProblemOf } from 'mdml-engine/linear'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0.1], [0.4], [0.9], [1.3], [1.8], [2.2], [2.6], [3.1], [0.2], [1.1], [2.9], [1.6]]
 * const y = [0, 0, 1, 0, 3, 2, 0, 5, 0, 2, 6, 0]
 * const problem = countProblemOf({ x, y }, 'constant', 'demo')
 * runAlgorithm(countModelAlgorithm(problem), { maxSteps: 100 }).status // => 'converged'
 * ```
 */
export function countModelAlgorithm(
  problem: CountProblem,
  options: CountModelOptions = {},
): Algorithm<CountModelState> {
  const kind = options.kind ?? 'zero-inflated'
  const family = options.family ?? 'poisson'
  const tol = options.tol ?? 1e-8
  const { p, q } = problem
  const m = p + q + (family === 'negative-binomial' ? 1 : 0)
  const total = problem.w.reduce((a, b) => a + b, 0)
  const evaluate = (theta: number[]): { objective: number; grad: Float64Array } => {
    const e = countModelObjective(problem, kind, family, theta)
    return { objective: e.objective / total, grad: Float64Array.from(e.gradient, (g) => g / total) }
  }
  const maxAbs = (v: ArrayLike<number>): number => Array.from(v).reduce((a, b) => Math.max(a, Math.abs(b)), 0)
  const start = (): number[] => {
    const positives = Array.from(problem.y).filter((v) => v > 0)
    const meanPositive = positives.length > 0 ? positives.reduce((a, b) => a + b, 0) / positives.length : 1
    const share = Math.min(0.9, Math.max(0.1, 1 - positives.length / problem.n))
    const theta = new Array<number>(m).fill(0)
    theta[p - 1] = Math.log(Math.max(meanPositive, 0.1))
    theta[p + q - 1] = Math.log(share / (1 - share)) - (kind === 'zero-inflated' ? 1 : 0)
    return theta
  }
  return {
    name: `${kind}-${family}`,
    config: { kind, family, tol },
    schema: channels({
      metrics: {
        objective: { description: 'the negative log-likelihood per unit weight', direction: 'lower' },
        gradient: { description: 'the largest gradient entry', direction: 'lower' },
      },
      tensors: { parameters: { role: 'param', description: 'count, zero-part and dispersion parameters', shape: [m] } },
    }),
    init: () => {
      const theta = start()
      const e = evaluate(theta)
      return { theta, objective: e.objective, gradient: maxAbs(e.grad), evaluations: 1, stalled: false }
    },
    step: (s) => {
      const g = evaluate(s.theta).grad
      const H = new Float64Array(m * m)
      for (let j = 0; j < m; j++) {
        const h = 1e-5 * Math.max(1, Math.abs(s.theta[j] ?? 0))
        const up = [...s.theta]
        const dn = [...s.theta]
        up[j] = (up[j] ?? 0) + h
        dn[j] = (dn[j] ?? 0) - h
        const gu = evaluate(up).grad
        const gd = evaluate(dn).grad
        for (let i = 0; i < m; i++) H[i * m + j] = ((gu[i] ?? 0) - (gd[i] ?? 0)) / (2 * h)
      }
      for (let i = 0; i < m; i++)
        for (let j = 0; j < i; j++) {
          const v = 0.5 * ((H[i * m + j] ?? 0) + (H[j * m + i] ?? 0))
          H[i * m + j] = v
          H[j * m + i] = v
        }
      const dir = newtonDirection(H, g, m)
      const found = backtrack(evaluate, s.theta, s.objective, g, dir)
      const evaluations = s.evaluations + 2 * m + 1 + (found?.evaluations ?? 50)
      if (found === undefined) return { ...s, evaluations, stalled: true }
      return {
        theta: found.u,
        objective: found.result.objective,
        gradient: maxAbs(found.result.grad),
        evaluations,
        stalled: false,
      }
    },
    done: (s) => (s.gradient <= tol ? 'converged' : s.stalled ? 'stopped' : undefined),
    observe: (s) => ({
      metrics: { objective: s.objective, gradient: s.gradient },
      tensors: { parameters: { role: 'param', value: s.theta, shape: [m] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CountModelState,
  }
}

/** The hyperparameters of {@link zeroInflated} and {@link hurdle}. */
export interface CountModelHyper {
  readonly family: CountFamily
  /** Which features drive the zero part (default only a constant, as statsmodels). */
  readonly inflation: Inflation
  readonly tol: number
  readonly maxIter: number
}

/** The fitted parameters of the count models. */
export type CountModelParams = {
  readonly kind: CountModelKind
  readonly family: CountFamily
  /** The count-part slopes. */
  readonly coef: number[]
  readonly intercept: number
  /** The zero-part slopes, for the features in `zeroFeatures`. */
  readonly zeroCoef: number[]
  readonly zeroIntercept: number
  readonly zeroFeatures: number[]
  /** The negative binomial dispersion alpha (zero for Poisson). */
  readonly alpha: number
  readonly logLikelihood: number
  /** The covariance of all parameters (inverse observed information), `[m][m]`. */
  readonly covariance: number[][]
  readonly iterations: number
  readonly converged: boolean
}

/** What a fitted count model offers. */
export interface CountModelVerbs {
  readonly nFeaturesIn: number
  /** The mean count. */
  readonly predict: (x: TensorLike) => Tensor
  /** The probability of the zero part (a structural zero for zero-inflated, any zero for a hurdle). */
  readonly predictInflation: (x: TensorLike) => Tensor
  /** The mean `mu` of the count distribution before zero-inflation or truncation. */
  readonly predictCount: (x: TensorLike) => Tensor
  /** The probability of observing a zero. */
  readonly predictZero: (x: TensorLike) => Tensor
  readonly featureImportances: number[]
  readonly score: (data: LearnData) => number
}

/** Parameters and information from the final state. */
function paramsOf(
  problem: CountProblem,
  h: CountModelHyper,
  kind: CountModelKind,
  s: CountModelState,
): CountModelParams {
  const { p, q } = problem
  const m = s.theta.length
  const grad = (t: ArrayLike<number>): Float64Array => countModelObjective(problem, kind, h.family, t).gradient
  const info = new Float64Array(m * m)
  for (let j = 0; j < m; j++) {
    const step = 1e-5 * Math.max(1, Math.abs(s.theta[j] ?? 0))
    const up = [...s.theta]
    const dn = [...s.theta]
    up[j] = (up[j] ?? 0) + step
    dn[j] = (dn[j] ?? 0) - step
    const gu = grad(up)
    const gd = grad(dn)
    for (let i = 0; i < m; i++) info[i * m + j] = ((gu[i] ?? 0) - (gd[i] ?? 0)) / (2 * step)
  }
  for (let i = 0; i < m; i++)
    for (let j = 0; j < i; j++) {
      const v = 0.5 * ((info[i * m + j] ?? 0) + (info[j * m + i] ?? 0))
      info[i * m + j] = v
      info[j * m + i] = v
    }
  const chol = dpotrf(info, m)
  const covariance = Array.from({ length: m }, () => new Array<number>(m).fill(Number.NaN))
  if (chol.info === 0)
    for (let c = 0; c < m; c++) {
      const e = new Float64Array(m)
      e[c] = 1
      const col = dpotrs(chol.factor, e, m)
      for (let r = 0; r < m; r++) (covariance[r] as number[])[c] = col[r] ?? 0
    }
  return {
    kind,
    family: h.family,
    coef: s.theta.slice(0, p - 1),
    intercept: s.theta[p - 1] ?? 0,
    zeroCoef: s.theta.slice(p, p + q - 1),
    zeroIntercept: s.theta[p + q - 1] ?? 0,
    zeroFeatures: [...problem.zeroFeatures],
    alpha: h.family === 'negative-binomial' ? Math.exp(s.theta[p + q] ?? 0) : 0,
    logLikelihood: -countModelObjective(problem, kind, h.family, s.theta).objective,
    covariance,
    iterations: s.evaluations,
    converged: s.gradient <= h.tol,
  }
}

function tablesOf(p: CountModelParams): CoefficientTable[] {
  const d = p.coef.length
  const m = d + 1 + p.zeroFeatures.length + 1 + (p.family === 'negative-binomial' ? 1 : 0)
  const cov = Float64Array.from(
    { length: m * m },
    (_, idx) => p.covariance[Math.floor(idx / m)]?.[idx % m] ?? Number.NaN,
  )
  const names = [
    ...Array.from({ length: d }, (_, j) => `count:x${j}`),
    'count:intercept',
    ...p.zeroFeatures.map((j) => `zero:x${j}`),
    'zero:intercept',
    ...(p.family === 'negative-binomial' ? ['ln alpha'] : []),
  ]
  const estimates = [
    ...p.coef,
    p.intercept,
    ...p.zeroCoef,
    p.zeroIntercept,
    ...(p.family === 'negative-binomial' ? [Math.log(p.alpha)] : []),
  ]
  return [waldTable(names, estimates, cov)]
}

const COUNT_DEFAULTS: CountModelHyper = { family: 'poisson', inflation: 'constant', tol: 1e-8, maxIter: 200 }

function countEstimator(
  kind: CountModelKind,
  hyper: Partial<CountModelHyper>,
): IterativeEstimator<CountModelHyper, CountModelState, CountModelParams, CountModelVerbs> {
  const who = kind === 'hurdle' ? 'hurdle' : 'zeroInflated'
  return iterative<CountModelHyper, CountModelState, CountModelParams, CountModelVerbs>(
    {
      name: kind,
      kind: 'regressor',
      defaults: COUNT_DEFAULTS,
      hyperSpace: (h) => dict({ tol: real(1e-12, 1e-3, { log: true, default: h.tol }) }),
      algorithm: (data, h) =>
        countModelAlgorithm(countProblemOf(data, h.inflation, who), { kind, family: h.family, tol: h.tol }),
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => paramsOf(countProblemOf(data, h.inflation, who), h, kind, s),
      verbs: (par) => {
        const d = par.coef.length
        const parts = (x: TensorLike): { mu: Float64Array; pi: Float64Array } => {
          const m = matrixView(x, who, { nFeatures: d })
          const mu = new Float64Array(m.n)
          const pi = new Float64Array(m.n)
          for (let i = 0; i < m.n; i++) {
            let eta = par.intercept
            for (let j = 0; j < d; j++) eta += (m.data[i * d + j] ?? 0) * (par.coef[j] ?? 0)
            let ez = par.zeroIntercept
            par.zeroFeatures.forEach((j, k) => {
              ez += (m.data[i * d + j] ?? 0) * (par.zeroCoef[k] ?? 0)
            })
            mu[i] = Math.exp(eta)
            pi[i] = logistic(ez)
          }
          return { mu, pi }
        }
        const f0 = (mu: number): number =>
          par.family === 'poisson' ? Math.exp(-mu) : (1 + par.alpha * mu) ** (-1 / par.alpha)
        const vec = (a: Float64Array): Tensor => fromStorage(a, [a.length], 'float64')
        const base = {
          nFeaturesIn: d,
          predict: (x: TensorLike): Tensor => {
            const { mu, pi } = parts(x)
            return vec(
              Float64Array.from(mu, (v, i) =>
                par.kind === 'zero-inflated' ? (1 - (pi[i] ?? 0)) * v : ((1 - (pi[i] ?? 0)) * v) / (1 - f0(v)),
              ),
            )
          },
          predictInflation: (x: TensorLike): Tensor => vec(parts(x).pi),
          predictCount: (x: TensorLike): Tensor => vec(parts(x).mu),
          predictZero: (x: TensorLike): Tensor => {
            const { mu, pi } = parts(x)
            return vec(
              Float64Array.from(mu, (v, i) =>
                par.kind === 'zero-inflated' ? (pi[i] ?? 0) + (1 - (pi[i] ?? 0)) * f0(v) : (pi[i] ?? 0),
              ),
            )
          },
          featureImportances: par.coef.map(Math.abs),
        }
        return { ...base, score: withDefaultScore('regressor', base).score ?? ((): number => Number.NaN) }
      },
      records: (s, data, h): readonly Exposition[] =>
        tablesOf(paramsOf(countProblemOf(data, h.inflation, who), h, kind, s)),
      logs: { 'solver/objective': Log(), 'solver/gradient': Log(), 'solver/parameters': Log({ detail: 'full' }) },
    },
    { ...COUNT_DEFAULTS, ...hyper },
  )
}

/**
 * The zero-inflated count regression (statsmodels' `ZeroInflatedPoisson` and `ZeroInflatedNegativeBinomialP`): with
 * probability `pi = logistic(z gamma)` a count is a structural zero, otherwise it follows a Poisson or negative binomial
 * (variance `mu + alpha mu^2`) with log mean `x beta`. By default `pi` depends on a constant only; `inflation: 'all'`
 * or a list of feature indices adds regressors. `predict` is the mean `(1 - pi) mu`; `predictInflation`,
 * `predictCount` and `predictZero` expose the parts.
 *
 * @param hyper - `family`, `inflation`, `tol`, `maxIter`.
 * @returns The estimator.
 * @reference D. Lambert, Zero-inflated Poisson regression, with an application to defects in manufacturing,
 * Technometrics 34 (1992).
 * @example
 * ```ts
 * import { zeroInflated } from 'mdml-engine/linear'
 * const x = [[0.1], [0.4], [0.9], [1.3], [1.8], [2.2], [2.6], [3.1], [0.2], [1.1], [2.9], [1.6]]
 * const y = [0, 0, 1, 0, 3, 2, 0, 5, 0, 2, 6, 0]
 * const f = zeroInflated().fit({ x, y })
 * (f.params.coef[0] ?? 0) > 0 // => true
 * ```
 */
export function zeroInflated(
  hyper: Partial<CountModelHyper> = {},
): IterativeEstimator<CountModelHyper, CountModelState, CountModelParams, CountModelVerbs> {
  return countEstimator('zero-inflated', hyper)
}

/**
 * The hurdle count regression (statsmodels' `HurdleCountModel`, R's `pscl::hurdle`): `P(y = 0) = pi` from a logistic
 * regression on whether the count is zero, and the positive counts follow a zero-truncated Poisson or negative binomial
 * with log mean `x beta`. The two parts have no parameters in common, so zeros cannot come from the count part.
 *
 * @param hyper - `family`, `inflation` (the features of the zero part), `tol`, `maxIter`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { hurdle } from 'mdml-engine/linear'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0.1], [0.4], [0.9], [1.3], [1.8], [2.2], [2.6], [3.1], [0.2], [1.1], [2.9], [1.6]]
 * const y = [0, 0, 1, 0, 3, 2, 0, 5, 0, 2, 6, 0]
 * const f = hurdle().fit({ x, y })
 * (toFlat(f.predictZero([[1]]))[0] ?? 1) < 1 // => true
 * ```
 */
export function hurdle(
  hyper: Partial<CountModelHyper> = {},
): IterativeEstimator<CountModelHyper, CountModelState, CountModelParams, CountModelVerbs> {
  return countEstimator('hurdle', hyper)
}
