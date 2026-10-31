/**
 * Ordinal regression for ordered categories: the cumulative-link family (proportional odds with the logit link, the
 * ordered probit, and the complementary log-log and log-log links), where `P(Y <= k | x) = F(theta_k - x beta)` with
 * increasing thresholds, and the sequential (continuation-ratio) model `P(Y = k | Y >= k, x) = F(theta_k - x beta)`.
 * `ordinalLogLikelihood` is the likelihood with its gradient; `ordinalAlgorithm` maximises it by damped Newton steps
 * (the thresholds parameterised as a first cutpoint and log increments, as statsmodels' `OrderedModel`); `ordinal` is
 * the classifier on the iterative base with class probabilities, the latent score and a Wald table.
 */

import { MdmlError } from '../base/index.ts'
import { categorical } from '../dists/index.ts'
import type { CoefficientTable, Exposition } from '../exposition/index.ts'
import {
  encodeLabels,
  featureNamesOf,
  type IterativeEstimator,
  iterative,
  type LearnData,
  type MatrixView,
  matrixView,
  sampleWeightsOf,
  withDefaultScore,
} from '../learn/index.ts'
import { dgemv, dpotrf, dpotrs } from '../numerics/linalg/index.ts'
import { ndtr, ndtri } from '../numerics/special/index.ts'
import { dict, real } from '../space/index.ts'
import { fromStorage, item, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { waldTable } from './inference.ts'

/** The link F of an ordinal model. */
export type OrdinalLink = 'logit' | 'probit' | 'cloglog' | 'loglog'

/** The structure of the category probabilities: cumulative (proportional odds) or sequential (continuation ratio). */
export type OrdinalMode = 'cumulative' | 'sequential'

interface LinkFunctions {
  readonly cdf: (z: number) => number
  readonly sf: (z: number) => number
  readonly pdf: (z: number) => number
  readonly inverse: (p: number) => number
}

const stableLogistic = (z: number): number => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)))

const LINKS: Readonly<Record<OrdinalLink, LinkFunctions>> = {
  logit: {
    cdf: stableLogistic,
    sf: (z) => stableLogistic(-z),
    pdf: (z) => stableLogistic(z) * stableLogistic(-z),
    inverse: (p) => Math.log(p / (1 - p)),
  },
  probit: {
    cdf: (z) => item(ndtr(z)),
    sf: (z) => item(ndtr(-z)),
    pdf: (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI),
    inverse: (p) => item(ndtri(p)),
  },
  cloglog: {
    cdf: (z) => -Math.expm1(-Math.exp(z)),
    sf: (z) => Math.exp(-Math.exp(z)),
    pdf: (z) => Math.exp(z - Math.exp(z)),
    inverse: (p) => Math.log(-Math.log(1 - p)),
  },
  loglog: {
    cdf: (z) => Math.exp(-Math.exp(-z)),
    sf: (z) => -Math.expm1(-Math.exp(-z)),
    pdf: (z) => Math.exp(-z - Math.exp(-z)),
    inverse: (p) => -Math.log(-Math.log(p)),
  },
}

const TINY = 1e-300

/** An ordinal problem: the features, the class index of each row (0 to K - 1) and the weights. */
export interface OrdinalProblem {
  readonly x: MatrixView
  readonly codes: ArrayLike<number>
  readonly w: ArrayLike<number>
  /** The number of categories K. */
  readonly nClasses: number
}

/** The log-likelihood of an ordinal model and its gradient with respect to the slopes and the thresholds. */
export interface OrdinalLikelihood {
  readonly logLikelihood: number
  readonly gradCoef: Float64Array
  readonly gradThresholds: Float64Array
}

/**
 * The weighted log-likelihood of an ordinal model and its gradient. Cumulative: `P(Y = k) = F(theta_k - eta) -
 * F(theta_{k-1} - eta)` with `eta = x beta` (the outer thresholds are infinite); sequential: `P(Y = k) = F(a_k)
 * prod_{j<k} (1 - F(a_j))` with `a_j = theta_j - eta` (the last category has no `F` factor).
 *
 * @param problem - Features, class indices and weights.
 * @param coef - The slopes `beta`.
 * @param thresholds - The K - 1 thresholds `theta` (increasing for the cumulative model).
 * @param link - The link F (default logit).
 * @param mode - Cumulative or sequential (default cumulative).
 * @returns The log-likelihood and its gradient.
 * @example
 * ```ts
 * import { ordinalLogLikelihood } from 'mdml-engine/linear'
 * import { matrixView } from 'mdml-engine/learn'
 * const problem = { x: matrixView([[0], [1]], 'demo'), codes: [0, 1], w: [1, 1], nClasses: 2 }
 * ordinalLogLikelihood(problem, [0], [0]).logLikelihood // => -1.3862943611198906
 * ```
 */
export function ordinalLogLikelihood(
  problem: OrdinalProblem,
  coef: ArrayLike<number>,
  thresholds: ArrayLike<number>,
  link: OrdinalLink = 'logit',
  mode: OrdinalMode = 'cumulative',
): OrdinalLikelihood {
  const { x, codes, w, nClasses: K } = problem
  const F = LINKS[link]
  const eta = dgemv(x.data, x.n, x.d, coef)
  const gradCoef = new Float64Array(x.d)
  const gradThresholds = new Float64Array(K - 1)
  let ll = 0
  const dA = new Float64Array(K - 1)
  for (let i = 0; i < x.n; i++) {
    const y = codes[i] ?? 0
    const wi = w[i] ?? 0
    if (wi === 0) continue
    dA.fill(0)
    const a = (k: number): number => (thresholds[k] ?? 0) - (eta[i] ?? 0)
    if (mode === 'cumulative') {
      const hasUpper = y < K - 1
      const hasLower = y > 0
      const upper = hasUpper ? a(y) : Number.POSITIVE_INFINITY
      const lower = hasLower ? a(y - 1) : Number.NEGATIVE_INFINITY
      // Work in the tail where the probabilities are small, so the difference does not cancel.
      const upperTail = hasLower && lower > 0
      const p = upperTail
        ? F.sf(lower) - (hasUpper ? F.sf(upper) : 0)
        : (hasUpper ? F.cdf(upper) : 1) - (hasLower ? F.cdf(lower) : 0)
      const safe = Math.max(p, TINY)
      ll += wi * Math.log(safe)
      if (hasUpper) dA[y] = (wi * F.pdf(upper)) / safe
      if (hasLower) dA[y - 1] = (dA[y - 1] ?? 0) - (wi * F.pdf(lower)) / safe
    } else {
      if (y < K - 1) {
        ll += wi * Math.log(Math.max(F.cdf(a(y)), TINY))
        dA[y] = (wi * F.pdf(a(y))) / Math.max(F.cdf(a(y)), TINY)
      }
      for (let j = 0; j < y; j++) {
        const s = Math.max(F.sf(a(j)), TINY)
        ll += wi * Math.log(s)
        dA[j] = (dA[j] ?? 0) - (wi * F.pdf(a(j))) / s
      }
    }
    for (let j = 0; j < K - 1; j++) {
      const g = dA[j] ?? 0
      gradThresholds[j] = (gradThresholds[j] ?? 0) + g
      // d a_j / d beta = -x_i.
      for (let c = 0; c < x.d; c++) gradCoef[c] = (gradCoef[c] ?? 0) - g * (x.data[i * x.d + c] ?? 0)
    }
  }
  return { logLikelihood: ll, gradCoef, gradThresholds }
}

/**
 * The category probabilities `[n, K]` for latent scores `eta = x beta`.
 *
 * @param eta - The scores.
 * @param thresholds - The thresholds.
 * @param link - The link (default logit).
 * @param mode - Cumulative or sequential (default cumulative).
 * @returns Row-major probabilities, `n * K` entries.
 * @example
 * ```ts
 * import { ordinalProbabilities } from 'mdml-engine/linear'
 * Array.from(ordinalProbabilities([0], [-1, 1])).map((p) => Math.round(p * 1e4) / 1e4) // => [0.2689, 0.4621, 0.2689]
 * ```
 */
export function ordinalProbabilities(
  eta: ArrayLike<number>,
  thresholds: ArrayLike<number>,
  link: OrdinalLink = 'logit',
  mode: OrdinalMode = 'cumulative',
): Float64Array {
  const K = thresholds.length + 1
  const F = LINKS[link]
  const out = new Float64Array(eta.length * K)
  for (let i = 0; i < eta.length; i++) {
    const a = (k: number): number => (thresholds[k] ?? 0) - (eta[i] ?? 0)
    if (mode === 'cumulative') {
      let prev = 0
      for (let k = 0; k < K; k++) {
        const c = k < K - 1 ? F.cdf(a(k)) : 1
        out[i * K + k] = Math.max(c - prev, 0)
        prev = c
      }
    } else {
      let alive = 1
      for (let k = 0; k < K; k++) {
        const hazard = k < K - 1 ? F.cdf(a(k)) : 1
        out[i * K + k] = alive * hazard
        alive *= 1 - hazard
      }
    }
  }
  return out
}

/** Options of {@link ordinalAlgorithm}. */
export interface OrdinalOptions {
  readonly link?: OrdinalLink
  readonly mode?: OrdinalMode
  /** A ridge penalty `alpha |beta|^2 / 2` on the slopes, on the per-weight scale of the objective (default 0). */
  readonly alpha?: number
  /** Stop when the largest gradient entry falls below this (default 1e-9). */
  readonly tol?: number
}

/** The state of {@link ordinalAlgorithm}. */
export type OrdinalState = {
  /** The slopes followed by the threshold parameters (first cutpoint and log increments when cumulative). */
  readonly u: number[]
  readonly objective: number
  /** The largest entry of the gradient of the objective. */
  readonly gradient: number
  readonly evaluations: number
  /** Whether no step decreased the objective (the search ended without meeting the tolerance). */
  readonly stalled: boolean
}

function thresholdsOf(delta: ArrayLike<number>, mode: OrdinalMode): number[] {
  if (mode === 'sequential') return Array.from(delta)
  const theta: number[] = []
  let acc = 0
  for (let k = 0; k < delta.length; k++) {
    acc += k === 0 ? (delta[0] ?? 0) : Math.exp(delta[k] ?? 0)
    theta.push(acc)
  }
  return theta
}

/** The start: slopes zero, thresholds at the inverse link of the marginal (cumulative or hazard) proportions. */
function startOf(problem: OrdinalProblem, link: OrdinalLink, mode: OrdinalMode): number[] {
  const { codes, w, nClasses: K } = problem
  const count = new Float64Array(K)
  for (let i = 0; i < problem.x.n; i++) count[codes[i] ?? 0] = (count[codes[i] ?? 0] ?? 0) + (w[i] ?? 0)
  const total = count.reduce((a, b) => a + b, 0)
  const clamp = (p: number): number => Math.min(1 - 1e-6, Math.max(1e-6, p))
  const theta: number[] = []
  let below = 0
  for (let k = 0; k < K - 1; k++) {
    const p = mode === 'cumulative' ? (below + (count[k] ?? 0)) / total : (count[k] ?? 0) / (total - below)
    theta.push(LINKS[link].inverse(clamp(p)))
    below += count[k] ?? 0
  }
  const delta =
    mode === 'sequential'
      ? theta
      : theta.map((t, k) => (k === 0 ? t : Math.log(Math.max(t - (theta[k - 1] ?? 0), 1e-3))))
  return [...new Array<number>(problem.x.d).fill(0), ...delta]
}

/** Read an ordinal problem from learn data (class labels are ordered by their value). */
export function ordinalProblemOf(
  data: LearnData,
  who: string,
): { problem: OrdinalProblem; classes: readonly number[] } {
  const x = matrixView(data.x, who)
  const { classes, codes } = encodeLabels(data.y ?? [], who, x.n)
  if (classes.length < 2) throw new MdmlError('value', `${who}: needs at least two categories`)
  return { problem: { x, codes, w: sampleWeightsOf(data.weights, x.n, who), nClasses: classes.length }, classes }
}

/**
 * Maximum-likelihood ordinal regression as a traced algorithm: damped Newton steps (a Hessian from central differences
 * of the analytic gradient, a Levenberg shift until it is positive definite, backtracking on the objective) on the
 * objective `-loglik / W + alpha |beta|^2 / 2`. It logs `objective`, `gradient` (the largest gradient entry) and the
 * tensor `parameters`.
 *
 * @param problem - Features, class indices and weights.
 * @param options - Link, mode, ridge penalty and tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { ordinalAlgorithm } from 'mdml-engine/linear'
 * import { matrixView } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = matrixView([[0], [1], [2], [3], [4], [5]], 'demo')
 * const problem = { x, codes: [0, 0, 1, 0, 1, 1], w: [1, 1, 1, 1, 1, 1], nClasses: 2 }
 * runAlgorithm(ordinalAlgorithm(problem), { maxSteps: 30 }).status // => 'converged'
 * ```
 */
export function ordinalAlgorithm(problem: OrdinalProblem, options: OrdinalOptions = {}): Algorithm<OrdinalState> {
  const link = options.link ?? 'logit'
  const mode = options.mode ?? 'cumulative'
  const alpha = options.alpha ?? 0
  const tol = options.tol ?? 1e-9
  const d = problem.x.d
  const K = problem.nClasses
  const m = d + K - 1
  const total = Array.from({ length: problem.x.n }, (_, i) => problem.w[i] ?? 0).reduce((a, b) => a + b, 0)
  // Objective and gradient in the unconstrained parameters u = (beta, delta).
  const evaluate = (u: ArrayLike<number>): { objective: number; grad: Float64Array } => {
    const beta = Array.from({ length: d }, (_, j) => u[j] ?? 0)
    const delta = Array.from({ length: K - 1 }, (_, k) => u[d + k] ?? 0)
    const theta = thresholdsOf(delta, mode)
    const e = ordinalLogLikelihood(problem, beta, theta, link, mode)
    const grad = new Float64Array(m)
    for (let j = 0; j < d; j++) grad[j] = -(e.gradCoef[j] ?? 0) / total + alpha * (beta[j] ?? 0)
    if (mode === 'sequential') for (let k = 0; k < K - 1; k++) grad[d + k] = -(e.gradThresholds[k] ?? 0) / total
    else {
      // theta_k = delta_0 + sum_{1<=j<=k} exp(delta_j).
      let tail = 0
      for (let k = K - 2; k >= 0; k--) {
        tail += e.gradThresholds[k] ?? 0
        grad[d + k] = (-(k === 0 ? 1 : Math.exp(delta[k] ?? 0)) * tail) / total
      }
    }
    const penalty = (alpha * beta.reduce((a, b) => a + b * b, 0)) / 2
    return { objective: -e.logLikelihood / total + penalty, grad }
  }
  const maxAbs = (v: ArrayLike<number>): number => {
    let s = 0
    for (let i = 0; i < v.length; i++) s = Math.max(s, Math.abs(v[i] ?? 0))
    return s
  }
  return {
    name: `ordinal-${mode}-${link}`,
    config: { link, mode, alpha, tol },
    schema: channels({
      metrics: {
        objective: { description: 'the penalised negative log-likelihood per unit weight', direction: 'lower' },
        gradient: { description: 'the largest gradient entry', direction: 'lower' },
      },
      tensors: { parameters: { role: 'param', description: 'slopes then threshold parameters', shape: [m] } },
    }),
    init: () => {
      const u = startOf(problem, link, mode)
      const e = evaluate(u)
      return { u, objective: e.objective, gradient: maxAbs(e.grad), evaluations: 1, stalled: false }
    },
    step: (s) => {
      const g = evaluate(s.u).grad
      // The Hessian by central differences of the gradient, symmetrised.
      const H = new Float64Array(m * m)
      for (let j = 0; j < m; j++) {
        const h = 1e-5 * Math.max(1, Math.abs(s.u[j] ?? 0))
        const up = Float64Array.from(s.u)
        const dn = Float64Array.from(s.u)
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
      // Newton direction with a Levenberg shift until the Hessian is positive definite.
      let shift = 0
      let dir: Float64Array = new Float64Array(m)
      for (let attempt = 0; attempt < 40; attempt++) {
        const a = Float64Array.from(H)
        for (let i = 0; i < m; i++) a[i * m + i] = (a[i * m + i] ?? 0) + shift
        const { factor, info } = dpotrf(a, m)
        if (info === 0) {
          dir = dpotrs(
            factor,
            Float64Array.from(g, (v) => -v),
            m,
          )
          break
        }
        shift = shift === 0 ? 1e-6 : shift * 10
      }
      // Backtrack until the objective decreases.
      let t = 1
      let next = { objective: Number.POSITIVE_INFINITY, grad: g }
      let u = s.u
      let evaluations = s.evaluations + 2 * m + 1
      for (let tries = 0; tries < 50; tries++) {
        const trial = s.u.map((v, i) => v + t * (dir[i] ?? 0))
        const e = evaluate(trial)
        evaluations++
        if (
          Number.isFinite(e.objective) &&
          e.objective <= s.objective + 1e-4 * t * dir.reduce((a, v, i) => a + v * (g[i] ?? 0), 0)
        ) {
          next = e
          u = trial
          break
        }
        t /= 2
      }
      if (u === s.u) return { ...s, evaluations, stalled: true }
      return { u, objective: next.objective, gradient: maxAbs(next.grad), evaluations, stalled: false }
    },
    done: (s) => (s.gradient <= tol ? 'converged' : s.stalled ? 'stopped' : undefined),
    observe: (s) => ({
      metrics: { objective: s.objective, gradient: s.gradient },
      tensors: { parameters: { role: 'param', value: s.u, shape: [m] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as OrdinalState,
  }
}

/** The hyperparameters of {@link ordinal}. */
export interface OrdinalHyper {
  readonly link: OrdinalLink
  readonly mode: OrdinalMode
  readonly alpha: number
  readonly tol: number
  readonly maxIter: number
}

/** The fitted parameters of {@link ordinal}. */
export type OrdinalParams = {
  readonly coef: number[]
  readonly thresholds: number[]
  /** The categories, ascending. */
  readonly classes: number[]
  readonly link: OrdinalLink
  readonly mode: OrdinalMode
  readonly logLikelihood: number
  /** The covariance of the slopes then thresholds (inverse observed information), `[m][m]`. */
  readonly covariance: number[][]
  readonly iterations: number
  readonly converged: boolean
}

/** What a fitted ordinal model offers. */
export interface OrdinalVerbs {
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  /** The most probable category. */
  readonly predict: (x: TensorLike) => Tensor
  /** Category probabilities `[m, K]`. */
  readonly predictProba: (x: TensorLike) => Tensor
  /** The latent score `x beta`. */
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly predictDistribution: (x: TensorLike) => ReturnType<typeof categorical>
  readonly featureImportances: number[]
  readonly score: (data: LearnData) => number
}

const ORDINAL_DEFAULTS: OrdinalHyper = { link: 'logit', mode: 'cumulative', alpha: 0, tol: 1e-9, maxIter: 100 }

/** The covariance of (slopes, thresholds): the inverse of the observed information at the fit, by central differences. */
function covarianceAt(
  problem: OrdinalProblem,
  h: OrdinalHyper,
  coef: readonly number[],
  thresholds: readonly number[],
): number[][] {
  const d = coef.length
  const m = d + thresholds.length
  const gradient = (v: ArrayLike<number>): Float64Array => {
    const e = ordinalLogLikelihood(problem, Array.from(v).slice(0, d), Array.from(v).slice(d), h.link, h.mode)
    return Float64Array.from([...e.gradCoef, ...e.gradThresholds], (g) => -g)
  }
  const theta = [...coef, ...thresholds]
  const info = new Float64Array(m * m)
  const total = Array.from({ length: problem.x.n }, (_, i) => problem.w[i] ?? 0).reduce((a, b) => a + b, 0)
  for (let j = 0; j < m; j++) {
    const step = 1e-5 * Math.max(1, Math.abs(theta[j] ?? 0))
    const up = [...theta]
    const dn = [...theta]
    up[j] = (up[j] ?? 0) + step
    dn[j] = (dn[j] ?? 0) - step
    const gu = gradient(up)
    const gd = gradient(dn)
    for (let i = 0; i < m; i++) info[i * m + j] = ((gu[i] ?? 0) - (gd[i] ?? 0)) / (2 * step)
  }
  for (let i = 0; i < m; i++) {
    info[i * m + i] = (info[i * m + i] ?? 0) + (i < d ? h.alpha * total : 0)
    for (let j = 0; j < i; j++) {
      const v = 0.5 * ((info[i * m + j] ?? 0) + (info[j * m + i] ?? 0))
      info[i * m + j] = v
      info[j * m + i] = v
    }
  }
  const { factor, info: status } = dpotrf(info, m)
  if (status !== 0) return Array.from({ length: m }, () => new Array<number>(m).fill(Number.NaN))
  const cov = Array.from({ length: m }, () => new Array<number>(m).fill(0))
  for (let c = 0; c < m; c++) {
    const e = new Float64Array(m)
    e[c] = 1
    const col = dpotrs(factor, e, m)
    for (let r = 0; r < m; r++) (cov[r] as number[])[c] = col[r] ?? 0
  }
  return cov
}

function tableOf(data: LearnData, p: OrdinalParams): CoefficientTable {
  const names = [
    ...featureNamesOf(data, p.coef.length),
    ...p.thresholds.map((_, k) => `${p.classes[k] ?? k}|${p.classes[k + 1] ?? k + 1}`),
  ]
  const estimates = [...p.coef, ...p.thresholds]
  const m = estimates.length
  const cov = Float64Array.from(
    { length: m * m },
    (_, idx) => p.covariance[Math.floor(idx / m)]?.[idx % m] ?? Number.NaN,
  )
  return waldTable(names, estimates, cov)
}

/**
 * Ordinal regression (R's `polr` and `ordinal::clm`, statsmodels' `OrderedModel`): a classifier for categories with a
 * natural order, `P(Y <= k | x) = F(theta_k - x beta)` for the cumulative mode (proportional odds with the logit link)
 * or the continuation-ratio form for the sequential mode. There is no intercept: the thresholds play that part. The
 * fit is by Newton steps (session: `solver/objective`, `solver/gradient`, `solver/parameters`), `predictProba` gives
 * the category probabilities, `decisionFunction` the latent score, and the exposition holds the Wald table of slopes
 * and thresholds.
 *
 * @param hyper - `link` (logit, probit, cloglog, loglog), `mode`, `alpha` (ridge on the slopes), `tol`, `maxIter`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { ordinal } from 'mdml-engine/linear'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0.1], [0.5], [1.2], [1.9], [2.3], [3.1], [3.4], [4.2], [0.9], [2.8], [1.5], [3.9]]
 * const y = [0, 0, 0, 1, 1, 2, 1, 2, 0, 2, 1, 2]
 * const f = ordinal().fit({ x, y })
 * f.params.thresholds.length // => 2
 * toFlat(f.predict([[0], [5]])) // => [0, 2]
 * ```
 */
export function ordinal(
  hyper: Partial<OrdinalHyper> = {},
): IterativeEstimator<OrdinalHyper, OrdinalState, OrdinalParams, OrdinalVerbs> {
  return iterative<OrdinalHyper, OrdinalState, OrdinalParams, OrdinalVerbs>(
    {
      name: 'ordinal',
      kind: 'classifier',
      defaults: ORDINAL_DEFAULTS,
      hyperSpace: (h) => dict({ alpha: real(0, Math.max(10, h.alpha), { default: h.alpha }) }),
      algorithm: (data, h) => ordinalAlgorithm(ordinalProblemOf(data, 'ordinal').problem, h),
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => {
        const { problem, classes } = ordinalProblemOf(data, 'ordinal')
        const d = problem.x.d
        const coef = s.u.slice(0, d)
        const thresholds = thresholdsOf(s.u.slice(d), h.mode)
        return {
          coef,
          thresholds,
          classes: [...classes],
          link: h.link,
          mode: h.mode,
          logLikelihood: ordinalLogLikelihood(problem, coef, thresholds, h.link, h.mode).logLikelihood,
          covariance: covarianceAt(problem, h, coef, thresholds),
          iterations: s.evaluations,
          converged: s.gradient <= h.tol,
        }
      },
      verbs: (p) => {
        const K = p.classes.length
        const scores = (x: TensorLike): Float64Array => {
          const m = matrixView(x, 'ordinal', { nFeatures: p.coef.length })
          return dgemv(m.data, m.n, m.d, p.coef)
        }
        const proba = (x: TensorLike): Float64Array => ordinalProbabilities(scores(x), p.thresholds, p.link, p.mode)
        const predict = (x: TensorLike): Tensor => {
          const pr = proba(x)
          const n = pr.length / K
          return fromStorage(
            Float64Array.from({ length: n }, (_, i) => {
              let best = 0
              for (let k = 1; k < K; k++) if ((pr[i * K + k] ?? 0) > (pr[i * K + best] ?? 0)) best = k
              return p.classes[best] ?? Number.NaN
            }),
            [n],
            'float64',
          )
        }
        const base = {
          classes: p.classes,
          nFeaturesIn: p.coef.length,
          predict,
          predictProba: (x: TensorLike): Tensor => fromStorage(proba(x), [proba(x).length / K, K], 'float64'),
          decisionFunction: (x: TensorLike): Tensor => {
            const s = scores(x)
            return fromStorage(s, [s.length], 'float64')
          },
          predictDistribution: (x: TensorLike): ReturnType<typeof categorical> =>
            categorical({ probs: fromStorage(proba(x), [proba(x).length / K, K], 'float64') }),
          featureImportances: p.coef.map(Math.abs),
        }
        return { ...base, score: withDefaultScore('classifier', base).score ?? ((): number => Number.NaN) }
      },
      records: (s, data, h): readonly Exposition[] => {
        const { problem, classes } = ordinalProblemOf(data, 'ordinal')
        const coef = s.u.slice(0, problem.x.d)
        const thresholds = thresholdsOf(s.u.slice(problem.x.d), h.mode)
        const p: OrdinalParams = {
          coef,
          thresholds,
          classes: [...classes],
          link: h.link,
          mode: h.mode,
          logLikelihood: 0,
          covariance: covarianceAt(problem, h, coef, thresholds),
          iterations: s.evaluations,
          converged: s.gradient <= h.tol,
        }
        return [tableOf(data, p)]
      },
      logs: { 'solver/objective': Log(), 'solver/gradient': Log(), 'solver/parameters': Log({ detail: 'full' }) },
    },
    { ...ORDINAL_DEFAULTS, ...hyper },
  )
}
