/**
 * Linear regression that tolerates outliers and heavy tails. `theilSen` is the multivariate Theil-Sen estimator
 * (scikit-learn's `TheilSenRegressor`): least squares on subsets of the rows, combined by the spatial median of the
 * resulting coefficient vectors, which Weiszfeld's iteration finds; `huber` is Huber regression with a concomitant scale
 * (scikit-learn's `HuberRegressor`), solved by majorise-minimise steps; `quantileRegression` is the pinball-loss fit
 * (scikit-learn's `QuantileRegressor`, statsmodels' `QuantReg`) solved exactly as a linear program of the library's
 * `optim`. The first two are `Algorithm`s on the iterative base.
 */

import { MdmlError } from '../base/index.ts'
import {
  defineEstimator,
  type Estimator,
  type Fitted,
  type IterativeEstimator,
  iterative,
  type LearnData,
} from '../learn/index.ts'
import { ddot, pinv } from '../numerics/linalg/index.ts'
import { linprog } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../random/index.ts'
import { dict, real } from '../space/index.ts'
import { median } from '../stats/index.ts'
import { primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { type AffineVerbs, affineVerbs } from './_verbs.ts'
import { type Design, designOf, matrixTensor, vectorTensor } from './design.ts'
import { leastSquaresSolve } from './solvers.ts'

/** The verbs every robust linear regressor shares: those of an affine map. */
export type RobustVerbs = AffineVerbs

const robustVerbs = affineVerbs

// ---------------------------------------------------------------------------------------------------------------------
// Theil-Sen

/** Options of {@link theilSenAlgorithm}. */
export interface TheilSenOptions {
  readonly fitIntercept?: boolean
  /** The size of the row subsets (default the number of coefficients, intercept included). */
  readonly nSubsamples?: number
  /** At most this many subsets: all of them when there are fewer, else random ones (default 10000). */
  readonly maxSubpopulation?: number
  /** Stop when the median moves less than this (default 1e-3, as scikit-learn). */
  readonly tol?: number
}

/** The state of {@link theilSenAlgorithm}: the current spatial median of the subset solutions. */
export type TheilSenState = {
  /** The median, the intercept first when one is fitted. */
  readonly median: number[]
  /** How far the last step moved it. */
  readonly change: number
  readonly iterations: number
}

const EPS = 2.220446049250313e-16

/** The number of subsets of size k out of n, capped at `cap + 1` (so a comparison with `cap` is exact). */
function combinationCount(n: number, k: number, cap: number): number {
  let c = 1
  for (let i = 1; i <= k; i++) {
    c = (c * (n - k + i)) / i
    if (c > cap) return cap + 1
  }
  return Math.round(c)
}

/** The next combination of `k` out of `n` in lexicographic order (in place); false when there is none. */
function nextCombination(idx: number[], n: number): boolean {
  const k = idx.length
  let i = k - 1
  while (i >= 0 && (idx[i] ?? 0) === n - k + i) i--
  if (i < 0) return false
  idx[i] = (idx[i] ?? 0) + 1
  for (let j = i + 1; j < k; j++) idx[j] = (idx[j - 1] ?? 0) + 1
  return true
}

/** The coefficients (intercept first) of the least-squares fit to the rows `rows`, minimum norm if rank deficient. */
function subsetFit(design: Design, rows: readonly number[], fitIntercept: boolean): Float64Array {
  const { x, y } = design
  const p = x.d + (fitIntercept ? 1 : 0)
  const a = new Float64Array(rows.length * p)
  const b = new Float64Array(rows.length)
  rows.forEach((r, i) => {
    if (fitIntercept) a[i * p] = 1
    for (let j = 0; j < x.d; j++) a[i * p + (fitIntercept ? 1 : 0) + j] = x.data[r * x.d + j] ?? 0
    b[i] = y[r] ?? 0
  })
  const sol = toFlat(primalOf(pinv(matrixTensor(a, rows.length, p)))) // [p, k]
  const out = new Float64Array(p)
  for (let j = 0; j < p; j++) {
    let s = 0
    for (let i = 0; i < rows.length; i++) s += (sol[j * rows.length + i] ?? 0) * (b[i] ?? 0)
    out[j] = s
  }
  return out
}

/** One step of the modified Weiszfeld iteration for the spatial median (Vardi and Zhang), as scikit-learn's. */
function weiszfeldStep(points: Float64Array, n: number, dim: number, old: readonly number[]): number[] {
  const weights: number[] = []
  const kept: number[] = []
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < dim; j++) s += ((points[i * dim + j] ?? 0) - (old[j] ?? 0)) ** 2
    const norm = Math.sqrt(s)
    if (norm >= EPS) {
      kept.push(i)
      weights.push(norm)
    }
  }
  const atOld = kept.length < n ? 1 : 0
  const quotient = new Float64Array(dim)
  kept.forEach((i, k) => {
    for (let j = 0; j < dim; j++)
      quotient[j] = (quotient[j] ?? 0) + ((points[i * dim + j] ?? 0) - (old[j] ?? 0)) / (weights[k] ?? 1)
  })
  let quotientNorm = Math.sqrt(quotient.reduce((a, v) => a + v * v, 0))
  let direction: number[]
  if (quotientNorm > EPS) {
    const sumInv = weights.reduce((a, w) => a + 1 / w, 0)
    direction = Array.from({ length: dim }, (_, j) => {
      let s = 0
      kept.forEach((i, k) => {
        s += (points[i * dim + j] ?? 0) / (weights[k] ?? 1)
      })
      return s / sumInv
    })
  } else {
    direction = new Array<number>(dim).fill(1)
    quotientNorm = 1
  }
  const a = Math.max(0, 1 - atOld / quotientNorm)
  const b = Math.min(1, atOld / quotientNorm)
  return direction.map((v, j) => a * v + b * (old[j] ?? 0))
}

/**
 * Theil-Sen regression as a traced algorithm. The subsets' least-squares solutions are formed up front (every subset
 * when there are at most `maxSubpopulation` of them, otherwise that many random ones drawn with `key`); each step is one
 * Weiszfeld step toward the spatial median of those solutions. It logs `change` (how far the median moved) and the
 * tensor `coefficients` (intercept first).
 *
 * @param design - Features, targets and weights (the weights are ignored).
 * @param options - The subset size and count, intercept and tolerance.
 * @param key - Randomness for the subset draw (used only when the subsets are subsampled).
 * @returns The algorithm.
 * @reference H. Theil, A rank-invariant method of linear and polynomial regression analysis (1950); P. Sen,
 * Estimates of the regression coefficient based on Kendall's tau, JASA 63 (1968); T. Kaufman and R. Rousseeuw,
 * Spatial median and multivariate Theil-Sen estimators (scikit-learn's `TheilSenRegressor`).
 * @example
 * ```ts
 * import { designOf, theilSenAlgorithm } from 'mdml-engine/linear'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const design = designOf({ x: [[0], [1], [2], [3], [4]], y: [1, 3, 5, 7, 60] }, 'demo')
 * runAlgorithm(theilSenAlgorithm(design), { maxSteps: 300 }).state.median.map((v) => Math.round(v)) // => [1, 2]
 * ```
 */
export function theilSenAlgorithm(
  design: Design,
  options: TheilSenOptions = {},
  key: Key = makeKey(0),
): Algorithm<TheilSenState> {
  const fitIntercept = options.fitIntercept ?? true
  const { n, d } = design.x
  const p = d + (fitIntercept ? 1 : 0)
  const subsetSize = options.nSubsamples ?? p
  const maxSub = options.maxSubpopulation ?? 10000
  const tol = options.tol ?? 1e-3
  if (subsetSize < p || subsetSize > n)
    throw new MdmlError('value', `theilSen: nSubsamples must be between ${p} and ${n}, got ${subsetSize}`)
  const total = combinationCount(n, subsetSize, maxSub)
  const solutions: Float64Array[] = []
  if (total <= maxSub) {
    const idx = Array.from({ length: subsetSize }, (_, i) => i)
    do solutions.push(subsetFit(design, idx, fitIntercept))
    while (nextCombination(idx, n))
  } else {
    // Draw each subset from its own fold of the key.
    for (let s = 0; s < maxSub; s++) solutions.push(subsetFit(design, drawRows(key, s, n, subsetSize), fitIntercept))
  }
  const m = solutions.length
  const points = new Float64Array(m * p)
  solutions.forEach((sol, i) => {
    points.set(sol, i * p)
  })
  return {
    name: 'theil-sen',
    config: { fitIntercept, nSubsamples: subsetSize, subsets: m, tol },
    schema: channels({
      metrics: { change: { description: 'how far the spatial median moved', direction: 'lower' } },
      tensors: { coefficients: { role: 'param', description: 'the spatial median (intercept first)', shape: [p] } },
    }),
    init: () => {
      const column = (j: number): Float64Array => Float64Array.from({ length: m }, (_, i) => points[i * p + j] ?? 0)
      // One coefficient: the median of the solutions. Otherwise Weiszfeld's iteration starts at their mean.
      if (p === 1) return { median: [median(column(0))], change: 0, iterations: 0 }
      const mean = Array.from({ length: p }, (_, j) => column(j).reduce((a, v) => a + v, 0) / m)
      return { median: mean, change: Number.POSITIVE_INFINITY, iterations: 0 }
    },
    step: (s) => {
      const next = weiszfeldStep(points, m, p, s.median)
      let move = 0
      for (let j = 0; j < p; j++) move += ((next[j] ?? 0) - (s.median[j] ?? 0)) ** 2
      return { median: next, change: Math.sqrt(move), iterations: s.iterations + 1 }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { change: Number.isFinite(s.change) ? s.change : 0 },
      tensors: { coefficients: { role: 'param', value: s.median, shape: [p] } },
    }),
    serialise: (s) => stateToJson({ ...s, change: Number.isFinite(s.change) ? s.change : -1 }),
    deserialise: (j) => {
      const s = stateFromJson(j) as TheilSenState
      return s.change < 0 ? { ...s, change: Number.POSITIVE_INFINITY } : s
    },
  }
}

/** The rows of the `s`-th random subset. */
function drawRows(key: Key, s: number, n: number, size: number): number[] {
  const perm = permutation(foldIn(key, s), n)
  return Array.from(perm.data.slice(0, size))
}

/** The hyperparameters of {@link theilSen}. */
export interface TheilSenHyper {
  readonly fitIntercept: boolean
  /** The subset size (default the number of coefficients). */
  readonly nSubsamples?: number
  readonly maxSubpopulation: number
  readonly tol: number
  readonly maxIter: number
}

/** The fitted parameters of {@link theilSen}. */
export type TheilSenParams = {
  readonly coef: number[]
  readonly intercept: number
  readonly iterations: number
  readonly converged: boolean
}

const THEIL_SEN_DEFAULTS: TheilSenHyper = { fitIntercept: true, maxSubpopulation: 10000, tol: 1e-3, maxIter: 300 }

/**
 * The Theil-Sen regressor (scikit-learn's `TheilSenRegressor`): resistant to outliers in the targets (a breakdown point
 * of roughly 29% for large samples) and in the features to a lesser degree, at the cost of the subsets it fits. A
 * session adds one Weiszfeld step per iteration (`solver/change`, `solver/coefficients`).
 *
 * @param hyper - `fitIntercept`, `nSubsamples`, `maxSubpopulation`, `tol`, `maxIter`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { theilSen } from 'mdml-engine/linear'
 * const f = theilSen().fit({ x: [[0], [1], [2], [3], [4], [5]], y: [1, 3, 5, 7, 9, 100] })
 * Math.round(f.params.coef[0] ?? 0) // => 2
 * ```
 */
export function theilSen(
  hyper: Partial<TheilSenHyper> = {},
): IterativeEstimator<TheilSenHyper, TheilSenState, TheilSenParams, RobustVerbs> {
  return iterative<TheilSenHyper, TheilSenState, TheilSenParams, RobustVerbs>(
    {
      name: 'theil-sen',
      kind: 'regressor',
      defaults: THEIL_SEN_DEFAULTS,
      hyperSpace: (h) => dict({ tol: real(1e-8, 1, { log: true, default: h.tol }) }),
      algorithm: (data, h, key) =>
        theilSenAlgorithm(
          designOf(data, 'theilSen'),
          {
            fitIntercept: h.fitIntercept,
            maxSubpopulation: h.maxSubpopulation,
            tol: h.tol,
            ...(h.nSubsamples === undefined ? {} : { nSubsamples: h.nSubsamples }),
          },
          key,
        ),
      maxSteps: (h) => h.maxIter,
      params: (s, _data, h) => ({
        coef: s.median.slice(h.fitIntercept ? 1 : 0),
        intercept: h.fitIntercept ? (s.median[0] ?? 0) : 0,
        iterations: s.iterations,
        converged: s.change < h.tol,
      }),
      verbs: (p) => robustVerbs(p.coef, p.intercept, 'theilSen'),
      logs: { 'solver/change': Log(), 'solver/coefficients': Log({ detail: 'full' }) },
    },
    { ...THEIL_SEN_DEFAULTS, ...hyper },
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Huber regression

/** Options of {@link huberAlgorithm}. */
export interface HuberOptions {
  /** The threshold (in scale units) beyond which residuals count linearly (default 1.35). */
  readonly epsilon?: number
  /** The L2 penalty `alpha |beta|^2` on the coefficients (default 1e-4, as scikit-learn). */
  readonly alpha?: number
  readonly fitIntercept?: boolean
  /** Stop when the parameters move less than this (default 1e-9). */
  readonly tol?: number
}

/** The state of {@link huberAlgorithm}. */
export type HuberState = {
  readonly coef: number[]
  readonly intercept: number
  /** The scale `sigma` estimated jointly with the coefficients. */
  readonly scale: number
  readonly objective: number
  /** The largest parameter change of the last step. */
  readonly change: number
  readonly iterations: number
}

/**
 * The Huber objective `sum_i w_i [sigma + sigma H(r_i / sigma)] + alpha |beta|^2` with `H(z) = z^2` inside `|z| <= eps` and
 * `2 eps |z| - eps^2` outside (scikit-learn's `HuberRegressor`), at the given parameters.
 *
 * @param design - Features, targets and weights.
 * @param coef - Coefficients.
 * @param intercept - Intercept.
 * @param scale - The scale sigma.
 * @param epsilon - The threshold.
 * @param alpha - The L2 penalty.
 * @returns The objective.
 * @example
 * ```ts
 * import { designOf, huberObjective } from 'mdml-engine/linear'
 * huberObjective(designOf({ x: [[0], [1]], y: [0, 1] }, 'demo'), [1], 0, 1, 1.35, 0) // => 2
 * ```
 */
export function huberObjective(
  design: Design,
  coef: ArrayLike<number>,
  intercept: number,
  scale: number,
  epsilon: number,
  alpha: number,
): number {
  const { n, d } = design.x
  let total = 0
  for (let i = 0; i < n; i++) {
    const r = (design.y[i] ?? 0) - intercept - ddot(design.x.data.subarray(i * d, (i + 1) * d), coef)
    const z = Math.abs(r) / scale
    const w = design.w[i] ?? 0
    total += w * (scale + (z <= epsilon ? (r * r) / scale : 2 * epsilon * Math.abs(r) - epsilon * epsilon * scale))
  }
  return total + alpha * ddot(coef, coef)
}

/**
 * Huber regression as a traced algorithm: alternately a weighted ridge least-squares step with the Huber weights
 * (`1 / sigma` inside the threshold, `eps / |r|` outside, a majorisation of the loss) and the closed-form update of the
 * scale, so the objective of scikit-learn's `HuberRegressor` decreases at every step. It logs `objective`, `scale`,
 * `change` and the tensor `coefficients`.
 *
 * @param design - Features, targets and weights.
 * @param options - Threshold, penalty, intercept and tolerance.
 * @returns The algorithm.
 * @reference P. Huber, Robust Statistics (1981), proposal 2; A. Owen, A robust hybrid of lasso and ridge regression
 * (2006).
 * @example
 * ```ts
 * import { designOf, huberAlgorithm } from 'mdml-engine/linear'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const design = designOf({ x: [[0], [1], [2], [3], [4], [5]], y: [0.1, 1, 2.1, 2.9, 4.1, 40] }, 'demo')
 * const s = runAlgorithm(huberAlgorithm(design, { alpha: 0 }), { maxSteps: 500 }).state
 * Math.round(s.coef[0] ?? 0) // => 1
 * ```
 */
export function huberAlgorithm(design: Design, options: HuberOptions = {}): Algorithm<HuberState> {
  const epsilon = options.epsilon ?? 1.35
  const alpha = options.alpha ?? 1e-4
  const fitIntercept = options.fitIntercept ?? true
  const tol = options.tol ?? 1e-9
  if (!(epsilon >= 1)) throw new MdmlError('value', 'huber: epsilon must be at least 1')
  const { n, d } = design.x
  const residuals = (coef: ArrayLike<number>, intercept: number): Float64Array =>
    Float64Array.from(
      { length: n },
      (_, i) => (design.y[i] ?? 0) - intercept - ddot(design.x.data.subarray(i * d, (i + 1) * d), coef),
    )
  return {
    name: 'huber',
    config: { epsilon, alpha, fitIntercept, tol },
    schema: channels({
      metrics: {
        objective: { description: 'the Huber objective with concomitant scale', direction: 'lower' },
        scale: { description: 'the estimated scale sigma' },
        change: { description: 'the largest parameter change', direction: 'lower' },
      },
      tensors: { coefficients: { role: 'param', description: 'the coefficients', shape: [d] } },
    }),
    init: () => {
      const coef = new Array<number>(d).fill(0)
      return {
        coef,
        intercept: 0,
        scale: 1,
        objective: huberObjective(design, coef, 0, 1, epsilon, alpha),
        change: Number.POSITIVE_INFINITY,
        iterations: 0,
      }
    },
    step: (s) => {
      const r = residuals(s.coef, s.intercept)
      const w = Float64Array.from(r, (v, i) => {
        const a = Math.abs(v) <= epsilon * s.scale ? 1 / s.scale : epsilon / Math.max(Math.abs(v), 1e-300)
        return (design.w[i] ?? 0) * a
      })
      const fit = leastSquaresSolve({ ...design, w }, alpha, fitIntercept, 'svd')
      const coef = Array.from(fit.coef)
      const r2 = residuals(coef, fit.intercept)
      // The scale: sum_in w r^2 = sigma^2 (sum w - eps^2 sum_out w).
      let inside = 0
      let outside = 0
      let weight = 0
      for (let i = 0; i < n; i++) {
        const wi = design.w[i] ?? 0
        weight += wi
        if (Math.abs(r2[i] ?? 0) <= epsilon * s.scale) inside += wi * (r2[i] ?? 0) ** 2
        else outside += wi
      }
      const denominator = weight - epsilon * epsilon * outside
      const scale = denominator > 0 && inside > 0 ? Math.sqrt(inside / denominator) : s.scale
      let change = Math.abs(scale - s.scale)
      coef.forEach((v, j) => {
        change = Math.max(change, Math.abs(v - (s.coef[j] ?? 0)))
      })
      change = Math.max(change, Math.abs(fit.intercept - s.intercept))
      return {
        coef,
        intercept: fit.intercept,
        scale,
        objective: huberObjective(design, coef, fit.intercept, scale, epsilon, alpha),
        change,
        iterations: s.iterations + 1,
      }
    },
    done: (s) => (s.change <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { objective: s.objective, scale: s.scale, change: Number.isFinite(s.change) ? s.change : 0 },
      tensors: { coefficients: { role: 'param', value: s.coef, shape: [d] } },
    }),
    serialise: (s) => stateToJson({ ...s, change: Number.isFinite(s.change) ? s.change : -1 }),
    deserialise: (j) => {
      const s = stateFromJson(j) as HuberState
      return s.change < 0 ? { ...s, change: Number.POSITIVE_INFINITY } : s
    },
  }
}

/** The hyperparameters of {@link huber}. */
export interface HuberHyper {
  readonly epsilon: number
  readonly alpha: number
  readonly fitIntercept: boolean
  readonly tol: number
  readonly maxIter: number
}

/** The fitted parameters of {@link huber}. */
export type HuberParams = {
  readonly coef: number[]
  readonly intercept: number
  /** The estimated scale of the errors. */
  readonly scale: number
  readonly objective: number
  /** Which training rows count as outliers (residual beyond `epsilon` scales). */
  readonly outliers: boolean[]
  readonly iterations: number
  readonly converged: boolean
}

const HUBER_DEFAULTS: HuberHyper = { epsilon: 1.35, alpha: 1e-4, fitIntercept: true, tol: 1e-9, maxIter: 2000 }

/**
 * Huber regression (scikit-learn's `HuberRegressor`): squared loss for small residuals and absolute loss beyond
 * `epsilon` estimated scales, with the scale fitted jointly, so a few wild targets cannot drag the fit. A session adds
 * one majorise-minimise step per iteration (`solver/objective`, `solver/scale`, `solver/change`,
 * `solver/coefficients`); `params.outliers` marks the rows treated linearly.
 *
 * @param hyper - `epsilon`, `alpha`, `fitIntercept`, `tol`, `maxIter`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { huber } from 'mdml-engine/linear'
 * const f = huber({ alpha: 0 }).fit({ x: [[0], [1], [2], [3], [4], [5]], y: [0.1, 1, 2.1, 2.9, 4.1, 40] })
 * f.params.outliers[5] // => true
 * ```
 */
export function huber(
  hyper: Partial<HuberHyper> = {},
): IterativeEstimator<HuberHyper, HuberState, HuberParams, RobustVerbs> {
  return iterative<HuberHyper, HuberState, HuberParams, RobustVerbs>(
    {
      name: 'huber',
      kind: 'regressor',
      defaults: HUBER_DEFAULTS,
      hyperSpace: (h) =>
        dict({
          epsilon: real(1, Math.max(5, h.epsilon), { default: h.epsilon }),
          alpha: real(0, Math.max(10, h.alpha), { default: h.alpha }),
        }),
      algorithm: (data, h) =>
        huberAlgorithm(designOf(data, 'huber'), {
          epsilon: h.epsilon,
          alpha: h.alpha,
          fitIntercept: h.fitIntercept,
          tol: h.tol,
        }),
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => {
        const design = designOf(data, 'huber')
        const { n, d } = design.x
        const outliers = Array.from({ length: n }, (_, i) => {
          const r = (design.y[i] ?? 0) - s.intercept - ddot(design.x.data.subarray(i * d, (i + 1) * d), s.coef)
          return Math.abs(r) > h.epsilon * s.scale
        })
        return {
          coef: s.coef,
          intercept: s.intercept,
          scale: s.scale,
          objective: s.objective,
          outliers,
          iterations: s.iterations,
          converged: s.change <= h.tol,
        }
      },
      verbs: (p) => robustVerbs(p.coef, p.intercept, 'huber'),
      logs: {
        'solver/objective': Log(),
        'solver/scale': Log(),
        'solver/change': Log(),
        'solver/coefficients': Log({ detail: 'full' }),
      },
    },
    { ...HUBER_DEFAULTS, ...hyper },
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Quantile regression

/** Options of {@link quantileRegression}. */
export interface QuantileOptions {
  /** The quantile tau in (0, 1) (default 0.5, the median regression). */
  readonly quantile?: number
  /** The L1 penalty `alpha |beta|_1` added to the mean pinball loss (default 0, as `QuantReg`). */
  readonly alpha?: number
  readonly fitIntercept?: boolean
}

/** A quantile regression fit. */
export type QuantileFit = {
  readonly coef: number[]
  readonly intercept: number
  readonly quantile: number
  /** The penalised mean pinball loss at the solution. */
  readonly objective: number
  /** The LP's verdict. */
  readonly status: string
}

/**
 * Quantile regression by linear programming: minimise `sum_i w_i rho_tau(y_i - x_i beta - b) / sum_i w_i + alpha |beta|_1`,
 * with `rho_tau(u) = u (tau - 1[u < 0])`, written with split residuals and split coefficients and solved exactly by the
 * library's `linprog` (scikit-learn's `QuantileRegressor` with the `highs` solver).
 *
 * @param data - Features, targets and weights.
 * @param options - The quantile, the L1 penalty and the intercept.
 * @returns The coefficients, intercept and objective.
 * @throws MdmlError `value` for a quantile outside (0, 1) or a failed LP.
 * @example
 * ```ts
 * import { quantileRegression } from 'mdml-engine/linear'
 * const x = [[0], [1], [2], [3], [4]]
 * const f = quantileRegression({ x, y: [0, 1, 2, 3, 50] }, { quantile: 0.5 })
 * f.coef.map((v) => Math.round(v * 1000) / 1000) // => [1]
 * ```
 */
export function quantileRegression(data: LearnData, options: QuantileOptions = {}): QuantileFit {
  const tau = options.quantile ?? 0.5
  const alpha = options.alpha ?? 0
  const fitIntercept = options.fitIntercept ?? true
  if (!(tau > 0 && tau < 1)) throw new MdmlError('value', `quantileRegression: quantile must be in (0, 1), got ${tau}`)
  const design = designOf(data, 'quantileRegression')
  const { n, d } = design.x
  const total = design.w.reduce((a, b) => a + b, 0)
  // Variables: beta+ (d), beta- (d), [b+, b-], u (n), v (n); X beta + b + u - v = y.
  const cols = 2 * d + (fitIntercept ? 2 : 0) + 2 * n
  const c = new Float64Array(cols)
  const a = new Float64Array(n * cols)
  for (let j = 0; j < 2 * d; j++) c[j] = alpha
  const uAt = 2 * d + (fitIntercept ? 2 : 0)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      const v = design.x.data[i * d + j] ?? 0
      a[i * cols + j] = v
      a[i * cols + d + j] = -v
    }
    if (fitIntercept) {
      a[i * cols + 2 * d] = 1
      a[i * cols + 2 * d + 1] = -1
    }
    a[i * cols + uAt + i] = 1
    a[i * cols + uAt + n + i] = -1
    c[uAt + i] = (tau * (design.w[i] ?? 0)) / total
    c[uAt + n + i] = ((1 - tau) * (design.w[i] ?? 0)) / total
  }
  const r = linprog({
    c: vectorTensor(c),
    aEq: matrixTensor(a, n, cols),
    bEq: vectorTensor(design.y),
    bounds: [0, null],
  })
  if (!r.success) throw new MdmlError('value', `quantileRegression: the linear program failed (${r.status})`)
  const x = toFlat(r.x)
  return {
    coef: Array.from({ length: d }, (_, j) => (x[j] ?? 0) - (x[d + j] ?? 0)),
    intercept: fitIntercept ? (x[2 * d] ?? 0) - (x[2 * d + 1] ?? 0) : 0,
    quantile: tau,
    objective: r.fun,
    status: r.status,
  }
}

/** The hyperparameters of {@link quantileRegressor}. */
export interface QuantileHyper {
  readonly quantile: number
  readonly alpha: number
  readonly fitIntercept: boolean
}

/** The fitted parameters of {@link quantileRegressor}. */
export type QuantileParams = QuantileFit

/**
 * The quantile regressor estimator: the conditional `quantile` of the target as a linear function of the features
 * (the median by default), exact by linear programming, with an optional L1 penalty.
 *
 * @param hyper - `quantile`, `alpha`, `fitIntercept`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { quantileRegressor } from 'mdml-engine/linear'
 * import { toFlat } from 'mdml-engine/tensor'
 * const f = quantileRegressor({ quantile: 0.5 }).fit({ x: [[0], [1], [2], [3], [4]], y: [0, 1, 2, 3, 50] })
 * toFlat(f.predict([[10]]))[0] // => 10
 * ```
 */
export function quantileRegressor(
  hyper: Partial<QuantileHyper> = {},
): Estimator<QuantileHyper, Fitted<QuantileParams> & RobustVerbs> {
  return defineEstimator<QuantileHyper, QuantileParams, RobustVerbs>(
    {
      name: 'quantile-regressor',
      kind: 'regressor',
      hyperSpace: (h) =>
        dict({
          quantile: real(0.01, 0.99, { default: h.quantile }),
          alpha: real(0, Math.max(10, h.alpha), { default: h.alpha }),
        }),
      fit: (data, h) => quantileRegression(data, h),
      methods: (p) => robustVerbs(p.coef, p.intercept, 'quantileRegressor'),
    },
    { quantile: 0.5, alpha: 0, fitIntercept: true, ...hyper },
  )
}
