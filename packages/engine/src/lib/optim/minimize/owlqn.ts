/**
 * OWL-QN (orthant-wise limited-memory quasi-Newton, Andrew and Gao), for a smooth function plus an L1 penalty: L-BFGS
 * on the pseudo-gradient with steps kept within one orthant, giving exact zeros at quasi-Newton speed for the lasso and
 * sparse logistic regression.
 */

import { OmaiError } from '../../base/index.ts'
import { daxpy, ddot, dnrm2, dscal } from '../../numerics/linalg/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import {
  CURVATURE_CHANNEL,
  configOf,
  curvatureMessage,
  observeVector,
  runVector,
  type VectorState,
  vectorDone,
  vectorSchema,
} from './_shared.ts'
import {
  type FlatObjective,
  flatObjective,
  infNorm,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'
import { twoLoopRecursion } from './quasi-newton.ts'

/** Options of {@link owlqn}. */
export interface OwlqnOptions extends VectorMethodOptions {
  /** The L1 weight C of f(x) + C |x|_1: a scalar, or one weight per flattened parameter (0 leaves it unpenalised). */
  readonly l1: number | ArrayLike<number>
  /** Curvature pairs kept (default 10). */
  readonly memory?: number
  /** Stop when the average relative improvement over the last `window` iterations is below this (default 1e-10). */
  readonly ftol?: number
  /** The improvement window (default 5, as Andrew and Gao's code). */
  readonly window?: number
  /** Sufficient-decrease constant of the line search (default 1e-4). */
  readonly c1?: number
  /** Backtracking factor (default 0.5; 0.1 on the first iteration, as the reference code). */
  readonly backoff?: number
  /** Line-search evaluations per iteration (default 50). */
  readonly maxLineSearch?: number
}

/** The state of an {@link owlqn} run: `value` is the full objective f + C |x|_1 and `grad` its pseudo-gradient. */
export interface OwlqnState extends VectorState {
  /** f(x), the smooth part. */
  readonly smoothValue: number
  /** The gradient of the smooth part. */
  readonly smoothGrad: Float64Array
  readonly s: readonly Float64Array[]
  readonly y: readonly Float64Array[]
  /** Recent objective values, newest last (for the improvement test). */
  readonly history: readonly number[]
  /** Nonzero parameters. */
  readonly nonzeros: number
  /** Line-search evaluations of the last step and its accepted step length. */
  readonly evaluations: number
  readonly alpha: number
  /** The curvature pair (from the smooth gradients) of the last step and whether it entered the memory. */
  readonly curvature?: {
    readonly sy: number
    readonly used: boolean
    readonly s: Float64Array
    readonly y: Float64Array
  }
}

/**
 * The pseudo-gradient of f(x) + sum_i c_i |x_i| (Andrew and Gao 2007): the gradient where x_i is nonzero, and at
 * x_i = 0 the one-sided derivative that points downhill (zero when both sides go uphill, the optimality condition).
 *
 * @param x - The point.
 * @param g - The gradient of f.
 * @param c - The L1 weights (scalar or per coordinate).
 * @returns The pseudo-gradient.
 * @concept sparsity
 * @example
 * ```ts
 * import { pseudoGradient } from 'omai-engine/optim'
 * Array.from(pseudoGradient([1, 0, 0, 0], [0.5, 0.5, -2, 2], 1)) // => [1.5, 0, -1, 1]
 * ```
 */
export function pseudoGradient(
  x: ArrayLike<number>,
  g: ArrayLike<number>,
  c: number | ArrayLike<number>,
): Float64Array {
  const out = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) {
    const ci = typeof c === 'number' ? c : (c[i] ?? 0)
    const xi = x[i] ?? 0
    const gi = g[i] ?? 0
    if (xi > 0) out[i] = gi + ci
    else if (xi < 0) out[i] = gi - ci
    else if (gi + ci < 0) out[i] = gi + ci
    else if (gi - ci > 0) out[i] = gi - ci
    else out[i] = 0
  }
  return out
}

/**
 * OWL-QN (orthant-wise limited-memory quasi-Newton, Andrew and Gao 2007) as a traced algorithm, for
 * f(x) + C |x|_1 with f smooth: L-BFGS on the pseudo-gradient, the direction projected onto the orthant of steepest
 * descent, and a backtracking search whose points are projected onto that orthant, so coordinates reach exact zeros.
 * Curvature pairs use the gradients of f alone. Records the objective, the pseudo-gradient norm, the number of nonzeros
 * and the line-search evaluations.
 *
 * @param f - The smooth part of the objective.
 * @param x0 - The start.
 * @param options - The L1 weights, memory, tolerances and line-search constants.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for negative or mismatched weights.
 * @concept sparsity
 * @reference G. Andrew and J. Gao, Scalable training of L1-regularized log-linear models, ICML 2007.
 * @example
 * ```ts
 * import { owlqnAlgorithm } from 'omai-engine/optim'
 * import { square, sum, type Value } from 'omai-engine/tensor'
 * owlqnAlgorithm((x: Value) => sum(square(x)), [1, 2], { l1: 0.1 }).name // => 'owlqn'
 * ```
 */
export function owlqnAlgorithm(f: Objective, x0: Parameters, options: OwlqnOptions): Algorithm<OwlqnState> {
  return owlqnOn(flatObjective(f, x0), options)
}

function owlqnOn(obj: FlatObjective, options: OwlqnOptions): Algorithm<OwlqnState> {
  const n = obj.n
  const c = options.l1
  if (typeof c !== 'number' && c.length !== n)
    throw new OmaiError('value', `owlqn: expected ${n} L1 weights, got ${c.length}`)
  const weights = typeof c === 'number' ? new Float64Array(n).fill(c) : Float64Array.from(c)
  if (weights.some((w) => w < 0)) throw new OmaiError('value', 'owlqn: L1 weights must be nonnegative')
  const memory = options.memory ?? 10
  const gtol = options.gradTol ?? 1e-5
  const ftol = options.ftol ?? 1e-10
  const window = options.window ?? 5
  const c1 = options.c1 ?? 1e-4
  const backoff = options.backoff ?? 0.5
  const maxls = options.maxLineSearch ?? 50
  const l1 = (x: ArrayLike<number>): number => {
    let s = 0
    for (let i = 0; i < n; i++) s += (weights[i] ?? 0) * Math.abs(x[i] ?? 0)
    return s
  }
  const nonzeros = (x: Float64Array): number => x.reduce((k, v) => k + (v === 0 ? 0 : 1), 0)
  return {
    name: 'owlqn',
    config: { ...(configOf(options) as Record<string, number>), l1: typeof c === 'number' ? c : -1 },
    schema: vectorSchema({
      metrics: {
        smoothLoss: { description: 'the smooth part f(x)' },
        nonzeros: { description: 'parameters away from zero' },
        alpha: { description: 'accepted step length' },
        evaluations: { description: 'line-search evaluations of the step' },
      },
      messages: CURVATURE_CHANNEL,
    }),
    init: () => {
      const { value, grad } = obj.valueAndGrad(obj.x0)
      const total = value + l1(obj.x0)
      const pg = pseudoGradient(obj.x0, grad, weights)
      return {
        x: obj.x0,
        value: total,
        grad: pg,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        step: 0,
        smoothValue: value,
        smoothGrad: grad,
        s: [],
        y: [],
        history: [total],
        nonzeros: nonzeros(obj.x0),
        evaluations: 0,
        alpha: 0,
        ...(infNorm(pg) <= gtol
          ? { stop: { status: 'converged' as const, message: 'pseudo-gradient norm below tolerance' } }
          : {}),
      }
    },
    step: (st) => {
      const pg = st.grad
      const dir = dscal(-1, twoLoopRecursion(st.s, st.y, pg))
      // Keep only the components that agree with steepest descent (-pg).
      for (let i = 0; i < n; i++) if ((dir[i] ?? 0) * (pg[i] ?? 0) >= 0) dir[i] = 0
      // The directional derivative of the full objective along dir (Andrew and Gao's DirDeriv).
      let dirDeriv = 0
      for (let i = 0; i < n; i++) {
        const di = dir[i] ?? 0
        if (di === 0) continue
        const xi = st.x[i] ?? 0
        const w = weights[i] ?? 0
        const sign = xi < 0 ? -1 : xi > 0 ? 1 : di < 0 ? -1 : 1
        dirDeriv += di * ((st.smoothGrad[i] ?? 0) + sign * w)
      }
      if (!(dirDeriv < 0)) {
        return { ...st, stop: { status: 'converged', message: 'no descent direction: the pseudo-gradient vanishes' } }
      }
      const first = st.s.length === 0 && st.step === 0
      let alpha = first ? 1 / dnrm2(dir) : 1
      const back = first ? 0.1 : backoff
      let evaluations = 0
      let next: { x: Float64Array; value: number; grad: Float64Array; total: number } | undefined
      while (evaluations < maxls) {
        const xn = daxpy(alpha, dir, st.x)
        // Project onto the orthant of x (coordinates at zero move into the orthant of -pg by construction).
        for (let i = 0; i < n; i++) if ((st.x[i] ?? 0) * (xn[i] ?? 0) < 0) xn[i] = 0
        const ev = obj.valueAndGrad(xn)
        evaluations += 1
        const total = ev.value + l1(xn)
        if (total <= st.value + c1 * dirDeriv * alpha) {
          next = { x: xn, value: ev.value, grad: ev.grad, total }
          break
        }
        alpha *= back
      }
      if (next === undefined) {
        return {
          ...st,
          nfev: st.nfev + evaluations,
          ngev: st.ngev + evaluations,
          evaluations,
          stop: { status: 'failed', message: 'line search failed to find a decrease' },
        }
      }
      const s = daxpy(-1, st.x, next.x)
      const y = daxpy(-1, st.smoothGrad, next.grad)
      const used = ddot(s, y) > Number.EPSILON * ddot(y, y)
      const pgNew = pseudoGradient(next.x, next.grad, weights)
      const history = [...st.history, next.total].slice(-(window + 1))
      const oldest = history[0] ?? next.total
      const improvement =
        history.length > window ? (oldest - next.total) / window / Math.abs(next.total || 1) : Infinity
      const stop =
        infNorm(pgNew) <= gtol
          ? { status: 'converged' as const, message: 'pseudo-gradient norm below tolerance' }
          : improvement < ftol
            ? { status: 'converged' as const, message: 'average relative improvement below ftol' }
            : undefined
      return {
        x: next.x,
        value: next.total,
        grad: pgNew,
        nfev: st.nfev + evaluations,
        ngev: st.ngev + evaluations,
        nhev: 0,
        step: dnrm2(s),
        smoothValue: next.value,
        smoothGrad: next.grad,
        s: used ? [...st.s, s].slice(-memory) : st.s,
        y: used ? [...st.y, y].slice(-memory) : st.y,
        history,
        nonzeros: nonzeros(next.x),
        evaluations,
        alpha,
        curvature: { sy: ddot(s, y), used, s, y },
        ...(stop === undefined ? {} : { stop }),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { smoothLoss: s.smoothValue, nonzeros: s.nonzeros, alpha: s.alpha, evaluations: s.evaluations },
        ...(s.curvature === undefined
          ? {}
          : { messages: [curvatureMessage(s.curvature.sy, s.curvature.s, s.curvature.y, s.curvature.used)] }),
      }),
  }
}

/**
 * Minimise f(x) + C |x|_1 by OWL-QN: L1-regularised regression and classification (lasso, sparse logistic regression)
 * with exact zeros, at quasi-Newton speed. The result's `fun` is the full objective and `grad` the pseudo-gradient.
 *
 * @param f - The smooth part of the objective.
 * @param x0 - The start (any parameter tree; weights apply to its flattened entries).
 * @param options - The L1 weights (required), memory, tolerances, line-search constants, iteration cap (default 1000)
 * and run options.
 * @returns The result.
 * @concept sparsity
 * @example
 * ```ts
 * import { owlqn } from 'omai-engine/optim'
 * import { mul, square, sub, sum, tensor, toFlat, type Value } from 'omai-engine/tensor'
 * // 0.5 |x - a|^2 + 0.5 |x|_1 soft-thresholds a = [2, 0.3, -1] by 0.5.
 * const r = owlqn((x: Value) => mul(sum(square(sub(x, tensor([2, 0.3, -1])))), 0.5), [0, 0, 0], { l1: 0.5 })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [1.5, 0, -0.5]
 * ```
 */
export function owlqn(f: Objective, x0: Parameters, options: OwlqnOptions): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(owlqnOn(obj, options), obj, options, 1000)
}
