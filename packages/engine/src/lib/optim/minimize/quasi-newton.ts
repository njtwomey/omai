/**
 * Quasi-Newton minimisers, which build curvature from gradient differences instead of computing Hessians: BFGS (a port
 * of scipy's, with the inverse-Hessian update exposed) and L-BFGS with the two-loop recursion over a limited memory of
 * curvature pairs, both as traced algorithms with gradients by autodiff.
 */

import { daxpy, ddot, dgemv, dnrm2, dscal } from '../../numerics/linalg/index.ts'
import { fromStorage, type Tensor } from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { type LineSearch, type LineSearchResult, lineFunction } from '../line-search.ts'
import { moreThuente, wolfe12 } from '../line-search-interval.ts'
import {
  CURVATURE_CHANNEL,
  configOf,
  curvatureMessage,
  LINE_SEARCH_CHANNEL,
  lineSearchMessage,
  observeVector,
  runVector,
  runVectorState,
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

/** Options of {@link bfgs}. */
export interface BfgsOptions extends VectorMethodOptions {
  /** Line search constants (scipy's c1 = 1e-4, c2 = 0.9). */
  readonly c1?: number
  readonly c2?: number
  /** Stop when the step is below xrtol (xrtol + |x|) (default 0, scipy's). */
  readonly xrtol?: number
  /** The initial inverse Hessian, n x n row-major (default the identity). */
  readonly hessInv0?: ArrayLike<number>
  /**
   * Powell's damping: when s'y < 0.2 s'Bs, replace y by theta y + (1 - theta) B s so that the update stays positive
   * definite (default false, which matches scipy).
   */
  readonly damped?: boolean
  /** The line search (default scipy's: More-Thuente with a zoom fallback). */
  readonly lineSearch?: LineSearch
}

/** The state of a {@link bfgs} run. */
export interface BfgsState extends VectorState {
  /** The inverse Hessian approximation, n x n row-major. */
  readonly hessInv: Float64Array
  readonly previousValue: number
  readonly lineSearch?: LineSearchResult
  readonly curvature?: {
    readonly sy: number
    readonly used: boolean
    readonly theta: number
    readonly s: Float64Array
    readonly y: Float64Array
  }
}

/**
 * The BFGS update of an inverse Hessian approximation, H+ = (I - rho s y') H (I - rho y s') + rho s s' with
 * rho = 1 / y's (scipy's rho = 1000 when y's = 0), in O(n^2).
 *
 * @param h - H, n x n row-major.
 * @param s - The step.
 * @param y - The gradient change.
 * @returns H+.
 * @concept quasi-newton
 * @example
 * ```ts
 * import { bfgsInverseUpdate } from 'mdml-engine/optim'
 * // After the update, H+ y = s (the secant equation).
 * const h = bfgsInverseUpdate(new Float64Array([1, 0, 0, 1]), new Float64Array([1, 0]), new Float64Array([2, 1]))
 * [h[0] * 2 + h[1] * 1, h[2] * 2 + h[3] * 1] // => [1, 0]
 * ```
 */
export function bfgsInverseUpdate(h: Float64Array, s: Float64Array, y: Float64Array): Float64Array {
  const n = s.length
  const sy = ddot(y, s)
  const rho = sy === 0 ? 1000 : 1 / sy
  const hy = dgemv(h, n, n, y)
  const yh = dgemv(h, n, n, y, true)
  const yhy = ddot(y, hy)
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const si = s[i] ?? 0
      const sj = s[j] ?? 0
      out[i * n + j] =
        (h[i * n + j] ?? 0) - rho * (si * (yh[j] ?? 0) + (hy[i] ?? 0) * sj) + (rho * rho * yhy + rho) * si * sj
    }
  }
  return out
}

/**
 * BFGS as a traced algorithm, a port of scipy's: the inverse Hessian approximation drives the direction p = -H g, the
 * Wolfe line search sets the step, and each curvature pair (s, y) updates H (optionally with Powell's damping).
 * Records the objective, gradient norm, step, line-search trials and curvature pairs.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Line search constants, xrtol, the initial inverse Hessian, damping and the gradient tolerance
 * (infinity norm, default 1e-5).
 * @returns The algorithm.
 * @concept quasi-newton
 * @example
 * ```ts
 * import { bfgsAlgorithm } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * bfgsAlgorithm((x: Value) => sum(square(x)), [1, 2]).name // => 'bfgs'
 * ```
 */
export function bfgsAlgorithm(f: Objective, x0: Parameters, options: BfgsOptions = {}): Algorithm<BfgsState> {
  return bfgsOn(flatObjective(f, x0), options)
}

function bfgsOn(obj: FlatObjective, options: BfgsOptions): Algorithm<BfgsState> {
  const n = obj.n
  const gtol = options.gradTol ?? 1e-5
  const xrtol = options.xrtol ?? 0
  const search =
    options.lineSearch ??
    wolfe12({ c1: options.c1 ?? 1e-4, c2: options.c2 ?? 0.9, alphaMin: 1e-100, alphaMax: 1e100, boundFallback: true })
  const identity = (): Float64Array => {
    const h = new Float64Array(n * n)
    for (let i = 0; i < n; i++) h[i * n + i] = 1
    return h
  }
  return {
    name: 'bfgs',
    config: configOf(options),
    schema: vectorSchema({
      metrics: { alpha: { description: 'accepted step length along the direction' } },
      messages: { ...LINE_SEARCH_CHANNEL, ...CURVATURE_CHANNEL },
    }),
    init: () => {
      const { value, grad } = obj.valueAndGrad(obj.x0)
      return {
        x: obj.x0,
        value,
        grad,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        step: 0,
        hessInv: options.hessInv0 === undefined ? identity() : Float64Array.from(options.hessInv0),
        previousValue: value + dnrm2(grad) / 2,
        ...(infNorm(grad) <= gtol
          ? { stop: { status: 'converged' as const, message: 'gradient norm below tolerance' } }
          : {}),
      }
    },
    step: (s) => {
      const p = dscal(-1, dgemv(s.hessInv, n, n, s.grad))
      const ls = search(lineFunction(obj, s.x, p), {
        value: s.value,
        slope: ddot(s.grad, p),
        previousValue: s.previousValue,
      })
      const nfev = s.nfev + ls.evaluations
      let ngev = s.ngev + ls.gradEvaluations
      if (!ls.success) {
        return {
          ...s,
          nfev,
          ngev,
          lineSearch: ls,
          stop: { status: 'failed', message: 'desired error not necessarily achieved due to precision loss' },
        }
      }
      const step = dscal(ls.alpha, p)
      const x = daxpy(1, step, s.x)
      let grad = ls.grad
      if (grad === undefined) {
        grad = obj.valueAndGrad(x).grad
        ngev += 1
      }
      let y = daxpy(-1, s.grad, grad)
      const base = {
        x,
        value: ls.value,
        grad,
        nfev,
        ngev,
        nhev: 0,
        step: dnrm2(step),
        previousValue: s.value,
        lineSearch: ls,
      }
      if (infNorm(grad) <= gtol) {
        return { ...base, hessInv: s.hessInv, stop: { status: 'converged', message: 'gradient norm below tolerance' } }
      }
      if (ls.alpha * dnrm2(p) <= xrtol * (xrtol + dnrm2(x))) {
        return { ...base, hessInv: s.hessInv, stop: { status: 'converged', message: 'step below xrtol' } }
      }
      if (!Number.isFinite(ls.value)) {
        return { ...base, hessInv: s.hessInv, stop: { status: 'failed', message: 'non-finite objective' } }
      }
      let theta = 1
      if (options.damped === true) {
        // B s = -alpha g for the current approximation B = H^-1, so s'Bs = -alpha s'g.
        const bs = dscal(-ls.alpha, s.grad)
        const sbs = ddot(step, bs)
        const sy = ddot(step, y)
        if (sy < 0.2 * sbs) {
          theta = (0.8 * sbs) / (sbs - sy)
          y = daxpy(1 - theta, bs, dscal(theta, y))
        }
      }
      const sy = ddot(step, y)
      return {
        ...base,
        hessInv: bfgsInverseUpdate(s.hessInv, step, y),
        curvature: { sy, used: true, theta, s: step, y },
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { alpha: s.lineSearch?.alpha ?? 0 },
        messages: [
          ...(s.lineSearch === undefined ? [] : [lineSearchMessage(s.lineSearch)]),
          ...(s.curvature === undefined
            ? []
            : [curvatureMessage(s.curvature.sy, s.curvature.s, s.curvature.y, true, s.curvature.theta)]),
        ],
      }),
  }
}

/**
 * Minimise by BFGS, a port of scipy.optimize.minimize with method 'BFGS' (gradients by autodiff). The result carries
 * the final inverse Hessian approximation.
 *
 * @param f - The objective.
 * @param x0 - The start (any parameter tree).
 * @param options - Line search constants, xrtol, initial inverse Hessian, damping, gradient tolerance (default 1e-5),
 * iteration cap (default 200 n) and run options.
 * @returns The result with `hessInv`.
 * @concept quasi-newton
 * @index function BFGS minimisation, a port of scipy's, gradients by autodiff.
 * @example
 * ```ts
 * import { bfgs } from 'mdml-engine/optim'
 * import { add, mul, square, sub, sum, toFlat, type Value } from 'mdml-engine/tensor'
 * const r = bfgs((x: Value) => add(sum(square(sub(x, 1))), mul(sum(square(x)), 0.5)), [3, -3])
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [0.666667, 0.666667]
 * ```
 */
export function bfgs(
  f: Objective,
  x0: Parameters,
  options: BfgsOptions = {},
): MinimizeResult & { readonly hessInv: Tensor } {
  const obj = flatObjective(f, x0)
  const { result, state } = runVectorState(bfgsOn(obj, options), obj, options, 200 * obj.n)
  return { ...result, hessInv: fromStorage(Float64Array.from(state.hessInv), [obj.n, obj.n], 'float64') }
}

/**
 * The L-BFGS two-loop recursion: H q for the limited-memory inverse Hessian built from curvature pairs (oldest first),
 * with the initial scaling gamma I (default s'y / y'y of the newest pair).
 *
 * @param s - Steps, oldest first.
 * @param y - Gradient changes, oldest first.
 * @param q - The vector (usually the gradient).
 * @param gamma - The initial scaling.
 * @returns H q.
 * @concept quasi-newton
 * @example
 * ```ts
 * import { twoLoopRecursion } from 'mdml-engine/optim'
 * // One pair: H y = s.
 * Array.from(twoLoopRecursion([new Float64Array([1, 0])], [new Float64Array([2, 0])], [2, 0])) // => [1, 0]
 * ```
 */
export function twoLoopRecursion(
  s: readonly Float64Array[],
  y: readonly Float64Array[],
  q: ArrayLike<number>,
  gamma?: number,
): Float64Array {
  const m = s.length
  let r: Float64Array = Float64Array.from(q)
  const alpha = new Float64Array(m)
  const rho = new Float64Array(m)
  for (let i = m - 1; i >= 0; i--) {
    const si = s[i] as Float64Array
    const yi = y[i] as Float64Array
    rho[i] = 1 / ddot(yi, si)
    alpha[i] = (rho[i] ?? 0) * ddot(si, r)
    r = daxpy(-(alpha[i] ?? 0), yi, r)
  }
  const newest = m - 1
  const g =
    gamma ??
    (m === 0
      ? 1
      : ddot(s[newest] as Float64Array, y[newest] as Float64Array) /
        ddot(y[newest] as Float64Array, y[newest] as Float64Array))
  r = dscal(g, r)
  for (let i = 0; i < m; i++) {
    const beta = (rho[i] ?? 0) * ddot(y[i] as Float64Array, r)
    r = daxpy((alpha[i] ?? 0) - beta, s[i] as Float64Array, r)
  }
  return r
}

/** Options of {@link lbfgs}. */
export interface LbfgsOptions extends VectorMethodOptions {
  /** The number of curvature pairs kept (default 10, scipy's maxcor). */
  readonly memory?: number
  /** Stop when the relative reduction (f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) is at most this (default 2.22e-9). */
  readonly ftol?: number
  /** The line search (default More-Thuente with c1 = 1e-4, c2 = 0.9). */
  readonly lineSearch?: LineSearch
}

/** The state of a limited-memory quasi-Newton run. */
export interface LbfgsState extends VectorState {
  /** Curvature pairs, oldest first. */
  readonly s: readonly Float64Array[]
  readonly y: readonly Float64Array[]
  readonly lineSearch?: LineSearchResult
  readonly curvature?: {
    readonly sy: number
    readonly used: boolean
    readonly s: Float64Array
    readonly y: Float64Array
  }
}

/**
 * L-BFGS as a traced algorithm: the two-loop recursion over the last `memory` curvature pairs gives the direction, a
 * More-Thuente line search the step (starting from 1 / |g| on the first iteration, as L-BFGS-B), and pairs with
 * s'y <= eps y'y are skipped. Stops on the gradient's infinity norm or a relative reduction below `ftol`.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Memory, tolerances and the line search.
 * @returns The algorithm.
 * @concept quasi-newton
 * @example
 * ```ts
 * import { lbfgsAlgorithm } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * lbfgsAlgorithm((x: Value) => sum(square(x)), [1, 2]).name // => 'lbfgs'
 * ```
 */
export function lbfgsAlgorithm(f: Objective, x0: Parameters, options: LbfgsOptions = {}): Algorithm<LbfgsState> {
  return lbfgsOn(flatObjective(f, x0), options)
}

function lbfgsOn(obj: FlatObjective, options: LbfgsOptions): Algorithm<LbfgsState> {
  const memory = options.memory ?? 10
  const gtol = options.gradTol ?? 1e-5
  const ftol = options.ftol ?? 2.220446049250313e-9
  const search = options.lineSearch ?? moreThuente({ alphaMax: 1e10 })
  return {
    name: 'lbfgs',
    config: configOf(options),
    schema: vectorSchema({
      metrics: {
        alpha: { description: 'accepted step length along the direction' },
        pairs: { description: 'curvature pairs held' },
      },
      messages: { ...LINE_SEARCH_CHANNEL, ...CURVATURE_CHANNEL },
    }),
    init: () => {
      const { value, grad } = obj.valueAndGrad(obj.x0)
      return {
        x: obj.x0,
        value,
        grad,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        step: 0,
        s: [],
        y: [],
        ...(infNorm(grad) <= gtol
          ? { stop: { status: 'converged' as const, message: 'gradient norm below tolerance' } }
          : {}),
      }
    },
    step: (st) => {
      const p = dscal(-1, twoLoopRecursion(st.s, st.y, st.grad))
      const slope = ddot(st.grad, p)
      const first = st.s.length === 0
      const alpha0 = first ? Math.min(1, 1 / dnrm2(p)) : 1
      const raw = search(lineFunction(obj, st.x, p), { value: st.value, slope, alpha: alpha0 })
      // More-Thuente's warnings (xtol, step bounds, rounding) still end at a usable trial step, as in L-BFGS-B.
      const usable =
        !raw.success &&
        Number.isFinite(raw.alpha) &&
        raw.value <= st.value &&
        /xtol|alphaMax|alphaMin|rounding/.test(raw.message)
      const ls = usable ? { ...raw, success: true } : raw
      const nfev = st.nfev + ls.evaluations
      let ngev = st.ngev + ls.gradEvaluations
      if (!ls.success) {
        if (!first) {
          // Restart from steepest descent with an empty memory, as L-BFGS-B does after a failed line search.
          return { ...st, nfev, ngev, s: [], y: [], lineSearch: ls }
        }
        return {
          ...st,
          nfev,
          ngev,
          lineSearch: ls,
          stop: { status: 'failed', message: `line search failed: ${ls.message}` },
        }
      }
      const step = dscal(ls.alpha, p)
      const x = daxpy(1, step, st.x)
      let grad = ls.grad
      if (grad === undefined) {
        grad = obj.valueAndGrad(x).grad
        ngev += 1
      }
      const y = daxpy(-1, st.grad, grad)
      const sy = ddot(step, y)
      const used = sy > Number.EPSILON * ddot(y, y)
      const S = used ? [...st.s, step].slice(-memory) : st.s
      const Y = used ? [...st.y, y].slice(-memory) : st.y
      const reduction = (st.value - ls.value) / Math.max(Math.abs(st.value), Math.abs(ls.value), 1)
      const stop =
        infNorm(grad) <= gtol
          ? { status: 'converged' as const, message: 'gradient norm below tolerance' }
          : reduction <= ftol
            ? { status: 'converged' as const, message: 'relative reduction of f below ftol' }
            : undefined
      return {
        x,
        value: ls.value,
        grad,
        nfev,
        ngev,
        nhev: 0,
        step: dnrm2(step),
        s: S,
        y: Y,
        lineSearch: ls,
        curvature: { sy, used, s: step, y },
        ...(stop === undefined ? {} : { stop }),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { alpha: s.lineSearch?.alpha ?? 0, pairs: s.s.length },
        messages: [
          ...(s.lineSearch === undefined ? [] : [lineSearchMessage(s.lineSearch)]),
          ...(s.curvature === undefined
            ? []
            : [curvatureMessage(s.curvature.sy, s.curvature.s, s.curvature.y, s.curvature.used)]),
        ],
      }),
  }
}

/**
 * Minimise by L-BFGS (limited-memory BFGS without bounds; {@link lbfgsb} adds box constraints).
 *
 * @param f - The objective.
 * @param x0 - The start (any parameter tree).
 * @param options - Memory (default 10), gradient tolerance (default 1e-5), ftol, line search, iteration cap (default
 * 15000) and run options.
 * @returns The result.
 * @concept quasi-newton
 * @index function L-BFGS minimisation without bounds.
 * @example
 * ```ts
 * import { lbfgs } from 'mdml-engine/optim'
 * import { square, sub, sum, type Value } from 'mdml-engine/tensor'
 * lbfgs((x: Value) => sum(square(sub(x, 3))), [0, 0, 0]).converged // => true
 * ```
 */
export function lbfgs(f: Objective, x0: Parameters, options: LbfgsOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(lbfgsOn(obj, options), obj, options, 15000)
}
