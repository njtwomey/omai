/**
 * Newton's method for minimisation: the modified Cholesky factorisation that shifts an indefinite Hessian until it is
 * positive definite (Nocedal and Wright, Algorithm 3.3), Newton with a line search, and Newton-CG (truncated Newton), a
 * port of scipy's that solves for the step by conjugate gradients on Hessian-vector products from autodiff.
 */

import { MdmlError } from '../../base/index.ts'
import { dasum, daxpy, ddot, dgemv, dnrm2, dpotrf, dpotrs, dscal, dsyev } from '../../numerics/linalg/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { backtracking, type LineSearch, type LineSearchResult, lineFunction } from '../line-search.ts'
import { wolfe12 } from '../line-search-interval.ts'
import {
  configOf,
  LINE_SEARCH_CHANNEL,
  lineSearchMessage,
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

/** Result of {@link modifiedCholesky}. */
export interface ModifiedCholesky {
  /** The upper factor U with H + tau I = U^T U. */
  readonly factor: Float64Array
  /** The multiple of the identity that was added (0 when H is already positive definite). */
  readonly tau: number
  /** Factorisations tried. */
  readonly attempts: number
}

/**
 * Cholesky with an added multiple of the identity (Nocedal and Wright, Algorithm 3.3): factor H + tau I with tau = 0
 * when the diagonal is positive (else -min h_ii + beta), doubling tau (at least beta) until the factorisation
 * succeeds. The modified Hessian is positive definite, so the Newton direction it gives is a descent direction.
 *
 * @param h - The symmetric matrix, n x n row-major.
 * @param n - Its order.
 * @param beta - The smallest shift (default 1e-3).
 * @returns The factor, the shift and the number of attempts.
 * @concept newton-method
 * @example
 * ```ts
 * import { modifiedCholesky } from 'mdml-engine/optim'
 * modifiedCholesky([1, 2, 2, 1], 2).tau > 1 // => true
 * modifiedCholesky([2, 0, 0, 3], 2).tau // => 0
 * ```
 */
export function modifiedCholesky(h: ArrayLike<number>, n: number, beta = 1e-3): ModifiedCholesky {
  let minDiag = Infinity
  for (let i = 0; i < n; i++) minDiag = Math.min(minDiag, h[i * n + i] ?? 0)
  let tau = minDiag > 0 ? 0 : -minDiag + beta
  for (let attempts = 1; attempts <= 200; attempts++) {
    const a = Float64Array.from(h)
    for (let i = 0; i < n; i++) a[i * n + i] = (a[i * n + i] ?? 0) + tau
    const { factor, info } = dpotrf(a, n)
    if (info === 0) return { factor, tau, attempts }
    tau = Math.max(2 * tau, beta)
  }
  throw new MdmlError('singular', 'modifiedCholesky: no shift made the matrix positive definite')
}

/** Options of {@link newton}. */
export interface NewtonOptions extends VectorMethodOptions {
  /** How the Hessian is made positive definite: an added multiple of the identity (default) or eigenvalue flooring. */
  readonly modification?: 'cholesky' | 'eigenvalue'
  /** The smallest shift of the Cholesky modification (default 1e-3), or the eigenvalue floor relative to max |lambda|. */
  readonly beta?: number
  /**
   * The line search (default safeguarded Armijo {@link backtracking} from the full Newton step, robust near maxima and
   * saddles).
   */
  readonly lineSearch?: LineSearch
}

/** The state of a {@link newton} run. */
export interface NewtonState extends VectorState {
  /** The shift added to the Hessian at the last step. */
  readonly tau: number
  readonly lineSearch?: LineSearchResult
}

function newtonDirection(
  h: Float64Array,
  g: Float64Array,
  n: number,
  options: NewtonOptions,
): { p: Float64Array; tau: number } {
  if (options.modification === 'eigenvalue') {
    const { w, v } = dsyev(h, n)
    let big = 0
    for (const lam of w) big = Math.max(big, Math.abs(lam))
    const floor = (options.beta ?? 1e-8) * Math.max(1, big)
    // p = -V diag(1 / max(|lambda|, floor)) V^T g
    const vtg = dgemv(v, n, n, g, true)
    let shift = 0
    for (let i = 0; i < n; i++) {
      const lam = w[i] ?? 0
      const mod = Math.max(Math.abs(lam), floor)
      shift = Math.max(shift, mod - lam)
      vtg[i] = (vtg[i] ?? 0) / mod
    }
    return { p: dscal(-1, dgemv(v, n, n, vtg)), tau: shift }
  }
  const { factor, tau } = modifiedCholesky(h, n, options.beta ?? 1e-3)
  return { p: dscal(-1, dpotrs(factor, g, n)), tau }
}

/**
 * Newton's method with a modified Hessian and a line search, as a traced algorithm: each step solves
 * (H + tau I) p = -g with the smallest shift that makes the Cholesky factorisation succeed (or floors the eigenvalues),
 * then backtracks along p from the full step. Quadratically convergent near a minimum with a positive definite Hessian;
 * the modification keeps it a descent method elsewhere.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Modification, shift, line search, gradient tolerance (infinity norm, default 1e-5).
 * @returns The algorithm.
 * @concept newton-method
 * @example
 * ```ts
 * import { newtonAlgorithm } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * newtonAlgorithm((x: Value) => sum(square(x)), [1, 2]).name // => 'newton'
 * ```
 */
export function newtonAlgorithm(f: Objective, x0: Parameters, options: NewtonOptions = {}): Algorithm<NewtonState> {
  return newtonOn(flatObjective(f, x0), options)
}

function newtonOn(obj: FlatObjective, options: NewtonOptions): Algorithm<NewtonState> {
  const gtol = options.gradTol ?? 1e-5
  const search = options.lineSearch ?? backtracking({ safeguard: [0.1, 0.5] })
  const n = obj.n
  return {
    name: 'newton',
    config: configOf(options),
    schema: vectorSchema({
      metrics: { tau: { description: 'multiple of the identity added to the Hessian' } },
      messages: LINE_SEARCH_CHANNEL,
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
        tau: 0,
        ...(infNorm(grad) <= gtol
          ? { stop: { status: 'converged' as const, message: 'gradient norm below tolerance' } }
          : {}),
      }
    },
    step: (s) => {
      const h = obj.hessian(s.x)
      let dir: { p: Float64Array; tau: number }
      try {
        dir = newtonDirection(h, s.grad, n, options)
      } catch (e) {
        return { ...s, nhev: s.nhev + 1, stop: { status: 'failed', message: (e as Error).message } }
      }
      const slope = ddot(s.grad, dir.p)
      const ls = search(lineFunction(obj, s.x, dir.p), { value: s.value, slope, alpha: 1 })
      if (!ls.success) {
        return {
          ...s,
          nfev: s.nfev + ls.evaluations,
          ngev: s.ngev + ls.gradEvaluations,
          nhev: s.nhev + 1,
          tau: dir.tau,
          lineSearch: ls,
          stop: { status: 'failed', message: `line search failed: ${ls.message}` },
        }
      }
      const x = daxpy(ls.alpha, dir.p, s.x)
      let grad = ls.grad
      let ngev = s.ngev + ls.gradEvaluations
      let value = ls.value
      if (grad === undefined) {
        const ev = obj.valueAndGrad(x)
        grad = ev.grad
        value = ev.value
        ngev += 1
      }
      return {
        x,
        value,
        grad,
        nfev: s.nfev + ls.evaluations,
        ngev,
        nhev: s.nhev + 1,
        step: ls.alpha * dnrm2(dir.p),
        tau: dir.tau,
        lineSearch: ls,
        ...(infNorm(grad) <= gtol
          ? { stop: { status: 'converged' as const, message: 'gradient norm below tolerance' } }
          : // An accepted step that leaves the objective and the gradient no better is a step made of rounding: the
            // tolerance is below the noise of the objective, and more steps would only walk to the cap.
            value >= s.value && infNorm(grad) >= infNorm(s.grad)
            ? {
                stop: {
                  status: 'failed' as const,
                  message:
                    'no decrease: the step changes neither the objective nor the gradient norm (the tolerance is below the noise floor)',
                },
              }
            : {}),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { tau: s.tau },
        ...(s.lineSearch === undefined ? {} : { messages: [lineSearchMessage(s.lineSearch)] }),
      }),
  }
}

/**
 * Minimise by Newton's method with a modified Cholesky factorisation (or eigenvalue flooring) and a line search.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Modification, shift, line search, gradient tolerance (default 1e-5), iteration cap (default 200 n)
 * and run options.
 * @returns The result.
 * @concept newton-method
 * @index function Newton's method with modified Cholesky and line search.
 * @example
 * ```ts
 * import { newton } from 'mdml-engine/optim'
 * import { add, mul, square, sub, sum, toFlat, type Value } from 'mdml-engine/tensor'
 * const r = newton((x: Value) => add(sum(square(sub(x, 1))), mul(sum(square(x)), 0.5)), [3, -3])
 * [r.steps, toFlat(r.x as Value).map((v) => Math.round(v * 1e12) / 1e12)] // => [1, [0.666666666667, 0.666666666667]]
 * ```
 */
export function newton(f: Objective, x0: Parameters, options: NewtonOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(newtonOn(obj, options), obj, options, 200 * obj.n)
}

/** Options of {@link newtonCg}. */
export interface NewtonCgOptions extends VectorMethodOptions {
  /** Stop when the L1 norm of the step falls to n xtol (default 1e-5, scipy's). */
  readonly xtol?: number
  /** Line search constants (scipy's c1 = 1e-4, c2 = 0.9). */
  readonly c1?: number
  readonly c2?: number
  /** Use Hessian-vector products (default) or form the dense Hessian once per iteration. */
  readonly hessian?: 'products' | 'dense'
}

/** The state of a {@link newtonCg} run. */
export interface NewtonCgState extends VectorState {
  readonly previousValue?: number
  /** The L1 norm of the last update. */
  readonly update: number
  readonly cgIterations: number
  readonly lineSearch?: LineSearchResult
}

/**
 * Newton-CG (truncated Newton) as a traced algorithm, a port of scipy's: each step solves H p = -g approximately by
 * conjugate gradients on Hessian-vector products (stopping at |r|_1 <= min(0.5, sqrt|g|_1) |g|_1, or at negative
 * curvature), then searches along p with the Wolfe line search. Stops when the step's L1 norm is below n xtol.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - xtol, line search constants and the Hessian mode.
 * @returns The algorithm.
 * @concept newton-method
 * @example
 * ```ts
 * import { newtonCgAlgorithm } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * newtonCgAlgorithm((x: Value) => sum(square(x)), [1, 2]).name // => 'newtonCg'
 * ```
 */
export function newtonCgAlgorithm(
  f: Objective,
  x0: Parameters,
  options: NewtonCgOptions = {},
): Algorithm<NewtonCgState> {
  return newtonCgOn(flatObjective(f, x0), options)
}

function newtonCgOn(obj: FlatObjective, options: NewtonCgOptions): Algorithm<NewtonCgState> {
  const n = obj.n
  const xtol = n * (options.xtol ?? 1e-5)
  const cgMax = 20 * n
  const search = wolfe12({ c1: options.c1 ?? 1e-4, c2: options.c2 ?? 0.9 })
  const dense = options.hessian === 'dense'
  const gtol = options.gradTol
  return {
    name: 'newtonCg',
    config: configOf(options),
    schema: vectorSchema({
      metrics: {
        update: { description: 'L1 norm of the last update' },
        cgIterations: { description: 'conjugate gradient iterations of the last step' },
      },
      messages: LINE_SEARCH_CHANNEL,
    }),
    init: () => {
      const { value, grad } = obj.valueAndGrad(obj.x0)
      return { x: obj.x0, value, grad, nfev: 1, ngev: 1, nhev: 0, step: 0, update: Number.MAX_VALUE, cgIterations: 0 }
    },
    step: (s) => {
      const b = dscal(-1, s.grad)
      const maggrad = dasum(b)
      const termcond = Math.min(0.5, Math.sqrt(maggrad)) * maggrad
      let xsupi: Float64Array = new Float64Array(n)
      let ri: Float64Array = Float64Array.from(s.grad)
      let psupi: Float64Array = dscal(-1, ri)
      let dri0 = ddot(ri, ri)
      let nhev = s.nhev
      const A = dense ? obj.hessian(s.x) : undefined
      if (dense) nhev += 1
      let i = 0
      let k2 = 0
      for (; k2 < cgMax; k2++) {
        if (dasum(ri) <= termcond) break
        const Ap = A === undefined ? obj.hvp(s.x, psupi) : dgemv(A, n, n, psupi)
        if (A === undefined) nhev += 1
        const curv = ddot(psupi, Ap)
        if (curv >= 0 && curv <= 3 * Number.EPSILON) break
        if (curv < 0) {
          if (i === 0) xsupi = dscal(dri0 / -curv, b)
          break
        }
        const alphai = dri0 / curv
        xsupi = daxpy(alphai, psupi, xsupi)
        ri = daxpy(alphai, Ap, ri)
        const dri1 = ddot(ri, ri)
        psupi = daxpy(dri1 / dri0, psupi, dscal(-1, ri))
        i += 1
        dri0 = dri1
      }
      if (k2 === cgMax) {
        return {
          ...s,
          nhev,
          cgIterations: k2,
          stop: { status: 'failed', message: "CG iterations didn't converge: the Hessian is not positive definite" },
        }
      }
      const pk = xsupi
      const ls = search(lineFunction(obj, s.x, pk), {
        value: s.value,
        slope: ddot(s.grad, pk),
        ...(s.previousValue === undefined ? {} : { previousValue: s.previousValue }),
      })
      if (!ls.success) {
        return {
          ...s,
          nfev: s.nfev + ls.evaluations,
          ngev: s.ngev + ls.gradEvaluations,
          nhev,
          cgIterations: i,
          lineSearch: ls,
          stop: { status: 'failed', message: 'desired error not necessarily achieved due to precision loss' },
        }
      }
      const update = dscal(ls.alpha, pk)
      const x = daxpy(1, update, s.x)
      let grad = ls.grad
      let ngev = s.ngev + ls.gradEvaluations
      if (grad === undefined) {
        grad = obj.valueAndGrad(x).grad
        ngev += 1
      }
      const l1 = dasum(update)
      const done = l1 <= xtol || (gtol !== undefined && infNorm(grad) <= gtol)
      return {
        x,
        value: ls.value,
        grad,
        nfev: s.nfev + ls.evaluations,
        ngev,
        nhev,
        step: dnrm2(update),
        previousValue: s.value,
        update: l1,
        cgIterations: i,
        lineSearch: ls,
        ...(done
          ? {
              stop:
                Number.isNaN(ls.value) || Number.isNaN(l1)
                  ? { status: 'diverged' as const, message: 'NaN result encountered' }
                  : { status: 'converged' as const, message: 'optimization terminated successfully' },
            }
          : {}),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { update: s.update, cgIterations: s.cgIterations },
        ...(s.lineSearch === undefined ? {} : { messages: [lineSearchMessage(s.lineSearch)] }),
      }),
  }
}

/**
 * Minimise by Newton-CG, a port of scipy.optimize.minimize with method 'Newton-CG' (Hessian-vector products by
 * autodiff).
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - xtol (default 1e-5), line search constants, Hessian mode, iteration cap (default 200 n) and run
 * options.
 * @returns The result.
 * @concept newton-method
 * @example
 * ```ts
 * import { newtonCg } from 'mdml-engine/optim'
 * import { square, sub, sum, type Value } from 'mdml-engine/tensor'
 * newtonCg((x: Value) => sum(square(sub(x, 3))), [0, 1]).converged // => true
 * ```
 */
export function newtonCg(f: Objective, x0: Parameters, options: NewtonCgOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(newtonCgOn(obj, options), obj, options, 200 * obj.n)
}
