/**
 * Gradient-based minimisers as traced algorithms: gradient descent with line searches, fixed rates or Barzilai-Borwein
 * steps, nonlinear conjugate gradients (a port of scipy's CG, with the Fletcher-Reeves, Polak-Ribiere, Hestenes-Stiefel
 * and Dai-Yuan choices), and coordinate descent with cyclic, random or greedy order.
 */

import { MdmlError } from '../../base/index.ts'
import { daxpy, ddot, dnrm2, dscal, idamax } from '../../numerics/linalg/index.ts'
import { permutation } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { rethrowUnless } from '../_catch.ts'
import { defaultGradTol } from '../_knobs.ts'
import { backtracking, type LineSearch, type LineSearchResult, lineFunction } from '../line-search.ts'
import { wolfe12 } from '../line-search-interval.ts'
import { brent } from '../scalar.ts'
import {
  configOf,
  gradientThreshold,
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

const converged = { status: 'converged' as const, message: 'gradient norm below tolerance' }

/** Options of {@link gradientDescent}. */
export interface GradientDescentOptions extends VectorMethodOptions {
  /**
   * How the step is chosen: a line search along -g (default, Armijo backtracking), a fixed learning rate, or the
   * Barzilai-Borwein steps s's / s'y (`bb1`) and s'y / y'y (`bb2`).
   */
  readonly step?: 'line-search' | 'fixed' | 'bb1' | 'bb2'
  /** The fixed step, and the first step of the Barzilai-Borwein rules (default 1e-3). */
  readonly learningRate?: number
  /** The line search (default safeguarded Armijo backtracking, starting from the previous accepted step times 2). */
  readonly lineSearch?: LineSearch
}

/** The state of a {@link gradientDescent} run. */
export interface GradientDescentState extends VectorState {
  readonly alpha: number
  readonly lineSearch?: LineSearchResult
}

/**
 * Gradient descent as a traced algorithm: x <- x - alpha g with alpha from a line search, a fixed learning rate or the
 * Barzilai-Borwein rules. Records the objective, gradient norm, step and line-search trials.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Step rule, learning rate, line search and the gradient tolerance (infinity norm, default 1e-5).
 * @returns The algorithm.
 * @concept gradient-descent
 * @example
 * ```ts
 * import { gradientDescentAlgorithm } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * gradientDescentAlgorithm((x: Value) => sum(square(x)), [1, 2]).name // => 'gradientDescent'
 * ```
 */
export function gradientDescentAlgorithm(
  f: Objective,
  x0: Parameters,
  options: GradientDescentOptions = {},
): Algorithm<GradientDescentState> {
  return gradientDescentOn(flatObjective(f, x0), options)
}

function gradientDescentOn(obj: FlatObjective, options: GradientDescentOptions): Algorithm<GradientDescentState> {
  const rule = options.step ?? 'line-search'
  const lr = options.learningRate ?? 1e-3
  const gtol = gradientThreshold(obj, options, defaultGradTol)
  const search = options.lineSearch ?? backtracking({ safeguard: [0.1, 0.5] })
  return {
    name: 'gradientDescent',
    config: configOf(options),
    schema: vectorSchema({ metrics: { alpha: { description: 'step size' } }, messages: LINE_SEARCH_CHANNEL }),
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
        alpha: lr,
        ...(infNorm(grad) <= gtol ? { stop: converged } : {}),
      }
    },
    step: (s) => {
      const p = dscal(-1, s.grad)
      if (rule === 'line-search') {
        const alpha0 = s.step === 0 ? Math.min(1, 1 / dnrm2(p)) : Math.min(2 * s.alpha, 1e10)
        const ls = search(lineFunction(obj, s.x, p), { value: s.value, slope: -ddot(p, p), alpha: alpha0 })
        if (!ls.success) {
          return {
            ...s,
            nfev: s.nfev + ls.evaluations,
            ngev: s.ngev + ls.gradEvaluations,
            lineSearch: ls,
            stop: { status: 'failed', message: `line search failed: ${ls.message}` },
          }
        }
        const x = daxpy(ls.alpha, p, s.x)
        const ev = ls.grad === undefined ? obj.valueAndGrad(x) : { value: ls.value, grad: ls.grad }
        const extra = ls.grad === undefined ? 1 : 0
        return {
          x,
          value: ev.value,
          grad: ev.grad,
          nfev: s.nfev + ls.evaluations + extra,
          ngev: s.ngev + ls.gradEvaluations + extra,
          nhev: 0,
          step: ls.alpha * dnrm2(p),
          alpha: ls.alpha,
          lineSearch: ls,
          ...(infNorm(ev.grad) <= gtol ? { stop: converged } : {}),
        }
      }
      const x = daxpy(s.alpha, p, s.x)
      const ev = obj.valueAndGrad(x)
      let alpha = s.alpha
      if (rule === 'bb1' || rule === 'bb2') {
        const sv = daxpy(-1, s.x, x)
        const y = daxpy(-1, s.grad, ev.grad)
        const sy = ddot(sv, y)
        const bb = rule === 'bb1' ? ddot(sv, sv) / sy : sy / ddot(y, y)
        alpha = Number.isFinite(bb) && bb > 0 ? bb : s.alpha
      }
      return {
        x,
        value: ev.value,
        grad: ev.grad,
        nfev: s.nfev + 1,
        ngev: s.ngev + 1,
        nhev: 0,
        step: s.alpha * dnrm2(p),
        alpha,
        ...(infNorm(ev.grad) <= gtol ? { stop: converged } : {}),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { alpha: s.alpha },
        ...(s.lineSearch === undefined ? {} : { messages: [lineSearchMessage(s.lineSearch)] }),
      }),
  }
}

/**
 * Minimise by gradient descent.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Step rule, learning rate, line search, gradient tolerance (default 1e-5), iteration cap (default
 * 10000) and run options.
 * @returns The result.
 * @concept gradient-descent
 * @example
 * ```ts
 * import { gradientDescent } from 'mdml-engine/optim'
 * import { square, sub, sum, type Value } from 'mdml-engine/tensor'
 * gradientDescent((x: Value) => sum(square(sub(x, 2))), [0, 0], { step: 'bb1' }).converged // => true
 * ```
 */
export function gradientDescent(f: Objective, x0: Parameters, options: GradientDescentOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(gradientDescentOn(obj, options), obj, options, 10000)
}

/** The choice of beta in nonlinear conjugate gradients. */
export type ConjugateGradientBeta =
  | 'polak-ribiere'
  | 'fletcher-reeves'
  | 'hestenes-stiefel'
  | 'dai-yuan'
  | 'hager-zhang'

/** Options of {@link conjugateGradient}. */
export interface ConjugateGradientOptions extends VectorMethodOptions {
  /** The update of the direction (default Polak-Ribiere+, as scipy's CG). */
  readonly beta?: ConjugateGradientBeta
  /** Line search constants (scipy's c1 = 1e-4, c2 = 0.4). */
  readonly c1?: number
  readonly c2?: number
  /** Restart with steepest descent every this many iterations (default never, as scipy). */
  readonly restart?: number
}

/** The state of a {@link conjugateGradient} run. */
export interface ConjugateGradientState extends VectorState {
  readonly direction: Float64Array
  readonly previousValue: number
  readonly beta: number
  readonly iterations: number
  readonly lineSearch?: LineSearchResult
}

function betaOf(kind: ConjugateGradientBeta, g: Float64Array, gNew: Float64Array, d: Float64Array): number {
  const y = daxpy(-1, g, gNew)
  const gg = ddot(g, g)
  switch (kind) {
    case 'polak-ribiere':
      return Math.max(0, ddot(y, gNew) / gg)
    case 'fletcher-reeves':
      return ddot(gNew, gNew) / gg
    case 'hestenes-stiefel':
      return Math.max(0, ddot(y, gNew) / ddot(d, y))
    case 'dai-yuan':
      return ddot(gNew, gNew) / ddot(d, y)
    case 'hager-zhang': {
      const dy = ddot(d, y)
      const t = daxpy((-2 * ddot(y, y)) / dy, d, y)
      const b = ddot(t, gNew) / dy
      // The lower bound of Hager and Zhang (2005) keeps the direction sufficiently descending.
      const eta = -1 / (dnrm2(d) * Math.min(0.01, dnrm2(g)))
      return Math.max(b, eta)
    }
  }
}

/**
 * Nonlinear conjugate gradients as a traced algorithm, a port of scipy's CG (Polak-Ribiere+ with the Wolfe line search
 * and its descent condition) with the Fletcher-Reeves, Hestenes-Stiefel, Dai-Yuan and Hager-Zhang updates as options.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - The beta rule, line search constants, restart period and gradient tolerance (default 1e-5).
 * @returns The algorithm.
 * @concept conjugate-gradient
 * @example
 * ```ts
 * import { conjugateGradientAlgorithm } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * conjugateGradientAlgorithm((x: Value) => sum(square(x)), [1, 2]).name // => 'conjugateGradient'
 * ```
 */
export function conjugateGradientAlgorithm(
  f: Objective,
  x0: Parameters,
  options: ConjugateGradientOptions = {},
): Algorithm<ConjugateGradientState> {
  return conjugateGradientOn(flatObjective(f, x0), options)
}

function conjugateGradientOn(obj: FlatObjective, options: ConjugateGradientOptions): Algorithm<ConjugateGradientState> {
  const gtol = gradientThreshold(obj, options, defaultGradTol)
  const kind = options.beta ?? 'polak-ribiere'
  const c1 = options.c1 ?? 1e-4
  const c2 = options.c2 ?? 0.4
  const sigma3 = 0.01
  return {
    name: 'conjugateGradient',
    config: configOf(options),
    schema: vectorSchema({
      metrics: { beta: { description: 'conjugacy coefficient of the last direction' } },
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
        direction: dscal(-1, grad),
        previousValue: value + dnrm2(grad) / 2,
        beta: 0,
        iterations: 0,
        ...(infNorm(grad) <= gtol ? { stop: converged } : {}),
      }
    },
    step: (s) => {
      const pk = s.direction
      const next = (gNew: Float64Array): { d: Float64Array; beta: number } => {
        const restart = options.restart !== undefined && (s.iterations + 1) % options.restart === 0
        const beta = restart ? 0 : betaOf(kind, s.grad, gNew, pk)
        return { d: daxpy(beta, pk, dscal(-1, gNew)), beta }
      }
      const search = wolfe12({
        c1,
        c2,
        alphaMin: 1e-100,
        alphaMax: 1e100,
        boundFallback: true,
        // scipy's descent condition: the next direction must descend (or the gradient already be small).
        extraCondition: (_alpha, point) => {
          if (point.grad === undefined) return true
          if (infNorm(point.grad) <= gtol) return true
          const { d } = next(point.grad)
          return ddot(d, point.grad) <= -sigma3 * ddot(point.grad, point.grad)
        },
      })
      const ls = search(lineFunction(obj, s.x, pk), {
        value: s.value,
        slope: ddot(s.grad, pk),
        previousValue: s.previousValue,
      })
      if (!ls.success) {
        return {
          ...s,
          nfev: s.nfev + ls.evaluations,
          ngev: s.ngev + ls.gradEvaluations,
          lineSearch: ls,
          stop: { status: 'failed', message: 'desired error not necessarily achieved due to precision loss' },
        }
      }
      const x = daxpy(ls.alpha, pk, s.x)
      const ev = ls.grad === undefined ? obj.valueAndGrad(x) : { value: ls.value, grad: ls.grad }
      const extra = ls.grad === undefined ? 1 : 0
      const { d, beta } = next(ev.grad)
      return {
        x,
        value: ev.value,
        grad: ev.grad,
        nfev: s.nfev + ls.evaluations + extra,
        ngev: s.ngev + ls.gradEvaluations + extra,
        nhev: 0,
        step: ls.alpha * dnrm2(pk),
        direction: d,
        previousValue: s.value,
        beta,
        iterations: s.iterations + 1,
        lineSearch: ls,
        ...(infNorm(ev.grad) <= gtol ? { stop: converged } : {}),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { beta: s.beta },
        ...(s.lineSearch === undefined ? {} : { messages: [lineSearchMessage(s.lineSearch)] }),
      }),
  }
}

/**
 * Minimise by nonlinear conjugate gradients (scipy.optimize.minimize with method 'CG' by default).
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - The beta rule, line search constants, restart period, gradient tolerance (default 1e-5), iteration
 * cap (default 200 n) and run options.
 * @returns The result.
 * @concept conjugate-gradient
 * @example
 * ```ts
 * import { conjugateGradient } from 'mdml-engine/optim'
 * import { square, sub, sum, type Value } from 'mdml-engine/tensor'
 * conjugateGradient((x: Value) => sum(square(sub(x, 2))), [0, 0, 0]).steps // => 1
 * ```
 */
export function conjugateGradient(
  f: Objective,
  x0: Parameters,
  options: ConjugateGradientOptions = {},
): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(conjugateGradientOn(obj, options), obj, options, 200 * obj.n)
}

/** Options of {@link coordinateDescent}. */
export interface CoordinateDescentOptions extends VectorMethodOptions {
  /** The order of coordinates in a sweep: cyclic (default), a random permutation per sweep, or greedy (Gauss-Southwell). */
  readonly order?: 'cyclic' | 'random' | 'greedy'
  /**
   * The update of one coordinate: `exact` minimises along it by Brent's method (default), `newton` takes
   * -g_i / H_ii (from a Hessian-vector product), `gradient` takes -learningRate g_i.
   */
  readonly update?: 'exact' | 'newton' | 'gradient'
  /** The step of the gradient update (default 1e-2). */
  readonly learningRate?: number
  /** Stop when a sweep changes the objective by at most ftol (|f| + 1) (default 1e-12). */
  readonly ftol?: number
}

/** The state of a {@link coordinateDescent} run (each step is a sweep over n coordinates). */
export interface CoordinateDescentState extends VectorState {
  readonly sweeps: number
  /** The largest coordinate change in the last sweep. */
  readonly maxChange: number
}

/**
 * Coordinate descent as a traced algorithm: each step is a sweep that updates the coordinates one at a time (cyclic,
 * random or greedy order) by exact one-dimensional minimisation, a coordinate Newton step or a gradient step. Random
 * orders draw from the step's key, so runs are reproducible.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Order, update rule, learning rate and tolerances.
 * @returns The algorithm.
 * @concept coordinate-descent
 * @example
 * ```ts
 * import { coordinateDescentAlgorithm } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * coordinateDescentAlgorithm((x: Value) => sum(square(x)), [1, 2]).name // => 'coordinateDescent'
 * ```
 */
export function coordinateDescentAlgorithm(
  f: Objective,
  x0: Parameters,
  options: CoordinateDescentOptions = {},
): Algorithm<CoordinateDescentState> {
  return coordinateDescentOn(flatObjective(f, x0), options)
}

function coordinateDescentOn(obj: FlatObjective, options: CoordinateDescentOptions): Algorithm<CoordinateDescentState> {
  const n = obj.n
  const order = options.order ?? 'cyclic'
  const update = options.update ?? 'exact'
  const lr = options.learningRate ?? 1e-2
  const gtol = gradientThreshold(obj, options, defaultGradTol)
  const ftol = options.ftol ?? 1e-12
  if (update === 'gradient' && !(lr > 0))
    throw new MdmlError('value', 'coordinateDescent: learningRate must be positive')
  return {
    name: 'coordinateDescent',
    config: configOf(options),
    schema: vectorSchema({ metrics: { maxChange: { description: 'largest coordinate change in the sweep' } } }),
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
        sweeps: 0,
        maxChange: 0,
        ...(infNorm(grad) <= gtol ? { stop: converged } : {}),
      }
    },
    step: (s, ctx) => {
      const x = Float64Array.from(s.x)
      let { nfev, ngev, nhev } = s
      let grad = s.grad
      let value = s.value
      let maxChange = 0
      const coords =
        order === 'random'
          ? toFlat(permutation(ctx.key, n))
          : order === 'cyclic'
            ? Array.from({ length: n }, (_, i) => i)
            : []
      const count = order === 'greedy' ? n : coords.length
      for (let k = 0; k < count; k++) {
        const i = order === 'greedy' ? idamax(grad) : (coords[k] ?? 0)
        const xi = x[i] ?? 0
        let next: number
        if (update === 'exact') {
          const along = (t: number): number => {
            const trial = Float64Array.from(x)
            trial[i] = t
            return obj.value(trial)
          }
          try {
            const r = brent(along, {
              bracket: [xi, xi - Math.sign(grad[i] ?? 0) * Math.max(1e-3, Math.abs(xi) * 1e-2 + 1e-3)],
            })
            nfev += r.nfev
            next = r.fun <= value ? r.x : xi
          } catch (error) {
            // No bracket along this coordinate (unbounded below, or flat): leave it for this sweep. An error of the
            // user's objective is not this and reaches the caller.
            rethrowUnless(error, 'convergence')
            next = xi
          }
        } else if (update === 'newton') {
          const e = new Float64Array(n)
          e[i] = 1
          const hii = obj.hvp(x, e)[i] ?? 0
          nhev += 1
          next = hii > 0 ? xi - (grad[i] ?? 0) / hii : xi - lr * (grad[i] ?? 0)
        } else next = xi - lr * (grad[i] ?? 0)
        maxChange = Math.max(maxChange, Math.abs(next - xi))
        x[i] = next
        const ev = obj.valueAndGrad(x)
        nfev += 1
        ngev += 1
        grad = ev.grad
        value = ev.value
      }
      const stop =
        infNorm(grad) <= gtol
          ? converged
          : Math.abs(s.value - value) <= ftol * (Math.abs(value) + 1)
            ? { status: 'converged' as const, message: 'objective change below ftol' }
            : undefined
      return {
        x,
        value,
        grad,
        nfev,
        ngev,
        nhev,
        step: dnrm2(daxpy(-1, s.x, x)),
        sweeps: s.sweeps + 1,
        maxChange,
        ...(stop === undefined ? {} : { stop }),
      }
    },
    done: vectorDone,
    observe: (s) => observeVector(s, { metrics: { maxChange: s.maxChange } }),
  }
}

/**
 * Minimise by coordinate descent.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Order, update rule, learning rate, tolerances, sweep cap (default 1000), key or seed and run
 * options.
 * @returns The result (`steps` counts sweeps).
 * @concept coordinate-descent
 * @example
 * ```ts
 * import { coordinateDescent } from 'mdml-engine/optim'
 * import { add, mul, square, sum, type Value } from 'mdml-engine/tensor'
 * // A separable quadratic: one sweep of exact coordinate minimisation solves it.
 * coordinateDescent((x: Value) => sum(add(square(x), mul(x, -2))), [0, 5], { update: 'newton' }).steps // => 1
 * ```
 */
export function coordinateDescent(
  f: Objective,
  x0: Parameters,
  options: CoordinateDescentOptions = {},
): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(coordinateDescentOn(obj, options), obj, options, 1000)
}
