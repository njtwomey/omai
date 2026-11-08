/**
 * Constraints in scipy's form (equalities and inequalities on the parameters, with Jacobians by autodiff) and the
 * projected gradient method for minimising over a convex set given by its projection, with spectral (Barzilai-Borwein)
 * steps and a nonmonotone line search (SPG2), as a traced algorithm.
 */

import { daxpy, ddot, dnrm2 } from '../../numerics/linalg/index.ts'
import { concat, reshape, type Value } from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import {
  type Bounds,
  type BoundVectors,
  boxOf,
  clipToBox,
  type FlatObjective,
  flatObjective,
  flatResiduals,
  infNorm,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type ResidualFunction,
  type VectorMethodOptions,
} from './objective.ts'

/** A constraint in scipy's form: c(x) (a scalar or a vector) = 0 for 'eq' and >= 0 for 'ineq'. */
export interface Constraint {
  readonly type: 'eq' | 'ineq'
  readonly fun: ResidualFunction
}

/** The constraints of a problem on the flat vector: values and Jacobians of the equalities and inequalities. */
export interface ConstraintSet {
  /** Number of equality and inequality components. */
  readonly me: number
  readonly mi: number
  /** The equality values h(x) and inequality values c(x) at a flat point. */
  values(x: ArrayLike<number>): { readonly eq: Float64Array; readonly ineq: Float64Array }
  /** Their Jacobians (me x n and mi x n, row-major). */
  jacobians(x: ArrayLike<number>): { readonly eq: Float64Array; readonly ineq: Float64Array }
  /** The constraint functions of a flat value, for building differentiable penalties. */
  readonly ofValue: (x: Value) => { readonly eq: Value | undefined; readonly ineq: Value | undefined }
}

/**
 * Collect constraints on the flattened parameters (Jacobians by autodiff).
 *
 * @param constraints - The constraints.
 * @param x0 - A point, fixing the parameter structure.
 * @returns The constraint set.
 * @concept constrained-optimisation
 * @example
 * ```ts
 * import { constraintSet } from 'omai-engine/optim'
 * import { sub, sum, type Value } from 'omai-engine/tensor'
 * const cs = constraintSet([{ type: 'eq', fun: (x: Value) => sub(sum(x), 1) }, { type: 'ineq', fun: (x: Value) => x }], [0.5, 0.5])
 * [cs.me, cs.mi, Array.from(cs.values([0.2, 0.3]).eq)] // => [1, 2, [-0.5]]
 * ```
 */
export function constraintSet(constraints: readonly Constraint[], x0: Parameters): ConstraintSet {
  const eqs = constraints.filter((c) => c.type === 'eq').map((c) => flatResiduals(c.fun, x0))
  const ins = constraints.filter((c) => c.type === 'ineq').map((c) => flatResiduals(c.fun, x0))
  const me = eqs.reduce((s, r) => s + r.m, 0)
  const mi = ins.reduce((s, r) => s + r.m, 0)
  const stackValues = (rs: typeof eqs, x: ArrayLike<number>, m: number): Float64Array => {
    const out = new Float64Array(m)
    let k = 0
    for (const r of rs) {
      out.set(r.residuals(x), k)
      k += r.m
    }
    return out
  }
  const stackJacobians = (rs: typeof eqs, x: ArrayLike<number>, m: number): Float64Array => {
    const n = x.length
    const out = new Float64Array(m * n)
    let k = 0
    for (const r of rs) {
      out.set(r.jacobian(x), k * n)
      k += r.m
    }
    return out
  }
  const ofValue = (rs: typeof eqs, v: Value): Value | undefined =>
    rs.length === 0 ? undefined : concat(rs.map((r) => reshape(r.ofValue(v), [-1])))
  return {
    me,
    mi,
    values: (x) => ({ eq: stackValues(eqs, x, me), ineq: stackValues(ins, x, mi) }),
    jacobians: (x) => ({ eq: stackJacobians(eqs, x, me), ineq: stackJacobians(ins, x, mi) }),
    ofValue: (v) => ({ eq: ofValue(eqs, v), ineq: ofValue(ins, v) }),
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Projected gradient

/** Options of {@link projectedGradient}. */
export interface ProjectedGradientOptions extends VectorMethodOptions {
  /** Box bounds (the feasible set when no `project` is given). */
  readonly bounds?: Bounds
  /** The Euclidean projection onto the feasible set, on flat vectors (for example a simplex projection). */
  readonly project?: (x: Float64Array) => Float64Array
  /**
   * The step: `spectral` (default, the spectral projected gradient of Birgin, Martinez and Raydan with a nonmonotone
   * line search), `armijo` (backtracking along the projection arc) or `fixed`.
   */
  readonly step?: 'spectral' | 'armijo' | 'fixed'
  /** The fixed step (default 1e-2). */
  readonly learningRate?: number
  /** Memory of the nonmonotone line search (default 10). */
  readonly memory?: number
}

/** The state of a {@link projectedGradient} run. */
export interface ProjectedGradientState extends VectorState {
  /** The spectral step length. */
  readonly lambda: number
  /** Recent objective values, for the nonmonotone test. */
  readonly recent: readonly number[]
  /** The infinity norm of P(x - g) - x, the optimality measure. */
  readonly projectedGradNorm: number
}

/**
 * The projected gradient method as a traced algorithm: steps along P(x - lambda g) - x with the spectral
 * (Barzilai-Borwein) step and Grippo-Lampariello-Lucidi nonmonotone line search (SPG2), Armijo backtracking along the
 * projection arc, or a fixed step. Stops when |P(x - g) - x|_inf <= gradTol (default 1e-6).
 *
 * @param f - The objective.
 * @param x0 - The start (projected onto the feasible set).
 * @param options - Bounds or a projection, the step rule and its parameters.
 * @returns The algorithm.
 * @concept constrained-optimisation
 * @reference E. G. Birgin, J. M. Martinez and M. Raydan, Nonmonotone spectral projected gradient methods on convex
 * sets, SIAM J. Optim. 10 (2000).
 * @example
 * ```ts
 * import { projectedGradientAlgorithm } from 'omai-engine/optim'
 * import { square, sum, type Value } from 'omai-engine/tensor'
 * projectedGradientAlgorithm((x: Value) => sum(square(x)), [1, 2], { bounds: { lower: 0.5 } }).name // => 'projectedGradient'
 * ```
 */
export function projectedGradientAlgorithm(
  f: Objective,
  x0: Parameters,
  options: ProjectedGradientOptions = {},
): Algorithm<ProjectedGradientState> {
  return projectedGradientOn(flatObjective(f, x0), options)
}

function projectedGradientOn(obj: FlatObjective, options: ProjectedGradientOptions): Algorithm<ProjectedGradientState> {
  const box: BoundVectors = boxOf(options.bounds, obj.n)
  const project = options.project ?? ((x: Float64Array) => clipToBox(x, box))
  const gtol = options.gradTol ?? 1e-6
  const rule = options.step ?? 'spectral'
  const lr = options.learningRate ?? 1e-2
  const memory = options.memory ?? 10
  const lmin = 1e-30
  const lmax = 1e30
  const pgNorm = (x: Float64Array, g: Float64Array): number => infNorm(daxpy(-1, x, project(daxpy(-1, g, x))))
  return {
    name: 'projectedGradient',
    config: configOf(options),
    schema: vectorSchema({
      metrics: {
        projectedGradNorm: { description: 'infinity norm of P(x - g) - x' },
        lambda: { description: 'spectral step length' },
      },
    }),
    init: () => {
      const x = project(Float64Array.from(obj.x0))
      const { value, grad } = obj.valueAndGrad(x)
      const pg = pgNorm(x, grad)
      return {
        x,
        value,
        grad,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        step: 0,
        lambda: rule === 'fixed' ? lr : Math.min(lmax, Math.max(lmin, 1 / Math.max(pg, Number.MIN_VALUE))),
        recent: [value],
        projectedGradNorm: pg,
        ...(pg <= gtol
          ? { stop: { status: 'converged' as const, message: 'projected gradient below tolerance' } }
          : {}),
      }
    },
    step: (s) => {
      const lambda = rule === 'armijo' ? Math.max(s.lambda, 1) : s.lambda
      let nfev = s.nfev
      let next: { x: Float64Array; value: number; grad: Float64Array } | undefined
      if (rule === 'fixed') {
        const x = project(daxpy(-lambda, s.grad, s.x))
        const ev = obj.valueAndGrad(x)
        nfev += 1
        next = { x, value: ev.value, grad: ev.grad }
      } else if (rule === 'spectral') {
        const d = daxpy(-1, s.x, project(daxpy(-lambda, s.grad, s.x)))
        const gd = ddot(s.grad, d)
        let fmax = -Infinity
        for (const v of s.recent) fmax = Math.max(fmax, v)
        let alpha = 1
        for (let k = 0; k < 60; k++) {
          const x = daxpy(alpha, d, s.x)
          const v = obj.value(x)
          nfev += 1
          if (v <= fmax + 1e-4 * alpha * gd) {
            const ev = obj.valueAndGrad(x)
            next = { x, value: ev.value, grad: ev.grad }
            break
          }
          const at = (-0.5 * alpha * alpha * gd) / (v - s.value - alpha * gd)
          alpha = at >= 0.1 && at <= 0.9 * alpha ? at : alpha / 2
        }
      } else {
        // Armijo along the projection arc x(t) = P(x - t g).
        let t = lambda
        for (let k = 0; k < 60; k++) {
          const x = project(daxpy(-t, s.grad, s.x))
          const v = obj.value(x)
          nfev += 1
          if (v <= s.value + 1e-4 * ddot(s.grad, daxpy(-1, s.x, x))) {
            const ev = obj.valueAndGrad(x)
            next = { x, value: ev.value, grad: ev.grad }
            break
          }
          t /= 2
        }
      }
      if (next === undefined) return { ...s, nfev, stop: { status: 'failed', message: 'line search failed' } }
      const sv = daxpy(-1, s.x, next.x)
      const y = daxpy(-1, s.grad, next.grad)
      const sy = ddot(sv, y)
      const newLambda = rule === 'fixed' ? lr : sy <= 0 ? lmax : Math.min(lmax, Math.max(lmin, ddot(sv, sv) / sy))
      const pg = pgNorm(next.x, next.grad)
      return {
        x: next.x,
        value: next.value,
        grad: next.grad,
        nfev,
        ngev: s.ngev + 1,
        nhev: 0,
        step: dnrm2(sv),
        lambda: newLambda,
        recent: [...s.recent, next.value].slice(-memory),
        projectedGradNorm: pg,
        ...(pg <= gtol
          ? { stop: { status: 'converged' as const, message: 'projected gradient below tolerance' } }
          : {}),
      }
    },
    done: vectorDone,
    observe: (s) => observeVector(s, { metrics: { projectedGradNorm: s.projectedGradNorm, lambda: s.lambda } }),
  }
}

/**
 * Minimise over a convex set by projected gradients (spectral by default).
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Bounds or a projection, step rule, tolerance (default 1e-6), iteration cap (default 10000) and run
 * options.
 * @returns The result.
 * @concept constrained-optimisation
 * @example
 * ```ts
 * import { projectedGradient } from 'omai-engine/optim'
 * import { square, sub, sum, tensor, toFlat, type Value } from 'omai-engine/tensor'
 * const r = projectedGradient((x: Value) => sum(square(sub(x, tensor([2, -1])))), [0, 0], { bounds: { lower: 0, upper: 1 } })
 * toFlat(r.x as Value) // => [1, 0]
 * ```
 */
export function projectedGradient(
  f: Objective,
  x0: Parameters,
  options: ProjectedGradientOptions = {},
): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(projectedGradientOn(obj, options), obj, options, 10000)
}
