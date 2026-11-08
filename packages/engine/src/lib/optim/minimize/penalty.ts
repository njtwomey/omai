/**
 * Constrained minimisation by sequences of unconstrained problems: the quadratic penalty method, the logarithmic
 * barrier method and the augmented Lagrangian method (with multipliers in scipy's sign convention), each an outer
 * traced algorithm that runs an inner local method as a child run.
 */

import { OmaiError } from '../../base/index.ts'
import { daxpy, ddot, dnrm2 } from '../../numerics/linalg/index.ts'
import {
  add,
  fromStorage,
  log,
  maximum,
  mul,
  neg,
  square,
  sub,
  sum,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import type { Algorithm, StepContext } from '../../trace/index.ts'
import {
  configOf,
  observeVector,
  runVector,
  runVectorState,
  type VectorState,
  vectorDone,
  vectorSchema,
} from './_shared.ts'
import { type Constraint, constraintSet } from './constrained.ts'
import { lbfgsb } from './lbfgsb.ts'
import { newton } from './newton.ts'
import {
  type Bounds,
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

/** The largest constraint violation: max |h_i| and max(0, -c_j) (and of the bounds). */
function violation(v: { readonly eq: Float64Array; readonly ineq: Float64Array }): number {
  let m = 0
  for (const h of v.eq) m = Math.max(m, Math.abs(h))
  for (const c of v.ineq) m = Math.max(m, -c)
  return m
}

/** A local method used by the outer loops, as a function value with the run options of a child run. */
export type InnerMethod = (
  f: Objective,
  x0: Parameters,
  options?: VectorMethodOptions & { readonly bounds?: Bounds },
) => MinimizeResult

const defaultInner: InnerMethod = (f, x0, options = {}) => lbfgsb(f, x0, { gradTol: 1e-8, ftol: 1e-12, ...options })

function inner(
  method: InnerMethod,
  f: (v: Value) => Value,
  x: Float64Array,
  options: VectorMethodOptions & { readonly bounds?: Bounds },
  ctx: StepContext,
): { x: Float64Array; fun: number; nfev: number; ngev: number; steps: number } {
  const r = method(f as Objective, Float64Array.from(x), { ...options, parent: ctx.run, key: ctx.key })
  return {
    x: Float64Array.from(toFlat(r.x as Value)),
    fun: r.fun,
    nfev: r.nfev,
    ngev: r.ngev,
    steps: r.steps,
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Penalty, barrier and augmented Lagrangian methods

/** The state of the outer loops of the penalty, barrier and augmented Lagrangian methods. */
export interface OuterState extends VectorState {
  /** The penalty parameter (penalty and augmented Lagrangian) or barrier parameter t. */
  readonly mu: number
  /** The largest constraint violation (or, for the barrier, the duality-gap bound). */
  readonly violation: number
  /** Multipliers of the equalities and inequalities (augmented Lagrangian and barrier estimates). */
  readonly multipliers: Float64Array
  /** Iterations of the last inner solve. */
  readonly innerSteps: number
}

/** Options shared by the penalty-type methods. */
export interface PenaltyOptions extends VectorMethodOptions {
  readonly constraints?: readonly Constraint[]
  /** Box bounds, passed to the inner method (L-BFGS-B by default handles them directly). */
  readonly bounds?: Bounds
  /** The initial penalty parameter (default 1 for the penalty method, 10 for the augmented Lagrangian). */
  readonly mu?: number
  /** The factor by which mu grows (default 10). */
  readonly growth?: number
  /** Stop when the constraint violation is at most this (default 1e-6). */
  readonly tol?: number
  /** The inner solver (default L-BFGS-B with tight tolerances). */
  readonly inner?: InnerMethod
}

const OUTER_SCHEMA = vectorSchema({
  metrics: {
    mu: { description: 'penalty or barrier parameter' },
    violation: { description: 'largest constraint violation (barrier: duality-gap bound)' },
    innerSteps: { description: 'iterations of the inner solve' },
  },
})

const observeOuter = (s: OuterState) =>
  observeVector(s, { metrics: { mu: s.mu, violation: s.violation, innerSteps: s.innerSteps } })

/**
 * The quadratic penalty method as a traced algorithm: each step minimises f + mu/2 (|h|^2 + |min(c, 0)|^2) by the inner
 * method (a child run) from the last point, then multiplies mu by `growth`, until the violation is at most `tol`.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Constraints, bounds, mu, growth, tolerance and the inner method.
 * @returns The algorithm.
 * @concept constrained-optimisation
 * @example
 * ```ts
 * import { penaltyMethodAlgorithm } from 'omai-engine/optim'
 * import { square, sub, sum, type Value } from 'omai-engine/tensor'
 * penaltyMethodAlgorithm((x: Value) => sum(square(x)), [1, 1], { constraints: [{ type: 'eq', fun: (x: Value) => sub(sum(x), 1) }] }).name // => 'penaltyMethod'
 * ```
 */
export function penaltyMethodAlgorithm(
  f: Objective,
  x0: Parameters,
  options: PenaltyOptions = {},
): Algorithm<OuterState> {
  const obj = flatObjective(f, x0)
  const cs = constraintSet(options.constraints ?? [], x0)
  const growth = options.growth ?? 10
  const tol = options.tol ?? 1e-6
  const method = options.inner ?? defaultInner
  const penalised =
    (mu: number) =>
    (v: Value): Value => {
      const { eq, ineq } = cs.ofValue(v)
      let p = obj.ofValue(v)
      if (eq !== undefined) p = add(p, mul(sum(square(eq)), mu / 2))
      if (ineq !== undefined) p = add(p, mul(sum(square(maximum(neg(ineq), 0))), mu / 2))
      return p
    }
  return {
    name: 'penaltyMethod',
    config: configOf(options),
    schema: OUTER_SCHEMA,
    init: () => {
      const x = Float64Array.from(obj.x0)
      const { value, grad } = obj.valueAndGrad(x)
      const vio = violation(cs.values(x))
      return {
        x,
        value,
        grad,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        step: 0,
        mu: options.mu ?? 1,
        violation: vio,
        multipliers: new Float64Array(cs.me + cs.mi),
        innerSteps: 0,
      }
    },
    step: (s, ctx) => {
      const r = inner(method, penalised(s.mu), s.x, options.bounds === undefined ? {} : { bounds: options.bounds }, ctx)
      const vals = cs.values(r.x)
      const vio = violation(vals)
      const { value, grad } = obj.valueAndGrad(r.x)
      // First-order multiplier estimates: lambda = mu h, nu = mu max(0, -c).
      const mult = new Float64Array(cs.me + cs.mi)
      for (let i = 0; i < cs.me; i++) mult[i] = s.mu * (vals.eq[i] ?? 0)
      for (let j = 0; j < cs.mi; j++) mult[cs.me + j] = s.mu * Math.max(0, -(vals.ineq[j] ?? 0))
      return {
        x: r.x,
        value,
        grad,
        nfev: s.nfev + r.nfev + 1,
        ngev: s.ngev + r.ngev + 1,
        nhev: 0,
        step: dnrm2(daxpy(-1, s.x, r.x)),
        mu: s.mu * growth,
        violation: vio,
        multipliers: mult,
        innerSteps: r.steps,
        ...(vio <= tol
          ? { stop: { status: 'converged' as const, message: 'constraint violation below tolerance' } }
          : {}),
      }
    },
    done: vectorDone,
    observe: observeOuter,
  }
}

/**
 * Minimise subject to constraints by the quadratic penalty method.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Constraints, bounds, mu (default 1), growth (default 10), tolerance (default 1e-6), the inner method,
 * outer iteration cap (default 30) and run options.
 * @returns The result.
 * @concept constrained-optimisation
 * @example
 * ```ts
 * import { penaltyMethod } from 'omai-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'omai-engine/tensor'
 * const r = penaltyMethod((x: Value) => sum(square(x)), [1, 1], { constraints: [{ type: 'eq', fun: (x: Value) => sub(sum(x), 1) }] })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e4) / 1e4) // => [0.5, 0.5]
 * ```
 */
export function penaltyMethod(f: Objective, x0: Parameters, options: PenaltyOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(penaltyMethodAlgorithm(f, x0, options), obj, options, 30)
}

/** Options of {@link barrierMethod}. */
export interface BarrierOptions extends VectorMethodOptions {
  /** Inequality constraints c(x) >= 0 (the start must satisfy them strictly). */
  readonly constraints?: readonly Constraint[]
  /** The initial barrier parameter t (default 1). */
  readonly t?: number
  /** The factor by which t grows (default 10). */
  readonly growth?: number
  /** Stop when the duality-gap bound m / t is at most this (default 1e-8). */
  readonly tol?: number
  /** The inner solver (default Newton's method with backtracking, which stays in the domain). */
  readonly inner?: InnerMethod
}

/**
 * The logarithmic barrier method as a traced algorithm (Boyd and Vandenberghe, 11.3): each step minimises
 * t f - sum log c_j by the inner method from the last (strictly feasible) point, then multiplies t by `growth`; m / t
 * bounds the suboptimality, and -1 / (t c_j) estimates the multipliers.
 *
 * @param f - The objective.
 * @param x0 - A strictly feasible start.
 * @param options - Inequality constraints, t, growth, tolerance and the inner method.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for equality constraints or an infeasible start.
 * @concept constrained-optimisation
 * @example
 * ```ts
 * import { barrierMethodAlgorithm } from 'omai-engine/optim'
 * import { square, sub, sum, type Value } from 'omai-engine/tensor'
 * barrierMethodAlgorithm((x: Value) => sum(square(x)), [2], { constraints: [{ type: 'ineq', fun: (x: Value) => sub(x, 1) }] }).name // => 'barrierMethod'
 * ```
 */
export function barrierMethodAlgorithm(
  f: Objective,
  x0: Parameters,
  options: BarrierOptions = {},
): Algorithm<OuterState> {
  const obj = flatObjective(f, x0)
  const constraints = options.constraints ?? []
  if (constraints.some((c) => c.type !== 'ineq'))
    throw new OmaiError('value', 'barrierMethod takes inequality constraints only')
  const cs = constraintSet(constraints, x0)
  const growth = options.growth ?? 10
  const tol = options.tol ?? 1e-8
  const method: InnerMethod = options.inner ?? ((g, x, o = {}) => newton(g, x, { gradTol: 1e-9, ...o }))
  const barrier =
    (t: number) =>
    (v: Value): Value => {
      const { ineq } = cs.ofValue(v)
      const fv = mul(obj.ofValue(v), t)
      return ineq === undefined ? fv : sub(fv, sum(log(ineq)))
    }
  return {
    name: 'barrierMethod',
    config: configOf(options),
    schema: OUTER_SCHEMA,
    init: () => {
      const x = Float64Array.from(obj.x0)
      const c = cs.values(x).ineq
      if (c.some((v) => !(v > 0))) throw new OmaiError('value', 'barrierMethod: the start must be strictly feasible')
      const { value, grad } = obj.valueAndGrad(x)
      const t = options.t ?? 1
      return {
        x,
        value,
        grad,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        step: 0,
        mu: t,
        violation: cs.mi / t,
        multipliers: new Float64Array(cs.mi),
        innerSteps: 0,
      }
    },
    step: (s, ctx) => {
      const r = inner(method, barrier(s.mu), s.x, {}, ctx)
      const c = cs.values(r.x).ineq
      const { value, grad } = obj.valueAndGrad(r.x)
      const gap = cs.mi / s.mu
      return {
        x: r.x,
        value,
        grad,
        nfev: s.nfev + r.nfev + 1,
        ngev: s.ngev + r.ngev + 1,
        nhev: 0,
        step: dnrm2(daxpy(-1, s.x, r.x)),
        mu: s.mu * growth,
        violation: gap,
        multipliers: c.map((cj) => 1 / (s.mu * cj)),
        innerSteps: r.steps,
        ...(gap <= tol ? { stop: { status: 'converged' as const, message: 'duality-gap bound below tolerance' } } : {}),
      }
    },
    done: vectorDone,
    observe: observeOuter,
  }
}

/**
 * Minimise subject to inequality constraints by the logarithmic barrier method.
 *
 * @param f - The objective.
 * @param x0 - A strictly feasible start.
 * @param options - Inequality constraints, t (default 1), growth (default 10), tolerance (default 1e-8), inner method,
 * outer iteration cap (default 30) and run options.
 * @returns The result.
 * @concept constrained-optimisation
 * @example
 * ```ts
 * import { barrierMethod } from 'omai-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'omai-engine/tensor'
 * const r = barrierMethod((x: Value) => sum(square(x)), [2], { constraints: [{ type: 'ineq', fun: (x: Value) => sub(x, 1) }] })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [1]
 * ```
 */
export function barrierMethod(f: Objective, x0: Parameters, options: BarrierOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(barrierMethodAlgorithm(f, x0, options), obj, options, 30)
}

/**
 * The augmented Lagrangian method (Powell-Hestenes-Rockafellar, with the shifted penalty for inequalities) as a traced
 * algorithm: each step minimises
 * f + sum (lambda_i h_i + mu / 2 h_i^2) + sum (max(0, nu_j - mu c_j)^2 - nu_j^2) / (2 mu) by the inner method,
 * updates lambda += mu h and nu = max(0, nu - mu c), and grows mu when the violation does not fall by a quarter.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Constraints, bounds, mu, growth, tolerance and the inner method.
 * @returns The algorithm.
 * @concept constrained-optimisation
 * @reference J. Nocedal and S. J. Wright, Numerical Optimization (2006), chapter 17.
 * @example
 * ```ts
 * import { augmentedLagrangianAlgorithm } from 'omai-engine/optim'
 * import { square, sub, sum, type Value } from 'omai-engine/tensor'
 * augmentedLagrangianAlgorithm((x: Value) => sum(square(x)), [1, 1], { constraints: [{ type: 'eq', fun: (x: Value) => sub(sum(x), 1) }] }).name // => 'augmentedLagrangian'
 * ```
 */
export function augmentedLagrangianAlgorithm(
  f: Objective,
  x0: Parameters,
  options: PenaltyOptions = {},
): Algorithm<OuterState> {
  const obj = flatObjective(f, x0)
  const cs = constraintSet(options.constraints ?? [], x0)
  const growth = options.growth ?? 10
  const tol = options.tol ?? 1e-6
  const method = options.inner ?? defaultInner
  const lagrangian =
    (mu: number, lambda: Float64Array, nu: Float64Array) =>
    (v: Value): Value => {
      const { eq, ineq } = cs.ofValue(v)
      let L = obj.ofValue(v)
      if (eq !== undefined) L = add(L, add(sum(mul(tensor(Array.from(lambda)), eq)), mul(sum(square(eq)), mu / 2)))
      if (ineq !== undefined) {
        const nuT = tensor(Array.from(nu))
        const shifted = maximum(sub(nuT, mul(ineq, mu)), 0)
        L = add(L, mul(sub(sum(square(shifted)), ddot(nu, nu)), 1 / (2 * mu)))
      }
      return L
    }
  return {
    name: 'augmentedLagrangian',
    config: configOf(options),
    schema: OUTER_SCHEMA,
    init: () => {
      const x = Float64Array.from(obj.x0)
      const { value, grad } = obj.valueAndGrad(x)
      return {
        x,
        value,
        grad,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        step: 0,
        mu: options.mu ?? 10,
        violation: violation(cs.values(x)),
        multipliers: new Float64Array(cs.me + cs.mi),
        innerSteps: 0,
      }
    },
    step: (s, ctx) => {
      const lambda = s.multipliers.subarray(0, cs.me)
      const nu = s.multipliers.subarray(cs.me)
      const r = inner(
        method,
        lagrangian(s.mu, lambda, nu),
        s.x,
        options.bounds === undefined ? {} : { bounds: options.bounds },
        ctx,
      )
      const vals = cs.values(r.x)
      const mult = new Float64Array(cs.me + cs.mi)
      for (let i = 0; i < cs.me; i++) mult[i] = (lambda[i] ?? 0) + s.mu * (vals.eq[i] ?? 0)
      for (let j = 0; j < cs.mi; j++) mult[cs.me + j] = Math.max(0, (nu[j] ?? 0) - s.mu * (vals.ineq[j] ?? 0))
      // The violation of the shifted inequalities: |min(c, nu / mu)|.
      let vio = 0
      for (const h of vals.eq) vio = Math.max(vio, Math.abs(h))
      for (let j = 0; j < cs.mi; j++) vio = Math.max(vio, Math.abs(Math.min(vals.ineq[j] ?? 0, (nu[j] ?? 0) / s.mu)))
      const { value, grad } = obj.valueAndGrad(r.x)
      const mu = vio > 0.25 * s.violation ? s.mu * growth : s.mu
      return {
        x: r.x,
        value,
        grad,
        nfev: s.nfev + r.nfev + 1,
        ngev: s.ngev + r.ngev + 1,
        nhev: 0,
        step: dnrm2(daxpy(-1, s.x, r.x)),
        mu,
        violation: vio,
        multipliers: mult,
        innerSteps: r.steps,
        ...(vio <= tol && dnrm2(daxpy(-1, s.x, r.x)) <= Math.sqrt(tol) * (1 + dnrm2(r.x))
          ? { stop: { status: 'converged' as const, message: 'constraint violation below tolerance' } }
          : {}),
      }
    },
    done: vectorDone,
    observe: observeOuter,
  }
}

/**
 * Minimise subject to constraints by the augmented Lagrangian method (multipliers reported in scipy's sign
 * convention for c(x) >= 0: nu >= 0). The multipliers are a tensor (migration, D-009: a `Float64Array`).
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Constraints, bounds, mu (default 10), growth (default 10), tolerance (default 1e-6), inner method,
 * outer iteration cap (default 50) and run options.
 * @returns The result with the final multipliers, a `[me + mi]` tensor (equalities first).
 * @concept constrained-optimisation
 * @example
 * ```ts
 * import { augmentedLagrangian } from 'omai-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'omai-engine/tensor'
 * const r = augmentedLagrangian((x: Value) => sum(square(x)), [1, 1], { constraints: [{ type: 'eq', fun: (x: Value) => sub(sum(x), 1) }] })
 * [toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6), Math.round(toFlat(r.multipliers)[0] * 1e4) / 1e4] // => [[0.5, 0.5], -1]
 * ```
 */
export function augmentedLagrangian(
  f: Objective,
  x0: Parameters,
  options: PenaltyOptions = {},
): MinimizeResult & { readonly multipliers: Tensor<'float64', [number]> } {
  const obj = flatObjective(f, x0)
  const { result, state } = runVectorState(augmentedLagrangianAlgorithm(f, x0, options), obj, options, 50)
  return { ...result, multipliers: fromStorage(state.multipliers, [state.multipliers.length], 'float64') }
}
