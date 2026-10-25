/**
 * Trust-region minimisation, which minimises a quadratic model within a radius and grows or shrinks the radius by how
 * well the model predicted the actual reduction: the traced driver (scipy's `_minimize_trust_region`), exact or
 * Hessian-vector-product curvature from autodiff, and the dogleg, trust-ncg and trust-exact methods.
 */

import { type Json, WhiteboxError } from '../../base/index.ts'
import { daxpy, dgemv, dnrm2 } from '../../numerics/linalg/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import {
  type FlatObjective,
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'
import {
  doglegStep,
  exactStep,
  type QuadraticModel,
  quadraticModelValue,
  steihaugStep,
  type TrustStep,
  type TrustSubproblem,
} from './trust-subproblems.ts'

/**
 * How a trust-region method models curvature: the exact Hessian by autodiff, or a quasi-Newton approximation updated
 * from steps and gradient changes (SR1, BFGS).
 */
export interface HessianModel<H> {
  readonly name: string
  /** The Hessian data at the start. */
  init(objective: FlatObjective, x: Float64Array): H
  /** The data at a new point x (after an accepted step), or after any step for quasi-Newton updates. */
  update(
    objective: FlatObjective,
    h: H,
    step: { readonly x: Float64Array; readonly s: Float64Array; readonly y: Float64Array; readonly accepted: boolean },
  ): H
  /** B v. */
  hessp(objective: FlatObjective, h: H, v: Float64Array): Float64Array
  /** The dense B. */
  hess(objective: FlatObjective, h: H): Float64Array
  /** Whether `update` needs the gradient at rejected points too (quasi-Newton updates do). */
  readonly needsRejectedGradients: boolean
  /** Hessian evaluations an init or update of this model costs. */
  readonly evaluations: number
}

/** The data of {@link exactHessian}: the point and, in dense mode, the Hessian there. */
export interface ExactHessianData {
  readonly x: Float64Array
  readonly matrix?: Float64Array
}

/**
 * The exact Hessian by autodiff, recomputed at each accepted point; `products` mode uses Hessian-vector products only
 * (forward over reverse), never forming the matrix.
 *
 * @param mode - `dense` (default) or `products`.
 * @returns The curvature model.
 * @example
 * ```ts
 * import { exactHessian } from 'whitebox-engine/optim'
 * exactHessian('products').name // => 'hessianProducts'
 * ```
 */
export function exactHessian(mode: 'dense' | 'products' = 'dense'): HessianModel<ExactHessianData> {
  const at = (obj: FlatObjective, x: Float64Array): ExactHessianData =>
    mode === 'dense' ? { x, matrix: obj.hessian(x) } : { x }
  return {
    name: mode === 'dense' ? 'exactHessian' : 'hessianProducts',
    init: at,
    update: (obj, h, step) => (step.accepted ? at(obj, step.x) : h),
    hessp: (obj, h, v) => (h.matrix === undefined ? obj.hvp(h.x, v) : dgemv(h.matrix, v.length, v.length, v)),
    hess: (obj, h) => h.matrix ?? obj.hessian(h.x),
    needsRejectedGradients: false,
    evaluations: mode === 'dense' ? 1 : 0,
  }
}

/** Options of {@link trustRegion} and the trust-region methods. */
export interface TrustRegionOptions extends VectorMethodOptions {
  /** The subproblem solver (default Steihaug CG). */
  readonly subproblem?: TrustSubproblem
  /** The initial radius (default 1). */
  readonly initialRadius?: number
  /** The largest radius (default 1000). */
  readonly maxRadius?: number
  /** Accept steps with actual / predicted reduction above this (default 0.15). */
  readonly eta?: number
}

/** The state of a trust-region run. */
export interface TrustRegionState extends VectorState {
  readonly radius: number
  /** The last ratio of actual to predicted reduction. */
  readonly rho: number
  readonly hessian: unknown
  readonly memo?: unknown
  readonly accepted: boolean
  readonly subproblemIterations: number
}

/**
 * The trust-region method as a traced algorithm (scipy's _minimize_trust_region): solve the subproblem, compare the
 * actual and predicted reductions, shrink the radius by 4 when the ratio is below 1/4, double it (to the maximum) when
 * above 3/4 on the boundary, and accept steps whose ratio exceeds eta. Stops when the gradient's 2-norm falls below
 * `gradTol` (default 1e-4, scipy's).
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param model - The curvature model (exact Hessian, products, or a quasi-Newton approximation).
 * @param options - Subproblem, radii, eta and tolerance.
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` for inconsistent radii or eta.
 * @concept trust-region
 * @example
 * ```ts
 * import { exactHessian, trustRegionAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * trustRegionAlgorithm((x: Value) => sum(square(x)), [1, 2], exactHessian()).name // => 'trustRegion'
 * ```
 */
export function trustRegionAlgorithm<H>(
  f: Objective,
  x0: Parameters,
  model: HessianModel<H>,
  options: TrustRegionOptions = {},
): Algorithm<TrustRegionState> {
  const obj = flatObjective(f, x0)
  return trustRegionOn(obj, model, options)
}

/**
 * The trust-region algorithm on a flat objective, the driver behind {@link trustRegionAlgorithm} and the quasi-Newton
 * trust-region methods (SR1).
 *
 * @param obj - The flat objective.
 * @param model - The curvature model.
 * @param options - Subproblem, radii, eta and tolerance.
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` for inconsistent radii or eta.
 * @concept trust-region
 * @example
 * ```ts
 * import { exactHessian, flatObjective, trustRegionOn } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * trustRegionOn(flatObjective((x: Value) => sum(square(x)), [1, 2]), exactHessian()).name // => 'trustRegion'
 * ```
 */
export function trustRegionOn<H>(
  obj: FlatObjective,
  model: HessianModel<H>,
  options: TrustRegionOptions = {},
): Algorithm<TrustRegionState> {
  const subproblem = options.subproblem ?? steihaugStep()
  const radius0 = options.initialRadius ?? 1
  const maxRadius = options.maxRadius ?? 1000
  const eta = options.eta ?? 0.15
  const gtol = options.gradTol ?? 1e-4
  if (!(eta >= 0 && eta < 0.25)) throw new WhiteboxError('value', 'trust region: eta must be in [0, 0.25)')
  if (!(maxRadius > 0 && radius0 > 0 && radius0 < maxRadius))
    throw new WhiteboxError('value', 'trust region: need 0 < initialRadius < maxRadius')
  // The model of a state, counting Hessian-vector products (scipy counts them as Hessian evaluations).
  let products = 0
  const quadratic = (s: TrustRegionState): QuadraticModel => ({
    value: s.value,
    grad: s.grad,
    hessp: (v) => {
      if (model.name === 'hessianProducts') products += 1
      return model.hessp(obj, s.hessian as H, v)
    },
    hess: () => model.hess(obj, s.hessian as H),
  })
  const converged = (g: Float64Array): boolean => dnrm2(g) < gtol
  return {
    name: 'trustRegion',
    config: { ...(configOf(options) as Record<string, Json>), model: model.name, subproblem: subproblem.name },
    schema: vectorSchema({
      metrics: {
        radius: { description: 'trust radius' },
        rho: { description: 'actual over predicted reduction of the last step' },
        subproblemIterations: { description: 'iterations of the last subproblem solve' },
      },
      messages: { step: { description: 'whether the last step was accepted and hit the boundary' } },
    }),
    init: () => {
      const { value, grad } = obj.valueAndGrad(obj.x0)
      const hessian = model.init(obj, obj.x0)
      return {
        x: obj.x0,
        value,
        grad,
        nfev: 1,
        ngev: 1,
        nhev: model.evaluations,
        step: 0,
        radius: radius0,
        rho: Number.NaN,
        hessian,
        accepted: true,
        subproblemIterations: 0,
        ...(converged(grad)
          ? { stop: { status: 'converged' as const, message: 'gradient norm below tolerance' } }
          : {}),
      }
    },
    step: (s) => {
      products = 0
      const m = quadratic(s)
      let trial: TrustStep
      try {
        trial = subproblem.solve(m, s.radius, s.memo)
      } catch (e) {
        return { ...s, stop: { status: 'failed', message: `linear algebra error: ${(e as Error).message}` } }
      }
      const predicted = quadraticModelValue(m, trial.p)
      const xp = daxpy(1, trial.p, s.x)
      const needGrad = model.needsRejectedGradients
      const ev = needGrad ? obj.valueAndGrad(xp) : { value: obj.value(xp), grad: undefined }
      let nfev = s.nfev + 1
      let ngev = s.ngev + (needGrad ? 1 : 0)
      const actualReduction = s.value - ev.value
      const predictedReduction = s.value - predicted
      if (predictedReduction <= 0) {
        return {
          ...s,
          nfev,
          ngev,
          stop: { status: 'failed', message: 'a bad approximation caused failure to predict improvement' },
        }
      }
      const rho = actualReduction / predictedReduction
      let radius = s.radius
      if (rho < 0.25) radius *= 0.25
      else if (rho > 0.75 && trial.hitsBoundary) radius = Math.min(2 * radius, maxRadius)
      const accepted = rho > eta
      let grad = s.grad
      let x = s.x
      let value = s.value
      let gradNew = ev.grad
      if (accepted) {
        if (gradNew === undefined) {
          gradNew = obj.valueAndGrad(xp).grad
          ngev += 1
          nfev += 0
        }
        x = xp
        value = ev.value
        grad = gradNew
      }
      let hessian = s.hessian as H
      let nhev = s.nhev + products
      if (accepted || needGrad) {
        hessian = model.update(obj, hessian, {
          x: xp,
          s: trial.p,
          y: gradNew === undefined ? new Float64Array(0) : daxpy(-1, s.grad, gradNew),
          accepted,
        })
        nhev += model.evaluations
      }
      const stop =
        accepted && converged(grad)
          ? { status: 'converged' as const, message: 'gradient norm below tolerance' }
          : undefined
      return {
        x,
        value,
        grad,
        nfev,
        ngev,
        nhev,
        step: accepted ? dnrm2(trial.p) : 0,
        radius,
        rho,
        hessian,
        ...(accepted || needGrad ? {} : trial.memo === undefined ? {} : { memo: trial.memo }),
        accepted,
        subproblemIterations: trial.iterations,
        ...(stop === undefined ? {} : { stop }),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { radius: s.radius, rho: s.rho, subproblemIterations: s.subproblemIterations },
        messages: [{ kind: 'step', payload: { accepted: s.accepted, rho: Number.isFinite(s.rho) ? s.rho : null } }],
      }),
  }
}

/**
 * Minimise by a trust-region method with any subproblem solver and the exact Hessian (dense when the solver needs it,
 * products otherwise).
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Subproblem (default Steihaug CG), radii, eta, gradient tolerance (2-norm, default 1e-4), iteration
 * cap (default 200 n) and run options.
 * @returns The result.
 * @concept trust-region
 * @example
 * ```ts
 * import { exactStep, trustRegion } from 'whitebox-engine/optim'
 * import { add, mul, pow, square, sub, sum, type Value } from 'whitebox-engine/tensor'
 * const r = trustRegion((x: Value) => sum(add(pow(x, 4), square(sub(x, 1)))), [2, -2], { subproblem: exactStep() })
 * r.converged // => true
 * ```
 */
export function trustRegion(f: Objective, x0: Parameters, options: TrustRegionOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  const dense = options.subproblem?.dense ?? false
  return runVector(trustRegionOn(obj, exactHessian(dense ? 'dense' : 'products'), options), obj, options, 200 * obj.n)
}

/**
 * Minimise by the dogleg trust-region method with the exact Hessian (scipy.optimize.minimize, method 'dogleg').
 *
 * @param f - The objective (with a positive definite Hessian along the path).
 * @param x0 - The start.
 * @param options - Radii, eta, gradient tolerance, iteration cap and run options.
 * @returns The result.
 * @concept trust-region
 * @example
 * ```ts
 * import { trustDogleg } from 'whitebox-engine/optim'
 * import { add, mul, square, sub, sum, toFlat, type Value } from 'whitebox-engine/tensor'
 * const r = trustDogleg((x: Value) => add(sum(square(sub(x, 1))), mul(sum(square(x)), 0.5)), [3, -3])
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e8) / 1e8) // => [0.66666667, 0.66666667]
 * ```
 */
export function trustDogleg(f: Objective, x0: Parameters, options: TrustRegionOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(
    trustRegionOn(obj, exactHessian('dense'), { ...options, subproblem: doglegStep() }),
    obj,
    options,
    200 * obj.n,
  )
}

/**
 * Minimise by the Newton conjugate-gradient trust-region method (scipy.optimize.minimize, method 'trust-ncg'): Steihaug
 * CG with Hessian-vector products by autodiff, never forming the Hessian.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Radii, eta, gradient tolerance, iteration cap and run options.
 * @returns The result.
 * @concept trust-region
 * @example
 * ```ts
 * import { trustNcg } from 'whitebox-engine/optim'
 * import { square, sub, sum, type Value } from 'whitebox-engine/tensor'
 * trustNcg((x: Value) => sum(square(sub(x, 2))), [0, 0, 0]).converged // => true
 * ```
 */
export function trustNcg(f: Objective, x0: Parameters, options: TrustRegionOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(
    trustRegionOn(obj, exactHessian('products'), { ...options, subproblem: steihaugStep() }),
    obj,
    options,
    200 * obj.n,
  )
}

/**
 * Minimise by the nearly exact trust-region method (scipy.optimize.minimize, method 'trust-exact'): More-Sorensen
 * steps on the dense Hessian, robust to indefinite Hessians and the hard case.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Radii, eta, gradient tolerance, iteration cap and run options.
 * @returns The result.
 * @concept trust-region
 * @example
 * ```ts
 * import { trustExact } from 'whitebox-engine/optim'
 * import { add, cos, sum, square, type Value } from 'whitebox-engine/tensor'
 * trustExact((x: Value) => sum(add(cos(x), square(x))), [2, -1]).converged // => true
 * ```
 */
export function trustExact(f: Objective, x0: Parameters, options: TrustRegionOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(
    trustRegionOn(obj, exactHessian('dense'), { ...options, subproblem: exactStep() }),
    obj,
    options,
    200 * obj.n,
  )
}
