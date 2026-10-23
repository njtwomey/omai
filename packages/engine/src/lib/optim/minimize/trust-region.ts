import { type Json, WhiteboxError } from '../../base/index.ts'
import { daxpy, ddot, dgemv, dnrm2, dpotrf, dpotrs, dscal, dtrsv } from '../../numerics/linalg/index.ts'
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

/** A quadratic model m(p) = f + g'p + p'Bp / 2 of the objective around a point. */
export interface QuadraticModel {
  readonly value: number
  readonly grad: Float64Array
  /** B v. */
  readonly hessp: (v: Float64Array) => Float64Array
  /** The dense B (n x n, row-major), for subproblem solvers that factorise it. */
  readonly hess?: () => Float64Array
}

/** A step of a trust-region subproblem solver. */
export interface TrustStep {
  readonly p: Float64Array
  readonly hitsBoundary: boolean
  /** Inner iterations (CG steps, factorisations). */
  readonly iterations: number
  /** State a solver keeps between solves of the same model at a smaller radius (the exact solver's lambda bounds). */
  readonly memo?: unknown
}

/**
 * A trust-region subproblem solver: approximately minimise the model within |p| <= radius. `memo` is what the solver
 * returned for the same model on the previous (rejected) step; `dense` says whether it factorises the Hessian.
 */
export interface TrustSubproblem {
  readonly name: string
  readonly dense: boolean
  solve(model: QuadraticModel, radius: number, memo?: unknown): TrustStep
}

/** The model value m(p). */
function modelAt(model: QuadraticModel, p: Float64Array): number {
  return model.value + ddot(model.grad, p) + 0.5 * ddot(p, model.hessp(p))
}

/** The two values of t where |z + t d| = radius (scipy's get_boundaries_intersections), sorted. */
function boundary(z: Float64Array, d: Float64Array, radius: number): [number, number] {
  const a = ddot(d, d)
  const b = 2 * ddot(z, d)
  const c = ddot(z, z) - radius * radius
  const sq = Math.sqrt(b * b - 4 * a * c)
  const aux = b + (b >= 0 ? sq : -sq)
  const ta = -aux / (2 * a)
  const tb = (-2 * c) / aux
  return ta <= tb ? [ta, tb] : [tb, ta]
}

function denseHessian(model: QuadraticModel, name: string): Float64Array {
  if (model.hess === undefined) throw new WhiteboxError('value', `${name} needs a dense Hessian`)
  return model.hess()
}

/**
 * Powell's dogleg step (scipy's trust-dogleg subproblem): the Newton step when it fits, else the path from the Cauchy
 * point towards the Newton step, cut at the boundary. Needs a positive definite model Hessian.
 *
 * @returns The subproblem solver.
 * @throws WhiteboxError with code `singular` when the model Hessian is not positive definite.
 * @concept trust-region
 * @example
 * ```ts
 * import { doglegStep } from 'whitebox-engine/optim'
 * const model = { value: 0, grad: new Float64Array([1, 0]), hessp: (v: Float64Array) => v, hess: () => new Float64Array([1, 0, 0, 1]) }
 * Array.from(doglegStep().solve(model, 0.5).p) // => [-0.5, 0]
 * ```
 */
export function doglegStep(): TrustSubproblem {
  return {
    name: 'dogleg',
    dense: true,
    solve: (model, radius) => {
      const g = model.grad
      const n = g.length
      const { factor, info } = dpotrf(denseHessian(model, 'dogleg'), n)
      if (info !== 0) throw new WhiteboxError('singular', 'dogleg: the model Hessian is not positive definite')
      const pB = dscal(-1, dpotrs(factor, g, n))
      if (dnrm2(pB) < radius) return { p: pB, hitsBoundary: false, iterations: 1 }
      const Bg = model.hessp(g)
      const pU = dscal(-ddot(g, g) / ddot(g, Bg), g)
      const nU = dnrm2(pU)
      if (nU >= radius) return { p: dscal(radius / nU, pU), hitsBoundary: true, iterations: 1 }
      const d = daxpy(-1, pU, pB)
      const [, tb] = boundary(pU, d, radius)
      return { p: daxpy(tb, d, pU), hitsBoundary: true, iterations: 1 }
    },
  }
}

/** Options of {@link steihaugStep}. */
export interface SteihaugOptions {
  /** CG iteration cap (default: unbounded, as scipy; CG ends in at most n steps in exact arithmetic). */
  readonly maxIterations?: number
}

/**
 * The Steihaug-Toint truncated conjugate gradient step (scipy's trust-ncg subproblem): CG on the model until the
 * residual is small (|r| < min(0.5, sqrt|g|) |g|), negative curvature is met or the iterate leaves the region. Uses
 * only Hessian-vector products.
 *
 * @param options - CG iteration cap.
 * @returns The subproblem solver.
 * @concept trust-region
 * @example
 * ```ts
 * import { steihaugStep } from 'whitebox-engine/optim'
 * const model = { value: 0, grad: new Float64Array([1, 0]), hessp: (v: Float64Array) => v.map((x) => -x) }
 * steihaugStep().solve(model, 2).hitsBoundary // => true
 * ```
 */
export function steihaugStep(options: SteihaugOptions = {}): TrustSubproblem {
  const cap = options.maxIterations ?? Infinity
  return {
    name: 'steihaug',
    dense: false,
    solve: (model, radius) => {
      const g = model.grad
      const gNorm = dnrm2(g)
      const tolerance = Math.min(0.5, Math.sqrt(gNorm)) * gNorm
      let z: Float64Array = new Float64Array(g.length)
      if (gNorm < tolerance) return { p: z, hitsBoundary: false, iterations: 0 }
      let r: Float64Array = Float64Array.from(g)
      let d = dscal(-1, r)
      for (let it = 1; ; it++) {
        const Bd = model.hessp(d)
        const dBd = ddot(d, Bd)
        if (dBd <= 0) {
          const [ta, tb] = boundary(z, d, radius)
          const pa = daxpy(ta, d, z)
          const pb = daxpy(tb, d, z)
          return { p: modelAt(model, pa) < modelAt(model, pb) ? pa : pb, hitsBoundary: true, iterations: it }
        }
        const rr = ddot(r, r)
        const alpha = rr / dBd
        const zNext = daxpy(alpha, d, z)
        if (dnrm2(zNext) >= radius) {
          const [, tb] = boundary(z, d, radius)
          return { p: daxpy(tb, d, z), hitsBoundary: true, iterations: it }
        }
        const rNext = daxpy(alpha, Bd, r)
        const rrNext = ddot(rNext, rNext)
        if (Math.sqrt(rrNext) < tolerance || it >= cap) return { p: zNext, hitsBoundary: false, iterations: it }
        d = daxpy(rrNext / rr, d, dscal(-1, rNext))
        z = zNext
        r = rNext
      }
    },
  }
}

/** Options of {@link exactStep}. */
export interface ExactStepOptions {
  /** Relative accuracy on the boundary (default 0.1). */
  readonly kEasy?: number
  /** Relative accuracy in the hard case (default 0.2). */
  readonly kHard?: number
  /** Factorisations per solve (default 25). */
  readonly maxIterations?: number
}

/** An estimate of the smallest singular value of an upper triangular U and its vector (scipy's, after Cline et al.). */
function smallestSingular(u: Float64Array, n: number): { s: number; z: Float64Array } {
  const p = new Float64Array(n)
  const w = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    const ukk = u[k * n + k] ?? 0
    const wp = (1 - (p[k] ?? 0)) / ukk
    const wm = (-1 - (p[k] ?? 0)) / ukk
    let np = Math.abs(wp)
    let nm = Math.abs(wm)
    for (let i = k + 1; i < n; i++) {
      np += Math.abs((p[i] ?? 0) + (u[k * n + i] ?? 0) * wp)
      nm += Math.abs((p[i] ?? 0) + (u[k * n + i] ?? 0) * wm)
    }
    const wk = np >= nm ? wp : wm
    w[k] = wk
    for (let i = k + 1; i < n; i++) p[i] = (p[i] ?? 0) + (u[k * n + i] ?? 0) * wk
  }
  const v = dtrsv(u, w, n)
  const vn = dnrm2(v)
  return { s: dnrm2(w) / vn, z: dscal(1 / vn, v) }
}

/**
 * The nearly exact trust-region step of More and Sorensen (scipy's trust-exact subproblem): Newton's method on the
 * secular equation |p(lambda)| = radius with Cholesky factorisations of B + lambda I, safeguarded by Gershgorin bounds,
 * handling the hard case with an approximate smallest eigenvector.
 *
 * @param options - Accuracy constants and the factorisation cap.
 * @returns The subproblem solver (needs a dense Hessian).
 * @concept trust-region
 * @reference J. J. More and D. C. Sorensen, Computing a trust region step, SIAM J. Sci. Stat. Comput. 4 (1983).
 * @example
 * ```ts
 * import { exactStep } from 'whitebox-engine/optim'
 * const model = { value: 0, grad: new Float64Array([1, 1]), hessp: (v: Float64Array) => v, hess: () => new Float64Array([1, 0, 0, 1]) }
 * Array.from(exactStep().solve(model, 10).p) // => [-1, -1]
 * ```
 */
export function exactStep(options: ExactStepOptions = {}): TrustSubproblem {
  const kEasy = options.kEasy ?? 0.1
  const kHard = options.kHard ?? 0.2
  const maxiter = options.maxIterations ?? 25
  const UPDATE = 0.01
  return {
    name: 'exact',
    dense: true,
    solve: (model, radius, memo) => {
      const H = denseHessian(model, 'exactStep')
      const g = model.grad
      const n = g.length
      const gNorm = dnrm2(g)
      let lbG = Infinity
      let ubG = -Infinity
      let hInf = 0
      let hFro = 0
      let minDiag = Infinity
      for (let i = 0; i < n; i++) {
        let row = 0
        for (let j = 0; j < n; j++) {
          const h = H[i * n + j] ?? 0
          row += Math.abs(h)
          hFro += h * h
        }
        const d = H[i * n + i] ?? 0
        minDiag = Math.min(minDiag, d)
        lbG = Math.min(lbG, d + Math.abs(d) - row)
        ubG = Math.max(ubG, d - Math.abs(d) + row)
        hInf = Math.max(hInf, row)
      }
      hFro = Math.sqrt(hFro)
      const closeToZero = n * Number.EPSILON * hInf
      let lambdaUb = Math.max(0, gNorm / radius + Math.min(-lbG, hFro, hInf))
      let lambdaLb = Math.max(0, -minDiag, gNorm / radius - Math.min(ubG, hFro, hInf))
      const previous = memo as { radius: number; lambdaLb: number } | undefined
      if (previous !== undefined && radius < previous.radius) lambdaLb = Math.max(previous.lambdaLb, lambdaLb)
      let lambda =
        lambdaLb === 0 ? 0 : Math.max(Math.sqrt(lambdaLb * lambdaUb), lambdaLb + UPDATE * (lambdaUb - lambdaLb))
      const shifted = (l: number): Float64Array => {
        const a = Float64Array.from(H)
        for (let i = 0; i < n; i++) a[i * n + i] = (a[i * n + i] ?? 0) + l
        return a
      }
      let p: Float64Array = new Float64Array(n)
      let hits = true
      let factored: { factor: Float64Array; info: number } | undefined
      let it = 0
      while (it < maxiter) {
        const Hl = shifted(lambda)
        const { factor: U, info } = factored ?? dpotrf(Hl, n)
        factored = undefined
        it += 1
        if (info === 0 && gNorm > closeToZero) {
          p = dpotrs(U, dscal(-1, g), n)
          const pNorm = dnrm2(p)
          if (pNorm <= radius && lambda === 0) {
            hits = false
            break
          }
          const w = dtrsv(U, p, n, { transpose: true })
          const wNorm = dnrm2(w)
          const lambdaNew = lambda + (pNorm / wNorm) ** 2 * ((pNorm - radius) / radius)
          if (pNorm < radius) {
            const { s, z } = smallestSingular(U, n)
            const [ta, tb] = boundary(p, z, radius)
            const stepLen = Math.abs(ta) < Math.abs(tb) ? ta : tb
            const quad = ddot(p, dgemv(Hl, n, n, p))
            const relErr = (stepLen * stepLen * s * s) / (quad + lambda * radius * radius)
            if (relErr <= kHard) {
              p = daxpy(stepLen, z, p)
              break
            }
            lambdaUb = lambda
            lambdaLb = Math.max(lambdaLb, lambda - s * s)
            const next = dpotrf(shifted(lambdaNew), n)
            if (next.info === 0) {
              lambda = lambdaNew
              factored = next
            } else {
              lambdaLb = Math.max(lambdaLb, lambdaNew)
              lambda = Math.max(Math.sqrt(Math.abs(lambdaLb * lambdaUb)), lambdaLb + UPDATE * (lambdaUb - lambdaLb))
            }
          } else {
            if (Math.abs(pNorm - radius) / radius <= kEasy) break
            lambdaLb = lambda
            lambda = lambdaNew
          }
        } else if (info === 0) {
          if (lambda === 0) {
            p = new Float64Array(n)
            hits = false
            break
          }
          const { s, z } = smallestSingular(U, n)
          p = dscal(radius, z)
          if (radius * radius * s * s <= kHard * lambda * radius * radius) break
          lambdaUb = lambda
          lambdaLb = Math.max(lambdaLb, lambda - s * s)
          lambda = Math.max(Math.sqrt(lambdaLb * lambdaUb), lambdaLb + UPDATE * (lambdaUb - lambdaLb))
        } else {
          // The leading k x k block is singular: delta and v with (A + delta e_k e_k^T) v = 0 bound lambda below.
          const k = info
          let delta = -(Hl[(k - 1) * n + (k - 1)] ?? 0)
          for (let i = 0; i < k - 1; i++) delta += (U[i * n + (k - 1)] ?? 0) ** 2
          const v = new Float64Array(n)
          v[k - 1] = 1
          if (k > 1) {
            const sub = new Float64Array((k - 1) * (k - 1))
            const rhs = new Float64Array(k - 1)
            for (let i = 0; i < k - 1; i++) {
              rhs[i] = -(U[i * n + (k - 1)] ?? 0)
              for (let j = 0; j < k - 1; j++) sub[i * (k - 1) + j] = U[i * n + j] ?? 0
            }
            v.set(dtrsv(sub, rhs, k - 1), 0)
          }
          lambdaLb = Math.max(lambdaLb, lambda + delta / ddot(v, v))
          lambda = Math.max(Math.sqrt(Math.abs(lambdaLb * lambdaUb)), lambdaLb + UPDATE * (lambdaUb - lambdaLb))
        }
      }
      return { p, hitsBoundary: hits, iterations: it, memo: { radius, lambdaLb } }
    },
  }
}

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

/** The trust-region algorithm on a flat objective (shared by the quasi-Newton trust-region methods). */
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
      const predicted = modelAt(m, trial.p)
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
