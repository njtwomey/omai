/**
 * Solvers for the trust-region subproblem, minimising a quadratic model within a ball: Powell's dogleg step, the
 * Steihaug-Toint truncated conjugate gradient step, and the nearly exact More-Sorensen step, the subproblems of scipy's
 * trust-dogleg, trust-ncg and trust-exact methods.
 */

import { OmaiError } from '../../base/index.ts'
import { daxpy, ddot, dgemv, dnrm2, dpotrf, dpotrs, dscal, dtrsv } from '../../numerics/linalg/index.ts'

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

/**
 * The value of a quadratic model at a step, m(p) = f + g'p + p'Bp / 2.
 *
 * @param model - The model.
 * @param p - The step.
 * @returns m(p).
 * @example
 * ```ts
 * import { quadraticModelValue } from 'omai-engine/optim'
 * const model = { value: 1, grad: new Float64Array([1, 0]), hessp: (v: Float64Array) => v }
 * quadraticModelValue(model, new Float64Array([-1, 0])) // => 0.5
 * ```
 */
export function quadraticModelValue(model: QuadraticModel, p: Float64Array): number {
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
  if (model.hess === undefined) throw new OmaiError('value', `${name} needs a dense Hessian`)
  return model.hess()
}

/**
 * Powell's dogleg step (scipy's trust-dogleg subproblem): the Newton step when it fits, else the path from the Cauchy
 * point towards the Newton step, cut at the boundary. Needs a positive definite model Hessian.
 *
 * @returns The subproblem solver.
 * @throws OmaiError with code `singular` when the model Hessian is not positive definite.
 * @concept trust-region
 * @example
 * ```ts
 * import { doglegStep } from 'omai-engine/optim'
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
      if (info !== 0) throw new OmaiError('singular', 'dogleg: the model Hessian is not positive definite')
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
 * import { steihaugStep } from 'omai-engine/optim'
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
          return {
            p: quadraticModelValue(model, pa) < quadraticModelValue(model, pb) ? pa : pb,
            hitsBoundary: true,
            iterations: it,
          }
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
 * import { exactStep } from 'omai-engine/optim'
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
