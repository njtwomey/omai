/**
 * Quasi-Newton Hessian approximations for trust-region methods: the symmetric rank-one (SR1) update, which can
 * represent indefinite curvature, and the direct BFGS update, with SR1 in a trust region (Nocedal and Wright, Algorithm
 * 6.2) as a traced algorithm that needs no Hessian evaluations.
 */

import { daxpy, ddot, dgemv, dnrm2, dscal } from '../../numerics/linalg/index.ts'
import { runVector } from './_shared.ts'
import { flatObjective, type MinimizeResult, type Objective, type Parameters } from './objective.ts'
import { type HessianModel, type TrustRegionOptions, trustRegionOn } from './trust-region.ts'
import { steihaugStep } from './trust-subproblems.ts'

/** The data of a dense quasi-Newton Hessian approximation for trust-region methods. */
export interface QuasiNewtonHessian {
  /** B, n x n row-major (created at the first update when `initialScale` is 'auto'). */
  readonly matrix: Float64Array
  readonly updates: number
  readonly skipped: number
}

/** Options of {@link sr1Hessian} and {@link bfgsHessian}. */
export interface QuasiNewtonHessianOptions {
  /** B0 = scale I; 'auto' (default) scales by y'y / y's at the first update, as scipy's HessianUpdateStrategy. */
  readonly initialScale?: number | 'auto'
  /** SR1 skips updates with |s'(y - Bs)| < r |s| |y - Bs| (default 1e-8). */
  readonly minDenominator?: number
}

function quasiHessian(
  name: string,
  options: QuasiNewtonHessianOptions,
  update: (b: Float64Array, s: Float64Array, y: Float64Array, n: number) => Float64Array | undefined,
): HessianModel<QuasiNewtonHessian> {
  const scale = options.initialScale ?? 'auto'
  return {
    name,
    init: (_obj, x) => {
      const n = x.length
      const b = new Float64Array(n * n)
      for (let i = 0; i < n; i++) b[i * n + i] = typeof scale === 'number' ? scale : 1
      return { matrix: b, updates: 0, skipped: 0 }
    },
    update: (_obj, h, step) => {
      const n = step.s.length
      if (step.y.length === 0) return h
      let b = h.matrix
      if (h.updates === 0 && scale === 'auto') {
        const sy = ddot(step.s, step.y)
        const yy = ddot(step.y, step.y)
        if (sy !== 0 && yy !== 0) b = dscal(Math.abs(yy / sy), b)
      }
      const next = update(b, step.s, step.y, n)
      return next === undefined
        ? { matrix: b, updates: h.updates + 1, skipped: h.skipped + 1 }
        : { matrix: next, updates: h.updates + 1, skipped: h.skipped }
    },
    hessp: (_obj, h, v) => dgemv(h.matrix, v.length, v.length, v),
    hess: (_obj, h) => h.matrix,
    needsRejectedGradients: true,
    evaluations: 0,
  }
}

/**
 * The symmetric rank-one approximation of the Hessian for trust-region methods: B+ = B + (y - Bs)(y - Bs)' / (y - Bs)'s,
 * skipped when the denominator is tiny. SR1 can become indefinite, which trust regions handle, and often approximates
 * the true Hessian better than BFGS.
 *
 * @param options - Initial scale and the skipping threshold.
 * @returns The curvature model.
 * @concept quasi-newton
 * @example
 * ```ts
 * import { sr1Hessian } from 'omai-engine/optim'
 * sr1Hessian().name // => 'sr1'
 * ```
 */
export function sr1Hessian(options: QuasiNewtonHessianOptions = {}): HessianModel<QuasiNewtonHessian> {
  const r = options.minDenominator ?? 1e-8
  return quasiHessian('sr1', options, (b, s, y, n) => {
    const v = daxpy(-1, dgemv(b, n, n, s), y)
    const den = ddot(v, s)
    if (Math.abs(den) < r * dnrm2(s) * dnrm2(v) || den === 0) return undefined
    const out = Float64Array.from(b)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) out[i * n + j] = (out[i * n + j] ?? 0) + ((v[i] ?? 0) * (v[j] ?? 0)) / den
    return out
  })
}

/**
 * The BFGS approximation of the Hessian itself (not its inverse) for trust-region methods:
 * B+ = B - Bss'B / s'Bs + yy' / y's, skipped when y's <= 0.
 *
 * @param options - Initial scale.
 * @returns The curvature model.
 * @concept quasi-newton
 * @example
 * ```ts
 * import { bfgsHessian } from 'omai-engine/optim'
 * bfgsHessian().name // => 'bfgsHessian'
 * ```
 */
export function bfgsHessian(options: QuasiNewtonHessianOptions = {}): HessianModel<QuasiNewtonHessian> {
  return quasiHessian('bfgsHessian', options, (b, s, y, n) => {
    const sy = ddot(s, y)
    if (!(sy > 0)) return undefined
    const bs = dgemv(b, n, n, s)
    const sbs = ddot(s, bs)
    const out = Float64Array.from(b)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        out[i * n + j] = (out[i * n + j] ?? 0) - ((bs[i] ?? 0) * (bs[j] ?? 0)) / sbs + ((y[i] ?? 0) * (y[j] ?? 0)) / sy
    return out
  })
}

/** Options of {@link sr1}. */
export interface Sr1Options extends TrustRegionOptions, QuasiNewtonHessianOptions {}

/**
 * SR1 with a trust region (Nocedal and Wright, Algorithm 6.2) as a traced algorithm: the SR1 approximation, updated
 * after every step (accepted or not), in the trust-region driver with a Steihaug CG subproblem by default.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Trust-region options and the SR1 update options.
 * @returns The algorithm.
 * @concept quasi-newton
 * @example
 * ```ts
 * import { sr1Algorithm } from 'omai-engine/optim'
 * import { square, sum, type Value } from 'omai-engine/tensor'
 * sr1Algorithm((x: Value) => sum(square(x)), [1, 2]).name // => 'trustRegion'
 * ```
 */
export function sr1Algorithm(f: Objective, x0: Parameters, options: Sr1Options = {}): ReturnType<typeof trustRegionOn> {
  return trustRegionOn(flatObjective(f, x0), sr1Hessian(options), { subproblem: steihaugStep(), ...options })
}

/**
 * Minimise by SR1 with a trust region: no Hessian evaluations, robust to indefinite curvature.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Subproblem (default Steihaug CG), radii, eta, gradient tolerance (2-norm, default 1e-4), SR1
 * options, iteration cap (default 200 n) and run options.
 * @returns The result.
 * @concept quasi-newton
 * @example
 * ```ts
 * import { sr1 } from 'omai-engine/optim'
 * import { square, sub, sum, type Value } from 'omai-engine/tensor'
 * sr1((x: Value) => sum(square(sub(x, 3))), [0, 0], { gradTol: 1e-8 }).converged // => true
 * ```
 */
export function sr1(f: Objective, x0: Parameters, options: Sr1Options = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(
    trustRegionOn(obj, sr1Hessian(options), { subproblem: steihaugStep(), ...options }),
    obj,
    options,
    200 * obj.n,
  )
}
