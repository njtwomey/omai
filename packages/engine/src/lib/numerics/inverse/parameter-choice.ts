/**
 * Rules for choosing the regularisation parameter of an ill-posed problem: the L-curve corner (Hansen), generalised
 * cross-validation, Morozov's discrepancy principle for a known noise level, and the quasi-optimality criterion, with a
 * parameter grid scaled to the singular values.
 */

import { OmaiError } from '../../base/index.ts'
import { primalOf, type Tensor, type TensorLike, toFlat, toVector } from '../../tensor/index.ts'
import { vectorOf } from '../_vector.ts'
import type { SpectralData } from './tikhonov.ts'

function arrays(data: SpectralData): { s: number[]; beta: number[] } {
  return { s: toFlat(primalOf(data.s)), beta: toFlat(primalOf(data.beta)) }
}

/**
 * A geometric grid of n parameters from the largest to a small fraction of the singular values. Returns a tensor
 * (migration, D-009: was `number[]`).
 *
 * @param data - The SVD of the problem (see `spectralData`).
 * @param n - Number of parameters (default 100).
 * @returns The grid `[n]`, decreasing, starting at the largest singular value.
 * @throws OmaiError with code `value` when the singular values are all zero.
 * @example
 * ```ts
 * import { lambdaGrid, spectralData } from 'omai-engine/numerics/inverse'
 * import { toFlat } from 'omai-engine/tensor'
 * const grid = toFlat(lambdaGrid(spectralData([[2, 0], [0, 1]], [1, 1]), 3))
 * [grid.length, grid[0]] // => [3, 2]
 * ```
 */
export function lambdaGrid(data: SpectralData, n = 100): Tensor<'float64', [number]> {
  return vectorOf(gridOf(arrays(data).s, n, 'lambdaGrid'))
}

function gridOf(s: readonly number[], n: number, name: string): number[] {
  const hi = s[0] ?? 1
  if (!(hi > 0)) throw new OmaiError('value', `${name}: the singular values are all zero`)
  const lo = Math.max((s[s.length - 1] ?? hi * 1e-16) / 10, hi * 1e-16)
  return Array.from({ length: n }, (_, i) => hi * (lo / hi) ** (i / (n - 1)))
}

/** The parameters to scan: the given vector, or the default geometric grid. */
function scan(lambdas: TensorLike | undefined, s: readonly number[], rule: string): readonly number[] {
  return lambdas === undefined ? gridOf(s, 100, 'lambdaGrid') : toVector(lambdas, rule)
}

/** Residual and solution norms of the Tikhonov solution, from the SVD. */
function norms(
  s: readonly number[],
  beta: readonly number[],
  lambda: number,
  outside: number,
): { rho: number; eta: number } {
  let rho = outside
  let eta = 0
  s.forEach((si, i) => {
    const f = (si * si) / (si * si + lambda * lambda)
    rho += ((1 - f) * (beta[i] ?? 0)) ** 2
    eta += ((f * (beta[i] ?? 0)) / si) ** 2
  })
  return { rho: Math.sqrt(rho), eta: Math.sqrt(eta) }
}

/** Result of a parameter-choice rule. */
export interface ParameterChoice {
  readonly lambda: number
  /** The parameters scanned `[k]`. */
  readonly lambdas: Tensor<'float64', [number]>
  /** The rule's criterion at each parameter `[k]` (curvature, GCV function, ...). */
  readonly criterion: Tensor<'float64', [number]>
}

/**
 * The L-curve corner (Hansen): the parameter of maximum curvature of (log |A x - b|, log |x|) over the Tikhonov
 * solutions, with the curvature from the closed form in terms of the SVD. Takes the parameters as a tensor-like and
 * returns them and the curvature as tensors (migration, D-009: were `readonly number[]`).
 *
 * @param data - From spectralData.
 * @param options - The parameters to scan `[k]` (default 100 geometric) and |b|^2 (to include the part outside
 *   range(A)).
 * @returns The corner parameter and the curvature curve.
 * @throws OmaiError with code `shape` unless the parameters are a vector, and `not-differentiable` for a traced value.
 * @concept l-curve
 * @example
 * ```ts
 * import { lCurve, spectralData } from 'omai-engine/numerics/inverse'
 * lCurve(spectralData([[1, 0, 0], [0, 0.1, 0], [0, 0, 0.001]], [1, 0.1, 0.01])).lambda > 0 // => true
 * ```
 */
export function lCurve(
  data: SpectralData,
  options: { readonly lambdas?: TensorLike; readonly bNormSquared?: number } = {},
): ParameterChoice {
  const { s, beta } = arrays(data)
  const lambdas = scan(options.lambdas, s, 'lCurve')
  const outside = Math.max(
    0,
    (options.bNormSquared ?? beta.reduce((acc, v) => acc + v * v, 0)) - beta.reduce((acc, v) => acc + v * v, 0),
  )
  const criterion = lambdas.map((lambda) => {
    // Hansen's lcfun: the curvature of (log rho, log eta) from the derivatives of eta = |x| and rho = |A x - b| in lambda.
    let eta2 = 0
    let rho2 = outside
    let phi = 0
    let psi = 0
    let dphi = 0
    let dpsi = 0
    s.forEach((si, i) => {
      const bi = beta[i] ?? 0
      const xi = bi / si
      const f = (si * si) / (si * si + lambda * lambda)
      const cf = 1 - f
      const f1 = (-2 * f * cf) / lambda
      const f2 = (-f1 * (3 - 4 * f)) / lambda
      eta2 += (f * xi) ** 2
      rho2 += (cf * bi) ** 2
      phi += f * f1 * xi * xi
      psi += cf * f1 * bi * bi
      dphi += (f1 * f1 + f * f2) * xi * xi
      dpsi += (-f1 * f1 + cf * f2) * bi * bi
    })
    const eta = Math.sqrt(eta2)
    const rho = Math.sqrt(rho2)
    const deta = phi / eta
    const drho = -psi / rho
    const ddeta = dphi / eta - deta * (deta / eta)
    const ddrho = -dpsi / rho - drho * (drho / rho)
    const dlogeta = deta / eta
    const dlogrho = drho / rho
    const ddlogeta = ddeta / eta - dlogeta * dlogeta
    const ddlogrho = ddrho / rho - dlogrho * dlogrho
    return (dlogrho * ddlogeta - ddlogrho * dlogeta) / (dlogrho * dlogrho + dlogeta * dlogeta) ** 1.5
  })
  let best = 0
  criterion.forEach((k, i) => {
    if (k > (criterion[best] ?? Number.NEGATIVE_INFINITY)) best = i
  })
  return { lambda: lambdas[best] ?? Number.NaN, lambdas: vectorOf(lambdas), criterion: vectorOf(criterion) }
}

/**
 * Generalised cross-validation (Golub, Heath and Wahba): minimise G(lambda) = |A x_lambda - b|^2 / (m - sum f_i)^2,
 * which estimates the predictive error without knowing the noise level. Takes the parameters as a tensor-like and
 * returns them and G as tensors (migration, D-009: were `readonly number[]`).
 *
 * @param data - From spectralData.
 * @param m - Number of data points.
 * @param options - Parameters to scan `[k]` and |b|^2.
 * @returns The minimiser and G along the scan.
 * @throws OmaiError with code `shape` unless the parameters are a vector, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { gcv, spectralData } from 'omai-engine/numerics/inverse'
 * gcv(spectralData([[1, 0], [0, 0.01], [0, 0]], [1, 0.02, 0.1]), 3).lambda > 0 // => true
 * ```
 */
export function gcv(
  data: SpectralData,
  m: number,
  options: { readonly lambdas?: TensorLike; readonly bNormSquared?: number } = {},
): ParameterChoice {
  const { s, beta } = arrays(data)
  const lambdas = scan(options.lambdas, s, 'gcv')
  const outside = Math.max(
    0,
    (options.bNormSquared ?? beta.reduce((acc, v) => acc + v * v, 0)) - beta.reduce((acc, v) => acc + v * v, 0),
  )
  const criterion = lambdas.map((lambda) => {
    const { rho } = norms(s, beta, lambda, outside)
    const trace = m - s.reduce((acc, si) => acc + (si * si) / (si * si + lambda * lambda), 0)
    return (rho * rho) / (trace * trace)
  })
  let best = 0
  criterion.forEach((g, i) => {
    if (g < (criterion[best] ?? Number.POSITIVE_INFINITY)) best = i
  })
  return { lambda: lambdas[best] ?? Number.NaN, lambdas: vectorOf(lambdas), criterion: vectorOf(criterion) }
}

/**
 * Morozov's discrepancy principle: the largest lambda with |A x_lambda - b| <= tau delta for noise level delta, found
 * by bisection in log lambda (the residual norm increases with lambda).
 *
 * @param data - From spectralData.
 * @param delta - The noise norm |e|.
 * @param options - Safety factor tau (default 1) and |b|^2.
 * @returns The parameter.
 * @example
 * ```ts
 * import { discrepancyPrinciple, spectralData } from 'omai-engine/numerics/inverse'
 * discrepancyPrinciple(spectralData([[1, 0], [0, 0.1]], [1, 0.1]), 0.05) > 0 // => true
 * ```
 */
export function discrepancyPrinciple(
  data: SpectralData,
  delta: number,
  options: { readonly tau?: number; readonly bNormSquared?: number } = {},
): number {
  const { s, beta } = arrays(data)
  const target = (options.tau ?? 1) * delta
  const outside = Math.max(
    0,
    (options.bNormSquared ?? beta.reduce((acc, v) => acc + v * v, 0)) - beta.reduce((acc, v) => acc + v * v, 0),
  )
  const residual = (lambda: number): number => norms(s, beta, lambda, outside).rho
  let lo = Math.log((s[s.length - 1] ?? 1) * 1e-8 || 1e-300)
  let hi = Math.log((s[0] ?? 1) * 1e4)
  if (residual(Math.exp(lo)) > target)
    throw new OmaiError('value', 'discrepancyPrinciple: the noise level is below the attainable residual')
  for (let it = 0; it < 200 && hi - lo > 1e-12; it++) {
    const mid = (lo + hi) / 2
    if (residual(Math.exp(mid)) > target) hi = mid
    else lo = mid
  }
  return Math.exp(lo)
}

/**
 * The quasi-optimality criterion: on a geometric grid lambda_k, choose the k minimising |x_{k+1} - x_k| (computed
 * from the SVD), a heuristic rule that needs no noise level. Takes the grid as a tensor-like and returns the grid and
 * the criterion as tensors (migration, D-009: were `readonly number[]`).
 *
 * @param data - From spectralData.
 * @param options - The grid `[k]` (default 100 geometric parameters).
 * @returns The choice and the criterion (the grid without its last point, `[k - 1]`).
 * @throws OmaiError with code `shape` unless the grid is a vector, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { quasiOptimality, spectralData } from 'omai-engine/numerics/inverse'
 * quasiOptimality(spectralData([[1, 0], [0, 0.01]], [1, 0.02])).lambda > 0 // => true
 * ```
 */
export function quasiOptimality(data: SpectralData, options: { readonly lambdas?: TensorLike } = {}): ParameterChoice {
  const { s, beta } = arrays(data)
  const lambdas = scan(options.lambdas, s, 'quasiOptimality')
  const coefficients = (lambda: number): number[] =>
    s.map((si, i) => ((si * si) / (si * si + lambda * lambda)) * ((beta[i] ?? 0) / si))
  const criterion = lambdas.slice(0, -1).map((lambda, k) => {
    const a = coefficients(lambda)
    const b = coefficients(lambdas[k + 1] ?? lambda)
    return Math.sqrt(a.reduce((acc, v, i) => acc + (v - (b[i] ?? 0)) ** 2, 0))
  })
  let best = 0
  criterion.forEach((q, i) => {
    if (q < (criterion[best] ?? Number.POSITIVE_INFINITY)) best = i
  })
  return {
    lambda: lambdas[best] ?? Number.NaN,
    lambdas: vectorOf(lambdas.slice(0, -1)),
    criterion: vectorOf(criterion),
  }
}
