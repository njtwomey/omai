import { WhiteboxError } from '../../base/index.ts'
import { primalOf, toFlat } from '../../tensor/index.ts'
import type { SpectralData } from './tikhonov.ts'

function arrays(data: SpectralData): { s: number[]; beta: number[] } {
  return { s: toFlat(primalOf(data.s)), beta: toFlat(primalOf(data.beta)) }
}

/** A geometric grid of n parameters from the largest to a small fraction of the singular values. */
export function lambdaGrid(data: SpectralData, n = 100): number[] {
  const { s } = arrays(data)
  const hi = s[0] ?? 1
  const lo = Math.max((s[s.length - 1] ?? hi * 1e-16) / 10, hi * 1e-16)
  return Array.from({ length: n }, (_, i) => hi * (lo / hi) ** (i / (n - 1)))
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
  readonly lambdas: readonly number[]
  /** The rule's criterion at each parameter (curvature, GCV function, ...). */
  readonly criterion: readonly number[]
}

/**
 * The L-curve corner (Hansen): the parameter of maximum curvature of (log |A x - b|, log |x|) over the Tikhonov
 * solutions, with the curvature from the closed form in terms of the SVD.
 *
 * @param data - From spectralData.
 * @param options - The parameters to scan (default 100 geometric) and |b|^2 (to include the part outside range(A)).
 * @returns The corner parameter and the curvature curve.
 * @concept l-curve
 * @example
 * ```ts
 * import { lCurve, spectralData } from 'whitebox-engine/numerics/inverse'
 * lCurve(spectralData([[1, 0, 0], [0, 0.1, 0], [0, 0, 0.001]], [1, 0.1, 0.01])).lambda > 0 // => true
 * ```
 */
export function lCurve(
  data: SpectralData,
  options: { readonly lambdas?: readonly number[]; readonly bNormSquared?: number } = {},
): ParameterChoice {
  const { s, beta } = arrays(data)
  const lambdas = options.lambdas ?? lambdaGrid(data)
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
  return { lambda: lambdas[best] ?? Number.NaN, lambdas, criterion }
}

/**
 * Generalised cross-validation (Golub, Heath and Wahba): minimise G(lambda) = |A x_lambda - b|^2 / (m - sum f_i)^2,
 * which estimates the predictive error without knowing the noise level.
 *
 * @param data - From spectralData.
 * @param m - Number of data points.
 * @param options - Parameters to scan and |b|^2.
 * @returns The minimiser and G along the scan.
 * @example
 * ```ts
 * import { gcv, spectralData } from 'whitebox-engine/numerics/inverse'
 * gcv(spectralData([[1, 0], [0, 0.01], [0, 0]], [1, 0.02, 0.1]), 3).lambda > 0 // => true
 * ```
 */
export function gcv(
  data: SpectralData,
  m: number,
  options: { readonly lambdas?: readonly number[]; readonly bNormSquared?: number } = {},
): ParameterChoice {
  const { s, beta } = arrays(data)
  const lambdas = options.lambdas ?? lambdaGrid(data)
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
  return { lambda: lambdas[best] ?? Number.NaN, lambdas, criterion }
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
 * import { discrepancyPrinciple, spectralData } from 'whitebox-engine/numerics/inverse'
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
    throw new WhiteboxError('value', 'discrepancyPrinciple: the noise level is below the attainable residual')
  for (let it = 0; it < 200 && hi - lo > 1e-12; it++) {
    const mid = (lo + hi) / 2
    if (residual(Math.exp(mid)) > target) hi = mid
    else lo = mid
  }
  return Math.exp(lo)
}

/**
 * The quasi-optimality criterion: on a geometric grid lambda_k, choose the k minimising |x_{k+1} - x_k| (computed
 * from the SVD), a heuristic rule that needs no noise level.
 *
 * @param data - From spectralData.
 * @param options - The grid (default 100 geometric parameters).
 * @returns The choice and the criterion.
 * @example
 * ```ts
 * import { quasiOptimality, spectralData } from 'whitebox-engine/numerics/inverse'
 * quasiOptimality(spectralData([[1, 0], [0, 0.01]], [1, 0.02])).lambda > 0 // => true
 * ```
 */
export function quasiOptimality(
  data: SpectralData,
  options: { readonly lambdas?: readonly number[] } = {},
): ParameterChoice {
  const { s, beta } = arrays(data)
  const lambdas = options.lambdas ?? lambdaGrid(data)
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
  return { lambda: lambdas[best] ?? Number.NaN, lambdas: lambdas.slice(0, -1), criterion }
}
