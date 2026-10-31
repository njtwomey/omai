/**
 * The damped Newton step shared by the maximum-likelihood fits of this folder (ordinal and mixed models): a direction
 * from a Hessian made positive definite by a Levenberg shift, and a backtracking line search on the objective.
 */

import { dpotrf, dpotrs } from '../numerics/linalg/index.ts'

/**
 * The Newton direction `-(H + shift I)^-1 g`, with the smallest power-of-ten shift (starting from none) that makes the
 * matrix positive definite.
 *
 * @param hessian - The symmetric Hessian, row-major `[m, m]`.
 * @param gradient - The gradient.
 * @param m - The dimension.
 * @returns The direction (steepest descent if no shift works).
 */
export function newtonDirection(hessian: ArrayLike<number>, gradient: ArrayLike<number>, m: number): Float64Array {
  let shift = 0
  for (let attempt = 0; attempt < 40; attempt++) {
    const a = Float64Array.from(hessian)
    for (let i = 0; i < m; i++) a[i * m + i] = (a[i * m + i] ?? 0) + shift
    const { factor, info } = dpotrf(a, m)
    if (info === 0)
      return dpotrs(
        factor,
        Float64Array.from(gradient, (v) => -v),
        m,
      )
    shift = shift === 0 ? 1e-6 : shift * 10
  }
  return Float64Array.from(gradient, (v) => -v)
}

/**
 * Halve a step until the objective decreases enough (the Armijo condition).
 *
 * @param evaluate - The objective (and whatever else the caller wants to keep) at a point.
 * @param point - The current point.
 * @param value - The objective at `point`.
 * @param gradient - The gradient at `point`.
 * @param direction - The search direction.
 * @returns The accepted point and its evaluation, or `undefined` when no step helps.
 */
export function backtrack<T extends { readonly objective: number }>(
  evaluate: (u: number[]) => T,
  point: readonly number[],
  value: number,
  gradient: ArrayLike<number>,
  direction: ArrayLike<number>,
): { readonly u: number[]; readonly result: T; readonly evaluations: number } | undefined {
  let slope = 0
  for (let i = 0; i < point.length; i++) slope += (direction[i] ?? 0) * (gradient[i] ?? 0)
  let t = 1
  for (let tries = 0; tries < 50; tries++) {
    const u = point.map((v, i) => v + t * (direction[i] ?? 0))
    const result = evaluate(u)
    if (Number.isFinite(result.objective) && result.objective <= value + 1e-4 * t * slope)
      return { u, result, evaluations: tries + 1 }
    t /= 2
  }
  return undefined
}
