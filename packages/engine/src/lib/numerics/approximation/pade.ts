/**
 * Pade approximants and continued fractions: the rational function matching the first terms of a power series
 * (`scipy.interpolate.pade`), and evaluation of continued fractions by the modified Lentz algorithm, the form many
 * special functions are computed in.
 */

import { MdmlError } from '../../base/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat, toVector } from '../../tensor/index.ts'
import type { RunStatus } from '../../trace/index.ts'
import { vectorOf } from '../_vector.ts'
import { solve } from '../linalg/index.ts'

/**
 * The Pade approximant [n/m] of a power series (scipy.interpolate.pade): p / q with deg p = n, deg q = m, q(0) = 1,
 * matching the first n + m + 1 Taylor coefficients. Coefficients lowest degree first. Takes and returns tensors
 * (migration, D-009: `an` was `readonly number[]`, the coefficients `number[]`).
 *
 * @param an - Taylor coefficients `[k]` a_0, a_1, ...
 * @param m - Denominator degree.
 * @param n - Numerator degree (default len(an) - 1 - m).
 * @returns Numerator `[n + 1]` and denominator `[m + 1]` coefficients, lowest first.
 * @throws MdmlError with code `value` without enough coefficients, `shape` unless `an` is a vector, and
 *   `not-differentiable` for a traced value.
 * @concept pade-approximation
 * @example
 * ```ts
 * import { pade } from 'mdml-engine/numerics/approximation'
 * // exp(x) [1/1] = (1 + x/2) / (1 - x/2).
 * pade([1, 1, 0.5], 1) // => { numerator: [1, 0.5], denominator: [1, -0.5] }
 * ```
 */
export function pade(
  taylor: TensorLike,
  m: number,
  n?: number,
): { numerator: Tensor<'float64', [number]>; denominator: Tensor<'float64', [number]> } {
  const an = toVector(taylor, 'pade')
  const num = n ?? an.length - 1 - m
  if (num < 0 || num + m + 1 > an.length) throw new MdmlError('value', 'pade: not enough Taylor coefficients')
  const a = (k: number): number => (k < 0 ? 0 : (an[k] ?? 0))
  let q: number[] = [1]
  if (m > 0) {
    // sum_{j=0}^{m} q_j a_{k-j} = 0 for k = n + 1 .. n + m, with q_0 = 1.
    const mat = new Float64Array(m * m)
    const rhs = new Float64Array(m)
    for (let i = 0; i < m; i++) {
      const k = num + 1 + i
      for (let j = 1; j <= m; j++) mat[i * m + j - 1] = a(k - j)
      rhs[i] = -a(k)
    }
    q = [1, ...toFlat(solve(fromStorage(mat, [m, m], 'float64'), fromStorage(rhs, [m], 'float64')) as Tensor)]
  }
  const p = Array.from({ length: num + 1 }, (_, k) => {
    let s = 0
    for (let j = 0; j <= Math.min(k, m); j++) s += (q[j] ?? 0) * a(k - j)
    return s
  })
  return { numerator: vectorOf(p), denominator: vectorOf(q) }
}

/**
 * Evaluate a continued fraction b_0 + a_1 / (b_1 + a_2 / (b_2 + ...)) by the modified Lentz algorithm (Numerical
 * Recipes 5.2), stopping when a convergent changes by less than tol.
 *
 * @param a - The partial numerators a_j (j >= 1).
 * @param b - The partial denominators b_j (j >= 0).
 * @param options - Tolerance (default 1e-15) and maximum terms (default 10000).
 * @returns The value, the number of terms used, `converged` and `status` (`converged`, or `completed` when `maxTerms`
 *   ran out: the value is then the last convergent, not an answer to `tol`; it used to throw `convergence`).
 * @concept continued-fractions
 * @example
 * ```ts
 * import { continuedFraction } from 'mdml-engine/numerics/approximation'
 * // The golden ratio 1 + 1/(1 + 1/(1 + ...)).
 * Math.abs(continuedFraction(() => 1, () => 1).value - (1 + Math.sqrt(5)) / 2) < 1e-14 // => true
 * ```
 */
export function continuedFraction(
  a: (j: number) => number,
  b: (j: number) => number,
  options: { readonly tol?: number; readonly maxTerms?: number } = {},
): { value: number; terms: number; converged: boolean; status: RunStatus } {
  const tiny = 1e-300
  const tol = options.tol ?? 1e-15
  let f = b(0)
  if (f === 0) f = tiny
  let c = f
  let d = 0
  for (let j = 1; j <= (options.maxTerms ?? 10000); j++) {
    const aj = a(j)
    const bj = b(j)
    d = bj + aj * d
    if (d === 0) d = tiny
    c = bj + aj / c
    if (c === 0) c = tiny
    d = 1 / d
    const delta = c * d
    f *= delta
    if (Math.abs(delta - 1) < tol) return { value: f, terms: j, converged: true, status: 'converged' }
  }
  return { value: f, terms: options.maxTerms ?? 10000, converged: false, status: 'completed' }
}
