/**
 * Pade approximants and continued fractions: the rational function matching the first terms of a power series
 * (`scipy.interpolate.pade`), and evaluation of continued fractions by the modified Lentz algorithm, the form many
 * special functions are computed in.
 */

import { WhiteboxError } from '../../base/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../tensor/index.ts'
import { solve } from '../linalg/index.ts'

/**
 * The Pade approximant [n/m] of a power series (scipy.interpolate.pade): p / q with deg p = n, deg q = m, q(0) = 1,
 * matching the first n + m + 1 Taylor coefficients. Coefficients lowest degree first.
 *
 * @param an - Taylor coefficients a_0, a_1, ...
 * @param m - Denominator degree.
 * @param n - Numerator degree (default len(an) - 1 - m).
 * @returns Numerator and denominator coefficients, lowest first.
 * @concept pade-approximation
 * @example
 * ```ts
 * import { pade } from 'whitebox-engine/numerics/approximation'
 * // exp(x) [1/1] = (1 + x/2) / (1 - x/2).
 * pade([1, 1, 0.5], 1) // => { numerator: [1, 0.5], denominator: [1, -0.5] }
 * ```
 */
export function pade(an: readonly number[], m: number, n?: number): { numerator: number[]; denominator: number[] } {
  const num = n ?? an.length - 1 - m
  if (num < 0 || num + m + 1 > an.length) throw new WhiteboxError('value', 'pade: not enough Taylor coefficients')
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
  return { numerator: p, denominator: q }
}

/**
 * Evaluate a continued fraction b_0 + a_1 / (b_1 + a_2 / (b_2 + ...)) by the modified Lentz algorithm (Numerical
 * Recipes 5.2), stopping when a convergent changes by less than tol.
 *
 * @param a - The partial numerators a_j (j >= 1).
 * @param b - The partial denominators b_j (j >= 0).
 * @param options - Tolerance (default 1e-15) and maximum terms (default 10000).
 * @returns The value and the number of terms used.
 * @concept continued-fractions
 * @example
 * ```ts
 * import { continuedFraction } from 'whitebox-engine/numerics/approximation'
 * // The golden ratio 1 + 1/(1 + 1/(1 + ...)).
 * Math.abs(continuedFraction(() => 1, () => 1).value - (1 + Math.sqrt(5)) / 2) < 1e-14 // => true
 * ```
 */
export function continuedFraction(
  a: (j: number) => number,
  b: (j: number) => number,
  options: { readonly tol?: number; readonly maxTerms?: number } = {},
): { value: number; terms: number } {
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
    if (Math.abs(delta - 1) < tol) return { value: f, terms: j }
  }
  throw new WhiteboxError('convergence', 'continuedFraction: no convergence')
}
