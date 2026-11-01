/**
 * Best (minimax) approximation: the Remez exchange algorithm, which finds the polynomial or rational function
 * minimising the maximum error on an interval by repeatedly solving on a reference set of points and moving the points
 * to the error's extrema until it equioscillates.
 */

import { MdmlError, maxOf } from '../../base/index.ts'
import { fromStorage, primalOf, type Tensor, toFlat, type Value } from '../../tensor/index.ts'
import { solve } from '../linalg/index.ts'

/** Result of the Remez algorithm. */
export interface MinimaxResult {
  /** Numerator coefficients in the Chebyshev basis of the mapped variable t in [-1, 1]. */
  readonly numerator: readonly number[]
  /** Denominator coefficients (Chebyshev basis, constant term 1); [1] for polynomials. */
  readonly denominator: readonly number[]
  /** The levelled error E (the minimax error at convergence). */
  readonly error: number
  /** The final reference points, where the error equioscillates. */
  readonly reference: readonly number[]
  readonly iterations: number
  readonly converged: boolean
  /** Evaluate the approximation at numbers. */
  readonly evaluate: (x: number) => number
}

function chebT(k: number, t: number): number {
  if (k === 0) return 1
  let p0 = 1
  let p1 = t
  for (let i = 2; i <= k; i++) [p0, p1] = [p1, 2 * t * p1 - p0]
  return p1
}

const series = (c: readonly number[], t: number): number => c.reduce((s, ck, k) => s + ck * chebT(k, t), 0)

/**
 * The best (minimax) approximation of f on [a, b] by a polynomial of degree m, or a rational function of type (m, n),
 * by the Remez exchange algorithm: on a reference of m + n + 2 points, solve for the approximation whose error
 * alternates in sign with equal magnitude E, move the reference to the alternating extrema of the error, and repeat
 * until the largest error equals |E| (the equioscillation theorem's characterisation of the best approximation).
 * For FIR filter design (weighted bands, linear-phase types) see `remez` of `signal`, which is scipy's Parks-McClellan.
 *
 * @param f - The function on numbers.
 * @param m - Numerator degree.
 * @param options - Denominator degree n (default 0), interval (default [-1, 1]), tolerance (1e-10) and iteration cap.
 * @returns Coefficients, the levelled error and the reference.
 * @concept minimax-approximation
 * @example
 * ```ts
 * import { remez } from 'mdml-engine/numerics/approximation'
 * // The best linear approximation of x^2 on [-1, 1] is 1/2, with error 1/2.
 * Math.abs(remez((x) => x * x, 1).error) // => 0.5
 * ```
 */
export function remez(
  f: (x: number) => number,
  m: number,
  options: {
    readonly n?: number
    readonly interval?: readonly [number, number]
    readonly tol?: number
    readonly maxIterations?: number
  } = {},
): MinimaxResult {
  const n = options.n ?? 0
  const [a, b] = options.interval ?? [-1, 1]
  const size = m + n + 2
  const toX = (t: number): number => (a + b) / 2 + ((b - a) / 2) * t
  // Initial reference: Chebyshev extrema of degree size - 1.
  let reference = Array.from({ length: size }, (_, i) => -Math.cos((i * Math.PI) / (size - 1)))
  let numerator: number[] = new Array<number>(m + 1).fill(0)
  let denominator: number[] = [1, ...new Array<number>(n).fill(0)]
  let error = 0
  const grid = Array.from({ length: 4001 }, (_, i) => -Math.cos((i * Math.PI) / 4000))
  for (let it = 1; it <= (options.maxIterations ?? 60); it++) {
    // Solve p(t_i) - f_i q(t_i) + (-1)^i E q_prev(t_i) = 0 (linearised in E for rational types), q_0 = 1.
    for (let inner = 0; inner < (n === 0 ? 1 : 20); inner++) {
      const unknowns = m + 1 + n + 1
      const mat = new Float64Array(size * unknowns)
      const rhs = new Float64Array(size)
      reference.forEach((t, i) => {
        const fi = f(toX(t))
        const qPrev = series(denominator, t)
        for (let k = 0; k <= m; k++) mat[i * unknowns + k] = chebT(k, t)
        for (let k = 1; k <= n; k++) mat[i * unknowns + m + k] = -fi * chebT(k, t)
        mat[i * unknowns + m + n + 1] = (i % 2 === 0 ? 1 : -1) * qPrev
        rhs[i] = fi
      })
      const sol = toFlat(
        solve(fromStorage(mat, [size, unknowns], 'float64'), fromStorage(rhs, [size], 'float64')) as Tensor,
      )
      numerator = sol.slice(0, m + 1)
      denominator = [1, ...sol.slice(m + 1, m + 1 + n)]
      const next = sol[m + n + 1] ?? 0
      const settled = Math.abs(next - error) <= 1e-14 * Math.max(1, Math.abs(next))
      error = next
      if (settled) break
    }
    const r = (t: number): number => series(numerator, t) / series(denominator, t)
    const e = (t: number): number => f(toX(t)) - r(t)
    // Alternating extrema of the error: the extreme point of each sign run on the grid, refined by golden section.
    const values = grid.map(e)
    const runs: { t: number; v: number }[] = []
    values.forEach((v, i) => {
      const t = grid[i] ?? 0
      const last = runs[runs.length - 1]
      if (last === undefined || Math.sign(v) !== Math.sign(last.v) || v === 0) runs.push({ t, v })
      else if (Math.abs(v) > Math.abs(last.v)) runs[runs.length - 1] = { t, v }
    })
    const refined = runs.map(({ t, v }) => {
      let lo = Math.max(-1, t - 2e-3)
      let hi = Math.min(1, t + 2e-3)
      const s = Math.sign(v)
      for (let k = 0; k < 60; k++) {
        const m1 = lo + (hi - lo) / 3
        const m2 = hi - (hi - lo) / 3
        if (s * e(m1) < s * e(m2)) lo = m1
        else hi = m2
      }
      const tt = (lo + hi) / 2
      return { t: tt, v: e(tt) }
    })
    // Keep size points: drop the smallest end extrema while too many.
    let chosen = refined
    while (chosen.length > size) {
      const first = Math.abs(chosen[0]?.v ?? 0)
      const last = Math.abs(chosen[chosen.length - 1]?.v ?? 0)
      chosen = first < last ? chosen.slice(1) : chosen.slice(0, -1)
    }
    if (chosen.length < size)
      throw new MdmlError('convergence', 'remez: the error has too few alternations (try a different degree)')
    const maxError = maxOf(chosen.map((p) => Math.abs(p.v)))
    reference = chosen.map((p) => p.t)
    if (maxError - Math.abs(error) <= (options.tol ?? 1e-10) * Math.max(Math.abs(error), 1e-300)) {
      return {
        numerator,
        denominator,
        error,
        reference: reference.map(toX),
        iterations: it,
        converged: true,
        evaluate: (x) => r((2 * x - a - b) / (b - a)),
      }
    }
  }
  const r = (t: number): number => series(numerator, t) / series(denominator, t)
  return {
    numerator,
    denominator,
    error,
    reference: reference.map(toX),
    iterations: options.maxIterations ?? 60,
    converged: false,
    evaluate: (x) => r((2 * x - a - b) / (b - a)),
  }
}

/**
 * The Chebyshev-basis evaluation used by {@link remez}, on values: sum_k c_k T_k(t).
 *
 * @param c - Coefficients.
 * @param t - Points in [-1, 1].
 * @returns Values.
 * @example
 * ```ts
 * import { chebyshevSeries } from 'mdml-engine/numerics/approximation'
 * chebyshevSeries([0, 0, 1], [0.5]) // => [-0.5]
 * ```
 */
export function chebyshevSeries(c: readonly number[], t: readonly number[] | Value): number[] {
  const ts = Array.isArray(t) ? (t as readonly number[]) : toFlat(primalOf(t as Value))
  return ts.map((v) => series(c, v))
}
