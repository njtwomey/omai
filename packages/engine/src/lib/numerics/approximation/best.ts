/**
 * Best (minimax) approximation: the Remez exchange algorithm, which finds the polynomial or rational function
 * minimising the maximum error on an interval by repeatedly solving on a reference set of points and moving the points
 * to the error's extrema until it equioscillates.
 */

import { type Json, maxOf } from '../../base/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat, toVector } from '../../tensor/index.ts'
import {
  type Algorithm,
  numberFromJson,
  numbersFromJson,
  numbersToJson,
  numberToJson,
  type RunStatus,
  runAlgorithm,
} from '../../trace/index.ts'
import { vectorOf } from '../_vector.ts'
import { solve } from '../linalg/index.ts'
import type { ApproximationPath } from './approximation-contract.ts'

/** Result of the Remez algorithm. Holds tensors (migration, D-009: the three vectors were `readonly number[]`). */
export interface MinimaxResult {
  /** Numerator coefficients in the Chebyshev basis of the mapped variable t in [-1, 1], `[m + 1]`. */
  readonly numerator: Tensor<'float64'>
  /** Denominator coefficients (Chebyshev basis, constant term 1), `[n + 1]`; `[1]` for polynomials. */
  readonly denominator: Tensor<'float64'>
  /** The levelled error E (the minimax error at convergence). */
  readonly error: number
  /** The final reference points, where the error equioscillates, `[m + n + 2]`. */
  readonly reference: Tensor<'float64'>
  readonly iterations: number
  readonly converged: boolean
  /** `converged`, `completed` when the iteration cap was reached, or `failed` when the error has too few alternations. */
  readonly status: RunStatus
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

/** What {@link remezAlgorithm} computes: degrees, interval and the iteration controls. */
export interface RemezOptions {
  /** Denominator degree n (default 0: a polynomial). */
  readonly n?: number
  /** The interval (default [-1, 1]). */
  readonly interval?: readonly [number, number]
  /** Stop when the largest error is within this relative amount of the levelled error (default 1e-10). */
  readonly tol?: number
  /** Most exchanges (default 60). */
  readonly maxIterations?: number
}

/**
 * The state of {@link remezAlgorithm}: the reference in the mapped variable t in [-1, 1], the Chebyshev coefficients of
 * numerator and denominator, the levelled error, and how the exchange stands.
 */
export interface RemezState {
  /** Reference points in t. */
  readonly reference: number[]
  readonly numerator: number[]
  readonly denominator: number[]
  /** The levelled error E. */
  readonly error: number
  /** The largest error on the grid at the last exchange (Infinity before the first). */
  readonly maxError: number
  /** Exchanges done. */
  readonly iterations: number
  readonly converged: boolean
  /**
   * True when the exchange could not continue: the error of the current approximation has fewer alternations than the
   * reference needs (the degree does not suit the function). The state then holds the last good approximation.
   */
  readonly failed?: boolean
}

/**
 * The Remez exchange as a stepped algorithm: each step solves for the approximation whose error alternates with equal
 * magnitude E on the reference (m + n + 2 points, iterating the linearisation for rational types), finds the
 * alternating extrema of the error on a fine grid, and moves the reference to them. It logs the reference, E, the
 * largest error and the alternations found, and is done (`converged`) when the largest error is within `tol` of |E|
 * (the equioscillation theorem), `completed` after `maxIterations` exchanges, `failed` when the error has too few
 * alternations (the state keeps the last good approximation). The state serialises.
 *
 * @param f - The function on numbers.
 * @param m - Numerator degree.
 * @param options - Denominator degree, interval, tolerance and iteration cap.
 * @returns The algorithm.
 * @concept minimax-approximation
 * @example
 * ```ts
 * import { remezAlgorithm } from 'omai-engine/numerics/approximation'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(remezAlgorithm((x) => x * x, 1), { maxSteps: 60 })
 * [run.status, Math.abs(run.state.error)] // => ['converged', 0.5]
 * ```
 */
export function remezAlgorithm(
  f: (x: number) => number,
  m: number,
  options: RemezOptions = {},
): Algorithm<RemezState, ApproximationPath> {
  const n = options.n ?? 0
  const [a, b] = options.interval ?? [-1, 1]
  const size = m + n + 2
  const maxIterations = options.maxIterations ?? 60
  const toX = (t: number): number => (a + b) / 2 + ((b - a) / 2) * t
  const grid = Array.from({ length: 4001 }, (_, i) => -Math.cos((i * Math.PI) / 4000))
  return {
    name: 'numerics/remez',
    config: { m, n, interval: [a, b], tol: options.tol ?? 1e-10, maxIterations },
    init: () => ({
      // Initial reference: Chebyshev extrema of degree size - 1.
      reference: Array.from({ length: size }, (_, i) => -Math.cos((i * Math.PI) / (size - 1))),
      numerator: new Array<number>(m + 1).fill(0),
      denominator: [1, ...new Array<number>(n).fill(0)],
      error: 0,
      maxError: Number.POSITIVE_INFINITY,
      iterations: 0,
      converged: false,
    }),
    step: (s, ctx) => {
      let { numerator, denominator, error } = s
      // Solve p(t_i) - f_i q(t_i) + (-1)^i E q_prev(t_i) = 0 (linearised in E for rational types), q_0 = 1.
      for (let inner = 0; inner < (n === 0 ? 1 : 20); inner++) {
        const unknowns = m + 1 + n + 1
        const mat = new Float64Array(size * unknowns)
        const rhs = new Float64Array(size)
        s.reference.forEach((t, i) => {
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
        const sg = Math.sign(v)
        for (let k = 0; k < 60; k++) {
          const m1 = lo + (hi - lo) / 3
          const m2 = hi - (hi - lo) / 3
          if (sg * e(m1) < sg * e(m2)) lo = m1
          else hi = m2
        }
        const tt = (lo + hi) / 2
        return { t: tt, v: e(tt) }
      })
      ctx.log('remez/alternation', refined.length)
      // Keep size points: drop the smallest end extrema while too many.
      let chosen = refined
      while (chosen.length > size) {
        const first = Math.abs(chosen[0]?.v ?? 0)
        const last = Math.abs(chosen[chosen.length - 1]?.v ?? 0)
        chosen = first < last ? chosen.slice(1) : chosen.slice(0, -1)
      }
      if (chosen.length < size) {
        return { ...s, iterations: s.iterations + 1, failed: true }
      }
      const maxError = maxOf(chosen.map((p) => Math.abs(p.v)))
      const reference = chosen.map((p) => p.t)
      const converged = maxError - Math.abs(error) <= (options.tol ?? 1e-10) * Math.max(Math.abs(error), 1e-300)
      ctx.log('remez/error', error)
      ctx.log('remez/maxError', maxError)
      if (ctx.wants('remez/reference')) ctx.log('remez/reference', reference.map(toX))
      return { reference, numerator, denominator, error, maxError, iterations: s.iterations + 1, converged }
    },
    done: (s) =>
      s.converged
        ? 'converged'
        : s.failed === true
          ? 'failed'
          : s.iterations >= maxIterations
            ? 'completed'
            : undefined,
    serialise: (s) => ({
      reference: s.reference,
      numerator: numbersToJson(s.numerator),
      denominator: numbersToJson(s.denominator),
      error: numberToJson(s.error),
      maxError: numberToJson(s.maxError),
      iterations: s.iterations,
      converged: s.converged,
      ...(s.failed === true ? { failed: true } : {}),
    }),
    deserialise: (j) => {
      const o = j as unknown as Record<string, Json>
      return {
        reference: numbersFromJson(o['reference']),
        numerator: numbersFromJson(o['numerator']),
        denominator: numbersFromJson(o['denominator']),
        error: numberFromJson(o['error']),
        maxError: numberFromJson(o['maxError']),
        iterations: o['iterations'] as number,
        converged: o['converged'] as boolean,
        ...(o['failed'] === true ? { failed: true } : {}),
      }
    },
  }
}

/**
 * The best (minimax) approximation of f on [a, b] by a polynomial of degree m, or a rational function of type (m, n),
 * by the Remez exchange algorithm: on a reference of m + n + 2 points, solve for the approximation whose error
 * alternates in sign with equal magnitude E, move the reference to the alternating extrema of the error, and repeat
 * until the largest error equals |E| (the equioscillation theorem's characterisation of the best approximation).
 * For FIR filter design (weighted bands, linear-phase types) see `remez` of `signal`, which is scipy's Parks-McClellan.
 * {@link remezAlgorithm} run to the end.
 *
 * @param f - The function on numbers.
 * @param m - Numerator degree.
 * @param options - Denominator degree n (default 0), interval (default [-1, 1]), tolerance (1e-10) and iteration cap.
 * @returns Coefficients, the levelled error and the reference (the vectors are tensors; they were `number[]`). A degree
 *   that does not suit the function (the error has too few alternations) ends with status `failed` and the last good
 *   approximation, instead of throwing.
 * @concept minimax-approximation
 * @example
 * ```ts
 * import { remez } from 'omai-engine/numerics/approximation'
 * // The best linear approximation of x^2 on [-1, 1] is 1/2, with error 1/2.
 * Math.abs(remez((x) => x * x, 1).error) // => 0.5
 * ```
 */
export function remez(f: (x: number) => number, m: number, options: RemezOptions = {}): MinimaxResult {
  const [a, b] = options.interval ?? [-1, 1]
  const run = runAlgorithm(remezAlgorithm(f, m, options), { maxSteps: options.maxIterations ?? 60 })
  const s = run.state
  const r = (t: number): number => series(s.numerator, t) / series(s.denominator, t)
  return {
    numerator: vectorOf(s.numerator),
    denominator: vectorOf(s.denominator),
    error: s.error,
    reference: vectorOf(s.reference.map((t) => (a + b) / 2 + ((b - a) / 2) * t)),
    iterations: s.converged || s.failed === true ? s.iterations : (options.maxIterations ?? 60),
    converged: s.converged,
    status: s.converged ? 'converged' : s.failed === true ? 'failed' : 'completed',
    evaluate: (x) => r((2 * x - a - b) / (b - a)),
  }
}

/**
 * The Chebyshev-basis evaluation used by {@link remez}, on values: sum_k c_k T_k(t). Takes and returns tensors
 * (migration, D-009: `c` was `readonly number[]`, `t` a `readonly number[]` or a value, the result `number[]`); a traced
 * `t` is rejected where it used to be read at its primal.
 *
 * @param c - Coefficients `[k]`.
 * @param t - Points `[m]` in [-1, 1].
 * @returns Values `[m]`.
 * @throws OmaiError with code `shape` unless `c` and `t` are vectors, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { chebyshevSeries } from 'omai-engine/numerics/approximation'
 * chebyshevSeries([0, 0, 1], [0.5]) // => [-0.5]
 * ```
 */
export function chebyshevSeries(c: TensorLike, t: TensorLike): Tensor<'float64', [number]> {
  const coefficients = toVector(c, 'chebyshevSeries')
  return vectorOf(toVector(t, 'chebyshevSeries').map((v) => series(coefficients, v)))
}
