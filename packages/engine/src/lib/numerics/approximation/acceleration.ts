/**
 * Convergence acceleration for sequences and series: Richardson extrapolation of step-size estimates, Aitken's
 * delta-squared process, Wynn's epsilon algorithm and the Shanks transformation built on it, and the Euler transform
 * for alternating series, each turning slowly converging partial results into a much better limit estimate.
 */

import { MdmlError } from '../../base/index.ts'
import { type Tensor, type TensorLike, toVector } from '../../tensor/index.ts'
import { vectorOf } from '../_vector.ts'

/**
 * Richardson extrapolation of estimates A(h), A(h / r), A(h / r^2), ... whose error expands in powers h^p, h^2p, ...:
 * the tableau R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / (r^(p j) - 1). Takes the estimates as a tensor-like
 * (migration, D-009: was `readonly number[]`); the tableau is a ragged triangle and stays a list of lists.
 *
 * @param estimates - Estimates `[k]` at successively smaller steps.
 * @param options - Step ratio r (default 2) and leading error order p (default 2, as for the trapezoid rule).
 * @returns The extrapolated value and the tableau.
 * @throws MdmlError with code `value` for no estimates, `shape` unless `estimates` is a vector, and
 *   `not-differentiable` for a traced value.
 * @concept extrapolation
 * @example
 * ```ts
 * import { richardson } from 'mdml-engine/numerics/approximation'
 * // Central differences of sin at 1 with h = 0.4, 0.2, 0.1 extrapolate to cos(1).
 * const d = (h) => (Math.sin(1 + h) - Math.sin(1 - h)) / (2 * h)
 * Math.abs(richardson([d(0.4), d(0.2), d(0.1)]).value - Math.cos(1)) < 1e-7 // => true
 * ```
 */
export function richardson(
  estimates: TensorLike,
  options: { readonly ratio?: number; readonly order?: number } = {},
): { value: number; table: number[][] } {
  const values = toVector(estimates, 'richardson')
  if (values.length === 0) throw new MdmlError('value', 'richardson: no estimates')
  const r = options.ratio ?? 2
  const p = options.order ?? 2
  const table: number[][] = []
  values.forEach((e, i) => {
    const row = [e]
    for (let j = 1; j <= i; j++) {
      const prev = table[i - 1]?.[j - 1] ?? 0
      row.push((row[j - 1] ?? 0) + ((row[j - 1] ?? 0) - prev) / (r ** (p * j) - 1))
    }
    table.push(row)
  })
  return { value: table.at(-1)?.at(-1) ?? Number.NaN, table }
}

/**
 * Aitken's delta-squared transformation: s'_n = s_n - (s_{n+1} - s_n)^2 / (s_{n+2} - 2 s_{n+1} + s_n), which removes a
 * geometric error term. Takes and returns tensors (migration, D-009: was `readonly number[]` to `number[]`).
 *
 * @param s - A convergent sequence `[k]`.
 * @returns The accelerated sequence `[k - 2]` (two shorter).
 * @throws MdmlError with code `shape` unless `s` is a vector, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { aitken } from 'mdml-engine/numerics/approximation'
 * // Partial sums of a geometric series become exact.
 * aitken([1, 1.5, 1.75]) // => [2]
 * ```
 */
export function aitken(s: TensorLike): Tensor<'float64', [number]> {
  const seq = toVector(s, 'aitken')
  const out: number[] = []
  for (let n = 0; n + 2 < seq.length; n++) {
    const a = seq[n] ?? 0
    const b = seq[n + 1] ?? 0
    const c = seq[n + 2] ?? 0
    const d = c - 2 * b + a
    out.push(d === 0 ? c : a - ((b - a) * (b - a)) / d)
  }
  return vectorOf(out)
}

/**
 * Wynn's epsilon algorithm: the epsilon table eps_{k+1}^(n) = eps_{k-1}^(n+1) + 1 / (eps_k^(n+1) - eps_k^(n)); its even
 * columns are Shanks transforms of the sequence. The best estimate is the last even-column entry. Takes the sequence as
 * a tensor-like (migration, D-009: was `readonly number[]`); the columns shrink by one entry each and stay a list of
 * lists.
 *
 * @param s - A sequence `[k]` (partial sums).
 * @returns The estimate and the even columns (column 0 is s).
 * @throws MdmlError with code `shape` unless `s` is a vector, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { wynnEpsilon } from 'mdml-engine/numerics/approximation'
 * // Partial sums of 1 - 1/2 + 1/3 - ... converge slowly to ln 2.
 * const s = []
 * for (let n = 1, acc = 0; n <= 12; n++) s.push((acc += (n % 2 ? 1 : -1) / n))
 * Math.abs(wynnEpsilon(s).value - Math.LN2) < 1e-8 // => true
 * ```
 */
export function wynnEpsilon(sequence: TensorLike): { value: number; columns: number[][] } {
  return wynnKernel(toVector(sequence, 'wynnEpsilon'))
}

function wynnKernel(s: readonly number[]): { value: number; columns: number[][] } {
  let prev: number[] = new Array<number>(s.length + 1).fill(0)
  let curr: number[] = [...s]
  const columns: number[][] = [[...s]]
  let value = s[s.length - 1] ?? Number.NaN
  for (let k = 1; curr.length > 1; k++) {
    const next: number[] = []
    for (let n = 0; n + 1 < curr.length; n++) {
      const lo = curr[n] ?? 0
      const hi = curr[n + 1] ?? 0
      // Equal finite entries make the next one infinite; two equal infinities (a constant sequence) make no contribution.
      const reciprocal = lo === hi ? (Number.isFinite(lo) ? Number.POSITIVE_INFINITY : 0) : 1 / (hi - lo)
      next.push((prev[n + 1] ?? 0) + reciprocal)
    }
    prev = curr
    curr = next
    if (k % 2 === 0) {
      columns.push([...curr])
      const last = curr[curr.length - 1]
      if (last !== undefined && Number.isFinite(last)) value = last
    }
  }
  return { value, columns }
}

/**
 * The Shanks transformation of order k (column 2k of Wynn's table). Takes and returns tensors (migration, D-009: was
 * `readonly number[]` to `number[]`).
 *
 * @param s - A sequence `[n]`.
 * @param k - Order (default 1).
 * @returns The transformed sequence.
 * @throws MdmlError with code `shape` unless `s` is a vector, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { shanks } from 'mdml-engine/numerics/approximation'
 * shanks([1, 1.5, 1.75, 1.875]) // => [2, 2]
 * ```
 */
export function shanks(s: TensorLike, k = 1): Tensor<'float64', [number]> {
  return vectorOf(wynnKernel(toVector(s, 'shanks')).columns[k] ?? [])
}

/**
 * The Euler transform of an alternating series sum (-1)^n a_n: sum_k (-1)^k Delta^k a_0 / 2^(k+1), which converges
 * much faster when a_n is smooth and decreasing. Takes a tensor-like (migration, D-009: was `readonly number[]`).
 *
 * @param a - The magnitudes `[k]` a_0, a_1, ... of the alternating terms.
 * @returns The accelerated sum.
 * @throws MdmlError with code `shape` unless `a` is a vector, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { eulerTransform } from 'mdml-engine/numerics/approximation'
 * // pi / 4 = 1 - 1/3 + 1/5 - ...
 * Math.abs(eulerTransform(Array.from({ length: 30 }, (_, n) => 1 / (2 * n + 1))) - Math.PI / 4) < 1e-9 // => true
 * ```
 */
export function eulerTransform(a: TensorLike): number {
  const magnitudes = toVector(a, 'eulerTransform')
  let diffs = [...magnitudes]
  let sum = 0
  for (let k = 0; k < magnitudes.length; k++) {
    sum += ((k % 2 === 0 ? 1 : -1) * (diffs[0] ?? 0)) / 2 ** (k + 1)
    diffs = diffs.slice(1).map((v, i) => v - (diffs[i] ?? 0))
  }
  return sum
}
