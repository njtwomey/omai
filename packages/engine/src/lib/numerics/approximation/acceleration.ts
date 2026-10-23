import { WhiteboxError } from '../../base/index.ts'

/**
 * Richardson extrapolation of estimates A(h), A(h / r), A(h / r^2), ... whose error expands in powers h^p, h^2p, ...:
 * the tableau R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / (r^(p j) - 1).
 *
 * @param estimates - Estimates at successively smaller steps.
 * @param options - Step ratio r (default 2) and leading error order p (default 2, as for the trapezoid rule).
 * @returns The extrapolated value and the tableau.
 * @concept extrapolation
 * @example
 * ```ts
 * import { richardson } from 'whitebox-engine/numerics/approximation'
 * // Central differences of sin at 1 with h = 0.4, 0.2, 0.1 extrapolate to cos(1).
 * const d = (h) => (Math.sin(1 + h) - Math.sin(1 - h)) / (2 * h)
 * Math.abs(richardson([d(0.4), d(0.2), d(0.1)]).value - Math.cos(1)) < 1e-7 // => true
 * ```
 */
export function richardson(
  estimates: readonly number[],
  options: { readonly ratio?: number; readonly order?: number } = {},
): { value: number; table: number[][] } {
  if (estimates.length === 0) throw new WhiteboxError('value', 'richardson: no estimates')
  const r = options.ratio ?? 2
  const p = options.order ?? 2
  const table: number[][] = []
  estimates.forEach((e, i) => {
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
 * geometric error term.
 *
 * @param s - A convergent sequence.
 * @returns The accelerated sequence (two shorter).
 * @example
 * ```ts
 * import { aitken } from 'whitebox-engine/numerics/approximation'
 * // Partial sums of a geometric series become exact.
 * aitken([1, 1.5, 1.75]) // => [2]
 * ```
 */
export function aitken(s: readonly number[]): number[] {
  const out: number[] = []
  for (let n = 0; n + 2 < s.length; n++) {
    const a = s[n] ?? 0
    const b = s[n + 1] ?? 0
    const c = s[n + 2] ?? 0
    const d = c - 2 * b + a
    out.push(d === 0 ? c : a - ((b - a) * (b - a)) / d)
  }
  return out
}

/**
 * Wynn's epsilon algorithm: the epsilon table eps_{k+1}^(n) = eps_{k-1}^(n+1) + 1 / (eps_k^(n+1) - eps_k^(n)); its even
 * columns are Shanks transforms of the sequence. The best estimate is the last even-column entry.
 *
 * @param s - A sequence (partial sums).
 * @returns The estimate and the even columns (column 0 is s).
 * @example
 * ```ts
 * import { wynnEpsilon } from 'whitebox-engine/numerics/approximation'
 * // Partial sums of 1 - 1/2 + 1/3 - ... converge slowly to ln 2.
 * const s = []
 * for (let n = 1, acc = 0; n <= 12; n++) s.push((acc += (n % 2 ? 1 : -1) / n))
 * Math.abs(wynnEpsilon(s).value - Math.LN2) < 1e-9 // => true
 * ```
 */
export function wynnEpsilon(s: readonly number[]): { value: number; columns: number[][] } {
  let prev: number[] = new Array<number>(s.length + 1).fill(0)
  let curr: number[] = [...s]
  const columns: number[][] = [[...s]]
  let value = s[s.length - 1] ?? Number.NaN
  for (let k = 1; curr.length > 1; k++) {
    const next: number[] = []
    for (let n = 0; n + 1 < curr.length; n++) {
      const diff = (curr[n + 1] ?? 0) - (curr[n] ?? 0)
      next.push((prev[n + 1] ?? 0) + (diff === 0 ? Number.POSITIVE_INFINITY : 1 / diff))
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
 * The Shanks transformation of order k (column 2k of Wynn's table).
 *
 * @param s - A sequence.
 * @param k - Order (default 1).
 * @returns The transformed sequence.
 * @example
 * ```ts
 * import { shanks } from 'whitebox-engine/numerics/approximation'
 * shanks([1, 1.5, 1.75, 1.875]) // => [2, 2]
 * ```
 */
export function shanks(s: readonly number[], k = 1): number[] {
  return wynnEpsilon(s).columns[k] ?? []
}

/**
 * The Euler transform of an alternating series sum (-1)^n a_n: sum_k (-1)^k Delta^k a_0 / 2^(k+1), which converges
 * much faster when a_n is smooth and decreasing.
 *
 * @param a - The magnitudes a_0, a_1, ... of the alternating terms.
 * @returns The accelerated sum.
 * @example
 * ```ts
 * import { eulerTransform } from 'whitebox-engine/numerics/approximation'
 * // pi / 4 = 1 - 1/3 + 1/5 - ...
 * Math.abs(eulerTransform(Array.from({ length: 30 }, (_, n) => 1 / (2 * n + 1))) - Math.PI / 4) < 1e-9 // => true
 * ```
 */
export function eulerTransform(a: readonly number[]): number {
  let diffs = [...a]
  let sum = 0
  for (let k = 0; k < a.length; k++) {
    sum += ((k % 2 === 0 ? 1 : -1) * (diffs[0] ?? 0)) / 2 ** (k + 1)
    diffs = diffs.slice(1).map((v, i) => v - (diffs[i] ?? 0))
  }
  return sum
}
