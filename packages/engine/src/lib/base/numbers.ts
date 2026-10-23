/**
 * Whether two numbers are close in the numpy sense: `|a - b| <= atol + rtol * |b|`. NaNs are never close; equal
 * infinities are.
 *
 * @param a - First number.
 * @param b - Second number (the reference).
 * @param rtol - Relative tolerance (default 1e-9).
 * @param atol - Absolute tolerance (default 0).
 * @returns `true` when close.
 * @example
 * ```ts
 * import { isClose } from 'whitebox-engine/base'
 * isClose(0.1 + 0.2, 0.3) // => true
 * ```
 */
export function isClose(a: number, b: number, rtol = 1e-9, atol = 0): boolean {
  if (a === b) return true
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false
  return Math.abs(a - b) <= atol + rtol * Math.abs(b)
}

/**
 * Whether every pair of numbers is close (see {@link isClose}).
 *
 * @param a - First list.
 * @param b - Second list (the reference), of the same length.
 * @param rtol - Relative tolerance.
 * @param atol - Absolute tolerance.
 * @returns `true` when the lengths match and every pair is close.
 * @example
 * ```ts
 * import { allClose } from 'whitebox-engine/base'
 * allClose([1, 2], [1, 2 + 1e-12]) // => true
 * ```
 */
export function allClose(a: ArrayLike<number>, b: ArrayLike<number>, rtol = 1e-9, atol = 0): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (!isClose(a[i] ?? Number.NaN, b[i] ?? Number.NaN, rtol, atol)) return false
  return true
}

/**
 * Clamp a number to an interval.
 *
 * @param x - The number.
 * @param lo - Lower bound.
 * @param hi - Upper bound.
 * @returns `x` limited to `[lo, hi]`.
 * @example
 * ```ts
 * import { clamp } from 'whitebox-engine/base'
 * clamp(5, 0, 1) // => 1
 * ```
 */
export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : Math.min(x, hi)
}

/**
 * Kahan-compensated sum, accurate to a few ulps whatever the length.
 *
 * @param xs - The numbers.
 * @returns Their sum.
 * @example
 * ```ts
 * import { kahanSum } from 'whitebox-engine/base'
 * kahanSum([1e16, 1, -1e16]) // => 1
 * ```
 */
export function kahanSum(xs: ArrayLike<number>): number {
  let sum = 0
  let c = 0
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i] ?? 0
    const t = sum + x
    // Neumaier's variant also handles terms larger than the running sum.
    c += Math.abs(sum) >= Math.abs(x) ? sum - t + x : x - t + sum
    sum = t
  }
  return sum + c
}

/**
 * Pairwise (cascade) summation, as numpy's sum: rounding error grows like O(log n) rather than O(n).
 *
 * @param xs - The numbers.
 * @returns Their sum.
 * @example
 * ```ts
 * import { pairwiseSum } from 'whitebox-engine/base'
 * pairwiseSum([1, 2, 3, 4]) // => 10
 * ```
 */
export function pairwiseSum(xs: ArrayLike<number>): number {
  const sumRange = (lo: number, hi: number): number => {
    if (hi - lo <= 8) {
      let s = 0
      for (let i = lo; i < hi; i++) s += xs[i] ?? 0
      return s
    }
    const mid = (lo + hi) >> 1
    return sumRange(lo, mid) + sumRange(mid, hi)
  }
  return sumRange(0, xs.length)
}

/**
 * Maximum of a list by a loop (never spreading into `Math.max`, which fails on long arrays).
 *
 * @param xs - The numbers.
 * @returns The maximum, `-Infinity` for an empty list, `NaN` if any element is NaN.
 * @example
 * ```ts
 * import { maxOf } from 'whitebox-engine/base'
 * maxOf([3, 9, 2]) // => 9
 * ```
 */
export function maxOf(xs: ArrayLike<number>): number {
  let m = -Infinity
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i] ?? Number.NaN
    if (Number.isNaN(x)) return Number.NaN
    if (x > m) m = x
  }
  return m
}

/**
 * Minimum of a list by a loop.
 *
 * @param xs - The numbers.
 * @returns The minimum, `Infinity` for an empty list, `NaN` if any element is NaN.
 * @example
 * ```ts
 * import { minOf } from 'whitebox-engine/base'
 * minOf([3, 9, 2]) // => 2
 * ```
 */
export function minOf(xs: ArrayLike<number>): number {
  let m = Infinity
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i] ?? Number.NaN
    if (Number.isNaN(x)) return Number.NaN
    if (x < m) m = x
  }
  return m
}

/**
 * Index of the first maximum of a list.
 *
 * @param xs - The numbers.
 * @returns The index, or -1 for an empty list.
 * @example
 * ```ts
 * import { argmaxOf } from 'whitebox-engine/base'
 * argmaxOf([3, 9, 9, 2]) // => 1
 * ```
 */
export function argmaxOf(xs: ArrayLike<number>): number {
  let best = -1
  let m = -Infinity
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i] ?? -Infinity
    if (best === -1 || x > m) {
      best = i
      m = x
    }
  }
  return best
}

/**
 * Index of the first minimum of a list.
 *
 * @param xs - The numbers.
 * @returns The index, or -1 for an empty list.
 * @example
 * ```ts
 * import { argminOf } from 'whitebox-engine/base'
 * argminOf([3, 1, 1, 2]) // => 1
 * ```
 */
export function argminOf(xs: ArrayLike<number>): number {
  let best = -1
  let m = Infinity
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i] ?? Infinity
    if (best === -1 || x < m) {
      best = i
      m = x
    }
  }
  return best
}

/**
 * The integers `start, start + step, ...` below `stop`.
 *
 * @param start - First value (or the stop when it is the only argument).
 * @param stop - Exclusive end.
 * @param step - Spacing (default 1).
 * @returns The list.
 * @example
 * ```ts
 * import { range } from 'whitebox-engine/base'
 * range(4) // => [0, 1, 2, 3]
 * range(1, 7, 2) // => [1, 3, 5]
 * ```
 */
export function range(start: number, stop?: number, step = 1): number[] {
  const [lo, hi] = stop === undefined ? [0, start] : [start, stop]
  const out: number[] = []
  if (step > 0) for (let i = lo; i < hi; i += step) out.push(i)
  else if (step < 0) for (let i = lo; i > hi; i += step) out.push(i)
  return out
}

/**
 * n evenly spaced numbers from start to stop inclusive (numpy.linspace on plain numbers); the last equals stop exactly.
 *
 * @param start - First value.
 * @param stop - Last value.
 * @param n - Count (default 50).
 * @returns The list.
 * @example
 * ```ts
 * import { linspaceNumbers } from 'whitebox-engine/base'
 * linspaceNumbers(0, 1, 5) // => [0, 0.25, 0.5, 0.75, 1]
 * ```
 */
export function linspaceNumbers(start: number, stop: number, n = 50): number[] {
  if (n <= 0) return []
  if (n === 1) return [start]
  const step = (stop - start) / (n - 1)
  return Array.from({ length: n }, (_, i) => (i === n - 1 ? stop : start + i * step))
}

/**
 * The `q` quantile of a list with linear interpolation between order statistics (numpy.quantile's default `linear`
 * method). The input is not modified.
 *
 * @param xs - The numbers (NaN entries make the result NaN).
 * @param q - The level in [0, 1].
 * @returns The quantile, `NaN` for an empty list.
 * @example
 * ```ts
 * import { quantileOf } from 'whitebox-engine/base'
 * quantileOf([4, 1, 3, 2], 0.5) // => 2.5
 * quantileOf([1, 2, 3, 4, 5], 0.25) // => 2
 * ```
 */
export function quantileOf(xs: ArrayLike<number>, q: number): number {
  const n = xs.length
  if (n === 0) return Number.NaN
  const sorted = Float64Array.from(xs as ArrayLike<number>)
  for (const v of sorted) if (Number.isNaN(v)) return Number.NaN
  sorted.sort()
  const h = (n - 1) * Math.min(1, Math.max(0, q))
  const lo = Math.floor(h)
  const hi = Math.min(n - 1, lo + 1)
  const a = sorted[lo] ?? Number.NaN
  const b = sorted[hi] ?? Number.NaN
  return a + (h - lo) * (b - a)
}
