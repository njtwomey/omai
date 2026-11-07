/**
 * Small helpers on plain numbers and number lists, used where tensors would be overkill: closeness tests in the numpy
 * sense, clamping, accurate summation (Kahan and pairwise), loop-based extrema and argmax that never spread into
 * `Math.max`, stable log-sum-exp and softmax, ranges, quantiles, means and variances, and bit counts, with the few constants that more than one domain needs.
 */

/** The Euler-Mascheroni constant, the limit of the harmonic sum minus the logarithm. */
export const EULER_GAMMA: number = 0.5772156649015329

/** The natural logarithm of 2 pi, the constant of the Gaussian log density (as `Math.log(2 * Math.PI)`). */
export const LOG_2PI: number = Math.log(2 * Math.PI)

/** Half the natural logarithm of 2 pi (as `0.5 * Math.log(2 * Math.PI)`): the standard normal log density at 0 is minus this. */
export const LOG_SQRT_2PI: number = 0.5 * Math.log(2 * Math.PI)

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
 * import { isClose } from 'mdml-engine/base'
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
 * import { allClose } from 'mdml-engine/base'
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
 * import { clamp } from 'mdml-engine/base'
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
 * import { kahanSum } from 'mdml-engine/base'
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
 * import { pairwiseSum } from 'mdml-engine/base'
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
 * import { maxOf } from 'mdml-engine/base'
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
 * import { minOf } from 'mdml-engine/base'
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
 * log(sum exp(x_i)) of a list, computed stably by shifting by the maximum (the number-array kernel behind log-domain
 * algorithms; differentiable code uses the tensor `logsumexp`).
 *
 * @param xs - The numbers.
 * @returns The log-sum-exp, `-Infinity` for an empty list or all `-Infinity`, `NaN` if any element is NaN.
 * @example
 * ```ts
 * import { logSumExpOf } from 'mdml-engine/base'
 * logSumExpOf([1000, 1000]) === 1000 + Math.LN2 // => true
 * ```
 */
export function logSumExpOf(xs: ArrayLike<number>): number {
  const m = maxOf(xs)
  if (Number.isNaN(m) || m === -Infinity) return m
  if (m === Infinity) return Infinity
  let s = 0
  for (let i = 0; i < xs.length; i++) s += Math.exp((xs[i] ?? 0) - m)
  return m + Math.log(s)
}

/**
 * Index of the first maximum of a list.
 *
 * @param xs - The numbers.
 * @returns The index, or -1 for an empty list.
 * @example
 * ```ts
 * import { argmaxOf } from 'mdml-engine/base'
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
 * import { argminOf } from 'mdml-engine/base'
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
 * import { range } from 'mdml-engine/base'
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
 * import { linspaceNumbers } from 'mdml-engine/base'
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
 * The softmax of a list of log-weights: `exp(xs) / sum(exp(xs))`, computed stably.
 *
 * @param xs - Unnormalised log probabilities.
 * @returns Probabilities summing to one.
 * @example
 * ```ts
 * import { softmaxOf } from 'mdml-engine/base'
 * softmaxOf([0, 0]) // => [0.5, 0.5]
 * ```
 */
export function softmaxOf(xs: ArrayLike<number>): number[] {
  const z = logSumExpOf(xs)
  return Array.from(xs as ArrayLike<number>, (x) => Math.exp(x - z))
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
 * import { quantileOf } from 'mdml-engine/base'
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

/**
 * log(exp(a) + exp(b)) on plain numbers, without overflow (numpy.logaddexp for scalars; the tensor op is
 * `numerics/special` `logaddexp`).
 *
 * @param a - First log value.
 * @param b - Second log value.
 * @returns The log of the sum; `-Infinity` when both are `-Infinity`.
 * @example
 * ```ts
 * import { logAddExp } from 'mdml-engine/base'
 * logAddExp(0, 0) === Math.LN2 // => true
 * logAddExp(-Infinity, 2) // => 2
 * ```
 */
export function logAddExp(a: number, b: number): number {
  if (a === Number.NEGATIVE_INFINITY) return b
  if (b === Number.NEGATIVE_INFINITY) return a
  if (a === Number.POSITIVE_INFINITY || b === Number.POSITIVE_INFINITY) return Number.POSITIVE_INFINITY
  return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b))
}

/**
 * Arithmetic mean of a list (pairwise summation), `NaN` for an empty list.
 *
 * @param xs - The numbers.
 * @returns The mean.
 * @example
 * ```ts
 * import { meanOf } from 'mdml-engine/base'
 * meanOf([1, 2, 6]) // => 3
 * ```
 */
export function meanOf(xs: ArrayLike<number>): number {
  return xs.length === 0 ? Number.NaN : pairwiseSum(xs) / xs.length
}

/**
 * Variance of a list about its mean, dividing by `n - ddof` (numpy's `var`), `NaN` when `n <= ddof`.
 *
 * @param xs - The numbers.
 * @param ddof - Delta degrees of freedom (default 0, the population variance; 1 for the sample variance).
 * @returns The variance.
 * @example
 * ```ts
 * import { varianceOf } from 'mdml-engine/base'
 * varianceOf([1, 2, 3, 4], 1) // => 1.6666666666666667
 * ```
 */
export function varianceOf(xs: ArrayLike<number>, ddof = 0): number {
  const n = xs.length
  if (n <= ddof) return Number.NaN
  const m = meanOf(xs)
  const d = new Float64Array(n)
  for (let i = 0; i < n; i++) d[i] = ((xs[i] ?? 0) - m) ** 2
  return pairwiseSum(d) / (n - ddof)
}

/**
 * The number of set bits of a non-negative 32-bit integer (the size of a subset given as a bitmask).
 *
 * @param mask - The bitmask.
 * @returns How many bits are 1.
 * @example
 * ```ts
 * import { popcount } from 'mdml-engine/base'
 * popcount(0b1011) // => 3
 * popcount(0) // => 0
 * ```
 */
export function popcount(mask: number): number {
  let c = 0
  let x = mask
  while (x !== 0) {
    x &= x - 1
    c += 1
  }
  return c
}
