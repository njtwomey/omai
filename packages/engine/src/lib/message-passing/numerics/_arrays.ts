/**
 * The one place where the plain number arrays of message passing (a message, a moment vector, a flat matrix, a mutable
 * accumulator) are created from the tensor library: zeros, a constant, a unit vector, the identity and an index range are
 * tensor creations read out once with `toFlat`, never hand-filled arrays. Messages are `number[]` by design of the core API
 * (the hot loops of the engine and of the quadrature kernels stay free of tensors), so this is the edge: build with tensor
 * creation, read out here, mutate the result as a plain accumulator. Internal to message passing.
 */

import { add, arange, div, eye, full, mul, oneHot, toFlat, type Value } from '../../tensor/index.ts'

/**
 * `n` zeros, as a fresh mutable array (a message or accumulator to fill).
 *
 * @param n - The length.
 * @returns The zeros.
 */
export function zeroArray(n: number): number[] {
  return toFlat(full([n], 0))
}

/**
 * `n` copies of a constant, as a fresh mutable array.
 *
 * @param n - The length.
 * @param value - The constant.
 * @returns The array.
 */
export function constantArray(n: number, value: number): number[] {
  return toFlat(full([n], value))
}

/**
 * The indicator of one index: 1 at `at`, 0 elsewhere (all zeros when `at` is outside `[0, n)`).
 *
 * @param n - The length.
 * @param at - The index of the one.
 * @returns The unit vector.
 */
export function unitArray(n: number, at: number): number[] {
  return toFlat(oneHot(at, n))
}

/**
 * The `d` by `d` identity matrix, flat in row-major order.
 *
 * @param d - The dimension.
 * @returns The `d * d` entries.
 */
export function identityArray(d: number): number[] {
  return toFlat(eye(d))
}

/**
 * The indices `0, 1, ..., n - 1`.
 *
 * @param n - The count.
 * @returns The range.
 */
export function rangeArray(n: number): number[] {
  return toFlat(arange(n))
}

/**
 * `n` NaNs: the moments of a message that has no density.
 *
 * @param n - The length.
 * @returns The array.
 */
export function nanArray(n: number): number[] {
  return toFlat(full([n], Number.NaN))
}

/**
 * The values of a tensor as a `Float64Array`, for the kernels (`_dense.ts`) whose results are typed arrays.
 *
 * @param t - A concrete value.
 * @returns A fresh copy of the elements.
 */
export function float64Of(t: Value): Float64Array {
  return Float64Array.from(toFlat(t))
}

/**
 * `n` equally spaced points from `lo` to `hi`, as `lo + (hi - lo) * i / (n - 1)` for `i = 0..n-1` (the grid formula of the
 * quadrature kernels, in this operation order so the points are the same bits as the hand loop `lo + ((hi - lo) * i) / (n - 1)`;
 * not `linspace`, whose step is rounded first).
 *
 * @param lo - The first point.
 * @param hi - The last point.
 * @param n - The number of points (at least 2).
 * @returns The grid as a `Float64Array`.
 */
export function gridArray(lo: number, hi: number, n: number): Float64Array {
  return float64Of(add(lo, div(mul(arange(n), hi - lo), n - 1)))
}
