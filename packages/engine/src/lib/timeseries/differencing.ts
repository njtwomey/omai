/**
 * Differencing and its inverse: the lag-`k` difference applied `d` times, the polynomial of a mixed regular and seasonal
 * difference (1 - B)^d (1 - B^s)^D, and integration, which rebuilds a series from its differences and the values that
 * start it. Missing values (NaN) propagate through differences, as in pandas.
 */

import { MdmlError } from '../base/index.ts'
import { polymul } from '../numerics/polynomial/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { asSeries } from './_series.ts'

/**
 * The lag-`lag` difference y_t - y_(t-lag), applied `order` times; the result is `order * lag` values shorter.
 *
 * @param y - The series.
 * @param order - How many times to difference (default 1).
 * @param lag - The lag, 12 for a monthly seasonal difference (default 1).
 * @returns The differenced series.
 * @throws MdmlError with code `value` for a non-positive order or lag.
 * @example
 * ```ts
 * import { difference } from 'mdml-engine/timeseries'
 * difference([1, 4, 9, 16, 25]) // => [3, 5, 7, 9]
 * difference([1, 4, 9, 16, 25], 2) // => [2, 2, 2]
 * difference([1, 2, 5, 6, 9, 10], 1, 2) // => [4, 4, 4, 4]
 * ```
 */
export function difference(y: TensorLike, order = 1, lag = 1): number[] {
  if (!(Number.isInteger(order) && order >= 0 && Number.isInteger(lag) && lag >= 1))
    throw new MdmlError('value', 'difference: order must be a non-negative integer and lag a positive integer')
  let x: number[] = Array.from(asSeries(y, 'difference'))
  for (let k = 0; k < order; k++) x = x.slice(lag).map((v, i) => v - (x[i] ?? 0))
  return x
}

/**
 * The coefficients of the differencing operator (1 - B)^d (1 - B^s)^D, lowest power first, so that applying it to a
 * series is the same as `difference(difference(y, d), D, s)`.
 *
 * @param d - Regular differences.
 * @param seasonal - Seasonal differences D (default 0).
 * @param period - The seasonal period s (default 1).
 * @returns The polynomial `[1, -c_1, ..., -c_k]`, so that y_t = z_t + sum c_j y_(t-j) integrates the differences z.
 * @example
 * ```ts
 * import { differencePolynomial } from 'mdml-engine/timeseries'
 * differencePolynomial(1, 1, 4) // => [1, -1, 0, 0, -1, 1]
 * differencePolynomial(2) // => [1, -2, 1]
 * ```
 */
export function differencePolynomial(d: number, seasonal = 0, period = 1): number[] {
  if (!(Number.isInteger(d) && d >= 0 && Number.isInteger(seasonal) && seasonal >= 0 && Number.isInteger(period)))
    throw new MdmlError('value', 'differencePolynomial: orders must be non-negative integers')
  let poly = [1]
  for (let i = 0; i < d; i++) poly = toFlat(polymul(poly, [1, -1]))
  for (let i = 0; i < seasonal; i++)
    poly = toFlat(polymul(poly, [1, ...new Array<number>(Math.max(period - 1, 0)).fill(0), -1]))
  return poly
}

/**
 * Integrate differences: the series whose lag-`lag` difference of the given order is `z`, started from `initial`
 * (the first `order * lag` values of the series). It is the inverse of {@link difference}.
 *
 * @param z - The differenced series.
 * @param initial - The first `order * lag` values of the original series.
 * @param order - The order of differencing that produced `z` (default 1).
 * @param lag - The lag (default 1).
 * @returns The original series, `initial.length + z.length` values.
 * @throws MdmlError with code `shape` when `initial` has the wrong length.
 * @example
 * ```ts
 * import { difference, integrate } from 'mdml-engine/timeseries'
 * const y = [1, 4, 9, 16, 25]
 * integrate(difference(y, 2), [1, 4], 2) // => [1, 4, 9, 16, 25]
 * ```
 */
export function integrate(z: TensorLike, initial: readonly number[], order = 1, lag = 1): number[] {
  if (initial.length !== order * lag)
    throw new MdmlError('shape', `integrate: expected ${order * lag} initial values, got ${initial.length}`)
  let x: number[] = Array.from(asSeries(z, 'integrate'))
  // Level k of the original series, the k-th difference w_k, starts with the first `lag` values of the k-th difference of
  // `initial`; each level is the running sum (at stride `lag`) of the next one.
  for (let k = order - 1; k >= 0; k--) {
    const out = difference(initial, k, lag).slice(0, lag)
    for (let i = 0; i < x.length; i++) out.push((x[i] ?? 0) + (out[out.length - lag] ?? 0))
    x = out
  }
  return x
}
