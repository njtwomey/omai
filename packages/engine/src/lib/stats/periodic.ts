/**
 * Periodic data (clock times, angles, phases) without a circular distribution: wrapping a value into one period, the copies
 * of a value a period or more away (the data a normal is scored on when the copies are summed in one log-sum-exp: shifting the
 * data instead of the means), the grid of copies of a point on a torus, the log density of the wrapped normal as that sum, and
 * the error of keeping only a few copies. The wrapped normal's density is N(x; mu + k period, sigma^2) summed over every integer k;
 * term by term N(x; mu + k period, sigma^2) = N(x - k period; mu, sigma^2), so the sum is an ordinary normal scored at the copies.
 */

import { logSumExpOf, OmaiError } from '../base/index.ts'

/**
 * A value wrapped into one period: `x` modulo `period`, in `[0, period)`.
 *
 * @param x - The value (an hour of the day, an angle).
 * @param period - The period (24, 2 pi).
 * @returns The value in `[0, period)`.
 * @example
 * ```ts
 * import { wrapToPeriod } from 'omai-engine/stats'
 * wrapToPeriod(-0.5, 24) // => 23.5
 * ```
 */
export function wrapToPeriod(x: number, period: number): number {
  if (!(period > 0)) throw new OmaiError('value', `wrapToPeriod: the period must be positive, got ${period}`)
  const r = x - period * Math.floor(x / period)
  // Rounding can leave a tiny negative value's result equal to the period itself.
  return r >= period ? 0 : r
}

/**
 * The copies of a value a whole number of periods away: `x - k period, ..., x, ..., x + k period`. Scoring a density at all of
 * them in one log-sum-exp is the wrapped density; adding them as extra rows of a dataset is not (it scores each copy
 * separately).
 *
 * @index function The copies of a periodic value a period or more away, to score in one log-sum-exp.
 * @param x - The value.
 * @param period - The period.
 * @param k - How many periods each way (default 1: three copies).
 * @returns The `2k + 1` copies in increasing order.
 * @example
 * ```ts
 * import { periodicCopies } from 'omai-engine/stats'
 * periodicCopies(23, 24) // => [-1, 23, 47]
 * ```
 */
export function periodicCopies(x: number, period: number, k = 1): number[] {
  if (!(period > 0)) throw new OmaiError('value', `periodicCopies: the period must be positive, got ${period}`)
  return Array.from({ length: 2 * k + 1 }, (_, j) => x + (j - k) * period)
}

/**
 * The copies of a point on a torus: the grid of every combination of shifts by whole periods in each coordinate, `(2k + 1)^d`
 * points (nine in two dimensions, twenty-seven in three, for the default `k = 1`).
 *
 * @param point - The coordinates.
 * @param periods - The period of each coordinate.
 * @param k - How many periods each way (default 1).
 * @returns The grid, the last coordinate varying fastest.
 * @example
 * ```ts
 * import { periodicGrid } from 'omai-engine/stats'
 * periodicGrid([1, 2], [24, 7]).length // => 9
 * ```
 */
export function periodicGrid(point: readonly number[], periods: readonly number[], k = 1): number[][] {
  if (point.length !== periods.length)
    throw new OmaiError('shape', 'periodicGrid: the point and the periods need the same length')
  let grid: number[][] = [[]]
  point.forEach((x, i) => {
    const copies = periodicCopies(x, periods[i] as number, k)
    grid = grid.flatMap((prefix) => copies.map((c) => [...prefix, c]))
  })
  return grid
}

/**
 * The log density of the wrapped normal at `x`: the log of the sum over the copies `k = -K..K` of the normal density with mean
 * `mu` and standard deviation `sigma`, scored at `x - k period` (the copies in one log-sum-exp). It is the density on the circle
 * of period `period`; `copies = 1` is the three-copy approximation, large `K` the wrapped normal itself.
 *
 * @index function Wrapped normal log density as a log-sum-exp of ordinary normals at the copies.
 * @param x - The value.
 * @param mu - The mean.
 * @param sigma - The standard deviation.
 * @param period - The period.
 * @param copies - How many periods each way are summed (default 10).
 * @returns The log density.
 * @example
 * ```ts
 * import { wrappedNormalLogPdf } from 'omai-engine/stats'
 * // Midnight is as close to 23:50 as to 00:10.
 * Math.abs(wrappedNormalLogPdf(23.5, 0, 1, 24) - wrappedNormalLogPdf(0.5, 0, 1, 24)) < 1e-12 // => true
 * ```
 */
export function wrappedNormalLogPdf(x: number, mu: number, sigma: number, period: number, copies = 10): number {
  if (!(sigma > 0)) throw new OmaiError('value', `wrappedNormalLogPdf: sigma must be positive, got ${sigma}`)
  const terms = periodicCopies(x, period, copies).map(
    (c) => -0.5 * ((c - mu) / sigma) ** 2 - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI),
  )
  return logSumExpOf(terms)
}

/**
 * The most the density of the three-copy approximation can be off, relative to the wrapped normal's, for a mean inside one
 * period: `2 e^(-3a/4) / (1 - e^(-3a))` with `a = period^2 / (2 sigma^2)`. About 8e-11 at sigma = 3 on a 24 hour day, under
 * 0.005 nats at sigma = 6, about 0.4 at sigma = 12: three copies are accurate while sigma is small against the period.
 *
 * @index function Relative error bound of the three-copy wrapped density.
 * @param sigma - The standard deviation.
 * @param period - The period.
 * @returns The bound on the relative error of the density.
 * @example
 * ```ts
 * import { threeCopiesBound } from 'omai-engine/stats'
 * threeCopiesBound(3, 24) < 1e-10 // => true
 * ```
 */
export function threeCopiesBound(sigma: number, period: number): number {
  if (!(sigma > 0) || !(period > 0)) throw new OmaiError('value', 'threeCopiesBound: sigma and period must be positive')
  const a = period ** 2 / (2 * sigma ** 2)
  return (2 * Math.exp(-0.75 * a)) / (1 - Math.exp(-3 * a))
}
