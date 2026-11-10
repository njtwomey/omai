/**
 * The simultaneous band of an ECDF-difference plot (Säilynoja, Bürkner and Vehtari 2022): for `n` independent uniform
 * values, the band around the ECDF at a grid of points inside which the whole ECDF stays with a stated probability,
 * found by an exact recursion over the counts at the grid points and a search over the level of the pointwise binomial
 * bands. The band answers "does this curve look like the ECDF of uniform values?" without the multiple-testing error of
 * reading a pointwise band at many points; the rank plots of simulation-based calibration and of MCMC chains draw it.
 */

import { OmaiError } from '../base/index.ts'

/** A band around the ECDF of uniform values at a grid of points: the ECDF stays between `lower` and `upper` everywhere with probability `level`. */
export interface EcdfBand {
  /** The grid, strictly increasing in (0, 1). */
  readonly z: readonly number[]
  /** The least and greatest value of the ECDF (a proportion, count over `n`) at each grid point. */
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  /** The number of values the ECDF is made of. */
  readonly n: number
  /** The simultaneous probability the band holds. */
  readonly level: number
  /** The level of each pointwise binomial band the search settled on (smaller than `1 - level`). */
  readonly gamma: number
}

/** log k! for k = 0..n. */
function logFactorials(n: number): Float64Array {
  const lf = new Float64Array(n + 1)
  for (let k = 1; k <= n; k++) lf[k] = (lf[k - 1] ?? 0) + Math.log(k)
  return lf
}

/** The probability mass of Binomial(m, p) at j. */
function binomialMass(lf: Float64Array, m: number, j: number, p: number): number {
  if (j < 0 || j > m) return 0
  if (p <= 0) return j === 0 ? 1 : 0
  if (p >= 1) return j === m ? 1 : 0
  return Math.exp((lf[m] ?? 0) - (lf[j] ?? 0) - (lf[m - j] ?? 0) + j * Math.log(p) + (m - j) * Math.log1p(-p))
}

function checkGrid(z: readonly number[], where: string): void {
  let previous = 0
  for (const v of z) {
    if (!(v > previous && v < 1))
      throw new OmaiError('value', `${where}: the grid must increase strictly inside (0, 1)`)
    previous = v
  }
}

/**
 * The probability that the ECDF of `n` uniform values has, at every grid point, a count within the given limits: the
 * exact probability behind a simultaneous band. The count at the next point is, given the count so far, the sum of a
 * binomial draw from the values left, so a recursion over the counts within the limits gives it exactly.
 *
 * @param n - The number of uniform values.
 * @param z - The grid, strictly increasing in (0, 1).
 * @param lower - The least count allowed at each point.
 * @param upper - The greatest count allowed at each point.
 * @returns The probability.
 * @throws OmaiError `value` for a grid outside (0, 1) or not increasing; `shape` when the limits do not match the grid.
 * @example
 * ```ts
 * import { bandCoverage } from 'omai-engine/exposition'
 * // One value, one point at 1/2: the count is 0 or 1 with probability 1/2 each.
 * bandCoverage(1, [0.5], [0], [0]) // => 0.5
 * bandCoverage(1, [0.5], [0], [1]) // => 1
 * ```
 */
export function bandCoverage(
  n: number,
  z: readonly number[],
  lower: readonly number[],
  upper: readonly number[],
): number {
  checkGrid(z, 'bandCoverage')
  if (lower.length !== z.length || upper.length !== z.length)
    throw new OmaiError('shape', 'bandCoverage: the limits must have one entry per grid point')
  return coverage(n, z, lower, upper, logFactorials(n))
}

function coverage(
  n: number,
  z: readonly number[],
  lower: readonly number[],
  upper: readonly number[],
  lf: Float64Array,
): number {
  // `mass[k - from]`: the probability that the count so far is k and every earlier point was within its limits.
  let from = 0
  let mass: number[] = [1]
  let previous = 0
  for (const [i, zi] of z.entries()) {
    const lo = Math.max(0, lower[i] ?? 0)
    const hi = Math.min(n, upper[i] ?? n)
    if (lo > hi) return 0
    const p = (zi - previous) / (1 - previous)
    const next = new Array<number>(hi - lo + 1).fill(0)
    for (const [a, w] of mass.entries()) {
      if (w === 0) continue
      const k = from + a
      const left = n - k
      for (let c = Math.max(lo, k); c <= Math.min(hi, k + left); c++) {
        next[c - lo] = (next[c - lo] ?? 0) + w * binomialMass(lf, left, c - k, p)
      }
    }
    from = lo
    mass = next
    previous = zi
  }
  return mass.reduce((s, v) => s + v, 0)
}

/** The least count whose cumulative probability reaches `prob`, from the cumulative sums of the binomial masses. */
function quantileCount(cumulative: Float64Array, prob: number): number {
  let lo = 0
  let hi = cumulative.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if ((cumulative[mid] ?? 0) >= prob) hi = mid
    else lo = mid + 1
  }
  return lo
}

/**
 * The simultaneous band of the ECDF of `n` uniform values over a grid: pointwise binomial bands whose level is searched
 * so that the ECDF stays inside all of them together with probability `level` (Säilynoja et al. 2022, the exact
 * recursion over the counts). Plot the ECDF minus the grid against the grid with the band minus the grid; a curve that
 * leaves the band is, at that level, not the ECDF of uniform values.
 *
 * @param n - The number of values (at least one).
 * @param z - The grid, strictly increasing in (0, 1).
 * @param level - The simultaneous probability (default 0.95).
 * @returns The band, as proportions.
 * @throws OmaiError `value` for fewer than one value, a level outside (0, 1) or a grid outside (0, 1).
 * @index function Simultaneous band of the ECDF of uniform values: `ecdfBand(100, [0.25, 0.5, 0.75])`.
 * @example
 * ```ts
 * import { bandCoverage, ecdfBand } from 'omai-engine/exposition'
 * const b = ecdfBand(50, [0.25, 0.5, 0.75], 0.95)
 * const counts = (v: readonly number[]) => v.map((x) => Math.round(x * 50))
 * bandCoverage(50, b.z, counts(b.lower), counts(b.upper)) >= 0.95 // => true
 * b.lower[1] < 0.5 && b.upper[1] > 0.5 // => true
 * ```
 */
export function ecdfBand(n: number, z: readonly number[], level = 0.95): EcdfBand {
  if (!Number.isInteger(n) || n < 1) throw new OmaiError('value', 'ecdfBand: n must be a positive integer')
  if (!(level > 0 && level < 1)) throw new OmaiError('value', 'ecdfBand: the level must be inside (0, 1)')
  checkGrid(z, 'ecdfBand')
  const lf = logFactorials(n)
  // The cumulative distribution of the count at each point.
  const cumulative = z.map((zi) => {
    const c = new Float64Array(n + 1)
    let acc = 0
    for (let k = 0; k <= n; k++) {
      acc += binomialMass(lf, n, k, zi)
      c[k] = acc
    }
    return c
  })
  const limits = (gamma: number): { lo: number[]; hi: number[] } => ({
    lo: cumulative.map((c) => quantileCount(c, gamma / 2)),
    hi: cumulative.map((c) => quantileCount(c, 1 - gamma / 2)),
  })
  const cover = (gamma: number): number => {
    const { lo, hi } = limits(gamma)
    return coverage(n, z, lo, hi, lf)
  }
  // The largest gamma whose simultaneous coverage is still at least the level: bisection on its logarithm.
  let low = Math.log(1e-14)
  let high = Math.log(1 - level)
  if (cover(Math.exp(high)) >= level) low = high
  else
    for (let it = 0; it < 30; it++) {
      const mid = (low + high) / 2
      if (cover(Math.exp(mid)) >= level) low = mid
      else high = mid
    }
  const gamma = Math.exp(low)
  const { lo, hi } = limits(gamma)
  return {
    z: [...z],
    lower: lo.map((k) => k / n),
    upper: hi.map((k) => k / n),
    n,
    level,
    gamma,
  }
}

/**
 * The central interval of a binomial count: the counts between the quantiles at the two tails of `1 - level` shared
 * equally, so that a count of `n` trials with success probability `p` falls inside with at least probability `level`
 * (the pointwise band of a histogram bin whose expected share is `p`).
 *
 * @param n - The number of trials.
 * @param p - The success probability.
 * @param level - The probability the interval holds (default 0.99).
 * @returns The least and greatest count.
 * @example
 * ```ts
 * import { binomialInterval } from 'omai-engine/exposition'
 * binomialInterval(100, 0.5, 0.95) // => [40, 60]
 * ```
 */
export function binomialInterval(n: number, p: number, level = 0.99): [number, number] {
  if (!Number.isInteger(n) || n < 1) throw new OmaiError('value', 'binomialInterval: n must be a positive integer')
  if (!(p >= 0 && p <= 1)) throw new OmaiError('value', 'binomialInterval: p must be a probability')
  if (!(level > 0 && level < 1)) throw new OmaiError('value', 'binomialInterval: the level must be inside (0, 1)')
  const lf = logFactorials(n)
  const cumulative = new Float64Array(n + 1)
  let acc = 0
  for (let k = 0; k <= n; k++) {
    acc += binomialMass(lf, n, k, p)
    cumulative[k] = acc
  }
  const tail = (1 - level) / 2
  return [quantileCount(cumulative, tail), quantileCount(cumulative, 1 - tail)]
}

/**
 * An evenly spaced grid of `points` values strictly inside (0, 1): `i / (points + 1)`, the points at which the ECDF of
 * uniform values on `points + 1` cells is read.
 *
 * @param points - How many points (at least one).
 * @returns The grid.
 * @example
 * ```ts
 * import { uniformGrid } from 'omai-engine/exposition'
 * uniformGrid(3) // => [0.25, 0.5, 0.75]
 * ```
 */
export function uniformGrid(points: number): number[] {
  return Array.from({ length: Math.max(1, points) }, (_, i) => (i + 1) / (Math.max(1, points) + 1))
}
