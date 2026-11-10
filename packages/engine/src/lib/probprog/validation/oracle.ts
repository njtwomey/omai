/**
 * What a posterior is compared with: a `MarginalOracle` is the one-dimensional marginal of a quantity (its mean,
 * standard deviation, CDF and quantile function), whether it comes from a closed form, a grid, or a long reference
 * run. The validation kernel (`agree`, simulation-based calibration) speaks this one interface.
 */

import { OmaiError } from '../../base/index.ts'
import type { Distribution } from '../../dists/index.ts'
import { item } from '../../tensor/index.ts'

/** The marginal distribution of one scalar quantity. */
export interface MarginalOracle {
  /** The quantity's name (`mu`, `theta[2]`). */
  readonly name: string
  readonly mean: number
  readonly sd: number
  /** P(X <= x). */
  cdf(x: number): number
  /** The p-quantile, for p in (0, 1). */
  quantile(p: number): number
}

/**
 * The oracle of a scalar distribution (a closed-form posterior).
 *
 * @param name - The quantity's name.
 * @param d - A scalar distribution with a CDF and a quantile function.
 * @returns The oracle.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { distributionOracle } from 'omai-engine/probprog/validation'
 * import { normal } from 'omai-engine/dists'
 * const o = distributionOracle('mu', normal(1, 2))
 * [o.mean, o.sd, o.cdf(1), o.quantile(0.5)] // => [1, 2, 0.5, 1]
 * ```
 */
export function distributionOracle(name: string, d: Distribution): MarginalOracle {
  if (d.batchShape.length > 0 || d.eventShape.length > 0)
    throw new OmaiError('shape', `distributionOracle: ${name} must be a scalar distribution`)
  return {
    name,
    mean: item(d.mean()),
    sd: item(d.stddev()),
    cdf: (x) => item(d.cdf(x)),
    quantile: (p) => item(d.ppf(p)),
  }
}

/**
 * The oracle of weighted draws (a long reference run): the empirical distribution, with a piecewise linear CDF through
 * the midpoints of the sorted draws so that its quantile function inverts it.
 *
 * @param name - The quantity's name.
 * @param draws - The draws.
 * @param weights - Their weights (default equal).
 * @returns The oracle.
 * @throws OmaiError with code `value` for fewer than two draws or negative weights.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { weightedOracle } from 'omai-engine/probprog/validation'
 * const o = weightedOracle('x', [1, 2, 3, 4])
 * [o.mean, o.quantile(0.5)] // => [2.5, 2.5]
 * ```
 */
export function weightedOracle(name: string, draws: ArrayLike<number>, weights?: ArrayLike<number>): MarginalOracle {
  const n = draws.length
  if (n < 2) throw new OmaiError('value', `weightedOracle: ${name} needs at least two draws`)
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (draws[a] ?? 0) - (draws[b] ?? 0))
  const x = order.map((i) => draws[i] ?? 0)
  const w = order.map((i) => weights?.[i] ?? 1)
  if (w.some((v) => !(v >= 0))) throw new OmaiError('value', 'weightedOracle: weights must be non-negative')
  const total = w.reduce((a, b) => a + b, 0)
  const mass = w.map((v) => v / total)
  const mean = mass.reduce((s, m, i) => s + m * (x[i] ?? 0), 0)
  const variance = mass.reduce((s, m, i) => s + m * ((x[i] ?? 0) - mean) ** 2, 0)
  // F at the sorted draws, taken at the middle of each draw's mass.
  const F: number[] = []
  let acc = 0
  for (const m of mass) {
    F.push(acc + m / 2)
    acc += m
  }
  const at = (grid: readonly number[], target: number): number => {
    let lo = 0
    let hi = grid.length - 1
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1
      if ((grid[mid] ?? 0) <= target) lo = mid
      else hi = mid
    }
    return lo
  }
  return {
    name,
    mean,
    sd: Math.sqrt(variance),
    cdf: (v) => {
      if (v <= (x[0] ?? 0)) return v < (x[0] ?? 0) ? 0 : (F[0] ?? 0)
      if (v >= (x[n - 1] ?? 0)) return 1
      const j = at(x, v)
      const span = (x[j + 1] ?? 0) - (x[j] ?? 0)
      const t = span === 0 ? 0 : (v - (x[j] ?? 0)) / span
      return (F[j] ?? 0) + t * ((F[j + 1] ?? 0) - (F[j] ?? 0))
    },
    quantile: (p) => {
      if (p <= (F[0] ?? 0)) return x[0] ?? 0
      if (p >= (F[n - 1] ?? 1)) return x[n - 1] ?? 0
      const j = at(F, p)
      const span = (F[j + 1] ?? 0) - (F[j] ?? 0)
      const t = span === 0 ? 0 : (p - (F[j] ?? 0)) / span
      return (x[j] ?? 0) + t * ((x[j + 1] ?? 0) - (x[j] ?? 0))
    },
  }
}
