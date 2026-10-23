import { betainc, betaincinv, digamma, lbeta } from '../../numerics/special/index.ts'
import { split, uniform as uniformNoise } from '../../random/index.ts'
import {
  add,
  clip,
  div,
  greater,
  less,
  log,
  log1p,
  logicalAnd,
  logicalOr,
  mul,
  neg,
  promoteFloat,
  square,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { gammaSample } from '../_sampling.ts'
import { batchShapeOf, type Distribution, defineDistribution } from '../distribution.ts'

/**
 * The continuous uniform distribution on [low, high] (scipy.stats.uniform with loc = low, scale = high - low).
 *
 * @param low - Lower bound.
 * @param high - Upper bound.
 * @returns The distribution; `rsample` is low + (high - low) U.
 * @example
 * ```ts
 * import { uniform } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(uniform(0, 4).cdf(1)) // => 0.25
 * ```
 */
export function uniform(low: TensorLike = 0, high: TensorLike = 1): Distribution {
  const [lo, hi] = promoteFloat(low, high) as [Value, Value]
  const batchShape = batchShapeOf(lo, hi)
  const width = sub(hi, lo)
  const outside = (x: Value): Value => logicalOr(less(x, lo), greater(x, hi))
  return defineDistribution({
    name: 'uniform',
    params: { low: lo, high: hi },
    support: { kind: 'interval', low: lo, high: hi },
    batchShape,
    logProb: (x) => where(outside(x), Number.NEGATIVE_INFINITY, neg(log(width))),
    cdf: (x) => clip(div(sub(x, lo), width), 0, 1),
    ppf: (q) => add(lo, mul(q, width)),
    rsample: (key, shape) => add(lo, mul(width, uniformNoise(key, [...shape, ...batchShape]))),
    mean: () => mul(add(lo, hi), 0.5),
    variance: () => div(square(width), 12),
    mode: () => mul(add(lo, hi), 0.5),
    entropy: () => log(width),
  })
}

/**
 * The beta distribution with shapes `a` and `b` (scipy.stats.beta, torch.distributions.Beta).
 *
 * @param a - First shape (positive).
 * @param b - Second shape (positive).
 * @returns The distribution; `rsample` is G1 / (G1 + G2) for implicitly reparameterised gamma variates, so samples
 *   differentiate in both shapes.
 * @concept distributions
 * @example
 * ```ts
 * import { beta } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(beta(2, 2).cdf(0.5)) // => 0.5
 * ```
 */
export function beta(a: TensorLike, b: TensorLike): Distribution {
  const [p, q] = promoteFloat(a, b) as [Value, Value]
  const batchShape = batchShapeOf(p, q)
  const inside = (x: Value): Value => logicalAnd(greater(x, 0), less(x, 1))
  const safe = (x: Value): Value => where(inside(x), x, 0.5)
  const total = add(p, q)
  return defineDistribution({
    name: 'beta',
    params: { a: p, b: q },
    support: { kind: 'unitInterval' },
    batchShape,
    logProb: (x) =>
      where(
        inside(x),
        sub(add(mul(sub(p, 1), log(safe(x))), mul(sub(q, 1), log1p(neg(safe(x))))), lbeta(p, q)),
        Number.NEGATIVE_INFINITY,
      ),
    cdf: (x) => betainc(p, q, clip(x, 0, 1)),
    ppf: (u) => betaincinv(p, q, u),
    rsample: (key, shape) => {
      const [k1, k2] = split(key)
      const g1 = gammaSample(k1 as never, add(p, mul(q, 0)), shape)
      const g2 = gammaSample(k2 as never, add(q, mul(p, 0)), shape)
      return div(g1, add(g1, g2))
    },
    mean: () => div(p, total),
    variance: () => div(mul(p, q), mul(square(total), add(total, 1))),
    mode: () => div(sub(p, 1), sub(total, 2)),
    entropy: () =>
      add(
        sub(sub(lbeta(p, q), mul(sub(p, 1), digamma(p))), mul(sub(q, 1), digamma(q))),
        mul(sub(total, 2), digamma(total)),
      ),
  })
}
