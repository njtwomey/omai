/**
 * Limiting spectral laws of random matrices, part of `omai-engine/dists`: the Marchenko-Pastur law of the eigenvalues
 * of a sample covariance matrix when the dimension grows with the sample size, with its closed-form density, CDF,
 * quantiles, moments, support edges and sampling. The first picture of high-dimensional statistics is the eigenvalue
 * histogram of pure noise under this curve.
 */

import { OmaiError } from '../../base/index.ts'
import {
  abs,
  add,
  arccos,
  arctan,
  clip,
  div,
  greater,
  greaterEqual,
  less,
  lessEqual,
  log,
  logicalAnd,
  mul,
  promoteFloat,
  sin,
  sqrt,
  square,
  sub,
  type TensorLike,
  tan,
  type Value,
  where,
} from '../../tensor/index.ts'
import { concreteTwin, numericEntropy, numericPpf } from '../_numeric.ts'
import { uniformOpen } from '../_sampling.ts'
import { batchShapeOf, broadcastBatch, type Distribution, defineDistribution, validateParams } from '../distribution.ts'

/**
 * The edges of the Marchenko-Pastur support, `variance (1 -+ sqrt(ratio))^2`: the smallest and largest eigenvalue a
 * pure-noise sample covariance matrix of aspect ratio `ratio = p / n` approaches as `p` and `n` grow.
 *
 * @param ratio - The aspect ratio `p / n` of dimension to sample size (positive).
 * @param variance - The variance of each noise entry (default 1).
 * @returns The lower and the upper edge.
 * @index function Edges of the Marchenko-Pastur support from the aspect ratio.
 * @example
 * ```ts
 * import { marchenkoPasturEdges } from 'omai-engine/dists'
 * marchenkoPasturEdges(0.25) // => { lower: 0.25, upper: 2.25 }
 * ```
 */
export function marchenkoPasturEdges(ratio: number, variance = 1): { readonly lower: number; readonly upper: number } {
  if (!(ratio > 0) || !Number.isFinite(ratio))
    throw new OmaiError('value', `marchenkoPasturEdges: ratio must be positive and finite (got ${ratio})`)
  if (!(variance > 0) || !Number.isFinite(variance))
    throw new OmaiError('value', `marchenkoPasturEdges: variance must be positive and finite (got ${variance})`)
  const s = Math.sqrt(ratio)
  return { lower: variance * (1 - s) ** 2, upper: variance * (1 + s) ** 2 }
}

/**
 * The Marchenko-Pastur law of the eigenvalues of a sample covariance matrix `(1 / n) X X^T` of `p x n` noise with
 * entries of the given variance, as `p, n -> infinity` with `p / n = ratio`. For `ratio <= 1` the law has the density
 * `sqrt((b - x)(x - a)) / (2 pi variance ratio x)` on `[a, b] = variance (1 -+ sqrt(ratio))^2`; for `ratio > 1` the
 * matrix is singular and the law also has an atom of mass `1 - 1 / ratio` at zero. The mean is `variance` and the
 * variance `variance^2 ratio` in both cases.
 *
 * @param ratio - The aspect ratio `p / n` (positive).
 * @param variance - The variance of the noise entries (default 1).
 * @returns The distribution. `logProb` and `prob` are those of the absolutely continuous part (the atom is not a
 *   density; it is `1 - 1 / ratio` at zero for a ratio above 1, and the `cdf` includes it); closed-form `cdf` (an
 *   arccosine and an arctangent), quantiles by inversion of the `cdf` with the atom at zero, `rsample` by the
 *   quantile of uniform noise, mean, variance, mode `variance (1 - ratio)^2 / (1 + ratio)`, and the entropy by
 *   quadrature (undefined, `NaN`, with an atom).
 * @index distribution Marchenko-Pastur law of sample covariance eigenvalues, by aspect ratio.
 * @example
 * ```ts
 * import { marchenkoPastur } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * item(marchenkoPastur(0.25).cdf(2.25)) // => 1
 * item(marchenkoPastur(0.25).mean()) // => 1
 * ```
 */
export function marchenkoPastur(ratio: TensorLike, variance: TensorLike = 1): Distribution {
  const [y, v] = promoteFloat(ratio, variance) as [Value, Value]
  validateParams('marchenkoPastur', { ratio: [y, 'positive'], variance: [v, 'positive'] })
  const batchShape = batchShapeOf(y, v)
  const bb = (z: Value): Value => broadcastBatch(z, batchShape)
  const sy = sqrt(y)
  const lower = mul(v, square(sub(1, sy)))
  const upper = mul(v, square(add(1, sy)))
  const atom = where(greater(y, 1), sub(1, div(1, y)), 0)
  const hasAtom = greater(y, 1)
  const insideBulk = (x: Value): Value => logicalAnd(greater(x, lower), less(x, upper))
  const safe = (x: Value): Value => where(insideBulk(x), x, mul(add(lower, upper), 0.5))
  // The integral of the density up to x: substitute x / variance = 1 + y - 2 sqrt(y) cos(phi), then integrate sin^2(phi) / (1 + y - 2 sqrt(y) cos(phi)).
  const bulkCdf = (x: Value): Value => {
    const u = div(x, v)
    const phi = arccos(clip(div(sub(add(1, y), u), mul(sy, 2)), -1, 1))
    const gap = abs(sub(1, y))
    const flat = less(gap, 1e-12)
    const angle = arctan(mul(div(add(1, sy), where(flat, 1, abs(sub(1, sy)))), tan(mul(phi, 0.5))))
    const last = where(flat, 0, mul(div(gap, mul(y, 2)), angle))
    const body = sub(add(div(sin(phi), mul(sy, 2)), div(mul(add(1, y), phi), mul(y, 4))), last)
    return mul(body, 2 / Math.PI)
  }
  const cdf = (x: Value): Value =>
    where(
      lessEqual(x, lower),
      where(logicalAnd(hasAtom, greaterEqual(x, 0)), atom, 0),
      where(greaterEqual(x, upper), 1, add(atom, bulkCdf(x))),
    )
  const twin = concreteTwin([y, v], (r, s) => marchenkoPastur(r, s))
  const self: Distribution = defineDistribution({
    name: 'marchenkoPastur',
    params: { ratio: y, variance: v },
    support: { kind: 'interval', low: where(hasAtom, 0, lower), high: upper },
    batchShape,
    logProb: (x) => {
      const s = safe(x)
      const dens = sub(mul(log(mul(sub(upper, s), sub(s, lower))), 0.5), log(mul(mul(mul(v, y), s), 2 * Math.PI)))
      return where(insideBulk(x), dens, Number.NEGATIVE_INFINITY)
    },
    cdf,
    ppf: (q) => {
      const inverted = numericPpf(self, twin, q)
      return where(logicalAnd(hasAtom, lessEqual(q, atom)), 0, inverted)
    },
    rsample: (key, shape) => self.ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => bb(v),
    variance: () => bb(mul(square(v), y)),
    mode: () => bb(div(mul(v, square(sub(1, y))), add(1, y))),
    entropy: () =>
      where(bb(hasAtom), Number.NaN, numericEntropy(self, { kind: 'finite', low: bb(lower), high: bb(upper) })),
  })
  return self
}
