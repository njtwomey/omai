/**
 * The distributions of extremes: the Gumbel law of maxima, the generalised extreme value (GEV) distribution, which
 * unites the Gumbel, Frechet and Weibull limit laws of block maxima, the generalised Pareto distribution of exceedances
 * over a high threshold, and the Frechet, as scipy.stats.
 */

import { gamma as gammaFn, log1mexp } from '../../numerics/special/index.ts'
import {
  abs,
  add,
  div,
  exp,
  expm1,
  greater,
  greaterEqual,
  less,
  lessEqual,
  log,
  log1p,
  logicalAnd,
  logicalOr,
  mul,
  neg,
  pow,
  primalOf,
  promoteFloat,
  square,
  sub,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { uniformOpen } from '../_sampling.ts'
import {
  batchShapeOf,
  broadcastBatch,
  type Distribution,
  defineDistribution,
  type Support,
  validateParams,
} from '../distribution.ts'

const EULER = 0.5772156649015329
// Below this |shape| the shape-zero (Gumbel or exponential) limits are used.
const SMALL = 1e-12

/** The support of a shape-dependent family: above the end when every shape is positive, below when every one is negative. */
function shapeSupport(shape: Value, end: Value, zeroLow: Value | null): Support {
  let values: number[]
  try {
    values = toFlat(primalOf(shape))
  } catch {
    return { kind: 'real' }
  }
  if (values.every((v) => v > 0))
    return zeroLow === null ? { kind: 'greaterThan', low: end } : { kind: 'greaterThan', low: zeroLow }
  if (values.every((v) => v < 0))
    return zeroLow === null ? { kind: 'lessThan', high: end } : { kind: 'interval', low: zeroLow, high: end }
  return zeroLow === null ? { kind: 'real' } : { kind: 'greaterThan', low: zeroLow }
}

/**
 * The Gumbel distribution of maxima (scipy.stats.gumbel_r, torch.distributions.Gumbel): F(x) =
 * exp(-exp(-(x - loc) / scale)).
 *
 * @param loc - Location.
 * @param scale - Scale (positive).
 * @returns The distribution; `rsample` is loc - scale log(-log U).
 * @concept distributions
 * @example
 * ```ts
 * import { gumbel } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(gumbel(0, 1).cdf(0)) === Math.exp(-1) // => true
 * ```
 */
export function gumbel(loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [mu, beta] = promoteFloat(loc, scale) as [Value, Value]
  validateParams('gumbel', { loc: [mu, 'real'], scale: [beta, 'positive'] })
  const batchShape = batchShapeOf(mu, beta)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const z = (x: Value): Value => div(sub(x, mu), beta)
  const ppf = (q: Value): Value => sub(mu, mul(beta, log(neg(log(q)))))
  return defineDistribution({
    name: 'gumbel',
    params: { loc: mu, scale: beta },
    support: { kind: 'real' },
    batchShape,
    logProb: (x) => {
      const v = z(x)
      return sub(neg(add(v, exp(neg(v)))), log(beta))
    },
    cdf: (x) => exp(neg(exp(neg(z(x))))),
    logCdf: (x) => neg(exp(neg(z(x)))),
    sf: (x) => neg(expm1(neg(exp(neg(z(x)))))),
    logSf: (x) => log1mexp(neg(exp(neg(z(x))))),
    ppf,
    isf: (q) => sub(mu, mul(beta, log(neg(log1p(neg(q)))))),
    rsample: (key, shape) => ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => b(add(mu, mul(beta, EULER))),
    variance: () => b(mul(square(beta), (Math.PI * Math.PI) / 6)),
    mode: () => b(mu),
    entropy: () => b(add(log(beta), EULER + 1)),
  })
}

/**
 * The generalised extreme value distribution GEV(loc, scale, shape xi), the limit law of normalised maxima: F(x) =
 * exp(-(1 + xi z)^(-1/xi)) with z = (x - loc) / scale (the Gumbel at xi = 0, Frechet-type for xi > 0 and reversed
 * Weibull-type for xi < 0). This is the extreme-value convention; scipy.stats.genextreme uses c = -xi.
 *
 * @param loc - Location.
 * @param scale - Scale (positive).
 * @param shape - Shape xi (real).
 * @returns The distribution with closed-form density, CDF, quantiles, moments (infinite where they diverge), mode and
 *   entropy; `rsample` is the quantile of uniform noise, differentiable in all three parameters.
 * @example
 * ```ts
 * import { generalisedExtremeValue } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(generalisedExtremeValue(0, 1, 0).cdf(0)) === Math.exp(-1) // => true
 * ```
 */
export function generalisedExtremeValue(
  loc: TensorLike = 0,
  scale: TensorLike = 1,
  shape: TensorLike = 0,
): Distribution {
  const [mu, sigma, xi] = promoteFloat(loc, scale, shape) as [Value, Value, Value]
  validateParams('generalisedExtremeValue', { loc: [mu, 'real'], scale: [sigma, 'positive'], shape: [xi, 'finite'] })
  const batchShape = batchShapeOf(mu, sigma, xi)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const zero = less(abs(xi), SMALL)
  const safeXi = where(zero, 1, xi)
  const z = (x: Value): Value => div(sub(x, mu), sigma)
  const inside = (x: Value): Value => logicalOr(zero, greater(add(mul(xi, z(x)), 1), 0))
  // log t(x), where t = (1 + xi z)^(-1/xi) (exp(-z) at xi = 0); F = exp(-t).
  const logT = (x: Value): Value => {
    const v = z(x)
    const arg = where(inside(x), add(mul(xi, v), 1), 1)
    return where(zero, neg(v), div(neg(log1p(sub(arg, 1))), safeXi))
  }
  // Outside the support F = 0 (below it, xi > 0) or F = 1 (above it, xi < 0).
  const outsideValue = (low: number, high: number): Value => where(greater(xi, 0), low, high)
  const fromQ = (logMinusLogQ: Value): Value =>
    add(mu, mul(sigma, where(zero, neg(logMinusLogQ), div(expm1(mul(neg(xi), logMinusLogQ)), safeXi))))
  const g = (k: number): Value => gammaFn(sub(1, mul(xi, k)))
  const self: Distribution = defineDistribution({
    name: 'generalisedExtremeValue',
    params: { loc: mu, scale: sigma, shape: xi },
    support: shapeSupport(xi, sub(mu, div(sigma, safeXi)), null),
    batchShape,
    logProb: (x) => {
      const lt = logT(x)
      return where(inside(x), sub(sub(mul(add(xi, 1), lt), exp(lt)), log(sigma)), Number.NEGATIVE_INFINITY)
    },
    cdf: (x) => where(inside(x), exp(neg(exp(logT(x)))), outsideValue(0, 1)),
    logCdf: (x) => where(inside(x), neg(exp(logT(x))), outsideValue(Number.NEGATIVE_INFINITY, 0)),
    sf: (x) => where(inside(x), neg(expm1(neg(exp(logT(x))))), outsideValue(1, 0)),
    ppf: (q) => fromQ(log(neg(log(q)))),
    isf: (q) => fromQ(log(neg(log1p(neg(q))))),
    rsample: (key, sampleShape) => fromQ(log(neg(log(uniformOpen(key, [...sampleShape, ...batchShape]))))),
    mean: () =>
      b(
        where(
          zero,
          add(mu, mul(sigma, EULER)),
          where(less(xi, 1), add(mu, div(mul(sigma, sub(g(1), 1)), safeXi)), Number.POSITIVE_INFINITY),
        ),
      ),
    variance: () =>
      b(
        where(
          zero,
          mul(square(sigma), (Math.PI * Math.PI) / 6),
          where(
            less(xi, 0.5),
            div(mul(square(sigma), sub(g(2), square(g(1)))), square(safeXi)),
            Number.POSITIVE_INFINITY,
          ),
        ),
      ),
    mode: () => b(add(mu, mul(sigma, where(zero, 0, div(expm1(mul(neg(xi), log1p(xi))), safeXi))))),
    entropy: () => b(add(add(log(sigma), mul(xi, EULER)), EULER + 1)),
  })
  return self
}

/**
 * The generalised Pareto distribution GPD(loc, scale, shape xi) of threshold exceedances (scipy.stats.genpareto(xi,
 * loc, scale)): survival (1 + xi z)^(-1/xi) for z = (x - loc) / scale >= 0, bounded above by loc - scale / xi when
 * xi < 0; xi = 0 is the exponential.
 *
 * @param loc - Location (the threshold).
 * @param scale - Scale (positive).
 * @param shape - Shape xi (real).
 * @returns The distribution with closed-form density, CDF, quantiles, moments, mode and entropy; `rsample` is the
 *   quantile of uniform noise.
 * @example
 * ```ts
 * import { generalisedPareto } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(generalisedPareto(0, 1, 0).mean()) // => 1
 * item(generalisedPareto(0, 2, 0.5).mean()) // => 4
 * ```
 */
export function generalisedPareto(loc: TensorLike = 0, scale: TensorLike = 1, shape: TensorLike = 0): Distribution {
  const [mu, sigma, xi] = promoteFloat(loc, scale, shape) as [Value, Value, Value]
  validateParams('generalisedPareto', { loc: [mu, 'real'], scale: [sigma, 'positive'], shape: [xi, 'finite'] })
  const batchShape = batchShapeOf(mu, sigma, xi)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const zero = less(abs(xi), SMALL)
  const safeXi = where(zero, 1, xi)
  const z = (x: Value): Value => div(sub(x, mu), sigma)
  const upper = sub(mu, div(sigma, safeXi))
  const inside = (x: Value): Value =>
    logicalAnd(greaterEqual(z(x), 0), logicalOr(greaterEqual(xi, 0), lessEqual(x, upper)))
  const safeZ = (x: Value): Value => where(inside(x), z(x), 0)
  // log S(x) = -log1p(xi z) / xi (-z at xi = 0).
  const logS = (x: Value): Value => {
    const v = safeZ(x)
    return where(zero, neg(v), div(neg(log1p(mul(xi, v))), safeXi))
  }
  const fromLogS = (ls: Value): Value => add(mu, mul(sigma, where(zero, neg(ls), div(expm1(mul(neg(xi), ls)), safeXi))))
  const belowLoc = (x: Value): Value => less(z(x), 0)
  return defineDistribution({
    name: 'generalisedPareto',
    params: { loc: mu, scale: sigma, shape: xi },
    support: shapeSupport(xi, upper, mu),
    batchShape,
    logProb: (x) => {
      const v = safeZ(x)
      const lp = where(zero, neg(v), mul(neg(add(div(1, safeXi), 1)), log1p(mul(xi, v))))
      return where(inside(x), sub(lp, log(sigma)), Number.NEGATIVE_INFINITY)
    },
    cdf: (x) => where(inside(x), neg(expm1(logS(x))), where(belowLoc(x), 0, 1)),
    sf: (x) => where(inside(x), exp(logS(x)), where(belowLoc(x), 1, 0)),
    logSf: (x) => where(inside(x), logS(x), where(belowLoc(x), 0, Number.NEGATIVE_INFINITY)),
    ppf: (q) => fromLogS(log1p(neg(q))),
    isf: (q) => fromLogS(log(q)),
    rsample: (key, sampleShape) => fromLogS(log(uniformOpen(key, [...sampleShape, ...batchShape]))),
    mean: () => b(where(less(xi, 1), add(mu, div(sigma, sub(1, xi))), Number.POSITIVE_INFINITY)),
    variance: () =>
      b(
        where(less(xi, 0.5), div(square(sigma), mul(square(sub(1, xi)), sub(1, mul(xi, 2)))), Number.POSITIVE_INFINITY),
      ),
    mode: () => b(where(less(xi, -1), upper, mu)),
    entropy: () => b(add(add(log(sigma), xi), 1)),
  })
}

/**
 * The Frechet distribution (scipy.stats.invweibull(alpha, loc, scale)): F(x) = exp(-z^(-alpha)) for
 * z = (x - loc) / scale > 0, the GEV with xi = 1 / alpha up to location and scale.
 *
 * @param alpha - Shape (positive).
 * @param loc - Location (the lower end of the support).
 * @param scale - Scale (positive).
 * @returns The distribution; `rsample` is the quantile of uniform noise. The mean is infinite for alpha <= 1 and the
 *   variance for alpha <= 2.
 * @example
 * ```ts
 * import { frechet } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(frechet(1).cdf(1)) === Math.exp(-1) // => true
 * ```
 */
export function frechet(alpha: TensorLike, loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [a, mu, s] = promoteFloat(alpha, loc, scale) as [Value, Value, Value]
  validateParams('frechet', { alpha: [a, 'positive'], loc: [mu, 'real'], scale: [s, 'positive'] })
  const batchShape = batchShapeOf(a, mu, s)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const inside = (x: Value): Value => greater(x, mu)
  const z = (x: Value): Value => div(sub(where(inside(x), x, add(mu, s)), mu), s)
  // z^(-alpha), the cumulative hazard -log F.
  const h = (x: Value): Value => pow(z(x), neg(a))
  const fromLogMinusLogQ = (l: Value): Value => add(mu, mul(s, exp(div(neg(l), a))))
  const g = (k: number): Value => gammaFn(sub(1, div(k, a)))
  return defineDistribution({
    name: 'frechet',
    params: { alpha: a, loc: mu, scale: s },
    support: { kind: 'greaterThan', low: mu },
    batchShape,
    logProb: (x) =>
      where(inside(x), sub(sub(log(div(a, s)), mul(add(a, 1), log(z(x)))), h(x)), Number.NEGATIVE_INFINITY),
    cdf: (x) => where(inside(x), exp(neg(h(x))), 0),
    logCdf: (x) => where(inside(x), neg(h(x)), Number.NEGATIVE_INFINITY),
    sf: (x) => where(inside(x), neg(expm1(neg(h(x)))), 1),
    ppf: (q) => fromLogMinusLogQ(log(neg(log(q)))),
    isf: (q) => fromLogMinusLogQ(log(neg(log1p(neg(q))))),
    rsample: (key, shape) => fromLogMinusLogQ(log(neg(log(uniformOpen(key, [...shape, ...batchShape]))))),
    mean: () => b(where(greater(a, 1), add(mu, mul(s, g(1))), Number.POSITIVE_INFINITY)),
    variance: () => b(where(greater(a, 2), mul(square(s), sub(g(2), square(g(1)))), Number.POSITIVE_INFINITY)),
    mode: () => b(add(mu, mul(s, pow(div(a, add(a, 1)), div(1, a))))),
    entropy: () => b(sub(add(add(log(s), 1 + EULER), div(EULER, a)), log(a))),
  })
}
