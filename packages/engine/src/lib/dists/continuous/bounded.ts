/**
 * Continuous distributions on a bounded interval: uniform, beta, Kumaraswamy (a beta-like family with a closed-form
 * CDF), triangular and logit-normal, each with densities, CDFs, quantiles, moments and reparameterised sampling as
 * scipy.stats and torch.distributions.
 */

import {
  betainc,
  betaincinv,
  digamma,
  lbeta,
  logit,
  logNdtr,
  logSigmoid,
  ndtr,
  ndtri,
  normalLogPdf,
} from '../../numerics/special/index.ts'
import { foldIn, normal as normalNoise } from '../../random/index.ts'
import {
  add,
  clip,
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
  promoteFloat,
  sigmoid,
  sqrt,
  square,
  sub,
  type Tensor,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { concreteTwin, integrate, numericMode } from '../_numeric.ts'
import { gammaSample, uniformOpen } from '../_sampling.ts'
import { batchShapeOf, broadcastBatch, type Distribution, defineDistribution, validateParams } from '../distribution.ts'

const EULER = 0.5772156649015329

/**
 * The continuous uniform distribution on [low, high] (scipy.stats.uniform with loc = low, scale = high - low).
 *
 * @param low - Lower bound.
 * @param high - Upper bound.
 * @returns The distribution; `rsample` is low + (high - low) U.
 * @example
 * ```ts
 * import { uniform } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(uniform(0, 4).cdf(1)) // => 0.25
 * ```
 */
export function uniform(low: TensorLike = 0, high: TensorLike = 1): Distribution {
  const [lo, hi] = promoteFloat(low, high) as [Value, Value]
  validateParams('uniform', { low: [lo, 'finite'], high: [hi, 'finite'] })
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
    sf: (x) => clip(div(sub(hi, x), width), 0, 1),
    ppf: (q) => add(lo, mul(q, width)),
    isf: (q) => sub(hi, mul(q, width)),
    rsample: (key, shape) => add(lo, mul(width, uniformOpen(key, [...shape, ...batchShape]))),
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
 * import { beta } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(beta(2, 2).cdf(0.5)) - 0.5) < 1e-14 // => true
 * ```
 */
export function beta(a: TensorLike, b: TensorLike): Distribution {
  const [p, q] = promoteFloat(a, b) as [Value, Value]
  validateParams('beta', { a: [p, 'positive'], b: [q, 'positive'] })
  const batchShape = batchShapeOf(p, q)
  const bb = (v: Value): Value => broadcastBatch(v, batchShape)
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
    sf: (x) => betainc(q, p, clip(sub(1, x), 0, 1)),
    ppf: (u) => betaincinv(p, q, u),
    isf: (u) => sub(1, betaincinv(q, p, u)),
    rsample: (key, shape) => {
      const g1 = gammaSample(foldIn(key, 0), bb(p), shape)
      const g2 = gammaSample(foldIn(key, 1), bb(q), shape)
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

/**
 * The Kumaraswamy distribution on (0, 1) (torch.distributions.Kumaraswamy with concentration1 = a, concentration0 =
 * b): density a b x^(a-1) (1 - x^a)^(b-1), a beta-like family with a closed-form CDF and quantile function.
 *
 * @param a - First shape (positive).
 * @param b - Second shape (positive).
 * @returns The distribution; `rsample` is the quantile of uniform noise, differentiable in both shapes.
 * @example
 * ```ts
 * import { kumaraswamy } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(kumaraswamy(1, 1).cdf(0.25)) - 0.25) < 1e-15 // => true
 * ```
 */
export function kumaraswamy(a: TensorLike, b: TensorLike): Distribution {
  const [p, q] = promoteFloat(a, b) as [Value, Value]
  validateParams('kumaraswamy', { a: [p, 'positive'], b: [q, 'positive'] })
  const batchShape = batchShapeOf(p, q)
  const bb = (v: Value): Value => broadcastBatch(v, batchShape)
  const inside = (x: Value): Value => logicalAnd(greater(x, 0), less(x, 1))
  const safe = (x: Value): Value => where(inside(x), x, 0.5)
  // log1p(-x^a), the log of the survival base.
  const logRest = (x: Value): Value => log1p(neg(pow(safe(x), p)))
  const ppf = (u: Value): Value => pow(neg(expm1(div(log1p(neg(u)), q))), div(1, p))
  // The n-th raw moment b B(1 + n / a, b).
  const moment = (n: number): Value => exp(add(log(q), lbeta(add(div(n, p), 1), q)))
  return defineDistribution({
    name: 'kumaraswamy',
    params: { a: p, b: q },
    support: { kind: 'unitInterval' },
    batchShape,
    logProb: (x) =>
      where(
        inside(x),
        add(add(log(mul(p, q)), mul(sub(p, 1), log(safe(x)))), mul(sub(q, 1), logRest(x))),
        Number.NEGATIVE_INFINITY,
      ),
    cdf: (x) => where(lessEqual(x, 0), 0, where(greaterEqual(x, 1), 1, neg(expm1(mul(q, logRest(x)))))),
    sf: (x) => where(lessEqual(x, 0), 1, where(greaterEqual(x, 1), 0, exp(mul(q, logRest(x))))),
    logSf: (x) => where(lessEqual(x, 0), 0, where(greaterEqual(x, 1), Number.NEGATIVE_INFINITY, mul(q, logRest(x)))),
    ppf,
    isf: (u) => pow(neg(expm1(div(log(u), q))), div(1, p)),
    rsample: (key, shape) => ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => bb(moment(1)),
    variance: () => bb(sub(moment(2), square(moment(1)))),
    mode: () => {
      const m = pow(div(sub(p, 1), sub(mul(p, q), 1)), div(1, p))
      return bb(
        where(logicalAnd(greaterEqual(p, 1), logicalAnd(greaterEqual(q, 1), greater(mul(p, q), 1))), m, Number.NaN),
      )
    },
    entropy: () => {
      const harmonic = add(digamma(add(q, 1)), EULER)
      return bb(sub(add(sub(1, div(1, q)), mul(sub(1, div(1, p)), harmonic)), log(mul(p, q))))
    },
  })
}

/**
 * The triangular distribution on [low, high] with its peak at `mode` (scipy.stats.triang(c, loc=low,
 * scale=high - low) with c = (mode - low) / (high - low)).
 *
 * @param low - Lower bound.
 * @param mode - Peak (low <= mode <= high).
 * @param high - Upper bound (above low).
 * @returns The distribution; `rsample` is the closed-form quantile of uniform noise, differentiable in all three
 *   parameters.
 * @example
 * ```ts
 * import { triangular } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(triangular(0, 1, 2).cdf(1)) // => 0.5
 * item(triangular(0, 1, 2).mean()) // => 1
 * ```
 */
export function triangular(low: TensorLike, mode: TensorLike, high: TensorLike): Distribution {
  const [a, c, b] = promoteFloat(low, mode, high) as [Value, Value, Value]
  validateParams('triangular', { low: [a, 'finite'], mode: [c, 'finite'], high: [b, 'finite'] })
  const batchShape = batchShapeOf(a, c, b)
  const bb = (v: Value): Value => broadcastBatch(v, batchShape)
  const width = sub(b, a)
  const left = sub(c, a)
  const right = sub(b, c)
  const safeLeft = where(greater(left, 0), left, 1)
  const safeRight = where(greater(right, 0), right, 1)
  const peak = div(left, width)
  const outside = (x: Value): Value => logicalOr(less(x, a), greater(x, b))
  const rising = (x: Value): Value => less(x, c)
  const ppf = (q: Value): Value =>
    where(lessEqual(q, peak), add(a, sqrt(mul(mul(q, width), left))), sub(b, sqrt(mul(mul(sub(1, q), width), right))))
  return defineDistribution({
    name: 'triangular',
    params: { low: a, mode: c, high: b },
    support: { kind: 'interval', low: a, high: b },
    batchShape,
    logProb: (x) => {
      const up = div(mul(sub(x, a), 2), mul(width, safeLeft))
      const down = div(mul(sub(b, x), 2), mul(width, safeRight))
      return where(outside(x), Number.NEGATIVE_INFINITY, log(where(rising(x), up, down)))
    },
    cdf: (x) => {
      const lower = div(square(sub(x, a)), mul(width, safeLeft))
      const upper = sub(1, div(square(sub(b, x)), mul(width, safeRight)))
      return where(lessEqual(x, a), 0, where(greaterEqual(x, b), 1, where(lessEqual(x, c), lower, upper)))
    },
    sf: (x) => {
      const lower = sub(1, div(square(sub(x, a)), mul(width, safeLeft)))
      const upper = div(square(sub(b, x)), mul(width, safeRight))
      return where(lessEqual(x, a), 1, where(greaterEqual(x, b), 0, where(lessEqual(x, c), lower, upper)))
    },
    ppf,
    rsample: (key, shape) => ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => bb(div(add(add(a, b), c), 3)),
    variance: () =>
      bb(div(sub(add(add(square(a), square(b)), square(c)), add(add(mul(a, b), mul(a, c)), mul(b, c))), 18)),
    mode: () => bb(c),
    entropy: () => bb(add(log(mul(width, 0.5)), 0.5)),
  })
}

/**
 * The logit-normal distribution of sigmoid(Y) for Y ~ N(loc, scale^2) on (0, 1) (TensorFlow Probability's
 * LogitNormal). It can be bimodal (for scale above about 1.4).
 *
 * @param loc - Mean of the logit.
 * @param scale - Standard deviation of the logit (positive).
 * @returns The distribution: closed-form density, CDF, quantiles and `rsample` (sigmoid(loc + scale eps)); the
 *   moments and entropy by quadrature over Y (they have no closed form), and the mode by search.
 * @example
 * ```ts
 * import { logitNormal } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(logitNormal(0, 1).cdf(0.5)) // => 0.5
 * Math.abs(item(logitNormal(0, 1).mean()) - 0.5) < 1e-14 // => true
 * ```
 */
export function logitNormal(loc: TensorLike = 0, scale: TensorLike = 1): Distribution {
  const [mu, sigma] = promoteFloat(loc, scale) as [Value, Value]
  validateParams('logitNormal', { loc: [mu, 'real'], scale: [sigma, 'positive'] })
  const batchShape = batchShapeOf(mu, sigma)
  const inside = (x: Value): Value => logicalAnd(greater(x, 0), less(x, 1))
  const safe = (x: Value): Value => where(inside(x), x, 0.5)
  const z = (x: Value): Value => div(sub(logit(safe(x)), mu), sigma)
  const fromNormal = (e: Value): Value => sigmoid(add(mu, mul(sigma, e)))
  // E[g(Y)] for the logit Y, by quadrature in standard units t: Y = loc + scale t.
  const overLogit = (g: (y: Value) => Value): Value =>
    integrate(
      (t) => mul(exp(normalLogPdf(t)), g(add(mu, mul(sigma, t)))),
      { kind: 'real', center: 0, scale: 1 },
      batchShape.length,
    )
  const mean = (): Value => broadcastBatch(overLogit(sigmoid), batchShape)
  const twin = concreteTwin([mu, sigma], (m, s) => logitNormal(m as Tensor, s as Tensor))
  const self: Distribution = defineDistribution({
    name: 'logitNormal',
    params: { loc: mu, scale: sigma },
    support: { kind: 'unitInterval' },
    batchShape,
    logProb: (x) =>
      where(
        inside(x),
        sub(sub(normalLogPdf(z(x)), log(sigma)), add(log(safe(x)), log1p(neg(safe(x))))),
        Number.NEGATIVE_INFINITY,
      ),
    cdf: (x) => where(lessEqual(x, 0), 0, where(greaterEqual(x, 1), 1, ndtr(z(x)))),
    sf: (x) => where(lessEqual(x, 0), 1, where(greaterEqual(x, 1), 0, ndtr(neg(z(x))))),
    logCdf: (x) => where(lessEqual(x, 0), Number.NEGATIVE_INFINITY, where(greaterEqual(x, 1), 0, logNdtr(z(x)))),
    logSf: (x) => where(lessEqual(x, 0), 0, where(greaterEqual(x, 1), Number.NEGATIVE_INFINITY, logNdtr(neg(z(x))))),
    ppf: (q) => fromNormal(ndtri(q)),
    isf: (q) => fromNormal(neg(ndtri(q))),
    rsample: (key, shape) => fromNormal(normalNoise(key, [...shape, ...batchShape])),
    mean,
    variance: () => {
      const m = mean()
      return broadcastBatch(
        overLogit((y) => square(sub(sigmoid(y), m))),
        batchShape,
      )
    },
    mode: () => numericMode(self, twin, 1e-9, 1 - 1e-9),
    // H(X) = H(Y) + E[log X + log(1 - X)], since dx/dy = x (1 - x).
    entropy: () =>
      broadcastBatch(
        add(
          add(log(sigma), 0.5 * Math.log(2 * Math.PI * Math.E)),
          overLogit((y) => add(logSigmoid(y), logSigmoid(neg(y)))),
        ),
        batchShape,
      ),
  })
  return self
}
