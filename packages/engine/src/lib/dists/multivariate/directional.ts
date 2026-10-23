import { maxOf, WhiteboxError } from '../../base/index.ts'
import { digamma, ive, lgamma, logIv } from '../../numerics/special/index.ts'
import {
  beta as betaDraw,
  foldIn,
  type Key,
  normal as normalNoise,
  uniform as uniformNoise,
  vonMises as vonMisesDraw,
} from '../../random/index.ts'
import {
  add,
  arange,
  arctan2,
  concat,
  cos,
  div,
  expandDims,
  fromStorage,
  greater,
  log,
  log1p,
  mul,
  neg,
  norm,
  primalOf,
  promoteFloat,
  reshape,
  sin,
  sqrt,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { concreteTwin, numericPpf } from '../_numeric.ts'
import { discreteSample, elementwiseSample, implicitReparam } from '../_sampling.ts'
import { beta } from '../continuous/bounded.ts'
import { batchShapeOf, broadcastBatch, type Distribution, defineDistribution, validateParams } from '../distribution.ts'

const LOG_2PI = Math.log(2 * Math.PI)

/**
 * The von Mises distribution on the circle (scipy.stats.vonmises, torch.distributions.VonMises): density
 * exp(kappa cos(x - loc)) / (2 pi I_0(kappa)). The CDF is the Fourier series
 * F = (theta + pi) / (2 pi) + sum_n I_n(kappa) sin(n theta) / (n pi I_0(kappa)) with theta = x - loc (so it rises by
 * one per turn, as scipy's does), summed over the terms that carry the weight.
 *
 * @param loc - Mean direction (radians).
 * @param concentration - kappa (non-negative).
 * @returns The distribution. `sample` uses Best and Fisher's rejection sampler and `rsample` differentiates it by
 *   implicit reparameterisation through the CDF (draws lie in [loc - pi, loc + pi]); the variance is the circular variance 1 - I_1(kappa) / I_0(kappa)
 *   (as in torch), and quantiles are numerical.
 * @example
 * ```ts
 * import { vonMises } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(vonMises(0, 1).cdf(0)) // => 0.5
 * ```
 */
export function vonMises(loc: TensorLike = 0, concentration: TensorLike = 1): Distribution {
  const [mu, kappa] = promoteFloat(loc, concentration) as [Value, Value]
  validateParams('vonMises', { loc: [mu, 'finite'], concentration: [kappa, 'nonNegative'] })
  const batchShape = batchShapeOf(mu, kappa)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const logNorm = add(LOG_2PI, logIv(0, kappa))
  // I_1 / I_0 without overflow.
  const ratio = div(ive(1, kappa), ive(0, kappa))
  const twin = concreteTwin([mu, kappa], (m, k) => vonMises(m as Tensor, k as Tensor))
  const cdf = (x: Value): Value => {
    const theta = sub(x, mu)
    const kmax = maxOf(toFlat(primalOf(kappa)))
    const terms = Math.ceil(30 + 12 * Math.sqrt(kmax))
    const rank = batchShapeOf(theta).length
    const n = reshape(arange(1, terms + 1), [terms, ...new Array<number>(rank).fill(1)])
    const series = sum(div(mul(div(ive(n, kappa), ive(0, kappa)), sin(mul(n, theta))), n), 0)
    return add(div(add(theta, Math.PI), 2 * Math.PI), div(series, Math.PI))
  }
  const self: Distribution = defineDistribution({
    name: 'vonMises',
    params: { loc: mu, concentration: kappa },
    support: { kind: 'circular' },
    batchShape,
    logProb: (x) => sub(mul(kappa, cos(sub(x, mu))), logNorm),
    cdf,
    ppf: (q) => numericPpf(self, twin, q),
    rsample: (key, shape) => {
      const draw = elementwiseSample(
        key,
        [mu, kappa],
        shape,
        batchShape,
        (k, m, kk) => vonMisesDraw(k, [], m ?? 0, kk ?? 0).data[0] ?? 0,
      )
      // Each draw within half a turn of loc, on the branch where the CDF runs from 0 to 1.
      const offset = sub(draw, mu)
      const centred = add(mu, arctan2(sin(offset), cos(offset)))
      return implicitReparam(centred, cdf, self.prob)
    },
    mean: () => b(mu),
    variance: () => b(sub(1, ratio)),
    mode: () => b(mu),
    entropy: () => b(sub(logNorm, mul(kappa, ratio))),
  })
  return self
}

/** Householder reflection taking e_1 to the unit vector mu, applied to y: y - 2 (u . y) u, u = (e_1 - mu) / |e_1 - mu|. */
function reflectFromFirstAxis(mu: Value, y: Value): Value {
  const d = mu.shape[mu.shape.length - 1] ?? 0
  const e1 = fromStorage(
    Float64Array.from({ length: d }, (_, i) => (i === 0 ? 1 : 0)),
    [d],
    'float64',
  )
  const diff = sub(e1, mu)
  const size = norm(diff, 2, -1, true)
  const ok = greater(size, 1e-12)
  const u = div(diff, where(ok, size, 1))
  const reflected = sub(y, mul(mul(sum(mul(u, y), -1, true), 2), u))
  return where(ok, reflected, y)
}

/** Uniform directions on the sphere S^{m-1} (normalised normals), `[...shape, m]`. */
function uniformDirections(key: Key, shape: readonly number[], m: number): Value {
  const z = normalNoise(key, [...shape, m])
  return div(z, norm(z, 2, -1, true))
}

/**
 * The von Mises-Fisher distribution on the unit sphere S^{d-1} (scipy.stats.vonmises_fisher): density
 * C_d(kappa) exp(kappa mu^T x), C_d = kappa^{d/2-1} / ((2 pi)^{d/2} I_{d/2-1}(kappa)).
 *
 * @param meanDirection - The unit mean direction mu `[..., d]`.
 * @param concentration - kappa (positive).
 * @returns The distribution with event shape `[d]`: closed-form mean A_d(kappa) mu, variances, mode and entropy, and
 *   `sample` by Wood's (1994) rejection sampler. It has no reparameterised sampler; use {@link powerSpherical} for
 *   pathwise gradients.
 * @throws WhiteboxError with code `value` when the mean direction is not a unit vector.
 * @example
 * ```ts
 * import { vonMisesFisher } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(vonMisesFisher([0, 0, 1], 2).mode()) // => [0, 0, 1]
 * ```
 */
export function vonMisesFisher(meanDirection: TensorLike, concentration: TensorLike): Distribution {
  const [mu, kappa] = promoteFloat(meanDirection, concentration) as [Value, Value]
  const d = mu.shape[mu.shape.length - 1] ?? 0
  if (d < 2) throw new WhiteboxError('shape', 'vonMisesFisher: the mean direction needs at least two coordinates')
  if (toFlat(primalOf(sum(square(mu), -1))).some((v) => Math.abs(v - 1) > 1e-8))
    throw new WhiteboxError('value', 'vonMisesFisher: the mean direction must be a unit vector')
  validateParams('vonMisesFisher', { concentration: [kappa, 'positive'] })
  const batchShape = batchShapeOf(sum(mu, -1), kappa)
  const order = d / 2 - 1
  const logC = sub(sub(mul(log(kappa), order), (d / 2) * LOG_2PI), logIv(order, kappa))
  const a = div(ive(d / 2, kappa), ive(order, kappa))
  const lifted = (v: Value): Value => expandDims(v, -1)
  return defineDistribution({
    name: 'vonMisesFisher',
    params: { meanDirection: mu, concentration: kappa },
    support: { kind: 'sphere' },
    batchShape,
    eventShape: [d],
    logProb: (x) => add(mul(kappa, sum(mul(mu, x), -1)), logC),
    sample: (key, shape) => {
      const full = [...shape, ...batchShape]
      // Wood's algorithm for the component w along mu, in the frame where mu = e_1.
      const w = discreteSample([kappa], full, 'float64', ([kt]) => {
        const ks = toFlat(broadcastBatch(kt as Tensor, batchShape))
        const size = full.reduce((p, q) => p * q, 1)
        const out = new Float64Array(size)
        for (let i = 0; i < size; i++) out[i] = woodDraw(foldIn(key, i), ks[i % Math.max(1, ks.length)] ?? 1, d)
        return fromStorage(out, full, 'float64')
      })
      const v = uniformDirections(foldIn(key, -1), full, d - 1)
      const y = concat([expandDims(w, -1), mul(expandDims(sqrt(sub(1, square(w))), -1), v)], -1)
      return reflectFromFirstAxis(mu, y)
    },
    mean: () => mul(lifted(a), mu),
    // E[x x^T] = (A / kappa) I + (1 - d A / kappa) mu mu^T.
    variance: () => {
      const ak = div(a, kappa)
      return sub(add(lifted(ak), mul(lifted(sub(1, mul(ak, d))), square(mu))), mul(lifted(square(a)), square(mu)))
    },
    mode: () => broadcastBatch(mu, [...batchShape, d]),
    entropy: () => sub(neg(mul(kappa, a)), logC),
  })
}

/** One draw of w = mu^T x for a von Mises-Fisher distribution (Wood, 1994). */
function woodDraw(key: Key, kappa: number, d: number): number {
  const m = d - 1
  const b = m / (Math.sqrt(4 * kappa * kappa + m * m) + 2 * kappa)
  const x0 = (1 - b) / (1 + b)
  const c = kappa * x0 + m * Math.log(1 - x0 * x0)
  for (let attempt = 0; attempt < 10000; attempt++) {
    const k = foldIn(key, attempt)
    const z = betaDraw(foldIn(k, 0), [], m / 2, m / 2).data[0] ?? 0.5
    const w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
    const u = uniformNoise(foldIn(k, 1), []).data[0] ?? 0.5
    if (kappa * w + m * Math.log(1 - x0 * w) - c >= Math.log(u)) return w
  }
  throw new WhiteboxError('convergence', 'vonMisesFisher: the rejection sampler did not accept')
}

/**
 * The power spherical distribution on the unit sphere S^{d-1} (De Cao and Aziz, 2020): density proportional to
 * (1 + mu^T x)^kappa, a von Mises-Fisher alternative with a reparameterised sampler and no Bessel functions.
 *
 * @param meanDirection - The unit mean direction mu `[..., d]`.
 * @param concentration - kappa (non-negative).
 * @returns The distribution with event shape `[d]`: `rsample` reflects (t, sqrt(1 - t^2) v) onto mu for
 *   t = 2 z - 1, z ~ Beta((d - 1) / 2 + kappa, (d - 1) / 2) (reparameterised), so draws differentiate in mu and kappa;
 *   closed-form mean, variances, mode and entropy.
 * @reference De Cao, N. and Aziz, W. (2020). The power spherical distribution. arXiv:2006.04437.
 * @example
 * ```ts
 * import { powerSpherical } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
 * // E[t] = (alpha - beta) / (alpha + beta) = 2 / 4 with d = 3, kappa = 2.
 * toArray(powerSpherical([0, 0, 1], 2).mean()) // => [0, 0, 0.5]
 * ```
 */
export function powerSpherical(meanDirection: TensorLike, concentration: TensorLike): Distribution {
  const [mu, kappa] = promoteFloat(meanDirection, concentration) as [Value, Value]
  const d = mu.shape[mu.shape.length - 1] ?? 0
  if (d < 2) throw new WhiteboxError('shape', 'powerSpherical: the mean direction needs at least two coordinates')
  if (toFlat(primalOf(sum(square(mu), -1))).some((v) => Math.abs(v - 1) > 1e-8))
    throw new WhiteboxError('value', 'powerSpherical: the mean direction must be a unit vector')
  validateParams('powerSpherical', { concentration: [kappa, 'nonNegative'] })
  const batchShape = batchShapeOf(sum(mu, -1), kappa)
  const half = (d - 1) / 2
  const alpha = add(kappa, half)
  const total = add(alpha, half)
  // log N = -[(alpha + beta) log 2 + beta log pi + lgamma(alpha) - lgamma(alpha + beta)].
  const logNorm = neg(sub(add(add(mul(total, Math.LN2), half * Math.log(Math.PI)), lgamma(alpha)), lgamma(total)))
  const meanT = div(sub(alpha, half), total)
  const varT = div(mul(mul(alpha, half), 4), mul(square(total), add(total, 1)))
  const lifted = (v: Value): Value => expandDims(v, -1)
  return defineDistribution({
    name: 'powerSpherical',
    params: { meanDirection: mu, concentration: kappa },
    support: { kind: 'sphere' },
    batchShape,
    eventShape: [d],
    logProb: (x) => add(logNorm, mul(kappa, log1p(sum(mul(mu, x), -1)))),
    rsample: (key, shape) => {
      const full = [...shape, ...batchShape]
      const z = beta(broadcastBatch(alpha, batchShape), half).rsample(foldIn(key, 0), shape)
      const t = sub(mul(z, 2), 1)
      const v = uniformDirections(foldIn(key, 1), full, d - 1)
      const y = concat([expandDims(t, -1), mul(expandDims(sqrt(sub(1, square(t))), -1), v)], -1)
      return reflectFromFirstAxis(mu, y)
    },
    mean: () => mul(lifted(meanT), mu),
    // E[x x^T] = E[t^2] mu mu^T + (1 - E[t^2]) / (d - 1) (I - mu mu^T).
    variance: () => {
      const t2 = add(varT, square(meanT))
      const along = mul(lifted(t2), square(mu))
      const across = mul(lifted(div(sub(1, t2), d - 1)), sub(1, square(mu)))
      return sub(add(along, across), mul(lifted(square(meanT)), square(mu)))
    },
    mode: () => broadcastBatch(mu, [...batchShape, d]),
    entropy: () => neg(add(logNorm, mul(kappa, sub(add(Math.LN2, digamma(alpha)), digamma(total))))),
  })
}
