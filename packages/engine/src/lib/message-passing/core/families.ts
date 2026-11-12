/**
 * The scalar message families: the Gaussian (precision form), the gamma, the beta, the Bernoulli and the normal-gamma,
 * each an exponential family whose messages are natural parameters, with the product, quotient and power of messages,
 * moments and the member with given moments, log-partition, entropy and KL. `defineFamily` is the one place that turns
 * a family's own pieces (moments, their inverse, the log-partition) into the full `Family`; the vector families
 * (multivariate Gaussian, discrete, Dirichlet) are in `families-vector.ts`.
 *
 * @index module The scalar message families: Gaussian, gamma, beta, Bernoulli, normal-gamma.
 */

import { type Json, LOG_2PI } from '../../base/index.ts'
import { ddot } from '../../numerics/linalg/index.ts'
import { div, toFlat } from '../../tensor/index.ts'
import { zeroArray } from '../numerics/_arrays.ts'
import { shiftedSum } from '../numerics/_moments.ts'
import { digammaOf, lgammaOf, trigammaOf } from '../numerics/_scalar.ts'
import { ImproperMessageError } from './errors.ts'
import type { Eta, Family, FamilyKind, FamilySpec, Quotient } from './types.ts'

/** What a family supplies: the pieces that differ; the rest of `Family` follows from them. */
export interface FamilyDefinition {
  readonly spec: FamilySpec
  readonly dim: number
  readonly size: number
  isProper(eta: Eta): boolean
  moments(eta: Eta): Eta
  fromMoments(moments: Eta): Eta
  logPartition(eta: Eta): number
  /** The pairing eta . T given E[T]: the dot product, except for the Gaussian's precision form. Default the dot product. */
  pair?(eta: Eta, moments: Eta): number
  change?(previous: Eta, next: Eta): number
  summary(eta: Eta): Readonly<Record<string, Json>>
  /** Applied to the result of a product, quotient or power (the discrete families renormalise their log weights). */
  normalise?(eta: Eta): Eta
  logProbs?(eta: Eta): readonly number[]
  fromLogProbs?(logProbs: readonly number[]): Eta
}

/** The largest absolute difference of two messages: the change of a message that has no moments. */
export function maxAbsDifference(a: Eta, b: Eta): number {
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] ?? 0) - (b[i] ?? 0)))
  return m
}

/**
 * A family from its pieces: product, quotient and power as the vector operations on natural parameters, entropy and KL
 * by the exponential-family identities H = A - eta.E[T] and KL(p||q) = (eta_p - eta_q).E_p[T] - A(eta_p) + A(eta_q),
 * and a change measure (the one given, or the largest difference of the natural parameters).
 *
 * @param def - The family's pieces.
 * @returns The family.
 */
export function defineFamily(def: FamilyDefinition): Family {
  const normalise = def.normalise ?? ((eta: Eta) => eta)
  const pair = def.pair ?? ddot
  const flat: Eta = zeroArray(def.size)
  // Exempt from the tensor rule, on purpose: a plain loop over a number array. A belief is the sum of every message of a
  // variable (hundreds of them for a parameter shared across time) and this is the innermost operation of a sweep, where a
  // tensor per message would allocate on every product.
  const combine = (a: Eta, b: Eta, sign: 1 | -1): Eta => {
    const out = new Array<number>(a.length)
    for (let i = 0; i < a.length; i++) out[i] = (a[i] as number) + sign * (b[i] ?? 0)
    return normalise(out)
  }
  const family: Family = {
    kind: def.spec.kind,
    spec: def.spec,
    dim: def.dim,
    size: def.size,
    flat,
    isProper: def.isProper,
    product: (a, b) => combine(a, b, 1),
    quotient: (a, b): Quotient => {
      const eta = combine(a, b, -1)
      return { eta, improper: !def.isProper(eta) }
    },
    power: (a, p) => (p === 0 ? flat : normalise(a.map((v) => v * p))),
    moments: def.moments,
    fromMoments: def.fromMoments,
    pair,
    logPartition: def.logPartition,
    entropy: (eta) => def.logPartition(eta) - pair(eta, def.moments(eta)),
    kl: (p, q) => {
      const mp = def.moments(p)
      const diff = p.map((v, i) => v - (q[i] ?? 0))
      return pair(diff, mp) - def.logPartition(p) + def.logPartition(q)
    },
    change: (previous, next) =>
      def.change !== undefined && def.isProper(previous) && def.isProper(next)
        ? def.change(previous, next)
        : maxAbsDifference(previous, next),
    summary: (eta) => ({ ...def.summary(eta), improper: !def.isProper(eta) }),
    ...(def.logProbs === undefined ? {} : { logProbs: def.logProbs }),
    ...(def.fromLogProbs === undefined ? {} : { fromLogProbs: def.fromLogProbs }),
  }
  return family
}

/** The Gaussian message from a mean and a variance (a precision form `[mean / variance, 1 / variance]`). */
export function gaussianEta(mean: number, variance: number): Eta {
  if (!(variance > 0)) throw new ImproperMessageError(`gaussianEta: the variance must be positive, got ${variance}`)
  return [mean / variance, 1 / variance]
}

/** The mean and variance of a proper Gaussian message. */
export function gaussianMoments(eta: Eta): { readonly mean: number; readonly variance: number } {
  const j = eta[1] ?? 0
  return { mean: (eta[0] ?? 0) / j, variance: 1 / j }
}

const logRatio = (a: number, b: number): number => Math.log(a / b)

/**
 * The Gaussian family on a real variable, messages in precision form `[h, J]` with `log q = h x - J x^2 / 2 - A`,
 * so `J` is the precision and `h / J` the mean; sufficient statistics (x, x^2) with E[T] = (mean, mean^2 + variance).
 *
 * @concept exponential-families
 * @index structure The Gaussian message family in precision form.
 * @example
 * ```ts
 * import { gaussian, gaussianEta } from 'omai-engine/message-passing'
 * const a = gaussianEta(1, 4)
 * const b = gaussianEta(3, 4)
 * gaussian.summary(gaussian.product(a, b)).mean // => 2
 * ```
 */
export const gaussian: Family = defineFamily({
  spec: { kind: 'gaussian' },
  dim: 1,
  size: 2,
  isProper: (eta) =>
    (eta[1] ?? 0) > 0 && Number.isFinite(eta[0] ?? Number.NaN) && Number.isFinite(eta[1] ?? Number.NaN),
  moments: (eta) => {
    const { mean, variance } = gaussianMoments(eta)
    return [mean, mean * mean + variance]
  },
  fromMoments: (m) => {
    const variance = (m[1] ?? 0) - (m[0] ?? 0) ** 2
    if (!(variance > 0))
      throw new ImproperMessageError(`gaussian.fromMoments: the variance ${variance} is not positive`)
    return gaussianEta(m[0] ?? 0, variance)
  },
  logPartition: (eta) => {
    const h = eta[0] ?? 0
    const j = eta[1] ?? 0
    return (h * h) / (2 * j) - 0.5 * Math.log(j) + 0.5 * LOG_2PI
  },
  pair: (eta, m) => (eta[0] ?? 0) * (m[0] ?? 0) - 0.5 * (eta[1] ?? 0) * (m[1] ?? 0),
  change: (a, b) => {
    const pa = gaussianMoments(a)
    const pb = gaussianMoments(b)
    return Math.max(
      Math.abs(pb.mean - pa.mean) / Math.sqrt(pb.variance),
      0.5 * Math.abs(logRatio(pb.variance, pa.variance)),
    )
  },
  summary: (eta) => {
    const j = eta[1] ?? 0
    return j > 0 ? { mean: (eta[0] ?? 0) / j, variance: 1 / j, precision: j } : { precision: j, h: eta[0] ?? 0 }
  },
})

/** The gamma message from a shape and a rate: `[shape - 1, -rate]`. */
export function gammaEta(shape: number, rate: number): Eta {
  return [shape - 1, -rate]
}

/** The shape and rate of a gamma message. */
export function gammaShapeRate(eta: Eta): { readonly shape: number; readonly rate: number } {
  return { shape: (eta[0] ?? 0) + 1, rate: -(eta[1] ?? 0) }
}

/**
 * The gamma shape that matches E[log x] and E[x]: the root of log a - digamma(a) = log E[x] - E[log x], from Minka's
 * starting value by Newton's method.
 *
 * @param logMean - E[log x].
 * @param mean - E[x].
 * @returns The shape a > 0 (the rate is a / mean).
 * @throws OmaiError with code `value` when log E[x] <= E[log x] (Jensen's inequality fails: the moments are not a gamma's).
 */
export function gammaShapeFromMoments(logMean: number, mean: number): number {
  const s = Math.log(mean) - logMean
  if (!(s > 0))
    throw new ImproperMessageError(`gamma: E[log x] = ${logMean} and E[x] = ${mean} violate Jensen's inequality`)
  let a = (3 - s + Math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
  for (let i = 0; i < 30; i++) {
    const f = Math.log(a) - digammaOf(a) - s
    const step = f / (1 / a - trigammaOf(a))
    const next = a - step
    a = next > 0 ? next : a / 2
    if (Math.abs(step) < 1e-13 * a) break
  }
  return a
}

/**
 * The gamma family on a positive variable, messages `[shape - 1, -rate]`; sufficient statistics (log x, x) with
 * E[T] = (digamma(shape) - log rate, shape / rate). Proper for a positive shape and rate.
 *
 * @concept exponential-families
 * @index structure The gamma message family on a positive variable.
 * @example
 * ```ts
 * import { gamma, gammaEta } from 'omai-engine/message-passing'
 * gamma.summary(gammaEta(3, 2)).mean // => 1.5
 * ```
 */
export const gamma: Family = defineFamily({
  spec: { kind: 'gamma' },
  dim: 1,
  size: 2,
  isProper: (eta) =>
    (eta[0] ?? -1) > -1 &&
    (eta[1] ?? 0) < 0 &&
    Number.isFinite(eta[0] ?? Number.NaN) &&
    Number.isFinite(eta[1] ?? Number.NaN),
  moments: (eta) => {
    const { shape, rate } = gammaShapeRate(eta)
    return [digammaOf(shape) - Math.log(rate), shape / rate]
  },
  fromMoments: (m) => {
    const shape = gammaShapeFromMoments(m[0] ?? 0, m[1] ?? 0)
    return gammaEta(shape, shape / (m[1] ?? 1))
  },
  logPartition: (eta) => {
    const { shape, rate } = gammaShapeRate(eta)
    return lgammaOf(shape) - shape * Math.log(rate)
  },
  change: (a, b) => {
    const pa = gammaShapeRate(a)
    const pb = gammaShapeRate(b)
    const mean = Math.abs(logRatio(pb.shape / pb.rate, pa.shape / pa.rate))
    return Math.max(mean, 0.5 * Math.abs(logRatio(pb.shape / pb.rate ** 2, pa.shape / pa.rate ** 2)))
  },
  summary: (eta) => {
    const { shape, rate } = gammaShapeRate(eta)
    return shape > 0 && rate > 0 ? { shape, rate, mean: shape / rate, variance: shape / rate ** 2 } : { shape, rate }
  },
})

/** The beta message from its two shapes: `[a - 1, b - 1]`. */
export function betaEta(a: number, b: number): Eta {
  return [a - 1, b - 1]
}

/**
 * The Dirichlet shape vector with E[log p_k] = `logMeans[k]`: Minka's Newton iteration for the Hessian that is a
 * diagonal plus a rank-one matrix, from the starting value that matches the mean and the log-variance of the largest
 * entry. The beta family uses it with two entries.
 *
 * @param logMeans - E[log p_k] for each of at least two entries.
 * @returns The shapes alpha_k > 0.
 */
export function dirichletShapesFromLogMeans(logMeans: readonly number[]): number[] {
  const k = logMeans.length
  const { weights: p, total: norm } = shiftedSum(logMeans)
  const mean = toFlat(div(p, norm))
  const spread = mean.reduce((acc, m, i) => acc + m * ((logMeans[i] ?? 0) - Math.log(m)), 0)
  const scale = Math.max((k - 1) / Math.max(-spread, 1e-12), 1e-3)
  let alpha = mean.map((m) => Math.max(m * scale, 1e-3))
  for (let it = 0; it < 60; it++) {
    const total = alpha.reduce((a, b) => a + b, 0)
    const psiTotal = digammaOf(total)
    const g = alpha.map((a, i) => psiTotal - digammaOf(a) + (logMeans[i] ?? 0))
    const q = alpha.map((a) => -trigammaOf(a))
    const z = trigammaOf(total)
    let num = 0
    let den = 1 / z
    for (let i = 0; i < k; i++) {
      num += (g[i] ?? 0) / (q[i] ?? 1)
      den += 1 / (q[i] ?? 1)
    }
    const b = num / den
    let step = 1
    let next = alpha.map((a, i) => a - ((g[i] ?? 0) - b) / (q[i] ?? 1))
    while (next.some((v) => !(v > 0)) && step > 1e-8) {
      step /= 2
      next = alpha.map((a, i) => a - (step * ((g[i] ?? 0) - b)) / (q[i] ?? 1))
    }
    const move = Math.max(...next.map((v, i) => Math.abs(v - (alpha[i] ?? 0)) / v))
    alpha = next
    if (move < 1e-13) break
  }
  return alpha
}

/**
 * The beta family on (0, 1), messages `[a - 1, b - 1]`; sufficient statistics (log x, log(1 - x)), E[T] =
 * (digamma(a) - digamma(a + b), digamma(b) - digamma(a + b)). `fromMoments` matches the log moments (the exact KL
 * projection), by a Newton iteration.
 *
 * @concept exponential-families
 * @index structure The beta message family on (0, 1).
 * @example
 * ```ts
 * import { beta, betaEta } from 'omai-engine/message-passing'
 * beta.summary(betaEta(2, 6)).mean // => 0.25
 * ```
 */
export const beta: Family = defineFamily({
  spec: { kind: 'beta' },
  dim: 1,
  size: 2,
  isProper: (eta) =>
    (eta[0] ?? -1) > -1 &&
    (eta[1] ?? -1) > -1 &&
    Number.isFinite(eta[0] ?? Number.NaN) &&
    Number.isFinite(eta[1] ?? Number.NaN),
  moments: (eta) => {
    const a = (eta[0] ?? 0) + 1
    const b = (eta[1] ?? 0) + 1
    const total = digammaOf(a + b)
    return [digammaOf(a) - total, digammaOf(b) - total]
  },
  fromMoments: (m) => {
    const [a, b] = dirichletShapesFromLogMeans([m[0] ?? 0, m[1] ?? 0])
    return betaEta(a ?? 1, b ?? 1)
  },
  logPartition: (eta) => {
    const a = (eta[0] ?? 0) + 1
    const b = (eta[1] ?? 0) + 1
    return lgammaOf(a) + lgammaOf(b) - lgammaOf(a + b)
  },
  change: (p, q) => {
    const mean = (e: Eta): number => ((e[0] ?? 0) + 1) / ((e[0] ?? 0) + (e[1] ?? 0) + 2)
    const sd = (e: Eta): number => {
      const a = (e[0] ?? 0) + 1
      const b = (e[1] ?? 0) + 1
      return Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))
    }
    return Math.max(Math.abs(mean(q) - mean(p)) / sd(q), 0.5 * Math.abs(logRatio(sd(q) ** 2, sd(p) ** 2)))
  },
  summary: (eta) => {
    const a = (eta[0] ?? 0) + 1
    const b = (eta[1] ?? 0) + 1
    return a > 0 && b > 0 ? { a, b, mean: a / (a + b), variance: (a * b) / ((a + b) ** 2 * (a + b + 1)) } : { a, b }
  },
})

const LOGIT_LIMIT = 700

/** The Bernoulli message with probability `p` of true: its logit. */
export function bernoulliEta(p: number): Eta {
  if (p <= 0) return [-LOGIT_LIMIT]
  if (p >= 1) return [LOGIT_LIMIT]
  return [Math.log(p) - Math.log1p(-p)]
}

const probability = (eta: Eta): number => {
  const x = eta[0] ?? 0
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x))
}

/**
 * The Bernoulli family on {0, 1}, messages the logit `[log(p / (1 - p))]`; sufficient statistic x, E[T] = p. Always
 * proper.
 *
 * @concept exponential-families
 * @index structure The Bernoulli message family: the logit of the probability of true.
 * @example
 * ```ts
 * import { bernoulli, bernoulliEta } from 'omai-engine/message-passing'
 * bernoulli.summary(bernoulliEta(0.25)).p // => 0.25
 * ```
 */
export const bernoulli: Family = defineFamily({
  spec: { kind: 'bernoulli' },
  dim: 1,
  size: 1,
  isProper: (eta) => Number.isFinite(eta[0] ?? Number.NaN),
  moments: (eta) => [probability(eta)],
  fromMoments: (m) => bernoulliEta(m[0] ?? 0.5),
  logPartition: (eta) => {
    const x = eta[0] ?? 0
    return x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x))
  },
  change: (a, b) => Math.abs(probability(b) - probability(a)),
  summary: (eta) => ({ p: probability(eta) }),
  logProbs: (eta) => {
    const x = eta[0] ?? 0
    const lse = x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x))
    return [-lse, x - lse]
  },
  fromLogProbs: (lp) => [(lp[1] ?? 0) - (lp[0] ?? 0)],
})

/** The normal-gamma message from `(m, beta, a, b)`: mu | lambda ~ N(m, 1 / (beta lambda)), lambda ~ Gamma(a, b). */
export function normalGammaEta(m: number, beta: number, a: number, b: number): Eta {
  return [beta * m, -beta / 2, -(beta * m * m) / 2 - b, a - 0.5]
}

/** The parameters `(m, beta, a, b)` of a normal-gamma message. */
export function normalGammaParams(eta: Eta): {
  readonly m: number
  readonly beta: number
  readonly a: number
  readonly b: number
} {
  const beta = -2 * (eta[1] ?? 0)
  const m = (eta[0] ?? 0) / beta
  return { m, beta, a: (eta[3] ?? 0) + 0.5, b: -(eta[2] ?? 0) - (eta[0] ?? 0) ** 2 / (2 * beta) }
}

/**
 * The normal-gamma family on a mean and a precision `(mu, lambda)` jointly, the conjugate prior of a Gaussian with both
 * unknown: `mu | lambda ~ N(m, 1 / (beta lambda))`, `lambda ~ Gamma(a, b)`. Messages are the coefficients of the
 * sufficient statistics `(lambda mu, lambda mu^2, lambda, log lambda)`, which is what the factors of a Gaussian with
 * unknown mean and precision (the components of a Gaussian mixture) multiply into; E[T] = (m a / b, m^2 a / b + 1 /
 * beta, a / b, digamma(a) - log b).
 *
 * @concept exponential-families
 * @index structure The normal-gamma message family on a mean and a precision jointly.
 * @example
 * ```ts
 * import { normalGamma, normalGammaEta } from 'omai-engine/message-passing'
 * normalGamma.summary(normalGammaEta(2, 4, 3, 1.5)).precisionMean // => 2
 * ```
 */
export const normalGamma: Family = defineFamily({
  spec: { kind: 'normal-gamma' },
  dim: 1,
  size: 4,
  isProper: (eta) => {
    if (!eta.every(Number.isFinite)) return false
    const { beta, a, b } = normalGammaParams(eta)
    return beta > 0 && a > 0 && b > 0
  },
  moments: (eta) => {
    const { m, beta, a, b } = normalGammaParams(eta)
    return [(m * a) / b, (m * m * a) / b + 1 / beta, a / b, digammaOf(a) - Math.log(b)]
  },
  fromMoments: (mo) => {
    const lambda = mo[2] ?? 1
    const m = (mo[0] ?? 0) / lambda
    const inverseBeta = (mo[1] ?? 0) - m * m * lambda
    if (!(inverseBeta > 0))
      throw new ImproperMessageError('normalGamma.fromMoments: the moments imply a non-positive 1 / beta')
    const a = gammaShapeFromMoments(mo[3] ?? 0, lambda)
    return normalGammaEta(m, 1 / inverseBeta, a, a / lambda)
  },
  logPartition: (eta) => {
    const { beta, a, b } = normalGammaParams(eta)
    return 0.5 * LOG_2PI - 0.5 * Math.log(beta) + lgammaOf(a) - a * Math.log(b)
  },
  change: (p, q) => {
    const x = normalGammaParams(p)
    const y = normalGammaParams(q)
    const mean = Math.abs(y.m - x.m) * Math.sqrt((y.beta * y.a) / y.b)
    return Math.max(mean, Math.abs(logRatio(y.a / y.b, x.a / x.b)), Math.abs(logRatio(y.beta, x.beta)))
  },
  summary: (eta) => {
    const { m, beta, a, b } = normalGammaParams(eta)
    return beta > 0 && a > 0 && b > 0 ? { m, beta, a, b, precisionMean: a / b, mean: m } : { m, beta, a, b }
  },
})

/** Every kind of family this entry builds without a size. */
export const scalarFamilies: Readonly<Partial<Record<FamilyKind, Family>>> = {
  gaussian,
  gamma,
  beta,
  bernoulli,
  'normal-gamma': normalGamma,
}
