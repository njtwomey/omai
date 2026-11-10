/**
 * The vector message families: the multivariate Gaussian in precision form, the discrete distribution on a finite set
 * of states (log weights) and the Dirichlet, with `familyOf`, which builds any family of the library from its JSON
 * description. Same contract as the scalar families of `families.ts`.
 *
 * @index module The vector message families: multivariate Gaussian, discrete, Dirichlet.
 */

import { type Json, logSumExpOf, OmaiError } from '../base/index.ts'
import { dgemv } from '../numerics/linalg/index.ts'
import { spd } from './_dense.ts'
import { digammaOf, lgammaOf } from './_scalar.ts'
import { bernoulli, beta, defineFamily, dirichletShapesFromLogMeans, gamma, gaussian, normalGamma } from './families.ts'
import type { Eta, Family, FamilySpec } from './types.ts'

const LOG_2PI = Math.log(2 * Math.PI)

/** A multivariate Gaussian message from a mean and a covariance (row-major): `[J m, J]` with `J = covariance^-1`. */
export function gaussianMvEta(mean: readonly number[], covariance: readonly number[]): Eta {
  const d = mean.length
  const factor = spd(covariance, d)
  if (factor === undefined) throw new OmaiError('value', 'gaussianMvEta: the covariance is not positive definite')
  const j = factor.inverse()
  return [...dgemv(j, d, d, mean), ...j]
}

/** The mean and covariance (row-major) of a proper multivariate Gaussian message of dimension d. */
export function gaussianMvMoments(
  eta: Eta,
  d: number,
): { readonly mean: Float64Array; readonly covariance: Float64Array } {
  const factor = spd(eta.slice(d), d)
  if (factor === undefined) throw new OmaiError('value', 'gaussianMvMoments: the precision is not positive definite')
  return { mean: factor.solve(eta.slice(0, d)), covariance: factor.inverse() }
}

/**
 * The multivariate Gaussian family of dimension d, messages in precision form `[h, J]` (h of length d, J row-major
 * d x d), `log q = h . x - x' J x / 2 - A`; sufficient statistics (x, x x') with E[T] = (mean, mean mean' +
 * covariance).
 *
 * @param d - The dimension.
 * @returns The family.
 * @throws OmaiError with code `value` for a dimension below 1.
 * @concept exponential-families
 * @index function The multivariate Gaussian message family of a given dimension.
 * @example
 * ```ts
 * import { gaussianMv, gaussianMvEta } from 'omai-engine/message-passing'
 * const f = gaussianMv(2)
 * f.summary(gaussianMvEta([1, 2], [1, 0, 0, 4])).mean // => [1, 2]
 * ```
 */
export function gaussianMv(d: number): Family {
  if (!(Number.isInteger(d) && d >= 1))
    throw new OmaiError('value', `gaussianMv: the dimension must be at least 1, got ${d}`)
  return defineFamily({
    spec: { kind: 'gaussian-mv', size: d },
    dim: d,
    size: d + d * d,
    isProper: (eta) => spd(eta.slice(d), d) !== undefined && eta.slice(0, d).every(Number.isFinite),
    moments: (eta) => {
      const { mean, covariance } = gaussianMvMoments(eta, d)
      const second = Array.from(covariance)
      for (let i = 0; i < d; i++)
        for (let j = 0; j < d; j++) second[i * d + j] = (second[i * d + j] ?? 0) + (mean[i] ?? 0) * (mean[j] ?? 0)
      return [...mean, ...second]
    },
    fromMoments: (m) => {
      const mean = m.slice(0, d)
      const covariance = m.slice(d).map((v, k) => v - (mean[Math.floor(k / d)] ?? 0) * (mean[k % d] ?? 0))
      return gaussianMvEta(mean, covariance)
    },
    logPartition: (eta) => {
      const factor = spd(eta.slice(d), d)
      if (factor === undefined) return Number.NaN
      const h = eta.slice(0, d)
      const x = factor.solve(h)
      let quad = 0
      for (let i = 0; i < d; i++) quad += (h[i] ?? 0) * (x[i] ?? 0)
      return 0.5 * quad - 0.5 * factor.logDet + 0.5 * d * LOG_2PI
    },
    pair: (eta, m) => {
      let s = 0
      for (let i = 0; i < d; i++) s += (eta[i] ?? 0) * (m[i] ?? 0)
      for (let k = 0; k < d * d; k++) s -= 0.5 * (eta[d + k] ?? 0) * (m[d + k] ?? 0)
      return s
    },
    change: (a, b) => {
      const pa = gaussianMvMoments(a, d)
      const pb = gaussianMvMoments(b, d)
      let worst = 0
      for (let i = 0; i < d; i++) {
        const va = pa.covariance[i * d + i] ?? 1
        const vb = pb.covariance[i * d + i] ?? 1
        worst = Math.max(
          worst,
          Math.abs((pb.mean[i] ?? 0) - (pa.mean[i] ?? 0)) / Math.sqrt(vb),
          0.5 * Math.abs(Math.log(vb / va)),
        )
      }
      return worst
    },
    summary: (eta) => {
      const factor = spd(eta.slice(d), d)
      if (factor === undefined) return { h: eta.slice(0, d), precision: eta.slice(d) }
      const covariance = factor.inverse()
      const rows: number[][] = Array.from({ length: d }, (_, i) => Array.from(covariance.subarray(i * d, (i + 1) * d)))
      return {
        mean: Array.from(factor.solve(eta.slice(0, d))),
        covariance: rows,
        variance: Array.from({ length: d }, (_, i) => covariance[i * d + i] ?? 0),
      }
    },
  })
}

const normalised = (eta: Eta): Eta => {
  const lse = logSumExpOf(eta)
  return Number.isFinite(lse) ? eta.map((v) => v - lse) : eta
}

/** The discrete message from probabilities (log weights, normalised). */
export function discreteEta(probs: readonly number[]): Eta {
  return normalised(probs.map((p) => (p > 0 ? Math.log(p) : Number.NEGATIVE_INFINITY)))
}

/** The probabilities of a discrete message. */
export function discreteProbs(eta: Eta): number[] {
  const lse = logSumExpOf(eta)
  return eta.map((v) => Math.exp(v - lse))
}

/**
 * The discrete family on k states, messages the log weights (defined up to an additive constant; products and
 * quotients renormalise them); sufficient statistic the one-hot indicator, E[T] the probabilities. Proper unless every
 * weight is zero.
 *
 * @param k - The number of states.
 * @returns The family.
 * @throws OmaiError with code `value` for fewer than two states.
 * @concept exponential-families
 * @index function The discrete message family on k states: log weights.
 * @example
 * ```ts
 * import { discrete, discreteEta } from 'omai-engine/message-passing'
 * discrete(3).summary(discreteEta([0.2, 0.3, 0.5])).probs // => [0.2, 0.3, 0.5]
 * ```
 */
export function discrete(k: number): Family {
  if (!(Number.isInteger(k) && k >= 2)) throw new OmaiError('value', `discrete: needs at least two states, got ${k}`)
  return defineFamily({
    spec: { kind: 'discrete', size: k },
    dim: k,
    size: k,
    normalise: normalised,
    isProper: (eta) =>
      eta.every((v) => !Number.isNaN(v) && v < Number.POSITIVE_INFINITY) && Math.max(...eta) > Number.NEGATIVE_INFINITY,
    moments: discreteProbs,
    fromMoments: (m) => discreteEta(m),
    logPartition: (eta) => logSumExpOf(eta),
    pair: (eta, m) => eta.reduce((s, v, i) => ((m[i] ?? 0) > 0 ? s + v * (m[i] ?? 0) : s), 0),
    change: (a, b) => {
      const pa = discreteProbs(a)
      const pb = discreteProbs(b)
      return Math.max(...pa.map((v, i) => Math.abs((pb[i] ?? 0) - v)))
    },
    summary: (eta) => (Math.max(...eta) > Number.NEGATIVE_INFINITY ? { probs: discreteProbs(eta) } : { probs: [] }),
    logProbs: (eta) => normalised(eta),
    fromLogProbs: (lp) => normalised(lp),
  })
}

/**
 * The Dirichlet family on the simplex of dimension k, messages `alpha - 1`; sufficient statistics log p, E[T] =
 * (digamma(alpha_j) - digamma(sum alpha)). `fromMoments` matches the log moments by Newton's method.
 *
 * @param k - The number of components.
 * @returns The family.
 * @throws OmaiError with code `value` for fewer than two components.
 * @concept exponential-families
 * @index function The Dirichlet message family of a given dimension.
 * @example
 * ```ts
 * import { dirichlet } from 'omai-engine/message-passing'
 * dirichlet(3).summary([0, 1, 2]).mean // => [0.1666666666666667, 0.3333333333333333, 0.5]
 * ```
 */
export function dirichlet(k: number): Family {
  if (!(Number.isInteger(k) && k >= 2))
    throw new OmaiError('value', `dirichlet: needs at least two components, got ${k}`)
  const shapes = (eta: Eta): number[] => eta.map((v) => v + 1)
  return defineFamily({
    spec: { kind: 'dirichlet', size: k },
    dim: k,
    size: k,
    isProper: (eta) => eta.every((v) => Number.isFinite(v) && v > -1),
    moments: (eta) => {
      const a = shapes(eta)
      const total = digammaOf(a.reduce((s, v) => s + v, 0))
      return a.map((v) => digammaOf(v) - total)
    },
    fromMoments: (m) => dirichletShapesFromLogMeans(m).map((v) => v - 1),
    logPartition: (eta) => {
      const a = shapes(eta)
      return a.reduce((s, v) => s + lgammaOf(v), 0) - lgammaOf(a.reduce((s, v) => s + v, 0))
    },
    change: (a, b) => {
      const pa = shapes(a)
      const pb = shapes(b)
      return Math.max(...pa.map((v, i) => Math.abs(Math.log((pb[i] ?? 1) / v))))
    },
    summary: (eta) => {
      const a = shapes(eta)
      const total = a.reduce((s, v) => s + v, 0)
      return a.every((v) => v > 0) ? { alpha: a, mean: a.map((v) => v / total) as Json } : { alpha: a }
    },
  })
}

/**
 * The family of a JSON description.
 *
 * @param spec - The kind and, for the vector families, the size.
 * @returns The family.
 * @throws OmaiError with code `value` for an unknown kind or a missing size.
 * @example
 * ```ts
 * import { familyOf } from 'omai-engine/message-passing'
 * familyOf({ kind: 'discrete', size: 3 }).size // => 3
 * ```
 */
export function familyOf(spec: FamilySpec): Family {
  const size = spec.size
  switch (spec.kind) {
    case 'gaussian':
      return gaussian
    case 'gamma':
      return gamma
    case 'beta':
      return beta
    case 'bernoulli':
      return bernoulli
    case 'normal-gamma':
      return normalGamma
    case 'gaussian-mv':
      if (size === undefined) throw new OmaiError('value', 'familyOf: gaussian-mv needs a size')
      return gaussianMv(size)
    case 'discrete':
      if (size === undefined) throw new OmaiError('value', 'familyOf: discrete needs a size')
      return discrete(size)
    case 'dirichlet':
      if (size === undefined) throw new OmaiError('value', 'familyOf: dirichlet needs a size')
      return dirichlet(size)
    default:
      throw new OmaiError('value', `familyOf: unknown family ${String((spec as { kind: unknown }).kind)}`)
  }
}
