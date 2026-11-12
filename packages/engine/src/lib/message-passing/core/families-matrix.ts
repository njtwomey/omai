/**
 * The matrix message families: the Wishart on a symmetric positive-definite matrix (the conjugate prior of the precision
 * of a multivariate Gaussian, the matrix analogue of the gamma) and the normal-Wishart on a mean vector and a precision
 * matrix jointly (the analogue of the normal-gamma). Same contract as the scalar families of `families.ts`: messages are
 * natural parameters, product and quotient add and subtract them, and the family knows its moments, the member with given
 * moments (the KL projection), the log-partition, the entropy and the KL.
 *
 * A matrix is carried as its full `d x d` row-major array (symmetric), so the pairing of a message with the sufficient
 * statistics is the plain dot product, and a message built by moment matching is symmetrised.
 *
 * @index module The matrix message families: Wishart and normal-Wishart.
 */

import { type Json, LOG_2PI } from '../../base/index.ts'
import { ddot, dgemv } from '../../numerics/linalg/index.ts'
import {
  add,
  diagonal,
  div,
  mul,
  outer,
  reshape,
  sub,
  type TensorLike,
  toArray,
  toFlat,
  transpose,
} from '../../tensor/index.ts'
import { float64Of, identityArray, nanArray } from '../numerics/_arrays.ts'
import { spd } from '../numerics/_dense.ts'
import { digammaOf, lgammaOf, trigammaOf } from '../numerics/_scalar.ts'
import { ImproperMessageError, ParameterError } from './errors.ts'
import { defineFamily } from './families.ts'
import type { Eta, Family } from './types.ts'

/** The log multivariate gamma function `log Gamma_d(k)`. */
export function logMultiGamma(k: number, d: number): number {
  let s = (d * (d - 1) * Math.log(Math.PI)) / 4
  for (let j = 0; j < d; j++) s += lgammaOf(k - j / 2)
  return s
}

/** The multivariate digamma `psi_d(k) = sum_j digamma(k - j / 2)`. */
export function multiDigamma(k: number, d: number): number {
  let s = 0
  for (let j = 0; j < d; j++) s += digammaOf(k - j / 2)
  return s
}

/** The multivariate trigamma `sum_j trigamma(k - j / 2)`, the derivative of `multiDigamma`. */
function multiTrigamma(k: number, d: number): number {
  let s = 0
  for (let j = 0; j < d; j++) s += trigammaOf(k - j / 2)
  return s
}

/** The mirror image of a square matrix across its diagonal averaged with it: the symmetric part, row-major. */
function symmetric(a: TensorLike, d: number): number[] {
  const m = reshape(a, [d, d])
  return toFlat(mul(add(m, transpose(m)), 0.5))
}

/** `f` times the entries of `a` (which has exactly `n` of them). */
const scaled = (a: TensorLike, f: number, n: number): number[] => toFlat(mul(reshape(a, [n]), f))

/** A `d x d` row-major array as nested rows, for a summary. */
const rowsOf = (a: TensorLike, d: number): number[][] => toArray(reshape(a, [d, d])) as number[][]

/**
 * The Wishart shape `k = nu / 2` that matches `E log |Lambda|` and `E Lambda`: the root of `d log k - psi_d(k) = log |E Lambda| - E log |Lambda|`
 * (the matrix version of the gamma shape equation), by a safeguarded Newton iteration.
 *
 * @param logDetMean - `E log |Lambda|`.
 * @param meanLogDet - `log |E Lambda|`.
 * @param d - The dimension.
 * @returns The shape `k > (d - 1) / 2`.
 * @throws OmaiError with code `value` when `log |E Lambda| <= E log |Lambda|` (Jensen's inequality fails).
 */
export function wishartShapeFromMoments(logDetMean: number, meanLogDet: number, d: number): number {
  const s = meanLogDet - logDetMean
  if (!(s > 0))
    throw new ParameterError(
      `wishart: E log|L| = ${logDetMean} and log|E L| = ${meanLogDet} violate Jensen's inequality`,
    )
  const floor = (d - 1) / 2
  const h = (k: number): number => d * Math.log(k) - multiDigamma(k, d) - s
  // h falls from +infinity at the floor to 0 at infinity: bracket the root, then Newton with bisection as the safeguard.
  let lo = floor + 1e-12 * (1 + floor)
  let hi = Math.max(d / (2 * s), floor + 1)
  while (h(hi) > 0 && hi < 1e15) hi *= 2
  let k = Math.min(Math.max(d / (2 * s), lo + 1e-9), hi)
  for (let it = 0; it < 200; it++) {
    const f = h(k)
    if (f > 0) lo = k
    else hi = k
    const slope = d / k - multiTrigamma(k, d)
    let next = k - f / slope
    if (!(next > lo && next < hi)) next = 0.5 * (lo + hi)
    if (Math.abs(next - k) < 1e-14 * k) {
      k = next
      break
    }
    k = next
  }
  return k
}

/** The Wishart message from the degrees of freedom `nu` and the scale matrix `S` (row-major `d x d`): `[(nu - d - 1) / 2, -S^-1 / 2]`. */
export function wishartEta(nu: number, scale: ArrayLike<number>, d: number): Eta {
  const factor = spd(scale, d)
  if (factor === undefined) throw new ImproperMessageError('wishartEta: the scale matrix is not positive definite')
  return [(nu - d - 1) / 2, ...scaled(symmetric(factor.inverse(), d), -0.5, d * d)]
}

/** The degrees of freedom, the scale `S` and `E Lambda = nu S` of a proper Wishart message of dimension `d`. */
export function wishartParams(
  eta: Eta,
  d: number,
): { readonly nu: number; readonly scale: Float64Array; readonly mean: Float64Array } | undefined {
  const factor = spd(scaled(eta.slice(1, 1 + d * d), -1, d * d), d)
  if (factor === undefined || !((eta[0] ?? -1) > -1)) return undefined
  const k = (eta[0] ?? 0) + (d + 1) / 2
  const inverse = factor.inverse()
  return { nu: 2 * k, scale: float64Of(mul(inverse, 0.5)), mean: float64Of(mul(inverse, k)) }
}

/** The `d x d` identity as a row-major array. */
export const identityMatrix = (d: number): number[] => identityArray(d)

/**
 * The Wishart family on `d x d` symmetric positive-definite matrices, messages `[a, B]` with `a = (nu - d - 1) / 2` and `B = -S^-1 / 2`
 * (a `d x d` row-major block): `log q = a log |L| + tr(B L) - A`, sufficient statistics `(log |L|, L)` with
 * `E[T] = (psi_d(nu / 2) - log |-B|, (nu / 2) (-B)^-1)`. The conjugate prior of the precision of a Gaussian, so a Gaussian
 * observation `N(y; mu, L^-1)` is the message `[1/2, -(y - mu)(y - mu)' / 2]` exactly; with `d = 1` it is the gamma
 * (shape `nu / 2`, rate `-B`). `fromMoments` matches `E log |L|` and `E L`, the exact KL projection. Proper for `a > -1` and `-B`
 * positive definite.
 *
 * @param d - The dimension.
 * @returns The family.
 * @throws OmaiError with code `value` for a dimension below 1.
 * @concept exponential-families
 * @index function The Wishart message family of a given dimension: the precision matrix of a Gaussian.
 * @example
 * ```ts
 * import { identityMatrix, wishart, wishartEta } from 'omai-engine/message-passing'
 * const f = wishart(2)
 * f.summary(wishartEta(5, identityMatrix(2), 2)).mean // => [[5, 0], [0, 5]]
 * ```
 */
export function wishart(d: number): Family {
  if (!(Number.isInteger(d) && d >= 1)) throw new ParameterError(`wishart: the dimension must be at least 1, got ${d}`)
  const n = d * d
  const blockOf = (eta: Eta): number[] => scaled(eta.slice(1, 1 + n), -1, n)
  return defineFamily({
    spec: { kind: 'wishart', size: d },
    dim: d,
    size: 1 + n,
    isProper: (eta) => eta.every(Number.isFinite) && (eta[0] ?? -1) > -1 && spd(blockOf(eta), d) !== undefined,
    moments: (eta) => {
      const p = wishartParams(eta, d)
      if (p === undefined) return nanArray(1 + n)
      const factor = spd(blockOf(eta), d) as NonNullable<ReturnType<typeof spd>>
      return [multiDigamma(p.nu / 2, d) - factor.logDet, ...p.mean]
    },
    fromMoments: (m) => {
      const mean = symmetric(m.slice(1), d)
      const factor = spd(mean, d)
      if (factor === undefined) throw new ImproperMessageError('wishart.fromMoments: E[L] is not positive definite')
      const k = wishartShapeFromMoments(m[0] ?? 0, factor.logDet, d)
      // -B = k (E L)^-1
      return [k - (d + 1) / 2, ...scaled(symmetric(factor.inverse(), d), -k, n)]
    },
    logPartition: (eta) => {
      const factor = spd(blockOf(eta), d)
      if (factor === undefined || !((eta[0] ?? -1) > -1)) return Number.NaN
      const k = (eta[0] ?? 0) + (d + 1) / 2
      return -k * factor.logDet + logMultiGamma(k, d)
    },
    change: (a, b) => {
      const pa = wishartParams(a, d)
      const pb = wishartParams(b, d)
      if (pa === undefined || pb === undefined) return Number.POSITIVE_INFINITY
      let worst = 0.5 * Math.abs(Math.log(pb.nu / pa.nu))
      for (let i = 0; i < d; i++)
        for (let j = 0; j < d; j++) {
          const sd = Math.sqrt(
            pb.nu * ((pb.scale[i * d + j] ?? 0) ** 2 + (pb.scale[i * d + i] ?? 0) * (pb.scale[j * d + j] ?? 0)),
          )
          worst = Math.max(worst, Math.abs((pb.mean[i * d + j] ?? 0) - (pa.mean[i * d + j] ?? 0)) / sd)
        }
      return worst
    },
    summary: (eta) => {
      const p = wishartParams(eta, d)
      const rows = (a: TensorLike): Json => rowsOf(a, d)
      if (p === undefined) return { a: eta[0] ?? 0, B: rows(eta.slice(1)) }
      // Var(L_ij) = nu (S_ij^2 + S_ii S_jj), the outer product of the diagonal with itself plus the squared scale.
      const scale = reshape(p.scale, [d, d])
      const diag = diagonal(scale)
      const variance = toFlat(mul(add(mul(scale, scale), outer(diag, diag)), p.nu))
      return { nu: p.nu, scale: rows(p.scale), mean: rows(p.mean), variance: rows(variance) }
    },
  })
}

/** The normal-Wishart message from `(m, beta, nu, S)`: `mu | L ~ N(m, (beta L)^-1)`, `L ~ W(nu, S)`. */
export function normalWishartEta(m: readonly number[], beta: number, nu: number, scale: ArrayLike<number>): Eta {
  const d = m.length
  const factor = spd(scale, d)
  if (factor === undefined)
    throw new ImproperMessageError('normalWishartEta: the scale matrix is not positive definite')
  const inverse = factor.inverse()
  const betaM = mul(m, beta)
  const b = toFlat(mul(add(reshape(inverse, [d, d]), outer(betaM, m)), -0.5))
  return [(nu - d) / 2, ...symmetric(b, d), ...toFlat(betaM), -beta / 2]
}

/** The parameters `(m, beta, nu, S)` of a normal-Wishart message of dimension `d`, or `undefined` when it has no density. */
export function normalWishartParams(
  eta: Eta,
  d: number,
):
  | {
      readonly m: Float64Array
      readonly beta: number
      readonly nu: number
      readonly scale: Float64Array
      readonly inverseScale: Float64Array
    }
  | undefined {
  const beta = -2 * (eta[1 + d * d + d] ?? 0)
  if (!(beta > 0)) return undefined
  const m = div(eta.slice(1 + d * d, 1 + d * d + d), beta)
  const inverseScale = sub(reshape(mul(eta.slice(1, 1 + d * d), -2), [d, d]), outer(mul(m, beta), m))
  const factor = spd(symmetric(toFlat(inverseScale), d), d)
  const nu = 2 * (eta[0] ?? 0) + d
  if (factor === undefined || !(nu > d - 1)) return undefined
  return {
    m: float64Of(m),
    beta,
    nu,
    scale: factor.inverse(),
    inverseScale: Float64Array.from(symmetric(toFlat(inverseScale), d)),
  }
}

/**
 * The normal-Wishart family on a mean vector and a precision matrix jointly, the conjugate prior of a multivariate Gaussian with
 * both unknown: `mu | L ~ N(m, (beta L)^-1)`, `L ~ W(nu, S)`. Messages are the coefficients of the sufficient statistics
 * `(log |L|, L, L mu, mu' L mu)`, which is what the Gaussian observations of a vector multiply into; `E[T] = (psi_d(nu / 2) + d log 2 +
 * log |S|, nu S, nu S m, d / beta + nu m' S m)`. The matrix version of the normal-gamma family.
 *
 * @param d - The dimension.
 * @returns The family.
 * @throws OmaiError with code `value` for a dimension below 1.
 * @concept exponential-families
 * @index function The normal-Wishart message family: a Gaussian vector's mean and precision matrix jointly.
 * @example
 * ```ts
 * import { identityMatrix, normalWishart, normalWishartEta } from 'omai-engine/message-passing'
 * const f = normalWishart(2)
 * f.summary(normalWishartEta([1, 2], 4, 5, identityMatrix(2))).mean // => [1, 2]
 * ```
 */
export function normalWishart(d: number): Family {
  if (!(Number.isInteger(d) && d >= 1))
    throw new ParameterError(`normalWishart: the dimension must be at least 1, got ${d}`)
  const n = d * d
  return defineFamily({
    spec: { kind: 'normal-wishart', size: d },
    dim: d,
    size: 2 + n + d,
    isProper: (eta) => eta.every(Number.isFinite) && normalWishartParams(eta, d) !== undefined,
    moments: (eta) => {
      const p = normalWishartParams(eta, d)
      if (p === undefined) return nanArray(2 + n + d)
      const factor = spd(p.inverseScale, d) as NonNullable<ReturnType<typeof spd>>
      const nuS = float64Of(mul(p.scale, p.nu))
      const nuSm = dgemv(nuS, d, d, p.m)
      const quad = ddot(nuSm, p.m)
      return [multiDigamma(p.nu / 2, d) + d * Math.LN2 - factor.logDet, ...nuS, ...nuSm, d / p.beta + quad]
    },
    fromMoments: (mo) => {
      const mean = symmetric(mo.slice(1, 1 + n), d)
      const factor = spd(mean, d)
      if (factor === undefined)
        throw new ImproperMessageError('normalWishart.fromMoments: E[L] is not positive definite')
      const g = mo.slice(1 + n, 1 + n + d)
      const m = factor.solve(g)
      const gm = ddot(g, m)
      const spread = (mo[1 + n + d] ?? 0) - gm
      if (!(spread > 0))
        throw new ImproperMessageError('normalWishart.fromMoments: the moments imply a non-positive 1 / beta')
      const k = wishartShapeFromMoments(mo[0] ?? 0, factor.logDet, d)
      // E log|L| = psi_d(nu / 2) + d log 2 + log |S| and E L = nu S, so the Wishart part is that of `wishart(d).fromMoments` with log|L| shifted.
      const nu = 2 * k
      const scale = scaled(symmetric(mean, d), 1 / nu, n)
      return normalWishartEta(Array.from(m), d / spread, nu, scale)
    },
    logPartition: (eta) => {
      const p = normalWishartParams(eta, d)
      if (p === undefined) return Number.NaN
      const factor = spd(p.inverseScale, d) as NonNullable<ReturnType<typeof spd>>
      return (
        0.5 * d * LOG_2PI -
        0.5 * d * Math.log(p.beta) +
        0.5 * p.nu * d * Math.LN2 -
        0.5 * p.nu * factor.logDet +
        logMultiGamma(p.nu / 2, d)
      )
    },
    summary: (eta) => {
      const p = normalWishartParams(eta, d)
      if (p === undefined) return {}
      const rows = (a: TensorLike): Json => rowsOf(a, d)
      return {
        m: Array.from(p.m),
        beta: p.beta,
        nu: p.nu,
        scale: rows(p.scale),
        mean: Array.from(p.m),
        precisionMean: rows(toFlat(mul(p.scale, p.nu))),
      }
    },
  })
}
