/**
 * Copulas, distributions on the unit cube with uniform marginals that describe dependence separately from the marginal
 * distributions (Sklar's theorem): the `Copula` contract and `defineCopula`, the Gaussian and Student-t copulas, and
 * `copulaDistribution`, which joins any marginals with a copula into one joint distribution.
 */

import { MdmlError } from '../../base/index.ts'
import { cholesky, solveTriangular } from '../../numerics/linalg/index.ts'
import { lgamma, ndtr, ndtri, owensT, stdtr, stdtrit } from '../../numerics/special/index.ts'
import { foldIn, type Key, normal as normalNoise } from '../../random/index.ts'
import {
  add,
  arcsin,
  clip,
  diagonal,
  div,
  equal,
  exp,
  expandDims,
  less,
  log,
  log1p,
  matmul,
  mul,
  primalOf,
  promoteFloat,
  sliceAxis,
  sqrt,
  square,
  squeeze,
  stack,
  stopGradient,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { integrate } from '../_numeric.ts'
import { gammaSample } from '../_sampling.ts'
import { type Distribution, defineDistribution, inSupport, unavailable, validateParams } from '../distribution.ts'
import { coordinate } from './_coordinates.ts'

/**
 * A copula: a distribution on the unit cube [0, 1]^d with uniform marginals, describing dependence separately from
 * the marginal distributions (Sklar's theorem). Values `u` have the d coordinates on the last axis.
 *
 * @concept copulas
 */
export interface Copula {
  readonly name: string
  readonly dim: number
  readonly params: Readonly<Record<string, Value>>
  readonly hasRsample: boolean
  logDensity(u: TensorLike): Value
  density(u: TensorLike): Value
  cdf(u: TensorLike): Value
  /** Draws `[...shape, d]` on the unit cube. */
  sample(key: Key, shape?: readonly number[]): Value
  /** Reparameterised draws (elliptical copulas), differentiable in the parameters. */
  rsample(key: Key, shape?: readonly number[]): Value
  /** Kendall's tau: the matrix of pairwise taus for elliptical copulas, the common tau for Archimedean ones. */
  kendallTau(): Value
}

/** What a copula must define; {@link defineCopula} fills in the rest. */
export interface CopulaSpec {
  readonly name: string
  readonly dim: number
  readonly params: Readonly<Record<string, Value>>
  logDensity(u: Value): Value
  cdf?(u: Value): Value
  sample?(key: Key, shape: readonly number[]): Value
  rsample?(key: Key, shape: readonly number[]): Value
  kendallTau(): Value
}

/**
 * Build a copula from its defining parts: `density = exp(logDensity)`, `sample = stopGradient(rsample)` when there is
 * a reparameterised sampler, and methods that are not given throw.
 *
 * @param spec - The defining parts.
 * @returns The copula.
 * @concept copulas
 * @example
 * ```ts
 * import { defineCopula } from 'mdml-engine/dists'
 * import { item, mul, sum } from 'mdml-engine/tensor'
 * const independence = defineCopula({
 *   name: 'independence', dim: 2, params: {},
 *   logDensity: (u) => mul(sum(u, -1), 0), kendallTau: () => mul(sum([0]), 0),
 * })
 * item(independence.density([0.3, 0.6])) // => 1
 * ```
 */
export function defineCopula(spec: CopulaSpec): Copula {
  const v = (u: TensorLike): Value => promoteFloat(u)[0] as Value
  const rsample = (key: Key, shape: readonly number[] = []): Value =>
    spec.rsample ? spec.rsample(key, shape) : unavailable(spec.name, 'rsample (no reparameterisation)')
  return {
    name: spec.name,
    dim: spec.dim,
    params: spec.params,
    hasRsample: spec.rsample !== undefined,
    logDensity: (u) => spec.logDensity(v(u)),
    density: (u) => exp(spec.logDensity(v(u))),
    cdf: (u) => (spec.cdf ? spec.cdf(v(u)) : unavailable(spec.name, 'cdf')),
    sample: (key, shape = []) => (spec.sample ? spec.sample(key, shape) : stopGradient(rsample(key, shape))),
    rsample,
    kendallTau: spec.kendallTau,
  }
}

/**
 * The bivariate standard normal CDF P(X <= h, Y <= k) with correlation rho, by Owen's T function (Owen, 1956);
 * zeros are nudged by 1e-12, where the formula's arguments are singular (an error below 1e-12).
 */
function bivariateNormalCdf(h0: Value, k0: Value, rho: Value): Value {
  const h = where(equal(h0, 0), 1e-12, h0)
  const k = where(equal(k0, 0), 1e-12, k0)
  const s = sqrt(sub(1, square(rho)))
  const a1 = div(sub(k, mul(rho, h)), mul(h, s))
  const a2 = div(sub(h, mul(rho, k)), mul(k, s))
  const beta = where(less(mul(h, k), 0), 0.5, 0)
  return sub(sub(sub(mul(add(ndtr(h), ndtr(k)), 0.5), owensT(h, a1)), owensT(k, a2)), beta)
}

/** Validate a correlation matrix and return it with its Cholesky factor and log-determinant. */
function correlationParts(correlation: TensorLike, family: string): { r: Value; l: Value; logDet: Value; d: number } {
  const [r] = promoteFloat(correlation) as [Value]
  if (r.shape.length !== 2 || r.shape[0] !== r.shape[1])
    throw new MdmlError('shape', `${family}: the correlation must be a single d x d matrix`)
  if (!toFlat(primalOf(inSupport({ kind: 'correlation' }, r))).every((ok) => ok === 1))
    throw new MdmlError('value', `${family}: not a positive-definite correlation matrix`)
  const l = cholesky(r)
  return { r, l, logDet: mul(sum(log(diagonal(l)), -1), 2), d: r.shape[0] ?? 0 }
}

const ellipticalTau = (r: Value): Value => mul(arcsin(r), 2 / Math.PI)

/**
 * The Gaussian copula with correlation matrix R (statsmodels' GaussianCopula): the dependence of a multivariate normal,
 * density phi_R(z) / prod phi(z_i) at z = Phi^{-1}(u).
 *
 * @param correlation - The correlation matrix `[d, d]`.
 * @returns The copula; `rsample` maps reparameterised normal draws through Phi. The CDF is the bivariate normal CDF
 *   (via Owen's T) for d = 2.
 * @concept copulas
 * @example
 * ```ts
 * import { gaussianCopula } from 'mdml-engine/dists'
 * import { toFlat } from 'mdml-engine/tensor'
 * // Kendall's tau of a Gaussian copula: (2 / pi) asin(rho).
 * toFlat(gaussianCopula([[1, 0.5], [0.5, 1]]).kendallTau()).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 0.333333333333, 0.333333333333, 1]
 * ```
 */
export function gaussianCopula(correlation: TensorLike): Copula {
  const { r, l, logDet, d } = correlationParts(correlation, 'gaussianCopula')
  const whiten = (z: Value): Value => squeeze(solveTriangular(l, expandDims(z, -1), { lower: true }), -1)
  return defineCopula({
    name: 'gaussianCopula',
    dim: d,
    params: { correlation: r },
    logDensity: (u) => {
      const z = ndtri(u)
      return add(mul(sub(sum(square(z), -1), sum(square(whiten(z)), -1)), 0.5), mul(logDet, -0.5))
    },
    cdf: (u) =>
      d === 2
        ? bivariateNormalCdf(
            ndtri(coordinate(u, 0)),
            ndtri(coordinate(u, 1)),
            squeeze(sliceAxis(sliceAxis(r, 0, 0, 1), 1, 1, 2)),
          )
        : unavailable('gaussianCopula', 'cdf for d > 2 (a multivariate normal CDF)'),
    rsample: (key, shape) => {
      const eps = normalNoise(key, [...shape, d])
      return ndtr(squeeze(matmul(l, expandDims(eps, -1)), -1))
    },
    kendallTau: () => ellipticalTau(r),
  })
}

/**
 * The Student-t copula with correlation matrix R and `df` degrees of freedom: the dependence of a multivariate t,
 * with tail dependence that the Gaussian copula lacks.
 *
 * @param correlation - The correlation matrix `[d, d]`.
 * @param df - Degrees of freedom (positive).
 * @returns The copula; `rsample` maps reparameterised multivariate-t draws (normal over the root of an implicitly
 *   reparameterised chi-square) through the t CDF. The bivariate CDF is E_S[Phi_2(h S, k S; rho)] for
 *   S = sqrt(V / df), by quadrature over S.
 * @example
 * ```ts
 * import { studentTCopula } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * // Independence-like correlation still has t-copula dependence: the density at the centre exceeds 1.
 * item(studentTCopula([[1, 0], [0, 1]], 3).density([0.5, 0.5])) > 1 // => true
 * ```
 */
export function studentTCopula(correlation: TensorLike, df: TensorLike): Copula {
  const { r, l, logDet, d } = correlationParts(correlation, 'studentTCopula')
  const [nu] = promoteFloat(df) as [Value]
  validateParams('studentTCopula', { df: [nu, 'positive'] })
  const whiten = (z: Value): Value => squeeze(solveTriangular(l, expandDims(z, -1), { lower: true }), -1)
  const halfNu = mul(nu, 0.5)
  const logT1 = (t: Value): Value =>
    sub(
      sub(sub(lgamma(mul(add(nu, 1), 0.5)), lgamma(halfNu)), mul(log(mul(nu, Math.PI)), 0.5)),
      mul(mul(add(nu, 1), 0.5), log1p(div(square(t), nu))),
    )
  // log density of S = sqrt(V / df), V chi-square: log 2 + (df/2) log(df/2) - lgamma(df/2) + (df-1) log s - df s^2/2.
  const logS = (s: Value): Value =>
    add(
      sub(add(Math.LN2, mul(halfNu, log(halfNu))), lgamma(halfNu)),
      sub(mul(sub(nu, 1), log(s)), mul(halfNu, square(s))),
    )
  return defineCopula({
    name: 'studentTCopula',
    dim: d,
    params: { correlation: r, df: nu },
    logDensity: (u) => {
      const t = stdtrit(nu, u)
      const maha = sum(square(whiten(t)), -1)
      const joint = sub(
        sub(sub(lgamma(mul(add(nu, d), 0.5)), lgamma(halfNu)), mul(log(mul(nu, Math.PI)), d / 2)),
        add(mul(logDet, 0.5), mul(mul(add(nu, d), 0.5), log1p(div(maha, nu)))),
      )
      return sub(joint, sum(logT1(t), -1))
    },
    cdf: (u) => {
      if (d !== 2) return unavailable('studentTCopula', 'cdf for d > 2')
      const h = stdtrit(nu, coordinate(u, 0))
      const k = stdtrit(nu, coordinate(u, 1))
      const rho = squeeze(sliceAxis(sliceAxis(r, 0, 0, 1), 1, 1, 2))
      // Over w = log s, centred at 0 with the spread of log S.
      const scale = div(1, sqrt(mul(nu, 2)))
      return integrate(
        (w) => {
          // Far nodes carry no weight; clipping keeps their (zero-weighted) derivatives finite.
          const s = exp(clip(w, -30, 30))
          return mul(exp(add(logS(s), w)), bivariateNormalCdf(mul(h, s), mul(k, s), rho))
        },
        { kind: 'real', center: 0, scale },
        h.shape.length,
      )
    },
    rsample: (key, shape) => {
      const eps = normalNoise(foldIn(key, 0), [...shape, d])
      const z = squeeze(matmul(l, expandDims(eps, -1)), -1)
      const v = mul(gammaSample(foldIn(key, 1), halfNu, shape), 2)
      return stdtr(nu, div(z, expandDims(sqrt(div(v, nu)), -1)))
    },
    kendallTau: () => ellipticalTau(r),
  })
}

/**
 * The joint distribution of marginals joined by a copula (Sklar's theorem): F(x) = C(F_1(x_1), ..., F_d(x_d)), with
 * log density log c(F_1(x_1), ...) + sum_i log f_i(x_i).
 *
 * @param copula - The copula.
 * @param marginals - One univariate distribution per coordinate (each with a CDF and quantile function).
 * @returns The distribution with event shape `[d]`: draws are marginal quantiles of copula draws (reparameterised when
 *   the copula is), the mean and variance are the marginals'; the mode and entropy throw.
 * @throws MdmlError with code `shape` when the number of marginals differs from the copula's dimension.
 * @example
 * ```ts
 * import { copulaDistribution, exponential, gaussianCopula, normal } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * const d = copulaDistribution(gaussianCopula([[1, 0.5], [0.5, 1]]), [normal(1, 2), exponential(2)])
 * toArray(d.mean()) // => [1, 0.5]
 * ```
 */
export function copulaDistribution(copula: Copula, marginals: readonly Distribution[]): Distribution {
  const d = copula.dim
  if (marginals.length !== d)
    throw new MdmlError('shape', `copulaDistribution: ${d} dimensions but ${marginals.length} marginals`)
  for (const m of marginals)
    if (m.eventShape.length > 0 || m.batchShape.length > 0)
      throw new MdmlError('shape', 'copulaDistribution: marginals must be scalar and unbatched')
  const toCube = (x: Value): Value =>
    stack(
      marginals.map((m, i) => m.cdf(coordinate(x, i))),
      -1,
    )
  const fromCube = (u: Value): Value =>
    stack(
      marginals.map((m, i) => m.ppf(coordinate(u, i))),
      -1,
    )
  const spec = {
    name: 'copulaDistribution',
    params: { ...copula.params },
    support: { kind: 'real' } as const,
    batchShape: [],
    eventShape: [d],
    logProb: (x: Value): Value =>
      add(
        copula.logDensity(toCube(x)),
        sum(
          stack(
            marginals.map((m, i) => m.logProb(coordinate(x, i))),
            -1,
          ),
          -1,
        ),
      ),
    cdf: (x: Value): Value => copula.cdf(toCube(x)),
    sample: (key: Key, shape: readonly number[]): Value => fromCube(copula.sample(key, shape)),
    mean: (): Value =>
      stack(
        marginals.map((m) => m.mean()),
        -1,
      ),
    variance: (): Value =>
      stack(
        marginals.map((m) => m.variance()),
        -1,
      ),
    mode: (): Value => unavailable('copulaDistribution', 'mode'),
    entropy: (): Value => unavailable('copulaDistribution', 'entropy (estimate it by Monte Carlo)'),
  }
  return defineDistribution(
    copula.hasRsample
      ? { ...spec, rsample: (key: Key, shape: readonly number[]) => fromCube(copula.rsample(key, shape)) }
      : spec,
  )
}
