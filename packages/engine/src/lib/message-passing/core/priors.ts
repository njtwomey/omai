/**
 * Priors as factors of a graph: the `prior` factor of a variable with the natural parameters of a Gaussian, gamma,
 * beta, Bernoulli, discrete, Dirichlet, normal-gamma or multivariate Gaussian, written from the parameters people
 * think in (a mean and a variance, a shape and a rate). Each returns the factor to put in a graph's `factors`.
 */

import { OmaiError } from '../../base/index.ts'
import { bernoulliEta, betaEta, gamma, gammaEta, gaussianEta, normalGammaEta } from './families.ts'
import { discreteEta, gaussianMvEta } from './families-vector.ts'
import type { Eta, FactorSpec } from './types.ts'

/**
 * The `prior` factor of a variable with the given natural parameters (in its family's layout).
 *
 * @param variable - The variable's name.
 * @param eta - The natural parameters of the normalised density.
 * @returns The factor.
 * @example
 * ```ts
 * import { priorFactor } from 'omai-engine/message-passing'
 * priorFactor('x', [0, 1]).operator // => 'prior'
 * ```
 */
export function priorFactor(variable: string, eta: Eta): FactorSpec {
  return { operator: 'prior', args: [variable], params: { eta: [...eta] } }
}

/**
 * A Gaussian prior N(mean, variance) on a variable.
 *
 * @param variable - The variable's name.
 * @param mean - The mean.
 * @param variance - The variance.
 * @returns The factor.
 * @example
 * ```ts
 * import { gaussianPrior } from 'omai-engine/message-passing'
 * gaussianPrior('x', 1, 4).params // => { eta: [0.25, 0.25] }
 * ```
 */
export const gaussianPrior = (variable: string, mean: number, variance: number): FactorSpec =>
  priorFactor(variable, gaussianEta(mean, variance))

/**
 * A multivariate Gaussian prior on a vector variable.
 *
 * @param variable - The variable's name.
 * @param mean - The mean vector.
 * @param covariance - The covariance, row-major.
 * @returns The factor.
 * @example
 * ```ts
 * import { gaussianMvPrior } from 'omai-engine/message-passing'
 * gaussianMvPrior('w', [0, 0], [1, 0, 0, 1]).params // => { eta: [0, 0, 1, 0, 0, 1] }
 * ```
 */
export const gaussianMvPrior = (variable: string, mean: readonly number[], covariance: readonly number[]): FactorSpec =>
  priorFactor(variable, gaussianMvEta(mean, covariance))

/**
 * A gamma prior on a positive variable.
 *
 * @param variable - The variable's name.
 * @param shape - The shape.
 * @param rate - The rate.
 * @returns The factor.
 * @example
 * ```ts
 * import { gammaPrior } from 'omai-engine/message-passing'
 * gammaPrior('tau', 2, 1).params // => { eta: [1, -1] }
 * ```
 */
export const gammaPrior = (variable: string, shape: number, rate: number): FactorSpec =>
  priorFactor(variable, gammaEta(shape, rate))

/**
 * A beta prior on a probability.
 *
 * @param variable - The variable's name.
 * @param a - The first shape.
 * @param b - The second shape.
 * @returns The factor.
 * @example
 * ```ts
 * import { betaPrior } from 'omai-engine/message-passing'
 * betaPrior('p', 2, 3).params // => { eta: [1, 2] }
 * ```
 */
export const betaPrior = (variable: string, a: number, b: number): FactorSpec => priorFactor(variable, betaEta(a, b))

/**
 * A Bernoulli prior with probability p of true.
 *
 * @param variable - The variable's name.
 * @param p - The probability.
 * @returns The factor.
 * @example
 * ```ts
 * import { bernoulliPrior } from 'omai-engine/message-passing'
 * bernoulliPrior('b', 0.5).params // => { eta: [0] }
 * ```
 */
export const bernoulliPrior = (variable: string, p: number): FactorSpec => priorFactor(variable, bernoulliEta(p))

/**
 * A discrete prior with the given probabilities.
 *
 * @param variable - The variable's name.
 * @param probs - The probabilities of the states.
 * @returns The factor.
 * @throws OmaiError with code `value` for probabilities that are negative or sum to zero.
 * @example
 * ```ts
 * import { discretePrior } from 'omai-engine/message-passing'
 * discretePrior('z', [0.5, 0.5]).operator // => 'prior'
 * ```
 */
export function discretePrior(variable: string, probs: readonly number[]): FactorSpec {
  if (probs.some((p) => p < 0) || probs.every((p) => p === 0))
    throw new OmaiError('value', 'discretePrior: needs non-negative probabilities that do not all vanish')
  return priorFactor(variable, discreteEta(probs))
}

/**
 * A Dirichlet prior with the given concentrations.
 *
 * @param variable - The variable's name.
 * @param alpha - The concentrations.
 * @returns The factor.
 * @example
 * ```ts
 * import { dirichletPrior } from 'omai-engine/message-passing'
 * dirichletPrior('pi', [1, 2, 3]).params // => { eta: [0, 1, 2] }
 * ```
 */
export const dirichletPrior = (variable: string, alpha: readonly number[]): FactorSpec =>
  priorFactor(
    variable,
    alpha.map((a) => a - 1),
  )

/**
 * A normal-gamma prior on a mean and precision: mu | lambda ~ N(m, 1 / (beta lambda)), lambda ~ Gamma(a, b).
 *
 * @param variable - The variable's name.
 * @param m - The location of the mean.
 * @param beta - The precision multiplier of the mean.
 * @param a - The gamma shape.
 * @param b - The gamma rate.
 * @returns The factor.
 * @example
 * ```ts
 * import { normalGammaPrior } from 'omai-engine/message-passing'
 * normalGammaPrior('theta', 0, 1, 2, 1).params // => { eta: [0, -0.5, -1, 1.5] }
 * ```
 */
export const normalGammaPrior = (variable: string, m: number, beta: number, a: number, b: number): FactorSpec =>
  priorFactor(variable, normalGammaEta(m, beta, a, b))

/** What {@link projectedPrior} needs to know of a density: where it may be positive, and where to start looking for its mass. */
export interface ProjectionOptions {
  /** The family the density is projected onto (default `gaussian`; `gamma` needs a positive density). */
  readonly family?: 'gaussian' | 'gamma'
  /** The support of the density, `[lo, hi]` (default the whole line, or the positive half line for a gamma). */
  readonly support?: readonly [number, number]
  /** A point near the bulk of the mass, where the search for it starts (default 0, or 1 for a gamma). */
  readonly start?: number
}

/** log density values on a grid of points, in the vectorised form the projection evaluates (one call per scan). */
export type LogDensity = (x: Float64Array) => Float64Array

const SCAN = 801
const FINE = 6001

/** The window of the line holding the mass of a log density: widened from `start` until the edges are negligible, then narrowed onto a spike. */
function massWindow(logDensity: LogDensity, lo: number, hi: number, start: number): readonly [number, number] {
  let a = Math.max(lo, start - 5)
  let b = Math.min(hi, start + 5)
  for (let round = 0; round < 40; round++) {
    const xs = Float64Array.from({ length: SCAN }, (_, i) => a + ((b - a) * i) / (SCAN - 1))
    const lp = logDensity(xs)
    let top = Number.NEGATIVE_INFINITY
    for (const v of lp) if (v > top) top = v
    if (!Number.isFinite(top)) {
      // Nothing finite in the window: look wider.
      const span = b - a
      a = Math.max(lo, a - 2 * span)
      b = Math.min(hi, b + 2 * span)
      continue
    }
    const span = b - a
    let first = 0
    let last = SCAN - 1
    while (first < SCAN - 1 && (lp[first] as number) < top - 40) first++
    while (last > 0 && (lp[last] as number) < top - 40) last--
    const wide = (first === 0 && a > lo) || (last === SCAN - 1 && b < hi)
    if (wide) {
      if (first === 0 && a > lo) a = Math.max(lo, a - 2 * span)
      if (last === SCAN - 1 && b < hi) b = Math.min(hi, b + 2 * span)
      continue
    }
    const pad = (2 * span) / (SCAN - 1)
    const na = Math.max(lo, a + (span * first) / (SCAN - 1) - pad)
    const nb = Math.min(hi, a + (span * last) / (SCAN - 1) + pad)
    // A window much wider than the mass is narrowed onto it and scanned again (a spike a grid of this size would miss).
    if (nb - na < 0.5 * span) {
      a = na
      b = nb
      continue
    }
    return [a, b]
  }
  return [a, b]
}

/**
 * The mean and variance of a density given by its log (up to a constant) on an interval: a scan finds where the mass is,
 * then a fine trapezoid rule integrates it. The moments the Gaussian projection of a density needs, found once.
 *
 * @param logDensity - The log density, on an array of points.
 * @param lo - The lower end of the support (default -Infinity).
 * @param hi - The upper end (default Infinity).
 * @param start - A point near the bulk of the mass (default 0).
 * @returns The mean and variance.
 * @example
 * ```ts
 * import { densityMoments } from 'omai-engine/message-passing'
 * // The density proportional to exp(-x^2 / 8): N(0, 4).
 * const m = densityMoments((x) => x.map((v) => -(v * v) / 8))
 * [m.mean.toFixed(4), m.variance.toFixed(3)] // => ['0.0000', '4.000']
 * ```
 */
export function densityMoments(
  logDensity: LogDensity,
  lo: number = Number.NEGATIVE_INFINITY,
  hi: number = Number.POSITIVE_INFINITY,
  start = 0,
): { readonly mean: number; readonly variance: number; readonly meanLog: number } {
  const [a, b] = massWindow(logDensity, lo, hi, Math.min(Math.max(start, lo), hi))
  const xs = Float64Array.from({ length: FINE }, (_, i) => a + ((b - a) * i) / (FINE - 1))
  const lp = logDensity(xs)
  let top = Number.NEGATIVE_INFINITY
  for (const v of lp) if (v > top) top = v
  let z = 0
  let m1 = 0
  let ml = 0
  for (let i = 0; i < FINE; i++) {
    const w = Math.exp((lp[i] as number) - top) * (i === 0 || i === FINE - 1 ? 0.5 : 1)
    z += w
    m1 += w * (xs[i] as number)
    ml += w * Math.log(Math.max(xs[i] as number, Number.MIN_VALUE))
  }
  const mean = m1 / z
  let m2 = 0
  for (let i = 0; i < FINE; i++)
    m2 += Math.exp((lp[i] as number) - top) * (i === 0 || i === FINE - 1 ? 0.5 : 1) * ((xs[i] as number) - mean) ** 2
  return { mean, variance: m2 / z, meanLog: ml / z }
}

/**
 * The prior factor of a variable whose density has no message family (a half-normal, half-Cauchy, uniform or truncated
 * normal): the density projected once onto the variable's family by matching moments (the KL projection onto a Gaussian:
 * its mean and variance; onto a gamma: `E[x]` and `E[log x]`), and put in the graph as an ordinary prior. The model the
 * graph then holds has the projected prior, not the original: the Gaussian cannot have a bound or a half line, the gamma a
 * heavy tail, and what the projection loses is the prior's shape, not the posterior's (which EP projects again, site by site).
 *
 * @param variable - The variable's name.
 * @param logDensity - The log density (up to a constant) on an array of points.
 * @param options - The family, the support and a starting point for the search for the mass.
 * @returns The prior factor.
 * @concept factor-operators
 * @index function A prior factor from any density: projected once onto a Gaussian or gamma by moment matching.
 * @example
 * ```ts
 * import { projectedPrior } from 'omai-engine/message-passing'
 * // Uniform on [0, 1]: mean 1/2, variance 1/12, a Gaussian with the same two moments (natural parameters h = mean / variance, J = 1 / variance).
 * const f = projectedPrior('u', (x) => x.map(() => 0), { support: [0, 1], start: 0.5 })
 * f.params?.['eta']?.map((v) => (v as number).toFixed(3)) // => ['6.000', '12.000']
 * ```
 */
export function projectedPrior(variable: string, logDensity: LogDensity, options: ProjectionOptions = {}): FactorSpec {
  const family = options.family ?? 'gaussian'
  const [lo, hi] =
    options.support ??
    (family === 'gamma' ? [0, Number.POSITIVE_INFINITY] : [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY])
  const m = densityMoments(logDensity, lo, hi, options.start ?? (family === 'gamma' ? 1 : 0))
  if (family === 'gamma') {
    if (!(lo >= 0)) throw new OmaiError('value', 'projectedPrior: a gamma needs a density on the positive half line')
    return priorFactor(variable, gamma.fromMoments([m.meanLog, m.mean]))
  }
  if (!(m.variance > 0) || !Number.isFinite(m.variance))
    throw new OmaiError('value', 'projectedPrior: the density has no finite variance')
  return gaussianPrior(variable, m.mean, m.variance)
}
