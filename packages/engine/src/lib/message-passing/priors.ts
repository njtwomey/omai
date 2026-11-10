/**
 * Priors as factors of a graph: the `prior` factor of a variable with the natural parameters of a Gaussian, gamma,
 * beta, Bernoulli, discrete, Dirichlet, normal-gamma or multivariate Gaussian, written from the parameters people
 * think in (a mean and a variance, a shape and a rate). Each returns the factor to put in a graph's `factors`.
 */

import { OmaiError } from '../base/index.ts'
import { bernoulliEta, betaEta, gammaEta, gaussianEta, normalGammaEta } from './families.ts'
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
