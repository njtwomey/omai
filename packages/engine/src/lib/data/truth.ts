import type { Value } from '../tensor/index.ts'

/**
 * What is known about how a dataset was generated, so methods can be checked against the truth rather than only against
 * held-out data. Every field is optional; generators fill what they know.
 *
 * @concept dataset-truth
 */
export interface Truth {
  /** E[y | x], the function a regressor should recover. */
  readonly regressionFunction?: (x: Value) => Value
  /** The standard deviation of additive Gaussian noise on y. */
  readonly noiseScale?: number
  /** The irreducible error of the best possible predictor under squared loss (the noise variance). */
  readonly bayesError?: number
  /** The features that influence y (indices into the columns of x). */
  readonly relevantFeatures?: readonly number[]
  /** Free-form further facts (true coefficients, latent variables). */
  readonly extra?: Readonly<Record<string, unknown>>
}
