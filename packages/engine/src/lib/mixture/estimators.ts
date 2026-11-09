/**
 * The mixture estimators on the iterative base: `gaussianMixture` (scikit-learn's `GaussianMixture`, EM with `nInit`
 * restarts) and `bayesianGaussianMixture` (`BayesianGaussianMixture`, variational inference), over the traced
 * algorithms of `gaussian.ts` and `variational.ts`. A fitted mixture predicts the most responsible component, gives the
 * responsibilities, the per-row log density (`scoreSamples`) and its mean (`score`), and for the maximum-likelihood
 * mixture the information criteria (`bic`, `aic`).
 */

import { type IterativeEstimator, iterative, type LearnData } from '../learn/index.ts'
import { choice, dict, int } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, tensor } from '../tensor/index.ts'
import {
  type CovarianceType,
  componentLabels,
  type GaussianEmState,
  gaussianEmAlgorithm,
  initialMixture,
  initialResponsibilities,
  type MixtureInit,
  mixtureParameterCount,
  mixturePosterior,
} from './gaussian.ts'
import {
  expectedWeights,
  type VariationalState,
  variationalMixtureAlgorithm,
  variationalPriors,
  variationalResponsibilities,
  type WeightPrior,
} from './variational.ts'

const argmaxRows = (resp: readonly (readonly number[])[]): Tensor => tensor(componentLabels(resp))

const matrixOf = (rows: readonly (readonly number[])[], k: number): Tensor =>
  fromStorage(Float64Array.from(rows.flat()), [rows.length, k], 'float64')

/** The fitted parameters of {@link gaussianMixture}: weights, means and covariances as full matrices, and the fit's summary. */
export type GaussianMixtureParams = {
  readonly weights: number[]
  readonly means: number[][]
  /** Component covariances `[k][d][d]` (the shared one repeated for `tied`, diagonal for `diag` and `spherical`). */
  readonly covariances: number[][][]
  readonly covarianceType: CovarianceType
  /** The mean log likelihood per row of the training data under these parameters. */
  readonly logLikelihood: number
}

/** The verbs of a fitted Gaussian mixture. */
export interface GaussianMixtureVerbs {
  /** The most responsible component of each row `[m]`. */
  readonly predict: (x: TensorLike) => Tensor
  /** The responsibilities `[m, k]`. */
  readonly predictProba: (x: TensorLike) => Tensor
  /** The log density of each row under the mixture `[m]` (scikit-learn's `score_samples`). */
  readonly scoreSamples: (x: TensorLike) => Tensor
  /** The mean log likelihood per row (scikit-learn's `score`; higher is better). */
  readonly score: (data: LearnData) => number
  /** The Bayesian information criterion of the rows (lower is better). */
  readonly bic: (x: TensorLike) => number
  /** The Akaike information criterion of the rows (lower is better). */
  readonly aic: (x: TensorLike) => number
  readonly nFeaturesIn: number
}

const gaussianMixtureVerbs = (p: GaussianMixtureParams): GaussianMixtureVerbs => {
  const d = p.means[0]?.length ?? 0
  const k = p.weights.length
  const post = (x: TensorLike): { readonly resp: number[][]; readonly logNorm: number[] } => mixturePosterior(x, p)
  const information = (x: TensorLike, penalty: (n: number) => number): number => {
    const { logNorm } = post(x)
    const n = logNorm.length
    const total = logNorm.reduce((a, v) => a + v, 0)
    return -2 * total + mixtureParameterCount(k, d, p.covarianceType) * penalty(n)
  }
  return {
    predict: (x) => argmaxRows(post(x).resp),
    predictProba: (x) => matrixOf(post(x).resp, k),
    scoreSamples: (x) => tensor(post(x).logNorm),
    score: (data) => {
      const { logNorm } = post(data.x)
      return logNorm.reduce((a, v) => a + v, 0) / logNorm.length
    },
    bic: (x) => information(x, Math.log),
    aic: (x) => information(x, () => 2),
    nFeaturesIn: d,
  }
}

/** Hyperparameters of {@link gaussianMixture}. */
export interface GaussianMixtureHyper {
  /** The number of components (default 1). */
  readonly nComponents: number
  /** How covariances are shared and shaped (default `full`). */
  readonly covarianceType: CovarianceType
  /** The tolerance on the change of the mean log likelihood (default 1e-3). */
  readonly tol: number
  /** Added to every covariance diagonal (default 1e-6). */
  readonly regCovar: number
  /** The step budget, one EM iteration each (default 100). */
  readonly maxIter: number
  /** Restarts from different starts; the highest likelihood wins (default 1; given parameters run once). */
  readonly nInit: number
  /** The start: `kmeans` (default), `random` responsibilities, or given weights, means and covariances. */
  readonly init: MixtureInit
}

/**
 * A Gaussian mixture by expectation-maximisation (scikit-learn's `GaussianMixture`) on the iterative base: full, tied,
 * diagonal or spherical covariances, a k-means or random start (or given parameters), `nInit` restarts keeping the
 * highest likelihood. The fitted value predicts components, gives responsibilities (`predictProba`), the per-row log
 * density (`scoreSamples`), the mean log likelihood (`score`) and `bic` and `aic`. `.session(data)` steps the EM.
 *
 * @index estimator Gaussian mixture by EM: predict, predictProba, scoreSamples, bic, aic.
 * @example
 * ```ts
 * import { gaussianMixture } from 'omai-engine/mixture'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0], [0.2], [-0.1], [10], [10.2], [9.9]]
 * const f = gaussianMixture.with({ nComponents: 2 }).fit({ x })
 * toFlat(f.predict([[0.1], [10.1]])).length // => 2
 * f.bic(x) < gaussianMixture.with({ nComponents: 1 }).fit({ x }).bic(x) // => true
 * ```
 */
export const gaussianMixture: IterativeEstimator<
  GaussianMixtureHyper,
  GaussianEmState,
  GaussianMixtureParams,
  GaussianMixtureVerbs
> = iterative<GaussianMixtureHyper, GaussianEmState, GaussianMixtureParams, GaussianMixtureVerbs>({
  name: 'mixture/gaussian',
  kind: 'clusterer',
  defaults: {
    nComponents: 1,
    covarianceType: 'full',
    tol: 1e-3,
    regCovar: 1e-6,
    maxIter: 100,
    nInit: 1,
    init: 'kmeans',
  },
  hyperSpace: (h) =>
    dict({
      nComponents: int(1, 12, { default: h.nComponents, label: 'components' }),
      covarianceType: choice(['full', 'tied', 'diag', 'spherical'] as const, { default: h.covarianceType }),
    }),
  algorithm: (data, h, key) =>
    gaussianEmAlgorithm(
      data.x,
      initialMixture(
        data.x,
        { nComponents: h.nComponents, covarianceType: h.covarianceType, regCovar: h.regCovar, init: h.init },
        key,
      ),
      { covarianceType: h.covarianceType, regCovar: h.regCovar, tol: h.tol },
    ),
  maxSteps: (h) => h.maxIter,
  restarts: {
    count: (h) => (typeof h.init === 'string' ? h.nInit : 1),
    better: (a, b) => a.logLikelihood > b.logLikelihood,
  },
  params: (s, data, h) => {
    const p = { weights: s.weights, means: s.means, covariances: s.covariances, covarianceType: h.covarianceType }
    const { logNorm } = mixturePosterior(data.x, p)
    return { ...p, logLikelihood: logNorm.reduce((a, v) => a + v, 0) / logNorm.length }
  },
  verbs: gaussianMixtureVerbs,
})

/** The variational posterior as plain data (the fields of `VariationalPosterior`, as a type with the shape JSON needs). */
export type PosteriorData = {
  readonly concentration: number[] | [number[], number[]]
  readonly meanPrecision: number[]
  readonly means: number[][]
  readonly degreesOfFreedom: number[]
  readonly covariances: number[][][]
}

/** The fitted parameters of {@link bayesianGaussianMixture}: the expected weights, the posterior and the lower bound. */
export type BayesianMixtureParams = {
  /** The expected mixing weights. */
  readonly weights: number[]
  readonly means: number[][]
  readonly covariances: number[][][]
  readonly weightPrior: WeightPrior
  readonly posterior: PosteriorData
}

/** The verbs of a fitted Bayesian Gaussian mixture. */
export interface BayesianMixtureVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  /** The log of the expected joint, summed over components (scikit-learn's `score_samples`), `[m]`. */
  readonly scoreSamples: (x: TensorLike) => Tensor
  /** The mean of `scoreSamples` over the rows. */
  readonly score: (data: LearnData) => number
  readonly nFeaturesIn: number
}

const bayesianMixtureVerbs = (p: BayesianMixtureParams): BayesianMixtureVerbs => {
  const k = p.weights.length
  return {
    predict: (x) => argmaxRows(variationalResponsibilities(x, p.posterior).resp),
    predictProba: (x) => matrixOf(variationalResponsibilities(x, p.posterior).resp, k),
    scoreSamples: (x) => tensor(variationalResponsibilities(x, p.posterior).logNorm),
    score: (data) => {
      const { logNorm } = variationalResponsibilities(data.x, p.posterior)
      return logNorm.reduce((a, v) => a + v, 0) / logNorm.length
    },
    nFeaturesIn: p.means[0]?.length ?? 0,
  }
}

/** Hyperparameters of {@link bayesianGaussianMixture}. */
export interface BayesianMixtureHyper {
  /** The number of components, an upper bound the data may leave unused (default 1). */
  readonly nComponents: number
  readonly covarianceType: CovarianceType
  /** `dirichlet_process` (default) or `dirichlet_distribution`. */
  readonly weightPrior: WeightPrior
  /** The weight concentration prior (default `1 / nComponents`). */
  readonly weightConcentrationPrior?: number
  /** The precision prior on the means (default 1). */
  readonly meanPrecisionPrior?: number
  /** The Wishart degrees of freedom prior (default the number of features). */
  readonly degreesOfFreedomPrior?: number
  readonly tol: number
  readonly regCovar: number
  readonly maxIter: number
  /** Restarts; the highest lower bound wins (default 1). */
  readonly nInit: number
  /** The initial responsibilities: `kmeans` (default) or `random`. */
  readonly init: 'kmeans' | 'random'
}

/**
 * A Bayesian Gaussian mixture by variational inference (scikit-learn's `BayesianGaussianMixture`) on the iterative
 * base: a Dirichlet or Dirichlet-process prior on the weights lets the data switch off components it does not need
 * (read `params.weights`), with data-dependent Normal-Wishart priors. `.session(data)` steps the coordinate ascent.
 *
 * @index estimator Bayesian Gaussian mixture by variational inference; unused components fade out.
 * @example
 * ```ts
 * import { bayesianGaussianMixture } from 'omai-engine/mixture'
 * const x = [[0], [0.2], [-0.1], [0.1], [10], [10.2], [9.9], [10.1]]
 * const f = bayesianGaussianMixture.with({ nComponents: 3, weightConcentrationPrior: 0.01 }).fit({ x })
 * f.params.weights.length // => 3
 * ```
 */
export const bayesianGaussianMixture: IterativeEstimator<
  BayesianMixtureHyper,
  VariationalState,
  BayesianMixtureParams,
  BayesianMixtureVerbs
> = iterative<BayesianMixtureHyper, VariationalState, BayesianMixtureParams, BayesianMixtureVerbs>({
  name: 'mixture/bayesian-gaussian',
  kind: 'clusterer',
  defaults: {
    nComponents: 1,
    covarianceType: 'full',
    weightPrior: 'dirichlet_process',
    tol: 1e-3,
    regCovar: 1e-6,
    maxIter: 100,
    nInit: 1,
    init: 'kmeans',
  },
  hyperSpace: (h) =>
    dict({
      nComponents: int(1, 12, { default: h.nComponents, label: 'components' }),
      covarianceType: choice(['full', 'tied', 'diag', 'spherical'] as const, { default: h.covarianceType }),
      weightPrior: choice(['dirichlet_process', 'dirichlet_distribution'] as const, { default: h.weightPrior }),
    }),
  algorithm: (data, h, key) =>
    variationalMixtureAlgorithm(
      data.x,
      initialResponsibilities(data.x, h.nComponents, h.init, key),
      variationalPriors(data.x, h.nComponents, {
        covarianceType: h.covarianceType,
        ...(h.weightConcentrationPrior === undefined ? {} : { weightConcentration: h.weightConcentrationPrior }),
        ...(h.meanPrecisionPrior === undefined ? {} : { meanPrecision: h.meanPrecisionPrior }),
        ...(h.degreesOfFreedomPrior === undefined ? {} : { degreesOfFreedom: h.degreesOfFreedomPrior }),
      }),
      { weightPrior: h.weightPrior, covarianceType: h.covarianceType, regCovar: h.regCovar, tol: h.tol },
    ),
  maxSteps: (h) => h.maxIter,
  restarts: { count: (h) => h.nInit, better: (a, b) => a.lowerBound > b.lowerBound },
  params: (s, _data, h) => ({
    weights: expectedWeights(s.posterior),
    means: s.posterior.means,
    covariances: s.posterior.covariances,
    weightPrior: h.weightPrior,
    posterior: s.posterior,
  }),
  verbs: bayesianMixtureVerbs,
})
