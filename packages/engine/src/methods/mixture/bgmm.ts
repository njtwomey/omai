/**
 * The Bayesian Gaussian mixture as a particular of the iterative base: coordinate-ascent variational inference with a
 * Dirichlet or Dirichlet-process prior on the weights, which switches off the components the data does not need. Every
 * step's expected weights, means and covariances and the lower bound are on the tape, so a page can watch the unneeded
 * components die off while the needed ones settle onto the clusters.
 */

import { type IterativeEstimator, iterative } from '../../lib/learn/index.ts'
import {
  type CovarianceType,
  componentLabels,
  expectedWeights,
  initialResponsibilities,
  type VariationalPosterior,
  type VariationalState,
  variationalMixtureAlgorithm,
  variationalPriors,
  variationalResponsibilities,
  type WeightPrior,
} from '../../lib/mixture/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, tensor } from '../../lib/tensor/index.ts'
import { type Contract, defineContract, derived, Log } from '../../lib/trace/index.ts'

/** The hyperparameters of {@link bgmm}. */
export interface BgmmHyper {
  /** The most components used; the prior switches off those the data does not need. */
  readonly nComponents: number
  /** A finite Dirichlet on the weights, or the stick-breaking Dirichlet process. */
  readonly weightConcentrationPriorType: WeightPrior
  /** The concentration of the weight prior: small values favour few components. */
  readonly weightConcentrationPrior: number
  /** How component covariances are shared and shaped. */
  readonly covarianceType: CovarianceType
  /**
   * Where inference starts: from a converged k-means partition, or from random responsibilities (the components start
   * alike and the algorithm does the work; watchable).
   */
  readonly init: 'kmeans' | 'random'
  readonly maxIter: number
  /** The change of the lower bound that stops the run. */
  readonly tol: number
}

/** The fitted parameters of {@link bgmm}: plain JSON. */
export type BgmmParams = {
  /** The expected mixing weights. */
  readonly weights: number[]
  readonly means: number[][]
  /** The component covariances `[k][d][d]` (full matrices, whatever the type). */
  readonly covariances: number[][][]
  readonly lowerBound: number
  /** The posterior weight concentrations (Dirichlet), or the two Beta parameters of each stick (process). */
  readonly concentration: number[] | number[][]
  readonly meanPrecision: number[]
  readonly degreesOfFreedom: number[]
}

/** What a fitted Bayesian Gaussian mixture offers. */
export interface BgmmVerbs {
  /** The most responsible component of each row. */
  readonly predict: (x: TensorLike) => Tensor
  /** The responsibilities `[m, k]`. */
  readonly predictProba: (x: TensorLike) => Tensor
}

const posteriorOf = (p: BgmmParams): VariationalPosterior => ({
  concentration: p.concentration as VariationalPosterior['concentration'],
  meanPrecision: p.meanPrecision,
  means: p.means,
  degreesOfFreedom: p.degreesOfFreedom,
  covariances: p.covariances,
})

/**
 * The paths of the Bayesian Gaussian mixture beyond the solver’s: the component means, derived.
 *
 * @example
 * ```ts
 * import type { BgmmPath } from 'mdml-engine/methods/mixture'
 * const path: BgmmPath = 'model/centres'
 * path // => 'model/centres'
 * ```
 */
export type BgmmPath = 'model/centres'

/**
 * A Bayesian Gaussian mixture fitted by variational inference: unneeded components end with vanishing weight.
 *
 * @example
 * ```ts
 * import { bgmm } from 'mdml-engine/methods/mixture'
 * const x = [[0, 0], [0, 1], [0.5, 0.5], [9, 9], [9, 8], [8.5, 8.5]]
 * bgmm.with({ nComponents: 3 }).fit({ x }).params.weights.length // => 3
 * ```
 */
export const bgmm: IterativeEstimator<BgmmHyper, VariationalState, BgmmParams, BgmmVerbs, BgmmPath> = iterative<
  BgmmHyper,
  VariationalState,
  BgmmParams,
  BgmmVerbs,
  BgmmPath
>({
  name: 'mixture/bgmm',
  kind: 'clusterer',
  defaults: {
    nComponents: 8,
    weightConcentrationPriorType: 'dirichlet_process',
    weightConcentrationPrior: 0.01,
    covarianceType: 'full',
    init: 'kmeans',
    maxIter: 100,
    tol: 1e-3,
  },
  hyperSpace: (h) =>
    dict({
      nComponents: int(1, 12, { default: h.nComponents, label: 'components' }),
      weightConcentrationPriorType: choice(['dirichlet_process', 'dirichlet_distribution'] as const, {
        default: h.weightConcentrationPriorType,
        label: 'weight prior',
      }),
      weightConcentrationPrior: real(1e-3, 1e3, {
        log: true,
        default: h.weightConcentrationPrior,
        label: 'weight concentration',
      }),
      covarianceType: choice(['full', 'tied', 'diag', 'spherical'] as const, {
        default: h.covarianceType,
        label: 'covariance type',
      }),
      init: choice(['kmeans', 'random'] as const, { default: h.init, label: 'start from' }),
    }),
  algorithm: (data, h, key) =>
    variationalMixtureAlgorithm(
      data.x,
      initialResponsibilities(data.x, h.nComponents, h.init, key),
      variationalPriors(data.x, h.nComponents, {
        weightConcentration: h.weightConcentrationPrior,
        covarianceType: h.covarianceType,
      }),
      { weightPrior: h.weightConcentrationPriorType, covarianceType: h.covarianceType, tol: h.tol },
    ),
  maxSteps: (h) => h.maxIter,
  params: (s) => ({
    weights: expectedWeights(s.posterior),
    means: s.posterior.means,
    covariances: s.posterior.covariances,
    lowerBound: s.lowerBound,
    concentration: s.posterior.concentration,
    meanPrecision: s.posterior.meanPrecision,
    degreesOfFreedom: s.posterior.degreesOfFreedom,
  }),
  verbs: (p) => ({
    predict: (x) => tensor(componentLabels(variationalResponsibilities(x, posteriorOf(p)).resp)),
    predictProba: (x) => {
      const { resp } = variationalResponsibilities(x, posteriorOf(p))
      return fromStorage(Float64Array.from(resp.flat()), [resp.length, p.weights.length], 'float64')
    },
  }),
  contract: (): Contract<VariationalState, BgmmPath> =>
    defineContract({
      'model/centres': derived<VariationalState>({
        description: 'the component means at this step',
        compute: (s) => s.posterior.means,
      }),
    }),
  logs: {
    'solver/lowerBound': Log(),
    'solver/weights': Log({ detail: 'full' }),
    'solver/means': Log({ detail: 'full' }),
    'solver/covariances': Log({ detail: 'full' }),
  },
})
