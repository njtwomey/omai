/**
 * The Gaussian mixture model as a particular of the iterative base: expectation-maximisation from a k-means start, with
 * restarts, the log likelihood, responsibilities, means and weights of every step on the tape, the mixture's density
 * over the plane as a probe, and the step-by-step record read back from that tape.
 */

import { mixtureSteps } from '../../lib/exposition/index.ts'
import { decisionSurfaceOf, type IterativeEstimator, iterative } from '../../lib/learn/index.ts'
import {
  type CovarianceType,
  componentLabels,
  type GaussianEmState,
  gaussianEmAlgorithm,
  initialMixture,
  mixturePosterior,
} from '../../lib/mixture/index.ts'
import { choice, dict, int } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, tensor } from '../../lib/tensor/index.ts'
import { type Contract, defineContract, derived, Log, metricSeries, tensorSeries } from '../../lib/trace/index.ts'

/** The hyperparameters of {@link gmm}. */
export interface GmmHyper {
  /** The number of mixture components. */
  readonly nComponents: number
  /** How component covariances are shared and shaped. */
  readonly covarianceType: CovarianceType
  /**
   * Where EM starts: from a converged k-means partition (scikit-learn's default, so EM has little left to do on
   * separated data) or from random responsibilities (EM does the work; watchable).
   */
  readonly init: 'kmeans' | 'random'
  /** Starts from fresh initialisations; the highest log likelihood wins. */
  readonly nInit: number
  readonly maxIter: number
  /** The mean log-likelihood change that stops EM. */
  readonly tol: number
}

/** The fitted parameters of {@link gmm}: plain JSON. */
export type GmmParams = {
  readonly weights: number[]
  readonly means: number[][]
  readonly covariances: number[][][]
  readonly logLikelihood: number
}

/** What a fitted Gaussian mixture offers. */
export interface GmmVerbs {
  /** The most responsible component of each row. */
  readonly predict: (x: TensorLike) => Tensor
  /** The responsibilities `[m, k]`. */
  readonly predictProba: (x: TensorLike) => Tensor
}

/**
 * The paths of the Gaussian mixture beyond the solver’s: the component means and the log density over the plane, both derived.
 *
 * @example
 * ```ts
 * import type { GmmPath } from 'mdml-engine/methods/mixture'
 * const path: GmmPath = 'model/centres'
 * path // => 'model/centres'
 * ```
 */
export type GmmPath = 'model/centres' | 'model/density'

/**
 * A Gaussian mixture fitted by expectation-maximisation.
 *
 * @example
 * ```ts
 * import { gmm } from 'mdml-engine/methods/mixture'
 * gmm.with({ nComponents: 2 }).fit({ x: [[0, 0], [0, 1], [9, 9], [9, 8]] }).params.means.length // => 2
 * ```
 */
export const gmm: IterativeEstimator<GmmHyper, GaussianEmState, GmmParams, GmmVerbs, GmmPath> = iterative<
  GmmHyper,
  GaussianEmState,
  GmmParams,
  GmmVerbs,
  GmmPath
>({
  name: 'mixture/gmm',
  kind: 'clusterer',
  defaults: { nComponents: 3, covarianceType: 'full', init: 'kmeans', nInit: 1, maxIter: 100, tol: 1e-3 },
  hyperSpace: (h) =>
    dict({
      nComponents: int(1, 8, { default: h.nComponents, label: 'components' }),
      covarianceType: choice(['full', 'tied', 'diag', 'spherical'] as const, {
        default: h.covarianceType,
        label: 'covariance type',
      }),
      init: choice(['kmeans', 'random'] as const, { default: h.init, label: 'start from' }),
    }),
  algorithm: (data, h, key) =>
    gaussianEmAlgorithm(data.x, initialMixture(data.x, h, key), { covarianceType: h.covarianceType, tol: h.tol }),
  maxSteps: (h) => h.maxIter,
  restarts: { count: (h) => h.nInit, better: (a, b) => a.logLikelihood > b.logLikelihood },
  params: (s) => ({ weights: s.weights, means: s.means, covariances: s.covariances, logLikelihood: s.logLikelihood }),
  verbs: (p) => ({
    predict: (x) => tensor(componentLabels(mixturePosterior(x, p).resp)),
    predictProba: (x) => {
      const { resp } = mixturePosterior(x, p)
      return fromStorage(Float64Array.from(resp.flat()), [resp.length, p.weights.length], 'float64')
    },
  }),
  contract: (data): Contract<GaussianEmState, GmmPath> =>
    defineContract({
      'model/centres': derived<GaussianEmState>({
        description: 'the component means at this step',
        compute: (s) => s.means,
      }),
      'model/density': derived<GaussianEmState>({
        description: 'the mixture’s log density over the plane of the first two features, with the training rows',
        cost: 'dear',
        compute: (s, args) => {
          const res = (args as { readonly resolution?: number } | undefined)?.resolution
          const names = (data as { readonly featureNames?: readonly string[] }).featureNames
          const fitted = {
            estimator: 'mixture/gmm',
            kind: 'density' as const,
            params: {},
            predict: (x: TensorLike): Tensor => tensor(mixturePosterior(x, s).logNorm),
          }
          return decisionSurfaceOf(fitted, data.x, {
            ...(res === undefined ? {} : { resolution: res }),
            ...(names === undefined ? {} : { featureNames: names }),
            field: { name: 'log density', smooth: true },
          })
        },
      }),
    }),
  logs: {
    'solver/logLikelihood': Log(),
    'solver/responsibilities': Log({ detail: 'full' }),
    'solver/means': Log({ detail: 'full' }),
    'solver/weights': Log({ detail: 'full' }),
    'solver/covariances': Log({ detail: 'full' }),
  },
  projections: {
    mixtureSteps: (run) => {
      const resp = tensorSeries(run, 'solver/responsibilities')
      const means = tensorSeries(run, 'solver/means')
      const weights = tensorSeries(run, 'solver/weights')
      const covs = tensorSeries(run, 'solver/covariances')
      const objective = metricSeries(run, 'solver/logLikelihood').values
      const t = Math.min(means.length, weights.length, objective.length)
      const matrix = (v: readonly number[], rows: number, cols: number): number[][] =>
        Array.from({ length: rows }, (_, i) => v.slice(i * cols, (i + 1) * cols))
      return mixtureSteps({
        family: 'gaussian',
        weights: weights.slice(0, t).map((e) => [...(e.values ?? [])]),
        means: means.slice(0, t).map((e) => matrix(e.values ?? [], e.shape[0] ?? 0, e.shape[1] ?? 1)),
        // covariances[t][k]: component k's d by d covariance, flattened row-major, as the record expects.
        ...(covs.length >= t
          ? {
              covariances: covs
                .slice(0, t)
                .map((e) => matrix(e.values ?? [], e.shape[0] ?? 0, (e.shape[1] ?? 1) * (e.shape[2] ?? 1))),
            }
          : {}),
        responsibilities: resp.slice(0, t).map((e) => matrix(e.values ?? [], e.shape[0] ?? 0, e.shape[1] ?? 1)),
        objective: objective.slice(0, t),
      })
    },
  },
})
