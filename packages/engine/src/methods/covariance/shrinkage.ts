import { chi2 } from '../../lib/dists/index.ts'
import { type AnomalyScores, anomalyScores } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  type LearnData,
  matrixView,
} from '../../lib/learn/index.ts'
import { linalg } from '../../lib/numerics/index.ts'
import { bool, choice, dict, real } from '../../lib/space/index.ts'
import { ledoitWolf, oas, shrunkCovariance, empiricalCovariance as stats } from '../../lib/stats/index.ts'
import { fromStorage, item, type Tensor, type TensorLike, toArray } from '../../lib/tensor/index.ts'

/** A fitted covariance: location, covariance and precision. */
export type CovarianceParams = {
  readonly location: number[]
  readonly covariance: number[][]
  readonly precision: number[][]
  /** The shrinkage intensity, for shrinkage estimators. */
  readonly shrinkage: number
}

/** The verbs of a fitted covariance: squared Mahalanobis distances and the Gaussian log likelihood. */
export interface CovarianceVerbs {
  /** Squared Mahalanobis distances `[m]` (sklearn's `mahalanobis`). */
  readonly mahalanobis: (x: TensorLike) => Tensor
  /** The mean Gaussian log likelihood of data under the fitted location and covariance. */
  readonly score: (data: LearnData) => number
  readonly nFeaturesIn: number
}

/** A fitted covariance estimator. */
export type FittedCovariance = ExplainedFitted<CovarianceParams, CovarianceVerbs>

/**
 * Squared Mahalanobis distances of rows to a location under a precision matrix.
 *
 * @param x - Points `[n, d]`.
 * @param location - The location.
 * @param precision - The precision `[d][d]`.
 * @returns The squared distances.
 */
export function squaredMahalanobis(
  x: TensorLike,
  location: readonly number[],
  precision: readonly (readonly number[])[],
): number[] {
  const d = location.length
  const m = matrixView(x, 'mahalanobis', { nFeatures: d })
  return Array.from({ length: m.n }, (_, i) => {
    let s = 0
    for (let a = 0; a < d; a++) {
      const da = (m.data[i * d + a] ?? 0) - (location[a] ?? 0)
      for (let b = 0; b < d; b++) s += da * (precision[a]?.[b] ?? 0) * ((m.data[i * d + b] ?? 0) - (location[b] ?? 0))
    }
    return s
  })
}

/**
 * The verbs of a covariance model.
 *
 * @param p - The parameters.
 * @returns Mahalanobis distances and the log-likelihood score.
 */
export function covarianceVerbs(p: CovarianceParams): CovarianceVerbs {
  const d = p.location.length
  const logdet = item(linalg.slogdet(p.covariance).logabsdet)
  return {
    mahalanobis: (x) => {
      const v = squaredMahalanobis(x, p.location, p.precision)
      return fromStorage(Float64Array.from(v), [v.length], 'float64')
    },
    score: (data) => {
      const v = squaredMahalanobis(data.x, p.location, p.precision)
      return v.reduce((a, q) => a - 0.5 * (q + d * Math.log(2 * Math.PI) + logdet), 0) / v.length
    },
    nFeaturesIn: d,
  }
}

/**
 * The exposition of a covariance fit: squared Mahalanobis distances as anomaly scores, flagged beyond the 97.5%
 * quantile of chi-squared with d degrees of freedom.
 *
 * @param method - The method name.
 * @param x - The training points.
 * @param p - The parameters.
 * @returns The record.
 */
export function mahalanobisRecord(method: string, x: TensorLike, p: CovarianceParams): AnomalyScores {
  const scores = squaredMahalanobis(x, p.location, p.precision)
  return anomalyScores({
    method,
    scores,
    threshold: item(chi2(p.location.length).ppf(0.975)),
    evidence: { name: 'squaredMahalanobis', values: scores },
  })
}

/** Hyperparameters of the plain and shrunk covariance estimators. */
export type CovarianceHyper = {
  /** The estimator (default empirical). */
  readonly method?: 'empirical' | 'shrunk' | 'ledoit-wolf' | 'oas'
  /** The fixed shrinkage of 'shrunk' (default 0.1). */
  readonly shrinkage?: number
  readonly assumeCentered?: boolean
}

/**
 * Covariance estimation (sklearn.covariance's EmpiricalCovariance, ShrunkCovariance, LedoitWolf and OAS) over the
 * estimators of `stats/covariance.ts`. Its exposition is an `anomalyScores` record of squared Mahalanobis distances.
 *
 * @param hyper - The estimator, its shrinkage and whether the data are centred.
 * @returns The estimator.
 */
export function covarianceEstimator(hyper: CovarianceHyper = {}): Estimator<CovarianceHyper, FittedCovariance> {
  return defineExplainedEstimator<CovarianceHyper, CovarianceParams, CovarianceVerbs>(
    {
      name: 'covariance',
      kind: 'density',
      hyperSpace: (h) =>
        dict({
          method: choice(['empirical', 'shrunk', 'ledoit-wolf', 'oas'], { default: h.method ?? 'empirical' }),
          shrinkage: real(0, 1, { default: h.shrinkage ?? 0.1 }),
          assumeCentered: bool({ default: h.assumeCentered ?? false }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'covariance', { minSamples: 2 })
        const options = { assumeCentered: h.assumeCentered ?? false }
        const method = h.method ?? 'empirical'
        let covariance: number[][]
        let shrinkage = 0
        if (method === 'empirical') covariance = toArray(stats(data.x, options)) as number[][]
        else {
          const r =
            method === 'shrunk'
              ? shrunkCovariance(data.x, { ...options, shrinkage: h.shrinkage ?? 0.1 })
              : method === 'ledoit-wolf'
                ? ledoitWolf(data.x, options)
                : oas(data.x, options)
          covariance = toArray(r.covariance) as number[][]
          shrinkage = r.shrinkage
        }
        const location = options.assumeCentered
          ? new Array<number>(m.d).fill(0)
          : Array.from({ length: m.d }, (_, j) => {
              let s = 0
              for (let i = 0; i < m.n; i++) s += m.data[i * m.d + j] ?? 0
              return s / m.n
            })
        const params: CovarianceParams = {
          location,
          covariance,
          precision: toArray(linalg.pinv(covariance)) as number[][],
          shrinkage,
        }
        return { params, exposition: [mahalanobisRecord(method, data.x, params)] }
      },
      methods: covarianceVerbs,
    },
    hyper,
  )
}

/** A covariance estimator as a class. */
export class CovarianceEstimator extends EstimatorClass<CovarianceHyper, FittedCovariance> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link covarianceEstimator}.
   */
  constructor(hyper: CovarianceHyper = {}) {
    super(covarianceEstimator(hyper))
  }
}
