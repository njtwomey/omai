/**
 * Covariance estimators, as in scikit-learn's `covariance` module: the empirical covariance and its shrunk forms (fixed
 * shrinkage, Ledoit-Wolf, OAS), each fitted with its precision matrix, Mahalanobis distances and Gaussian log
 * likelihood on the iterative base. The exposition shows squared Mahalanobis distances as anomaly scores against a
 * chi-squared cut-off.
 */

import { chi2 } from '../dists/index.ts'
import { type AnomalyScores, anomalyScores } from '../exposition/index.ts'
import {
  ExplainedEstimatorClass,
  type Fitted,
  type IterativeEstimator,
  type IterativeFitted,
  iterative,
  type LearnData,
  matrixView,
} from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { bool, choice, dict, real } from '../space/index.ts'
import {
  type CovarianceOptions,
  empiricalCovariance,
  ledoitWolf,
  oas,
  type ShrunkCovariance,
  shrunkCovariance,
} from '../stats/index.ts'
import { fromStorage, item, type Tensor, type TensorLike, toArray } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'

export { type CovarianceOptions, empiricalCovariance, ledoitWolf, oas, type ShrunkCovariance, shrunkCovariance }

/** Which covariance estimator to fit. */
export type CovarianceMethod = 'empirical' | 'shrunk' | 'ledoit-wolf' | 'oas'

/** A fitted covariance: location, covariance, precision and shrinkage intensity. */
export type CovarianceParams = {
  readonly location: number[]
  readonly covariance: number[][]
  readonly precision: number[][]
  /** The shrinkage intensity, for shrinkage estimators (0 for empirical). */
  readonly shrinkage: number
}

/** The verbs of a fitted covariance: squared Mahalanobis distances and the mean Gaussian log likelihood. */
export interface CovarianceVerbs {
  /** Squared Mahalanobis distances `[m]` (sklearn's `mahalanobis`). */
  readonly mahalanobis: (x: TensorLike) => Tensor
  /** The mean Gaussian log likelihood of data under the fitted location and covariance. */
  readonly score: (data: LearnData) => number
  readonly nFeaturesIn: number
}

/** The state of a covariance fit on the iterative base. */
export interface CovarianceState {
  readonly location: number[]
  readonly covariance: number[][]
  readonly precision: number[][]
  readonly shrinkage: number
  readonly logdet: number
  readonly done: boolean
}

/** A fitted covariance estimator. */
export type FittedCovariance = IterativeFitted<CovarianceParams, CovarianceVerbs>

/**
 * Squared Mahalanobis distances `(x - mu)^T Precision (x - mu)` of rows to a location under a precision matrix.
 *
 * @param x - Points `[n, d]`.
 * @param location - The location `[d]`.
 * @param precision - The precision matrix `[d][d]`.
 * @returns The squared Mahalanobis distances, one per row.
 * @example
 * ```ts
 * import { squaredMahalanobis } from 'mdml-engine/covariance'
 * squaredMahalanobis([[0, 0], [2, 0]], [0, 0], [[1, 0], [0, 1]]) // => [0, 4]
 * ```
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
    return Math.max(0, s)
  })
}

/**
 * Mahalanobis distances of rows to a location under a precision matrix, as a 1-D tensor (unsquared by default, or
 * squared when `options.squared` is true).
 *
 * @param x - Points `[n, d]`.
 * @param location - The location `[d]`.
 * @param precision - The precision matrix `[d][d]`.
 * @param options - Whether to return squared distances (default false).
 * @returns A 1-D tensor `[n]` of distances.
 * @example
 * ```ts
 * import { mahalanobisDistances } from 'mdml-engine/covariance'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(mahalanobisDistances([[0, 0], [3, 4]], [0, 0], [[1, 0], [0, 1]])) // => [0, 5]
 * ```
 */
export function mahalanobisDistances(
  x: TensorLike,
  location: readonly number[],
  precision: readonly (readonly number[])[],
  options: { readonly squared?: boolean } = {},
): Tensor {
  const sq = squaredMahalanobis(x, location, precision)
  const out = options.squared === true ? Float64Array.from(sq) : Float64Array.from(sq, (v) => Math.sqrt(Math.max(0, v)))
  return fromStorage(out, [out.length], 'float64')
}

/**
 * The verbs of a fitted covariance model: squared Mahalanobis distances and mean Gaussian log likelihood.
 *
 * @param p - The fitted covariance parameters.
 * @returns Mahalanobis distances and the log-likelihood score.
 * @example
 * ```ts
 * import { covarianceVerbs } from 'mdml-engine/covariance'
 * import { toFlat } from 'mdml-engine/tensor'
 * const v = covarianceVerbs({ location: [0, 0], covariance: [[1, 0], [0, 1]], precision: [[1, 0], [0, 1]], shrinkage: 0 })
 * toFlat(v.mahalanobis([[3, 4]])) // => [25]
 * ```
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
 * quantile of chi-squared with `d` degrees of freedom.
 *
 * @param method - The method name.
 * @param x - The training points `[n, d]`.
 * @param p - The fitted covariance parameters.
 * @returns The `anomalyScores` record.
 * @example
 * ```ts
 * import { mahalanobisRecord } from 'mdml-engine/covariance'
 * const p = { location: [0, 0], covariance: [[1, 0], [0, 1]], precision: [[1, 0], [0, 1]], shrinkage: 0 }
 * mahalanobisRecord('empirical', [[0, 0], [2, 0]], p).scores // => [0, 4]
 * ```
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

/** Hyperparameters of the empirical and shrunk covariance estimators. */
export interface CovarianceHyper {
  /** The estimator (default `'empirical'`). */
  readonly method?: CovarianceMethod
  /** The fixed shrinkage intensity of `'shrunk'` in `[0, 1]` (default 0.1). */
  readonly shrinkage?: number
  /** Whether the data are already centred (default false). */
  readonly assumeCentered?: boolean
}

function computeCovarianceState(x: TensorLike, h: Required<CovarianceHyper>, done: boolean): CovarianceState {
  const m = matrixView(x, 'covariance', { minSamples: 2 })
  const options = { assumeCentered: h.assumeCentered }
  let covariance: number[][]
  let shrinkage = 0
  if (h.method === 'empirical') {
    covariance = toArray(empiricalCovariance(x, options)) as number[][]
  } else {
    const r =
      h.method === 'shrunk'
        ? shrunkCovariance(x, { ...options, shrinkage: h.shrinkage })
        : h.method === 'ledoit-wolf'
          ? ledoitWolf(x, options)
          : oas(x, options)
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
  const precision = toArray(linalg.pinv(covariance)) as number[][]
  const logdet = item(linalg.slogdet(covariance).logabsdet)
  return { location, covariance, precision, shrinkage, logdet, done }
}

/**
 * Empirical and shrinkage covariance estimation as a stepped {@link Algorithm}: computes the location, covariance,
 * precision and shrinkage intensity for `'empirical'`, `'shrunk'`, `'ledoit-wolf'` or `'oas'`, observing the shrinkage
 * intensity and log-determinant.
 *
 * @param x - Observations `[n, d]`.
 * @param options - Estimator method, fixed shrinkage intensity and centring option.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { shrinkageAlgorithm } from 'mdml-engine/covariance'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(shrinkageAlgorithm([[0, 0], [2, 2]], { method: 'shrunk', shrinkage: 0.5 }), { maxSteps: 2 }).state.covariance // => [[1, 0.5], [0.5, 1]]
 * ```
 */
export function shrinkageAlgorithm(x: TensorLike, options: CovarianceHyper = {}): Algorithm<CovarianceState> {
  const h: Required<CovarianceHyper> = {
    method: options.method ?? 'empirical',
    shrinkage: options.shrinkage ?? 0.1,
    assumeCentered: options.assumeCentered ?? false,
  }
  const m = matrixView(x, 'shrinkageAlgorithm', { minSamples: 2 })
  return {
    name: `covariance-${h.method}`,
    config: { method: h.method, shrinkage: h.shrinkage, assumeCentered: h.assumeCentered },
    schema: channels({
      metrics: {
        shrinkage: { description: 'the shrinkage intensity pulling the covariance towards a scaled identity' },
        logdet: { description: 'log-determinant of the estimated covariance matrix' },
      },
      tensors: {
        covariance: { role: 'param', description: 'the estimated covariance matrix', shape: ['d', 'd'] },
        precision: { role: 'param', description: 'the pseudo-inverse precision matrix', shape: ['d', 'd'] },
      },
    }),
    init: () => computeCovarianceState(x, h, false),
    step: (s) => ({ ...s, done: true }),
    done: (s) => (s.done ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        shrinkage: s.shrinkage,
        ...(Number.isFinite(s.logdet) ? { logdet: s.logdet } : {}),
      },
      tensors: {
        covariance: { role: 'param', value: s.covariance.flat(), shape: [m.d, m.d] },
        precision: { role: 'param', value: s.precision.flat(), shape: [m.d, m.d] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CovarianceState,
  }
}

/**
 * Covariance estimation (sklearn.covariance's `EmpiricalCovariance`, `ShrunkCovariance`, `LedoitWolf` and `OAS`) on the
 * iterative base over the estimators of `stats/covariance.ts`. Its exposition is an `anomalyScores` record of squared
 * Mahalanobis distances.
 *
 * @param hyper - The estimator method, its shrinkage and whether the data are centred.
 * @returns The iterative estimator.
 * @index estimator Empirical, Ledoit-Wolf and OAS shrunk covariance.
 * @example
 * ```ts
 * import { covarianceEstimator } from 'mdml-engine/covariance'
 * covarianceEstimator({ method: 'shrunk', shrinkage: 0.5 }).fit({ x: [[0, 0], [2, 2]] }).params.covariance // => [[1, 0.5], [0.5, 1]]
 * ```
 */
export function covarianceEstimator(
  hyper: CovarianceHyper = {},
): IterativeEstimator<Required<CovarianceHyper>, CovarianceState, CovarianceParams, CovarianceVerbs> {
  const defaults: Required<CovarianceHyper> = {
    method: hyper.method ?? 'empirical',
    shrinkage: hyper.shrinkage ?? 0.1,
    assumeCentered: hyper.assumeCentered ?? false,
  }
  return iterative<Required<CovarianceHyper>, CovarianceState, CovarianceParams, CovarianceVerbs>({
    name: 'covariance',
    kind: 'density',
    defaults,
    hyperSpace: (h) =>
      dict({
        method: choice(['empirical', 'shrunk', 'ledoit-wolf', 'oas'] as const, {
          label: 'estimator',
          description: 'The empirical covariance, fixed shrinkage, Ledoit-Wolf or OAS.',
          default: h.method,
        }),
        shrinkage: real(0, 1, {
          label: 'shrinkage',
          description: 'How far the covariance is pulled towards a scaled identity (0 none, 1 all the way).',
          default: h.shrinkage,
        }),
        assumeCentered: bool({
          label: 'assume centred',
          description: 'Treat the data as already centred (do not subtract the mean).',
          default: h.assumeCentered,
        }),
      }),
    algorithm: (data, h) => shrinkageAlgorithm(data.x, h),
    maxSteps: () => 1,
    params: (s) => ({
      location: s.location,
      covariance: s.covariance,
      precision: s.precision,
      shrinkage: s.shrinkage,
    }),
    verbs: covarianceVerbs,
    records: (s, data, h) => [mahalanobisRecord(h.method, data.x, s)],
    logs: { 'solver/shrinkage': Log() },
  })
}

/**
 * A covariance estimator as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { CovarianceEstimator } from 'mdml-engine/covariance'
 * new CovarianceEstimator({ method: 'empirical' }).fit([[0, 0], [2, 2]]).model.params.location // => [1, 1]
 * ```
 */
export class CovarianceEstimator extends ExplainedEstimatorClass<
  Required<CovarianceHyper>,
  Fitted<CovarianceParams> & CovarianceVerbs
> {
  /**
   * Make an unfitted covariance model.
   *
   * @param hyper - The hyperparameters of {@link covarianceEstimator}.
   */
  constructor(hyper: CovarianceHyper = {}) {
    super(covarianceEstimator(hyper))
  }
}
