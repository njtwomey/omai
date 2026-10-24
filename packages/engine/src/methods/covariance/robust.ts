import { WhiteboxError } from '../../lib/base/index.ts'
import { chi2 } from '../../lib/dists/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  fitRunOptions,
  type MatrixView,
  matrixView,
} from '../../lib/learn/index.ts'
import { linalg } from '../../lib/numerics/index.ts'
import { foldIn, type Key, keys, permutation } from '../../lib/random/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import { item, toArray } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import {
  type CovarianceParams,
  type CovarianceVerbs,
  covarianceVerbs,
  mahalanobisRecord,
  squaredMahalanobis,
} from './shrinkage.ts'

/** The state of the C-step iteration: a support of h rows, its mean and covariance, and its log determinant. */
export interface CStepState {
  readonly support: number[]
  readonly location: number[]
  readonly covariance: number[][]
  readonly logdet: number
  readonly previous: number
}

/** The mean and (biased) covariance of some rows. */
function meanAndCovariance(m: MatrixView, rows: readonly number[]): { location: number[]; covariance: number[][] } {
  const d = m.d
  const location = Array.from(
    { length: d },
    (_, j) => rows.reduce((a, i) => a + (m.data[i * d + j] ?? 0), 0) / rows.length,
  )
  const covariance = Array.from({ length: d }, (_, a) =>
    Array.from(
      { length: d },
      (_, b) =>
        rows.reduce(
          (s, i) =>
            s + ((m.data[i * d + a] ?? 0) - (location[a] ?? 0)) * ((m.data[i * d + b] ?? 0) - (location[b] ?? 0)),
          0,
        ) / rows.length,
    ),
  )
  return { location, covariance }
}

const logdetOf = (c: readonly (readonly number[])[]): number => {
  const r = linalg.slogdet(c as number[][])
  return item(r.sign) > 0 ? item(r.logabsdet) : Number.NEGATIVE_INFINITY
}

/**
 * The concentration steps of FastMCD (Rousseeuw and Van Driessen 1999; sklearn's `_c_step`) as a traced algorithm:
 * each step keeps the h rows nearest (in Mahalanobis distance) to the current mean and covariance; the covariance
 * determinant never increases, and the iteration stops when it does not decrease.
 *
 * @param x - Points `[n, d]`.
 * @param support - The initial support (h row indices).
 * @returns The algorithm.
 */
export function cStepAlgorithm(x: MatrixView, support: readonly number[]): Algorithm<CStepState> {
  const h = support.length
  return {
    name: 'mcd-c-step',
    config: { h },
    init: () => {
      const rows = [...support]
      const { location, covariance } = meanAndCovariance(x, rows)
      return { support: rows, location, covariance, logdet: logdetOf(covariance), previous: Number.POSITIVE_INFINITY }
    },
    step: (s) => {
      const precision = toArray(linalg.pinv(s.covariance)) as number[][]
      const flat = Array.from({ length: x.n }, (_, i) => Array.from(x.data.subarray(i * x.d, (i + 1) * x.d)))
      const dist = squaredMahalanobis(flat, s.location, precision)
      const rows = dist
        .map((v, i) => [v, i] as const)
        .sort((a, b) => a[0] - b[0] || a[1] - b[1])
        .slice(0, h)
        .map(([, i]) => i)
      const { location, covariance } = meanAndCovariance(x, rows)
      return { support: rows, location, covariance, logdet: logdetOf(covariance), previous: s.logdet }
    },
    done: (s) =>
      !Number.isFinite(s.logdet) || !(s.logdet < s.previous) || Math.abs(s.logdet - s.previous) < 1e-12
        ? 'converged'
        : undefined,
    observe: (s) => ({ metrics: Number.isFinite(s.logdet) ? { logdet: s.logdet } : {} }),
  }
}

/** Hyperparameters of the minimum covariance determinant. */
export type MinCovDetHyper = {
  /** The fraction of rows in the support (default (n + d + 1) / (2n)). */
  readonly supportFraction?: number
  /** Random starting supports (default 30, as sklearn for small n). */
  readonly nTrials?: number
}

/** The parameters of a fitted MCD: the reweighted estimate, the raw one, and the support. */
export type MinCovDetParams = CovarianceParams & {
  readonly rawLocation: number[]
  readonly rawCovariance: number[][]
  readonly support: boolean[]
}

/** A fitted minimum covariance determinant. */
export type FittedMinCovDet = ExplainedFitted<MinCovDetParams, CovarianceVerbs>

/**
 * The minimum covariance determinant (sklearn.covariance.MinCovDet) by FastMCD: random h-subsets each refined by two
 * C-steps, the best ten refined to convergence (traced), then the consistency correction and reweighting at the
 * 97.5% chi-squared quantile. Its exposition is an `anomalyScores` record of robust squared Mahalanobis distances.
 *
 * @param hyper - Support fraction and the number of random starts.
 * @returns The estimator.
 */
export function minCovDet(hyper: MinCovDetHyper = {}): Estimator<MinCovDetHyper, FittedMinCovDet> {
  return defineExplainedEstimator<MinCovDetHyper, MinCovDetParams, CovarianceVerbs>(
    {
      name: 'min-cov-det',
      kind: 'density',
      hyperSpace: (h) =>
        dict({
          supportFraction: real(0.5, 1, { default: h.supportFraction ?? 0.75, label: 'support fraction' }),
          nTrials: int(1, 500, { default: h.nTrials ?? 30, label: 'random starts' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'minCovDet', { minSamples: 3 })
        const hSize =
          h.supportFraction === undefined ? Math.ceil(0.5 * (m.n + m.d + 1)) : Math.ceil(h.supportFraction * m.n)
        if (hSize <= m.d) throw new WhiteboxError('value', 'minCovDet: the support must have more rows than features')
        const run = fitRunOptions(options, 'min-cov-det')
        const key = run.key as Key
        const refine = (start: readonly number[], steps: number) =>
          runAlgorithm(cStepAlgorithm(m, start), { maxSteps: steps }).state
        const trials = keys(foldIn(key, 0), h.nTrials ?? 30).map((k) =>
          refine(Array.from(permutation(k, m.n).data).slice(0, hSize), 2),
        )
        const best = trials.sort((a, b) => a.logdet - b.logdet).slice(0, 10)
        const finals = best.map((s) => runAlgorithm(cStepAlgorithm(m, s.support), { ...run, maxSteps: 30 }).state)
        const raw = finals.sort((a, b) => a.logdet - b.logdet)[0] as CStepState
        const flat = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const rawPrecision = toArray(linalg.pinv(raw.covariance)) as number[][]
        const rawDist = squaredMahalanobis(flat, raw.location, rawPrecision)
        const sorted = [...rawDist].sort((a, b) => a - b)
        const median =
          m.n % 2 === 1 ? (sorted[(m.n - 1) / 2] ?? 0) : ((sorted[m.n / 2 - 1] ?? 0) + (sorted[m.n / 2] ?? 0)) / 2
        const correction = median / item(chi2(m.d).ppf(0.5))
        const corrected = rawDist.map((v) => v / correction)
        const cut = item(chi2(m.d).ppf(0.975))
        const support = corrected.map((v) => v < cut)
        const { location, covariance } = meanAndCovariance(
          m,
          support.flatMap((s, i) => (s ? [i] : [])),
        )
        const params: MinCovDetParams = {
          location,
          covariance,
          precision: toArray(linalg.pinv(covariance)) as number[][],
          shrinkage: 0,
          rawLocation: raw.location,
          rawCovariance: raw.covariance.map((r) => r.map((v) => v * correction)),
          support,
        }
        return { params, exposition: [mahalanobisRecord('minCovDet', data.x, params)] }
      },
      methods: covarianceVerbs,
    },
    hyper,
  )
}

/** The minimum covariance determinant as a class. */
export class MinCovDet extends EstimatorClass<MinCovDetHyper, FittedMinCovDet> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link minCovDet}.
   */
  constructor(hyper: MinCovDetHyper = {}) {
    super(minCovDet(hyper))
  }
}
