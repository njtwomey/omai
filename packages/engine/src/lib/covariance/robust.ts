/**
 * The minimum covariance determinant by FastMCD (Rousseeuw and Van Driessen, 1999; as in scikit-learn's `MinCovDet`)
 * and the elliptic envelope outlier detector (`EllipticEnvelope`): a robust location and covariance from the subset of
 * rows with the smallest covariance determinant, found by concentration steps from random subsets as a traced
 * `Algorithm`, then corrected for consistency and reweighted.
 */

import { contaminationThreshold, type OutlierVerbs, outlierExposition, outlierVerbs } from '../anomaly/index.ts'
import { MdmlError } from '../base/index.ts'
import { chi2 } from '../dists/index.ts'
import {
  ExplainedEstimatorClass,
  type Fitted,
  type IterativeEstimator,
  type IterativeFitted,
  iterative,
  type MatrixView,
  matrixView,
} from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { foldIn, type Key, keys, key as makeKey, permutation } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, item, type Tensor, type TensorLike, toArray } from '../tensor/index.ts'
import { type Algorithm, channels, Log, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  type CovarianceMethod,
  type CovarianceParams,
  type CovarianceVerbs,
  covarianceVerbs,
  mahalanobisRecord,
  shrinkageAlgorithm,
  squaredMahalanobis,
} from './shrinkage.ts'

/** The state of the C-step iteration: a support of `h` rows, its mean and covariance, and its log determinant. */
export interface CStepState {
  readonly support: number[]
  readonly location: number[]
  readonly covariance: number[][]
  readonly logdet: number
  readonly previous: number
}

function isMatrixView(x: TensorLike | MatrixView): x is MatrixView {
  if (typeof x !== 'object' || x === null) return false
  const v = x as Partial<MatrixView>
  return v.data instanceof Float64Array && typeof v.n === 'number' && typeof v.d === 'number'
}

function toView(x: TensorLike | MatrixView, who: string): MatrixView {
  return isMatrixView(x) ? x : matrixView(x, who, { minSamples: 3 })
}

/** The mean and (biased) covariance of a subset of rows. */
function meanAndCovariance(m: MatrixView, rows: readonly number[]): { location: number[]; covariance: number[][] } {
  const d = m.d
  const count = Math.max(rows.length, 1)
  const location = Array.from({ length: d }, (_, j) => rows.reduce((a, i) => a + (m.data[i * d + j] ?? 0), 0) / count)
  const covariance = Array.from({ length: d }, (_, a) =>
    Array.from(
      { length: d },
      (_, b) =>
        rows.reduce(
          (s, i) =>
            s + ((m.data[i * d + a] ?? 0) - (location[a] ?? 0)) * ((m.data[i * d + b] ?? 0) - (location[b] ?? 0)),
          0,
        ) / count,
    ),
  )
  return { location, covariance }
}

const logdetOf = (c: readonly (readonly number[])[]): number => {
  const r = linalg.slogdet(c as number[][])
  return item(r.sign) > 0 ? item(r.logabsdet) : Number.NEGATIVE_INFINITY
}

function oneCStep(m: MatrixView, s: CStepState, h: number): CStepState {
  const precision = toArray(linalg.pinv(s.covariance)) as number[][]
  const flat = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
  const dist = squaredMahalanobis(flat, s.location, precision)
  const rows = dist
    .map((v, i) => [v, i] as const)
    .sort((a, b) => a[0] - b[0] || a[1] - b[1])
    .slice(0, h)
    .map(([, i]) => i)
  const { location, covariance } = meanAndCovariance(m, rows)
  return { support: rows, location, covariance, logdet: logdetOf(covariance), previous: s.logdet }
}

/**
 * The concentration steps of FastMCD (Rousseeuw and Van Driessen 1999; sklearn's `_c_step`) as a traced
 * {@link Algorithm}: each step keeps the `h` rows nearest (in Mahalanobis distance) to the current mean and
 * covariance; the covariance determinant never increases, and the iteration stops when it does not decrease.
 *
 * @param x - Points `[n, d]` (or a `MatrixView`).
 * @param support - The initial support (`h` row indices).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { cStepAlgorithm } from 'mdml-engine/covariance'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0, 0], [1, 0], [0, 1], [1, 1], [10, 10]]
 * runAlgorithm(cStepAlgorithm(x, [0, 1, 2, 4]), { maxSteps: 10 }).state.support // => [0, 1, 2, 3]
 * ```
 */
export function cStepAlgorithm(x: TensorLike | MatrixView, support: readonly number[]): Algorithm<CStepState> {
  const m = toView(x, 'cStepAlgorithm')
  const h = support.length
  return {
    name: 'mcd-c-step',
    config: { h },
    schema: channels({
      metrics: {
        logdet: { description: 'log-determinant of the current h-subset covariance', direction: 'lower' },
        determinant: { description: 'determinant of the current h-subset covariance', direction: 'lower' },
      },
      tensors: {
        subset: { role: 'state', description: 'row indices in the current h-subset', shape: ['h'] },
      },
    }),
    init: () => {
      const rows = [...support]
      const { location, covariance } = meanAndCovariance(m, rows)
      return { support: rows, location, covariance, logdet: logdetOf(covariance), previous: Number.POSITIVE_INFINITY }
    },
    step: (s) => oneCStep(m, s, h),
    done: (s) =>
      !Number.isFinite(s.logdet) || !(s.logdet < s.previous) || Math.abs(s.logdet - s.previous) < 1e-12
        ? 'converged'
        : undefined,
    observe: (s) => ({
      metrics: Number.isFinite(s.logdet) ? { logdet: s.logdet, determinant: Math.exp(s.logdet) } : {},
      tensors: { subset: { role: 'state', value: s.support, shape: [h] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CStepState,
  }
}

/** The state of FastMCD: the active `h`-subset, the consistency-corrected raw and reweighted estimates, and `logdet`. */
export interface FastMcdState {
  readonly subset: number[]
  readonly rawLocation: number[]
  readonly rawCovariance: number[][]
  readonly location: number[]
  readonly covariance: number[][]
  readonly precision: number[][]
  readonly support: boolean[]
  readonly logdet: number
  readonly determinant: number
  readonly previous: number
}

/** Options of {@link fastMcdAlgorithm}. */
export interface FastMcdOptions {
  /** The fraction of rows in the concentration support (default `(n + d + 1) / (2n)`). */
  readonly supportFraction?: number
  /** Random starting `h`-subsets screened in `init` (default 30). */
  readonly nTrials?: number
}

/**
 * The consistency factor of Croux and Haesbroeck (1999, Eq. 4.2; Pison et al. 2002), `alpha / F_{chi2(d + 2)}(q_alpha)`
 * with `q_alpha` the `alpha` quantile of `chi2(d)`: the factor by which the covariance of the central `alpha` fraction
 * of a normal sample is too small.
 */
function consistencyFactor(d: number, alpha: number): number {
  const q = item(chi2(d).ppf(alpha))
  return alpha / item(chi2(d + 2).cdf(q))
}

/** The raw FastMCD estimate, then scikit-learn's consistency correction of the distances and the reweighting. */
function reweightMcd(m: MatrixView, raw: CStepState): FastMcdState {
  const flat = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
  const rawPrecision = toArray(linalg.pinv(raw.covariance)) as number[][]
  // The distances under the raw estimate, divided by the factor for the support's fraction of the rows (scikit-learn's
  // `correct_covariance`), are cut at the 97.5% chi-squared quantile (`reweight_covariance`).
  const rawFactor = consistencyFactor(m.d, raw.support.length / m.n)
  const corrected = squaredMahalanobis(flat, raw.location, rawPrecision).map((v) => v / rawFactor)
  const support = corrected.map((v) => v < item(chi2(m.d).ppf(0.975)))
  const inliers = support.flatMap((s, i) => (s ? [i] : []))
  const chosen = inliers.length > m.d ? inliers : raw.support
  const { location, covariance: biased } = meanAndCovariance(m, chosen)
  const factor = consistencyFactor(m.d, 0.975)
  const covariance = biased.map((r) => r.map((v) => v * factor))
  const precision = toArray(linalg.pinv(covariance)) as number[][]
  return {
    subset: raw.support,
    rawLocation: raw.location,
    rawCovariance: raw.covariance,
    location,
    covariance,
    precision,
    support,
    logdet: raw.logdet,
    determinant: Number.isFinite(raw.logdet) ? Math.exp(raw.logdet) : 0,
    previous: raw.previous,
  }
}

/**
 * FastMCD (Rousseeuw and Van Driessen 1999; sklearn.covariance.MinCovDet) as a stepped {@link Algorithm}: `init`
 * screens `nTrials` random `h`-subsets by two C-steps to select the best starting support, and each `step` performs one
 * concentration step (`solver/determinant` falling, `solver/subset` updating) with consistency correction and
 * chi-squared reweighting.
 *
 * @param x - Points `[n, d]`.
 * @param options - Support fraction and number of random trials.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { fastMcdAlgorithm } from 'mdml-engine/covariance'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [-0.1, 0], [0, -0.1], [10, 10]]
 * runAlgorithm(fastMcdAlgorithm(x, { nTrials: 5 }), { maxSteps: 10 }).state.support[6] // => false
 * ```
 */
export function fastMcdAlgorithm(x: TensorLike, options: FastMcdOptions = {}): Algorithm<FastMcdState> {
  const m = matrixView(x, 'minCovDet', { minSamples: 3 })
  // scikit-learn: ceil((n + d + 1) / 2) by default, otherwise int(support_fraction * n).
  const hSize =
    options.supportFraction === undefined
      ? Math.ceil(0.5 * (m.n + m.d + 1))
      : Math.floor(options.supportFraction * m.n + 1e-9)
  if (hSize <= m.d || hSize > m.n)
    throw new MdmlError('value', 'minCovDet: the support must have more rows than features and at most n rows')
  const nTrials = Math.max(1, options.nTrials ?? 30)
  return {
    name: 'fast-mcd',
    config: { h: hSize, nTrials },
    schema: channels({
      metrics: {
        determinant: { description: 'determinant of the current h-subset covariance', direction: 'lower' },
        logdet: { description: 'log-determinant of the current h-subset covariance', direction: 'lower' },
      },
      tensors: {
        subset: { role: 'state', description: 'row indices in the current h-subset', shape: ['h'] },
      },
    }),
    init: (key: Key) => {
      const k0 = key ?? makeKey(0)
      const starts = keys(foldIn(k0, 0), nTrials).map((k) => Array.from(permutation(k, m.n).data).slice(0, hSize))
      const twoStep = starts.map((start, idx) => ({
        idx,
        state: runAlgorithm(cStepAlgorithm(m, start), { maxSteps: 2 }).state,
      }))
      const top = twoStep.sort((a, b) => a.state.logdet - b.state.logdet).slice(0, 10)
      const finals = top.map(({ idx, state }) => ({
        idx,
        state: runAlgorithm(cStepAlgorithm(m, state.support), { maxSteps: 30 }).state,
      }))
      const winnerIdx = finals.sort((a, b) => a.state.logdet - b.state.logdet)[0]?.idx ?? 0
      const startSupport = starts[winnerIdx] ?? starts[0] ?? []
      const { location, covariance } = meanAndCovariance(m, startSupport)
      const c0: CStepState = {
        support: startSupport,
        location,
        covariance,
        logdet: logdetOf(covariance),
        previous: Number.POSITIVE_INFINITY,
      }
      return reweightMcd(m, c0)
    },
    step: (s) => {
      const cState: CStepState = {
        support: s.subset,
        location: s.rawLocation,
        covariance: meanAndCovariance(m, s.subset).covariance,
        logdet: s.logdet,
        previous: s.previous,
      }
      const next = oneCStep(m, cState, hSize)
      return reweightMcd(m, next)
    },
    done: (s) =>
      !Number.isFinite(s.logdet) || !(s.logdet < s.previous) || Math.abs(s.logdet - s.previous) < 1e-12
        ? 'converged'
        : undefined,
    observe: (s) => ({
      metrics: Number.isFinite(s.logdet) ? { determinant: s.determinant, logdet: s.logdet } : {},
      tensors: { subset: { role: 'state', value: s.subset, shape: [hSize] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as FastMcdState,
  }
}

/** Hyperparameters of the minimum covariance determinant estimator. */
export interface MinCovDetHyper {
  /** The fraction of rows in the support (default `(n + d + 1) / (2n)`). */
  readonly supportFraction?: number
  /** Random starting supports (default 30, as sklearn for small `n`). */
  readonly nTrials?: number
}

/** The parameters of a fitted MCD: the reweighted estimate, the raw one, and the inlier support mask. */
export type MinCovDetParams = CovarianceParams & {
  readonly rawLocation: number[]
  readonly rawCovariance: number[][]
  readonly support: boolean[]
}

/** A fitted minimum covariance determinant estimator. */
export type FittedMinCovDet = IterativeFitted<MinCovDetParams, CovarianceVerbs>

/**
 * The minimum covariance determinant (sklearn.covariance.MinCovDet) by FastMCD on the iterative base: random
 * `h`-subsets screened by two C-steps, refined step by step to convergence, then consistency-corrected and reweighted
 * at the 97.5% chi-squared quantile. Its exposition is an `anomalyScores` record of robust squared Mahalanobis
 * distances.
 *
 * @param hyper - Support fraction and the number of random starts.
 * @returns The iterative estimator.
 * @example
 * ```ts
 * import { minCovDet } from 'mdml-engine/covariance'
 * const x = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [-0.1, 0], [0, -0.1], [20, 20]]
 * minCovDet({ nTrials: 5 }).fit({ x }).params.support[6] // => false
 * ```
 */
export function minCovDet(
  hyper: MinCovDetHyper = {},
): IterativeEstimator<MinCovDetHyper, FastMcdState, MinCovDetParams, CovarianceVerbs> {
  return iterative<MinCovDetHyper, FastMcdState, MinCovDetParams, CovarianceVerbs>({
    name: 'min-cov-det',
    kind: 'density',
    defaults: {
      ...(hyper.supportFraction === undefined ? {} : { supportFraction: hyper.supportFraction }),
      nTrials: hyper.nTrials ?? 30,
    },
    hyperSpace: (h) => dict({ nTrials: int(1, 500, { default: h.nTrials ?? 30, label: 'random starts' }) }),
    algorithm: (data, h) => fastMcdAlgorithm(data.x, h),
    maxSteps: () => 30,
    params: (s) => ({
      location: s.location,
      covariance: s.covariance,
      precision: s.precision,
      shrinkage: 0,
      rawLocation: s.rawLocation,
      rawCovariance: s.rawCovariance,
      support: s.support,
    }),
    verbs: covarianceVerbs,
    records: (s, data) => [
      mahalanobisRecord('minCovDet', data.x, {
        location: s.location,
        covariance: s.covariance,
        precision: s.precision,
        shrinkage: 0,
      }),
    ],
    logs: { 'solver/determinant': Log(), 'solver/logdet': Log() },
  })
}

/**
 * The minimum covariance determinant as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { MinCovDet } from 'mdml-engine/covariance'
 * const x = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [-0.1, 0], [0, -0.1], [20, 20]]
 * new MinCovDet({ nTrials: 5 }).fit(x).model.params.support[6] // => false
 * ```
 */
export class MinCovDet extends ExplainedEstimatorClass<MinCovDetHyper, Fitted<MinCovDetParams> & CovarianceVerbs> {
  /**
   * Make an unfitted MCD model.
   *
   * @param hyper - The hyperparameters of {@link minCovDet}.
   */
  constructor(hyper: MinCovDetHyper = {}) {
    super(minCovDet(hyper))
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Elliptic envelope and Mahalanobis outlier detector

/** Hyperparameters of the elliptic envelope: the Mahalanobis outlier detector over a chosen covariance estimator. */
export interface EllipticEnvelopeHyper {
  /** The expected fraction of outliers in `(0, 0.5]` (default 0.1). */
  readonly contamination: number
  /** The covariance behind the ellipse: `'mcd'` (default, robust), `'empirical'`, `'shrunk'`, `'ledoit-wolf'` or `'oas'`. */
  readonly method: 'mcd' | CovarianceMethod
  /** Fixed shrinkage intensity when `method` is `'shrunk'` (default 0.1). */
  readonly shrinkage: number
  /** Support fraction of FastMCD when `method` is `'mcd'` (default `(n + d + 1) / (2n)`). */
  readonly supportFraction?: number
  /** Number of random starts of FastMCD (default 30). */
  readonly nTrials: number
}

/** The parameters of a fitted elliptic envelope or Mahalanobis detector. */
export type EllipticEnvelopeParams = {
  readonly location: number[]
  readonly covariance: number[][]
  readonly precision: number[][]
  readonly threshold: number
}

/** The verbs of a fitted elliptic envelope: the outlier verbs, and the squared Mahalanobis distances themselves. */
export interface EllipticEnvelopeVerbs extends OutlierVerbs {
  /** Squared Mahalanobis distances `[m]` (the anomaly score). */
  readonly mahalanobis: (x: TensorLike) => Tensor
}

/** A fitted elliptic envelope detector. */
export type FittedEllipticEnvelope = IterativeFitted<EllipticEnvelopeParams, EllipticEnvelopeVerbs>

function envelopeVerbs(p: EllipticEnvelopeParams): EllipticEnvelopeVerbs {
  const scores = (x: TensorLike): number[] => squaredMahalanobis(x, p.location, p.precision)
  return {
    ...outlierVerbs(scores, p.threshold, p.location.length),
    mahalanobis: (x) => fromStorage(Float64Array.from(scores(x)), [matrixView(x, 'mahalanobis').n], 'float64'),
  }
}

/**
 * The elliptic envelope (sklearn.covariance.EllipticEnvelope) on the iterative base: squared Mahalanobis distances
 * under the minimum covariance determinant (or empirical, shrunk, Ledoit-Wolf or OAS covariance), thresholded at the
 * `(1 - contamination)` quantile. Its exposition is an `anomalyScores` record with the squared distances and, for one
 * or two features, a `decisionSurface`.
 *
 * @param hyper - Contamination, the covariance `method` and the FastMCD support fraction and starts.
 * @returns The iterative outlier detector.
 * @example
 * ```ts
 * import { ellipticEnvelope } from 'mdml-engine/covariance'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [-0.1, 0], [0, -0.1], [20, 20]]
 * toFlat(ellipticEnvelope({ contamination: 0.15, nTrials: 5 }).fit({ x }).predict([[0, 0], [20, 20]])) // => [1, -1]
 * ```
 */
export function ellipticEnvelope(
  hyper: Partial<EllipticEnvelopeHyper> = {},
): IterativeEstimator<EllipticEnvelopeHyper, FastMcdState, EllipticEnvelopeParams, EllipticEnvelopeVerbs> {
  return iterative<EllipticEnvelopeHyper, FastMcdState, EllipticEnvelopeParams, EllipticEnvelopeVerbs>({
    name: 'elliptic-envelope',
    kind: 'outlier',
    defaults: {
      contamination: hyper.contamination ?? 0.1,
      method: hyper.method ?? 'mcd',
      shrinkage: hyper.shrinkage ?? 0.1,
      ...(hyper.supportFraction === undefined ? {} : { supportFraction: hyper.supportFraction }),
      nTrials: hyper.nTrials ?? 30,
    },
    hyperSpace: (h) =>
      dict({
        contamination: real(0.01, 0.5, {
          label: 'contamination',
          description: 'The expected fraction of outliers: sets the threshold on the scores.',
          default: h.contamination,
        }),
        method: choice(['mcd', 'empirical', 'shrunk', 'ledoit-wolf', 'oas'] as const, {
          label: 'covariance',
          description: 'Which covariance estimator defines the Mahalanobis ellipse.',
          default: h.method,
        }),
      }),
    algorithm: (data, h) => {
      const mode = h.method
      if (mode === 'mcd') {
        return fastMcdAlgorithm(data.x, {
          ...(h.supportFraction === undefined ? {} : { supportFraction: h.supportFraction }),
          ...(h.nTrials === undefined ? {} : { nTrials: h.nTrials }),
        })
      }
      const inner = shrinkageAlgorithm(data.x, { method: mode, shrinkage: h.shrinkage })
      const n = matrixView(data.x, 'ellipticEnvelope').n
      const wrap = (s: ReturnType<typeof inner.init>, prev: number): FastMcdState => ({
        subset: Array.from({ length: n }, (_, i) => i),
        rawLocation: s.location,
        rawCovariance: s.covariance,
        location: s.location,
        covariance: s.covariance,
        precision: s.precision,
        support: new Array<boolean>(n).fill(true),
        logdet: s.logdet,
        determinant: Number.isFinite(s.logdet) ? Math.exp(s.logdet) : 0,
        previous: prev,
      })
      return {
        name: inner.name,
        config: inner.config,
        schema: channels({
          metrics: {
            determinant: { description: 'determinant of the estimated covariance' },
            logdet: { description: 'log-determinant of the estimated covariance' },
          },
        }),
        init: (k) => wrap(inner.init(k), Number.POSITIVE_INFINITY),
        step: (s) => ({ ...s, previous: s.logdet }),
        done: (s) => (s.previous === s.logdet ? 'completed' : undefined),
        observe: (s) => ({
          metrics: Number.isFinite(s.logdet) ? { determinant: s.determinant, logdet: s.logdet } : {},
        }),
        serialise: (s) => stateToJson(s),
        deserialise: (j) => stateFromJson(j) as FastMcdState,
      }
    },
    maxSteps: (h) => (h.method === 'mcd' ? 30 : 1),
    params: (s, data, h) => {
      const train = squaredMahalanobis(data.x, s.location, s.precision)
      const threshold = contaminationThreshold(train, h.contamination)
      return {
        location: s.location,
        covariance: s.covariance,
        precision: s.precision,
        threshold,
      }
    },
    verbs: envelopeVerbs,
    records: (s, data, h) => {
      const train = squaredMahalanobis(data.x, s.location, s.precision)
      const threshold = contaminationThreshold(train, h.contamination)
      const params: EllipticEnvelopeParams = {
        location: s.location,
        covariance: s.covariance,
        precision: s.precision,
        threshold,
      }
      const fitted = {
        estimator: 'elliptic-envelope',
        kind: 'outlier' as const,
        params,
        ...envelopeVerbs(params),
      }
      return outlierExposition(
        fitted,
        data.x,
        h.method === 'mcd' ? 'ellipticEnvelope' : 'mahalanobis',
        threshold,
        { name: 'squaredMahalanobis', values: train },
        train,
      )
    },
    logs: { 'solver/determinant': Log(), 'solver/logdet': Log() },
  })
}

/**
 * The elliptic envelope as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { EllipticEnvelope } from 'mdml-engine/covariance'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [-0.1, 0], [0, -0.1], [20, 20]]
 * toFlat(new EllipticEnvelope({ contamination: 0.15, nTrials: 5 }).fit(x).predict([[0, 0], [20, 20]])) // => [1, -1]
 * ```
 */
export class EllipticEnvelope extends ExplainedEstimatorClass<
  EllipticEnvelopeHyper,
  Fitted<EllipticEnvelopeParams> & EllipticEnvelopeVerbs
> {
  /**
   * Make an unfitted elliptic envelope detector.
   *
   * @param hyper - The hyperparameters of {@link ellipticEnvelope}.
   */
  constructor(hyper: Partial<EllipticEnvelopeHyper> = {}) {
    super(ellipticEnvelope(hyper))
  }
}
