/**
 * Nonlinear shrinkage of the sample covariance eigenvalues, the analytical quadratic-inverse form of Ledoit and Wolf
 * (2022, "Quadratic shrinkage for large covariance matrices"): each sample eigenvalue is replaced by its own optimal value
 * under the Frobenius loss, estimated in closed form from a kernel smoothing of the spectrum (the Stein shrinker and its
 * Hilbert conjugate, by a Cauchy kernel with a bandwidth proportional to the eigenvalue), the eigenvectors left as they
 * are. Unlike linear shrinkage to an identity it bends the spectrum: the large eigenvalues are pulled down and the small
 * ones up by different amounts, and it stays invertible for more variables than observations.
 */

import { OmaiError } from '../base/index.ts'
import { curveSet, type Exposition } from '../exposition/index.ts'
import {
  ExplainedEstimatorClass,
  type Fitted,
  type IterativeEstimator,
  iterative,
  type LearnData,
} from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { bool, dict } from '../space/index.ts'
import { type TensorLike, toArray } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { conditionOf, fromEigen, moments, symmetricEigen } from './_matrix.ts'
import { type CovarianceParams, type CovarianceVerbs, covarianceVerbs, mahalanobisRecord } from './shrinkage.ts'

/** Options of {@link quadraticInverseShrinkage}. */
export interface NonlinearShrinkageOptions {
  /**
   * Whether the data are already centred (default false): the mean is then not subtracted and the sample covariance is
   * divided by `n`; otherwise by `n - 1`, as Ledoit and Wolf's code does.
   */
  readonly assumeCentered?: boolean
}

/** The result of {@link quadraticInverseShrinkage}. */
export interface NonlinearShrinkage {
  /** The shrunk covariance `[p, p]`. */
  readonly covariance: number[][]
  /** The eigenvalues of the sample covariance (divided by the effective sample size), descending. */
  readonly sampleEigenvalues: number[]
  /** The shrunk eigenvalues in the same order, with the same sum. */
  readonly shrunkEigenvalues: number[]
  /** The column means (zeros when `assumeCentered`). */
  readonly location: number[]
  /** The concentration ratio `p / n` with the effective sample size `n`. */
  readonly ratio: number
  /** The smoothing bandwidth of the kernel. */
  readonly bandwidth: number
}

/**
 * The quadratic-inverse shrinkage (QIS) estimator of the covariance matrix (Ledoit and Wolf 2022): with the sample
 * eigenvalues `l_j`, `x_j = 1 / l_j`, the bandwidth `h = min(c^2, c^-2)^0.35 / p^0.35` for the concentration `c = p / n`
 * and the smoothed Stein shrinker `theta_j = mean_i x_j (x_j - x_i) / ((x_j - x_i)^2 + h^2 x_j^2)` with its conjugate
 * `H_j = mean_i h x_j^2 / ((x_j - x_i)^2 + h^2 x_j^2)`, the shrunk eigenvalue is
 * `1 / ((1 - c)^2 x_j + 2 c (1 - c) x_j theta_j + c^2 x_j (theta_j^2 + H_j^2))`; for `p > n` the non-null ones are
 * `1 / (x_j (theta_j^2 + H_j^2))` and the null eigenvalues all become `1 / ((c - 1) mean(x))`. The eigenvalues are
 * rescaled to keep the trace.
 *
 * @param x - Observations `[n, p]` (at least 3 rows).
 * @param options - Whether the data are already centred.
 * @returns The shrunk covariance with the sample and shrunk eigenvalues, the location, the concentration and the bandwidth.
 * @throws OmaiError with code `shape` for fewer than 3 observations.
 * @index function Analytical nonlinear (quadratic-inverse) shrinkage of the covariance eigenvalues.
 * @example
 * ```ts
 * import { quadraticInverseShrinkage } from 'omai-engine/covariance'
 * const x = [[1, 2], [3, 6], [5, 4], [2, 2], [4, 5], [0, 1]]
 * quadraticInverseShrinkage(x).shrunkEigenvalues.length // => 2
 * ```
 */
export function quadraticInverseShrinkage(x: TensorLike, options: NonlinearShrinkageOptions = {}): NonlinearShrinkage {
  const centredInput = options.assumeCentered === true
  const m = moments(x, 'quadraticInverseShrinkage', { assumeCentered: centredInput, minSamples: 3 })
  const p = m.d
  const n = centredInput ? m.n : m.n - 1
  if (n < 2) throw new OmaiError('shape', 'quadraticInverseShrinkage: needs at least 3 observations')
  const c = p / n
  const scale = m.n / n
  const sample = m.covariance.map((r) => r.map((v) => v * scale))
  const { values, vectors } = symmetricEigen(sample)
  const kept = Math.min(p, n)
  const floor = Math.max(values[0] ?? 0, Number.MIN_VALUE) * 1e-14
  const lambda = values.slice(0, kept).map((v) => Math.max(v, floor))
  const h = Math.min(c * c, 1 / (c * c)) ** 0.35 / p ** 0.35
  const inv = lambda.map((v) => 1 / v)
  const theta = new Array<number>(kept).fill(0)
  const conj = new Array<number>(kept).fill(0)
  for (let j = 0; j < kept; j++) {
    const xj = inv[j] ?? 0
    let a = 0
    let b = 0
    for (let i = 0; i < kept; i++) {
      const xi = inv[i] ?? 0
      const diff = xi - xj
      const den = diff * diff + h * h * xi * xi
      a += (xi * diff) / den
      b += (h * xi * xi) / den
    }
    theta[j] = a / kept
    conj[j] = b / kept
  }
  const amplitude = theta.map((t, j) => t * t + (conj[j] ?? 0) ** 2)
  let delta: number[]
  if (p <= n) {
    delta = inv.map(
      (xj, j) => 1 / ((1 - c) ** 2 * xj + 2 * c * (1 - c) * xj * (theta[j] ?? 0) + c * c * xj * (amplitude[j] ?? 0)),
    )
  } else {
    const meanInv = inv.reduce((a, v) => a + v, 0) / kept
    const nonNull = inv.map((xj, j) => 1 / (xj * (amplitude[j] ?? 0)))
    delta = [...nonNull, ...new Array<number>(p - kept).fill(1 / ((c - 1) * meanInv))]
  }
  const trace = values.reduce((a, v) => a + v, 0)
  const total = delta.reduce((a, v) => a + v, 0)
  const shrunk = delta.map((v) => (v * trace) / total)
  return {
    covariance: fromEigen(shrunk, vectors),
    sampleEigenvalues: values,
    shrunkEigenvalues: shrunk,
    location: m.location,
    ratio: c,
    bandwidth: h,
  }
}

/** Hyperparameters of {@link nonlinearShrinkageEstimator}. */
export interface NonlinearShrinkageHyper {
  /** Whether the data are already centred (default false). */
  readonly assumeCentered?: boolean
}

/** The state of a nonlinear shrinkage fit. */
export interface NonlinearShrinkageState {
  readonly location: number[]
  readonly covariance: number[][]
  readonly precision: number[][]
  readonly shrinkage: number
  readonly logdet: number
  readonly sampleEigenvalues: number[]
  readonly shrunkEigenvalues: number[]
  readonly ratio: number
  readonly bandwidth: number
  readonly done: boolean
}

/** The fitted parameters of the nonlinear shrinkage estimator. */
export type NonlinearShrinkageParams = CovarianceParams & { readonly ratio: number; readonly bandwidth: number }

function fitState(x: TensorLike, h: Required<NonlinearShrinkageHyper>, done: boolean): NonlinearShrinkageState {
  const r = quadraticInverseShrinkage(x, { assumeCentered: h.assumeCentered })
  const precision = toArray(linalg.pinv(r.covariance)) as number[][]
  return {
    location: r.location,
    covariance: r.covariance,
    precision,
    shrinkage: 0,
    logdet: r.shrunkEigenvalues.reduce((a, v) => a + Math.log(Math.max(v, Number.MIN_VALUE)), 0),
    sampleEigenvalues: r.sampleEigenvalues,
    shrunkEigenvalues: r.shrunkEigenvalues,
    ratio: r.ratio,
    bandwidth: r.bandwidth,
    done,
  }
}

/**
 * Quadratic-inverse shrinkage as a one-step {@link Algorithm}: the covariance, precision and log-determinant on the tape
 * with the condition number before and after and the sample and shrunk eigenvalues.
 *
 * @param x - Observations `[n, p]`.
 * @param options - Whether the data are already centred.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { nonlinearShrinkageAlgorithm } from 'omai-engine/covariance'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[1, 2], [3, 6], [5, 4], [2, 2], [4, 5], [0, 1]]
 * runAlgorithm(nonlinearShrinkageAlgorithm(x), { maxSteps: 2 }).state.shrunkEigenvalues.length // => 2
 * ```
 */
export function nonlinearShrinkageAlgorithm(
  x: TensorLike,
  options: NonlinearShrinkageHyper = {},
): Algorithm<NonlinearShrinkageState> {
  const h: Required<NonlinearShrinkageHyper> = { assumeCentered: options.assumeCentered ?? false }
  const p = moments(x, 'nonlinearShrinkageAlgorithm', { minSamples: 3 }).d
  return {
    name: 'nonlinear-shrinkage',
    config: { assumeCentered: h.assumeCentered },
    schema: channels({
      metrics: {
        logdet: { description: 'log-determinant of the estimated covariance matrix' },
        conditionBefore: { description: 'condition number of the sample covariance' },
        conditionAfter: { description: 'condition number of the shrunk covariance', direction: 'lower' },
        bandwidth: { description: 'the smoothing bandwidth of the kernel' },
      },
      tensors: {
        covariance: { role: 'param', description: 'the shrunk covariance matrix', shape: ['p', 'p'] },
        precision: { role: 'param', description: 'the pseudo-inverse precision matrix', shape: ['p', 'p'] },
        sampleEigenvalues: { role: 'state', description: 'the sample eigenvalues, descending', shape: ['p'] },
        shrunkEigenvalues: { role: 'state', description: 'the shrunk eigenvalues, descending', shape: ['p'] },
      },
    }),
    init: () => fitState(x, h, false),
    step: (s) => ({ ...s, done: true }),
    done: (s) => (s.done ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        bandwidth: s.bandwidth,
        ...(Number.isFinite(s.logdet) ? { logdet: s.logdet } : {}),
        ...(Number.isFinite(conditionOf(s.sampleEigenvalues))
          ? { conditionBefore: conditionOf(s.sampleEigenvalues) }
          : {}),
        ...(Number.isFinite(conditionOf(s.shrunkEigenvalues))
          ? { conditionAfter: conditionOf(s.shrunkEigenvalues) }
          : {}),
      },
      tensors: {
        covariance: { role: 'param', value: s.covariance.flat(), shape: [p, p] },
        precision: { role: 'param', value: s.precision.flat(), shape: [p, p] },
        sampleEigenvalues: { role: 'state', value: s.sampleEigenvalues, shape: [p] },
        shrunkEigenvalues: { role: 'state', value: s.shrunkEigenvalues, shape: [p] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as NonlinearShrinkageState,
  }
}

function spectrumRecord(s: NonlinearShrinkageState): Exposition {
  const ranks = s.sampleEigenvalues.map((_, i) => i + 1)
  const positive = s.sampleEigenvalues.every((v) => v > 1e-12) && s.shrunkEigenvalues.every((v) => v > 1e-12)
  return curveSet({
    x: ranks,
    xLabel: 'eigenvalue rank',
    yLabel: 'eigenvalue',
    yLog: positive,
    series: [
      { name: 'sample', y: s.sampleEigenvalues },
      { name: 'shrunk', y: s.shrunkEigenvalues },
    ],
  })
}

/**
 * Analytical nonlinear shrinkage (Ledoit and Wolf's quadratic-inverse form) as a covariance estimator on the iterative
 * base: location, covariance, precision (pseudo-inverse), `mahalanobis` and `score`. The exposition is the sample
 * eigenvalues against the shrunk ones by rank, and the Mahalanobis anomaly scores.
 *
 * @param hyper - Whether the data are already centred.
 * @returns The iterative estimator.
 * @index estimator Covariance by analytical nonlinear (quadratic-inverse) eigenvalue shrinkage.
 * @example
 * ```ts
 * import { nonlinearShrinkageEstimator } from 'omai-engine/covariance'
 * const x = [[1, 2], [3, 6], [5, 4], [2, 2], [4, 5], [0, 1]]
 * nonlinearShrinkageEstimator().fit({ x }).params.covariance.length // => 2
 * ```
 */
export function nonlinearShrinkageEstimator(
  hyper: NonlinearShrinkageHyper = {},
): IterativeEstimator<
  Required<NonlinearShrinkageHyper>,
  NonlinearShrinkageState,
  NonlinearShrinkageParams,
  CovarianceVerbs
> {
  return iterative<
    Required<NonlinearShrinkageHyper>,
    NonlinearShrinkageState,
    NonlinearShrinkageParams,
    CovarianceVerbs
  >({
    name: 'nonlinear-shrinkage',
    kind: 'density',
    defaults: { assumeCentered: hyper.assumeCentered ?? false },
    hyperSpace: (h) =>
      dict({
        assumeCentered: bool({
          label: 'assume centred',
          description: 'Treat the data as already centred (do not subtract the mean).',
          default: h.assumeCentered,
        }),
      }),
    algorithm: (data: LearnData, h) => nonlinearShrinkageAlgorithm(data.x, h),
    maxSteps: () => 1,
    params: (s) => ({
      location: s.location,
      covariance: s.covariance,
      precision: s.precision,
      shrinkage: s.shrinkage,
      ratio: s.ratio,
      bandwidth: s.bandwidth,
    }),
    verbs: covarianceVerbs,
    records: (s, data) => [
      spectrumRecord(s),
      mahalanobisRecord('nonlinearShrinkage', data.x, {
        location: s.location,
        covariance: s.covariance,
        precision: s.precision,
        shrinkage: 0,
      }),
    ],
    logs: { 'solver/conditionAfter': Log() },
  })
}

/**
 * The nonlinear shrinkage estimator as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { NonlinearShrinkageEstimator } from 'omai-engine/covariance'
 * new NonlinearShrinkageEstimator().fit([[1, 2], [3, 6], [5, 4], [2, 2], [4, 5], [0, 1]]).model.params.location.length // => 2
 * ```
 */
export class NonlinearShrinkageEstimator extends ExplainedEstimatorClass<
  Required<NonlinearShrinkageHyper>,
  Fitted<NonlinearShrinkageParams> & CovarianceVerbs
> {
  /**
   * Make an unfitted nonlinear shrinkage model.
   *
   * @param hyper - The hyperparameters of {@link nonlinearShrinkageEstimator}.
   */
  constructor(hyper: NonlinearShrinkageHyper = {}) {
    super(nonlinearShrinkageEstimator(hyper))
  }
}
