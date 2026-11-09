/**
 * The nearest valid matrix to a given one, for the covariance and correlation matrices that pairwise, thresholded,
 * denoised or hand-edited estimates leave indefinite: Higham's alternating projections with Dykstra's correction for
 * the nearest correlation matrix (a stepped, traced algorithm: the distance to the input and the number of negative
 * eigenvalues at each sweep), the fast eigenvalue clipping that is not the nearest but costs one decomposition, and the
 * covariance and symmetric positive semidefinite versions with the conventions of statsmodels' `corr_nearest`,
 * `corr_clipped` and `cov_nearest`.
 */

import { OmaiError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  frobeniusDistance,
  fromCorrelation,
  fromEigen,
  type Rows,
  squareRows,
  symmetricEigen,
  symmetrise,
  toCorrelation,
} from './_matrix.ts'

/** The state of Higham's alternating projections: the unit-diagonal iterate, the Dykstra correction and what is recorded. */
export interface NearestCorrelationState {
  /** The iterate `Y_k` with a unit diagonal (the projection onto the unit-diagonal set); it tends to the answer. */
  readonly y: number[][]
  /** The positive semidefinite iterate `X_k` (the projection onto the cone); it tends to the same answer. */
  readonly x: number[][]
  /** Dykstra's correction `Delta S_k`. */
  readonly correction: number[][]
  /** The sweeps made. */
  readonly sweeps: number
  /** The Frobenius distance from the input to `Y_k`. */
  readonly distance: number
  /** The relative change of `Y_k` and `X_k` in the last sweep. */
  readonly change: number
  /** The number of eigenvalues of `Y_k` below `-1e-12`. */
  readonly negativeEigenvalues: number
  /** The smallest eigenvalue of `Y_k`. */
  readonly minEigenvalue: number
}

/** Options of {@link nearestCorrelationAlgorithm}. */
export interface NearestCorrelationOptions {
  /** The smallest eigenvalue allowed (the floor the cone projection clips to; default 0). */
  readonly threshold?: number
  /** Stops when the relative change of the iterates in a sweep is at most this (default 1e-13). */
  readonly tol?: number
}

function countBelow(values: readonly number[], limit: number): number {
  let c = 0
  for (const v of values) if (v < limit) c++
  return c
}

function setUnitDiagonal(a: Rows): Rows {
  return a.map((row, i) => row.map((v, j) => (i === j ? 1 : v)))
}

function norm(a: readonly (readonly number[])[]): number {
  let s = 0
  for (const r of a) for (const v of r) s += v * v
  return Math.sqrt(s)
}

/** The matrix with eigenvalues below `value` raised to it (statsmodels' `clip_evals`), and whether any was. */
function clipEvals(a: Rows, value: number): { matrix: Rows; clipped: boolean } {
  const { values, vectors } = symmetricEigen(a)
  const clipped = values.some((v) => v < value)
  return {
    matrix: fromEigen(
      values.map((v) => Math.max(v, value)),
      vectors,
    ),
    clipped,
  }
}

/**
 * Higham's alternating projections for the nearest correlation matrix (Higham 2002) with Dykstra's correction, as a
 * traced {@link Algorithm}: each sweep subtracts the correction, projects onto the positive semidefinite cone by
 * clipping eigenvalues, updates the correction and projects onto the unit-diagonal matrices. It records the distance
 * to the input, the number of negative eigenvalues and the smallest eigenvalue of the iterate, and the change per sweep.
 * The iterates converge linearly to the unique correlation matrix nearest to the input in the Frobenius norm.
 *
 * @param a - The matrix to repair `[p, p]` (its symmetric part is used).
 * @param options - The eigenvalue floor and the stopping tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { nearestCorrelationAlgorithm } from 'omai-engine/covariance'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const out = runAlgorithm(nearestCorrelationAlgorithm([[1, 0.9, 0.2], [0.9, 1, 0.9], [0.2, 0.9, 1]]), { maxSteps: 300 })
 * out.state.negativeEigenvalues // => 0
 * ```
 */
export function nearestCorrelationAlgorithm(
  a: readonly (readonly number[])[],
  options: NearestCorrelationOptions = {},
): Algorithm<NearestCorrelationState> {
  const input = symmetrise(a)
  const p = input.length
  const threshold = options.threshold ?? 0
  const tol = options.tol ?? 1e-13
  const summarise = (y: Rows): { negativeEigenvalues: number; minEigenvalue: number } => {
    const values = symmetricEigen(y).values
    return { negativeEigenvalues: countBelow(values, -1e-12), minEigenvalue: values[p - 1] ?? 0 }
  }
  return {
    name: 'nearest-correlation',
    config: { threshold, tol },
    schema: channels({
      metrics: {
        distance: { description: 'Frobenius distance from the input to the iterate', direction: 'lower' },
        negativeEigenvalues: { description: 'the number of negative eigenvalues of the iterate', direction: 'lower' },
        minEigenvalue: { description: 'the smallest eigenvalue of the iterate' },
        change: { description: 'the relative change of the iterates in the sweep', direction: 'lower' },
      },
      tensors: {
        matrix: { role: 'param', description: 'the unit-diagonal iterate', shape: ['p', 'p'] },
        psd: { role: 'state', description: 'the positive semidefinite iterate', shape: ['p', 'p'] },
      },
    }),
    init: () => ({
      y: input,
      x: input,
      correction: input.map((r) => r.map(() => 0)),
      sweeps: 0,
      distance: 0,
      change: Number.POSITIVE_INFINITY,
      ...summarise(input),
    }),
    step: (s) => {
      const r = s.y.map((row, i) => row.map((v, j) => v - (s.correction[i]?.[j] ?? 0)))
      const x = clipEvals(r, threshold).matrix
      const correction = x.map((row, i) => row.map((v, j) => v - (r[i]?.[j] ?? 0)))
      const y = setUnitDiagonal(x)
      const change = Math.max(
        frobeniusDistance(y, s.y) / Math.max(norm(y), Number.MIN_VALUE),
        frobeniusDistance(x, s.x) / Math.max(norm(x), Number.MIN_VALUE),
      )
      return {
        y,
        x,
        correction,
        sweeps: s.sweeps + 1,
        distance: frobeniusDistance(y, input),
        change,
        ...summarise(y),
      }
    },
    done: (s) => (s.change <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        distance: s.distance,
        negativeEigenvalues: s.negativeEigenvalues,
        minEigenvalue: s.minEigenvalue,
        ...(Number.isFinite(s.change) ? { change: s.change } : {}),
      },
      tensors: {
        matrix: { role: 'param', value: s.y.flat(), shape: [p, p] },
        psd: { role: 'state', value: s.x.flat(), shape: [p, p] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as NearestCorrelationState,
  }
}

/** The result of a nearest-matrix computation. */
export interface NearestResult {
  /** The repaired matrix. */
  readonly matrix: number[][]
  /** The Frobenius distance from the input's symmetric part (for a covariance, from its correlation matrix). */
  readonly distance: number
  /** The sweeps made (0 for a direct clipping). */
  readonly iterations: number
  /** Whether the iteration met its tolerance within the sweep budget. */
  readonly converged: boolean
  /** The smallest eigenvalue of the result. */
  readonly minEigenvalue: number
}

/** Options of {@link nearestCorrelation}. */
export interface NearestCorrelationFitOptions extends NearestCorrelationOptions {
  /** The most sweeps (default `100 p`, as statsmodels' `n_fact = 100`). */
  readonly maxIter?: number
}

/**
 * The correlation matrix nearest to `a` in the Frobenius norm (Higham 2002): symmetric, unit diagonal, positive
 * semidefinite. A matrix that is already a correlation matrix comes back unchanged (to rounding). The Dykstra
 * iteration is {@link nearestCorrelationAlgorithm}; the result is finished by one eigenvalue clip when the last sweep
 * left an eigenvalue below the floor by more than rounding, so it is a valid correlation matrix whatever the tolerance.
 *
 * @param a - The matrix to repair `[p, p]`.
 * @param options - The eigenvalue floor, the tolerance and the sweep budget.
 * @returns The matrix, its distance from the input, the sweeps made, whether it converged, and its smallest eigenvalue.
 * @index function Nearest correlation matrix by Higham's alternating projections with Dykstra's correction.
 * @example
 * ```ts
 * import { nearestCorrelation } from 'omai-engine/covariance'
 * const r = nearestCorrelation([[1, 0.9, 0.2], [0.9, 1, 0.9], [0.2, 0.9, 1]])
 * r.minEigenvalue > -1e-12 // => true
 * ```
 */
export function nearestCorrelation(a: TensorLike, options: NearestCorrelationFitOptions = {}): NearestResult {
  const input = symmetrise(squareRows(a, 'nearestCorrelation'))
  const p = input.length
  const threshold = options.threshold ?? 0
  const out = runAlgorithm(nearestCorrelationAlgorithm(input, options), { maxSteps: options.maxIter ?? 100 * p })
  let matrix = out.state.y
  const values = symmetricEigen(matrix).values
  if ((values[p - 1] ?? 0) < threshold - 1e-12) {
    // Not converged far enough to be valid: clip once more and rescale, as `corr_clipped` does.
    matrix = toCorrelation(clipEvals(matrix, threshold).matrix).correlation
  }
  return {
    matrix,
    distance: frobeniusDistance(matrix, input),
    iterations: out.steps,
    converged: out.status === 'converged',
    minEigenvalue: symmetricEigen(matrix).values[p - 1] ?? 0,
  }
}

/**
 * The fast repair of a correlation matrix by clipping (statsmodels' `corr_clipped`): eigenvalues below `threshold` are
 * raised to it, then the matrix is rescaled to a unit diagonal. One eigendecomposition, but farther from the input than
 * {@link nearestCorrelation}; a matrix with no eigenvalue below the threshold is returned as it is.
 *
 * @param a - A symmetric matrix with a unit diagonal `[p, p]`.
 * @param threshold - The smallest eigenvalue allowed (default 1e-15, as statsmodels).
 * @returns The clipped correlation matrix, its distance from the input and its smallest eigenvalue.
 * @index function Fast correlation repair by eigenvalue clipping and rescaling.
 * @example
 * ```ts
 * import { clipCorrelation } from 'omai-engine/covariance'
 * clipCorrelation([[1, 0.5], [0.5, 1]]).distance // => 0
 * ```
 */
export function clipCorrelation(a: TensorLike, threshold = 1e-15): NearestResult {
  const input = symmetrise(squareRows(a, 'clipCorrelation'))
  const { matrix, clipped } = clipEvals(input, threshold)
  const out = clipped ? toCorrelation(matrix).correlation : input
  const p = input.length
  return {
    matrix: out,
    distance: frobeniusDistance(out, input),
    iterations: 0,
    converged: true,
    minEigenvalue: symmetricEigen(out).values[p - 1] ?? 0,
  }
}

/** Options of {@link nearestCovariance}. */
export interface NearestCovarianceOptions {
  /** `'clipped'` (the default; fast) or `'nearest'` (Higham's alternating projections), as statsmodels' `cov_nearest`. */
  readonly method?: 'clipped' | 'nearest'
  /** The smallest eigenvalue of the intermediate correlation matrix (default 1e-15). */
  readonly threshold?: number
  /** The sweep budget of `'nearest'` is `nFact` times the dimension (default 100). */
  readonly nFact?: number
  /** The stopping tolerance of `'nearest'` (default 1e-13). */
  readonly tol?: number
}

/** The result of {@link nearestCovariance}. */
export interface NearestCovariance {
  /** The repaired covariance, with the input's variances. */
  readonly covariance: number[][]
  /** The repaired correlation matrix behind it. */
  readonly correlation: number[][]
  /** The standard deviations, the square roots of the input's diagonal. */
  readonly std: number[]
  /** The Frobenius distance of the correlation matrices (input to repaired). */
  readonly distance: number
}

/**
 * The positive semidefinite covariance matrix nearest to `a` with its variances kept (statsmodels' `cov_nearest`): the
 * matrix is turned into a correlation matrix by its standard deviations, repaired by clipping or by Higham's alternating
 * projections, and scaled back.
 *
 * @param a - A symmetric matrix with a positive diagonal `[p, p]`.
 * @param options - The method, the eigenvalue floor and the sweep budget.
 * @returns The covariance, correlation and standard deviations, and the distance in correlation space.
 * @throws OmaiError with code `value` when a diagonal entry is not positive.
 * @index function Nearest covariance matrix keeping the variances, by clipping or alternating projections.
 * @example
 * ```ts
 * import { nearestCovariance } from 'omai-engine/covariance'
 * nearestCovariance([[4, 0], [0, 9]]).covariance // => [[4, 0], [0, 9]]
 * ```
 */
export function nearestCovariance(a: TensorLike, options: NearestCovarianceOptions = {}): NearestCovariance {
  const input = symmetrise(squareRows(a, 'nearestCovariance'))
  for (const [i, row] of input.entries())
    if (!((row[i] ?? 0) > 0)) throw new OmaiError('value', `nearestCovariance: the diagonal entry ${i} is not positive`)
  const { correlation, scale } = toCorrelation(input)
  const threshold = options.threshold ?? 1e-15
  const repaired =
    options.method === 'nearest'
      ? nearestCorrelation(correlation, {
          threshold,
          maxIter: Math.trunc(input.length * (options.nFact ?? 100)),
          ...(options.tol === undefined ? {} : { tol: options.tol }),
        })
      : clipCorrelation(correlation, threshold)
  return {
    covariance: fromCorrelation(repaired.matrix, scale),
    correlation: repaired.matrix,
    std: scale,
    distance: repaired.distance,
  }
}

/** Options of {@link nearestSpd}. */
export interface NearestSpdOptions {
  /** The smallest eigenvalue allowed (default 0: positive semidefinite; a small positive value makes it definite). */
  readonly floor?: number
}

/**
 * The symmetric positive semidefinite matrix nearest to `a` in the Frobenius norm (Higham 1988): the symmetric part
 * with its negative eigenvalues set to zero (or raised to `floor`), which is the exact projection onto the matrices
 * with no eigenvalue below the floor. The diagonal is not kept; for a correlation matrix use
 * {@link nearestCorrelation}, for a covariance with fixed variances {@link nearestCovariance}.
 *
 * @param a - A square matrix `[p, p]`.
 * @param options - The eigenvalue floor.
 * @returns The repaired matrix, its distance from the symmetric part of the input and its smallest eigenvalue.
 * @index function Nearest symmetric positive semidefinite matrix in the Frobenius norm.
 * @example
 * ```ts
 * import { nearestSpd } from 'omai-engine/covariance'
 * nearestSpd([[1, 0], [0, -1]]).matrix // => [[1, 0], [0, 0]]
 * ```
 */
export function nearestSpd(a: TensorLike, options: NearestSpdOptions = {}): NearestResult {
  const input = symmetrise(squareRows(a, 'nearestSpd'))
  const floor = options.floor ?? 0
  const { values, vectors } = symmetricEigen(input)
  const matrix = fromEigen(
    values.map((v) => Math.max(v, floor)),
    vectors,
  )
  return {
    matrix,
    distance: frobeniusDistance(matrix, input),
    iterations: 0,
    converged: true,
    minEigenvalue: Math.max(values[values.length - 1] ?? 0, floor),
  }
}

/**
 * Whether a matrix is a correlation matrix within a tolerance: symmetric, with a unit diagonal and no eigenvalue below
 * `-tol`.
 *
 * @param a - The matrix `[p, p]`.
 * @param tol - The tolerance of each of the three conditions (default 1e-10).
 * @returns Whether it is a valid correlation matrix.
 * @example
 * ```ts
 * import { isCorrelationMatrix } from 'omai-engine/covariance'
 * isCorrelationMatrix([[1, 0.5], [0.5, 1]]) // => true
 * isCorrelationMatrix([[1, 2], [2, 1]]) // => false
 * ```
 */
export function isCorrelationMatrix(a: TensorLike, tol = 1e-10): boolean {
  const m = squareRows(a, 'isCorrelationMatrix')
  const p = m.length
  for (let i = 0; i < p; i++) {
    if (Math.abs((m[i]?.[i] ?? 0) - 1) > tol) return false
    for (let j = 0; j < i; j++) if (Math.abs((m[i]?.[j] ?? 0) - (m[j]?.[i] ?? 0)) > tol) return false
  }
  return (symmetricEigen(m).values[p - 1] ?? 0) >= -tol
}
