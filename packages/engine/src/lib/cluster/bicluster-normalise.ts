/**
 * The normalisations that spectral biclustering applies before its SVD (Kluger et al. 2003; as scikit-learn's
 * `cluster/_bicluster.py`): independent scaling of rows and columns, the bistochastic iteration that repeats it until
 * every row sums to one constant and every column to another (a traced `Algorithm`, one scaling per step), and the
 * double centring of the log of the matrix.
 */

import { OmaiError } from '../base/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'

type Rows = readonly (readonly number[])[]

/**
 * Shifts a matrix so its smallest entry is at least `minValue` (scikit-learn's `make_nonnegative`): adds
 * `minValue - min` when the minimum is below it.
 *
 * @param x - The matrix `[n, d]`.
 * @param minValue - The floor (default 0).
 * @returns The shifted matrix.
 * @throws OmaiError `shape` for an empty or ragged matrix, `value` for non-finite entries.
 * @example
 * ```ts
 * import { makeNonnegative } from 'omai-engine/cluster'
 * makeNonnegative([[-1, 2], [3, 4]]) // => [[0, 3], [4, 5]]
 * ```
 */
export function makeNonnegative(x: Rows, minValue = 0): number[][] {
  const d = x[0]?.length ?? 0
  if (x.length === 0 || d === 0) throw new OmaiError('shape', 'makeNonnegative: the matrix is empty')
  if (x.some((r) => r.length !== d)) throw new OmaiError('shape', 'makeNonnegative: the rows differ in length')
  let min = Number.POSITIVE_INFINITY
  for (const r of x)
    for (const v of r) {
      if (!Number.isFinite(v)) throw new OmaiError('value', 'makeNonnegative: the matrix has non-finite entries')
      if (v < min) min = v
    }
  const shift = min < minValue ? minValue - min : 0
  return x.map((r) => r.map((v) => v + shift))
}

/**
 * Scales rows and columns independently (scikit-learn's `_scale_normalize`): `diag(1 / sqrt(row sums)) X diag(1 /
 * sqrt(column sums))` of the matrix made non-negative.
 *
 * @param x - The matrix `[n, d]`.
 * @returns The scaled matrix and the row and column scaling factors.
 * @throws OmaiError `value` when a row or a column sums to zero (the scaling is undefined there).
 * @example
 * ```ts
 * import { scaleNormalize } from 'omai-engine/cluster'
 * scaleNormalize([[1, 1], [1, 1]]).matrix // => [[0.5, 0.5], [0.5, 0.5]]
 * ```
 */
export function scaleNormalize(x: Rows): { matrix: number[][]; rowScale: number[]; columnScale: number[] } {
  const a = makeNonnegative(x)
  const d = a[0]?.length ?? 0
  const rowSums = a.map((r) => r.reduce((s, v) => s + v, 0))
  const colSums = Array.from({ length: d }, (_, j) => a.reduce((s, r) => s + (r[j] ?? 0), 0))
  if (rowSums.some((s) => !(s > 0)) || colSums.some((s) => !(s > 0)))
    throw new OmaiError('value', 'scaleNormalize: a row or a column of the matrix sums to zero')
  const rowScale = rowSums.map((s) => 1 / Math.sqrt(s))
  const columnScale = colSums.map((s) => 1 / Math.sqrt(s))
  return {
    matrix: a.map((r, i) => r.map((v, j) => (rowScale[i] ?? 0) * v * (columnScale[j] ?? 0))),
    rowScale,
    columnScale,
  }
}

/** The state of {@link bistochasticNormalizeAlgorithm}: the matrix so far and how far the last scaling moved it. */
export interface BistochasticState {
  readonly matrix: number[][]
  /** The Frobenius norm of the change of the last scaling (infinite before the first). */
  readonly distance: number
}

/** Options of {@link bistochasticNormalizeAlgorithm}. */
export interface BistochasticOptions {
  /** The change in Frobenius norm below which the iteration stops (default 1e-5). */
  readonly tol?: number
}

/**
 * The bistochastic normalisation as a traced {@link Algorithm} (scikit-learn's `_bistochastic_normalize`): each step
 * rescales rows and columns of the current matrix ({@link scaleNormalize}), until the change is below `tol`. The
 * rows and columns then each sum to a constant. It observes the size of the last change and the matrix.
 *
 * @param x - The matrix `[n, d]`.
 * @param options - The tolerance.
 * @returns The algorithm.
 * @throws OmaiError `value` for a zero row or column, `shape` for an empty or ragged matrix.
 * @example
 * ```ts
 * import { bistochasticNormalizeAlgorithm } from 'omai-engine/cluster'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const out = runAlgorithm(bistochasticNormalizeAlgorithm([[1, 2], [3, 4]]), { maxSteps: 100 })
 * out.state.distance < 1e-5 // => true
 * ```
 */
export function bistochasticNormalizeAlgorithm(
  x: Rows,
  options: BistochasticOptions = {},
): Algorithm<BistochasticState> {
  const tol = options.tol ?? 1e-5
  const start = makeNonnegative(x)
  const n = start.length
  const d = start[0]?.length ?? 0
  return {
    name: 'bistochastic-normalise',
    config: { tol },
    schema: channels({
      metrics: { distance: { description: 'Frobenius norm of the change of the last scaling', direction: 'lower' } },
      tensors: { matrix: { role: 'state', description: 'the matrix being scaled', shape: ['n', 'd'] } },
    }),
    init: () => ({ matrix: start.map((r) => [...r]), distance: Number.POSITIVE_INFINITY }),
    step: (s) => {
      const next = scaleNormalize(s.matrix).matrix
      const change = next.flatMap((r, i) => r.map((v, j) => v - (s.matrix[i]?.[j] ?? 0)))
      return { matrix: next, distance: dnrm2(change) }
    },
    done: (s) => (s.distance < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { distance: Number.isFinite(s.distance) ? s.distance : -1 },
      tensors: { matrix: { role: 'state', value: s.matrix.flat(), shape: [n, d] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BistochasticState,
  }
}

/**
 * Rescales rows and columns until every row sums to one constant and every column to another (scikit-learn's
 * `_bistochastic_normalize`): {@link bistochasticNormalizeAlgorithm} run to its tolerance.
 *
 * @param x - The matrix `[n, d]`.
 * @param options - The tolerance and the most iterations (default 1000).
 * @returns The normalised matrix.
 * @example
 * ```ts
 * import { bistochasticNormalize } from 'omai-engine/cluster'
 * const b = bistochasticNormalize([[1, 2], [3, 4]])
 * Math.abs((b[0]?.[0] ?? 0) + (b[0]?.[1] ?? 0) - (b[1]?.[0] ?? 0) - (b[1]?.[1] ?? 0)) < 1e-3 // => true
 * ```
 */
export function bistochasticNormalize(
  x: Rows,
  options: BistochasticOptions & { readonly maxIter?: number } = {},
): number[][] {
  return runAlgorithm(bistochasticNormalizeAlgorithm(x, options), { maxSteps: options.maxIter ?? 1000 }).state.matrix
}

/**
 * Kluger's log-interactions normalisation (scikit-learn's `_log_normalize`): the log of the matrix (made at least 1)
 * with row means, column means and the grand mean removed, `L - row mean - column mean + mean`.
 *
 * @param x - The matrix `[n, d]`.
 * @returns The double-centred log matrix.
 * @example
 * ```ts
 * import { logNormalize } from 'omai-engine/cluster'
 * logNormalize([[1, 1], [1, 1]]) // => [[0, 0], [0, 0]]
 * ```
 */
export function logNormalize(x: Rows): number[][] {
  const l = makeNonnegative(x, 1).map((r) => r.map(Math.log))
  const n = l.length
  const d = l[0]?.length ?? 0
  const rowMean = l.map((r) => r.reduce((s, v) => s + v, 0) / d)
  const colMean = Array.from({ length: d }, (_, j) => l.reduce((s, r) => s + (r[j] ?? 0), 0) / n)
  const mean = rowMean.reduce((s, v) => s + v, 0) / n
  return l.map((r, i) => r.map((v, j) => v - (rowMean[i] ?? 0) - (colMean[j] ?? 0) + mean))
}
