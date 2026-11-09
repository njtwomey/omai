/**
 * Rank-based graph input: the nonparanormal (Gaussian copula) model of Liu, Lafferty and Wasserman, in which the data
 * are a monotone transform of a Gaussian vector and the graph is the Gaussian's. `nonparanormalTransform` maps each
 * variable to normal scores of its Winsorised empirical distribution, `nonparanormalCorrelation` estimates the latent
 * correlation matrix from normal scores, Spearman's rho (`2 sin(pi rho / 6)`) or Kendall's tau (`sin(pi tau / 2)`),
 * and `glassoNonparanormal` feeds it to the graphical lasso. The rank-based estimates are exactly invariant to
 * monotone transforms of the variables, so heavy tails and skew no longer hide the graph.
 */

import { OmaiError } from '../base/index.ts'
import { kendallTauMatrix } from '../dists/index.ts'
import { matrixView } from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import { empiricalCovariance, ranksOf, standardise } from '../stats/index.ts'
import { fromStorage, type Tensor, type TensorLike, toArray, toFlat } from '../tensor/index.ts'
import type { Algorithm } from '../trace/index.ts'
import {
  type GraphicalLassoAlgorithmOptions,
  type GraphicalLassoPenalty,
  type GraphicalLassoState,
  graphicalLasso,
  graphicalLassoAlgorithm,
} from './graphical-lasso.ts'

/**
 * The truncation level of the Winsorised empirical distribution of Liu, Lafferty and Wasserman,
 * `1 / (4 n^(1/4) sqrt(pi log n))`: empirical distribution values are clipped to `[delta, 1 - delta]` before the
 * normal quantile, which keeps the transform's variance finite and gives the estimator its rate.
 *
 * @param n - The number of observations (at least 2).
 * @returns The truncation level.
 * @example
 * ```ts
 * import { winsorisationLevel } from 'omai-engine/covariance'
 * winsorisationLevel(100) // => 0.020784626763613683
 * ```
 */
export function winsorisationLevel(n: number): number {
  return 1 / (4 * n ** 0.25 * Math.sqrt(Math.PI * Math.log(n)))
}

/** Options of {@link nonparanormalTransform}. */
export interface NonparanormalTransformOptions {
  /**
   * Clip the empirical distribution (ranks over `n`) to `[delta, 1 - delta]` with {@link winsorisationLevel} (the
   * truncated estimator, default true); false uses ranks over `n + 1`, the shrunk estimator, which never reaches 0 or 1.
   */
  readonly winsorise?: boolean
  /** Keep each variable's sample mean and standard deviation (as the paper's transform does) or standardise (default `'unit'`). */
  readonly scale?: 'unit' | 'original'
}

/**
 * The nonparanormal transform: each column replaced by the normal quantile of its Winsorised empirical distribution
 * (ties get their average rank), so a variable that is a monotone transform of a Gaussian becomes approximately that
 * Gaussian. Columns come out with zero mean and unit variance (or the original mean and standard deviation, with
 * `scale: 'original'`).
 *
 * @param x - Observations `[n, p]`.
 * @param options - Whether to Winsorise and how to scale.
 * @returns The transformed observations `[n, p]`.
 * @throws OmaiError with code `value` for fewer than three observations.
 * @index function Nonparanormal transform: normal scores of the Winsorised empirical distribution.
 * @example
 * ```ts
 * import { nonparanormalTransform } from 'omai-engine/covariance'
 * import { toFlat } from 'omai-engine/tensor'
 * const z = toFlat(nonparanormalTransform([[1], [10], [100], [1000]], { winsorise: false }))
 * [Math.abs((z[0] ?? 0) + (z[3] ?? 0)) < 1e-12, Math.abs((z[1] ?? 0) + (z[2] ?? 0)) < 1e-12, (z[0] ?? 0) < (z[1] ?? 0)] // => [true, true, true]
 * ```
 */
export function nonparanormalTransform(x: TensorLike, options: NonparanormalTransformOptions = {}): Tensor {
  const m = matrixView(x, 'nonparanormalTransform', { minSamples: 3 })
  const { n, d } = m
  const winsorise = options.winsorise ?? true
  const delta = winsorisationLevel(n)
  const out = new Float64Array(n * d)
  for (let j = 0; j < d; j++) {
    const col = Float64Array.from({ length: n }, (_, i) => m.data[i * d + j] ?? 0)
    const ranks = toFlat(ranksOf(col))
    const probs = Float64Array.from(ranks, (r) =>
      winsorise ? Math.min(Math.max(r / n, delta), 1 - delta) : r / (n + 1),
    )
    const scores = toFlat(ndtri(fromStorage(probs, [n], 'float64')))
    let mean = 0
    for (const v of scores) mean += v / n
    let variance = 0
    for (const v of scores) variance += (v - mean) ** 2 / n
    const sd = Math.sqrt(variance)
    let mu = 0
    let sigma = 1
    if (options.scale === 'original') {
      for (const v of col) mu += v / n
      let s2 = 0
      for (const v of col) s2 += (v - mu) ** 2 / n
      sigma = Math.sqrt(s2)
    }
    for (let i = 0; i < n; i++) out[i * d + j] = mu + (sigma * ((scores[i] ?? 0) - mean)) / (sd > 0 ? sd : 1)
  }
  return fromStorage(out, [n, d], 'float64')
}

/** How the latent correlation of the nonparanormal model is estimated. */
export type RankCorrelationMethod = 'spearman' | 'kendall' | 'normal-scores'

/** Options of {@link nonparanormalCorrelation}. */
export interface NonparanormalCorrelationOptions {
  /**
   * `'spearman'` (default): `2 sin(pi rho / 6)` of the Spearman correlation; `'kendall'`: `sin(pi tau / 2)` of Kendall's
   * tau-b; `'normal-scores'`: the correlation of the {@link nonparanormalTransform}. The first two are invariant to
   * monotone transforms of the variables exactly.
   */
  readonly method?: RankCorrelationMethod
  /** Winsorise, for `'normal-scores'` (see {@link NonparanormalTransformOptions}). */
  readonly winsorise?: boolean
  /**
   * The sine transforms need not give a positive definite matrix: shrink it towards the identity just enough that its
   * smallest eigenvalue is at least 1e-6 (default true).
   */
  readonly positiveDefinite?: boolean
}

/** Shrink a correlation matrix towards the identity until its smallest eigenvalue reaches `floor`. */
function shrinkToDefinite(r: number[][], floor: number): number[][] {
  const lowest = Math.min(...toFlat(linalg.eigvalsh(r)))
  if (lowest >= floor) return r
  const t = (floor - lowest) / (1 - lowest)
  return r.map((row, a) => row.map((v, b) => (1 - t) * v + (a === b ? t : 0)))
}

/**
 * The latent correlation matrix of a nonparanormal sample: the correlation of the normal scores, or the sine
 * transform of Spearman's rho (`2 sin(pi rho / 6)`) or of Kendall's tau (`sin(pi tau / 2)`), which for a monotone
 * transform of a Gaussian vector are consistent for its correlation and depend on the data through the ranks alone.
 *
 * @param x - Observations `[n, p]`.
 * @param options - The estimator and whether to repair a matrix that is not positive definite.
 * @returns The correlation matrix `[p][p]`, with a unit diagonal.
 * @throws OmaiError with code `value` for fewer than three observations.
 * @index function Latent correlation of a nonparanormal sample from Spearman, Kendall or normal scores.
 * @example
 * ```ts
 * import { nonparanormalCorrelation } from 'omai-engine/covariance'
 * const x = [[1, 1], [2, 4], [3, 9], [4, 16]]
 * nonparanormalCorrelation(x, { method: 'kendall', positiveDefinite: false })[0]?.[1] // => 1
 * ```
 */
export function nonparanormalCorrelation(x: TensorLike, options: NonparanormalCorrelationOptions = {}): number[][] {
  const method = options.method ?? 'spearman'
  const m = matrixView(x, 'nonparanormalCorrelation', { minSamples: 3 })
  const { n, d } = m
  let r: number[][]
  if (method === 'kendall') {
    const tau = toArray(kendallTauMatrix(fromStorage(Float64Array.from(m.data), [n, d], 'float64'))) as number[][]
    r = tau.map((row, a) => row.map((v, b) => (a === b ? 1 : Math.sin((Math.PI * v) / 2))))
  } else if (method === 'normal-scores') {
    r = pearson(nonparanormalTransform(x, options.winsorise === undefined ? {} : { winsorise: options.winsorise }))
  } else if (method === 'spearman') {
    const ranks = new Float64Array(n * d)
    for (let j = 0; j < d; j++) {
      const col = toFlat(ranksOf(Float64Array.from({ length: n }, (_, i) => m.data[i * d + j] ?? 0)))
      for (let i = 0; i < n; i++) ranks[i * d + j] = col[i] ?? 0
    }
    r = pearson(fromStorage(ranks, [n, d], 'float64')).map((row, a) =>
      row.map((v, b) => (a === b ? 1 : 2 * Math.sin((Math.PI * v) / 6))),
    )
  } else throw new OmaiError('value', `nonparanormalCorrelation: unknown method '${String(method)}'`)
  return options.positiveDefinite === false ? r : shrinkToDefinite(r, 1e-6)
}

/** The Pearson correlation matrix of the columns. */
function pearson(x: Tensor): number[][] {
  const z = standardise(x, { ddof: 0 }).transformed
  const c = toArray(empiricalCovariance(z, { assumeCentered: true })) as number[][]
  return c.map((row, a) => row.map((v, b) => (a === b ? 1 : Math.max(-1, Math.min(1, v)))))
}

/** Options of {@link glassoNonparanormal}: those of {@link nonparanormalCorrelation} and of the graphical lasso. */
export type GlassoNonparanormalOptions = NonparanormalCorrelationOptions &
  GraphicalLassoAlgorithmOptions & { readonly maxIter?: number }

/**
 * The graphical lasso on the nonparanormal correlation matrix, as a traced {@link Algorithm} (the graphical lasso's
 * sweeps).
 *
 * @param x - Observations `[n, p]`.
 * @param alpha - The L1 penalty (on the correlation scale).
 * @param options - The rank correlation and the graphical lasso's tolerances.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { glassoNonparanormalAlgorithm } from 'omai-engine/covariance'
 * glassoNonparanormalAlgorithm([[1, 2], [2, 1], [3, 5], [4, 3]], 0.1).name // => 'graphical-lasso'
 * ```
 */
export function glassoNonparanormalAlgorithm(
  x: TensorLike,
  alpha: GraphicalLassoPenalty,
  options: GlassoNonparanormalOptions = {},
): Algorithm<GraphicalLassoState> {
  return graphicalLassoAlgorithm(nonparanormalCorrelation(x, options), alpha, options)
}

/**
 * The nonparanormal graphical lasso (the `huge` pipeline `huge.npn` then `huge(method = "glasso")`): a sparse
 * precision matrix for the latent Gaussian of a Gaussian-copula sample, estimated from its rank correlation. To choose
 * the penalty as well, apply {@link nonparanormalTransform} and pass the result to `selectGraphicalLassoPenalty`.
 *
 * @param x - Observations `[n, p]`.
 * @param alpha - The L1 penalty (on the correlation scale), or a matrix of penalties.
 * @param options - The rank correlation (default Spearman) and the graphical lasso's tolerances and sweeps.
 * @returns The graphical lasso state of the latent model, the correlation it was fitted to and the sweeps.
 * @index function Graphical lasso on a nonparanormal (rank-based) correlation matrix.
 * @example
 * ```ts
 * import { glassoNonparanormal } from 'omai-engine/covariance'
 * const x = [[1, 1, 0], [2, 3, 1], [3, 2, 5], [4, 5, 4], [5, 4, 2], [6, 7, 7]]
 * glassoNonparanormal(x, 0.5).precision.length // => 3
 * ```
 */
export function glassoNonparanormal(
  x: TensorLike,
  alpha: GraphicalLassoPenalty,
  options: GlassoNonparanormalOptions = {},
): GraphicalLassoState & { readonly nIter: number; readonly correlation: number[][] } {
  const correlation = nonparanormalCorrelation(x, options)
  return { ...graphicalLasso(correlation, alpha, options), correlation }
}
