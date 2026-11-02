/**
 * The graphical lasso (Friedman, Hastie and Tibshirani, 2008; as in scikit-learn): a sparse precision matrix estimated
 * by an L1-penalised Gaussian log-likelihood, whose zeros represent conditional independences between variables.
 * Solved by coordinate descent column by column as a traced `Algorithm`, with the objective, the solver and an
 * estimator on the iterative base.
 */

import { MdmlError } from '../base/index.ts'
import { regularisationPath } from '../exposition/index.ts'
import {
  ExplainedEstimatorClass,
  type Fitted,
  featureNamesOf,
  type IterativeEstimator,
  type IterativeFitted,
  iterative,
  matrixView,
} from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { lassoCoordinateDescent } from '../optim/index.ts'
import { dict, int, real } from '../space/index.ts'
import { empiricalCovariance } from '../stats/index.ts'
import { asTensor, item, type TensorLike, toArray, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, Log, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { type CovarianceParams, type CovarianceVerbs, covarianceVerbs, mahalanobisRecord } from './shrinkage.ts'

/** The state of the graphical lasso: the covariance and precision estimates, the duality gap and the objective. */
export interface GraphicalLassoState {
  readonly covariance: number[][]
  readonly precision: number[][]
  readonly gap: number
  readonly objective: number
}

/**
 * The graphical lasso objective `-log det(Theta) + tr(S Theta) + alpha |Theta|_1` off the diagonal (scikit-learn's
 * convention).
 *
 * @param s - The empirical covariance `[p][p]`.
 * @param precision - The precision matrix `[p][p]`.
 * @param alpha - The L1 penalty.
 * @returns The objective value.
 * @example
 * ```ts
 * import { graphicalLassoObjective } from 'mdml-engine/covariance'
 * graphicalLassoObjective([[1, 0], [0, 1]], [[1, 0], [0, 1]], 0.1) // => 2
 * ```
 */
export function graphicalLassoObjective(
  s: readonly (readonly number[])[],
  precision: readonly (readonly number[])[],
  alpha: number,
): number {
  const p = s.length
  let trace = 0
  let l1 = 0
  for (let a = 0; a < p; a++)
    for (let b = 0; b < p; b++) {
      trace += (s[a]?.[b] ?? 0) * (precision[b]?.[a] ?? 0)
      if (a !== b) l1 += Math.abs(precision[a]?.[b] ?? 0)
    }
  return -item(linalg.slogdet(precision as number[][]).logabsdet) + trace + alpha * l1
}

/** sklearn's `_dual_gap`: tr(S Theta) - p + alpha |Theta|_1 off the diagonal. */
function dualGap(s: readonly (readonly number[])[], precision: readonly (readonly number[])[], alpha: number): number {
  const p = s.length
  let gap = -p
  for (let a = 0; a < p; a++)
    for (let b = 0; b < p; b++) {
      gap += (s[a]?.[b] ?? 0) * (precision[b]?.[a] ?? 0)
      if (a !== b) gap += alpha * Math.abs(precision[a]?.[b] ?? 0)
    }
  return gap
}

function activeEdges(precision: readonly (readonly number[])[]): number {
  const p = precision.length
  let edges = 0
  for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) if (Math.abs(precision[a]?.[b] ?? 0) > 1e-10) edges++
  return edges
}

/** Options of {@link graphicalLassoAlgorithm}. */
export interface GraphicalLassoAlgorithmOptions {
  /** Tolerance on the duality gap (default 1e-4). */
  readonly tol?: number
  /** Tolerance on the inner coordinate-descent lasso solver (default 1e-4). */
  readonly enetTol?: number
}

/**
 * The graphical lasso (Friedman, Hastie and Tibshirani 2008; sklearn's `graphical_lasso` with coordinate descent) as a
 * traced {@link Algorithm}: each step sweeps the columns, solving each column's lasso in the Gram form through
 * `optim`'s lasso coordinate descent (the Gram matrix factored by Cholesky), and records the duality gap, objective
 * and number of active off-diagonal edges.
 *
 * @param s - The empirical covariance `[p][p]`.
 * @param alpha - The L1 penalty.
 * @param options - Tolerance on the duality gap (`1e-4`) and the inner lasso tolerance (`1e-4`).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { graphicalLassoAlgorithm } from 'mdml-engine/covariance'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const out = runAlgorithm(graphicalLassoAlgorithm([[1, 0.5], [0.5, 1]], 0.6), { maxSteps: 20 })
 * out.state.precision[0]?.[1] // => 0
 * ```
 */
export function graphicalLassoAlgorithm(
  s: readonly (readonly number[])[],
  alpha: number,
  options: GraphicalLassoAlgorithmOptions = {},
): Algorithm<GraphicalLassoState> {
  const p = s.length
  const tol = options.tol ?? 1e-4
  return {
    name: 'graphical-lasso',
    config: { alpha, tol },
    schema: channels({
      metrics: {
        gap: { description: 'the primal-dual gap', direction: 'lower' },
        objective: { description: 'the L1-penalised negative log-likelihood objective', direction: 'lower' },
        edges: { description: 'the number of active off-diagonal edges in the precision matrix' },
      },
      tensors: {
        precision: { role: 'param', description: 'the sparse precision matrix', shape: ['p', 'p'] },
        covariance: { role: 'state', description: 'the estimated covariance matrix', shape: ['p', 'p'] },
      },
    }),
    init: () => {
      const covariance = s.map((r, a) => r.map((v, b) => (a === b ? v : 0.95 * v)))
      const precision = toArray(linalg.pinv(covariance)) as number[][]
      return {
        covariance,
        precision,
        gap: Number.POSITIVE_INFINITY,
        objective: graphicalLassoObjective(s, precision, alpha),
      }
    },
    step: (st) => {
      const cov = st.covariance.map((r) => [...r])
      const prec = st.precision.map((r) => [...r])
      for (let idx = 0; idx < p; idx++) {
        const others = Array.from({ length: p }, (_, j) => j).filter((j) => j !== idx)
        const sub = others.map((a) => others.map((b) => cov[a]?.[b] ?? 0))
        const row = others.map((j) => s[idx]?.[j] ?? 0)
        const start = others.map((j) => -((prec[j]?.[idx] ?? 0) / ((prec[idx]?.[idx] ?? 0) + 1000 * Number.EPSILON)))
        // 0.5 w^T S11 w - s12^T w + alpha |w|_1 = 0.5 |L^T w - L^-1 s12|^2 + alpha |w|_1 + const, S11 = L L^T.
        const l = toArray(linalg.cholesky(sub)) as number[][]
        const design = others.map((_, a) => others.map((__, b) => l[b]?.[a] ?? 0))
        const target = toFlat(linalg.solveTriangular(l, row, { lower: true }))
        const coefs = toFlat(
          lassoCoordinateDescent(design, target, alpha, {
            x0: start,
            tol: options.enetTol ?? 1e-4,
            maxIterations: 1000,
          }).coef,
        )
        const diag =
          1 / ((cov[idx]?.[idx] ?? 0) - others.reduce((a, j, k) => a + (cov[j]?.[idx] ?? 0) * (coefs[k] ?? 0), 0))
        ;(prec[idx] as number[])[idx] = diag
        for (const [k, j] of others.entries()) {
          ;(prec[j] as number[])[idx] = -diag * (coefs[k] ?? 0)
          ;(prec[idx] as number[])[j] = -diag * (coefs[k] ?? 0)
        }
        const newRow = sub.map((r) => r.reduce((a, v, k) => a + v * (coefs[k] ?? 0), 0))
        for (const [k, j] of others.entries()) {
          ;(cov[idx] as number[])[j] = newRow[k] ?? 0
          ;(cov[j] as number[])[idx] = newRow[k] ?? 0
        }
      }
      return {
        covariance: cov,
        precision: prec,
        gap: dualGap(s, prec, alpha),
        objective: graphicalLassoObjective(s, prec, alpha),
      }
    },
    done: (st) => (Math.abs(st.gap) < tol ? 'converged' : undefined),
    observe: (st) => ({
      metrics: Number.isFinite(st.gap)
        ? { gap: st.gap, objective: st.objective, edges: activeEdges(st.precision) }
        : { edges: activeEdges(st.precision) },
      tensors: {
        precision: { role: 'param', value: st.precision.flat(), shape: [p, p] },
        covariance: { role: 'state', value: st.covariance.flat(), shape: [p, p] },
      },
    }),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as GraphicalLassoState,
  }
}

/**
 * Solve the graphical lasso from an empirical covariance matrix.
 *
 * @param s - The empirical covariance `[p, p]`.
 * @param alpha - The non-negative L1 penalty.
 * @param options - Tolerances and the iteration cap (default 100).
 * @returns The covariance, precision, final duality gap, objective and iteration count.
 * @example
 * ```ts
 * import { graphicalLasso } from 'mdml-engine/covariance'
 * graphicalLasso([[1, 0.5], [0.5, 1]], 0.6).precision[0]?.[1] // => 0
 * ```
 */
export function graphicalLasso(
  s: TensorLike,
  alpha: number,
  options: { readonly tol?: number; readonly enetTol?: number; readonly maxIter?: number } = {},
): GraphicalLassoState & { readonly nIter: number } {
  const rows = toArray(asTensor(s)) as number[][]
  if (!(alpha >= 0)) throw new MdmlError('value', 'graphicalLasso: alpha must be non-negative')
  const out = runAlgorithm(graphicalLassoAlgorithm(rows, alpha, options), { maxSteps: options.maxIter ?? 100 })
  return { ...out.state, nIter: out.steps }
}

/** Hyperparameters of the graphical lasso estimator. */
export interface GraphicalLassoHyper {
  /** The L1 penalty (default 0.01). */
  readonly alpha?: number
  /** Duality-gap convergence tolerance (default 1e-4). */
  readonly tol?: number
  /** Inner coordinate-descent lasso tolerance (default 1e-4). */
  readonly enetTol?: number
  /** Maximum coordinate-descent sweeps (default 100). */
  readonly maxIter?: number
  /** Penalties of the exposition's regularisation path (default 8, log-spaced from the largest off-diagonal entry). */
  readonly pathPenalties?: readonly number[]
}

/** Fitted parameters of the graphical lasso estimator. */
export type GraphicalLassoParams = CovarianceParams & {
  readonly gap: number
  readonly objective: number
}

/** A fitted graphical lasso estimator. */
export type FittedGraphicalLasso = IterativeFitted<GraphicalLassoParams, CovarianceVerbs>

/**
 * The graphical lasso estimator (sklearn.covariance.GraphicalLasso) on the iterative base: a sparse precision matrix,
 * traced sweep by sweep. Its exposition is a `regularisationPath` of the off-diagonal partial correlations over a
 * penalty grid (the edges of the precision-matrix graph entering as the penalty falls) and an `anomalyScores` record of
 * squared Mahalanobis distances.
 *
 * @param hyper - The penalty, tolerances, iteration cap and the regularisation path's penalties.
 * @returns The iterative estimator.
 * @index estimator Sparse precision matrix by graphical lasso.
 * @example
 * ```ts
 * import { graphicalLassoEstimator } from 'mdml-engine/covariance'
 * const x = [[1, 1, 0], [-1, -1, 0], [1, 1, 1], [-1, -1, -1]]
 * graphicalLassoEstimator({ alpha: 0.5 }).fit({ x }).params.precision[0]?.[2] // => 0
 * ```
 */
export function graphicalLassoEstimator(
  hyper: GraphicalLassoHyper = {},
): IterativeEstimator<GraphicalLassoHyper, GraphicalLassoState, GraphicalLassoParams, CovarianceVerbs> {
  return iterative<GraphicalLassoHyper, GraphicalLassoState, GraphicalLassoParams, CovarianceVerbs>({
    name: 'graphical-lasso',
    kind: 'density',
    defaults: {
      alpha: hyper.alpha ?? 0.01,
      tol: hyper.tol ?? 1e-4,
      enetTol: hyper.enetTol ?? 1e-4,
      maxIter: hyper.maxIter ?? 100,
      ...(hyper.pathPenalties === undefined ? {} : { pathPenalties: hyper.pathPenalties }),
    },
    hyperSpace: (h) =>
      dict({
        alpha: real(1e-4, 1, { log: true, default: h.alpha ?? 0.01, label: 'penalty' }),
        maxIter: int(1, 500, { default: h.maxIter ?? 100, label: 'sweeps' }),
      }),
    algorithm: (data, h) => {
      matrixView(data.x, 'graphicalLasso', { minSamples: 2 })
      const s = toArray(empiricalCovariance(data.x)) as number[][]
      const alpha = h.alpha ?? 0.01
      if (!(alpha >= 0)) throw new MdmlError('value', 'graphicalLasso: alpha must be non-negative')
      return graphicalLassoAlgorithm(s, alpha, {
        ...(h.tol === undefined ? {} : { tol: h.tol }),
        ...(h.enetTol === undefined ? {} : { enetTol: h.enetTol }),
      })
    },
    maxSteps: (h) => h.maxIter ?? 100,
    params: (st, data) => {
      const m = matrixView(data.x, 'graphicalLasso', { minSamples: 2 })
      const location = Array.from({ length: m.d }, (_, j) => {
        let t = 0
        for (let i = 0; i < m.n; i++) t += m.data[i * m.d + j] ?? 0
        return t / m.n
      })
      return {
        location,
        covariance: st.covariance,
        precision: st.precision,
        shrinkage: 0,
        gap: st.gap,
        objective: st.objective,
      }
    },
    verbs: covarianceVerbs,
    records: (st, data, h) => {
      const m = matrixView(data.x, 'graphicalLasso', { minSamples: 2 })
      const s = toArray(empiricalCovariance(data.x)) as number[][]
      const alpha = h.alpha ?? 0.01
      const opts = {
        ...(h.tol === undefined ? {} : { tol: h.tol }),
        ...(h.enetTol === undefined ? {} : { enetTol: h.enetTol }),
      }
      const location = Array.from({ length: m.d }, (_, j) => {
        let t = 0
        for (let i = 0; i < m.n; i++) t += m.data[i * m.d + j] ?? 0
        return t / m.n
      })
      const names = featureNamesOf(data, m.d)
      const pairs: [number, number][] = []
      for (let a = 0; a < m.d; a++) for (let b = a + 1; b < m.d; b++) pairs.push([a, b])
      const largest = Math.max(...pairs.map(([a, b]) => Math.abs(s[a]?.[b] ?? 0)), 1e-12)
      const penalties = h.pathPenalties ?? Array.from({ length: 8 }, (_, i) => largest * 10 ** (-2 * (i / 7)))
      const path = penalties.map((pen) => {
        const pState = runAlgorithm(graphicalLassoAlgorithm(s, pen, opts), { maxSteps: h.maxIter ?? 100 }).state
        const coef = pairs.map(
          ([a, b]) =>
            -(pState.precision[a]?.[b] ?? 0) /
            Math.sqrt((pState.precision[a]?.[a] ?? 1) * (pState.precision[b]?.[b] ?? 1)),
        )
        return { coef, gap: pState.gap }
      })
      const params: CovarianceParams = {
        location,
        covariance: st.covariance,
        precision: st.precision,
        shrinkage: 0,
      }
      return [
        regularisationPath({
          penalties: [...penalties],
          features: pairs.map(([a, b]) => `${names[a]}~${names[b]}`),
          coefficients: path.map((r) => r.coef),
          active: path.map((r) => r.coef.flatMap((v, j) => (Math.abs(v) > 1e-10 ? [j] : []))),
          gaps: path.map((r) => r.gap),
          chosen: alpha,
        }),
        mahalanobisRecord('graphicalLasso', data.x, params),
      ]
    },
    logs: { 'solver/gap': Log(), 'solver/objective': Log(), 'solver/edges': Log() },
  })
}

/**
 * The graphical lasso as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { GraphicalLasso } from 'mdml-engine/covariance'
 * const x = [[1, 1, 0], [-1, -1, 0], [1, 1, 1], [-1, -1, -1]]
 * new GraphicalLasso({ alpha: 0.5 }).fit(x).model.params.precision[0]?.[2] // => 0
 * ```
 */
export class GraphicalLasso extends ExplainedEstimatorClass<
  GraphicalLassoHyper,
  Fitted<GraphicalLassoParams> & CovarianceVerbs
> {
  /**
   * Make an unfitted graphical lasso model.
   *
   * @param hyper - The hyperparameters of {@link graphicalLassoEstimator}.
   */
  constructor(hyper: GraphicalLassoHyper = {}) {
    super(graphicalLassoEstimator(hyper))
  }
}
