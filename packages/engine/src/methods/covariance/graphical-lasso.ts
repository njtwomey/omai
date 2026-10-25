/**
 * The graphical lasso (Friedman, Hastie and Tibshirani, 2008; as in scikit-learn): a sparse precision matrix estimated
 * by an L1-penalised Gaussian likelihood, whose zeros are conditional independences between variables. Solved by
 * coordinate descent column by column as a traced algorithm, with the objective, the solver and an estimator with a
 * class wrapper.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { regularisationPath } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  featureNamesOf,
  fitRunOptions,
  matrixView,
} from '../../lib/learn/index.ts'
import { linalg } from '../../lib/numerics/index.ts'
import { lassoCoordinateDescent } from '../../lib/optim/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import { empiricalCovariance } from '../../lib/stats/index.ts'
import { asTensor, item, type TensorLike, toArray, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import { type CovarianceParams, type CovarianceVerbs, covarianceVerbs, mahalanobisRecord } from './shrinkage.ts'

/** The state of the graphical lasso: the covariance and precision estimates and the duality gap. */
export interface GraphicalLassoState {
  readonly covariance: number[][]
  readonly precision: number[][]
  readonly gap: number
  readonly objective: number
}

/**
 * The graphical lasso objective -log det(Theta) + tr(S Theta) + alpha |Theta|_1 off the diagonal (sklearn's).
 *
 * @param s - The empirical covariance.
 * @param precision - The precision.
 * @param alpha - The penalty.
 * @returns The objective.
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

/**
 * The graphical lasso (Friedman, Hastie and Tibshirani 2008; sklearn's `graphical_lasso` with coordinate descent) as a
 * traced algorithm: each step sweeps the columns, solving each column's lasso in the Gram form through `optim`'s
 * lasso coordinate descent (the Gram matrix factored by Cholesky), and records the duality gap.
 *
 * @param s - The empirical covariance `[p][p]`.
 * @param alpha - The L1 penalty.
 * @param options - Tolerance on the duality gap (1e-4) and the inner lasso tolerance (1e-4).
 * @returns The algorithm.
 */
export function graphicalLassoAlgorithm(
  s: readonly (readonly number[])[],
  alpha: number,
  options: { readonly tol?: number; readonly enetTol?: number } = {},
): Algorithm<GraphicalLassoState> {
  const p = s.length
  const tol = options.tol ?? 1e-4
  return {
    name: 'graphical-lasso',
    config: { alpha, tol },
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
    observe: (st) => ({ metrics: Number.isFinite(st.gap) ? { gap: st.gap, objective: st.objective } : {} }),
  }
}

/**
 * The graphical lasso from an empirical covariance.
 *
 * @param s - The empirical covariance.
 * @param alpha - The penalty.
 * @param options - Tolerances and the iteration cap (100).
 * @returns The covariance, precision and final gap.
 */
export function graphicalLasso(
  s: TensorLike,
  alpha: number,
  options: { readonly tol?: number; readonly enetTol?: number; readonly maxIter?: number } = {},
): GraphicalLassoState & { readonly nIter: number } {
  const rows = toArray(asTensor(s)) as number[][]
  if (!(alpha >= 0)) throw new WhiteboxError('value', 'graphicalLasso: alpha must be non-negative')
  const out = runAlgorithm(graphicalLassoAlgorithm(rows, alpha, options), { maxSteps: options.maxIter ?? 100 })
  return { ...out.state, nIter: out.steps }
}

/** Hyperparameters of the graphical lasso estimator. */
export type GraphicalLassoHyper = {
  /** The L1 penalty (default 0.01). */
  readonly alpha?: number
  readonly tol?: number
  readonly enetTol?: number
  readonly maxIter?: number
  /** Penalties of the exposition's path (default 8, log spaced from the largest useful penalty). */
  readonly pathPenalties?: readonly number[]
}

/** A fitted graphical lasso. */
export type FittedGraphicalLasso = ExplainedFitted<CovarianceParams & { readonly nIter: number }, CovarianceVerbs>

/**
 * The graphical lasso estimator (sklearn.covariance.GraphicalLasso): a sparse precision matrix, traced sweep by sweep.
 * Its exposition is a `regularisationPath` of the off-diagonal partial correlations over a penalty grid (the edges
 * of the precision-matrix graph entering as the penalty falls) and an `anomalyScores` record of squared Mahalanobis
 * distances. TODO: cross-validated selection (GraphicalLassoCV).
 *
 * @param hyper - The penalty, tolerances, iteration cap and the path's penalties.
 * @returns The estimator.
 */
export function graphicalLassoEstimator(
  hyper: GraphicalLassoHyper = {},
): Estimator<GraphicalLassoHyper, FittedGraphicalLasso> {
  return defineExplainedEstimator<GraphicalLassoHyper, CovarianceParams & { readonly nIter: number }, CovarianceVerbs>(
    {
      name: 'graphical-lasso',
      kind: 'density',
      hyperSpace: (h) =>
        dict({
          alpha: real(1e-4, 1, { log: true, default: h.alpha ?? 0.01, label: 'penalty' }),
          maxIter: int(1, 500, { default: h.maxIter ?? 100, label: 'sweeps' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'graphicalLasso', { minSamples: 2 })
        const s = toArray(empiricalCovariance(data.x)) as number[][]
        const alpha = h.alpha ?? 0.01
        const opts = {
          ...(h.tol === undefined ? {} : { tol: h.tol }),
          ...(h.enetTol === undefined ? {} : { enetTol: h.enetTol }),
        }
        const out = runAlgorithm(graphicalLassoAlgorithm(s, alpha, opts), {
          ...fitRunOptions(options, 'graphical-lasso'),
          maxSteps: h.maxIter ?? 100,
        })
        const location = Array.from({ length: m.d }, (_, j) => {
          let t = 0
          for (let i = 0; i < m.n; i++) t += m.data[i * m.d + j] ?? 0
          return t / m.n
        })
        const params = {
          location,
          covariance: out.state.covariance,
          precision: out.state.precision,
          shrinkage: 0,
          nIter: out.steps,
        }
        const names = featureNamesOf(data, m.d)
        const pairs: [number, number][] = []
        for (let a = 0; a < m.d; a++) for (let b = a + 1; b < m.d; b++) pairs.push([a, b])
        const largest = Math.max(...pairs.map(([a, b]) => Math.abs(s[a]?.[b] ?? 0)), 1e-12)
        const penalties = h.pathPenalties ?? Array.from({ length: 8 }, (_, i) => largest * 10 ** (-2 * (i / 7)))
        const path = penalties.map((pen) => {
          const st = runAlgorithm(graphicalLassoAlgorithm(s, pen, opts), { maxSteps: h.maxIter ?? 100 }).state
          const coef = pairs.map(
            ([a, b]) =>
              -(st.precision[a]?.[b] ?? 0) / Math.sqrt((st.precision[a]?.[a] ?? 1) * (st.precision[b]?.[b] ?? 1)),
          )
          return { coef, gap: st.gap }
        })
        return {
          params,
          exposition: [
            regularisationPath({
              penalties: [...penalties],
              features: pairs.map(([a, b]) => `${names[a]}~${names[b]}`),
              coefficients: path.map((r) => r.coef),
              active: path.map((r) => r.coef.flatMap((v, j) => (Math.abs(v) > 1e-10 ? [j] : []))),
              gaps: path.map((r) => r.gap),
              chosen: alpha,
            }),
            mahalanobisRecord('graphicalLasso', data.x, params),
          ],
        }
      },
      methods: covarianceVerbs,
    },
    hyper,
  )
}

/** The graphical lasso as a class. */
export class GraphicalLasso extends EstimatorClass<GraphicalLassoHyper, FittedGraphicalLasso> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link graphicalLassoEstimator}.
   */
  constructor(hyper: GraphicalLassoHyper = {}) {
    super(graphicalLassoEstimator(hyper))
  }
}
