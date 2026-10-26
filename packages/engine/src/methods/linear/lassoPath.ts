/**
 * The lasso path as a particular of the iterative base: an algorithm whose step solves the next penalty of a
 * decreasing grid by coordinate descent, warm-started from the last. Each penalty's coefficients, duality gap and
 * active count are on the tape, and the regularisation-path record is read back from it.
 */

import { regularisationPath } from '../../lib/exposition/index.ts'
import { featureNamesOf, type IterativeEstimator, iterative } from '../../lib/learn/index.ts'
import { alphaGrid, coordinateDescentSolve, designOf } from '../../lib/linear/index.ts'
import { bool, dict, int } from '../../lib/space/index.ts'
import { channels, Log, metricSeries, stateFromJson, stateToJson, tensorSeries } from '../../lib/trace/index.ts'
import { type LinearParams, type LinearRegressorVerbs, linearRegressorVerbs } from './_verbs.ts'

/** The hyperparameters of {@link lassoPath}. */
export interface LassoPathHyper {
  /** The number of penalties on the grid, from the smallest that zeroes every coefficient downwards. */
  readonly nAlphas: number
  /** The grid's lower end relative to its upper end. */
  readonly eps: number
  readonly fitIntercept: boolean
  readonly tol: number
  readonly maxIter: number
}

/** The fitted parameters of {@link lassoPath}: the model at the penalty reached. */
export type LassoPathParams = LinearParams & { readonly alpha: number }

/** The state of {@link lassoPath}: the penalty reached and its solution. */
export interface LassoPathState {
  /** Index of the penalty solved. */
  readonly i: number
  readonly alpha: number
  readonly coef: number[]
  readonly intercept: number
  readonly gap: number
}

/**
 * The lasso along a penalty grid, one penalty per step. Stepping the session walks down the path; the fitted model is
 * the one at the smallest penalty reached, and any step's model can be probed.
 *
 * @example
 * ```ts
 * import { lassoPath } from 'mdml-engine/methods/linear'
 * lassoPath.with({ nAlphas: 5 }).fit({ x: [[0], [1], [2], [3]], y: [0, 1, 2, 3] }).params.coef.length // => 1
 * ```
 */
export const lassoPath: IterativeEstimator<LassoPathHyper, LassoPathState, LassoPathParams, LinearRegressorVerbs> =
  iterative<LassoPathHyper, LassoPathState, LassoPathParams, LinearRegressorVerbs>({
    name: 'linear/lasso-path',
    kind: 'regressor',
    defaults: { nAlphas: 30, eps: 1e-3, fitIntercept: true, tol: 1e-4, maxIter: 1000 },
    hyperSpace: (h) =>
      dict({
        nAlphas: int(5, 100, { default: h.nAlphas, label: 'penalties' }),
        fitIntercept: bool({ default: h.fitIntercept }),
      }),
    algorithm: (data, h) => {
      const design = designOf(data, 'lassoPath')
      const alphas = alphaGrid(design, { l1Ratio: 1, nAlphas: h.nAlphas, eps: h.eps, fitIntercept: h.fitIntercept })
      const solve = (i: number, start?: LassoPathState): LassoPathState => {
        const s = coordinateDescentSolve(design, alphas[i] ?? 0, 1, h.fitIntercept, {
          tol: h.tol,
          maxIter: h.maxIter,
          ...(start === undefined ? {} : { start: { coef: start.coef, intercept: start.intercept } }),
        })
        return { i, alpha: alphas[i] ?? 0, coef: Array.from(s.coef), intercept: s.intercept, gap: s.gap ?? Number.NaN }
      }
      return {
        name: 'lasso-path',
        config: { nAlphas: h.nAlphas, eps: h.eps, fitIntercept: h.fitIntercept },
        schema: channels({
          metrics: {
            alpha: { description: 'the penalty solved' },
            gap: { description: 'the duality gap at this penalty', direction: 'lower' },
            active: { description: 'the number of non-zero coefficients' },
          },
          tensors: { coefficients: { role: 'param', description: 'the coefficients at this penalty', shape: ['d'] } },
        }),
        init: () => solve(0),
        step: (s) => solve(s.i + 1, s),
        done: (s) => (s.i >= alphas.length - 1 ? 'converged' : undefined),
        observe: (s) => ({
          metrics: { alpha: s.alpha, gap: s.gap, active: s.coef.filter((c) => c !== 0).length },
          tensors: { coefficients: { role: 'param', value: s.coef, shape: [s.coef.length] } },
        }),
        serialise: (s) => stateToJson(s),
        deserialise: (j) => stateFromJson(j) as LassoPathState,
      }
    },
    maxSteps: (h) => h.nAlphas,
    params: (s) => ({ coef: s.coef, intercept: s.intercept, alpha: s.alpha }),
    verbs: (p) => linearRegressorVerbs(p, 'lassoPath'),
    logs: { 'solver/alpha': Log(), 'solver/gap': Log(), 'solver/coefficients': Log({ detail: 'full' }) },
    projections: {
      regularisationPath: (run, _state, data) => {
        const coefficients = tensorSeries(run, 'solver/coefficients').map((e) => [...(e.values ?? [])])
        const d = coefficients[0]?.length ?? 0
        return regularisationPath({
          penalties: metricSeries(run, 'solver/alpha').values,
          features: featureNamesOf(data, d),
          coefficients,
          active: coefficients.map((c) => c.flatMap((v, j) => (v !== 0 ? [j] : []))),
          gaps: metricSeries(run, 'solver/gap').values,
        })
      },
    },
  })
