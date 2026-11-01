/**
 * Ridge regression as a particular of the iterative base: the closed-form penalised least squares of the linear
 * library, run as a one-step algorithm so that it has a tape like any other fit.
 */

import { type IterativeEstimator, iterative } from '../../lib/learn/index.ts'
import { designOf, designPredictor, leastSquaresSolve } from '../../lib/linear/index.ts'
import { bool, dict, real } from '../../lib/space/index.ts'
import { channels, Log, stateFromJson, stateToJson } from '../../lib/trace/index.ts'
import { type LinearParams, type LinearRegressorVerbs, linearRegressorVerbs } from './_verbs.ts'

/** The hyperparameters of {@link ridge}. */
export interface RidgeHyper {
  /** The penalty on the squared coefficients (scikit-learn's Ridge scaling). */
  readonly alpha: number
  readonly fitIntercept: boolean
}

/** The state of {@link ridge}: the coefficients once solved. */
export interface RidgeState {
  readonly solved: boolean
  readonly coef: number[]
  readonly intercept: number
  readonly objective: number
}

/**
 * Ridge regression: least squares with an L2 penalty on the coefficients.
 *
 * @example
 * ```ts
 * import { ridge } from 'mdml-engine/methods/linear'
 * ridge.fit({ x: [[0], [1], [2]], y: [1, 3, 5] }).params.coef.length // => 1
 * ```
 */
export const ridge: IterativeEstimator<RidgeHyper, RidgeState, LinearParams, LinearRegressorVerbs> = iterative<
  RidgeHyper,
  RidgeState,
  LinearParams,
  LinearRegressorVerbs
>({
  name: 'linear/ridge',
  kind: 'regressor',
  defaults: { alpha: 1, fitIntercept: true },
  hyperSpace: (h) =>
    dict({
      alpha: real(1e-4, 100, { log: true, default: Math.max(h.alpha, 1e-4), label: 'penalty strength' }),
      fitIntercept: bool({ default: h.fitIntercept }),
    }),
  algorithm: (data, h) => {
    const design = designOf(data, 'ridge')
    return {
      name: 'ridge',
      config: { alpha: h.alpha, fitIntercept: h.fitIntercept },
      schema: channels({
        metrics: { objective: { description: 'half the penalised sum of squares', direction: 'lower' } },
        tensors: { coefficients: { role: 'param', description: 'the coefficients', shape: ['d'] } },
      }),
      init: () => ({ solved: false, coef: new Array<number>(design.x.d).fill(0), intercept: 0, objective: 0 }),
      step: () => {
        const s = leastSquaresSolve(design, h.alpha, h.fitIntercept, 'cholesky')
        const coef = Array.from(s.coef)
        const fit = designPredictor(design, coef, s.intercept)
        let rss = 0
        let norm = 0
        for (let i = 0; i < design.y.length; i++) rss += (design.w[i] ?? 0) * ((design.y[i] ?? 0) - (fit[i] ?? 0)) ** 2
        for (const c of coef) norm += c * c
        return { solved: true, coef, intercept: s.intercept, objective: (rss + h.alpha * norm) / 2 }
      },
      done: (s) => (s.solved ? 'converged' : undefined),
      observe: (s) => ({
        metrics: { objective: s.objective },
        tensors: { coefficients: { role: 'param', value: s.coef, shape: [s.coef.length] } },
      }),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as RidgeState,
    }
  },
  params: (s) => ({ coef: s.coef, intercept: s.intercept }),
  verbs: (p) => linearRegressorVerbs(p, 'ridge'),
  logs: { 'solver/objective': Log(), 'solver/coefficients': Log({ detail: 'full' }) },
})
