/**
 * Estimating a parameter of the response family together with the smooth terms: the size theta of a negative
 * binomial, the power p of a Tweedie, or any scalar the family is built from. The Laplace-approximate REML score of
 * the outer iteration is a marginal likelihood of the family parameter too (the log-likelihoods of the families are exact,
 * not quasi), so the parameter is the minimiser of the converged score, found by bounded one-dimensional
 * minimisation, each evaluation a full outer fit warm-started at the previous smoothing parameters.
 */

import type { DispersionFamily } from '../dists/families/index.ts'
import { fminbound } from '../optim/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { fitGamOuter, type OuterState } from './outer.ts'
import { type GamHyper, type GamProblem, gamProblem } from './pirls.ts'

/** The result of {@link profileFamilyParameter}. */
export interface ProfiledFit {
  /** The estimated family parameter. */
  readonly parameter: number
  /** The REML score at it. */
  readonly criterion: number
  readonly problem: GamProblem
  readonly state: OuterState
  /** The evaluations made: the parameter and its score. */
  readonly evaluations: readonly { readonly parameter: number; readonly criterion: number }[]
}

/**
 * Fit an additive model whose response family has a scalar parameter, estimating the parameter by profiling the outer
 * REML score over it (the negative binomial's size, the Tweedie power, a beta's precision).
 *
 * @param x - Covariates `[n, d]`.
 * @param y - The response.
 * @param hyper - The hyperparameters; the family is replaced by `make(parameter)`, the iteration is `outer`, the method `reml`.
 * @param options - `make` (the family at a parameter), `bounds` (the search interval), `log` (search the logarithm,
 *   default false), `tol` (on the parameter, or its logarithm; default 1e-3) and the prior `weights`.
 * @returns The fit at the estimated parameter and the evaluations made.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, profileFamilyParameter, s } from 'mdml-engine/gam'
 * import { families } from 'mdml-engine/dists'
 * const x = Array.from({ length: 80 }, (_, i) => [i / 79])
 * const y = x.map((r, i) => Math.round(3 * Math.exp(Math.sin(4 * (r[0] ?? 0))) + ((i * 7) % 5) - 2))
 * const fit = profileFamilyParameter(x, y.map((v) => Math.max(0, v)), { ...GAM_DEFAULTS, terms: [s(0, { k: 6 })] }, {
 *   make: (alpha) => families.negativeBinomial(alpha),
 *   bounds: [0.01, 5],
 *   log: true,
 * })
 * fit.parameter > 0.01 // => true
 * ```
 */
export function profileFamilyParameter(
  x: TensorLike,
  y: TensorLike,
  hyper: GamHyper,
  options: {
    readonly make: (parameter: number) => DispersionFamily
    readonly bounds: readonly [number, number]
    readonly log?: boolean
    readonly tol?: number
    readonly weights?: TensorLike
  },
): ProfiledFit {
  const evaluations: { parameter: number; criterion: number }[] = []
  let warm: number[] | undefined
  const fitAt = (parameter: number): { problem: GamProblem; state: OuterState } => {
    const h: GamHyper = {
      ...hyper,
      family: options.make(parameter),
      iteration: 'outer',
      method: 'reml',
      sp: warm === undefined ? hyper.sp : warm.map(Math.exp),
    }
    const problem = gamProblem(x, y, h, options.weights)
    const { state } = fitGamOuter(problem, h)
    warm = state.logLambda
    return { problem, state }
  }
  const toParameter = (t: number): number => (options.log === true ? Math.exp(t) : t)
  const [lo, hi] = options.bounds
  const search = options.log === true ? [Math.log(lo), Math.log(hi)] : [lo, hi]
  const result = fminbound(
    (t) => {
      const parameter = toParameter(t)
      const value = fitAt(parameter).state.criterion
      evaluations.push({ parameter, criterion: value })
      return value
    },
    { bounds: [search[0] ?? lo, search[1] ?? hi], tol: options.tol ?? 1e-3 },
  )
  const parameter = toParameter(result.x)
  const best = fitAt(parameter)
  return { parameter, criterion: best.state.criterion, problem: best.problem, state: best.state, evaluations }
}
