/**
 * Cox proportional hazards with smooth terms: the log hazard ratio is an additive predictor (smooths `s`, tensor
 * products, linear and factor terms, with no intercept since the baseline hazard absorbs it) and the coefficients
 * maximise the penalised partial log-likelihood of `survival`'s `coxProblem` (Breslow or Efron ties, strata), with the
 * smoothing parameters chosen by the Laplace-approximate marginal likelihood of the extended-family machinery, whose
 * likelihood in the coefficients this file supplies (value, score and observed information from `coxProblem`). The
 * baseline cumulative hazard is Breslow's estimator at the fitted linear predictor, and survival curves are
 * `exp(-H0(t) exp(eta))`.
 */

import { MdmlError } from '../base/index.ts'
import { matrixView, targetVector } from '../learn/index.ts'
import type { Key } from '../random/index.ts'
import { type CoxTies, coxData, coxProblem } from '../survival/index.ts'
import { fromStorage, type TensorLike } from '../tensor/index.ts'
import {
  differencedLikelihood,
  EXTENDED_DEFAULTS,
  type ExtendedFitResult,
  type ExtendedHyper,
  type ExtendedModel,
  fitExtended,
  modelFromDesigns,
  predictExtended,
} from './extended.ts'
import { buildDesign, type GamTerm, s } from './terms.ts'

/** A Cox model with smooth terms: an extended model plus the survival data. */
export interface CoxGamModel extends ExtendedModel {
  readonly time: Float64Array
  readonly event: Float64Array
  readonly strata: Float64Array
  readonly ties: CoxTies
}

/** The options of {@link coxGamModel}. */
export interface CoxGamOptions {
  /** Tied event times: Efron's approximation (default) or Breslow's. */
  readonly ties?: CoxTies
  /** A stratum per subject: each has its own baseline hazard. */
  readonly strata?: TensorLike
}

/**
 * Prepare a Cox model with smooth terms: the design of the terms without its intercept column, and the partial
 * likelihood of `coxProblem` as the likelihood in the coefficients.
 *
 * @param x - Covariates `[n, d]`.
 * @param time - Observed times `[n]`.
 * @param event - Event indicators `[n]` (1 event, 0 censored).
 * @param terms - The terms (default one smooth per covariate).
 * @param options - The tie handling and the strata.
 * @returns The model.
 * @throws MdmlError `shape` for inconsistent lengths; `value` without any event.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { coxGamModel, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 12 }, (_, i) => [i / 11])
 * const model = coxGamModel(x, x.map((_, i) => 1 + i), x.map((_, i) => (i % 4 === 3 ? 0 : 1)), [s(0, { k: 4 })])
 * model.likelihood.p // => 3
 * ```
 */
export function coxGamModel(
  x: TensorLike,
  time: TensorLike,
  event: TensorLike,
  terms?: readonly GamTerm[],
  options: CoxGamOptions = {},
): CoxGamModel {
  const m = matrixView(x, 'gam')
  const design = buildDesign(x, terms ?? Array.from({ length: m.d }, (_, j) => s(j)))
  const w = design.p - 1
  if (w < 1) throw new MdmlError('value', 'coxGam: needs at least one term')
  const t = targetVector(time, 'gam', m.n)
  const ev = targetVector(event, 'gam', m.n)
  const strata = options.strata === undefined ? new Float64Array(m.n) : targetVector(options.strata, 'gam', m.n)
  const ties = options.ties ?? 'efron'
  const base = modelFromDesigns(
    m.n,
    [{ name: 'logHazard', design, dropIntercept: true }],
    (columns) => {
      const cols = columns[0] as { X: Float64Array; p: number }
      const problem = coxProblem(coxData(t, ev, fromStorage(cols.X, [m.n, cols.p], 'float64'), { strata }), { ties })
      return differencedLikelihood(cols.p, (beta) => ({
        logLik: problem.logLikelihood(beta),
        gradient: Float64Array.from(problem.score(beta)),
        information: problem.information(beta),
      }))
    },
    [],
    [0],
  )
  return { ...base, time: t, event: ev, strata, ties }
}

/** The baseline cumulative hazard of one stratum: Breslow's estimator at the fitted linear predictor. */
export interface CoxBaseline {
  readonly stratum: number
  /** The distinct event times, increasing. */
  readonly time: readonly number[]
  /** `H0` after each of them. */
  readonly cumulativeHazard: readonly number[]
}

/**
 * Breslow's estimate of the baseline cumulative hazard of each stratum, `H0(t) = sum_{t_j <= t} d_j / sum_{i in R_j}
 * exp(eta_i)`, at the fitted linear predictor `eta` of the subjects.
 *
 * @param model - The model.
 * @param fit - The fit.
 * @returns One baseline per stratum.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { coxBaseline, coxGamModel, fitCoxGam, linear } from 'mdml-engine/gam'
 * const x = [[0.1], [0.9], [0.4], [1.3], [0.7], [0.2], [1.1], [0.5]]
 * const model = coxGamModel(x, [1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 0, 1, 1, 1, 0, 1], [linear(0)])
 * coxBaseline(model, fitCoxGam(model, { method: 'fixed', sp: undefined, aux: undefined, maxIter: 3, solver: undefined }))[0]?.time.length // => 6
 * ```
 */
export function coxBaseline(model: CoxGamModel, fit: ExtendedFitResult): CoxBaseline[] {
  const eta = fit.eta[0] as Float64Array
  const strata = [...new Set(model.strata)].sort((a, b) => a - b)
  return strata.map((stratum) => {
    const members = Array.from({ length: model.n }, (_, i) => i).filter((i) => model.strata[i] === stratum)
    const times = [...new Set(members.filter((i) => model.event[i] === 1).map((i) => model.time[i] as number))].sort(
      (a, b) => a - b,
    )
    let H = 0
    const cumulative = times.map((tj) => {
      let deaths = 0
      let risk = 0
      for (const i of members) {
        if ((model.time[i] ?? 0) >= tj) risk += Math.exp(eta[i] ?? 0)
        if (model.time[i] === tj && model.event[i] === 1) deaths++
      }
      H += deaths / risk
      return H
    })
    return { stratum, time: times, cumulativeHazard: cumulative }
  })
}

/**
 * Fit a Cox model with smooth terms: the extended outer iteration on the partial likelihood.
 *
 * @param model - The model.
 * @param hyper - The hyperparameters (default REML).
 * @param key - The key for a stochastic inner solver.
 * @returns The fit.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { coxGamModel, fitCoxGam, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const time = x.map((r, i) => 1 + 3 * (1 - (r[0] ?? 0)) * (1 + 0.5 * Math.sin(i * 12.9898)))
 * const model = coxGamModel(x, time, x.map((_, i) => (i % 5 === 4 ? 0 : 1)), [s(0, { k: 5 })])
 * fitCoxGam(model).converged // => true
 * ```
 */
export function fitCoxGam(model: CoxGamModel, hyper: ExtendedHyper = EXTENDED_DEFAULTS, key?: Key): ExtendedFitResult {
  return fitExtended(model, hyper, key)
}

/**
 * The survival curves `S(t | x) = exp(-H0(t) exp(eta(x)))` of new subjects at the given times.
 *
 * @param model - The model.
 * @param fit - The fit.
 * @param x - New covariates `[m, d]`.
 * @param times - The times.
 * @param stratum - The baseline to use (default the first stratum).
 * @returns `[m][times]`.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { coxGamModel, coxSurvival, fitCoxGam, linear } from 'mdml-engine/gam'
 * const x = [[0.1], [0.9], [0.4], [1.3], [0.7], [0.2], [1.1], [0.5]]
 * const model = coxGamModel(x, [1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 0, 1, 1, 1, 0, 1], [linear(0)])
 * const fit = fitCoxGam(model, { method: 'fixed', sp: undefined, aux: undefined, maxIter: 3, solver: undefined })
 * coxSurvival(model, fit, [[0.5]], [0, 100])[0]?.[0] // => 1
 * ```
 */
export function coxSurvival(
  model: CoxGamModel,
  fit: ExtendedFitResult,
  x: TensorLike,
  times: ArrayLike<number>,
  stratum?: number,
): number[][] {
  const baseline = coxBaseline(model, fit)
  const base = stratum === undefined ? baseline[0] : baseline.find((b) => b.stratum === stratum)
  if (base === undefined) throw new MdmlError('value', `coxSurvival: no stratum ${String(stratum)}`)
  const eta = predictExtended(model, fit, x).eta[0] as Float64Array
  const H = (t: number): number => {
    let v = 0
    for (let k = 0; k < base.time.length && (base.time[k] ?? Number.POSITIVE_INFINITY) <= t; k++)
      v = base.cumulativeHazard[k] ?? v
    return v
  }
  return Array.from(eta, (e) => Array.from(times, (t) => Math.exp(-H(t) * Math.exp(e))))
}
