/**
 * GAMLSS (Rigby and Stasinopoulos 2005): every parameter of a distribution (location, scale, shape) is an additive
 * predictor with its own smooth terms and smoothing parameters. The model is an extended model whose family has one
 * predictor per parameter (`gamlssExtended`: normal, gamma, negative binomial, Student-t, log-normal), fitted in one of
 * three modes. `newton` is the generic extended iteration (a converged penalised Newton fit at each trial smoothing
 * parameter, the Laplace marginal likelihood minimised by Newton steps). `cg` is the Cole-Green algorithm: one joint
 * penalised Newton step on all coefficients per iteration, using the cross derivatives between the parameters, with the
 * smoothing parameters re-chosen on each working model. `rs` is the Rigby-Stasinopoulos algorithm: the parameters are
 * updated in turn, each by one penalised Newton step on its own block with the others held fixed (cross derivatives
 * ignored), the smoothing parameters of the block chosen on the working model. At fixed smoothing parameters all three
 * reach the same penalised maximum. The global deviance is minus twice the log-likelihood; quantile residuals
 * (randomised for discrete responses) and their worm-plot data check the fitted distribution.
 */

import { maxOf, OmaiError } from '../base/index.ts'
import { dgemv } from '../numerics/linalg/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import { nelderMead } from '../optim/index.ts'
import { type Key, key as makeKey, uniform } from '../random/index.ts'
import { asTensor, fromStorage, item, toFlat, type Value } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import {
  EXTENDED_DEFAULTS,
  type ExtendedFitResult,
  type ExtendedHyper,
  type ExtendedModel,
  type ExtendedPath,
  type ExtendedState,
  evaluateExtended,
  extendedAlgorithm,
  extendedContract,
  invertSafely,
  predictorsAt,
} from './extended.ts'
import { assemblePenalty, type PenaltyStructure, penaltyLogDet } from './penalised.ts'

/** How a GAMLSS model is fitted. */
export type GamlssMode = 'newton' | 'cg' | 'rs'

/** The paths {@link gamlssAlgorithm} logs: those of the extended iteration and the global deviance. */
export type GamlssPath = ExtendedPath | 'gamlss/globalDeviance'

/** The contract of {@link gamlssAlgorithm}. */
export const gamlssContract: Contract<ExtendedState, GamlssPath> = defineContract({
  ...extendedContract,
  'gamlss/globalDeviance': scalar({ description: 'minus twice the log-likelihood', direction: 'lower' }),
})

/** The hyperparameters of {@link gamlssAlgorithm}. */
export interface GamlssHyper extends ExtendedHyper {
  /** `newton` (default), `cg` (Cole-Green) or `rs` (Rigby-Stasinopoulos). */
  readonly mode: GamlssMode
  /** Convergence tolerance on the relative change of the penalised log-likelihood in `cg` and `rs` (default 1e-8). */
  readonly tol: number
}

/** The defaults of {@link GamlssHyper}. */
export const GAMLSS_DEFAULTS: GamlssHyper = { ...EXTENDED_DEFAULTS, mode: 'newton', tol: 1e-8, maxIter: 100 }

const LOG_LAMBDA = { lower: -20, upper: 25 } as const

/** The penalty blocks and terms of a structure restricted to coefficients `[from, to)`, reindexed from zero. */
function restrict(structure: PenaltyStructure, from: number, to: number): { sub: PenaltyStructure; index: number[] } {
  const index: number[] = []
  structure.blocks.forEach((b, r) => {
    if (b.offset >= from && b.offset + b.size <= to) index.push(r)
  })
  const position = new Map(index.map((r, i) => [r, i]))
  return {
    sub: {
      p: to - from,
      blocks: index.map((r) => {
        const b = structure.blocks[r] as PenaltyStructure['blocks'][number]
        return { ...b, offset: b.offset - from }
      }),
      terms: structure.terms
        .filter((t) => t.offset >= from && t.offset + t.size <= to)
        .map((t) => ({ ...t, offset: t.offset - from, blocks: t.blocks.map((b) => position.get(b) as number) })),
    },
    index,
  }
}

/** `-F`: the penalised log-likelihood of `beta` at smoothing parameters `logLambda`. */
function penalisedValue(
  model: ExtendedModel,
  beta: Float64Array,
  logLambda: readonly number[],
  aux: readonly number[],
): number {
  const ev = model.likelihood.evaluate(beta, aux)
  if (!Number.isFinite(ev.logLik)) return Number.NEGATIVE_INFINITY
  const S = assemblePenalty(model.structure, logLambda)
  const Sb = dgemv(S, beta.length, beta.length, beta)
  let q = 0
  for (let i = 0; i < beta.length; i++) q += (beta[i] ?? 0) * (Sb[i] ?? 0)
  return ev.logLik - 0.5 * q
}

/**
 * The Newton target of the coefficients `[from, to)` at smoothing parameters `rho`, with the others fixed: the maximiser
 * of the quadratic approximation of the penalised log-likelihood, `(D + S)^-1 (D beta + g)` on the block (the full
 * Newton step when the block is everything), and the Laplace criterion of that working model, whose minimum over `rho`
 * chooses the smoothing parameters.
 */
function workingModel(
  p: number,
  from: number,
  to: number,
  beta: Float64Array,
  gradient: Float64Array,
  information: Float64Array,
  sub: PenaltyStructure,
  rho: readonly number[],
): { target: Float64Array; value: number } {
  const w = to - from
  const D = new Float64Array(w * w)
  for (let i = 0; i < w; i++) for (let j = 0; j < w; j++) D[i * w + j] = information[(from + i) * p + from + j] ?? 0
  // The block's Newton step also moves against the other blocks only through their fixed values (RS ignores the cross terms).
  const S = assemblePenalty(sub, rho)
  const A = Float64Array.from(D, (v, i) => v + (S[i] ?? 0))
  const { inverse, logDet } = invertSafely(A, w)
  const bk = Float64Array.from({ length: w }, (_, i) => beta[from + i] ?? 0)
  const rhs = Float64Array.from(dgemv(D, w, w, bk), (v, i) => v + (gradient[from + i] ?? 0))
  const target = dgemv(inverse, w, w, rhs)
  const move = Float64Array.from(target, (v, i) => v - (bk[i] ?? 0))
  const Dm = dgemv(D, w, w, move)
  const St = dgemv(S, w, w, target)
  let q = 0
  for (let i = 0; i < w; i++)
    q +=
      (gradient[from + i] ?? 0) * (move[i] ?? 0) -
      0.5 * (move[i] ?? 0) * (Dm[i] ?? 0) -
      0.5 * (target[i] ?? 0) * (St[i] ?? 0)
  const { logDet: logS, nullDim } = penaltyLogDet(sub, rho)
  return { target, value: -q + 0.5 * logDet - 0.5 * logS - (nullDim / 2) * Math.log(2 * Math.PI) }
}

/** Minimise the working criterion over the log smoothing parameters of a block by the simplex. */
function chooseRho(evaluate: (rho: readonly number[]) => number, start: readonly number[]): number[] {
  const m = start.length
  if (m === 0) return []
  const x0 = start.map((v) => Math.min(Math.max(v, LOG_LAMBDA.lower), LOG_LAMBDA.upper))
  const simplex = [x0, ...x0.map((v, r) => x0.map((u, c) => (c === r ? v + (v + 2 > LOG_LAMBDA.upper ? -2 : 2) : u)))]
  const result = nelderMead(
    (x) => {
      const value = evaluate(Array.from(toFlat(x as Value)))
      return asTensor(Number.isFinite(value) ? value : 1e300)
    },
    x0,
    { xatol: 1e-4, fatol: 1e-9, initialSimplex: simplex, bounds: LOG_LAMBDA, maxEvaluations: 300 * m },
  )
  return Array.from(toFlat(result.x as Value))
}

/**
 * The GAMLSS fit as an algorithm. `newton` is the extended iteration with the global deviance logged; `cg` and `rs` are
 * the two classical algorithms, one iteration per step (see the file summary), each step ending with a step-halving line
 * search on the penalised log-likelihood, so the penalised log-likelihood never falls. The state is an `ExtendedState`
 * and serialises.
 *
 * @param model - The model from `extendedModel` with a `gamlssExtended` family.
 * @param hyper - The mode, the criterion (`reml` or `fixed`), starting or fixed smoothing parameters, tolerance, cap.
 * @param key - The run's key.
 * @returns The algorithm.
 * @throws OmaiError `value` for method `fixed` without `sp`.
 * @concept distributional-regression
 * @reference R. A. Rigby and D. M. Stasinopoulos, Generalized additive models for location, scale and shape (JRSS C 54,
 *   2005).
 * @example
 * ```ts
 * import { GAMLSS_DEFAULTS, extendedModel, gamlssAlgorithm, s } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = Array.from({ length: 60 }, (_, i) => [i / 59])
 * const y = x.map((r, i) => Math.sin(4 * (r[0] ?? 0)) + (0.2 + 0.3 * (r[0] ?? 0)) * Math.sin(i * 12.9898))
 * const model = extendedModel(x, y, families.gamlssExtended('normal'), [[s(0, { k: 6 })], [s(0, { k: 4 })]])
 * runAlgorithm(gamlssAlgorithm(model, { ...GAMLSS_DEFAULTS, mode: 'rs' }), { maxSteps: 60 }).state.converged // => true
 * ```
 */
export function gamlssAlgorithm(
  model: ExtendedModel,
  hyper: GamlssHyper = GAMLSS_DEFAULTS,
  key: Key = makeKey(0),
): Algorithm<ExtendedState, GamlssPath> {
  const base = extendedAlgorithm(model, hyper, key)
  const logDeviance = (ctx: { log: (path: 'gamlss/globalDeviance', v: number) => void }, s: ExtendedState): void => {
    ctx.log('gamlss/globalDeviance', -2 * s.logLik)
  }
  if (hyper.mode === 'newton') {
    return {
      ...base,
      name: 'gamlss-newton',
      step: (s, ctx) => {
        const next = base.step(s, ctx)
        logDeviance(ctx, next)
        return next
      },
    }
  }
  const { structure, likelihood } = model
  const p = likelihood.p
  const m = structure.blocks.length
  const reml = hyper.method === 'reml'
  const rs = hyper.mode === 'rs'
  const aux = hyper.aux ?? model.aux.map((a) => a.start)
  const groups = rs
    ? model.predictors.map((pr) => ({ from: pr.offset, to: pr.offset + pr.width }))
    : [{ from: 0, to: p }]
  return {
    ...base,
    name: `gamlss-${hyper.mode}`,
    step: (s, ctx) => {
      let beta = Float64Array.from(s.beta)
      const rho = [...s.logLambda]
      const before = penalisedValue(model, beta, rho, aux)
      for (const { from, to } of groups) {
        const ev = likelihood.evaluate(beta, aux)
        const { sub, index } = restrict(structure, from, to)
        if (reml && index.length > 0) {
          const chosen = chooseRho(
            (r) => workingModel(p, from, to, beta, ev.gradient, ev.information, sub, r).value,
            index.map((r) => rho[r] ?? 0),
          )
          index.forEach((r, i) => {
            rho[r] = chosen[i] ?? rho[r] ?? 0
          })
        }
        const { target } = workingModel(
          p,
          from,
          to,
          beta,
          ev.gradient,
          ev.information,
          sub,
          index.map((r) => rho[r] ?? 0),
        )
        const startValue = penalisedValue(model, beta, rho, aux)
        let a = 1
        let next = beta
        for (let tries = 0; tries < 30; tries++) {
          next = Float64Array.from(beta)
          for (let i = 0; i < to - from; i++)
            next[from + i] = (beta[from + i] ?? 0) + a * ((target[i] ?? 0) - (beta[from + i] ?? 0))
          const value = penalisedValue(model, next, rho, aux)
          if (Number.isFinite(value) && value >= startValue - 1e-12 * (Math.abs(startValue) + 1)) break
          a /= 2
          next = beta
        }
        beta = next
      }
      const ev = likelihood.evaluate(beta, aux)
      const after = penalisedValue(model, beta, rho, aux)
      const S = assemblePenalty(structure, rho)
      const A = Float64Array.from(ev.information, (v, i) => v + (S[i] ?? 0))
      const { inverse, logDet } = invertSafely(A, p)
      const edfCoef = new Float64Array(p)
      for (let i = 0; i < p; i++) {
        let t = 0
        for (let j = 0; j < p; j++) t += (inverse[i * p + j] ?? 0) * (ev.information[j * p + i] ?? 0)
        edfCoef[i] = t
      }
      const { logDet: logS, nullDim } = penaltyLogDet(structure, rho)
      const sb = dgemv(S, p, p, beta)
      let pen = 0
      for (let i = 0; i < p; i++) pen += (beta[i] ?? 0) * (sb[i] ?? 0)
      const criterion = -ev.logLik + 0.5 * pen + 0.5 * logDet - 0.5 * logS - (nullDim / 2) * Math.log(2 * Math.PI)
      const dRho = rho.reduce((acc, v, r) => Math.max(acc, Math.abs(v - (s.logLambda[r] ?? 0))), 0)
      const converged =
        s.iteration > 0 &&
        Math.abs(after - before) <= hyper.tol * (Math.abs(after) + 0.1) &&
        (!reml || dRho < 0.05 || m === 0)
      const next: ExtendedState = {
        iteration: s.iteration + 1,
        beta: Array.from(beta),
        logLambda: rho,
        aux: [...s.aux],
        logLik: ev.logLik,
        penalisedLogLik: after,
        criterion,
        gradNorm: Math.max(0, maxOf(Array.from(ev.gradient, (g, i) => Math.abs(g - (sb[i] ?? 0))))),
        edf: edfCoef.reduce((x, y) => x + y, 0),
        edfCoef: Array.from(edfCoef),
        converged,
      }
      ctx.log('extended/logLik', next.logLik)
      ctx.log('extended/penalisedLogLik', next.penalisedLogLik)
      ctx.log('extended/criterion', next.criterion)
      ctx.log('extended/gradNorm', next.gradNorm)
      ctx.log('extended/edf', next.edf)
      ctx.log('extended/coef', next.beta)
      if (m > 0) ctx.log('extended/logLambda', next.logLambda)
      logDeviance(ctx, next)
      return next
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ExtendedState,
  }
}

/** A fitted GAMLSS model: the extended fit and the global deviance. */
export interface GamlssFit extends ExtendedFitResult {
  /** Minus twice the log-likelihood. */
  readonly globalDeviance: number
  /** The effective degrees of freedom of each term of each parameter, labelled `parameter: term`. */
  readonly edfByTerm: Readonly<Record<string, number>>
  /** The smoothing parameters `lambda` of each penalty, labelled by parameter and term. */
  readonly smoothing: Readonly<Record<string, number[]>>
}

/**
 * Fit a GAMLSS model in the chosen mode. After the iteration the state is polished by the converged penalised fit at
 * the chosen smoothing parameters (so the coefficients are the exact penalised maximum whichever mode found the
 * smoothing parameters), and the posterior covariance and effective degrees of freedom come from the penalised
 * observed information.
 *
 * @param model - The model.
 * @param hyper - The hyperparameters.
 * @param key - The run's key.
 * @returns The fit.
 * @concept distributional-regression
 * @example
 * ```ts
 * import { GAMLSS_DEFAULTS, extendedModel, fitGamlss, s } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * const x = Array.from({ length: 60 }, (_, i) => [i / 59])
 * const y = x.map((r, i) => Math.sin(4 * (r[0] ?? 0)) + (0.2 + 0.3 * (r[0] ?? 0)) * Math.sin(i * 12.9898))
 * const model = extendedModel(x, y, families.gamlssExtended('normal'), [[s(0, { k: 6 })], [s(0, { k: 4 })]])
 * fitGamlss(model, { ...GAMLSS_DEFAULTS, mode: 'cg' }).globalDeviance < 0 // => true
 * ```
 */
export function fitGamlss(
  model: ExtendedModel,
  hyper: GamlssHyper = GAMLSS_DEFAULTS,
  key: Key = makeKey(0),
): GamlssFit {
  const result = runAlgorithm(gamlssAlgorithm(model, hyper, key), { key, maxSteps: hyper.maxIter })
  const state = result.state
  const evaluation = evaluateExtended(model, state.logLambda, state.aux, {
    warm: Float64Array.from(state.beta),
    gradient: false,
    ...(hyper.solver === undefined ? {} : { solver: hyper.solver }),
    key,
  })
  const final: ExtendedState = {
    ...state,
    beta: Array.from(evaluation.fit.beta),
    logLik: evaluation.fit.logLik,
    penalisedLogLik: evaluation.fit.logLik - 0.5 * evaluation.fit.penalty,
    criterion: evaluation.value,
    edf: evaluation.edf,
    edfCoef: Array.from(evaluation.edfCoef),
  }
  const edfByTerm: Record<string, number> = {}
  for (const t of model.terms) {
    let a = 0
    for (let j = 0; j < t.width; j++) a += evaluation.edfCoef[t.offset + j] ?? 0
    edfByTerm[`${t.predictor}: ${t.label}`] = a
  }
  const smoothing: Record<string, number[]> = {}
  model.structure.blocks.forEach((b, r) => {
    const term = model.terms.find((t) => t.offset === b.offset)
    const label = `${term?.predictor ?? '?'}: ${term?.label ?? b.offset}`
    const list = smoothing[label] ?? []
    list.push(Math.exp(evaluation.logLambda[r] ?? 0))
    smoothing[label] = list
  })
  return {
    state: final,
    converged: state.converged,
    evaluation,
    covariance: evaluation.covariance,
    edfTerms: model.terms.map((t) => edfByTerm[`${t.predictor}: ${t.label}`] ?? 0),
    eta: predictorsAt(model, Float64Array.from(final.beta)),
    globalDeviance: -2 * final.logLik,
    edfByTerm,
    smoothing,
  }
}

/**
 * The (randomised) quantile residuals of a fitted distributional model, `Phi^-1(F(y_i))` with the fitted distribution
 * of each observation: normal if the model is right. For a discrete response `F(y_i)` is replaced by a uniform draw on
 * `[F(y_i - 1), F(y_i)]` (Dunn and Smyth), from the key.
 *
 * @param model - The model (its family must carry a `distribution`).
 * @param fit - The fit.
 * @param y - The response `[n]`.
 * @param key - The key for the randomisation.
 * @returns The residuals `[n]`.
 * @throws OmaiError `value` if the family has no distribution.
 * @concept distributional-regression
 * @example
 * ```ts
 * import { GAMLSS_DEFAULTS, extendedModel, fitGamlss, quantileResiduals, s } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 60 }, (_, i) => [i / 59])
 * const y = x.map((r, i) => Math.sin(4 * (r[0] ?? 0)) + 0.3 * Math.sin(i * 12.9898))
 * const model = extendedModel(x, y, families.gamlssExtended('normal'), [[s(0, { k: 6 })], []])
 * quantileResiduals(model, fitGamlss(model), y, key(0)).length // => 60
 * ```
 */
export function quantileResiduals(
  model: ExtendedModel,
  fit: ExtendedFitResult,
  y: ArrayLike<number>,
  key: Key,
  options: { readonly discrete?: boolean } = {},
): Float64Array {
  const family = model.family
  if (family?.distribution === undefined)
    throw new OmaiError('value', 'quantileResiduals: the family has no distribution')
  const n = y.length
  const dist = family.distribution(fit.eta, fit.state.aux)
  const yv = fromStorage(Float64Array.from(y), [n], 'float64')
  const upper = Array.from(toFlat(dist.cdf(yv)))
  let u = upper
  if (options.discrete === true) {
    const lower = Array.from(
      toFlat(
        dist.cdf(
          fromStorage(
            Float64Array.from(y, (v) => v - 1),
            [n],
            'float64',
          ),
        ),
      ),
    )
    const r = Array.from(toFlat(uniform(key, [n])))
    u = upper.map((hi, i) => (lower[i] ?? 0) + (r[i] ?? 0.5) * (hi - (lower[i] ?? 0)))
  }
  return Float64Array.from(u, (v) => item(ndtri(Math.min(Math.max(v, 1e-12), 1 - 1e-12))))
}

/** The data of a worm plot: the deviations of the ordered residuals from the normal quantiles, with the band. */
export interface WormPlot {
  /** The theoretical normal quantiles `[n]`, increasing. */
  readonly quantile: Float64Array
  /** The ordered residual minus its theoretical quantile `[n]`. */
  readonly deviation: Float64Array
  /** The half-width of the approximate 95% band at each quantile. */
  readonly band: Float64Array
}

/**
 * The data of a worm plot (van Buuren and Fredriks): the detrended normal QQ plot of the quantile residuals, with
 * the approximate 95% pointwise band `1.96 sqrt(p (1 - p) / n) / phi(z)`. A model that fits has its worm inside the
 * band; a level offset says the mean is off, a slope the variance, a U shape skewness, an S shape kurtosis.
 *
 * @param residuals - Quantile residuals.
 * @returns The worm.
 * @concept distributional-regression
 * @example
 * ```ts
 * import { wormPlot } from 'omai-engine/gam'
 * wormPlot([0.5, -1, 0.2, 1.1, -0.3]).quantile.length // => 5
 * ```
 */
export function wormPlot(residuals: ArrayLike<number>): WormPlot {
  const n = residuals.length
  const sorted = Float64Array.from(residuals).sort()
  const quantile = new Float64Array(n)
  const deviation = new Float64Array(n)
  const band = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const p = (i + 0.5) / n
    const z = item(ndtri(p))
    quantile[i] = z
    deviation[i] = (sorted[i] ?? 0) - z
    band[i] = (1.96 * Math.sqrt((p * (1 - p)) / n)) / (Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI))
  }
  return { quantile, deviation, band }
}
