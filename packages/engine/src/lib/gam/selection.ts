/**
 * Comparing and selecting generalised additive models: the log-likelihood, the AIC and BIC with the effective degrees of
 * freedom of the fit and with the smoothing-parameter-uncertainty correction of Wood, Pya and Safken (2016), where the
 * covariance of the coefficients is widened by the uncertainty in the REML smoothing parameters; and the approximate
 * F or chi-square test between nested models (mgcv's `anova.gam`). Terms are selected out of a model by the double
 * penalty (`s(x, { select: true })`, in `terms.ts`), which adds a penalty on the null space of the smoothness penalty.
 */

import { MdmlError } from '../base/index.ts'
import { dgemm, dsyev } from '../numerics/linalg/index.ts'
import { betainc, gammaincc } from '../numerics/special/index.ts'
import { key as makeKey } from '../random/index.ts'
import { fromStorage, item } from '../tensor/index.ts'
import { fitContext, type GamFitContext } from './inference.ts'
import { evaluateOuter, type OuterSetup, outerHessian } from './outer.ts'
import type { GamHyper, GamProblem, GamState } from './pirls.ts'

/** The information criteria of a fitted model. */
export interface GamInformation {
  /** The log-likelihood at the fit, with the fitted scale. */
  readonly logLik: number
  /** The effective degrees of freedom of the fit (the trace of the influence matrix). */
  readonly edf: number
  /** The effective degrees of freedom corrected for smoothing-parameter uncertainty (REML or ML selection only). */
  readonly edf2: number
  /** Parameters counted besides the coefficients: one for an estimated scale. */
  readonly extra: number
  /** `-2 logLik + 2 (edf + extra)`: the conditional AIC (smoothing parameters taken as known). */
  readonly aicConditional: number
  /** `-2 logLik + 2 (edf2 + extra)`: the AIC corrected for the uncertainty in the smoothing parameters. */
  readonly aic: number
  /** `-2 logLik + log(n) (edf2 + extra)`. */
  readonly bic: number
}

/**
 * The covariance of the coefficients corrected for smoothing-parameter uncertainty, `Vc = Vp + sum_rs d_r Vrho_rs d_s'`,
 * with `d_r = d beta / d log lambda_r = -(X'WX + S)^-1 lambda_r S_r beta` and `Vrho` the inverse Hessian of the REML
 * score in `log lambda` (eigenvalues floored, so a flat direction contributes nothing), and the effective degrees of
 * freedom `edf2 = tr(Vc X'WX) / scale` it implies (never less than the conditional ones).
 *
 * @param problem - The problem.
 * @param state - A converged state of the fit.
 * @param hyper - The hyperparameters (the method must be `reml` or `ml`).
 * @returns The corrected covariance `[p, p]` and the corrected total degrees of freedom.
 * @concept generalized-additive-models
 * @reference S. N. Wood, N. Pya and B. Safken, Smoothing parameter and model selection for general smooth models
 *   (JASA 111, 2016).
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitGamOuter, gamProblem, s, smoothingCorrection } from 'mdml-engine/gam'
 * const x = Array.from({ length: 50 }, (_, i) => [i / 49])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 7 })], iteration: 'outer' as const }
 * const problem = gamProblem(x, x.map((r, i) => Math.sin(5 * (r[0] ?? 0)) + 0.2 * Math.sin(i * 12.9898)), hyper)
 * const { state } = fitGamOuter(problem, hyper)
 * smoothingCorrection(problem, state, hyper).edf2 >= state.edf - 1e-9 // => true
 * ```
 */
export function smoothingCorrection(
  problem: GamProblem,
  state: GamState,
  hyper: GamHyper,
): { readonly vc: Float64Array; readonly edf2: number } {
  const ctx = fitContext(problem, state)
  const p = problem.design.p
  const m = problem.structure.blocks.length
  const base = { vc: ctx.vp, edf2: ctx.edfTotal }
  if (m === 0 || (hyper.method !== 'reml' && hyper.method !== 'ml')) return base
  const setup: OuterSetup = { problem, hyper, key: makeKey(0) }
  const at = evaluateOuter(setup, state.logLambda, {
    warm: { beta: Float64Array.from(state.beta), eta: Float64Array.from(state.eta) },
  })
  const H = outerHessian(setup, at)
  const { w, v } = dsyev(H, m)
  const floor = 1e-6 * Math.max(...Array.from(w, Math.abs), 1e-12)
  // Vrho = H^-1 on the eigenvalues above the floor.
  const vrho = new Float64Array(m * m)
  for (let k = 0; k < m; k++) {
    if ((w[k] ?? 0) <= floor) continue
    for (let r = 0; r < m; r++)
      for (let s = 0; s < m; s++)
        vrho[r * m + s] = (vrho[r * m + s] ?? 0) + ((v[r * m + k] ?? 0) * (v[s * m + k] ?? 0)) / (w[k] ?? 1)
  }
  // d_r = -(H^-1) lambda_r S_r beta, with the Bayesian covariance Vp = scale H^-1 giving H^-1 b = Vp b / scale.
  const D = new Float64Array(p * m)
  problem.structure.blocks.forEach((blk, r) => {
    const lam = Math.exp(state.logLambda[r] ?? 0)
    const sb = new Float64Array(p)
    for (let i = 0; i < blk.size; i++) {
      let a = 0
      for (let j = 0; j < blk.size; j++) a += (blk.matrix[i * blk.size + j] ?? 0) * (state.beta[blk.offset + j] ?? 0)
      sb[blk.offset + i] = lam * a
    }
    for (let i = 0; i < p; i++) {
      let a = 0
      for (let j = 0; j < p; j++) a += (ctx.vp[i * p + j] ?? 0) * (sb[j] ?? 0)
      D[i * m + r] = -a / ctx.scale
    }
  })
  const left = dgemm(D, vrho, p, m, m)
  const extra = dgemm(left, D, p, m, p, { transposeB: true })
  const vc = Float64Array.from(ctx.vp, (a, i) => a + (extra[i] ?? 0))
  // edf2 = tr(Vc X'WX) / scale, bounded between the conditional edf and the number of coefficients.
  const xtwx = ctx.information
  let edf2 = 0
  for (let i = 0; i < p; i++)
    for (let j = 0; j < p; j++) edf2 += ((vc[i * p + j] ?? 0) * (xtwx[j * p + i] ?? 0)) / ctx.scale
  return { vc, edf2: Math.min(Math.max(edf2, ctx.edfTotal), p) }
}

/**
 * The log-likelihood, AIC and BIC of a fitted model. The AIC counts the effective degrees of freedom `edf2` corrected for
 * the estimation of the smoothing parameters when the selection was REML or ML (the conditional AIC, which takes them as
 * known and is biased towards over-fitting, is also given), plus one for an estimated scale.
 *
 * @param problem - The problem.
 * @param state - The fitted state.
 * @param hyper - The hyperparameters it was fitted with.
 * @returns The criteria.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitGam, gamInformation, gamProblem, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 6 })] }
 * const problem = gamProblem(x, x.map((r, i) => Math.sin(5 * (r[0] ?? 0)) + 0.2 * Math.sin(i * 12.9898)), hyper)
 * const info = gamInformation(problem, fitGam(problem, hyper).state, hyper)
 * info.aic >= info.aicConditional - 1e-9 // => true
 * ```
 */
export function gamInformation(problem: GamProblem, state: GamState, hyper: GamHyper): GamInformation {
  const ctx = fitContext(problem, state)
  const n = ctx.mu.length
  const vec = (a: ArrayLike<number>) => fromStorage(Float64Array.from(a), [a.length], 'float64')
  const logLik = item(problem.family.logLikelihood(vec(problem.y), vec(ctx.mu), ctx.scale, vec(problem.prior)))
  const extra = problem.knownScale === undefined ? 1 : 0
  const { edf2 } = smoothingCorrection(problem, state, hyper)
  let count = 0
  for (const w of problem.prior) if (w > 0) count++
  return {
    logLik,
    edf: ctx.edfTotal,
    edf2,
    extra,
    aicConditional: -2 * logLik + 2 * (ctx.edfTotal + extra),
    aic: -2 * logLik + 2 * (edf2 + extra),
    bic: -2 * logLik + Math.log(count || n) * (edf2 + extra),
  }
}

/** The result of {@link compareGams}. */
export interface GamComparison {
  /** The residual deviances of the smaller and the larger model. */
  readonly deviance: readonly [number, number]
  readonly edf: readonly [number, number]
  /** The difference in effective degrees of freedom. */
  readonly df: number
  /** The deviance drop, scaled by the larger model's scale (a chi-square) or by it and `df` (an F). */
  readonly statistic: number
  readonly statisticKind: 'chi2' | 'F'
  readonly pValue: number
}

/**
 * The approximate F or chi-square test between two nested models fitted to the same data (mgcv's `anova.gam`): the
 * drop in deviance from the smaller to the larger model over the difference in effective degrees of freedom, scaled by
 * the larger model's scale estimate (an F on `df` and the larger model's residual degrees of freedom) or, when the scale
 * is known, referred to a chi-square on `df`. It is approximate: the effective degrees of freedom are estimates and
 * the distribution under the null is not exactly either; the test is conservative when the smaller model is nested by
 * penalisation only.
 *
 * @param small - The smaller model's fitted context ({@link fitContext}).
 * @param large - The larger model's.
 * @returns The comparison.
 * @throws MdmlError `value` if the models are not nested in the order given (the larger has no more effective degrees of freedom).
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, compareGams, fitContext, fitGam, gamProblem, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 80 }, (_, i) => [i / 79, ((i * 7) % 80) / 79])
 * const y = x.map((r, i) => Math.sin(5 * (r[0] ?? 0)) + 0.3 * (r[1] ?? 0) + 0.2 * Math.sin(i * 12.9898))
 * const fit = (terms: ReturnType<typeof s>[]) => {
 *   const hyper = { ...GAM_DEFAULTS, terms }
 *   const problem = gamProblem(x, y, hyper)
 *   return fitContext(problem, fitGam(problem, hyper).state)
 * }
 * compareGams(fit([s(0, { k: 6 })]), fit([s(0, { k: 6 }), s(1, { k: 5 })])).pValue < 0.05 // => true
 * ```
 */
export function compareGams(small: GamFitContext, large: GamFitContext): GamComparison {
  const df = large.edfTotal - small.edfTotal
  if (!(df > 0))
    throw new MdmlError('value', 'compareGams: the larger model must have more effective degrees of freedom')
  const drop = small.deviance - large.deviance
  if (large.scaleEstimated) {
    const statistic = drop / df / large.scale
    const d2 = large.residualDf
    const pValue = statistic <= 0 ? 1 : item(betainc(d2 / 2, df / 2, d2 / (d2 + df * statistic)))
    return {
      deviance: [small.deviance, large.deviance],
      edf: [small.edfTotal, large.edfTotal],
      df,
      statistic,
      statisticKind: 'F',
      pValue,
    }
  }
  const statistic = drop / large.scale
  return {
    deviance: [small.deviance, large.deviance],
    edf: [small.edfTotal, large.edfTotal],
    df,
    statistic,
    statisticKind: 'chi2',
    pValue: statistic <= 0 ? 1 : item(gammaincc(df / 2, statistic / 2)),
  }
}
