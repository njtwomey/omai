/**
 * Outer-iteration smoothing-parameter selection for generalised additive models, as in Wood (2011) and mgcv's
 * `gam(method = "REML")`. The model is fitted to convergence at a trial `log lambda` (the inner fit: penalised IRLS, or
 * any solver of the `glmSolvers` slot), and the smoothing-parameter criterion is evaluated at the converged fit and
 * optimised in `log lambda`. For REML and ML the criterion is the Laplace-approximate marginal likelihood with the
 * observed information, its gradient is analytic (the derivative of the penalised coefficients is the implicit one,
 * `-(X'WX + S)^(-1) lambda_r S_r beta`, and the weights are differentiated in the linear predictor), and the iteration
 * is Newton with the Hessian by differences of gradients, step-halving and an eigenvalue safeguard. GCV and UBRE are
 * evaluated at the converged fits and minimised with the simplex. The performance iteration of `pirls.ts` remains the
 * other mode (`iteration: 'performance'`).
 */

import { maxOf, OmaiError } from '../base/index.ts'
import { dgemm, dgemv, dpotrf, dpotrs, dsyev } from '../numerics/linalg/index.ts'
import { fitGlm, fminbound, glmSolvers, nelderMead } from '../optim/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { asTensor, fromStorage, item, primalOf, type Tensor, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  assemblePenalty,
  gram,
  type PenalisedSolution,
  type PenaltyStructure,
  penaltyLogDet,
  solvePenalised,
  termEdf,
} from './penalised.ts'
import {
  type GamHyper,
  type GamProblem,
  type GamState,
  pearsonScale,
  startingPredictor,
  workingResponse,
} from './pirls.ts'

const vec = (a: ArrayLike<number>): Tensor => fromStorage(Float64Array.from(a), [a.length], 'float64')
const flat = (v: Value): Float64Array => Float64Array.from(toFlat(primalOf(v)))

/** The bounds `log lambda` is searched in. */
const BOUNDS = { lower: -20, upper: 25 } as const

/** The Fisher and observed (Hessian) working weights at a linear predictor. */
export function observedWeights(
  problem: GamProblem,
  eta: ArrayLike<number>,
): { readonly fisher: Float64Array; readonly observed: Float64Array } {
  const { link, family } = problem
  const mu = link.inverse(vec(eta))
  const g1 = flat(link.derivative(mu))
  const g2 = flat(link.secondDerivative(mu))
  const v = flat(family.variance(mu))
  const v1 = flat(family.varianceDerivative(mu))
  const m = flat(mu)
  const n = m.length
  const fisher = new Float64Array(n)
  const observed = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const w = (problem.prior[i] ?? 1) / ((g1[i] ?? 1) ** 2 * (v[i] ?? 1))
    const alpha = 1 + ((problem.y[i] ?? 0) - (m[i] ?? 0)) * ((v1[i] ?? 0) / (v[i] ?? 1) + (g2[i] ?? 0) / (g1[i] ?? 1))
    fisher[i] = Number.isFinite(w) ? w : 0
    observed[i] = Number.isFinite(w * alpha) ? w * alpha : 0
  }
  return { fisher, observed }
}

/** A model fitted to convergence at one set of smoothing parameters. */
export interface InnerFit {
  readonly beta: Float64Array
  readonly eta: Float64Array
  readonly deviance: number
  readonly penalisedDeviance: number
  readonly iterations: number
}

const deviance = (problem: GamProblem, eta: ArrayLike<number>): number =>
  item(problem.family.deviance(vec(problem.y), problem.link.inverse(vec(eta)), vec(problem.prior)))

function quadratic(S: Float64Array, beta: ArrayLike<number>, p: number): number {
  const sb = dgemv(S, p, p, beta)
  let a = 0
  for (let i = 0; i < p; i++) a += (beta[i] ?? 0) * (sb[i] ?? 0)
  return a
}

/**
 * The penalised fit at fixed smoothing parameters, run to convergence: penalised IRLS with step halving on the
 * penalised deviance (warm-started at `warm`), or, for another mode of the `solver` slot, `fitGlm` on the penalised
 * deviance warm-started at the coefficients.
 *
 * @param problem - The problem.
 * @param S - The assembled penalty `[p, p]`.
 * @param options - A warm start, the inner `solver`, a tolerance on the relative change of the penalised deviance
 *   (default 1e-12), the iteration cap (default 200) and the key for stochastic solvers.
 * @returns The converged fit.
 * @throws OmaiError `value` if the penalised normal equations are singular.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, assemblePenalty, gamProblem, innerFit, s } from 'omai-engine/gam'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const problem = gamProblem(x, x.map((r) => 2 * (r[0] ?? 0)), { ...GAM_DEFAULTS, terms: [s(0, { k: 5 })] })
 * innerFit(problem, assemblePenalty(problem.structure, [0])).beta.length === problem.design.p // => true
 * ```
 */
export function innerFit(
  problem: GamProblem,
  S: Float64Array,
  options: {
    readonly warm?: { readonly beta: ArrayLike<number>; readonly eta: ArrayLike<number> }
    readonly solver?: SpaceValue
    readonly tol?: number
    readonly maxIter?: number
    readonly key?: Key
  } = {},
): InnerFit {
  const { design } = problem
  const { n, p } = design
  const tol = options.tol ?? 1e-12
  const slot = options.solver !== undefined && glmSolvers.modeOf(options.solver).name !== 'irls'
  if (slot) {
    const fit = fitGlm(
      {
        x: fromStorage(design.X, [n, p], 'float64'),
        y: problem.y,
        weights: problem.prior,
        family: problem.family,
        link: problem.link,
        penalty: { matrix: fromStorage(S, [p, p], 'float64') },
        normalise: false,
        ...(options.warm === undefined ? {} : { start: Float64Array.from(options.warm.beta) }),
      },
      options.solver as SpaceValue,
      { key: options.key ?? makeKey(0) },
    )
    const fitted = Float64Array.from(toFlat(fit.x))
    const eta = dgemv(design.X, n, p, fitted)
    const d = deviance(problem, eta)
    return { beta: fitted, eta, deviance: d, penalisedDeviance: d + quadratic(S, fitted, p), iterations: fit.steps }
  }
  let eta: ArrayLike<number> = options.warm?.eta ?? startingPredictor(problem)
  let beta: ArrayLike<number> = options.warm?.beta ?? new Float64Array(p)
  let pdev = options.warm === undefined ? Number.POSITIVE_INFINITY : deviance(problem, eta) + quadratic(S, beta, p)
  let iterations = 0
  for (; iterations < (options.maxIter ?? 200); iterations++) {
    const work = workingResponse(problem, eta)
    const sol = solvePenalised(gram(design.X, n, p, work.w, work.z), S)
    let a = 1
    let nb: ArrayLike<number> = sol.beta
    let ne: ArrayLike<number> = sol.eta
    let next = deviance(problem, ne) + quadratic(S, nb, p)
    while (Number.isFinite(pdev) && !(next <= pdev + 1e-12 * (Math.abs(pdev) + 1)) && a > 1e-6) {
      a /= 2
      nb = Float64Array.from(sol.beta, (v, i) => (beta[i] ?? 0) + a * (v - (beta[i] ?? 0)))
      ne = dgemv(design.X, n, p, nb)
      next = deviance(problem, ne) + quadratic(S, nb, p)
    }
    const done = Number.isFinite(pdev) && Math.abs(next - pdev) <= tol * (Math.abs(pdev) + 1)
    beta = nb
    eta = ne
    pdev = next
    if (done) {
      iterations++
      break
    }
  }
  return {
    beta: Float64Array.from(beta),
    eta: Float64Array.from(eta),
    deviance: deviance(problem, eta),
    penalisedDeviance: pdev,
    iterations,
  }
}

/** The criterion of the outer iteration at one `log lambda`, with its gradient where it has one. */
export interface OuterEvaluation {
  readonly logLambda: readonly number[]
  readonly fit: InnerFit
  /** The criterion (lower is better). */
  readonly value: number
  /** The gradient in `log lambda` (analytic for REML and ML, absent for GCV and UBRE). */
  readonly gradient: Float64Array | undefined
  /** The scale the criterion used: the known scale, or the one profiled out. */
  readonly scaleUsed: number
  /** The total effective degrees of freedom of the converged model (Fisher weights). */
  readonly edf: number
}

/**
 * The derivative of `log |S_lambda|_+` in each `log lambda_r`: `tr(S_lambda^+ lambda_r S_r)` on the term's block.
 *
 * @param structure - The penalty structure.
 * @param logLambda - The log smoothing parameters, one per block.
 * @returns The derivatives, one per block.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { buildDesign, logDetDerivative, penaltyStructure, s } from 'omai-engine/gam'
 * const design = buildDesign([[0], [1], [2], [3], [4], [5], [6]], [s(0, { k: 5 })])
 * Math.round(logDetDerivative(penaltyStructure(design), [0])[0] ?? 0) // => 3
 * ```
 */
export function logDetDerivative(structure: PenaltyStructure, logLambda: ArrayLike<number>): Float64Array {
  const out = new Float64Array(structure.blocks.length)
  for (const t of structure.terms) {
    const S = new Float64Array(t.size * t.size)
    for (const b of t.blocks) {
      const blk = structure.blocks[b]
      if (blk === undefined) continue
      const lam = Math.exp(logLambda[b] ?? 0)
      for (let i = 0; i < S.length; i++) S[i] = (S[i] ?? 0) + lam * (blk.matrix[i] ?? 0)
    }
    const { w, v } = dsyev(S, t.size)
    const order = Array.from(w, (value, i) => ({ value, i })).sort((a, b) => b.value - a.value)
    for (const b of t.blocks) {
      const blk = structure.blocks[b]
      if (blk === undefined) continue
      const lam = Math.exp(logLambda[b] ?? 0)
      let trace = 0
      for (let r = 0; r < t.rank; r++) {
        const entry = order[r]
        if (entry === undefined) continue
        let q = 0
        for (let i = 0; i < t.size; i++)
          for (let j = 0; j < t.size; j++)
            q += (v[i * t.size + entry.i] ?? 0) * (blk.matrix[i * t.size + j] ?? 0) * (v[j * t.size + entry.i] ?? 0)
        trace += (lam * q) / Math.max(entry.value, 1e-300)
      }
      out[b] = trace
    }
  }
  return out
}

/**
 * The Laplace-approximate marginal likelihood criterion (REML or ML) and its gradient in `log lambda` at a converged fit.
 *
 * `V = -l(beta) + beta'S beta / (2 phi) + log|H| / 2 - log|S|_+ / 2 - (M/2) log(2 pi phi)` for REML (the last term's
 * `log phi` dropped for ML), with `H = X'WX + S` and `W` the observed-information weights, `M` the unpenalised dimension,
 * and `phi` the known scale or the minimiser of `V` over `phi`.
 */
function laplaceCriterion(
  problem: GamProblem,
  logLambda: readonly number[],
  fit: InnerFit,
  method: 'reml' | 'ml',
  wantGradient: boolean,
): { value: number; gradient: Float64Array | undefined; scaleUsed: number; edf: number } {
  const { design, structure, family, link } = problem
  const { n, p } = design
  const S = assemblePenalty(structure, logLambda)
  const weights = observedWeights(problem, fit.eta)
  let w = weights.observed
  const work = workingResponse(problem, fit.eta)
  const factorOf = (wt: Float64Array): ReturnType<typeof dpotrf> => {
    const g = gram(design.X, n, p, wt, work.z)
    const A = Float64Array.from(g.xtwx)
    for (let i = 0; i < A.length; i++) A[i] = (A[i] ?? 0) + (S[i] ?? 0)
    return dpotrf(A, p)
  }
  let chol = factorOf(w)
  if (chol.info !== 0) {
    // The observed information is not positive definite away from the optimum: fall back on the expected one.
    w = weights.fisher
    chol = factorOf(w)
  }
  if (chol.info !== 0) throw new OmaiError('value', 'gam: the penalised Hessian is singular')
  const { logDet, nullDim } = penaltyLogDet(structure, logLambda)
  let logDetH = 0
  for (let i = 0; i < p; i++) logDetH += 2 * Math.log(chol.factor[i * p + i] ?? 1)
  const pen = quadratic(S, fit.beta, p)
  const mu = link.inverse(vec(fit.eta))
  const logLik = (phi: number): number => item(family.logLikelihood(vec(problem.y), mu, phi, vec(problem.prior)))
  let phi = problem.knownScale ?? 1
  const phiTerm = method === 'reml' ? nullDim / 2 : 0
  if (problem.knownScale === undefined) {
    const guess = Math.max((fit.deviance + pen) / Math.max(n - nullDim, 1), 1e-300)
    const f = (lphi: number): number => -logLik(Math.exp(lphi)) + pen / (2 * Math.exp(lphi)) - phiTerm * lphi
    phi = Math.exp(fminbound(f, { bounds: [Math.log(guess) - 8, Math.log(guess) + 8], tol: 1e-10 }).x)
  }
  const value =
    -logLik(phi) +
    pen / (2 * phi) +
    0.5 * logDetH -
    0.5 * logDet -
    phiTerm * Math.log(phi) -
    (nullDim / 2) * Math.log(2 * Math.PI)
  // The effective degrees of freedom at the converged fit: tr(H^-1 X'WX) with the Fisher weights.
  const fisherSol = solvePenalised(gram(design.X, n, p, weights.fisher, work.z), S)
  if (!wantGradient) return { value, gradient: undefined, scaleUsed: phi, edf: fisherSol.edfTotal }
  // H^-1 (explicit) and the leverages h_i = x_i' H^-1 x_i.
  const Hinv = new Float64Array(p * p)
  const e = new Float64Array(p)
  for (let c = 0; c < p; c++) {
    e.fill(0)
    e[c] = 1
    const col = dpotrs(chol.factor, e, p)
    for (let r = 0; r < p; r++) Hinv[r * p + c] = col[r] ?? 0
  }
  const XH = dgemm(design.X, Hinv, n, p, p)
  const lev = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let a = 0
    for (let j = 0; j < p; j++) a += (XH[i * p + j] ?? 0) * (design.X[i * p + j] ?? 0)
    lev[i] = a
  }
  // dW/d eta by central differences of the observed (or Fisher) weights.
  const step = Float64Array.from(fit.eta, (v) => 1e-5 * (1 + Math.abs(v)))
  const plus = observedWeights(
    problem,
    Float64Array.from(fit.eta, (v, i) => v + (step[i] ?? 0)),
  )
  const minus = observedWeights(
    problem,
    Float64Array.from(fit.eta, (v, i) => v - (step[i] ?? 0)),
  )
  const useObserved = w === weights.observed
  const dw = Float64Array.from(
    fit.eta,
    (_, i) =>
      ((useObserved ? plus.observed[i] : plus.fisher[i]) ?? 0) -
      ((useObserved ? minus.observed[i] : minus.fisher[i]) ?? 0),
  ).map((v, i) => v / (2 * (step[i] ?? 1)))
  const dlogdet = logDetDerivative(structure, logLambda)
  const gradient = new Float64Array(structure.blocks.length)
  structure.blocks.forEach((blk, r) => {
    const lam = Math.exp(logLambda[r] ?? 0)
    // lambda_r S_r beta, placed in the full coefficient vector.
    const sb = new Float64Array(p)
    for (let i = 0; i < blk.size; i++) {
      let a = 0
      for (let j = 0; j < blk.size; j++) a += (blk.matrix[i * blk.size + j] ?? 0) * (fit.beta[blk.offset + j] ?? 0)
      sb[blk.offset + i] = lam * a
    }
    let betaSb = 0
    for (let i = 0; i < p; i++) betaSb += (fit.beta[i] ?? 0) * (sb[i] ?? 0)
    // d beta / d rho_r = -H^-1 sb; d eta = X d beta.
    const dBeta = dgemv(Hinv, p, p, sb).map((v) => -v)
    const dEta = dgemv(design.X, n, p, dBeta)
    let tr = 0
    for (let i = 0; i < blk.size; i++)
      for (let j = 0; j < blk.size; j++)
        tr += (Hinv[(blk.offset + i) * p + blk.offset + j] ?? 0) * lam * (blk.matrix[j * blk.size + i] ?? 0)
    for (let i = 0; i < n; i++) tr += (dw[i] ?? 0) * (dEta[i] ?? 0) * (lev[i] ?? 0)
    gradient[r] = betaSb / (2 * phi) + 0.5 * tr - 0.5 * (dlogdet[r] ?? 0)
  })
  return { value, gradient, scaleUsed: phi, edf: fisherSol.edfTotal }
}

/** The inputs the outer iteration keeps across evaluations. */
export interface OuterSetup {
  readonly problem: GamProblem
  readonly hyper: GamHyper
  readonly key: Key
}

/**
 * The criterion of the outer iteration at `log lambda`: fit to convergence (warm-started at `warm`), then score. REML
 * and ML are the Laplace-approximate marginal likelihood with its analytic gradient; GCV and UBRE are the deviance-based
 * scores of the converged fit.
 *
 * @param setup - The problem, hyperparameters and key.
 * @param logLambda - The log smoothing parameters.
 * @param options - A warm start and whether to compute the gradient (default true for REML and ML).
 * @returns The evaluation.
 * @throws OmaiError `value` for UBRE without a known scale or for a shape-constrained design.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, evaluateOuter, gamProblem, s } from 'omai-engine/gam'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 6 })], iteration: 'outer' as const }
 * const problem = gamProblem(x, x.map((r) => Math.sin(5 * (r[0] ?? 0))), hyper)
 * Number.isFinite(evaluateOuter({ problem, hyper, key: key(0) }, [0]).value) // => true
 * ```
 */
export function evaluateOuter(
  setup: OuterSetup,
  logLambda: readonly number[],
  options: { readonly warm?: Pick<InnerFit, 'beta' | 'eta'>; readonly gradient?: boolean } = {},
): OuterEvaluation {
  const { problem, hyper } = setup
  if (problem.design.constraints !== undefined) {
    throw new OmaiError('value', 'gam: the outer iteration does not support shape-constrained terms')
  }
  const S = assemblePenalty(problem.structure, logLambda)
  const fit = innerFit(problem, S, {
    ...(options.warm === undefined ? {} : { warm: options.warm }),
    solver: hyper.solver,
    key: setup.key,
  })
  if (hyper.method === 'reml' || hyper.method === 'ml') {
    const c = laplaceCriterion(problem, logLambda, fit, hyper.method, options.gradient ?? true)
    return { logLambda, fit, value: c.value, gradient: c.gradient, scaleUsed: c.scaleUsed, edf: c.edf }
  }
  const work = workingResponse(problem, fit.eta)
  const sol = solvePenalised(gram(problem.design.X, problem.design.n, problem.design.p, work.w, work.z), S)
  let n = 0
  for (const w of problem.prior) if (w > 0) n++
  let value: number
  if (hyper.method === 'gcv') {
    const denom = n - hyper.gamma * sol.edfTotal
    value = denom <= 0 ? Number.POSITIVE_INFINITY : (n * fit.deviance) / (denom * denom)
  } else if (hyper.method === 'ubre') {
    if (problem.knownScale === undefined) throw new OmaiError('value', 'gam: UBRE needs a known scale')
    value = fit.deviance / n - problem.knownScale + (2 * problem.knownScale * hyper.gamma * sol.edfTotal) / n
  } else throw new OmaiError('value', `gam: the outer iteration does not choose by "${hyper.method}"`)
  return { logLambda, fit, value, gradient: undefined, scaleUsed: problem.knownScale ?? 1, edf: sol.edfTotal }
}

/**
 * The Hessian of the REML or ML criterion in `log lambda` at a converged evaluation, by central differences of the
 * analytic gradient (each refit warm-started), symmetrised.
 *
 * @param setup - The problem, hyperparameters and key.
 * @param at - The evaluation (with its gradient) to differentiate at.
 * @param h - The difference step in `log lambda` (default 0.02).
 * @returns The Hessian `[m, m]`, row-major.
 * @throws OmaiError `value` when `at` has no gradient (GCV and UBRE).
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, evaluateOuter, gamProblem, outerHessian, s } from 'omai-engine/gam'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 6 })], iteration: 'outer' as const }
 * const problem = gamProblem(x, x.map((r, i) => Math.sin(5 * (r[0] ?? 0)) + 0.1 * Math.sin(i * 12.9898)), hyper)
 * const setup = { problem, hyper, key: key(0) }
 * outerHessian(setup, evaluateOuter(setup, [2])).length // => 1
 * ```
 */
export function outerHessian(setup: OuterSetup, at: OuterEvaluation, h = 0.02): Float64Array {
  if (at.gradient === undefined) throw new OmaiError('value', 'outerHessian: needs the analytic gradient (REML or ML)')
  const m = at.logLambda.length
  const out = new Float64Array(m * m)
  for (let s = 0; s < m; s++) {
    const move = (sign: number): Float64Array => {
      const ll = at.logLambda.map((v, r) => (r === s ? v + sign * h : v))
      return evaluateOuter(setup, ll, { warm: at.fit }).gradient as Float64Array
    }
    const up = move(1)
    const down = move(-1)
    for (let r = 0; r < m; r++) out[r * m + s] = ((up[r] ?? 0) - (down[r] ?? 0)) / (2 * h)
  }
  for (let r = 0; r < m; r++)
    for (let s = 0; s < r; s++) out[r * m + s] = out[s * m + r] = ((out[r * m + s] ?? 0) + (out[s * m + r] ?? 0)) / 2
  return out
}

/** A starting `log lambda` for each penalty: the ratio of the term's information to its penalty's trace (mgcv's rule). */
function startingLogLambda(problem: GamProblem, sp: readonly number[] | undefined): number[] {
  const { design, structure } = problem
  const { n, p } = design
  const work = workingResponse(problem, startingPredictor(problem))
  const g = gram(design.X, n, p, work.w, work.z)
  return structure.blocks.map((b, r) => {
    if (sp?.[r] !== undefined) return Math.log(Math.max(sp[r] ?? 1, 1e-300))
    let info = 0
    let pen = 0
    for (let i = 0; i < b.size; i++) {
      info += g.xtwx[(b.offset + i) * p + b.offset + i] ?? 0
      pen += b.matrix[i * b.size + i] ?? 0
    }
    return Math.min(Math.max(Math.log(Math.max(info, 1e-300) / Math.max(pen, 1e-300)), -8), 12)
  })
}

/** The state of {@link gamOuterAlgorithm}: a {@link GamState} and the criterion's gradient. */
export type OuterState = GamState & { readonly gradNorm: number }

/**
 * The outer iteration as an algorithm, one outer iteration per step. For REML and ML a step is one Newton step on
 * `log lambda`: the criterion, its analytic gradient and the finite-difference Hessian at the converged fit, the
 * Hessian's eigenvalues made positive (absolute values, floored), the step capped at 4 in `log lambda` and halved until
 * the criterion does not rise; it ends when the gradient is below `1e-5` or no step improves the criterion. For GCV and
 * UBRE the first step runs the simplex to convergence. The state is a {@link GamState}, so everything built on a fitted
 * state (inference, the estimators) applies. Logs `deviance`, `penalisedDeviance`, `edf`, `criterion`, `gradNorm` and
 * `scale` as metrics and the coefficients and `logLambda` as tensors, as the performance iteration does.
 *
 * @param problem - The prepared problem (no shape-constrained terms).
 * @param hyper - The criterion (`method`), the starting `sp`, the inner `solver` and the iteration cap.
 * @param key - The run's key.
 * @returns The algorithm.
 * @throws OmaiError `value` for method `fixed` (use the performance iteration, which fits at fixed `sp`).
 * @concept generalized-additive-models
 * @reference S. N. Wood, Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric
 *   generalized linear models (JRSS B 73, 2011).
 * @example
 * ```ts
 * import { GAM_DEFAULTS, gamOuterAlgorithm, gamProblem, s } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = Array.from({ length: 60 }, (_, i) => [i / 59])
 * const y = x.map((r) => Math.round(Math.exp(1 + Math.sin(5 * (r[0] ?? 0)))))
 * const hyper = { ...GAM_DEFAULTS, family: families.poisson(), terms: [s(0, { k: 8 })], iteration: 'outer' as const }
 * const problem = gamProblem(x, y, hyper)
 * runAlgorithm(gamOuterAlgorithm(problem, hyper), { maxSteps: 30 }).state.converged // => true
 * ```
 */
export function gamOuterAlgorithm(problem: GamProblem, hyper: GamHyper, key: Key = makeKey(0)): Algorithm<OuterState> {
  const { design, structure } = problem
  const m = structure.blocks.length
  if (hyper.method === 'fixed') {
    throw new OmaiError('value', 'gam: method "fixed" has no smoothing parameters to select (iteration performance)')
  }
  if (hyper.sp !== undefined && hyper.sp.length !== m) {
    throw new OmaiError('shape', `gam: sp has ${hyper.sp.length} entries for ${m} penalties`)
  }
  const setup: OuterSetup = { problem, hyper, key }
  const derivative = hyper.method === 'reml' || hyper.method === 'ml'
  const stateAt = (ev: OuterEvaluation, iteration: number, converged: boolean): OuterState => {
    const S = assemblePenalty(structure, ev.logLambda)
    const work = workingResponse(problem, ev.fit.eta)
    const sol: PenalisedSolution = solvePenalised(gram(design.X, design.n, design.p, work.w, work.z), S)
    const mu = flat(problem.link.inverse(vec(ev.fit.eta)))
    return {
      iteration,
      beta: Array.from(ev.fit.beta),
      eta: Array.from(ev.fit.eta),
      logLambda: [...ev.logLambda],
      deviance: ev.fit.deviance,
      penalisedDeviance: ev.fit.penalisedDeviance,
      edf: sol.edfTotal,
      edfTerms: termEdf(design, sol.edf),
      criterion: ev.value,
      scale: pearsonScale(problem, mu, sol.edfTotal),
      converged,
      gradNorm: ev.gradient === undefined ? Number.NaN : Math.max(0, maxOf(Array.from(ev.gradient, Math.abs))),
    }
  }
  const clamp = (v: readonly number[]): number[] => v.map((x) => Math.min(Math.max(x, BOUNDS.lower), BOUNDS.upper))
  return {
    name: 'gam-outer',
    config: { method: hyper.method, family: problem.family.name, link: problem.link.name, gamma: hyper.gamma },
    schema: channels({
      metrics: {
        deviance: { description: 'the deviance of the model', direction: 'lower' },
        penalisedDeviance: { description: 'the deviance plus the wiggliness penalty', direction: 'lower' },
        edf: { description: 'the total effective degrees of freedom' },
        criterion: { description: 'the smoothing-parameter criterion of the converged fit' },
        gradNorm: { description: 'the largest absolute gradient of the criterion in log lambda' },
        scale: { description: 'the scale (dispersion) estimate' },
      },
      tensors: {
        coef: { role: 'param', description: 'the coefficients', shape: [design.p] },
        ...(m > 0
          ? { logLambda: { role: 'param' as const, description: 'the log smoothing parameters', shape: [m] } }
          : {}),
      },
    }),
    init: () => {
      const start = clamp(startingLogLambda(problem, hyper.sp))
      return stateAt(evaluateOuter(setup, start, { gradient: false }), 0, m === 0)
    },
    step: (s) => {
      const current = clamp(s.logLambda)
      let next: OuterState
      if (!derivative) {
        const x0 = current
        const simplex = [x0, ...x0.map((v, r) => x0.map((u, c) => (c === r ? v + (v + 2 > BOUNDS.upper ? -2 : 2) : u)))]
        const result = nelderMead(
          (x) => asTensor(evaluateOuter(setup, Array.from(toFlat(x as Value)), { gradient: false }).value),
          x0,
          {
            xatol: 1e-5,
            fatol: 1e-10,
            initialSimplex: simplex,
            bounds: BOUNDS,
            maxEvaluations: 300 * Math.max(m, 1),
          },
        )
        next = stateAt(
          evaluateOuter(setup, clamp(Array.from(toFlat(result.x as Value))), { gradient: false }),
          s.iteration + 1,
          true,
        )
      } else {
        const warm = { beta: Float64Array.from(s.beta), eta: Float64Array.from(s.eta) }
        const ev0 = evaluateOuter(setup, current, { warm })
        const g = ev0.gradient as Float64Array
        const gradTol = 1e-5 * Math.max(1, Math.abs(ev0.value) / 100)
        if (Math.max(0, maxOf(Array.from(g, Math.abs))) < gradTol) {
          next = stateAt(ev0, s.iteration + 1, true)
        } else {
          const H = outerHessian(setup, ev0)
          const { w, v } = dsyev(H, m)
          const floor = Math.max(1e-8, 1e-8 * maxOf(Array.from(w, Math.abs)))
          const proj = new Float64Array(m)
          for (let k = 0; k < m; k++) {
            let a = 0
            for (let r = 0; r < m; r++) a += (v[r * m + k] ?? 0) * (g[r] ?? 0)
            proj[k] = -a / Math.max(Math.abs(w[k] ?? 0), floor)
          }
          const delta = new Float64Array(m)
          for (let r = 0; r < m; r++)
            for (let k = 0; k < m; k++) delta[r] = (delta[r] ?? 0) + (v[r * m + k] ?? 0) * (proj[k] ?? 0)
          const big = Math.max(0, maxOf(Array.from(delta, Math.abs)))
          const cap = big > 4 ? 4 / big : 1
          let a = cap
          let best: OuterEvaluation | undefined
          for (let tries = 0; tries < 12; tries++) {
            const trial = clamp(current.map((c, r) => c + a * (delta[r] ?? 0)))
            const ev1 = evaluateOuter(setup, trial, { warm: ev0.fit })
            if (ev1.value <= ev0.value + 1e-10 * (Math.abs(ev0.value) + 1)) {
              best = ev1
              break
            }
            a /= 2
          }
          // No step helps, or the criterion is flat to rounding (a term penalised towards its null space): done.
          const gain = best === undefined ? 0 : ev0.value - best.value
          const flat = gain <= 1e-9 * (Math.abs(ev0.value) + 1) && Math.max(0, maxOf(Array.from(g, Math.abs))) < 1e-3
          next =
            best === undefined || gain <= 1e-12 * (Math.abs(ev0.value) + 1)
              ? stateAt(ev0, s.iteration + 1, true)
              : stateAt(best, s.iteration + 1, flat)
        }
      }
      return next
    },
    done: (s) => (s.converged ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        deviance: s.deviance,
        penalisedDeviance: s.penalisedDeviance,
        edf: s.edf,
        ...(Number.isFinite(s.criterion) ? { criterion: s.criterion } : {}),
        ...(Number.isFinite(s.gradNorm) ? { gradNorm: s.gradNorm } : {}),
        scale: s.scale,
      },
      tensors: {
        coef: { role: 'param', value: s.beta, shape: [design.p] },
        ...(m > 0 ? { logLambda: { role: 'param' as const, value: s.logLambda, shape: [m] } } : {}),
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as OuterState,
  }
}

/**
 * Run {@link gamOuterAlgorithm} to the end.
 *
 * @param problem - The prepared problem.
 * @param hyper - The hyperparameters.
 * @param key - The key for a stochastic inner solver.
 * @returns The final state and whether it converged.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitGamOuter, gamProblem, s } from 'omai-engine/gam'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 6 })], iteration: 'outer' as const }
 * fitGamOuter(gamProblem(x, x.map((r) => Math.sin(5 * (r[0] ?? 0))), hyper), hyper).converged // => true
 * ```
 */
export function fitGamOuter(
  problem: GamProblem,
  hyper: GamHyper,
  key: Key = makeKey(0),
): { readonly state: OuterState; readonly converged: boolean } {
  const result = runAlgorithm(gamOuterAlgorithm(problem, hyper, key), { maxSteps: hyper.maxIter, key })
  return { state: result.state, converged: result.state.converged }
}
