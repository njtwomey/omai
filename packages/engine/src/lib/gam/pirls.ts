/**
 * Penalised iteratively reweighted least squares, the fitting algorithm of generalised additive models, as an
 * `Algorithm`: one iteration per step. Each step forms the working response of the exponential-family model at the
 * current linear predictor, chooses the smoothing parameters for it by GCV, UBRE or REML (the performance iteration of
 * Gu, which for a Gaussian model with identity link is exact in one step), solves the penalised normal equations and
 * halves the step if the penalised deviance rose. The tape holds the deviance, the effective degrees of freedom, the
 * criterion and the smoothing parameters at every iteration.
 */

import { MdmlError } from '../base/index.ts'
import { type DispersionFamily, gaussian, type Link } from '../dists/families/index.ts'
import { matrixView, sampleWeightsOf, targetVector } from '../learn/index.ts'
import { dpotrs } from '../numerics/linalg/index.ts'
import { fitGlm, glmSolvers, solverConfigOf } from '../optim/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { fromStorage, item, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  assemblePenalty,
  type Gram,
  gram,
  type PenalisedSolution,
  type PenaltyStructure,
  penaltyStructure,
  type SmoothingMethod,
  selectLogLambda,
  solvePenalised,
  termEdf,
} from './penalised.ts'
import { buildDesign, type GamDesign, type GamTerm } from './terms.ts'

/** The hyperparameters of a generalised additive model fit. */
export interface GamHyper {
  /** The terms of the additive predictor (default one `s` per column). */
  readonly terms: readonly GamTerm[] | undefined
  /** The exponential-family response (default Gaussian). */
  readonly family: DispersionFamily
  /** The link (default the family's canonical one). */
  readonly link: Link | undefined
  /** How the smoothing parameters are chosen; `fixed` uses `sp` (default `reml`). */
  readonly method: SmoothingMethod
  /** Smoothing parameters `lambda`, one per penalty: used as given by `fixed`, as starting values otherwise. */
  readonly sp: readonly number[] | undefined
  /** The inflation of the effective degrees of freedom in GCV and UBRE (default 1). */
  readonly gamma: number
  /** The scale, when known (default 1 for binomial, Poisson and negative binomial; estimated otherwise). */
  readonly scale: number | undefined
  /** The most iterations (default 100). */
  readonly maxIter: number
  /** Convergence tolerance on the relative change of the penalised deviance (default 1e-8). */
  readonly tol: number
  /**
   * How the coefficients are found at the smoothing parameters each iteration chose: a value of `glmSolvers.space`.
   * `irls` (the default) is one penalised reweighted least-squares step, Wood's performance iteration; any other mode
   * minimises the penalised deviance at those smoothing parameters, warm-started at the current coefficients.
   */
  readonly solver: SpaceValue
}

/** The defaults of {@link GamHyper}. */
export const GAM_DEFAULTS: GamHyper = {
  terms: undefined,
  family: gaussian(),
  link: undefined,
  method: 'reml',
  sp: undefined,
  gamma: 1,
  scale: undefined,
  maxIter: 100,
  tol: 1e-8,
  solver: solverConfigOf(glmSolvers, { kind: 'irls' }),
}

/** A model prepared on data: the design, its penalties and the response. */
export interface GamProblem {
  readonly design: GamDesign
  readonly structure: PenaltyStructure
  readonly y: Float64Array
  /** Prior weights. */
  readonly prior: Float64Array
  readonly family: DispersionFamily
  readonly link: Link
  /** The scale when it is known. */
  readonly knownScale: number | undefined
}

const KNOWN_SCALE = new Set(['binomial', 'poisson', 'negativeBinomial'])

/**
 * Prepare a model: build the design of the terms on the covariates and read the response and weights.
 *
 * @param x - Covariates `[n, d]`.
 * @param y - The response `[n]`.
 * @param hyper - The hyperparameters.
 * @param weights - Prior weights (default 1).
 * @returns The problem.
 * @throws MdmlError `shape` or `value` for inconsistent data or fewer observations than coefficients.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, gamProblem } from 'mdml-engine/gam'
 * const x = Array.from({ length: 12 }, (_, i) => [i])
 * gamProblem(x, x.map((r) => r[0] ?? 0), GAM_DEFAULTS).design.p // => 10
 * ```
 */
export function gamProblem(x: TensorLike, y: TensorLike, hyper: GamHyper, weights?: TensorLike): GamProblem {
  const m = matrixView(x, 'gam')
  const response = targetVector(y, 'gam', m.n)
  const design = buildDesign(x, hyper.terms)
  if (design.p > m.n) {
    throw new MdmlError('value', `gam: ${design.p} coefficients for ${m.n} observations; use fewer or smaller terms`)
  }
  const link = hyper.link ?? hyper.family.defaultLink
  return {
    design,
    structure: penaltyStructure(design),
    y: response,
    prior: sampleWeightsOf(weights, m.n, 'gam'),
    family: hyper.family,
    link,
    knownScale: hyper.scale ?? (KNOWN_SCALE.has(hyper.family.name) ? 1 : undefined),
  }
}

const vec = (a: ArrayLike<number>): Tensor => fromStorage(Float64Array.from(a), [a.length], 'float64')
const flat = (v: Parameters<typeof primalOf>[0]): Float64Array => Float64Array.from(toFlat(primalOf(v)))

/**
 * The working weights and response of the iteratively reweighted fit at a linear predictor:
 * `w = prior / (g'(mu)^2 V(mu))` and `z = eta + (y - mu) g'(mu)`.
 *
 * @param problem - The problem.
 * @param eta - The linear predictor `[n]`.
 * @returns The weights, the working response and the means.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, gamProblem, workingResponse } from 'mdml-engine/gam'
 * const p = gamProblem([[0], [1], [2], [3]], [1, 2, 3, 4], { ...GAM_DEFAULTS, terms: [] })
 * Array.from(workingResponse(p, [1, 1, 1, 1]).z) // => [1, 2, 3, 4]
 * ```
 */
export function workingResponse(
  problem: GamProblem,
  eta: ArrayLike<number>,
): { readonly w: Float64Array; readonly z: Float64Array; readonly mu: Float64Array } {
  const e = vec(eta)
  const mu = problem.link.inverse(e)
  const slope = flat(problem.link.derivative(mu))
  const variance = flat(problem.family.variance(mu))
  const m = flat(mu)
  const n = m.length
  const w = new Float64Array(n)
  const z = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const g = slope[i] ?? 1
    const v = variance[i] ?? 1
    const wi = (problem.prior[i] ?? 1) / (g * g * v)
    w[i] = Number.isFinite(wi) ? wi : 0
    z[i] = (eta[i] ?? 0) + ((problem.y[i] ?? 0) - (m[i] ?? 0)) * g
  }
  return { w, z, mu: m }
}

/**
 * The linear predictor the fit starts from: the link of the family's starting means.
 *
 * @param problem - The problem.
 * @returns The predictor `[n]`.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, gamProblem, startingPredictor } from 'mdml-engine/gam'
 * const p = gamProblem([[0], [1], [2], [3]], [0, 2, 2, 4], { ...GAM_DEFAULTS, terms: [] })
 * Array.from(startingPredictor(p)) // => [1, 2, 2, 3]
 * ```
 */
export function startingPredictor(problem: GamProblem): Float64Array {
  return flat(problem.link.link(problem.family.startingMu(vec(problem.y))))
}

function devianceAt(problem: GamProblem, eta: ArrayLike<number>): number {
  const mu = problem.link.inverse(vec(eta))
  return item(problem.family.deviance(vec(problem.y), mu, vec(problem.prior)))
}

/** The state of {@link gamAlgorithm}. */
export type GamState = {
  /** Iterations done. */
  readonly iteration: number
  readonly beta: number[]
  /** The linear predictor at the data. */
  readonly eta: number[]
  /** The log smoothing parameters, one per penalty. */
  readonly logLambda: number[]
  readonly deviance: number
  readonly penalisedDeviance: number
  /** The total effective degrees of freedom. */
  readonly edf: number
  readonly edfTerms: number[]
  /** The smoothing criterion at the last selection (NaN when fixed). */
  readonly criterion: number
  readonly scale: number
  readonly converged: boolean
}

/**
 * The fitted pieces at a state: the penalised solution at the state's working response and smoothing parameters, the
 * working weights, and the scale.
 *
 * @param problem - The problem.
 * @param state - A state of the fit.
 * @returns The gram, the solution, the penalty matrix and the scale.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, gamAlgorithm, gamProblem, gamSolution } from 'mdml-engine/gam'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const problem = gamProblem(x, x.map((r) => Math.sin(5 * (r[0] ?? 0))), GAM_DEFAULTS)
 * const state = runAlgorithm(gamAlgorithm(problem, GAM_DEFAULTS), { maxSteps: 20 }).state
 * gamSolution(problem, state).solution.beta.length // => problem.design.p
 * ```
 */
export function gamSolution(
  problem: GamProblem,
  state: GamState,
): { readonly g: Gram; readonly solution: PenalisedSolution; readonly S: Float64Array; readonly scale: number } {
  const { design } = problem
  const work = workingResponse(problem, state.eta)
  const g = gram(design.X, design.n, design.p, work.w, work.z)
  const S = assemblePenalty(problem.structure, state.logLambda)
  const solution = solvePenalised(g, S)
  return { g, solution, S, scale: state.scale }
}

/**
 * The influence (hat) matrix diagonal of the fitted model, `h_i = w_i x_i' (X'WX + S)^(-1) x_i` at the working weights:
 * the leverage of each observation. Its sum is the effective degrees of freedom, and `(y_i - mu_i) / (1 - h_i)` is the
 * leave-one-out residual of the working model.
 *
 * @param problem - The problem.
 * @param state - A state of the fit.
 * @returns The leverages `[n]`.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitGam, gamProblem, influenceDiagonal } from 'mdml-engine/gam'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const problem = gamProblem(x, x.map((r) => Math.cos(4 * (r[0] ?? 0))), GAM_DEFAULTS)
 * const h = influenceDiagonal(problem, fitGam(problem, GAM_DEFAULTS).state)
 * Math.abs(h.reduce((a, b) => a + b, 0) - fitGam(problem, GAM_DEFAULTS).state.edf) < 1e-8 // => true
 * ```
 */
export function influenceDiagonal(problem: GamProblem, state: GamState): Float64Array {
  const { g, solution } = gamSolution(problem, state)
  const { n, p } = g
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const wi = g.w[i] ?? 0
    if (wi === 0) continue
    const row = g.X.subarray(i * p, (i + 1) * p)
    const a = dpotrs(solution.factor, row, p)
    let h = 0
    for (let j = 0; j < p; j++) h += (row[j] ?? 0) * (a[j] ?? 0)
    out[i] = wi * h
  }
  return out
}

function pearsonScale(problem: GamProblem, mu: Float64Array, edf: number): number {
  if (problem.knownScale !== undefined) return problem.knownScale
  const variance = flat(problem.family.variance(vec(mu)))
  let chi = 0
  let n = 0
  for (let i = 0; i < mu.length; i++) {
    const wi = problem.prior[i] ?? 0
    if (wi > 0) n++
    chi += (wi * ((problem.y[i] ?? 0) - (mu[i] ?? 0)) ** 2) / (variance[i] ?? 1)
  }
  return chi / Math.max(n - edf, 1)
}

/**
 * Penalised IRLS for a generalised additive model as an algorithm, one iteration per step (see the file summary). It
 * logs `deviance`, `penalisedDeviance`, `edf`, `criterion` and `scale` as metrics and the coefficients and `logLambda`
 * as tensors; it ends when the penalised deviance and the smoothing parameters stop moving.
 *
 * @param problem - The prepared problem.
 * @param hyper - The method, starting smoothing parameters, tolerance, iteration limit and inner solver.
 * @param key - The run's key (for a stochastic inner solver).
 * @returns The algorithm.
 * @concept generalized-additive-models
 * @reference S. N. Wood, Generalized Additive Models: An Introduction with R (2nd ed., 2017), chapters 3 and 6.
 * @example
 * ```ts
 * import { GAM_DEFAULTS, gamAlgorithm, gamProblem } from 'mdml-engine/gam'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const y = x.map((r) => Math.sin(6 * (r[0] ?? 0)))
 * const problem = gamProblem(x, y, GAM_DEFAULTS)
 * const run = runAlgorithm(gamAlgorithm(problem, GAM_DEFAULTS), { maxSteps: 30 })
 * run.state.edf > 2 // => true
 * ```
 */
export function gamAlgorithm(problem: GamProblem, hyper: GamHyper, key: Key = makeKey(0)): Algorithm<GamState> {
  const { design, structure } = problem
  const n = design.n
  const nPen = structure.blocks.length
  if (hyper.sp !== undefined && hyper.sp.length !== nPen) {
    throw new MdmlError('shape', `gam: sp has ${hyper.sp.length} entries for ${nPen} penalties`)
  }
  if (hyper.method === 'fixed' && nPen > 0 && hyper.sp === undefined) {
    throw new MdmlError('value', 'gam: method "fixed" needs sp')
  }
  if (hyper.method === 'ubre' && problem.knownScale === undefined) {
    throw new MdmlError('value', 'gam: UBRE needs a known scale (a binomial or Poisson family, or the scale option)')
  }
  const start = Array.from({ length: nPen }, (_, r) => Math.log(Math.max(hyper.sp?.[r] ?? 1, 1e-300)))
  const penalisedDeviance = (eta: ArrayLike<number>, beta: ArrayLike<number>, S: Float64Array): number => {
    let pen = 0
    const p = design.p
    for (let a = 0; a < p; a++) {
      let row = 0
      for (let b = 0; b < p; b++) row += (S[a * p + b] ?? 0) * (beta[b] ?? 0)
      pen += (beta[a] ?? 0) * row
    }
    return devianceAt(problem, eta) + pen
  }
  const inner = glmSolvers.modeOf(hyper.solver).name === 'irls' ? undefined : hyper.solver
  const slotSolution = (S: Float64Array, start: ArrayLike<number>): { beta: Float64Array; eta: Float64Array } => {
    const p = design.p
    const fit = fitGlm(
      {
        x: design.X,
        n,
        d: p,
        y: problem.y,
        weights: problem.prior,
        family: problem.family,
        link: problem.link,
        penalty: { matrix: S },
        normalise: false,
        start,
      },
      inner as SpaceValue,
      { key },
    )
    const eta = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let t = 0
      for (let j = 0; j < p; j++) t += (design.X[i * p + j] ?? 0) * (fit.x[j] ?? 0)
      eta[i] = t
    }
    return { beta: fit.x, eta }
  }
  return {
    name: 'gam-pirls',
    config: {
      method: hyper.method,
      family: problem.family.name,
      link: problem.link.name,
      gamma: hyper.gamma,
      solver: hyper.solver,
    },
    schema: channels({
      metrics: {
        deviance: { description: 'the deviance of the model', direction: 'lower' },
        penalisedDeviance: { description: 'the deviance plus the wiggliness penalty', direction: 'lower' },
        edf: { description: 'the total effective degrees of freedom' },
        criterion: { description: 'the smoothing-parameter criterion at the chosen smoothing parameters' },
        scale: { description: 'the scale (dispersion) estimate' },
      },
      tensors: {
        coef: { role: 'param', description: 'the coefficients', shape: [design.p] },
        ...(nPen > 0
          ? { logLambda: { role: 'param' as const, description: 'the log smoothing parameters', shape: [nPen] } }
          : {}),
      },
    }),
    init: () => {
      const eta0 = Array.from(startingPredictor(problem))
      return {
        iteration: 0,
        beta: new Array<number>(design.p).fill(0),
        eta: eta0,
        logLambda: start,
        deviance: devianceAt(problem, eta0),
        penalisedDeviance: devianceAt(problem, eta0),
        edf: design.p,
        edfTerms: design.terms.map((t) => t.width),
        criterion: Number.NaN,
        scale: problem.knownScale ?? 1,
        converged: false,
      }
    },
    step: (s) => {
      const work = workingResponse(problem, s.eta)
      const g = gram(design.X, n, design.p, work.w, work.z)
      let logLambda = s.logLambda
      let criterion = s.criterion
      if (hyper.method !== 'fixed' && nPen > 0) {
        const sel = selectLogLambda(g, structure, s.logLambda, hyper.method, {
          gamma: hyper.gamma,
          ...(problem.knownScale === undefined ? {} : { scale: problem.knownScale }),
        })
        logLambda = sel.logLambda
        criterion = sel.value
      }
      const S = assemblePenalty(structure, logLambda)
      const sol = solvePenalised(g, S)
      // The coefficients at these smoothing parameters: the reweighted least-squares step, or a solver of the slot.
      const target = inner === undefined ? sol : slotSolution(S, s.iteration === 0 ? sol.beta : s.beta)
      let beta: ArrayLike<number> = target.beta
      let eta: ArrayLike<number> = target.eta
      let pdev = penalisedDeviance(eta, beta, S)
      if (s.iteration > 0) {
        const before = penalisedDeviance(s.eta, s.beta, S)
        let a = 1
        while (pdev > before + 1e-12 * (Math.abs(before) + 1) && a > 1e-4) {
          a /= 2
          beta = Float64Array.from(target.beta, (v, i) => (s.beta[i] ?? 0) + a * (v - (s.beta[i] ?? 0)))
          eta = Float64Array.from(target.eta, (v, i) => (s.eta[i] ?? 0) + a * (v - (s.eta[i] ?? 0)))
          pdev = penalisedDeviance(eta, beta, S)
        }
      }
      const mu = flat(problem.link.inverse(vec(eta)))
      const edf = sol.edfTotal
      const dLambda = logLambda.reduce((m, v, r) => Math.max(m, Math.abs(v - (s.logLambda[r] ?? 0))), 0)
      const converged =
        s.iteration > 0 && Math.abs(pdev - s.penalisedDeviance) < hyper.tol * (Math.abs(pdev) + 0.1) && dLambda < 0.05
      return {
        iteration: s.iteration + 1,
        beta: Array.from(beta),
        eta: Array.from(eta),
        logLambda,
        deviance: devianceAt(problem, eta),
        penalisedDeviance: pdev,
        edf,
        edfTerms: termEdf(design, sol.edf),
        criterion,
        scale: pearsonScale(problem, mu, edf),
        converged,
      }
    },
    done: (s) => (s.converged ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        deviance: s.deviance,
        penalisedDeviance: s.penalisedDeviance,
        edf: s.edf,
        ...(Number.isFinite(s.criterion) ? { criterion: s.criterion } : {}),
        scale: s.scale,
      },
      tensors: {
        coef: { role: 'param', value: s.beta, shape: [design.p] },
        ...(nPen > 0 ? { logLambda: { role: 'param' as const, value: s.logLambda, shape: [nPen] } } : {}),
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as GamState,
  }
}

/**
 * Run {@link gamAlgorithm} to the end.
 *
 * @param problem - The prepared problem.
 * @param hyper - The hyperparameters.
 * @returns The final state and whether it converged.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitGam, gamProblem } from 'mdml-engine/gam'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const problem = gamProblem(x, x.map((r) => 2 * (r[0] ?? 0) + 1), GAM_DEFAULTS)
 * fitGam(problem, GAM_DEFAULTS).converged // => true
 * ```
 */
export function fitGam(
  problem: GamProblem,
  hyper: GamHyper,
): { readonly state: GamState; readonly converged: boolean } {
  const result = runAlgorithm(gamAlgorithm(problem, hyper), { maxSteps: hyper.maxIter })
  return { state: result.state, converged: result.state.converged }
}
