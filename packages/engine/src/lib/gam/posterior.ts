/**
 * Sampling the coefficient posterior of an additive model, in place of the Gaussian approximation of
 * `sampleCoefficients`. At the fitted smoothing parameters the Bayesian posterior of the coefficients is proportional to
 * `exp(l(beta) - beta' S_lambda beta / (2 scale))`: a Gaussian with covariance `(X'WX + S)^-1` to second order, but
 * skewed where the likelihood is (counts with small means, binary data), which is what a Metropolis or Hamiltonian chain
 * recovers (mgcv's `gam.mh`). The targets are `LogDensity` values of `sampling` (the exponential-family models through
 * automatic differentiation, the extended models through the analytic gradient of their likelihood); the kernels are
 * `sampling`'s random-walk Metropolis and Hamiltonian Monte Carlo, both preconditioned by the Gaussian approximation's
 * covariance, so the chain starts from a distribution already close to the target; `posteriorSampler` is the chain as a
 * traced algorithm whose state serialises.
 */

import { MdmlError } from '../base/index.ts'
import { dgemv } from '../numerics/linalg/index.ts'
import type { Key } from '../random/index.ts'
import {
  hmcKernel,
  type LogDensity,
  type MCMCKernel,
  numericTarget,
  randomWalkMetropolis,
  sampleChain,
  samplerAlgorithm,
  target as tensorTarget,
} from '../sampling/index.ts'
import { fromStorage, matmul, mul, sub, sum, type Value } from '../tensor/index.ts'
import { type Algorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import type { ExtendedModel } from './extended.ts'
import { assemblePenalty } from './penalised.ts'
import type { GamProblem, GamState } from './pirls.ts'

/**
 * The coefficient posterior of an exponential-family additive model at its fitted smoothing parameters as a log density:
 * `l(beta) - beta' S beta / (2 scale)`, with the gradient by automatic differentiation.
 *
 * @param problem - The problem.
 * @param state - The fitted state (its smoothing parameters and scale).
 * @returns The log density over the coefficients.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { GAM_DEFAULTS, fitGam, gamPosteriorTarget, gamProblem, s } from 'mdml-engine/gam'
 * import { families } from 'mdml-engine/dists'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const hyper = { ...GAM_DEFAULTS, family: families.poisson(), terms: [s(0, { k: 5 })] }
 * const problem = gamProblem(x, x.map((r) => Math.round(3 * (r[0] ?? 0) + 1)), hyper)
 * gamPosteriorTarget(problem, fitGam(problem, hyper).state).dim // => 5
 * ```
 */
export function gamPosteriorTarget(problem: GamProblem, state: GamState): LogDensity {
  const { design } = problem
  const { n, p } = design
  const S = fromStorage(assemblePenalty(problem.structure, state.logLambda), [p, p], 'float64')
  const X = fromStorage(design.X, [n, p], 'float64')
  const y = fromStorage(problem.y, [n], 'float64')
  const w = fromStorage(problem.prior, [n], 'float64')
  const scale = state.scale
  return tensorTarget((b: Value) => {
    const mu = problem.link.inverse(matmul(X, b))
    const logLik = problem.family.logLikelihood(y, mu, scale, w)
    return sub(logLik, mul(sum(mul(b, matmul(S, b))), 0.5 / scale))
  }, p)
}

/**
 * The coefficient posterior of an extended model at its smoothing parameters and aux parameters as a log density, with
 * the analytic gradient of the likelihood.
 *
 * @param model - The model.
 * @param logLambda - The log smoothing parameters.
 * @param aux - The aux parameters.
 * @returns The log density over the coefficients.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { extendedModel, extendedPosteriorTarget, linear } from 'mdml-engine/gam'
 * import { families } from 'mdml-engine/dists'
 * const m = extendedModel([[0], [1], [2], [3]], [0.2, 0.4, 0.5, 0.8], families.betaRegression(), [[linear(0)]])
 * extendedPosteriorTarget(m, [], [2]).dim // => 2
 * ```
 */
export function extendedPosteriorTarget(
  model: ExtendedModel,
  logLambda: readonly number[],
  aux: readonly number[],
): LogDensity {
  const p = model.likelihood.p
  const S = assemblePenalty(model.structure, logLambda)
  const penalised = (b: Float64Array): { value: number; grad: Float64Array } => {
    const ev = model.likelihood.evaluate(b, aux)
    const sb = dgemv(S, p, p, b)
    let q = 0
    for (let i = 0; i < p; i++) q += (b[i] ?? 0) * (sb[i] ?? 0)
    return { value: ev.logLik - 0.5 * q, grad: Float64Array.from(ev.gradient, (g, i) => g - (sb[i] ?? 0)) }
  }
  return numericTarget((b) => penalised(b).value, p, { grad: (b) => penalised(b).grad })
}

/** The options of {@link posteriorKernel}. */
export interface PosteriorKernelOptions {
  /** `hmc` (default) or `metropolis` (random walk). */
  readonly method?: 'hmc' | 'metropolis'
  /** The Gaussian approximation's covariance `[p, p]`, row-major: the mass matrix inverse and the proposal covariance. */
  readonly covariance: ArrayLike<number>
  /** The HMC step size in the preconditioned coordinates (default `0.9 p^(-1/4)`). */
  readonly stepSize?: number
  /** The leapfrog steps per transition (default 10). */
  readonly steps?: number
  /** The random walk's scale factor on the covariance (default the optimal `2.38 / sqrt(p)`). */
  readonly scale?: number
}

/**
 * A Markov kernel for the coefficient posterior, preconditioned by the Gaussian approximation's covariance: Hamiltonian
 * Monte Carlo with that covariance as the inverse mass matrix, or a random-walk Metropolis whose proposal covariance is
 * that covariance scaled by `2.38^2 / p`.
 *
 * @param target - The posterior log density.
 * @param options - The method, the covariance and the tuning.
 * @returns The kernel.
 * @throws MdmlError `shape` if the covariance does not match the target's dimension.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { posteriorKernel } from 'mdml-engine/gam'
 * import { numericTarget } from 'mdml-engine/sampling'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * posteriorKernel(t, { covariance: [1] }).name // => 'hmc'
 * ```
 */
export function posteriorKernel(target: LogDensity, options: PosteriorKernelOptions): MCMCKernel {
  const p = target.dim
  if (options.covariance.length !== p * p) {
    throw new MdmlError('shape', `posteriorKernel: a ${p} x ${p} covariance needs ${p * p} entries`)
  }
  if (options.method === 'metropolis') {
    const f = options.scale ?? 2.38 / Math.sqrt(p)
    return randomWalkMetropolis(target, {
      covariance: fromStorage(
        Float64Array.from(options.covariance, (v) => v * f * f),
        [p, p],
        'float64',
      ),
    })
  }
  return hmcKernel(target, {
    stepSize: options.stepSize ?? 0.9 * p ** -0.25,
    numIntegrationSteps: options.steps ?? 10,
    inverseMassMatrix: Float64Array.from(options.covariance),
  }) as MCMCKernel
}

/** The draws of a posterior chain and how it moved. */
export interface PosteriorDraws {
  /** The draws `[count, dim]`, row-major. */
  readonly draws: Float64Array
  readonly count: number
  readonly dim: number
  /** The fraction of transitions accepted after warm-up. */
  readonly acceptance: number
}

/**
 * Draw from the coefficient posterior: a warm-up (discarded) from the fitted coefficients, then the kept draws of the
 * kernel of {@link posteriorKernel}. The posterior mode is the starting point, so the warm-up only has to forget the
 * start's distance from the typical set, a few dozen transitions.
 *
 * @param target - The posterior log density.
 * @param start - The starting coefficients (the fit).
 * @param options - The kernel options, the key and the number of draws, warm-up and thinning.
 * @returns The draws and the acceptance rate.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { samplePosterior } from 'mdml-engine/gam'
 * import { key } from 'mdml-engine/random'
 * import { numericTarget } from 'mdml-engine/sampling'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * samplePosterior(t, [0], { covariance: [1], key: key(0), numDraws: 50 }).count // => 50
 * ```
 */
export function samplePosterior(
  target: LogDensity,
  start: ArrayLike<number>,
  options: PosteriorKernelOptions & {
    readonly key: Key
    readonly numDraws: number
    readonly numWarmup?: number
    readonly thin?: number
  },
): PosteriorDraws {
  const kernel = posteriorKernel(target, options)
  const result = sampleChain(kernel, start, {
    key: options.key,
    numDraws: options.numDraws,
    numWarmup: options.numWarmup ?? 100,
    thin: options.thin ?? 1,
  })
  const accepted = result.stats['accepted']
  let hit = 0
  if (accepted !== undefined) for (let i = 0; i < result.count; i++) hit += accepted[i] ?? 0
  return {
    draws: result.draws,
    count: result.count,
    dim: target.dim,
    acceptance: result.count === 0 ? 0 : hit / result.count,
  }
}

/**
 * The posterior chain as a traced algorithm: one transition per step (the log density, acceptance and the position on
 * the tape), with a state that serialises, so a chain pauses and resumes where it was. The paths are those of
 * `samplerAlgorithm` (the sampler is `sampling`'s; its contract is not typed here).
 *
 * @param target - The posterior log density.
 * @param start - The starting coefficients.
 * @param options - The kernel options.
 * @returns The algorithm.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { posteriorSampler } from 'mdml-engine/gam'
 * import { numericTarget } from 'mdml-engine/sampling'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * runAlgorithm(posteriorSampler(t, [0], { covariance: [1] }), { seed: 1, maxSteps: 20 }).steps // => 20
 * ```
 */
export function posteriorSampler(
  target: LogDensity,
  start: ArrayLike<number>,
  options: PosteriorKernelOptions,
): Algorithm<ReturnType<ReturnType<typeof samplerAlgorithm>['init']>> {
  const algorithm = samplerAlgorithm(posteriorKernel(target, options), start)
  return {
    ...algorithm,
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ReturnType<typeof algorithm.init>,
  }
}
