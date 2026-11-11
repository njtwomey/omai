/**
 * Approximate message passing (Donoho, Maleki and Montanari 2009; Bayati and Montanari 2011) for the linear model
 * y = A x + w with a known matrix A, noise w ~ N(0, sigma^2 I) and an independent prior on every entry of x: the
 * message-passing algorithm that results from belief propagation on the dense bipartite graph of the rows and columns of A
 * once each message is replaced by a Gaussian and the messages of a node are replaced by their average. Each iteration
 * forms the residual with a memory (Onsager) correction `z = y - A x + z <eta'> / delta`, treats `r = x + A' z` as the entry
 * plus Gaussian noise of variance `tau^2 = |z|^2 / M`, and denoises it entry by entry with the posterior mean under the
 * prior (`eta`). For a matrix with independent entries of variance 1 / M the noise level of every iteration is predicted by
 * a scalar recursion, the state evolution `tau_{t+1}^2 = sigma^2 + mmse(tau_t^2) / delta` (here `delta = M / N`), and the
 * run follows it as M and N grow. The denoisers are the posterior means of a Gaussian prior and of a Bernoulli-Gaussian
 * (sparse) prior; the state evolution of a prior and the algorithm are given side by side, so a run puts its own noise
 * level and the predicted one on the tape.
 *
 * @index module Approximate message passing for linear models with iid priors, with its state evolution.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { dgemv, dnrm2 } from '../../numerics/linalg/index.ts'
import { gaussLegendre } from '../../numerics/quadrature/index.ts'
import { foldIn, type Key, normal, uniform } from '../../random/index.ts'
import { choice, dict, int, real } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  channels,
  defineContract,
  defineMethod,
  frame,
  type Method,
  message,
  type RunStatus,
  scalar,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'
import type { MpSnapshot } from '../engine/engine.ts'

/** The prior of every entry of x: a Gaussian of mean zero, or a Bernoulli-Gaussian (zero, or a Gaussian, with a given probability). */
export type AmpPrior =
  | { readonly kind: 'gaussian'; readonly variance: number }
  | { readonly kind: 'bernoulli-gaussian'; readonly sparsity: number; readonly variance: number }

function checkPrior(prior: AmpPrior): void {
  if (!(prior.variance > 0)) throw new OmaiError('value', 'AMP: the prior variance must be positive')
  if (prior.kind === 'bernoulli-gaussian' && !(prior.sparsity > 0 && prior.sparsity <= 1))
    throw new OmaiError('value', 'AMP: the sparsity must be in (0, 1]')
}

const GL16 = ((): { readonly x: readonly number[]; readonly w: readonly number[] } => {
  const r = gaussLegendre(16)
  return { x: toFlat(r.nodes), w: toFlat(r.weights) }
})()

/** The variance of one entry under the prior. */
const priorSecondMoment = (prior: AmpPrior): number =>
  prior.kind === 'gaussian' ? prior.variance : prior.sparsity * prior.variance

/**
 * The denoiser: the posterior mean of an entry given `r = x + tau Z`, and its derivative in r, for an iid prior.
 *
 * @param prior - The prior of the entry.
 * @param r - The noisy observation.
 * @param tau2 - The variance of its noise.
 * @returns The posterior mean and the derivative of the posterior mean with respect to `r`.
 * @throws OmaiError with code `value` for a non-positive variance or sparsity outside (0, 1].
 * @concept approximate-message-passing
 * @index function The AMP denoiser: posterior mean and derivative for a Gaussian or Bernoulli-Gaussian prior.
 * @example
 * ```ts
 * import { ampDenoiser } from 'omai-engine/message-passing'
 * // a Gaussian prior of variance 1 and noise of variance 1 halves the observation
 * ampDenoiser({ kind: 'gaussian', variance: 1 }, 2, 1) // => { mean: 1, derivative: 0.5 }
 * ```
 */
export function ampDenoiser(prior: AmpPrior, r: number, tau2: number): { mean: number; derivative: number } {
  checkPrior(prior)
  if (!(tau2 > 0)) throw new OmaiError('value', 'AMP: the noise variance must be positive')
  const gain = prior.variance / (prior.variance + tau2)
  if (prior.kind === 'gaussian') return { mean: gain * r, derivative: gain }
  // The posterior probability that the entry is nonzero, from the odds of the two components of the marginal of r.
  const logOdds =
    Math.log(prior.sparsity / (1 - prior.sparsity + 1e-300)) +
    0.5 * Math.log(tau2 / (prior.variance + tau2)) +
    0.5 * r * r * (1 / tau2 - 1 / (prior.variance + tau2))
  const pi = prior.sparsity >= 1 ? 1 : 1 / (1 + Math.exp(-logOdds))
  return {
    mean: pi * gain * r,
    derivative: pi * gain + pi * (1 - pi) * gain * r * r * (1 / tau2 - 1 / (prior.variance + tau2)),
  }
}

/**
 * The minimum mean squared error of one entry given `x + tau Z`: `E[(eta(X + tau Z) - X)^2]`, in closed form for the Gaussian
 * prior and by Gauss-Hermite quadrature over the two components of the marginal of `r` for the Bernoulli-Gaussian prior
 * (the posterior mean's error is the prior's variance less the variance of the posterior mean).
 *
 * @param prior - The prior.
 * @param tau2 - The noise variance.
 * @returns The mean squared error of the Bayes-optimal denoiser.
 * @concept approximate-message-passing
 * @example
 * ```ts
 * import { ampMmse } from 'omai-engine/message-passing'
 * ampMmse({ kind: 'gaussian', variance: 1 }, 1) // => 0.5
 * ```
 */
export function ampMmse(prior: AmpPrior, tau2: number): number {
  checkPrior(prior)
  if (prior.kind === 'gaussian') return (prior.variance * tau2) / (prior.variance + tau2)
  // E[Var(X | y)] over the marginal of y = x + tau z: the integrand is smooth and positive (no cancellation as the
  // noise falls), and the panels widen geometrically from zero so the transition of the posterior at the scale of tau is resolved.
  const g = prior.variance / (prior.variance + tau2)
  const tau = Math.sqrt(tau2)
  const spread = Math.sqrt(prior.variance + tau2)
  const edges = [0]
  for (let e = tau / 4; e < 14 * spread; e *= 1.6) edges.push(e)
  edges.push(14 * spread)
  const normal = (y: number, variance: number): number =>
    Math.exp((-0.5 * y * y) / variance) / Math.sqrt(2 * Math.PI * variance)
  let total = 0
  for (let k = 0; k + 1 < edges.length; k++) {
    const lo = edges[k] as number
    const hi = edges[k + 1] as number
    for (const [i, node] of GL16.x.entries()) {
      const y = lo + ((node + 1) / 2) * (hi - lo)
      const { mean: m } = ampDenoiser(prior, y, tau2)
      const pi = g * y === 0 ? 1 : m / (g * y)
      const variance = pi * g * tau2 + pi * (1 - pi) * g * g * y * y
      const density = (1 - prior.sparsity) * normal(y, tau2) + prior.sparsity * normal(y, prior.variance + tau2)
      total += (((GL16.w[i] ?? 0) * (hi - lo)) / 2) * density * variance
    }
  }
  return 2 * total
}

/**
 * The state evolution of AMP (Bayati and Montanari 2011): the noise variance of the effective Gaussian channel at each
 * iteration, `tau_0^2 = sigma^2 + E[X^2] / delta` and `tau_{t+1}^2 = sigma^2 + mmse(tau_t^2) / delta`, with `delta = M / N`
 * the number of measurements per unknown. It is what a run of AMP on a matrix of independent entries of variance 1 / M
 * follows as M and N grow, and its fixed point gives the error AMP reaches.
 *
 * @param prior - The prior of the entries.
 * @param delta - Measurements per unknown, M / N.
 * @param noiseVariance - The variance of the measurement noise.
 * @param iterations - The number of iterations to predict.
 * @returns `tau^2` for iterations 0 to `iterations`.
 * @throws OmaiError with code `value` for a non-positive delta or a negative noise variance.
 * @concept approximate-message-passing
 * @reference Bayati, M. and Montanari, A. (2011). The dynamics of message passing on dense graphs, with applications to compressed sensing. IEEE Transactions on Information Theory 57(2).
 * @index function The state evolution of AMP: the predicted noise level of every iteration.
 * @example
 * ```ts
 * import { ampStateEvolution } from 'omai-engine/message-passing'
 * // a Gaussian prior of variance 1, delta 1/2 and noise 0.1: tau^2 goes from 2.1 to the fixed point of tau^2 = 0.1 + 2 tau^2 / (1 + tau^2)
 * ampStateEvolution({ kind: 'gaussian', variance: 1 }, 0.5, 0.1, 3).map((v) => v.toFixed(4)) // => ['2.1000', '1.4548', '1.2853', '1.2248']
 * ```
 */
export function ampStateEvolution(prior: AmpPrior, delta: number, noiseVariance: number, iterations: number): number[] {
  if (!(delta > 0)) throw new OmaiError('value', 'AMP: delta must be positive')
  if (!(noiseVariance >= 0)) throw new OmaiError('value', 'AMP: the noise variance must not be negative')
  const out = [noiseVariance + priorSecondMoment(prior) / delta]
  for (let t = 0; t < iterations; t++) out.push(noiseVariance + ampMmse(prior, out[t] as number) / delta)
  return out
}

/** The linear model AMP runs on: the measurements and the row-major M by N matrix that produced them. */
export interface AmpProblem {
  /** The matrix, row-major (M rows, N columns); AMP's theory takes independent entries of variance 1 / M. */
  readonly matrix: ArrayLike<number>
  readonly rows: number
  readonly columns: number
  /** The measurements, length M. */
  readonly y: ArrayLike<number>
}

/** What {@link ampProblem} takes besides the key. */
export interface AmpProblemOptions {
  /** The number of unknowns (default 400). */
  readonly columns?: number
  /** Measurements per unknown, `M / N` (default 0.5). */
  readonly delta?: number
  /** The probability that an entry of the signal is nonzero (default 0.1). */
  readonly sparsity?: number
  /** The variance of the measurement noise (default 0.01). */
  readonly noiseVariance?: number
}

/**
 * A compressed-sensing problem with its truth: a matrix of independent Gaussian entries of variance `1 / M` (the case the state
 * evolution is exact for as the size grows), a signal whose entries are zero or standard normal draws (nonzero with
 * probability `sparsity`), and measurements with Gaussian noise. The truth is kept, so the error of the run can go on the tape.
 *
 * @param key - The random key.
 * @param options - The size, measurements per unknown, sparsity and noise.
 * @returns The problem, the signal that made it, and the prior and noise variance that describe it.
 * @concept approximate-message-passing
 * @index dataset A compressed-sensing problem (Gaussian matrix, sparse signal, noise) with its truth, for AMP.
 * @example
 * ```ts
 * import { ampProblem } from 'omai-engine/message-passing'
 * import { key } from 'omai-engine/random'
 * const p = ampProblem(key(1), { columns: 50, delta: 0.5 })
 * [p.problem.rows, p.problem.columns, p.truth.length] // => [25, 50, 50]
 * ```
 */
export function ampProblem(
  key: Key,
  options: AmpProblemOptions = {},
): {
  readonly problem: AmpProblem
  readonly truth: readonly number[]
  readonly prior: AmpPrior
  readonly noiseVariance: number
} {
  const n = options.columns ?? 400
  const m = Math.max(1, Math.round(n * (options.delta ?? 0.5)))
  const rho = options.sparsity ?? 0.1
  const noiseVariance = options.noiseVariance ?? 0.01
  const matrix = toFlat(normal(foldIn(key, 1), [m, n], { scale: 1 / Math.sqrt(m) }))
  const u = toFlat(uniform(foldIn(key, 2), [n]))
  const g = toFlat(normal(foldIn(key, 3), [n]))
  const w = toFlat(normal(foldIn(key, 4), [m], { scale: Math.sqrt(noiseVariance) }))
  const truth = Array.from(g, (v, i) => ((u[i] as number) < rho ? v : 0))
  const y = Array.from({ length: m }, (_, i) => {
    let s = w[i] as number
    for (let j = 0; j < n; j++) s += (matrix[i * n + j] as number) * (truth[j] as number)
    return s
  })
  return {
    problem: { matrix, rows: m, columns: n, y },
    truth,
    prior: { kind: 'bernoulli-gaussian', sparsity: rho, variance: 1 },
    noiseVariance,
  }
}

/** The knobs of AMP. */
export interface AmpOptions {
  /** The prior of every entry. */
  readonly prior: AmpPrior
  /** The variance of the measurement noise, when known: only for the state evolution on the tape (AMP estimates its own noise level). */
  readonly noiseVariance?: number
  /** The true signal, when known: only for the error on the tape. */
  readonly truth?: ArrayLike<number>
  /** Damping in [0, 1): the new x and z are `(1 - damping) new + damping old` (0 is AMP; damping is outside the theory and helps a matrix that is not iid). */
  readonly damping?: number
  /** Stop when the estimate moves by less than this, relative to its norm, in an iteration. */
  readonly tol?: number
  /** Stop, uncompleted, after this many iterations. */
  readonly maxIterations?: number
}

/** The state of an AMP run. */
export interface AmpState {
  /** The current estimate (the posterior mean of every entry). */
  readonly x: readonly number[]
  /** The residual with the Onsager correction. */
  readonly z: readonly number[]
  /** The derivative of the denoiser at every entry (it carries the posterior variance and the correction of the next iteration). */
  readonly derivative: readonly number[]
  /** The noise variance of the effective channel the last iteration denoised, `|z|^2 / M`. */
  readonly tau2: number
  readonly iteration: number
  /** The relative change of the estimate over the last iteration (Infinity before the first). */
  readonly change: number
}

/** The paths of an AMP run. */
export type AmpPath =
  | 'amp/tau2'
  | 'amp/stateEvolution'
  | 'amp/mse'
  | 'amp/change'
  | 'amp/iteration'
  | 'amp/marginals'
  | 'amp/frame'
  | 'amp/snapshot'

/**
 * What an AMP run logs, by path.
 *
 * @concept approximate-message-passing
 */
export const ampContract: Contract<AmpState, AmpPath> = defineContract({
  'amp/tau2': scalar({ from: 'tau2', description: 'the noise variance of the effective channel, |z|^2 / M' }),
  'amp/stateEvolution': scalar({
    from: 'stateEvolution',
    description: 'the noise variance the state evolution predicts for this iteration',
  }),
  'amp/mse': scalar({
    from: 'mse',
    description: 'the mean squared error of the estimate against the true signal (when it is given)',
  }),
  'amp/change': scalar({ from: 'change', description: 'the relative change of the estimate over the last iteration' }),
  'amp/iteration': scalar({ from: 'iteration', description: 'completed iterations' }),
  'amp/marginals': message({ from: 'marginals', description: 'the posterior mean and variance of every entry' }),
  'amp/frame': frame({ description: 'the posterior mean and variance of every entry' }),
  'amp/snapshot': message({
    description: 'an anytime snapshot (see runAnytimeWith): where the run is and the estimate',
  }),
})

const mean = (x: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i] ?? 0
  return s / x.length
}

/**
 * An anytime snapshot of an AMP state, in the shape of the engine's `MpSnapshot`: the iteration as `sweeps`, the change,
 * and one marginal `x` with the posterior mean and variance of every entry (`family: 'factorised'`). Pass it to
 * `runAnytimeWith`.
 *
 * @param options - The options the algorithm was built with.
 * @returns A function of a state alone.
 * @concept approximate-message-passing
 * @example
 * ```ts
 * import { ampSnapshot } from 'omai-engine/message-passing'
 * const state = { x: [0.5], z: [0], derivative: [0.5], tau2: 1, iteration: 3, change: 0.1 }
 * ampSnapshot()(state).marginals.x // => { family: 'factorised', mean: [0.5], variance: [0.5] }
 * ```
 */
export function ampSnapshot(): (state: AmpState) => MpSnapshot {
  return (state) => ({
    sweeps: state.iteration,
    updates: state.iteration,
    change: state.change,
    improper: { events: 0, skipped: 0, damped: 0, reset: 0 },
    marginals: { x: marginalOf(state) },
  })
}

const marginalOf = (state: AmpState): Record<string, Json> => ({
  family: 'factorised',
  mean: [...state.x],
  variance: state.derivative.map((d) => state.tau2 * d),
})

/**
 * Approximate message passing on the linear model `y = A x + w` with an iid prior on x: the iteration of the module
 * description, with the posterior mean of the prior as denoiser, its derivative in the Onsager term, and the noise level
 * estimated from the residual. The posterior variance of an entry is reported as `tau^2 eta'`. The tape carries the
 * estimated noise level, the one the state evolution predicts (when `noiseVariance` is given), the error against the truth
 * (when it is given), the change of the estimate and the posterior mean and variance. A matrix whose entries are not
 * independent can make the iteration diverge: damping, outside the theory, is the usual repair, and a run that stops being
 * finite ends as `diverged`. The state is plain numbers: a run serialises, resumes to the same state, and has anytime
 * snapshots through `ampSnapshot`.
 *
 * @param problem - The matrix and the measurements.
 * @param options - The prior (with the noise variance and the truth for the tape), damping, tolerance and iteration cap.
 * @returns The algorithm; its tape is described by {@link ampContract}.
 * @throws OmaiError with code `shape` for a matrix or measurements of the wrong size, `value` for a bad prior or damping.
 * @concept approximate-message-passing
 * @reference Donoho, D. L., Maleki, A. and Montanari, A. (2009). Message-passing algorithms for compressed sensing. Proceedings of the National Academy of Sciences 106(45).
 * @reference Bayati, M. and Montanari, A. (2011). The dynamics of message passing on dense graphs, with applications to compressed sensing. IEEE Transactions on Information Theory 57(2).
 * @index algorithm AMP for y = A x + w with an iid prior, with state evolution.
 * @example
 * ```ts
 * import { ampAlgorithm } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // two measurements of two unknowns: A = I, y = (1, -1), a standard normal prior and noise variance 1 give the shrunk estimate y / 2.
 * const alg = ampAlgorithm({ matrix: [1, 0, 0, 1], rows: 2, columns: 2, y: [1, -1] }, { prior: { kind: 'gaussian', variance: 1 }, maxIterations: 1 })
 * runAlgorithm(alg, { maxSteps: 1 }).state.iteration // => 1
 * ```
 */
export function ampAlgorithm(problem: AmpProblem, options: AmpOptions): Algorithm<AmpState, AmpPath> {
  const { rows: m, columns: n } = problem
  if (problem.matrix.length !== m * n || problem.y.length !== m)
    throw new OmaiError('shape', `AMP: a ${m} by ${n} matrix and ${m} measurements are needed`)
  checkPrior(options.prior)
  const damping = options.damping ?? 0
  if (!(damping >= 0 && damping < 1)) throw new OmaiError('value', `AMP: damping must be in [0, 1), got ${damping}`)
  const tol = options.tol ?? 1e-8
  const maxIterations = options.maxIterations ?? 200
  const truth = options.truth
  if (truth !== undefined && truth.length !== n) throw new OmaiError('shape', `AMP: the truth needs ${n} entries`)
  const delta = m / n
  const predicted =
    options.noiseVariance === undefined
      ? undefined
      : ampStateEvolution(options.prior, delta, options.noiseVariance, maxIterations)
  const matrix = Float64Array.from(problem.matrix)
  const y = Float64Array.from(problem.y)
  return {
    name: 'amp',
    config: { prior: options.prior as unknown as Json, damping, tol, maxIterations, rows: m, columns: n } as Json,
    schema: channels({
      metrics: {
        tau2: { description: 'the noise variance of the effective channel, |z|^2 / M' },
        stateEvolution: { description: 'the noise variance the state evolution predicts' },
        mse: { description: 'the mean squared error against the true signal', direction: 'lower' },
        change: { description: 'the relative change of the estimate over the last iteration' },
        iteration: { description: 'completed iterations' },
      },
      messages: { marginals: { description: 'the posterior mean and variance of every entry' } },
      frames: true,
    }),
    init: (): AmpState => ({
      x: new Array<number>(n).fill(0),
      z: Array.from(y),
      derivative: new Array<number>(n).fill(0),
      tau2: dnrm2(y) ** 2 / m,
      iteration: 0,
      change: Number.POSITIVE_INFINITY,
    }),
    step: (state): AmpState => {
      // The residual with its Onsager correction (the mean derivative of the last denoising over delta).
      const onsager = mean(state.derivative) / delta
      const ax = dgemv(matrix, m, n, state.x)
      const z = Array.from({ length: m }, (_, i) => {
        const next = (y[i] ?? 0) - (ax[i] ?? 0) + (state.z[i] ?? 0) * onsager
        return damping === 0 ? next : (1 - damping) * next + damping * (state.z[i] ?? 0)
      })
      const tau2 = Math.max(dnrm2(z) ** 2 / m, 1e-300)
      const back = dgemv(matrix, m, n, z, true)
      const x = new Array<number>(n)
      const derivative = new Array<number>(n)
      for (let i = 0; i < n; i++) {
        const d = ampDenoiser(options.prior, (state.x[i] ?? 0) + (back[i] ?? 0), tau2)
        x[i] = damping === 0 ? d.mean : (1 - damping) * d.mean + damping * (state.x[i] ?? 0)
        derivative[i] = d.derivative
      }
      const moved = dnrm2(x.map((v, i) => v - (state.x[i] ?? 0)))
      const scale = Math.max(dnrm2(x), 1e-300)
      return { x, z, derivative, tau2, iteration: state.iteration + 1, change: moved / scale }
    },
    done: (state): RunStatus | undefined => {
      if (state.x.some((v) => !Number.isFinite(v)) || !Number.isFinite(state.tau2)) return 'diverged'
      if (state.iteration >= 1 && state.change < tol) return 'converged'
      return state.iteration >= maxIterations ? 'completed' : undefined
    },
    observe: (state) => {
      const marginals = marginalOf(state)
      const se = predicted?.[Math.max(0, state.iteration - 1)]
      let mse: number | undefined
      if (truth !== undefined) {
        let s = 0
        for (let i = 0; i < n; i++) s += ((state.x[i] ?? 0) - (truth[i] ?? 0)) ** 2
        mse = s / n
      }
      return {
        metrics: {
          tau2: state.tau2,
          iteration: state.iteration,
          ...(Number.isFinite(state.change) ? { change: state.change } : {}),
          ...(se === undefined || state.iteration === 0 ? {} : { stateEvolution: se }),
          ...(mse === undefined ? {} : { mse }),
        },
        messages: [{ kind: 'marginals', payload: marginals as Json }],
        frame: { marginals: { x: marginals } as Json },
      }
    },
    serialise: (state) => stateToJson(state),
    deserialise: (json) => stateFromJson(json) as AmpState,
  }
}

/**
 * AMP on a problem as a method: the knobs as a space (the prior's family and its parameters, damping, tolerance,
 * iteration cap), the tape as the typed contract {@link ampContract}. The matrix and the measurements are the
 * constructables; the prior's kind and parameters are what a viewer chooses.
 *
 * @param problem - The matrix and the measurements.
 * @param tape - The noise variance and the truth, for the state evolution and the error on the tape.
 * @returns The method, named `amp`.
 * @concept approximate-message-passing
 * @index function The AMP method of a linear problem: the prior's kind and parameters as knobs, typed tape.
 * @example
 * ```ts
 * import { ampMethod } from 'omai-engine/message-passing'
 * ampMethod({ matrix: [1, 0, 0, 1], rows: 2, columns: 2, y: [1, -1] }).name // => 'amp'
 * ```
 */
export function ampMethod(
  problem: AmpProblem,
  tape: Pick<AmpOptions, 'noiseVariance' | 'truth'> = {},
): Method<AmpState, Config, AmpPath> {
  return defineMethod<AmpState, Config, AmpPath>({
    name: 'amp',
    config: dict({
      prior: choice(['bernoulli-gaussian', 'gaussian'] as const, {
        default: 'bernoulli-gaussian',
        description: 'the prior of every entry',
      }),
      sparsity: real(1e-3, 1, {
        log: true,
        default: 0.1,
        description: 'the probability that an entry is nonzero (Bernoulli-Gaussian prior)',
      }),
      variance: real(1e-3, 1e3, {
        log: true,
        default: 1,
        description: 'the variance of a nonzero entry (or of every entry, for the Gaussian prior)',
      }),
      damping: real(0, 0.95, { default: 0, description: 'weight of the old estimate in the new (outside the theory)' }),
      tol: real(1e-14, 1e-2, {
        log: true,
        default: 1e-8,
        description: 'stop when the estimate moves less than this relative to its norm',
      }),
      maxIterations: int(1, 10_000, { default: 200, description: 'the iteration cap' }),
    }),
    make: (c) => {
      const variance = typeof c['variance'] === 'number' ? c['variance'] : 1
      const sparsity = typeof c['sparsity'] === 'number' ? c['sparsity'] : 0.1
      const prior: AmpPrior =
        c['prior'] === 'gaussian' ? { kind: 'gaussian', variance } : { kind: 'bernoulli-gaussian', sparsity, variance }
      return ampAlgorithm(problem, {
        prior,
        ...tape,
        ...(typeof c['damping'] === 'number' ? { damping: c['damping'] } : {}),
        ...(typeof c['tol'] === 'number' ? { tol: c['tol'] } : {}),
        ...(typeof c['maxIterations'] === 'number' ? { maxIterations: c['maxIterations'] } : {}),
      })
    },
    contract: ampContract,
  })
}
