/**
 * Sequential Monte Carlo (SMC) samplers, which move a population of weighted particles from the prior to the posterior
 * through a ladder of tempered targets: systematic, stratified, multinomial and residual resampling, adaptive choice of
 * the next temperature, the default adaptive random-walk move, the tempered SMC sampler with its log-evidence estimate
 * (as BlackJAX), and annealed importance sampling.
 */

import { type Estimate, type Json, logSumExpOf, OmaiError } from '../base/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  type RunOptions,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { welfordCovariance, welfordInit, welfordUpdate } from './_mcmc.ts'
import type { MCMCKernel, MCMCState } from './chains.ts'
import { randomWalkMetropolis } from './metropolis.ts'
import { type ChainProposal, type LogDensity, numericTarget } from './target.ts'

function weightsOf(w: TensorLike, name: string): Float64Array {
  const v = Float64Array.from(toFlat(asTensor(w) as Tensor))
  if (v.length === 0) throw new OmaiError('value', `${name}: no weights`)
  let s = 0
  for (const x of v) {
    if (!(x >= 0)) throw new OmaiError('value', `${name}: weights must be non-negative`)
    s += x
  }
  if (!(s > 0)) throw new OmaiError('value', `${name}: the weights sum to zero`)
  return v.map((x) => x / s)
}

/** The first index whose cumulative weight reaches u (numpy searchsorted, side left), clipped to the last index. */
function inverseCdf(cumsum: Float64Array, us: ArrayLike<number>): Int32Array {
  const n = cumsum.length
  const out = new Int32Array(us.length)
  let j = 0
  for (let i = 0; i < us.length; i++) {
    const u = us[i] ?? 0
    // us is sorted, so the search continues from the last index.
    while (j < n && (cumsum[j] ?? 0) < u) j++
    out[i] = Math.min(j, n - 1)
  }
  return out
}

function cumulative(w: Float64Array): Float64Array {
  const c = new Float64Array(w.length)
  let s = 0
  for (let i = 0; i < w.length; i++) {
    s += w[i] ?? 0
    c[i] = s
  }
  return c
}

/**
 * Systematic resampling with a given uniform u (BlackJAX `systematic`): index i is chosen for the points
 * (k + u) / n, k = 0..n-1, on the cumulative weights. One uniform for all draws gives the lowest variance of the
 * standard schemes.
 *
 * @param weights - Non-negative weights (normalised here).
 * @param u - The uniform in [0, 1).
 * @param n - The number of indices (default the number of weights).
 * @returns The ancestor indices, sorted.
 * @throws OmaiError with code `value` for negative weights or weights summing to zero.
 * @concept resampling
 * @reference N. Chopin and O. Papaspiliopoulos, An introduction to sequential Monte Carlo (2020), chapter 9.
 * @example
 * ```ts
 * import { systematicResample } from 'omai-engine/sampling'
 * Array.from(systematicResample([0.5, 0.25, 0.25], 0.6)) // => [0, 1, 2]
 * ```
 */
export function systematicResample(weights: TensorLike, u: number, n?: number): Int32Array {
  const w = weightsOf(weights, 'systematicResample')
  const m = n ?? w.length
  return inverseCdf(
    cumulative(w),
    Float64Array.from({ length: m }, (_, k) => (k + u) / m),
  )
}

/**
 * Stratified resampling with given uniforms (BlackJAX `stratified`): one point (k + u_k) / n per stratum.
 *
 * @param weights - Non-negative weights.
 * @param u - n uniforms in [0, 1).
 * @param n - The number of indices (default the number of weights).
 * @returns The ancestor indices, sorted.
 * @throws OmaiError with code `shape` when the number of uniforms differs from n.
 * @concept resampling
 * @example
 * ```ts
 * import { stratifiedResample } from 'omai-engine/sampling'
 * Array.from(stratifiedResample([0.5, 0.5], [0.9, 0.1])) // => [0, 1]
 * ```
 */
export function stratifiedResample(weights: TensorLike, u: ArrayLike<number>, n?: number): Int32Array {
  const w = weightsOf(weights, 'stratifiedResample')
  const m = n ?? w.length
  if (u.length !== m) throw new OmaiError('shape', `stratifiedResample: need ${m} uniforms, got ${u.length}`)
  return inverseCdf(
    cumulative(w),
    Float64Array.from({ length: m }, (_, k) => (k + (u[k] ?? 0)) / m),
  )
}

/** Sorted uniforms from n + 1 uniforms by normalised exponential spacings -log u (Chopin's trick, as BlackJAX). */
function sortedUniforms(u: ArrayLike<number>, m: number): Float64Array {
  const z = new Float64Array(m + 1)
  let s = 0
  for (let i = 0; i <= m; i++) {
    const v = u[i] ?? 0
    s += -Math.log(v > 0 ? v : Number.MIN_VALUE)
    z[i] = s
  }
  return Float64Array.from({ length: m }, (_, i) => (z[i] ?? 0) / s)
}

/**
 * Multinomial resampling with given uniforms (BlackJAX `multinomial`): n independent draws from the weights, by
 * inverting the cumulative weights at n sorted uniforms built from n + 1 exponential spacings.
 *
 * @param weights - Non-negative weights.
 * @param u - n + 1 uniforms in [0, 1).
 * @param n - The number of indices (default the number of weights).
 * @returns The ancestor indices, sorted.
 * @throws OmaiError with code `shape` when the number of uniforms is not n + 1.
 * @concept resampling
 * @example
 * ```ts
 * import { multinomialResample } from 'omai-engine/sampling'
 * Array.from(multinomialResample([0, 1, 0], [0.3, 0.6, 0.2, 0.9])) // => [1, 1, 1]
 * ```
 */
export function multinomialResample(weights: TensorLike, u: ArrayLike<number>, n?: number): Int32Array {
  const w = weightsOf(weights, 'multinomialResample')
  const m = n ?? w.length
  if (u.length !== m + 1) throw new OmaiError('shape', `multinomialResample: need ${m + 1} uniforms, got ${u.length}`)
  return inverseCdf(cumulative(w), sortedUniforms(u, m))
}

/**
 * Residual resampling with given uniforms (BlackJAX `residual`): floor(n w_i) copies of each index
 * deterministically, then multinomial draws from the residual weights for the rest (in that order; BlackJAX permutes
 * the residual draws, which does not change the distribution).
 *
 * @param weights - Non-negative weights.
 * @param u - n + 1 uniforms for the multinomial part.
 * @param n - The number of indices (default the number of weights).
 * @returns The ancestor indices.
 * @throws OmaiError with code `shape` when the number of uniforms is not n + 1.
 * @concept resampling
 * @example
 * ```ts
 * import { residualResample } from 'omai-engine/sampling'
 * // n w = (1.5, 1.5): one copy of each, then one residual draw.
 * Array.from(residualResample([0.5, 0.5], [0.1, 0.2, 0.3, 0.4], 3)).slice(0, 2) // => [0, 1]
 * ```
 */
export function residualResample(weights: TensorLike, u: ArrayLike<number>, n?: number): Int32Array {
  const w = weightsOf(weights, 'residualResample')
  const m = n ?? w.length
  if (u.length !== m + 1) throw new OmaiError('shape', `residualResample: need ${m + 1} uniforms, got ${u.length}`)
  const out = new Int32Array(m)
  let k = 0
  const residual = new Float64Array(w.length)
  for (let i = 0; i < w.length; i++) {
    const copies = Math.floor(m * (w[i] ?? 0))
    for (let c = 0; c < copies && k < m; c++) out[k++] = i
    residual[i] = m * (w[i] ?? 0) - copies
  }
  const rest = m - k
  if (rest > 0) {
    const draws = inverseCdf(cumulative(weightsOf(residual, 'residualResample')), sortedUniforms(u, m))
    for (let i = 0; i < rest; i++) out[k + i] = draws[i] ?? 0
  }
  return out
}

/** The resampling schemes of {@link resample}. */
export type ResamplingScheme = 'multinomial' | 'systematic' | 'stratified' | 'residual'

/**
 * Draw ancestor indices from weights with a resampling scheme, using uniforms from the key.
 *
 * @param key - The random key.
 * @param weights - Non-negative weights.
 * @param options - The number of indices (default the number of weights) and the scheme (default `systematic`).
 * @returns The ancestor indices.
 * @concept resampling
 * @example
 * ```ts
 * import { resample } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * resample(key(0), [0, 0, 1], { scheme: 'stratified' }).every((i) => i === 2) // => true
 * ```
 */
export function resample(
  key: Key,
  weights: TensorLike,
  options: { readonly n?: number; readonly scheme?: ResamplingScheme } = {},
): Int32Array {
  const m = options.n ?? (asTensor(weights) as Tensor).shape[0] ?? 0
  const scheme = options.scheme ?? 'systematic'
  switch (scheme) {
    case 'systematic':
      return systematicResample(weights, uniform(key, [1]).data[0] ?? 0, m)
    case 'stratified':
      return stratifiedResample(weights, uniform(key, [m]).data, m)
    case 'multinomial':
      return multinomialResample(weights, uniform(key, [m + 1]).data, m)
    case 'residual':
      return residualResample(weights, uniform(key, [m + 1]).data, m)
    default:
      throw new OmaiError('value', `resample: unknown scheme ${String(scheme)}`)
  }
}

/** numpy's nan_to_num: NaN to 0 and infinities to the largest finite values. */
function finiteOf(v: number): number {
  if (Number.isNaN(v)) return 0
  if (v === Number.POSITIVE_INFINITY) return Number.MAX_VALUE
  if (v === Number.NEGATIVE_INFINITY) return -Number.MAX_VALUE
  return v
}

function logEss(logWeights: Float64Array): number {
  return 2 * logSumExpOf(logWeights) - logSumExpOf(logWeights.map((v) => 2 * v))
}

/** Options of {@link nextTemperature}. */
export interface TemperingOptions {
  /** The target ESS as a fraction of the particles (default 0.5). */
  readonly targetEss?: number
  /** The tolerance on the ESS gap of the bisection (default 1e-4, as BlackJAX). */
  readonly tol?: number
  /** The most bisection steps (default 100). */
  readonly maxIter?: number
}

/**
 * The next temperature of adaptive tempering (BlackJAX `adaptive_tempered` with its `ess_solver` and `dichotomy`):
 * the largest increment delta in [0, 1 - lambda] whose incremental weights exp(delta l_i) keep the ESS at
 * `targetEss` times the number of particles, found by bisection. When the full step keeps enough ESS the temperature
 * goes straight to 1.
 *
 * @param logLikelihoods - log L(x_i) of the current particles.
 * @param temperature - The current temperature lambda in [0, 1].
 * @param options - Target ESS fraction, tolerance and iteration cap.
 * @returns The next temperature.
 * @throws OmaiError with code `value` for a temperature outside [0, 1] or a target outside (0, 1).
 * @concept sequential-monte-carlo
 * @reference P. Del Moral, A. Doucet and A. Jasra, An adaptive sequential Monte Carlo method for approximate Bayesian
 *   computation (Statistics and Computing 2012).
 * @example
 * ```ts
 * import { nextTemperature } from 'omai-engine/sampling'
 * // Equal likelihoods never reduce the ESS: go straight to 1.
 * nextTemperature([-1, -1, -1, -1], 0.2) // => 1
 * ```
 */
export function nextTemperature(
  logLikelihoods: TensorLike,
  temperature: number,
  options: TemperingOptions = {},
): number {
  const target = options.targetEss ?? 0.5
  if (!(temperature >= 0 && temperature <= 1))
    throw new OmaiError('value', 'nextTemperature: temperature must lie in [0, 1]')
  if (!(target > 0 && target < 1)) throw new OmaiError('value', 'nextTemperature: targetEss must lie in (0, 1)')
  const tol = options.tol ?? 1e-4
  const maxIter = options.maxIter ?? 100
  const l = Float64Array.from(toFlat(asTensor(logLikelihoods) as Tensor))
  const goal = Math.log(l.length * target)
  const f = (delta: number): number => logEss(l.map((v) => finiteOf(delta * v))) - goal
  const maxDelta = 1 - temperature
  let fb = f(maxDelta)
  if (fb > 0) return 1
  let a = 0
  let b = maxDelta
  let fa = f(0)
  if (!(fa > 0)) return temperature
  for (let i = 0; i < maxIter && fa - fb > tol; i++) {
    const mid = 0.5 * (a + b)
    const fm = f(mid)
    if (fm < 0) {
      b = mid
      fb = fm
    } else {
      a = mid
      fa = fm
    }
  }
  return temperature + Math.min(Math.max(a, 0), maxDelta)
}

/** Builds the MCMC move of an SMC stage: a kernel for the tempered target, given the current particles. */
export type MoveKernel = (target: LogDensity, particles: Float64Array, temperature: number) => MCMCKernel

/**
 * The default SMC move: random-walk Metropolis with the proposal covariance (2.38^2 / dim) times the particles'
 * covariance (with a little jitter), the usual adaptive choice for SMC samplers.
 *
 * @param target - The tempered target.
 * @param particles - The current particles, row-major `[n, dim]`.
 * @returns The kernel.
 * @concept sequential-monte-carlo
 * @example
 * ```ts
 * import { adaptiveRandomWalkMove, numericTarget } from 'omai-engine/sampling'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * adaptiveRandomWalkMove(t, new Float64Array([-1, 0, 1])).name // => 'randomWalkMetropolis'
 * ```
 */
export function adaptiveRandomWalkMove(target: LogDensity, particles: Float64Array): MCMCKernel {
  const dim = target.dim
  let w = welfordInit(dim)
  for (let i = 0; i < particles.length / dim; i++) w = welfordUpdate(w, particles.subarray(i * dim, (i + 1) * dim))
  const scale = (2.38 * 2.38) / dim
  const cov = welfordCovariance(w).map((v, k) => scale * v + (k % (dim + 1) === 0 ? 1e-12 : 0))
  const rows = Array.from({ length: dim }, (_, i) => Array.from(cov.subarray(i * dim, (i + 1) * dim)))
  return randomWalkMetropolis(target, { covariance: rows })
}

/** The state of {@link smcAlgorithm}. */
export interface SMCState {
  /** The particles, row-major `[n, dim]`. */
  readonly particles: Float64Array
  readonly logLikelihood: Float64Array
  readonly temperature: number
  /** The running estimate of log Z = log integral of prior x likelihood. */
  readonly logEvidence: number
  /** The ESS of the incremental weights before resampling. */
  readonly ess: number
  /** The mean acceptance rate of the last MCMC moves. */
  readonly acceptanceRate: number
  readonly stage: number
}

/** Options of {@link smcAlgorithm}. */
export interface SMCOptions extends TemperingOptions {
  /** The number of particles (default 1000). */
  readonly numParticles?: number
  /** The resampling scheme (default `systematic`). */
  readonly resampling?: ResamplingScheme
  /** MCMC steps per particle per stage (default 5). */
  readonly mcmcSteps?: number
  /** The MCMC move (default {@link adaptiveRandomWalkMove}). */
  readonly move?: MoveKernel
}

function temperedTarget(
  prior: ChainProposal,
  logLikelihood: (x: Float64Array) => number,
  temperature: number,
): LogDensity {
  return numericTarget((x) => {
    const lp = prior.logProb(x)
    if (lp === Number.NEGATIVE_INFINITY) return lp
    const ll = logLikelihood(x)
    return lp + (temperature === 0 ? 0 : temperature * (Number.isNaN(ll) ? Number.NEGATIVE_INFINITY : ll))
  }, prior.dim)
}

/** Move every particle with `steps` transitions of a kernel; returns the moved particles and the mean acceptance. */
function moveParticles(
  kernel: MCMCKernel,
  particles: Float64Array,
  dim: number,
  steps: number,
  key: Key,
): { readonly particles: Float64Array; readonly acceptance: number } {
  const n = particles.length / dim
  const out = new Float64Array(particles.length)
  let acceptance = 0
  for (let i = 0; i < n; i++) {
    let s: MCMCState = kernel.init(particles.subarray(i * dim, (i + 1) * dim))
    for (let t = 0; t < steps; t++) {
      const tr = kernel.step(foldIn(foldIn(key, i), t), s)
      s = tr.state
      acceptance += tr.info.acceptanceRate / (n * steps)
    }
    out.set(s.position, i * dim)
  }
  return { particles: out, acceptance }
}

/**
 * A sequential Monte Carlo sampler with adaptive tempering (Del Moral, Doucet and Jasra 2006; BlackJAX
 * `adaptive_tempered_smc`): particles drawn from the prior move through the tempered targets
 * prior(x) L(x)^lambda, lambda from 0 to 1. Each stage picks the next lambda so the incremental weights keep the
 * target ESS ({@link nextTemperature}), accumulates log Z, resamples, and rejuvenates every particle with MCMC moves
 * at the new temperature. As a traced algorithm each step is one stage, recording the temperature, ESS, log evidence
 * and acceptance rate, and the particles.
 *
 * @param prior - The prior (draws and log density).
 * @param logLikelihood - log L(x) (NaN reads as -Infinity).
 * @param options - Particles, target ESS, resampling scheme, MCMC steps and move, tempering tolerance.
 * @returns The algorithm; it converges when lambda reaches 1.
 * @throws OmaiError with code `value` for a non-positive number of particles or MCMC steps.
 * @concept sequential-monte-carlo
 * @reference P. Del Moral, A. Doucet and A. Jasra, Sequential Monte Carlo samplers (JRSS B 2006).
 * @index algorithm Sequential Monte Carlo, one tempering step at a time.
 * @example
 * ```ts
 * import { proposalFromDistribution, smcAlgorithm } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const smc = smcAlgorithm(proposalFromDistribution(normal(0, 3)), (x) => -0.5 * ((x[0] ?? 0) - 1) ** 2, { numParticles: 200 })
 * runAlgorithm(smc, { seed: 0, maxSteps: 50 }).state.temperature // => 1
 * ```
 */
export function smcAlgorithm(
  prior: ChainProposal,
  logLikelihood: (x: Float64Array) => number,
  options: SMCOptions = {},
): Algorithm<SMCState> {
  const n = options.numParticles ?? 1000
  const steps = options.mcmcSteps ?? 5
  if (!(Number.isInteger(n) && n >= 2 && Number.isInteger(steps) && steps >= 0)) {
    throw new OmaiError('value', 'smcAlgorithm: need at least 2 particles and a non-negative number of MCMC steps')
  }
  const dim = prior.dim
  const scheme = options.resampling ?? 'systematic'
  const move = options.move ?? adaptiveRandomWalkMove
  const ll = (x: Float64Array): number => {
    const v = logLikelihood(x)
    return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
  }
  const likelihoods = (p: Float64Array): Float64Array =>
    Float64Array.from({ length: n }, (_, i) => ll(p.subarray(i * dim, (i + 1) * dim)))
  return {
    name: 'smc',
    config: { particles: n, dim, mcmcSteps: steps, resampling: scheme, targetEss: options.targetEss ?? 0.5 } as Json,
    schema: channels({
      metrics: {
        temperature: { description: 'the tempering parameter lambda' },
        ess: { description: 'ESS of the incremental weights before resampling' },
        logEvidence: { description: 'running estimate of log Z' },
        acceptanceRate: { description: 'mean acceptance rate of the MCMC moves', direction: 'higher' },
      },
      tensors: { particles: { role: 'state', description: 'the particles', shape: [n, dim] } },
    }),
    init: (key) => {
      const particles = prior.sampleMany(key, n)
      return {
        particles,
        logLikelihood: likelihoods(particles),
        temperature: 0,
        logEvidence: 0,
        ess: n,
        acceptanceRate: Number.NaN,
        stage: 0,
      }
    },
    step: (s, ctx) => {
      const next = nextTemperature(s.logLikelihood, s.temperature, options)
      const delta = next - s.temperature
      const inc = s.logLikelihood.map((v) => (delta === 0 ? 0 : finiteOf(delta * v)))
      const logZ = s.logEvidence + logSumExpOf(inc) - Math.log(n)
      const ess = Math.exp(logEss(inc))
      const z = logSumExpOf(inc)
      const idx = resample(
        foldIn(ctx.key, 0),
        inc.map((v) => Math.exp(v - z)),
        { n, scheme },
      )
      const resampled = new Float64Array(n * dim)
      idx.forEach((a, i) => {
        resampled.set(s.particles.subarray(a * dim, (a + 1) * dim), i * dim)
      })
      const kernel = move(temperedTarget(prior, ll, next), resampled, next)
      const moved =
        steps > 0
          ? moveParticles(kernel, resampled, dim, steps, foldIn(ctx.key, 1))
          : { particles: resampled, acceptance: Number.NaN }
      return {
        particles: moved.particles,
        logLikelihood: likelihoods(moved.particles),
        temperature: next,
        logEvidence: logZ,
        ess,
        acceptanceRate: moved.acceptance,
        stage: s.stage + 1,
      }
    },
    done: (s) => (s.temperature >= 1 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        temperature: s.temperature,
        ess: s.ess,
        logEvidence: s.logEvidence,
        ...(Number.isNaN(s.acceptanceRate) ? {} : { acceptanceRate: s.acceptanceRate }),
      },
      tensors: { particles: { role: 'state', value: s.particles, shape: [n, dim] } },
    }),
  }
}

/** The result of {@link smcSampler}. */
export interface SMCResult {
  /** Equally weighted draws from the posterior `[n, dim]`. */
  readonly particles: Tensor<'float64'>
  readonly logEvidence: number
  /** The temperatures of the stages, from 0 to 1. */
  readonly temperatures: readonly number[]
  readonly stages: number
  /** Whether lambda reached 1 within the stage cap. */
  readonly converged: boolean
}

/**
 * Run the adaptive-tempering SMC sampler ({@link smcAlgorithm}) to lambda = 1 and return the posterior particles,
 * the log evidence and the temperature ladder.
 *
 * @param prior - The prior.
 * @param logLikelihood - log L(x).
 * @param options - SMC options, a cap on the number of stages (default 500), key or seed, tracer and policy.
 * @returns The particles, log Z, temperatures and whether the run reached lambda = 1.
 * @concept sequential-monte-carlo
 * @index function Sequential Monte Carlo sampler with adaptive tempering.
 * @example
 * ```ts
 * import { proposalFromDistribution, smcSampler } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * // Prior N(0, 3^2), likelihood N(1; x, 1): Z = N(1; 0, 10) on the natural scale.
 * const r = smcSampler(proposalFromDistribution(normal(0, 3)), (x) => -0.5 * (1 - (x[0] ?? 0)) ** 2 - 0.5 * Math.log(2 * Math.PI), { numParticles: 400, key: key(0) })
 * Math.abs(r.logEvidence - (-0.5 * Math.log(2 * Math.PI * 10) - 0.05)) < 0.15 // => true
 * ```
 */
export function smcSampler(
  prior: ChainProposal,
  logLikelihood: (x: Float64Array) => number,
  options: SMCOptions & { readonly maxStages?: number } & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): SMCResult {
  const { maxStages, numParticles, resampling, mcmcSteps, move, targetEss, tol, maxIter, ...run } = options
  const algorithm = smcAlgorithm(prior, logLikelihood, {
    ...(numParticles === undefined ? {} : { numParticles }),
    ...(resampling === undefined ? {} : { resampling }),
    ...(mcmcSteps === undefined ? {} : { mcmcSteps }),
    ...(move === undefined ? {} : { move }),
    ...(targetEss === undefined ? {} : { targetEss }),
    ...(tol === undefined ? {} : { tol }),
    ...(maxIter === undefined ? {} : { maxIter }),
  })
  const temperatures: number[] = [0]
  const recording: Algorithm<SMCState> = {
    ...algorithm,
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      temperatures.push(next.temperature)
      return next
    },
  }
  const r = runAlgorithm(recording, { ...run, maxSteps: maxStages ?? 500 })
  const n = r.state.particles.length / prior.dim
  return {
    particles: fromStorage(r.state.particles, [n, prior.dim], 'float64'),
    logEvidence: r.state.logEvidence,
    temperatures,
    stages: r.steps,
    converged: r.status === 'converged',
  }
}

/** Options of {@link annealedImportanceSampling}. */
export interface AISOptions {
  readonly key: Key
  /** The number of independent runs (default 500). */
  readonly numParticles?: number
  /** The temperature ladder from 0 to 1 (default 50 steps of a geometric-in-lambda^4 ladder). */
  readonly temperatures?: readonly number[]
  /** MCMC steps per temperature (default 2). */
  readonly mcmcSteps?: number
  /** The MCMC move per temperature (default random-walk Metropolis scaled by the particles' spread). */
  readonly move?: MoveKernel
}

/** The result of {@link annealedImportanceSampling}. */
export interface AISResult {
  readonly particles: Tensor<'float64'>
  readonly logWeights: Tensor<'float64'>
  readonly logEvidence: Estimate
  readonly ess: number
}

/** The state of {@link aisAlgorithm}: the rung of the ladder reached, the particles and their accumulated log weights. */
export interface AISState {
  /** Rungs done: the particles are at temperature `ladder[rung]`. */
  readonly rung: number
  /** The particles, `dim` numbers each. */
  readonly particles: readonly number[]
  /** The accumulated log weight of every particle. */
  readonly logWeights: readonly number[]
}

/** The paths {@link aisAlgorithm} logs, as a type. */
export type AISPath = 'ais/beta' | 'ais/ess' | 'ais/log-weight' | 'ais/log-z' | 'ais/particles'

/**
 * What {@link aisAlgorithm} logs, by path, one rung of the ladder per step: the temperature reached, the effective
 * sample size of the weights, the mean log weight, the running log-evidence estimate and the particles.
 */
export const aisContract: Contract<AISState, AISPath> = defineContract({
  'ais/beta': scalar({ description: 'the temperature reached', direction: 'higher' }),
  'ais/ess': scalar({ description: 'the effective sample size of the weights', direction: 'higher' }),
  'ais/log-weight': scalar({ description: 'the mean log weight' }),
  'ais/log-z': scalar({ description: 'the log-evidence estimate so far: log of the mean weight' }),
  'ais/particles': tensor({ role: 'state', shape: ['particles', 'dim'], description: 'the particles after the move' }),
})

function aisLadder(options: AISOptions): readonly number[] {
  const ladder = options.temperatures ?? Array.from({ length: 51 }, (_, i) => (i / 50) ** 4)
  if (
    ladder[0] !== 0 ||
    ladder[ladder.length - 1] !== 1 ||
    ladder.some((v, i) => i > 0 && !(v > (ladder[i - 1] ?? 0)))
  ) {
    throw new OmaiError('value', 'annealedImportanceSampling: temperatures must increase from 0 to 1')
  }
  return ladder
}

const effectiveSize = (logW: readonly number[]): number => {
  const z = logSumExpOf(logW)
  let s2 = 0
  for (const v of logW) s2 += Math.exp(2 * (v - z))
  return 1 / s2
}

/**
 * Annealed importance sampling (Neal 2001) as a steppable algorithm: `init` draws the particles from the prior; one
 * step takes the next rung of the temperature ladder, adding (lambda_t - lambda_{t-1}) log L(x) to every log weight and
 * moving the particles with MCMC transitions that leave the tempered target prior(x) L(x)^lambda_t invariant. The state
 * is plain numbers; the step's randomness is derived from the options' key and the rung, so a resumed run moves the same
 * way.
 *
 * @param prior - The prior.
 * @param logLikelihood - log L(x).
 * @param options - Key, particles, temperature ladder, MCMC steps and move.
 * @returns The algorithm; its state is an {@link AISState}.
 * @throws OmaiError with code `value` for a ladder that does not increase from 0 to 1 or a particle count below 1.
 * @concept annealed-importance-sampling
 * @example
 * ```ts
 * import { aisAlgorithm, proposalFromDistribution } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = aisAlgorithm(proposalFromDistribution(normal(0, 3)), (x) => -0.5 * (x[0] ?? 0) ** 2, { key: key(0), numParticles: 20, temperatures: [0, 0.5, 1] })
 * runAlgorithm(alg, { maxSteps: 10 }).state.rung // => 2
 * ```
 */
export function aisAlgorithm(
  prior: ChainProposal,
  logLikelihood: (x: Float64Array) => number,
  options: AISOptions,
): Algorithm<AISState, AISPath> {
  const n = options.numParticles ?? 500
  const steps = options.mcmcSteps ?? 2
  const ladder = aisLadder(options)
  if (!(Number.isInteger(n) && n >= 1)) throw new OmaiError('value', 'annealedImportanceSampling: need particles')
  const dim = prior.dim
  const move = options.move ?? adaptiveRandomWalkMove
  const ll = (x: Float64Array): number => {
    const v = logLikelihood(x)
    return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
  }
  return {
    name: 'annealed-importance-sampling',
    config: { particles: n, mcmcSteps: steps, rungs: ladder.length - 1, dim },
    init: () => ({
      rung: 0,
      particles: Array.from(prior.sampleMany(foldIn(options.key, 0), n)),
      logWeights: new Array<number>(n).fill(0),
    }),
    step: (s, ctx) => {
      const t = s.rung + 1
      const delta = (ladder[t] ?? 1) - (ladder[t - 1] ?? 0)
      const x = Float64Array.from(s.particles)
      const logW = s.logWeights.map((w, i) => w + delta * ll(x.subarray(i * dim, (i + 1) * dim)))
      const temperature = ladder[t] ?? 1
      const kernel = move(temperedTarget(prior, ll, temperature), x, temperature)
      const moved = moveParticles(kernel, x, dim, steps, foldIn(options.key, t)).particles
      if (ctx.wants('ais/beta')) ctx.log('ais/beta', temperature)
      if (ctx.wants('ais/ess')) ctx.log('ais/ess', effectiveSize(logW))
      if (ctx.wants('ais/log-weight')) ctx.log('ais/log-weight', logW.reduce((a, b) => a + b, 0) / n)
      if (ctx.wants('ais/log-z')) ctx.log('ais/log-z', logSumExpOf(logW) - Math.log(n))
      if (ctx.wants('ais/particles')) ctx.log('ais/particles', moved)
      return { rung: t, particles: Array.from(moved), logWeights: logW }
    },
    done: (s) => (s.rung >= ladder.length - 1 ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as AISState,
  }
}

/**
 * Annealed importance sampling (Neal 2001): each particle starts at the prior and moves through the fixed ladder of
 * tempered targets prior(x) L(x)^lambda_t with MCMC transitions, accumulating the log weight
 * sum_t (lambda_t - lambda_{t-1}) log L(x_{t-1}); there is no resampling, so the runs are independent and the mean
 * weight is an unbiased estimate of Z.
 *
 * @param prior - The prior.
 * @param logLikelihood - log L(x).
 * @param options - Key, particles, temperature ladder, MCMC steps and move.
 * @returns The final particles, their log weights, the log-evidence estimate and the weights' ESS.
 * @throws OmaiError with code `value` for a ladder that does not increase from 0 to 1.
 * @concept annealed-importance-sampling
 * @reference R. M. Neal, Annealed importance sampling (Statistics and Computing 2001).
 * @example
 * ```ts
 * import { annealedImportanceSampling, proposalFromDistribution } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const r = annealedImportanceSampling(proposalFromDistribution(normal(0, 3)), (x) => -0.5 * (1 - (x[0] ?? 0)) ** 2 - 0.5 * Math.log(2 * Math.PI), { key: key(0), numParticles: 200 })
 * Math.abs(r.logEvidence.value - (-0.5 * Math.log(2 * Math.PI * 10) - 0.05)) < 0.1 // => true
 * ```
 */
export function annealedImportanceSampling(
  prior: ChainProposal,
  logLikelihood: (x: Float64Array) => number,
  options: AISOptions,
): AISResult {
  const n = options.numParticles ?? 500
  const dim = prior.dim
  const algorithm = aisAlgorithm(prior, logLikelihood, options)
  const rungs = aisLadder(options).length - 1
  const s = runAlgorithm(algorithm, { maxSteps: rungs }).state
  const logW = Float64Array.from(s.logWeights)
  const value = logSumExpOf(logW) - Math.log(n)
  let sq = 0
  for (const v of logW) sq += (Math.exp(v - value) - 1) ** 2
  return {
    particles: fromStorage(Float64Array.from(s.particles), [n, dim], 'float64'),
    logWeights: fromStorage(logW, [n], 'float64'),
    logEvidence: {
      value,
      standardError: n > 1 ? Math.sqrt(sq / (n - 1) / n) : Number.NaN,
      method: 'annealed importance sampling',
    },
    ess: effectiveSize(s.logWeights),
  }
}
