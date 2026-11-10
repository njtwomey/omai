/**
 * Particle filters: sequential Monte Carlo estimation of a hidden state from a weighted cloud of samples. The
 * resampling schemes (multinomial, stratified, systematic, residual), the bootstrap and auxiliary particle filters with
 * resampling on low effective sample size, and particle smoothing by forward filtering and backward simulation.
 */

import { logSumExpOf, OmaiError } from '../../base/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../../random/index.ts'
import {
  asTensor,
  primalOf,
  reshape,
  sliceAxis,
  type Tensor,
  type TensorLike,
  take,
  type Value,
} from '../../tensor/index.ts'
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
} from '../../trace/index.ts'
import { fromNumbers, numbersOf } from '../ode/_vec.ts'

/** The resampling schemes. */
export type Resampling = 'multinomial' | 'systematic' | 'stratified' | 'residual'

function cumulative(weights: ArrayLike<number>): Float64Array {
  const c = new Float64Array(weights.length)
  let s = 0
  for (let i = 0; i < weights.length; i++) {
    s += weights[i] ?? 0
    c[i] = s
  }
  // Guard the last entry against rounding.
  if (c.length > 0) c[c.length - 1] = 1
  return c
}

function searchSorted(c: Float64Array, u: number): number {
  let lo = 0
  let hi = c.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if ((c[mid] ?? 0) < u) lo = mid + 1
    else hi = mid
  }
  return lo
}

/**
 * Resample particle indices in proportion to normalised weights from uniform variates (filterpy's
 * monte_carlo resamplers, with the randomness passed in): multinomial (N independent uniforms), stratified (one
 * uniform per stratum [i/N, (i+1)/N)), systematic (a single uniform shifted by i/N) and residual (the integer parts
 * floor(N w_i) deterministically, the rest multinomially).
 *
 * @param weights - Normalised weights.
 * @param scheme - The scheme.
 * @param u - Uniform variates in [0, 1): N for multinomial, stratified and residual, the first one for systematic.
 * @returns N particle indices.
 * @throws OmaiError with code `value` when there are too few uniforms.
 * @concept particle-filter
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * filtering.resampleIndices([0.1, 0.2, 0.7], 'systematic', [0.5]) // => [1, 2, 2]
 * ```
 */
export function resampleIndices(weights: readonly number[], scheme: Resampling, u: readonly number[]): number[] {
  const N = weights.length
  const needed = scheme === 'systematic' ? 1 : N
  if (u.length < needed) throw new OmaiError('value', `resampleIndices: ${scheme} needs ${needed} uniforms`)
  if (scheme === 'residual') {
    const out: number[] = []
    const residual = new Float64Array(N)
    weights.forEach((w, i) => {
      const copies = Math.floor(N * w)
      for (let k = 0; k < copies; k++) out.push(i)
      residual[i] = N * w - copies
    })
    const left = N - out.length
    const total = residual.reduce((a, b) => a + b, 0)
    if (left > 0 && total > 0) {
      const c = cumulative(Float64Array.from(residual, (r) => r / total))
      for (let k = 0; k < left; k++) out.push(searchSorted(c, u[k] ?? 0))
    }
    return out
  }
  const c = cumulative(weights)
  const positions =
    scheme === 'multinomial'
      ? u.slice(0, N)
      : scheme === 'stratified'
        ? Array.from({ length: N }, (_, i) => (i + (u[i] ?? 0)) / N)
        : Array.from({ length: N }, (_, i) => (i + (u[0] ?? 0)) / N)
  return positions.map((p) => searchSorted(c, p))
}

/**
 * A state-space model for particle filtering: draw initial particles, propagate them, and score observations; the
 * transition density is needed for smoothing.
 */
export interface ParticleModel {
  /** N initial particles `[N, d]`. */
  readonly initial: (key: Key, n: number) => Value
  /** Propagate particles `[N, d]` to time t. */
  readonly transition: (key: Key, particles: Value, t: number) => Value
  /** log p(z_t | x) for each particle `[N]`. */
  readonly logLikelihood: (z: Value, particles: Value, t: number) => Value
  /** log p(x_{t+1} = next | x_t = particle) for each particle `[N]` (for smoothing). */
  readonly logTransition?: (particles: Value, next: Value, t: number) => Value
  /** A point prediction of each particle's next state (for the auxiliary filter's first stage), `[N, d]`. */
  readonly predict?: (particles: Value, t: number) => Value
  /**
   * A guided proposal: when present, particles are moved by it instead of by `transition` and weighted by the ratio of the
   * transition to the proposal density, so a proposal that looks at the observation wastes fewer particles. Unbiasedness of
   * the evidence estimate holds whatever the proposal, as long as it covers the transition's support.
   */
  readonly guide?: ParticleGuide
}

/**
 * A proposal of the particle filters that sees the observation: draws x_t from q(x_t | x_{t-1}, z_t) for every particle
 * and returns the log of p(x_t | x_{t-1}) / q(x_t | x_{t-1}, z_t) per particle, the correction the weights need.
 */
export interface ParticleGuide {
  /**
   * @param key - The step's key for the draws.
   * @param particles - The particles `[N, d]` before the move.
   * @param z - The observation at this step.
   * @param t - The step.
   * @returns The moved particles `[N, d]` and the log ratio `[N]`.
   */
  readonly propose: (
    key: Key,
    particles: Value,
    z: Value,
    t: number,
  ) => { readonly particles: Value; readonly logRatio: Value }
}

/** The output of the particle filters. */
export interface ParticleFilterResult {
  /** Weighted means `[T, d]`. */
  readonly means: Tensor
  /** The estimate of log p(z_1..T) (unbiased in p, not in log p). */
  readonly logEvidence: number
  /** Effective sample size before any resampling, per step. */
  readonly ess: readonly number[]
  /** Whether each step resampled. */
  readonly resampled: readonly boolean[]
  /** Particles `[N, d]` and normalised weights at every step (for smoothing). */
  readonly particles: readonly Tensor[]
  readonly weights: readonly Float64Array[]
}

/**
 * The state of the particle filters after `t` observations: the cloud (particles and normalised log-weights), the
 * evidence so far, and what every step said.
 */
export interface PfState {
  /** Observations consumed. */
  readonly t: number
  /** The particles `[N, d]` after the last step. */
  readonly particles: Value
  /** Their normalised log-weights `[N]`. */
  readonly logW: Float64Array
  /** The estimate of log p(z_1..t). */
  readonly logEvidence: number
  /** For each particle after the last step, the index of the particle it descends from (the identity without resampling). */
  readonly ancestors: readonly number[]
  /** Weighted means per step. */
  readonly means: readonly Float64Array[]
  /** The effective sample size per step, and whether the step resampled. */
  readonly ess: readonly number[]
  readonly resampled: readonly boolean[]
  /** Particles and normalised weights at every step (for smoothing). */
  readonly history: readonly Tensor[]
  readonly weights: readonly Float64Array[]
}

/** The paths the particle filters log, as a type. */
export type PfPath =
  | 'pf/ess'
  | 'pf/log-evidence'
  | 'pf/resampled'
  | 'pf/mean'
  | 'pf/weights'
  | 'pf/particles'
  | 'pf/ancestors'

/**
 * What the particle filters log, by path: the effective sample size before resampling, the evidence so far, whether
 * the step resampled, the weighted mean, the normalised weights, the particles and the ancestry of each particle.
 */
export const pfContract: Contract<PfState, PfPath> = defineContract({
  'pf/ess': scalar({ description: 'effective sample size before resampling' }),
  'pf/log-evidence': scalar({ description: 'running estimate of log p(z_1..t)', direction: 'higher' }),
  'pf/resampled': scalar({ description: '1 when this step resampled, else 0' }),
  'pf/mean': tensor({ role: 'state', shape: ['state'], description: 'the weighted mean of the particles' }),
  'pf/weights': tensor({ role: 'state', shape: ['particle'], description: 'the normalised weights after this step' }),
  'pf/particles': tensor({ role: 'state', shape: ['particle', 'state'], description: 'the particles after this step' }),
  'pf/ancestors': tensor({
    role: 'state',
    shape: ['particle'],
    description: 'the index of the particle each one descends from at this step',
  }),
})

function normalise(logW: Float64Array): { w: Float64Array; lse: number } {
  const lse = logSumExpOf(logW)
  return { w: Float64Array.from(logW, (v) => Math.exp(v - lse)), lse }
}

function weightedMean(particles: Value, w: Float64Array): Float64Array {
  const p = primalOf(particles)
  const N = p.shape[0] ?? 0
  const d = (p.data.length / Math.max(1, N)) | 0
  const out = new Float64Array(d)
  for (let i = 0; i < N; i++)
    for (let j = 0; j < d; j++) out[j] = (out[j] ?? 0) + (w[i] ?? 0) * (p.data[i * d + j] ?? 0)
  return out
}

function uniforms(key: Key, n: number): number[] {
  return Array.from(uniform(key, [n]).data)
}

/** Options of the particle filters. */
export interface ParticleOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** Number of particles (default 500). */
  readonly particles?: number
  readonly resampling?: Resampling
  /** Resample when ESS / N falls below this (default 0.5; 1 resamples every step). */
  readonly essThreshold?: number
  readonly key?: Key
}

/** Options of the particle filter algorithms. */
export interface ParticleAlgorithmOptions {
  /** Number of particles (default 500). */
  readonly particles?: number
  readonly resampling?: Resampling
  /** Resample when ESS / N falls below this (default 0.5; 1 resamples every step). */
  readonly essThreshold?: number
  readonly key?: Key
}

function pfAlgorithm(
  name: string,
  model: ParticleModel,
  observations: TensorLike,
  options: ParticleAlgorithmOptions,
  auxiliary: boolean,
): Algorithm<PfState, PfPath> {
  const Z = asTensor(observations)
  const T = Z.shape[0] ?? 0
  const N = options.particles ?? 500
  const scheme = options.resampling ?? 'systematic'
  const threshold = options.essThreshold ?? 0.5
  const root = options.key ?? makeKey(0)
  if (auxiliary && model.predict === undefined)
    throw new OmaiError('value', 'auxiliaryParticleFilter: the model needs predict')
  const zAt = (t: number): Value => reshape(sliceAxis(Z, 0, t, t + 1), Z.shape.slice(1))
  const identity = Array.from({ length: N }, (_, i) => i)
  return {
    name,
    config: { particles: N, resampling: scheme, threshold, steps: T },
    schema: channels({
      metrics: {
        ess: { description: 'effective sample size before resampling' },
        logEvidence: { description: 'running log p(z_1..t) estimate', direction: 'higher' },
      },
    }),
    init: () => ({
      t: 0,
      particles: model.initial(foldIn(root, 0), N),
      logW: new Float64Array(N).fill(-Math.log(N)),
      logEvidence: 0,
      ancestors: identity,
      means: [],
      ess: [],
      resampled: [],
      history: [],
      weights: [],
    }),
    step: (s, ctx) => {
      const t = ctx.t
      const kt = foldIn(root, t + 1)
      let particles = s.particles
      let logW = s.logW
      let firstStage = new Float64Array(N)
      let increment = 0
      let resampledNow = false
      let ancestors = identity
      const { w: wPrev } = normalise(logW)
      const essPrev = 1 / wPrev.reduce((acc, v) => acc + v * v, 0)
      if (auxiliary) {
        // First stage: weight by the likelihood of each particle's point prediction and resample.
        const pred = (model.predict as NonNullable<ParticleModel['predict']>)(particles, t)
        const lhat = numbersOf(model.logLikelihood(zAt(t), pred, t))
        const { w, lse } = normalise(Float64Array.from(logW, (v, i) => v + (lhat[i] ?? 0)))
        const idx = resampleIndices(Array.from(w), scheme, uniforms(foldIn(kt, 0), N))
        particles = take(particles, idx, 0)
        firstStage = Float64Array.from(idx, (i) => lhat[i] ?? 0)
        logW = new Float64Array(N).fill(-Math.log(N))
        increment = lse
        resampledNow = true
        ancestors = idx
      } else if (essPrev / N < threshold) {
        const idx = resampleIndices(Array.from(wPrev), scheme, uniforms(foldIn(kt, 0), N))
        particles = take(particles, idx, 0)
        logW = new Float64Array(N).fill(-Math.log(N))
        resampledNow = true
        ancestors = idx
      }
      let logRatio: ArrayLike<number> | undefined
      if (model.guide === undefined) particles = model.transition(foldIn(kt, 1), particles, t)
      else {
        const moved = model.guide.propose(foldIn(kt, 1), particles, zAt(t), t)
        particles = moved.particles
        logRatio = numbersOf(moved.logRatio)
      }
      const ll = numbersOf(model.logLikelihood(zAt(t), particles, t))
      // logW is normalised, so the log-sum of the updated weights estimates log p(z_t | z_1..t-1).
      const { w, lse } = normalise(
        Float64Array.from(logW, (v, i) => v + (ll[i] ?? 0) + (logRatio?.[i] ?? 0) - (firstStage[i] ?? 0)),
      )
      const ess = 1 / w.reduce((acc, v) => acc + v * v, 0)
      const mean = weightedMean(particles, w)
      const shown = auxiliary ? ess : essPrev
      const next: PfState = {
        t: t + 1,
        particles,
        logW: Float64Array.from(w, (v) => Math.log(v)),
        logEvidence: s.logEvidence + increment + lse,
        ancestors,
        means: [...s.means, mean],
        ess: [...s.ess, shown],
        resampled: [...s.resampled, resampledNow],
        history: [...s.history, primalOf(particles)],
        weights: [...s.weights, w],
      }
      if (ctx.wants('pf/ess')) ctx.log('pf/ess', shown)
      if (ctx.wants('pf/log-evidence')) ctx.log('pf/log-evidence', next.logEvidence)
      if (ctx.wants('pf/resampled')) ctx.log('pf/resampled', resampledNow ? 1 : 0)
      if (ctx.wants('pf/mean')) ctx.log('pf/mean', mean)
      if (ctx.wants('pf/weights')) ctx.log('pf/weights', w)
      if (ctx.wants('pf/particles')) ctx.log('pf/particles', primalOf(particles))
      if (ctx.wants('pf/ancestors')) ctx.log('pf/ancestors', ancestors)
      return next
    },
    done: (s) => (s.t >= T ? 'converged' : undefined),
    observe: (s) => ({ metrics: { ess: s.ess.at(-1) ?? 0, logEvidence: s.logEvidence } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PfState,
  }
}

/**
 * The bootstrap particle filter (Gordon, Salmond and Smith 1993) as a steppable algorithm: each step propagates the
 * particles through the transition, weights them by the likelihood of one observation and resamples when the effective
 * sample size falls below a threshold. The state is plain numbers and tensors and serialises; every step logs the ESS,
 * the evidence so far, whether it resampled, the weighted mean, the weights, the particles and their ancestry (see
 * {@link pfContract}).
 *
 * @param model - Initial distribution, transition and likelihood.
 * @param observations - `[T, ...]`.
 * @param options - Particle count, resampling scheme, ESS threshold and key.
 * @returns The algorithm; it is done after the last observation.
 * @concept particle-filter
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { key, normal } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import { add, mul, sub, sum, square } from 'omai-engine/tensor'
 * const model = {
 *   initial: (k, n) => normal(k, [n, 1]),
 *   transition: (k, x) => add(x, mul(normal(k, x.shape), 0.1)),
 *   logLikelihood: (z, x) => mul(sum(square(sub(x, z)), -1), -0.5),
 * }
 * const alg = filtering.bootstrapParticleAlgorithm(model, [[0.5], [0.4]], { particles: 50, key: key(1) })
 * runAlgorithm(alg, { maxSteps: 2 }).state.ess.length // => 2
 * ```
 */
export function bootstrapParticleAlgorithm(
  model: ParticleModel,
  observations: TensorLike,
  options: ParticleAlgorithmOptions = {},
): Algorithm<PfState, PfPath> {
  return pfAlgorithm('bootstrap-particle-filter', model, observations, options, false)
}

/**
 * The auxiliary particle filter (Pitt and Shephard 1999) as a steppable algorithm: each step resamples first by the
 * likelihood of each particle's point prediction, propagates, and corrects the weights by the ratio of the true to the
 * predicted likelihood. Every step resamples; the paths and state are those of {@link bootstrapParticleAlgorithm}.
 *
 * @param model - The model, with `predict`.
 * @param observations - `[T, ...]`.
 * @param options - Particle count, resampling scheme and key.
 * @returns The algorithm; it is done after the last observation.
 * @throws OmaiError with code `value` when the model has no `predict`.
 * @concept particle-filter
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { key, normal } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import { add, mul, sub, sum, square } from 'omai-engine/tensor'
 * const model = {
 *   initial: (k, n) => normal(k, [n, 1]),
 *   transition: (k, x) => add(x, mul(normal(k, x.shape), 0.1)),
 *   predict: (x) => x,
 *   logLikelihood: (z, x) => mul(sum(square(sub(x, z)), -1), -0.5),
 * }
 * const alg = filtering.auxiliaryParticleAlgorithm(model, [[0.5]], { particles: 50, key: key(2) })
 * runAlgorithm(alg, { maxSteps: 1 }).state.resampled // => [true]
 * ```
 */
export function auxiliaryParticleAlgorithm(
  model: ParticleModel,
  observations: TensorLike,
  options: Omit<ParticleAlgorithmOptions, 'essThreshold'> = {},
): Algorithm<PfState, PfPath> {
  return pfAlgorithm('auxiliary-particle-filter', model, observations, options, true)
}

function runPf(
  name: string,
  model: ParticleModel,
  observations: TensorLike,
  options: ParticleOptions,
  auxiliary: boolean,
): ParticleFilterResult {
  const { particles, resampling, essThreshold, key, ...run } = options
  const algorithm = pfAlgorithm(
    name,
    model,
    observations,
    {
      ...(particles === undefined ? {} : { particles }),
      ...(resampling === undefined ? {} : { resampling }),
      ...(essThreshold === undefined ? {} : { essThreshold }),
      ...(key === undefined ? {} : { key }),
    },
    auxiliary,
  )
  const T = asTensor(observations).shape[0] ?? 0
  const st = runAlgorithm(algorithm, { maxSteps: T, ...run }).state
  const d = st.means[0]?.length ?? 0
  const flat = new Float64Array(T * d)
  st.means.forEach((m, t) => {
    flat.set(m, t * d)
  })
  return {
    means: fromNumbers(flat, [T, d]),
    logEvidence: st.logEvidence,
    ess: st.ess,
    resampled: st.resampled,
    particles: st.history,
    weights: st.weights,
  }
}

/**
 * The bootstrap particle filter (Gordon, Salmond and Smith 1993): propagate particles through the transition, weight
 * them by the likelihood, and resample when the effective sample size falls below a threshold. The incremental
 * normalisers give an unbiased estimate of the evidence p(z_1..T). A traced algorithm with the ESS per step. It filters
 * a numeric state-space model (`initial`, `transition`, `logLikelihood` over tensors); `particleFilterAlgorithm` of
 * `probprog` filters a probabilistic program's markov loop instead, with the weights from its observed sites. The
 * resamplers differ on purpose: this one's (`resampleIndices`) follows filterpy and draws unsorted multinomial indices,
 * `sampling`'s (`resample`) follows BlackJAX and returns them sorted.
 *
 * @param model - Initial distribution, transition and likelihood.
 * @param observations - `[T, ...]`.
 * @param options - Particle count, resampling scheme and threshold, key and tracing.
 * @returns Weighted means, the log-evidence, ESS and the particle history.
 * @concept particle-filter
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { key, normal } from 'omai-engine/random'
 * import { add, mul, sub, sum, square } from 'omai-engine/tensor'
 * const model = {
 *   initial: (k, n) => normal(k, [n, 1]),
 *   transition: (k, x) => add(x, mul(normal(k, x.shape), 0.1)),
 *   logLikelihood: (z, x) => mul(sum(square(sub(x, z)), -1), -0.5),
 * }
 * filtering.bootstrapParticleFilter(model, [[0.5], [0.4]], { particles: 200, key: key(1) }).means.shape // => [2, 1]
 * ```
 */
export function bootstrapParticleFilter(
  model: ParticleModel,
  observations: TensorLike,
  options: ParticleOptions = {},
): ParticleFilterResult {
  return runPf('bootstrap-particle-filter', model, observations, options, false)
}

/**
 * The auxiliary particle filter (Pitt and Shephard 1999): resample first by the likelihood of each particle's point
 * prediction, propagate, and correct the weights by the ratio of the true to the predicted likelihood, which helps
 * when observations are informative.
 *
 * @param model - The model, with `predict`.
 * @param observations - `[T, ...]`.
 * @param options - Particle count, resampling scheme, key and tracing.
 * @returns Weighted means, the log-evidence, ESS and the particle history.
 * @throws OmaiError with code `value` when the model has no `predict`.
 * @concept particle-filter
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { key, normal } from 'omai-engine/random'
 * import { add, mul, sub, sum, square } from 'omai-engine/tensor'
 * const model = {
 *   initial: (k, n) => normal(k, [n, 1]),
 *   transition: (k, x) => add(x, mul(normal(k, x.shape), 0.1)),
 *   predict: (x) => x,
 *   logLikelihood: (z, x) => mul(sum(square(sub(x, z)), -1), -0.5),
 * }
 * filtering.auxiliaryParticleFilter(model, [[0.5]], { particles: 100, key: key(2) }).ess.length // => 1
 * ```
 */
export function auxiliaryParticleFilter(
  model: ParticleModel,
  observations: TensorLike,
  options: ParticleOptions = {},
): ParticleFilterResult {
  return runPf('auxiliary-particle-filter', model, observations, options, true)
}

/**
 * Particle smoothing by forward filtering, backward simulation (Godsill, Doucet and West 2004): draw M trajectories
 * backwards, choosing each earlier state among the filter's particles with weights w_t p(x_{t+1} | x_t).
 *
 * @param model - The model (with `logTransition`).
 * @param filtered - A particle filter's output.
 * @param options - Number of trajectories (default 100) and key.
 * @returns The smoothed means `[T, d]` and the sampled trajectories `[M, T, d]`.
 * @throws OmaiError with code `value` when the model has no transition density.
 * @concept particle-smoother
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { key, normal } from 'omai-engine/random'
 * import { add, mul, sub, sum, square } from 'omai-engine/tensor'
 * const model = {
 *   initial: (k, n) => normal(k, [n, 1]),
 *   transition: (k, x) => add(x, mul(normal(k, x.shape), 0.1)),
 *   logTransition: (x, next) => mul(sum(square(sub(x, next)), -1), -50),
 *   logLikelihood: (z, x) => mul(sum(square(sub(x, z)), -1), -0.5),
 * }
 * const f = filtering.bootstrapParticleFilter(model, [[0.5], [0.4]], { particles: 100, key: key(1) })
 * filtering.particleSmoother(model, f, { trajectories: 20, key: key(3) }).means.shape // => [2, 1]
 * ```
 */
export function particleSmoother(
  model: ParticleModel,
  filtered: ParticleFilterResult,
  options: { readonly trajectories?: number; readonly key?: Key } = {},
): { readonly means: Tensor; readonly trajectories: Tensor } {
  const logT = model.logTransition
  if (logT === undefined) throw new OmaiError('value', 'particleSmoother: the model needs logTransition')
  const M = options.trajectories ?? 100
  const T = filtered.particles.length
  const root = options.key ?? makeKey(0)
  const d = (filtered.particles[0]?.shape[1] ?? 1) | 0
  const paths = new Float64Array(M * T * d)
  const lastW = filtered.weights[T - 1] as Float64Array
  const lastC = cumulative(lastW)
  const pick = uniforms(foldIn(root, T), M).map((u) => searchSorted(lastC, u))
  const rowOf = (p: Tensor, i: number): number[] => Array.from(p.data.slice(i * d, (i + 1) * d))
  for (let j = 0; j < M; j++) {
    let row = rowOf(filtered.particles[T - 1] as Tensor, pick[j] ?? 0)
    paths.set(row, (j * T + T - 1) * d)
    for (let t = T - 2; t >= 0; t--) {
      const P = filtered.particles[t] as Tensor
      const w = filtered.weights[t] as Float64Array
      const lt = numbersOf(logT(P, fromNumbers(row, [d]), t))
      const lw = Float64Array.from(w, (v, i) => Math.log(v) + (lt[i] ?? 0))
      const { w: bw } = normalise(lw)
      const idx = searchSorted(cumulative(bw), uniforms(foldIn(foldIn(root, t), j), 1)[0] ?? 0)
      row = rowOf(P, idx)
      paths.set(row, (j * T + t) * d)
    }
  }
  const means = new Float64Array(T * d)
  for (let j = 0; j < M; j++)
    for (let k = 0; k < T * d; k++) means[k] = (means[k] ?? 0) + (paths[j * T * d + k] ?? 0) / M
  return { means: fromNumbers(means, [T, d]), trajectories: fromNumbers(paths, [M, T, d]) }
}
