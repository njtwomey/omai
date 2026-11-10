/**
 * Particle Markov chain Monte Carlo for state-space models (Andrieu, Doucet and Holenstein 2010): the parameters of a model
 * whose likelihood is only reached through a particle filter. Particle marginal Metropolis-Hastings runs a random walk on
 * the parameters with the filter's unbiased likelihood estimate (the pseudo-marginal kernel of `sampling`); particle Gibbs
 * alternates a conditional SMC sweep, which redraws the whole state path given the parameters (with ancestor sampling,
 * Lindsten, Jordan and Schon 2014), with a Metropolis move of the parameters given the path.
 */

import { logSumExpOf, OmaiError } from '../../base/index.ts'
import { foldIn, type Key, normal, uniform } from '../../random/index.ts'
import {
  type ChainProposal,
  type MCMCKernel,
  type MCMCState,
  metropolisAcceptance,
  type PseudoMarginalState,
  pseudoMarginalMetropolis,
} from '../../sampling/index.ts'
import { asTensor, reshape, sliceAxis, type TensorLike, type Value } from '../../tensor/index.ts'
import { fromNumbers, numbersOf } from '../ode/_vec.ts'
import {
  auxiliaryParticleFilter,
  bootstrapParticleFilter,
  type ParticleModel,
  type Resampling,
  resampleIndices,
} from './particle.ts'

/**
 * A family of state-space models indexed by parameters on the scale the sampler moves on: the prior over the parameters
 * and, for each value of them, the particle form of the model. Particle Gibbs also needs the model's transition density
 * (`logTransition`, for ancestor sampling) and the density of the state before the first transition (`logInitial`), because it
 * evaluates the density of a complete path.
 */
export interface StateSpaceFamily {
  /** The number of parameters. */
  readonly dim: number
  /** The prior over the parameters (its `logProb` is used). */
  readonly prior: Pick<ChainProposal, 'dim' | 'logProb' | 'sample'>
  /** The model at given parameters (initial state, transition, likelihood; `logTransition` for particle Gibbs). */
  readonly model: (theta: Float64Array) => ParticleModel
  /** log density of the state before the first transition, given the parameters (particle Gibbs). */
  readonly logInitial?: (theta: Float64Array, x: Float64Array) => number
  /** A name per parameter (default `theta[0]`, ...). */
  readonly names?: readonly string[]
}

/** Options of {@link particleLogLikelihood}. */
export interface ParticleLikelihoodOptions {
  /** Particles (default 100). */
  readonly particles?: number
  readonly resampling?: Resampling
  /** Resample when ESS / N falls below this (default 0.5). */
  readonly essThreshold?: number
  /** `bootstrap` (default; a model with a `guide` is moved by it) or `auxiliary` (the model needs `predict`). */
  readonly filter?: 'bootstrap' | 'auxiliary'
}

/**
 * The particle filter's estimate of the log likelihood of a series under a model: the log of an estimate of p(z_1..T) that is
 * unbiased on the natural scale. This is the quantity the pseudo-marginal kernel needs.
 *
 * @param model - The particle model.
 * @param observations - `[T, ...]`.
 * @param options - Particles, resampling, threshold and filter, and the key.
 * @returns The log evidence estimate.
 * @concept particle-mcmc
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { key } from 'omai-engine/random'
 * const model = filtering.gaussianParticleModel({ f: (x) => x, h: (x) => x, Q: [[0.1]], R: [[1]], x0: [0], P0: [[1]] })
 * Number.isFinite(filtering.particleLogLikelihood(model, [[0.5], [0.2]], { key: key(0), particles: 50 })) // => true
 * ```
 */
export function particleLogLikelihood(
  model: ParticleModel,
  observations: TensorLike,
  options: ParticleLikelihoodOptions & { readonly key: Key },
): number {
  const filter = options.filter === 'auxiliary' ? auxiliaryParticleFilter : bootstrapParticleFilter
  return filter(model, observations, {
    particles: options.particles ?? 100,
    ...(options.resampling === undefined ? {} : { resampling: options.resampling }),
    ...(options.essThreshold === undefined ? {} : { essThreshold: options.essThreshold }),
    key: options.key,
  }).logEvidence
}

/** Options of {@link pmmhKernel}. */
export interface PmmhOptions extends ParticleLikelihoodOptions {
  /** The key of the first likelihood estimate. */
  readonly key: Key
  /** The random-walk standard deviation on the parameters (default 2.38 / sqrt(dim)). */
  readonly scale?: number
  /** A full proposal covariance `[dim, dim]` (overrides `scale`). */
  readonly covariance?: TensorLike
}

/**
 * Particle marginal Metropolis-Hastings: the pseudo-marginal kernel of `sampling` with a particle filter as the likelihood
 * estimator. The chain on the parameters targets the exact posterior for any number of particles; more particles lower the
 * estimator's variance and raise the acceptance rate.
 *
 * @param family - The state-space family and the prior over its parameters.
 * @param observations - `[T, ...]`.
 * @param options - Particles, resampling, filter, proposal scale or covariance and the first key.
 * @returns The kernel; the state carries the likelihood estimate of the current parameters.
 * @concept particle-mcmc
 * @index function Particle marginal Metropolis-Hastings kernel: random walk on parameters with a particle filter's likelihood.
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { key, normal } from 'omai-engine/random'
 * const family = {
 *   dim: 1,
 *   prior: { dim: 1, logProb: (t) => -0.5 * (t[0] ?? 0) ** 2, sample: () => new Float64Array(1) },
 *   model: (t) => filtering.gaussianParticleModel({ f: (x) => x, h: (x) => x, Q: [[0.1]], R: [[Math.exp(t[0] ?? 0)]], x0: [0], P0: [[1]] }),
 * }
 * const kernel = filtering.pmmhKernel(family, [[0.5], [0.2], [0.4]], { key: key(0), particles: 40, scale: 0.3 })
 * Number.isFinite(kernel.init([0]).logLikelihood) // => true
 * ```
 */
export function pmmhKernel(
  family: StateSpaceFamily,
  observations: TensorLike,
  options: PmmhOptions,
): MCMCKernel<PseudoMarginalState> {
  const { key, scale, covariance, ...filter } = options
  return pseudoMarginalMetropolis(
    family.prior,
    (theta, k) => particleLogLikelihood(family.model(theta), observations, { ...filter, key: k }),
    {
      key,
      ...(scale === undefined ? {} : { scale }),
      ...(covariance === undefined ? {} : { covariance }),
    },
  )
}

const uniformsOf = (key: Key, n: number): number[] => Array.from(uniform(key, [n]).data)

function normalise(logW: Float64Array): Float64Array {
  const z = logSumExpOf(logW)
  return Float64Array.from(logW, (v) => Math.exp(v - z))
}

/** The first index whose cumulative weight reaches u. */
function categorical(w: ArrayLike<number>, u: number): number {
  let c = 0
  for (let i = 0; i < w.length; i++) {
    c += w[i] ?? 0
    if (u < c) return i
  }
  return w.length - 1
}

/** Options of {@link conditionalSmc}. */
export interface ConditionalSmcOptions {
  /** Particles including the reference (default 100). */
  readonly particles?: number
  /** `sampling` (default): the reference's ancestor is redrawn with ancestor sampling; `none`: it keeps its own ancestry. */
  readonly ancestor?: 'sampling' | 'none'
  readonly key: Key
}

/** The outcome of {@link conditionalSmc}. */
export interface ConditionalSmcResult {
  /** The new path `[(T + 1) * d]`, time-major: the state before the first transition, then the state after each. */
  readonly path: Float64Array
  /** The fraction of times at which the new path differs from the reference: near 0 means the sweep barely moved it. */
  readonly refresh: number
}

/**
 * One sweep of conditional SMC (the kernel of particle Gibbs): a particle filter that holds one particle on a given
 * reference path, then draws a new path from the final weights by tracing the ancestry back. With `ancestor: 'sampling'`
 * the reference particle's ancestor at each step is redrawn in proportion to the previous weights times the transition
 * density to the reference state, which lets the new path leave the reference's past even with few particles. The other
 * particles are resampled multinomially (the scheme the validity proof covers) at every step. The sweep leaves the
 * distribution of the path given the parameters and the observations invariant. Guided proposals are not used here; the
 * transition is the proposal.
 *
 * @param model - The particle model at fixed parameters (`logTransition` is required for ancestor sampling).
 * @param observations - `[T, ...]`.
 * @param reference - The current path `[(T + 1) * d]`, time-major.
 * @param options - Particles, ancestor rule and key.
 * @returns The new path and how much of it changed.
 * @throws OmaiError with code `value` for fewer than two particles, a reference of the wrong length, or ancestor
 *   sampling without a `logTransition`.
 * @concept particle-mcmc
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { key } from 'omai-engine/random'
 * const model = filtering.gaussianParticleModel({ f: (x) => x, h: (x) => x, Q: [[0.1]], R: [[1]], x0: [0], P0: [[1]] })
 * const r = filtering.conditionalSmc(model, [[0.5], [0.2]], new Float64Array(3), { key: key(0), particles: 20 })
 * r.path.length // => 3
 * ```
 */
export function conditionalSmc(
  model: ParticleModel,
  observations: TensorLike,
  reference: Float64Array,
  options: ConditionalSmcOptions,
): ConditionalSmcResult {
  const Z = asTensor(observations)
  const T = Z.shape[0] ?? 0
  const N = options.particles ?? 100
  const ancestorRule = options.ancestor ?? 'sampling'
  if (!(Number.isInteger(N) && N >= 2)) throw new OmaiError('value', 'conditionalSmc: need at least 2 particles')
  if (ancestorRule === 'sampling' && model.logTransition === undefined)
    throw new OmaiError('value', "conditionalSmc: ancestor sampling needs the model's logTransition")
  const d = reference.length / (T + 1)
  if (!Number.isInteger(d) || d < 1)
    throw new OmaiError(
      'value',
      `conditionalSmc: a reference of length ${reference.length} does not fit ${T} observations`,
    )
  const key = options.key
  const zAt = (t: number): Value => reshape(sliceAxis(Z, 0, t, t + 1), Z.shape.slice(1))
  const refAt = (s: number): Float64Array => reference.subarray(s * d, (s + 1) * d)
  const states: Float64Array[] = []
  const ancestors: Int32Array[] = []
  const x0 = Float64Array.from(numbersOf(model.initial(foldIn(key, 0), N)))
  x0.set(refAt(0), (N - 1) * d)
  states.push(x0)
  let logW = new Float64Array(N).fill(-Math.log(N))
  for (let t = 0; t < T; t++) {
    const kt = foldIn(key, t + 1)
    const prev = states[t] as Float64Array
    const w = normalise(logW)
    const idx = resampleIndices(Array.from(w), 'multinomial', uniformsOf(foldIn(kt, 0), N))
    const parents = new Int32Array(N)
    for (let i = 0; i < N - 1; i++) parents[i] = idx[i] ?? 0
    const parentRows = new Float64Array(N * d)
    for (let i = 0; i < N - 1; i++)
      parentRows.set(prev.subarray((parents[i] ?? 0) * d, ((parents[i] ?? 0) + 1) * d), i * d)
    const next = Float64Array.from(numbersOf(model.transition(foldIn(kt, 1), fromNumbers(parentRows, [N, d]), t)))
    const refNext = refAt(t + 1)
    next.set(refNext, (N - 1) * d)
    let refParent = N - 1
    if (ancestorRule === 'sampling') {
      const lt = numbersOf(
        (model.logTransition as NonNullable<ParticleModel['logTransition']>)(
          fromNumbers(prev, [N, d]),
          fromNumbers(refNext, [d]),
          t,
        ),
      )
      const lw = Float64Array.from(w, (v, j) => Math.log(v) + (lt[j] ?? 0))
      refParent = categorical(normalise(lw), uniformsOf(foldIn(kt, 2), 1)[0] ?? 0)
    }
    parents[N - 1] = refParent
    ancestors.push(parents)
    states.push(next)
    const ll = numbersOf(model.logLikelihood(zAt(t), fromNumbers(next, [N, d]), t))
    logW = Float64Array.from(ll, (v) => (Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v))
    if (!logW.some((v) => Number.isFinite(v)))
      throw new OmaiError('value', `conditionalSmc: every particle has zero weight at step ${t}`)
  }
  let b = categorical(normalise(logW), uniformsOf(foldIn(key, T + 1), 1)[0] ?? 0)
  const path = new Float64Array((T + 1) * d)
  for (let s = T; s >= 0; s--) {
    path.set((states[s] as Float64Array).subarray(b * d, (b + 1) * d), s * d)
    if (s > 0) b = (ancestors[s - 1] as Int32Array)[b] ?? 0
  }
  let changed = 0
  for (let s = 0; s <= T; s++) {
    let same = true
    for (let a = 0; a < d; a++) if (path[s * d + a] !== reference[s * d + a]) same = false
    if (!same) changed += 1
  }
  return { path, refresh: changed / (T + 1) }
}

/**
 * The log density of a complete path and the observations at given parameters: the log prior, the log density of the first
 * state, the transition densities along the path and the likelihood of every observation. Particle Gibbs moves the parameters
 * on this density.
 *
 * @param family - The family (with `logInitial` and a model with `logTransition`).
 * @param theta - The parameters.
 * @param observations - `[T, ...]`.
 * @param path - `[(T + 1) * d]`, time-major.
 * @returns log p(theta) + log p(x, z | theta).
 * @throws OmaiError with code `value` when the family has no `logInitial` or the model no `logTransition`.
 * @concept particle-mcmc
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * const family = {
 *   dim: 1,
 *   prior: { dim: 1, logProb: () => 0, sample: () => new Float64Array(1) },
 *   model: () => filtering.gaussianParticleModel({ f: (x) => x, h: (x) => x, Q: [[1]], R: [[1]], x0: [0], P0: [[1]] }),
 *   logInitial: (_t, x) => -0.5 * (x[0] ?? 0) ** 2 - 0.5 * Math.log(2 * Math.PI),
 * }
 * Number.isFinite(filtering.completeDataLogDensity(family, new Float64Array(1), [[0.1]], new Float64Array(2))) // => true
 * ```
 */
export function completeDataLogDensity(
  family: StateSpaceFamily,
  theta: Float64Array,
  observations: TensorLike,
  path: Float64Array,
): number {
  if (family.logInitial === undefined)
    throw new OmaiError('value', 'completeDataLogDensity: the family needs logInitial')
  // Outside the prior's support the model may not even exist (a negative variance), so the prior goes first.
  const lp = family.prior.logProb(theta)
  if (lp === Number.NEGATIVE_INFINITY) return lp
  const model = family.model(theta)
  const logT = model.logTransition
  if (logT === undefined) throw new OmaiError('value', 'completeDataLogDensity: the model needs logTransition')
  const Z = asTensor(observations)
  const T = Z.shape[0] ?? 0
  const d = path.length / (T + 1)
  let total = lp + family.logInitial(theta, path.subarray(0, d))
  for (let t = 0; t < T; t++) {
    const prev = fromNumbers(path.subarray(t * d, (t + 1) * d), [1, d])
    const next = path.subarray((t + 1) * d, (t + 2) * d)
    total += numbersOf(logT(prev, fromNumbers(next, [d]), t))[0] ?? Number.NEGATIVE_INFINITY
    const z = reshape(sliceAxis(Z, 0, t, t + 1), Z.shape.slice(1))
    total += numbersOf(model.logLikelihood(z, fromNumbers(next, [1, d]), t))[0] ?? Number.NEGATIVE_INFINITY
  }
  return Number.isNaN(total) ? Number.NEGATIVE_INFINITY : total
}

/** The state of {@link particleGibbs}: the parameters, the current path and what the last sweep did. */
export interface ParticleGibbsState extends MCMCState {
  /** The current state path `[(T + 1) * d]`, time-major. */
  readonly path: Float64Array
  /** The fraction of times at which the last conditional SMC sweep changed the path. */
  readonly refresh: number
  /** The acceptance rate of the last parameter moves. */
  readonly thetaAcceptance: number
}

/** Options of {@link particleGibbs}. */
export interface ParticleGibbsOptions {
  /** The key of the starting path (`init` has none of its own). */
  readonly key: Key
  /** Particles of the conditional SMC sweep, the reference included (default 50). */
  readonly particles?: number
  /** `sampling` (default) or `none`: see {@link conditionalSmc}. */
  readonly ancestor?: 'sampling' | 'none'
  /** Metropolis moves of the parameters per sweep (default 1). */
  readonly thetaMoves?: number
  /** The random-walk standard deviation of the parameter moves (default 2.38 / sqrt(dim)). */
  readonly scale?: number
}

/**
 * Particle Gibbs (Andrieu, Doucet and Holenstein 2010) with ancestor sampling: each transition redraws the whole state path
 * by a conditional SMC sweep given the parameters, then moves the parameters by random-walk Metropolis on the complete-data
 * density given the path. The chain's marginal on the parameters is the exact posterior; unlike PMMH it needs the model's
 * transition density, but its cost per sweep is one filter, and it does not need a likelihood estimate. The starting path is a
 * draw of the model forward from its prior, which the chain forgets.
 *
 * @param family - The family, with `logInitial` and a model with `logTransition`.
 * @param observations - `[T, ...]`.
 * @param options - The key of the starting path, particles, ancestor rule, parameter moves and scale.
 * @returns The kernel; the position is the parameters, the state also carries the path.
 * @throws OmaiError with code `value` when the family has no `logInitial`.
 * @concept particle-mcmc
 * @index function Particle Gibbs with ancestor sampling: conditional SMC path sweeps and parameter moves.
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { key } from 'omai-engine/random'
 * const family = {
 *   dim: 1,
 *   prior: { dim: 1, logProb: (t) => -0.5 * (t[0] ?? 0) ** 2, sample: () => new Float64Array(1) },
 *   model: (t) => filtering.gaussianParticleModel({ f: (x) => x, h: (x) => x, Q: [[Math.exp(t[0] ?? 0)]], R: [[1]], x0: [0], P0: [[1]] }),
 *   logInitial: (_t, x) => -0.5 * (x[0] ?? 0) ** 2 - 0.5 * Math.log(2 * Math.PI),
 * }
 * const kernel = filtering.particleGibbs(family, [[0.5], [0.2]], { key: key(0), particles: 20 })
 * kernel.init([0]).path.length // => 3
 * ```
 */
export function particleGibbs(
  family: StateSpaceFamily,
  observations: TensorLike,
  options: ParticleGibbsOptions,
): MCMCKernel<ParticleGibbsState> {
  if (family.logInitial === undefined) throw new OmaiError('value', 'particleGibbs: the family needs logInitial')
  const Z = asTensor(observations)
  const T = Z.shape[0] ?? 0
  const dim = family.dim
  const scale = options.scale ?? 2.38 / Math.sqrt(dim)
  const moves = options.thetaMoves ?? 1
  const N = options.particles ?? 50
  if (!(scale > 0 && Number.isFinite(scale))) throw new OmaiError('value', 'particleGibbs: the scale must be positive')
  return {
    name: 'particleGibbs',
    dim,
    config: { dim, particles: N, ancestor: options.ancestor ?? 'sampling', thetaMoves: moves, scale },
    init: (position) => {
      if (position.length !== dim) throw new OmaiError('shape', `particleGibbs: the position needs ${dim} values`)
      const theta = Float64Array.from(position)
      const model = family.model(theta)
      let x = model.initial(foldIn(options.key, 0), 1)
      const d = numbersOf(x).length
      const path = new Float64Array((T + 1) * d)
      path.set(numbersOf(x), 0)
      for (let t = 0; t < T; t++) {
        x = model.transition(foldIn(options.key, t + 1), x, t)
        path.set(numbersOf(x), (t + 1) * d)
      }
      return {
        position: theta,
        path,
        refresh: 1,
        thetaAcceptance: Number.NaN,
        logDensity: completeDataLogDensity(family, theta, observations, path),
      }
    },
    step: (key, state) => {
      const swept = conditionalSmc(family.model(state.position), observations, state.path, {
        key: foldIn(key, 0),
        particles: N,
        ...(options.ancestor === undefined ? {} : { ancestor: options.ancestor }),
      })
      let theta = state.position
      let logDensity = completeDataLogDensity(family, theta, observations, swept.path)
      let accepted = 0
      for (let m = 0; m < moves; m++) {
        const km = foldIn(foldIn(key, 1), m)
        const z = normal(foldIn(km, 0), [dim]).data
        const proposal = theta.map((v, i) => v + scale * (z[i] ?? 0))
        const lp = completeDataLogDensity(family, proposal, observations, swept.path)
        const alpha = metropolisAcceptance(logDensity, lp, 0, 0)
        if ((uniformsOf(foldIn(km, 1), 1)[0] ?? 1) < alpha) {
          theta = proposal
          logDensity = lp
          accepted += 1
        }
      }
      return {
        state: {
          position: theta,
          path: swept.path,
          refresh: swept.refresh,
          thetaAcceptance: accepted / moves,
          logDensity,
        },
        info: { acceptanceRate: accepted / moves, accepted: accepted > 0 },
      }
    },
  }
}
