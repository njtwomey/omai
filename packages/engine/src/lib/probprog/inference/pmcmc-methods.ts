/**
 * Particle MCMC as two inference methods for state-space models whose parameters are unknown: particle marginal
 * Metropolis-Hastings (a random walk on the parameters with a particle filter's unbiased likelihood estimate) and particle
 * Gibbs with ancestor sampling (a conditional SMC sweep over the whole state path, then a Metropolis move of the
 * parameters). Each is its own method with its own knobs and tape (`pmcmc/accept`, `pmcmc/logLik`, the parameters and, for
 * particle Gibbs, the path), one iteration per step, so a run cancels within one filter, reads its posterior over the
 * parameters at any step and resumes from a serialised state. The model is a `StateSpaceFamily` (a prior over the
 * parameters and a particle model for each value of them) with the observations, not a spec: the spec layer has no
 * recurrences yet.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import {
  type ParticleGibbsState,
  particleGibbs,
  pmmhKernel,
  type StateSpaceFamily,
} from '../../dynamics/filtering/index.ts'
import { foldIn, type Key, key as makeKey } from '../../random/index.ts'
import type { PseudoMarginalState } from '../../sampling/index.ts'
import { choice, type DictSpace, dict, int, real } from '../../space/index.ts'
import {
  add,
  asTensor,
  div,
  eye,
  mul,
  outer,
  reshape,
  sliceAxis,
  sub,
  type Tensor,
  type TensorLike,
  take,
  toArray,
  toFlat,
  zeros,
} from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { definePosterior, type Posterior, samplePosterior } from '../posterior.ts'
import { appended, no, num, pick, posteriorJson, yes } from './_support.ts'
import { defineInferenceMethod, type InferenceMethod, type Verdict } from './method.ts'

/** The paths particle marginal Metropolis-Hastings logs. */
export type PmmhPath = 'pmcmc/accept' | 'pmcmc/acceptRate' | 'pmcmc/logLik' | 'pmcmc/theta' | 'pmcmc/snapshot'

/** The paths particle Gibbs logs: those of PMMH and the state path and how much of it a sweep changed. */
export type PmcmcPath = PmmhPath | 'pmcmc/path' | 'pmcmc/refresh'

/** The state of {@link pmmhMethod}: the chain so far and what the adaptive proposal has seen. */
export interface PmmhState {
  /** Iterations done. */
  readonly t: number
  /** The current parameters `[dim]` and the log of the likelihood estimate carried with them. */
  readonly position: Float64Array
  readonly logLikelihood: number
  /** log prior + log likelihood estimate at the current point. */
  readonly logDensity: number
  /** The chain, row-major `[t, dim]`, and the log likelihood estimate at each iteration. */
  readonly draws: Float64Array
  readonly logLiks: Float64Array
  readonly accepted: number
  /** Running sums of the chain and of its outer products (the adaptive proposal's covariance). */
  readonly moments: Moments
}

/** Running sums of a chain: the count, the sum of the points and the sum of their outer products `[dim, dim]`. */
export interface Moments {
  readonly count: number
  readonly sum: Float64Array
  readonly sumSq: Float64Array
}

const emptyMoments = (dim: number): Moments => ({
  count: 0,
  sum: Float64Array.from(toFlat(zeros([dim]))),
  sumSq: Float64Array.from(toFlat(zeros([dim * dim]))),
})

function addMoment(m: Moments, x: Float64Array): Moments {
  const sum = Float64Array.from(toFlat(add(asTensor(m.sum), asTensor(x))))
  const sumSq = Float64Array.from(toFlat(add(asTensor(m.sumSq), reshape(outer(x, x), [-1]))))
  return { count: m.count + 1, sum, sumSq }
}

/** The sample covariance of the sums, row-major `[dim, dim]`. */
function covarianceOf(m: Moments): Float64Array {
  const n = m.count
  const centred = sub(asTensor(m.sumSq), div(reshape(outer(m.sum, m.sum), [-1]), n))
  return Float64Array.from(toFlat(div(centred, n - 1)))
}

/** The state of {@link particleGibbsMethod}. */
export interface PgState {
  readonly t: number
  readonly position: Float64Array
  readonly path: Float64Array
  /** log p(theta) + log p(path, data | theta) at the current point. */
  readonly logDensity: number
  readonly refresh: number
  readonly draws: Float64Array
  readonly logJoints: Float64Array
  /** Parameter moves accepted and attempted. */
  readonly accepted: number
  readonly attempted: number
}

const FILTERS = ['bootstrap', 'auxiliary'] as const
const SCHEMES = ['systematic', 'multinomial', 'stratified', 'residual'] as const

/** The knobs of {@link pmmhMethod} (the budget and snapshot knobs are added to them). */
export const pmmhKnobs: DictSpace = dict({
  numIterations: int(1, 100_000_000, {
    default: 1000,
    label: 'iterations',
    description: 'the run ends after this many iterations',
  }),
  particles: int(2, 1_000_000, {
    default: 100,
    label: 'particles',
    description:
      'particles of the filter that estimates the likelihood; more lowers its variance and raises acceptance',
  }),
  filter: choice([...FILTERS], { default: 'bootstrap', label: 'filter' }),
  resampling: choice([...SCHEMES], { default: 'systematic', label: 'resampling' }),
  essThreshold: real(0.01, 1, { default: 0.5, label: 'resampling threshold (ESS / N)' }),
  proposal: choice(['random-walk', 'adaptive'], {
    default: 'random-walk',
    label: 'proposal',
    description: 'a fixed isotropic random walk, or the covariance of the chain so far once it has enough draws',
  }),
  stepSize: real(1e-6, 100, { default: 0.3, label: 'step size', description: 'the random-walk standard deviation' }),
  burnIn: real(0, 0.9, {
    default: 0.2,
    label: 'burn-in',
    description: 'the fraction of the chain the posterior discards',
  }),
})

/** The knobs of {@link particleGibbsMethod} (the budget and snapshot knobs are added to them). */
export const particleGibbsKnobs: DictSpace = dict({
  numIterations: int(1, 100_000_000, {
    default: 1000,
    label: 'iterations',
    description: 'the run ends after this many sweeps',
  }),
  particles: int(2, 1_000_000, {
    default: 50,
    label: 'particles',
    description: 'particles of the conditional SMC sweep, the reference included',
  }),
  ancestor: choice(['sampling', 'none'], {
    default: 'sampling',
    label: 'ancestor',
    description: 'ancestor sampling for the reference path, or none (the reference keeps its own ancestry)',
  }),
  thetaMoves: int(1, 1000, { default: 1, label: 'parameter moves per sweep' }),
  stepSize: real(1e-6, 100, {
    default: 0.3,
    label: 'step size',
    description: 'the random-walk standard deviation of the parameter moves',
  }),
  burnIn: real(0, 0.9, {
    default: 0.2,
    label: 'burn-in',
    description: 'the fraction of the chain the posterior discards',
  }),
})

const namesOf = (family: StateSpaceFamily): string[] =>
  Array.from({ length: family.dim }, (_, i) => family.names?.[i] ?? (family.dim === 1 ? 'theta' : `theta[${i}]`))

/** The posterior over the parameters from a chain `[n, dim]`, after discarding the burn-in fraction. */
function chainPosterior(
  family: StateSpaceFamily,
  draws: Float64Array,
  burnIn: number,
  status: Posterior['status'],
  engine: string,
  diagnostics: Readonly<Record<string, Json>>,
): Posterior {
  const dim = family.dim
  const n = draws.length / dim
  const from = Math.min(n - 2, Math.floor(burnIn * n))
  const m = n - from
  if (m < 2)
    return definePosterior({
      form: 'samples',
      engine,
      sites: [],
      status: 'failed',
      draws: {},
      diagnostics: { reason: 'fewer than two draws so far', ...diagnostics },
    })
  const names = namesOf(family)
  const sites: Record<string, Tensor> = {}
  const kept = sliceAxis(reshape(asTensor(draws), [n, dim]), 0, from, n)
  for (let k = 0; k < dim; k++) sites[names[k] as string] = take(kept, k, 1) as Tensor
  return samplePosterior(sites, {
    engine,
    status,
    chainShape: [1, m],
    diagnostics: { ...diagnostics, burnIn: from, draws: m },
  })
}

const accept = (state: { readonly accepted: number }, t: number): number => (t > 0 ? state.accepted / t : Number.NaN)

/**
 * Particle marginal Metropolis-Hastings on a state-space model as a method with a fine step: one Metropolis iteration per
 * step, each with a fresh particle-filter likelihood estimate at the proposed parameters (the estimate at the current ones
 * is kept, which is what makes the chain exact whatever the filter's noise). The tape has the acceptance, the running
 * acceptance rate, the likelihood estimate and the parameters; `posterior(state)` is the chain so far after the burn-in. The
 * proposal is a fixed isotropic random walk or, once there are draws, the chain's own covariance. The state serialises, so
 * a run resumes and forks exactly.
 *
 * @param family - The state-space family: a prior over the parameters and a particle model at each value of them.
 * @param observations - `[T, ...]`.
 * @param options - The name, the starting parameters (default a draw from the prior) and the first key.
 * @returns The method, with `check` and `posterior`.
 * @concept particle-mcmc
 * @index estimator Particle marginal Metropolis-Hastings: random walk on state-space parameters with a particle filter.
 * @example
 * ```ts
 * import { pmmhMethod } from 'omai-engine/probprog/inference'
 * import { filtering } from 'omai-engine/dynamics'
 * import { session } from 'omai-engine/trace'
 * const family = {
 *   dim: 1,
 *   names: ['logR'],
 *   prior: { dim: 1, logProb: (t) => -0.5 * (t[0] ?? 0) ** 2, sample: () => new Float64Array(1) },
 *   model: (t) => filtering.gaussianParticleModel({ f: (x) => x, h: (x) => x, Q: [[0.1]], R: [[Math.exp(t[0] ?? 0)]], x0: [0], P0: [[1]] }),
 * }
 * const method = pmmhMethod(family, [[0.5], [0.2], [0.4], [0.1]])
 * const s = session(method, { steps: 20, mode: 'eager', config: { numIterations: 20, particles: 30 } })
 * [s.t, method.posterior?.(s.state, s.config).sites] // => [20, ['logR']]
 * ```
 */
export function pmmhMethod(
  family: StateSpaceFamily,
  observations: TensorLike,
  options: { readonly name?: string; readonly start?: readonly number[] } = {},
): InferenceMethod<PmmhState, Config, PmmhPath, StateSpaceFamily> {
  const name = options.name ?? 'pmmh'
  const dim = family.dim
  const check = (f: StateSpaceFamily): Verdict =>
    f.dim >= 1
      ? yes('a particle filter estimates the likelihood at each proposed parameter')
      : no('no parameters', ['parameters'])
  const kernelOf = (
    config: Config,
    key: Key,
    moments: Moments | undefined,
    t: number,
  ): ReturnType<typeof pmmhKernel> => {
    const adaptive =
      pick(config, 'proposal') === 'adaptive' && moments !== undefined && moments.count >= Math.max(20, 5 * dim)
    const cov = adaptive && moments !== undefined ? covarianceOf(moments) : undefined
    const covariance =
      cov === undefined
        ? undefined
        : (toArray(
            add(mul(reshape(asTensor(cov), [dim, dim]), (2.38 * 2.38) / dim), mul(eye(dim), 1e-9)),
          ) as number[][])
    void t
    return pmmhKernel(family, observations, {
      key,
      particles: num(config, 'particles'),
      filter: pick(config, 'filter') as (typeof FILTERS)[number],
      resampling: pick(config, 'resampling') as (typeof SCHEMES)[number],
      essThreshold: num(config, 'essThreshold'),
      ...(covariance === undefined ? { scale: num(config, 'stepSize') } : { covariance }),
    })
  }
  const posterior = (s: PmmhState, config: Config): Posterior =>
    chainPosterior(
      family,
      s.draws,
      num(config, 'burnIn'),
      s.t >= num(config, 'numIterations') ? 'completed' : 'stopped',
      'pmmh',
      {
        method: 'particle marginal Metropolis-Hastings',
        particles: num(config, 'particles'),
        acceptanceRate: accept(s, s.t),
      },
    )
  return defineInferenceMethod<PmmhState, Config, PmmhPath, StateSpaceFamily>({
    name,
    family: 'smc',
    description:
      'particle marginal Metropolis-Hastings: a random walk on the parameters with a particle-filter likelihood',
    config: pmmhKnobs,
    check,
    posterior,
    defaults: { 'pmcmc/accept': Log(), 'pmcmc/logLik': Log() },
    contract: (config): Contract<PmmhState, PmmhPath> =>
      defineContract({
        'pmcmc/accept': scalar({ description: '1 when the proposed parameters were accepted, else 0' }),
        'pmcmc/acceptRate': scalar({ description: 'the fraction of iterations accepted so far' }),
        'pmcmc/logLik': scalar({
          description: "the log of the particle filter's likelihood estimate at the current parameters",
          direction: 'higher',
        }),
        'pmcmc/theta': tensor({ role: 'state', shape: ['param'], description: 'the current parameters' }),
        'pmcmc/snapshot': derived<PmmhState>({
          cost: 'dear',
          description: 'the chain so far after the burn-in as JSON: means, sds, quantiles, ESS',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config, key): Algorithm<PmmhState, PmmhPath> => {
      const total = num(config, 'numIterations')
      return {
        name: 'probprog/pmmh',
        config: { numIterations: total, particles: num(config, 'particles') },
        init: () => {
          const start =
            options.start === undefined ? family.prior.sample(foldIn(key, 0)) : Float64Array.from(options.start)
          if (start.length !== dim) throw new OmaiError('shape', `${name}: the start needs ${dim} parameters`)
          const k = kernelOf(config, foldIn(key, 1), undefined, 0)
          const s = k.init(start)
          return {
            t: 0,
            position: s.position,
            logLikelihood: s.logLikelihood,
            logDensity: s.logDensity,
            draws: new Float64Array(0),
            logLiks: new Float64Array(0),
            accepted: 0,
            moments: emptyMoments(dim),
          }
        },
        step: (s, ctx) => {
          const k = kernelOf(config, foldIn(key, 1), s.moments, s.t)
          const from: PseudoMarginalState = {
            position: s.position,
            logLikelihood: s.logLikelihood,
            logDensity: s.logDensity,
          }
          const r = k.step(ctx.key, from)
          const draws = appended(s.draws, r.state.position)
          const logLiks = appended(s.logLiks, [r.state.logLikelihood])
          const next: PmmhState = {
            t: s.t + 1,
            position: r.state.position,
            logLikelihood: r.state.logLikelihood,
            logDensity: r.state.logDensity,
            draws,
            logLiks,
            accepted: s.accepted + (r.info.accepted ? 1 : 0),
            moments: addMoment(s.moments, r.state.position),
          }
          ctx.log('pmcmc/accept', r.info.accepted ? 1 : 0)
          ctx.log('pmcmc/acceptRate', accept(next, next.t))
          ctx.log('pmcmc/logLik', r.state.logLikelihood)
          if (ctx.wants('pmcmc/theta')) ctx.log('pmcmc/theta', r.state.position)
          return next
        },
        done: (s) => (s.t >= total ? 'completed' : undefined),
        serialise: (s) => stateToJson(s),
        deserialise: (j: Json) => stateFromJson(j) as PmmhState,
      }
    },
  })
}

/**
 * Particle Gibbs with ancestor sampling on a state-space model as a method with a fine step: one sweep per step, a conditional
 * SMC sweep that redraws the whole state path given the parameters, then Metropolis moves of the parameters given the path.
 * The tape has the acceptance of the parameter moves, the complete-data log density, how much of the path the sweep
 * changed (`pmcmc/refresh`: near 0 means the path is stuck to its reference, which ancestor sampling prevents), the
 * parameters and, when asked for, the path. The model needs its transition density and the density of its first state
 * (`logInitial`); in exchange it needs no likelihood estimate and its cost per sweep is one filter.
 *
 * @param family - The family, with `logInitial` and a model with `logTransition`.
 * @param observations - `[T, ...]`.
 * @param options - The name, the starting parameters (default a draw from the prior) and the first key.
 * @returns The method, with `check` and `posterior`.
 * @concept particle-mcmc
 * @index estimator Particle Gibbs with ancestor sampling: conditional SMC paths and parameter moves.
 * @example
 * ```ts
 * import { particleGibbsMethod } from 'omai-engine/probprog/inference'
 * import { filtering } from 'omai-engine/dynamics'
 * import { session } from 'omai-engine/trace'
 * const family = {
 *   dim: 1,
 *   names: ['logQ'],
 *   prior: { dim: 1, logProb: (t) => -0.5 * (t[0] ?? 0) ** 2, sample: () => new Float64Array(1) },
 *   model: (t) => filtering.gaussianParticleModel({ f: (x) => x, h: (x) => x, Q: [[Math.exp(t[0] ?? 0)]], R: [[1]], x0: [0], P0: [[1]] }),
 *   logInitial: (_t, x) => -0.5 * (x[0] ?? 0) ** 2 - 0.5 * Math.log(2 * Math.PI),
 * }
 * const method = particleGibbsMethod(family, [[0.5], [0.2], [0.4], [0.1]])
 * const s = session(method, { steps: 10, mode: 'eager', config: { numIterations: 10, particles: 12 } })
 * [s.t, method.posterior?.(s.state, s.config).sites] // => [10, ['logQ']]
 * ```
 */
export function particleGibbsMethod(
  family: StateSpaceFamily,
  observations: TensorLike,
  options: { readonly name?: string; readonly start?: readonly number[] } = {},
): InferenceMethod<PgState, Config, PmcmcPath, StateSpaceFamily> {
  const name = options.name ?? 'particle-gibbs'
  const dim = family.dim
  const check = (f: StateSpaceFamily): Verdict => {
    if (f.logInitial === undefined)
      return no(
        'particle Gibbs needs the density of the first state',
        ['logInitial'],
        'give the family a logInitial, or use pmmh',
      )
    const model = f.model(f.prior.sample(makeKey(0)))
    if (model.logTransition === undefined)
      return no(
        'particle Gibbs needs the transition density',
        ['logTransition'],
        'give the model a logTransition, or use pmmh',
      )
    return yes('conditional SMC with ancestor sampling over the path, then moves of the parameters')
  }
  const ensure = (): void => {
    const v = check(family)
    if (!v.applies) throw new OmaiError('value', `${name}: ${v.reason}${v.hint === undefined ? '' : ` (${v.hint})`}`)
  }
  const posterior = (s: PgState, config: Config): Posterior =>
    chainPosterior(
      family,
      s.draws,
      num(config, 'burnIn'),
      s.t >= num(config, 'numIterations') ? 'completed' : 'stopped',
      'particle-gibbs',
      {
        method: 'particle Gibbs with ancestor sampling',
        particles: num(config, 'particles'),
        acceptanceRate: s.attempted > 0 ? s.accepted / s.attempted : Number.NaN,
        refresh: s.refresh,
      },
    )
  return defineInferenceMethod<PgState, Config, PmcmcPath, StateSpaceFamily>({
    name,
    family: 'smc',
    description:
      'particle Gibbs with ancestor sampling: conditional SMC over the state path, then moves of the parameters',
    config: particleGibbsKnobs,
    check,
    posterior,
    defaults: { 'pmcmc/accept': Log(), 'pmcmc/logLik': Log(), 'pmcmc/refresh': Log() },
    contract: (config): Contract<PgState, PmcmcPath> =>
      defineContract({
        'pmcmc/accept': scalar({ description: "the fraction of this sweep's parameter moves accepted" }),
        'pmcmc/acceptRate': scalar({ description: 'the fraction of parameter moves accepted so far' }),
        'pmcmc/logLik': scalar({
          description: 'log p(theta) + log p(path, data | theta), the complete-data log density at the current state',
          direction: 'higher',
        }),
        'pmcmc/refresh': scalar({ description: 'the fraction of times at which the sweep changed the path' }),
        'pmcmc/theta': tensor({ role: 'state', shape: ['param'], description: 'the current parameters' }),
        'pmcmc/path': tensor({
          role: 'state',
          shape: ['time', 'state'],
          description: 'the current state path, the state before the first transition first',
        }),
        'pmcmc/snapshot': derived<PgState>({
          cost: 'dear',
          description: 'the chain so far after the burn-in as JSON: means, sds, quantiles, ESS',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config, key): Algorithm<PgState, PmcmcPath> => {
      ensure()
      const total = num(config, 'numIterations')
      const kernel = particleGibbs(family, observations, {
        key: foldIn(key, 1),
        particles: num(config, 'particles'),
        ancestor: pick(config, 'ancestor') as 'sampling' | 'none',
        thetaMoves: num(config, 'thetaMoves'),
        scale: num(config, 'stepSize'),
      })
      const moves = num(config, 'thetaMoves')
      return {
        name: 'probprog/particle-gibbs',
        config: { numIterations: total, particles: num(config, 'particles') },
        init: () => {
          ensure()
          const start =
            options.start === undefined ? family.prior.sample(foldIn(key, 0)) : Float64Array.from(options.start)
          if (start.length !== dim) throw new OmaiError('shape', `${name}: the start needs ${dim} parameters`)
          const s = kernel.init(start)
          return {
            t: 0,
            position: s.position,
            path: s.path,
            logDensity: s.logDensity,
            refresh: s.refresh,
            draws: new Float64Array(0),
            logJoints: new Float64Array(0),
            accepted: 0,
            attempted: 0,
          }
        },
        step: (s, ctx) => {
          const r = kernel.step(ctx.key, {
            position: s.position,
            path: s.path,
            logDensity: s.logDensity,
            refresh: s.refresh,
            thetaAcceptance: Number.NaN,
          } satisfies ParticleGibbsState)
          const draws = appended(s.draws, r.state.position)
          const logJoints = appended(s.logJoints, [r.state.logDensity])
          const gained = Math.round(r.state.thetaAcceptance * moves)
          const next: PgState = {
            t: s.t + 1,
            position: r.state.position,
            path: r.state.path,
            logDensity: r.state.logDensity,
            refresh: r.state.refresh,
            draws,
            logJoints,
            accepted: s.accepted + gained,
            attempted: s.attempted + moves,
          }
          ctx.log('pmcmc/accept', r.state.thetaAcceptance)
          ctx.log('pmcmc/acceptRate', next.accepted / next.attempted)
          ctx.log('pmcmc/logLik', r.state.logDensity)
          ctx.log('pmcmc/refresh', r.state.refresh)
          if (ctx.wants('pmcmc/theta')) ctx.log('pmcmc/theta', r.state.position)
          if (ctx.wants('pmcmc/path')) ctx.log('pmcmc/path', r.state.path)
          return next
        },
        done: (s) => (s.t >= total ? 'completed' : undefined),
        serialise: (s) => stateToJson(s),
        deserialise: (j: Json) => stateFromJson(j) as PgState,
      }
    },
  })
}
