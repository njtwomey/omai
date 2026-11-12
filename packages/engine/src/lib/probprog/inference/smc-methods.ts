/**
 * Sequential Monte Carlo as two inference methods. Tempered SMC moves a population of particles from the prior to the
 * posterior, one temperature per step (`smcAlgorithm` of `sampling`, on the prior and likelihood of the program), so a page
 * watches the temperatures fill in and reads the particles at the current temperature while it runs. The particle filter
 * runs the bootstrap filter over the program's `markov` loop, one time step per step, with the filtering posterior at
 * every step. Both serialise their states, so a run resumes and forks exactly; the one-call `temperedSmc`, `particleFilter`
 * and the planner's `smc` engine are unchanged.
 */

import type { Json } from '../../base/index.ts'
import { type Key, key as makeKey } from '../../random/index.ts'
import { type SMCState, smcAlgorithm } from '../../sampling/index.ts'
import { choice, type DictSpace, dict, int, real } from '../../space/index.ts'
import { asTensor, primalOf, type Tensor } from '../../tensor/index.ts'
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
import {
  filterBlocker,
  type ParticleFilterState,
  particleFilterAlgorithm,
  particleFilterPosterior,
  smcEngine,
  type TemperedProblem,
  temperedPosterior,
  temperedProblem,
} from '../engines/smc.ts'
import type { ProgramTrace } from '../handlers.ts'
import type { Posterior } from '../posterior.ts'
import {
  checkOf,
  guard,
  isProgramAnalysis,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  pick,
  posteriorJson,
  priorOnly,
  programOf,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod, type Verdict } from './method.ts'

/** The paths the tempered SMC method logs. */
export type TemperedSmcPath =
  | 'smc/temperature'
  | 'smc/ess'
  | 'smc/logEvidence'
  | 'smc/acceptance'
  | 'smc/particles'
  | 'smc/snapshot'

const SCHEMES = ['systematic', 'multinomial', 'stratified', 'residual'] as const

/** The knobs of {@link temperedSmcMethod} (the budget and snapshot knobs are added to them). */
export const temperedSmcKnobs: DictSpace = dict({
  numParticles: int(2, 1_000_000, { default: 500, label: 'particles' }),
  mcmcSteps: int(0, 1000, { default: 5, label: 'MCMC steps per stage' }),
  targetEss: real(0.05, 0.99, {
    default: 0.5,
    label: 'target ESS',
    description: 'the fraction of particles the ESS is held at between temperatures',
  }),
  resampling: choice([...SCHEMES], { default: 'systematic', label: 'resampling' }),
})

/**
 * Adaptive-tempering SMC on a model as a method with a fine step: one temperature per step (the next temperature picked by
 * bisection to hold the ESS, resample, rejuvenate by random-walk moves). `posterior(state)` is the particles at the current
 * temperature with the running evidence estimate (it is the posterior once the temperature reaches 1, when the run ends
 * `converged`). The run is capped by the session's `steps`.
 *
 * @param model - A bound spec, or a program closed over its data (continuous latent sites).
 * @returns The method, with `check` (a likelihood and only continuous latent sites) and `posterior`.
 * @concept sequential-monte-carlo
 * @index estimator Adaptive-tempering SMC, one temperature per step, with the evidence.
 * @example
 * ```ts
 * import { temperedSmcMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const method = temperedSmcMethod(model)
 * const s = session(method, { steps: 100, mode: 'eager', config: { numParticles: 300 } })
 * [s.status, Math.abs(item(method.posterior?.(s.state, s.config).mean('m') ?? 0) - 0.5) < 0.2] // => ['converged', true]
 * ```
 */
export function temperedSmcMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<SMCState, Config, TemperedSmcPath, ModelAnalysis> {
  const program = programOf(model)
  let cached: TemperedProblem | undefined
  const problem = (): TemperedProblem => {
    cached ??= temperedProblem(program, undefined, makeKey(0))
    return cached
  }
  const posterior = (state: SMCState): Posterior =>
    temperedPosterior(problem(), state.particles, {
      logEvidence: state.logEvidence,
      status: state.temperature >= 1 ? 'converged' : 'stopped',
      diagnostics: {
        method: 'tempered',
        stage: state.stage,
        temperature: state.temperature,
        ess: state.ess,
        partial: state.temperature < 1,
      },
    })
  const check = checkOf(smcEngine(), (a) => {
    if (a.continuous.length === 0) return no('no continuous latent nodes', ['continuous latents'], 'use enumeration')
    if (a.finiteDiscrete.length > 0 || a.unboundedDiscrete.length > 0)
      return no(
        `discrete latent nodes (${[...a.finiteDiscrete, ...a.unboundedDiscrete].join(', ')}) prevent tempering`,
        ['continuous latents only'],
        'use Gibbs, or sum them out and use NUTS',
      )
    return yes(`continuous latents: tempering from the prior to the posterior${priorOnly(a)}`)
  })
  const ensure = guard(options.name ?? 'tempered-smc', model, check)
  return defineInferenceMethod<SMCState, Config, TemperedSmcPath, ModelAnalysis>({
    name: options.name ?? 'tempered-smc',
    family: 'smc',
    description: 'adaptive-tempering SMC with the evidence, one temperature per step',
    config: temperedSmcKnobs,
    check,
    posterior,
    defaults: { 'smc/temperature': Log(), 'smc/ess': Log(), 'smc/logEvidence': Log() },
    contract: (): Contract<SMCState, TemperedSmcPath> =>
      defineContract({
        'smc/temperature': scalar({
          description: 'the tempering parameter lambda, 0 at the prior and 1 at the posterior',
        }),
        'smc/ess': scalar({ description: 'the ESS of the incremental weights before resampling' }),
        'smc/logEvidence': scalar({ description: 'the running estimate of log p(data)' }),
        'smc/acceptance': scalar({ description: 'the mean acceptance rate of the MCMC moves', direction: 'higher' }),
        'smc/particles': tensor({
          role: 'state',
          shape: ['particles', 'dim'],
          description: 'the particles, on the unconstrained scale',
        }),
        'smc/snapshot': derived<SMCState>({
          cost: 'dear',
          description: 'the particles at the current temperature as JSON: means, sds, quantiles',
          compute: (s) => posteriorJson(posterior(s)),
        }),
      }),
    make: (config): Algorithm<SMCState, TemperedSmcPath> => {
      ensure()
      const p = problem()
      const inner = smcAlgorithm(p.prior, p.logLikelihood, {
        numParticles: num(config, 'numParticles'),
        mcmcSteps: num(config, 'mcmcSteps'),
        targetEss: num(config, 'targetEss'),
        resampling: pick(config, 'resampling') as (typeof SCHEMES)[number],
      })
      return {
        name: inner.name,
        config: inner.config,
        init: (key: Key) => {
          ensure()
          return inner.init(key)
        },
        step: (s, ctx) => {
          const next = inner.step(s, ctx as never)
          ctx.log('smc/temperature', next.temperature)
          ctx.log('smc/ess', next.ess)
          ctx.log('smc/logEvidence', next.logEvidence)
          if (!Number.isNaN(next.acceptanceRate)) ctx.log('smc/acceptance', next.acceptanceRate)
          if (ctx.wants('smc/particles')) ctx.log('smc/particles', next.particles)
          return next
        },
        done: (s, t) => inner.done?.(s, t),
        serialise: (s) => stateToJson(s),
        deserialise: (j: Json) => stateFromJson(j) as SMCState,
      }
    },
  })
}

/** The paths the particle filter method logs. */
export type ParticleFilterPath =
  | 'filter/ess'
  | 'filter/logEvidence'
  | 'filter/resampled'
  | 'filter/step'
  | 'filter/snapshot'

/** The knobs of {@link particleFilterMethod} (the budget and snapshot knobs are added to them). */
export const particleFilterKnobs: DictSpace = dict({
  numParticles: int(1, 1_000_000, { default: 500, label: 'particles' }),
  resampleThreshold: real(0, 1, {
    default: 0.5,
    label: 'resampling threshold',
    description: 'resample when the ESS falls below this fraction (1: every step)',
  }),
  scheme: choice([...SCHEMES], { default: 'systematic', label: 'resampling' }),
})

// A particle's path as the values of its latent and deterministic sites at each step: what serialises of a trace, and all
// the posterior reads of one (`stackSiteValues` takes `type`, `isObserved` and `value`).
type PathRecord = Readonly<Record<string, Tensor>>

const recordOf = (tr: ProgramTrace): PathRecord => {
  const out: Record<string, Tensor> = {}
  for (const x of tr.sites.values())
    if ((x.type === 'sample' && !x.isObserved) || x.type === 'deterministic') out[x.name] = primalOf(x.value)
  return out
}

const traceOfRecord = (r: PathRecord): ProgramTrace =>
  ({
    sites: new Map(
      Object.entries(r).map(([name, value]) => [name, { type: 'sample', name, value, isObserved: false }]),
    ),
    returnValue: undefined,
  }) as unknown as ProgramTrace

/**
 * The bootstrap particle filter over a program's `markov` loop as a method with a fine step: one time step per step, with
 * the effective sample size and the running evidence on the tape and the filtering posterior (the weighted particle paths
 * up to the step) at any state. The state serialises (a particle's path is kept as the values of its sites), so a run
 * resumes and forks exactly. The run ends `completed` after the loop's last step.
 *
 * @param model - A program closed over its data with every latent site inside one `markov` loop, or a bound spec whose
 *   only latent is one `scan` (its other nodes data and emissions: no parameters to infer, no derived nodes after it).
 * @returns The method, with `check` and `posterior`.
 * @concept sequential-monte-carlo
 * @index estimator Bootstrap particle filter over a markov loop, one time step per step.
 * @example
 * ```ts
 * import { particleFilterMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { markov, observe, sample } from 'omai-engine/probprog'
 * import type { Value } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const walk = () => markov('t', 3, (t, x: Value | number) => { const z = sample(`x${t}`, normal(x, 1)); observe(`y${t}`, normal(z, 1), t); return z }, 0)
 * const method = particleFilterMethod(walk)
 * const s = session(method, { steps: 10, mode: 'eager', config: { numParticles: 50 } })
 * [s.t, method.posterior?.(s.state, s.config).sites] // => [3, ['x0', 'x1', 'x2']]
 * ```
 */
export function particleFilterMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<ParticleFilterState, Config, ParticleFilterPath, ModelAnalysis> {
  const program = programOf(model)
  const posterior = (state: ParticleFilterState): Posterior =>
    particleFilterPosterior(state, state.ess === 0 && state.t > 0 ? 'failed' : 'completed')
  const check = (a: ModelAnalysis): Verdict => {
    if (!isProgramAnalysis(a)) {
      if (a.scans.length !== 1)
        return no(
          'the model has no single scan',
          ['scan'],
          'give a spec with one scan, or a program with one markov loop',
        )
      if (!a.hasLikelihood) return no('observed nodes have no likelihood', ['likelihood'], 'add an emission')
      const outside = a.latent.filter((l) => l !== a.scans[0]?.name)
      if (outside.length > 0)
        return no(
          `latent nodes outside the scan (${outside.join(', ')})`,
          ['scan'],
          'fix them as data, or use NUTS or tempered SMC',
        )
      const after = [...a.derived, ...a.generated, ...a.factors, ...a.params]
      if (after.length > 0)
        return no(
          `nodes beside the scan (${after.join(', ')}) that the filter does not carry`,
          ['scan'],
          'leave them out of the model the filter runs',
        )
      return yes(`a scan of ${a.scans[0]?.steps ?? 'several'} steps over '${a.scans[0]?.over}'`)
    }
    if (!a.hasLikelihood) return no('observed sites have no likelihood', ['likelihood'], 'add an observation')
    const blocker = filterBlocker(a)
    return blocker === undefined
      ? yes(`a markov loop of ${a.markov[0]?.length ?? 0} steps`)
      : no(blocker, ['markov loop'], 'give a program with one markov loop holding every latent site')
  }
  const ensure = guard(options.name ?? 'particle-filter', model, check)
  return defineInferenceMethod<ParticleFilterState, Config, ParticleFilterPath, ModelAnalysis>({
    name: options.name ?? 'particle-filter',
    family: 'smc',
    description: 'bootstrap particle filter over a markov loop, one time step per step',
    config: particleFilterKnobs,
    check,
    posterior,
    defaults: { 'filter/ess': Log(), 'filter/logEvidence': Log() },
    contract: (): Contract<ParticleFilterState, ParticleFilterPath> =>
      defineContract({
        'filter/ess': scalar({ description: 'the effective sample size after the step' }),
        'filter/logEvidence': scalar({ description: 'the running estimate of log p(y_0..t)' }),
        'filter/resampled': scalar({ description: '1 when the particles were resampled at the step' }),
        'filter/step': scalar({ description: 'the time step reached' }),
        'filter/snapshot': derived<ParticleFilterState>({
          cost: 'dear',
          description: 'the filtering posterior up to the step as JSON',
          compute: (s) => posteriorJson(posterior(s)),
        }),
      }),
    make: (config): Algorithm<ParticleFilterState, ParticleFilterPath> => {
      ensure()
      const inner = particleFilterAlgorithm(program, undefined, {
        numParticles: num(config, 'numParticles'),
        resampleThreshold: num(config, 'resampleThreshold'),
        scheme: pick(config, 'scheme') as (typeof SCHEMES)[number],
      })
      return {
        name: inner.name,
        config: inner.config,
        init: (key: Key) => {
          ensure()
          return inner.init(key)
        },
        step: (s, ctx) => {
          const next = inner.step(s, ctx as never)
          ctx.log('filter/step', next.t)
          ctx.log('filter/ess', next.ess)
          ctx.log('filter/logEvidence', next.logEvidence)
          ctx.log('filter/resampled', next.resampled ? 1 : 0)
          return next
        },
        done: (s, t) => inner.done?.(s, t),
        serialise: (s) =>
          stateToJson({
            t: s.t,
            carries: s.carries,
            logWeights: s.logWeights,
            logEvidence: s.logEvidence,
            ess: s.ess,
            resampled: s.resampled,
            resamplings: s.resamplings,
            paths: s.paths.map((path) => path.map(recordOf)),
          }),
        deserialise: (j: Json) => {
          const o = stateFromJson(j) as Omit<ParticleFilterState, 'paths'> & { paths: PathRecord[][] }
          // A state saved before the weights were a tensor has a typed array here.
          return {
            ...o,
            logWeights: asTensor(o.logWeights) as Tensor,
            paths: o.paths.map((path) => path.map(traceOfRecord)),
          }
        },
      }
    },
  })
}
