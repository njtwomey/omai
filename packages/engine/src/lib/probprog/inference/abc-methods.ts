/**
 * Likelihood-free inference as two methods. ABC-SMC shrinks a tolerance over generations of weighted particles, one
 * generation per step, with the tolerance, the acceptance rate and the ESS on the tape and the weighted population as the
 * posterior of any state; its state serialises (the population is kept as its draws), so a run resumes and forks. Rejection
 * ABC is one computation and is honestly one step. The one-call `abcSmc`, `abcRejection` and the planner's `abc` engine are
 * unchanged.
 */

import type { Json } from '../../base/index.ts'
import { foldIn, key as makeKey } from '../../random/index.ts'
import { type DictSpace, dict, int, real } from '../../space/index.ts'
import { asTensor, exp, item, mul, sum, type Tensor } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import {
  type AbcSmcState,
  abcEngine,
  abcRejection,
  abcSmcAlgorithm,
  abcSmcDraws,
  abcSmcPosterior,
} from '../engines/abc.ts'
import type { ProgramTrace } from '../handlers.ts'
import type { Posterior } from '../posterior.ts'
import {
  checkOf,
  guard,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  posteriorJson,
  programOf,
  restorable,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod } from './method.ts'
import { type OneShotPath, type OneShotState, oneShotMethod } from './one-step.ts'

/** The paths the ABC-SMC method logs. */
export type AbcSmcPath =
  | 'abc/generation'
  | 'abc/epsilon'
  | 'abc/acceptance'
  | 'abc/ess'
  | 'abc/particles'
  | 'abc/snapshot'

/** The state of {@link abcSmcMethod}: the algorithm's state and the population as draws (the traces do not serialise). */
export interface AbcSmcRunState {
  readonly smc: AbcSmcState
  readonly draws: Readonly<Record<string, Tensor>>
}

/** The knobs of {@link abcSmcMethod} (the budget and snapshot knobs are added to them). */
export const abcSmcKnobs: DictSpace = dict({
  numParticles: int(10, 100_000, { default: 200, label: 'particles' }),
  quantile: real(0.05, 0.95, {
    default: 0.5,
    label: 'tolerance quantile',
    description: "each generation's tolerance is this quantile of the last one's distances",
  }),
  minEpsilon: real(0, 1_000_000, { default: 0, label: 'minimum tolerance' }),
  minAcceptance: real(0, 0.5, {
    default: 0.01,
    label: 'minimum acceptance',
    description: 'stop when a generation accepts less than this fraction',
  }),
  maxSimulationsPerParticle: int(1, 100_000, { default: 100, label: 'simulation cap per particle' }),
})

const emptyTrace = { sites: new Map(), returnValue: undefined } as unknown as ProgramTrace

/**
 * ABC-SMC on a simulator model as a method with a fine step: one generation per step (perturb, simulate, accept within the
 * shrinking tolerance, weight). `posterior(state)` is the weighted population. The run ends `converged` at the minimum
 * tolerance or acceptance rate, and `stopped` when a generation hits its simulation cap; the session's `steps` or a budget
 * also end it. A population is kept as its draws, so the state serialises and a run resumes and forks exactly.
 *
 * @param model - A bound spec, or a program closed over its data (continuous latent sites; observed nodes may be simulators).
 * @returns The method, with `check` (observed and continuous latent nodes) and `posterior`.
 * @concept approximate-bayesian-computation
 * @index estimator ABC-SMC, one generation per step, with the tolerance shrinking on the tape.
 * @example
 * ```ts
 * import { abcSmcMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, plate, sample } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const y = [0.5, 1.2, 1.6, 1.9, 2, 2.1, 2.4, 2.8, 3.5, 2]
 * const model = () => { const m = sample('m', normal(0, 3)); plate('n', y.length, () => observe('y', normal(m, 1), y)) }
 * const method = abcSmcMethod(model)
 * const s = session(method, { steps: 4, mode: 'eager', config: { numParticles: 150 } })
 * Math.abs(item(method.posterior?.(s.state, s.config).mean('m') ?? 0) - 2) < 0.5 // => true
 * ```
 */
export function abcSmcMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<AbcSmcRunState, Config, AbcSmcPath, ModelAnalysis> {
  const program = programOf(model)
  const posterior = (state: AbcSmcRunState): Posterior =>
    abcSmcPosterior(state.draws, state.smc, state.smc.generation === 0 ? 'stopped' : 'completed')
  const check = checkOf(abcEngine(), (a) => {
    if (a.latent.length === 0) return no('no latent nodes', ['latent nodes'])
    if (a.observed.length === 0)
      return no('no observed nodes', ['observed nodes'], 'ABC compares simulated with observed data')
    if (a.finiteDiscrete.length > 0 || a.unboundedDiscrete.length > 0)
      return no('discrete latent nodes are not supported by ABC-SMC', ['continuous latents only'], 'use rejection ABC')
    return yes(
      a.hasLikelihood
        ? 'simulation (a likelihood exists, so other methods are better)'
        : 'simulators and no likelihood',
    )
  })
  const ensure = guard(options.name ?? 'abc-smc', model, check)
  return defineInferenceMethod<AbcSmcRunState, Config, AbcSmcPath, ModelAnalysis>({
    name: options.name ?? 'abc-smc',
    family: 'likelihood-free',
    description: 'ABC-SMC: generations of weighted particles under a shrinking tolerance',
    config: abcSmcKnobs,
    check,
    posterior,
    defaults: { 'abc/epsilon': Log(), 'abc/acceptance': Log(), 'abc/ess': Log() },
    contract: (): Contract<AbcSmcRunState, AbcSmcPath> =>
      defineContract({
        'abc/generation': scalar({ description: 'generations done' }),
        'abc/epsilon': scalar({ description: 'the tolerance of the generation', direction: 'lower' }),
        'abc/acceptance': scalar({ description: 'the fraction of simulations accepted', direction: 'higher' }),
        'abc/ess': scalar({ description: 'the effective sample size of the weighted population' }),
        'abc/particles': tensor({
          role: 'state',
          shape: ['particles', 'dim'],
          description: 'the population, on the unconstrained scale',
        }),
        'abc/snapshot': derived<AbcSmcRunState>({
          cost: 'dear',
          description: 'the weighted population as JSON: means, sds, quantiles',
          compute: (s) => posteriorJson(posterior(s)),
        }),
      }),
    make: (config, key): Algorithm<AbcSmcRunState, AbcSmcPath> => {
      const particles = num(config, 'numParticles')
      const base = restorable(
        abcSmcAlgorithm(program, undefined, {
          numParticles: particles,
          quantile: num(config, 'quantile'),
          minEpsilon: num(config, 'minEpsilon'),
          minAcceptance: num(config, 'minAcceptance'),
          maxSimulations: num(config, 'maxSimulationsPerParticle') * particles,
        }),
        foldIn(key, 0),
      )
      return {
        name: 'probprog/abc-smc-method',
        config: base.config,
        init: (k) => {
          ensure()
          const smc = base.init(k)
          return { smc, draws: abcSmcDraws(smc.traces) }
        },
        step: (s, ctx) => {
          const smc = base.step(s.smc, ctx as unknown as StepContext)
          const draws = smc.traces === s.smc.traces ? s.draws : abcSmcDraws(smc.traces)
          ctx.log('abc/generation', smc.generation)
          if (Number.isFinite(smc.epsilon)) ctx.log('abc/epsilon', smc.epsilon)
          ctx.log('abc/acceptance', smc.acceptance)
          ctx.log('abc/ess', 1 / item(sum(exp(mul(smc.logWeights, 2)))))
          if (ctx.wants('abc/particles')) ctx.log('abc/particles', smc.particles)
          return { smc, draws }
        },
        done: (s, t) => base.done?.(s.smc, t),
        // The traces hold distributions and do not serialise; the population is kept as its draws, and the count stands in
        // for the traces (the algorithm reads only how many there are, before the next generation replaces them).
        serialise: (s) => stateToJson({ ...s.smc, traces: s.smc.traces.length, draws: s.draws }),
        deserialise: (j: Json) => {
          const o = stateFromJson(j) as Omit<AbcSmcState, 'traces'> & { traces: number; draws: Record<string, Tensor> }
          base.prime()
          const { draws, traces, ...rest } = o
          // A state saved before the population was held as tensors has typed arrays in these fields.
          const population = {
            ...rest,
            particles: asTensor(rest.particles) as Tensor,
            logWeights: asTensor(rest.logWeights) as Tensor,
            distances: asTensor(rest.distances) as Tensor,
          }
          return { smc: { ...population, traces: Array.from({ length: traces }, () => emptyTrace) }, draws }
        },
      }
    },
  })
}

/** The knobs of {@link abcRejectionMethod}. */
export const abcRejectionKnobs: DictSpace = dict({
  numSamples: int(10, 10_000_000, { default: 2000, label: 'simulations' }),
  quantile: real(0.0001, 1, {
    default: 0.05,
    label: 'accepted fraction',
    description: 'keep this fraction of the closest simulations',
  }),
  seed: int(0, 2_147_483_647, {
    default: 0,
    label: 'seed',
    description: 'the key of the simulations (a one-step method has no step keys)',
  }),
})

/**
 * Rejection ABC as a method: simulate parameters from the prior and data from the program, keep the draws whose summary
 * statistics fall closest to the observed ones. It is one computation, so honestly one step; its key is the `seed` knob.
 *
 * @param model - A bound spec, or a program closed over its data (observed nodes may be simulators).
 * @returns The method (one step; `check` needs observed and latent nodes).
 * @concept approximate-bayesian-computation
 * @index estimator Rejection ABC with summary statistics, as a one-step method.
 * @example
 * ```ts
 * import { abcRejectionMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 3)); observe('y', normal(m, 1), 1) }
 * const method = abcRejectionMethod(model)
 * const s = session(method, { steps: 1, mode: 'eager', config: { numSamples: 500 } })
 * method.posterior?.(s.state, s.config).status // => 'completed'
 * ```
 */
export function abcRejectionMethod(
  model: MethodModel,
): InferenceMethod<OneShotState, Config, OneShotPath<'abc-rejection'>, ModelAnalysis> {
  const program = programOf(model)
  return oneShotMethod({
    name: 'abc-rejection',
    family: 'likelihood-free',
    description: 'rejection ABC with summary statistics',
    config: abcRejectionKnobs,
    model,
    check: checkOf(abcEngine(), (a) => {
      if (a.latent.length === 0) return no('no latent nodes', ['latent nodes'])
      if (a.observed.length === 0)
        return no('no observed nodes', ['observed nodes'], 'ABC compares simulated with observed data')
      return yes('simulation')
    }),
    run: (config) =>
      abcRejection(program, undefined, {
        key: foldIn(makeKey(num(config, 'seed')), 0),
        numSamples: num(config, 'numSamples'),
        quantile: num(config, 'quantile'),
      }),
  })
}
