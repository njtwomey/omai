/**
 * Nested sampling as an inference method: the evidence and the posterior of a model from a population of live points
 * squeezed towards higher likelihood (`nestedAlgorithm` of `sampling`, on the prior and likelihood of the program on the
 * unconstrained scale). One live-point replacement is a step, so a page watches the evidence estimate and its error settle
 * and the likelihood threshold climb, reads the weighted posterior at any step (the removed points by their volume share
 * times likelihood, plus the live points), and cancels or resumes within one replacement.
 */

import type { Json } from '../../base/index.ts'
import { type Key, key as makeKey } from '../../random/index.ts'
import {
  type NestedPath,
  type NestedState,
  nestedAlgorithm,
  nestedContract,
  nestedEvidence,
  nestedPosterior,
} from '../../sampling/index.ts'
import { choice, type DictSpace, dict, int, real } from '../../space/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'
import { constrainDraws } from '../density.ts'
import { type TemperedProblem, temperedProblem } from '../engines/smc.ts'
import { type Posterior, weightedPosterior } from '../posterior.ts'
import {
  guard,
  isProgramAnalysis,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  pick,
  posteriorJson,
  programOf,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod, type Verdict } from './method.ts'

/** The paths the nested sampling method logs: those of the algorithm and the snapshot. */
export type NestedMethodPath = NestedPath | 'ns/snapshot'

/** The knobs of {@link nestedMethod} (the budget and snapshot knobs are added to them). */
export const nestedKnobs: DictSpace = dict({
  numLive: int(2, 1_000_000, {
    default: 200,
    label: 'live points',
    description: 'the resolution in prior volume; the error of log Z falls as 1 / sqrt(live points)',
  }),
  inner: choice(['slice', 'random-walk'], {
    default: 'slice',
    label: 'constrained-prior sampler',
    description: 'how a removed point is replaced: slice sweeps or random-walk moves from another live point',
  }),
  innerSteps: int(1, 10_000, { default: 5, label: 'inner steps', description: 'sweeps or moves per replacement' }),
  tolerance: real(1e-6, 100, {
    default: 0.05,
    label: 'tolerance',
    description: 'stop when the live points could add less than this to log Z',
  }),
})

/**
 * Nested sampling on a model as a method with a fine step: one live-point replacement per step. The tape has the log
 * evidence so far (live points included) with its error `sqrt(H / n)`, the log prior volume, the likelihood threshold, the
 * information and the replacement's cost, and the live points when asked for. `posterior(state)` is the weighted posterior
 * of everything seen so far with the evidence estimate and its error (it is final once the run ends `converged`). The run
 * is capped by the session's `steps`; the evidence needs about `live points x information` of them.
 *
 * @param model - A bound spec, or a program closed over its data (continuous latent sites).
 * @param options - The name.
 * @returns The method, with `check` (a likelihood and only continuous latent sites) and `posterior`.
 * @concept nested-sampling
 * @index estimator Nested sampling, one live-point replacement per step, with the evidence and its error.
 * @example
 * ```ts
 * import { nestedMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const method = nestedMethod(model)
 * const s = session(method, { steps: 5000, mode: 'eager', config: { numLive: 100 } })
 * [s.status, Math.abs((method.posterior?.(s.state, s.config).logEvidence?.value ?? 0) - (-1.5155)) < 0.3] // => ['converged', true]
 * ```
 */
export function nestedMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<NestedState, Config, NestedMethodPath, ModelAnalysis> {
  const program = programOf(model)
  let cached: TemperedProblem | undefined
  const problem = (): TemperedProblem => {
    cached ??= temperedProblem(program, undefined, makeKey(0))
    return cached
  }
  const posterior = (state: NestedState, config: Config): Posterior => {
    const p = problem()
    const n = num(config, 'numLive')
    const e = nestedEvidence(state, n)
    const w = nestedPosterior(state, p.prior.dim, n)
    return weightedPosterior(constrainDraws(p.um, w.points), w.logWeights, {
      engine: 'nested',
      status: state.finished ? 'converged' : 'stopped',
      logEvidence: { value: e.logZ, standardError: e.logZError, method: 'nested sampling' },
      diagnostics: {
        method: 'nested sampling',
        iterations: state.iteration,
        information: e.information,
        evaluations: state.totalEvaluations,
        liveLogLikelihoodThreshold: Number.isFinite(state.logLStar) ? state.logLStar : null,
        partial: !state.finished,
      },
    })
  }
  const check = (a: ModelAnalysis): Verdict => {
    if (!a.hasLikelihood) return no('observed nodes have no likelihood', ['likelihood'], 'add an observed node')
    if (isProgramAnalysis(a)) {
      if (a.continuous.length === 0) return no('no continuous latent sites', ['continuous latents'], 'use enumeration')
      if (a.discrete.sites.length > 0)
        return no(
          `discrete latent sites (${a.discrete.sites.join(', ')}) cannot be sampled by the constrained-prior moves`,
          ['continuous latents only'],
          'use Gibbs, or sum them out',
        )
    } else {
      if (a.continuous.length === 0) return no('no continuous latent nodes', ['continuous latents'], 'use enumeration')
      if (a.finiteDiscrete.length > 0 || a.unboundedDiscrete.length > 0)
        return no(
          `discrete latent nodes (${[...a.finiteDiscrete, ...a.unboundedDiscrete].join(', ')}) cannot be sampled by the constrained-prior moves`,
          ['continuous latents only'],
          'use Gibbs, or sum them out',
        )
    }
    return yes('continuous latents with a proper prior: the evidence is the likelihood integrated over prior volume')
  }
  const ensure = guard(options.name ?? 'nested', model, check)
  return defineInferenceMethod<NestedState, Config, NestedMethodPath, ModelAnalysis>({
    name: options.name ?? 'nested',
    family: 'nested',
    description: 'nested sampling with live points: the evidence with an error bar and the weighted posterior',
    config: nestedKnobs,
    check,
    posterior,
    defaults: { 'ns/logZ': Log(), 'ns/logZError': Log(), 'ns/logL': Log() },
    contract: (config): Contract<NestedState, NestedMethodPath> =>
      defineContract({
        ...nestedContract,
        'ns/snapshot': derived<NestedState>({
          cost: 'dear',
          description:
            'the weighted posterior of the points so far as JSON: means, sds, quantiles, evidence and its error',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config): Algorithm<NestedState, NestedMethodPath> => {
      ensure()
      const p = problem()
      const inner = nestedAlgorithm(p.prior, p.logLikelihood, {
        numLive: num(config, 'numLive'),
        inner: pick(config, 'inner') as 'slice' | 'random-walk',
        innerSteps: num(config, 'innerSteps'),
        tolerance: num(config, 'tolerance'),
      })
      return {
        name: inner.name,
        config: inner.config,
        init: (key: Key) => {
          ensure()
          return inner.init(key)
        },
        step: (s, ctx) => inner.step(s, ctx as never),
        done: (s, t) => inner.done?.(s, t),
        serialise: (s) => stateToJson(s),
        deserialise: (j: Json) => stateFromJson(j) as NestedState,
      }
    },
  })
}
