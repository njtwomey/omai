/**
 * Gibbs sampling as an inference method: one sweep per step over the blocks the program's structure gives (exact discrete
 * conditionals by enumeration, conjugate and mixture conditionals, collapsed Dirichlet and beta sites, slice steps for
 * the rest), the draws kept so far in the state, which serialises so a run resumes and forks, and `posterior` reading the
 * draws so far (the current sweep while it is still warming up). The one-call `gibbs` and its planner entry are unchanged.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { foldIn } from '../../random/index.ts'
import { bool, type DictSpace, dict, int, real } from '../../space/index.ts'
import { primalOf, type Value } from '../../tensor/index.ts'
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
import { type GibbsState, gibbsAlgorithm, gibbsEngine, gibbsPosterior } from '../engines/gibbs.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import {
  checkOf,
  flag,
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

/** The paths the Gibbs method logs. */
export type GibbsPath = 'gibbs/logJoint' | 'gibbs/sweep' | 'gibbs/kept' | `gibbs/value/${string}` | 'gibbs/snapshot'

/** The state of {@link gibbsMethod}: the sampler's state, the sweeps done and the draws kept after warm-up. */
export interface GibbsRunState {
  readonly sweep: number
  readonly current: GibbsState
  readonly kept: readonly Readonly<Record<string, Value>>[]
}

/** The knobs of {@link gibbsMethod} (the budget and snapshot knobs are added to them). */
export const gibbsKnobs: DictSpace = dict({
  numWarmup: int(0, 1_000_000, { default: 200, label: 'warm-up sweeps' }),
  numDraws: int(1, 1_000_000, { default: 1000, label: 'draws' }),
  thin: int(1, 1000, { default: 1, label: 'thinning' }),
  collapse: bool({ default: true, label: 'collapse Dirichlet and beta sites' }),
  sliceWidth: real(0.01, 100, {
    log: true,
    default: 1,
    label: 'slice width',
    description: 'for sites with no conjugate conditional',
  }),
})

/**
 * Gibbs sampling on a model as a method with a fine step: one sweep per step. The state holds the draws kept after
 * warm-up and serialises, so the run resumes and forks exactly; `posterior` reads them (during warm-up, the current sweep as
 * a single draw). The run ends `completed` after `numWarmup + numDraws * thin` sweeps.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @returns The method, with `check` (discrete or conjugate conditionals) and `posterior`.
 * @concept gibbs-sampling
 * @index estimator Gibbs sampling, one sweep per step, resumable, with anytime draws.
 * @example
 * ```ts
 * import { gibbsMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, plate, sample } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); plate('n', 3, () => observe('y', normal(m, 1), [1, 2, 3])) }
 * const method = gibbsMethod(model)
 * const s = session(method, { steps: 5000, mode: 'eager', config: { numDraws: 1500, numWarmup: 50 } })
 * Math.abs(item(method.posterior?.(s.state, s.config).mean('m') ?? 0) - 1.5) < 0.1 // => true
 * ```
 */
export function gibbsMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<GibbsRunState, Config, GibbsPath, ModelAnalysis> {
  const program = programOf(model)
  const posterior = (state: GibbsRunState, config: Config): Posterior => {
    const warming = state.kept.length === 0
    const draws = warming ? [state.current.values] : state.kept
    try {
      return gibbsPosterior(
        draws,
        warming
          ? 'stopped'
          : state.sweep >= num(config, 'numWarmup') + num(config, 'numDraws') * num(config, 'thin')
            ? 'completed'
            : 'stopped',
        {
          sweeps: state.sweep,
          kept: state.kept.length,
          warmup: warming,
        },
      )
    } catch (e) {
      return definePosterior({
        form: 'samples',
        engine: 'gibbs',
        sites: [],
        status: 'failed',
        draws: {},
        diagnostics: { reason: e instanceof Error ? e.message : String(e) },
      })
    }
  }
  const check = checkOf(gibbsEngine(), (a) => {
    if (!a.hasLikelihood) return no('observed nodes have no likelihood', ['likelihood'], 'add an observed node')
    if (a.latent.length === 0) return no('no latent nodes', ['latent nodes'])
    if (a.unboundedDiscrete.length > 0)
      return no(
        `unbounded discrete latents (${a.unboundedDiscrete.join(', ')})`,
        ['finite discrete latents'],
        'bound them',
      )
    if (a.finiteDiscrete.length === 0 && a.conjugate.length === 0)
      return no('no discrete or conjugate conditionals', ['finite discrete latents', 'conjugate sites'], 'use NUTS')
    return yes('discrete or conjugate conditionals')
  })
  const ensure = guard(options.name ?? 'gibbs', model, check)
  return defineInferenceMethod<GibbsRunState, Config, GibbsPath, ModelAnalysis>({
    name: options.name ?? 'gibbs',
    family: 'mcmc',
    description: 'Gibbs sampling with discrete, conjugate and collapsed conditionals, one sweep per step',
    config: gibbsKnobs,
    check,
    posterior,
    defaults: { 'gibbs/logJoint': Log(), 'gibbs/kept': Log() },
    contract: (config): Contract<GibbsRunState, GibbsPath> =>
      defineContract({
        'gibbs/logJoint': scalar({ description: 'the log density of the current state', direction: 'higher' }),
        'gibbs/sweep': scalar({ description: 'sweeps done' }),
        'gibbs/kept': scalar({ description: 'draws kept so far' }),
        'gibbs/value/*': tensor({
          role: 'state',
          description: 'the current value of a latent site (one path per site)',
        }),
        'gibbs/snapshot': derived<GibbsRunState>({
          cost: 'dear',
          description: 'the posterior so far as JSON: means, sds, quantiles, ESS',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config, key): Algorithm<GibbsRunState, GibbsPath> => {
      const numWarmup = num(config, 'numWarmup')
      const numDraws = num(config, 'numDraws')
      const thin = num(config, 'thin')
      const base = restorable(
        gibbsAlgorithm(program, undefined, {
          collapse: flag(config, 'collapse'),
          sliceWidth: num(config, 'sliceWidth'),
        }),
        foldIn(key, 0),
      )
      return {
        name: 'probprog/gibbs-method',
        config: { numWarmup, numDraws, thin },
        init: (k) => {
          ensure()
          return { sweep: 0, current: base.init(k), kept: [] }
        },
        step: (s, ctx) => {
          const next = base.step(s.current, ctx as unknown as StepContext)
          const sweep = s.sweep + 1
          const keep = sweep > numWarmup && (sweep - numWarmup) % thin === 0
          ctx.log('gibbs/sweep', sweep)
          if (Number.isFinite(next.logJoint)) ctx.log('gibbs/logJoint', next.logJoint)
          for (const [name, value] of Object.entries(next.values)) {
            const path = `gibbs/value/${name}` as const
            if (ctx.wants(path)) ctx.log(path, primalOf(value))
          }
          const kept = keep ? [...s.kept, next.values] : s.kept
          ctx.log('gibbs/kept', kept.length)
          return { sweep, current: next, kept }
        },
        done: (s) => base.done?.(s.current, s.sweep) ?? (s.kept.length >= numDraws ? 'completed' : undefined),
        serialise: (s) => ({
          sweep: s.sweep,
          current: stateToJson(s.current),
          kept: stateToJson(s.kept),
        }),
        deserialise: (j: Json) => {
          const o = j as unknown as { sweep: number; current: Json; kept: Json }
          if (o.sweep === undefined) throw new OmaiError('schema', 'gibbs: not a serialised state')
          return {
            sweep: o.sweep,
            current: base.deserialise?.(o.current) as GibbsState,
            kept: stateFromJson(o.kept) as GibbsRunState['kept'],
          }
        },
      }
    },
  })
}
