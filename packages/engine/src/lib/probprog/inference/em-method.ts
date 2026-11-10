/**
 * Expectation maximisation as an inference method: one EM iteration per step (SQUAREM-accelerated by default), the
 * objective and the estimates on the tape, a state that serialises so a run resumes and forks, and `posterior` reading
 * the point estimates and exact E-step marginals of any state. The one-call `em` and its planner entry are unchanged.
 */

import { foldIn, key as makeKey } from '../../random/index.ts'
import { bool, choice, type DictSpace, dict, int, real } from '../../space/index.ts'
import { primalOf, type Value } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  type RunStatus,
  type StepContext,
  scalar,
  tensor,
} from '../../trace/index.ts'
import { type EmState, emEngine, emWithPosterior } from '../engines/em.ts'
import type { Posterior } from '../posterior.ts'
import {
  checkOf,
  flag,
  guard,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  pick,
  posteriorJson,
  programOf,
  restorable,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod } from './method.ts'

/** The paths the EM method logs. */
export type EmPath = 'em/objective' | 'em/iteration' | `em/estimate/${string}` | 'em/snapshot'

/** The state of {@link emMethod}: the algorithm's state and the iterations done. */
export interface EmRunState {
  readonly iteration: number
  readonly em: EmState
}

/** The knobs of {@link emMethod} (the budget and snapshot knobs are added to them). */
export const emKnobs: DictSpace = dict({
  tol: real(1e-14, 1e-2, {
    log: true,
    default: 1e-9,
    label: 'tolerance',
    description: 'stop when the objective improves by less than this, relatively',
  }),
  mStep: choice(['auto', 'gradient'], {
    default: 'auto',
    label: 'M-step',
    description: 'closed form where recognised, or always the gradient',
  }),
  gradientIterations: int(1, 1000, {
    default: 50,
    label: 'M-step iterations',
    description: 'L-BFGS iterations per gradient M-step',
  }),
  accelerate: bool({
    default: true,
    label: 'SQUAREM',
    description: 'extrapolate two EM maps, falling back to plain EM when it does not help',
  }),
})

/**
 * EM on a model as a method with a fine step: one EM iteration per step. The state serialises, so the run resumes and
 * forks exactly; `posterior(state)` is the point estimates of the params and MAP sites with the exact marginals of the
 * integrated discrete or Gaussian sites given them. The run ends `converged` when the objective settles and `diverged`
 * when it is not finite; otherwise the session's `steps` or a budget stops it.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @returns The method, with `check` (params, and discrete or plated latent sites to integrate) and `posterior`.
 * @concept expectation-maximisation
 * @index estimator Expectation maximisation, one iteration per step, with exact E-steps.
 * @example
 * ```ts
 * import { emMethod } from 'omai-engine/probprog/inference'
 * import { categorical, normal } from 'omai-engine/dists'
 * import { observe, param, plate, sample } from 'omai-engine/probprog'
 * import { take, toArray } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const y = [-2.1, -1.9, -2.2, 2.0, 1.8, 2.3]
 * const gmm = () => {
 *   const mu = param('mu', [-1, 1])
 *   plate('n', y.length, () => { const z = sample('z', categorical({ probs: [0.5, 0.5] })); observe('y', normal(take(mu, z), 0.5), y) })
 * }
 * const method = emMethod(gmm)
 * const s = session(method, { steps: 200, mode: 'eager' })
 * toArray(method.posterior?.(s.state, s.config).mean('mu') ?? 0).map((v) => Math.round(v * 100) / 100) // => [-2.07, 2.03]
 * ```
 */
export function emMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<EmRunState, Config, EmPath, ModelAnalysis> {
  const program = programOf(model)
  const parts = new Map<string, ReturnType<typeof emWithPosterior<void>>>()
  const emOptions = (
    config: Config,
  ): { tol: number; mStep: 'auto' | 'gradient'; gradientIterations: number; accelerate: boolean } => ({
    tol: num(config, 'tol'),
    mStep: pick(config, 'mStep') === 'gradient' ? 'gradient' : 'auto',
    gradientIterations: num(config, 'gradientIterations'),
    accelerate: flag(config, 'accelerate'),
  })
  const partsFor = (config: Config): ReturnType<typeof emWithPosterior<void>> => {
    const id = JSON.stringify(config)
    let p = parts.get(id)
    if (p === undefined) {
      p = emWithPosterior(program, undefined, emOptions(config))
      parts.set(id, p)
    }
    return p
  }
  const primed = new WeakSet<object>()
  const posterior = (state: EmRunState, config: Config): Posterior => {
    const p = partsFor(config)
    // The posterior reads the problem the algorithm sets up in `init`: set it up once for this configuration.
    if (!primed.has(p)) {
      p.algorithm.init(makeKey(0))
      primed.add(p)
    }
    const verdict: RunStatus | undefined = p.algorithm.done?.(state.em, state.iteration)
    return p.posterior(state.em, { status: verdict ?? 'completed', iterations: state.iteration })
  }
  const check = checkOf(emEngine(), (a) => {
    if (!a.hasLikelihood) return no('observed nodes have no likelihood', ['likelihood'], 'add an observed node')
    if (a.unboundedDiscrete.length > 0)
      return no(
        `unbounded discrete latents (${a.unboundedDiscrete.join(', ')})`,
        ['finite discrete latents'],
        'bound them',
      )
    if (a.params.length === 0)
      return no('no params to estimate', ['param nodes'], 'declare the parameters as params, or use a sampler')
    const platedContinuous = a.nodes.some((n) => n.kind === 'latent' && !n.discrete && n.dims.length > 0)
    if (a.finiteDiscrete.length === 0 && !platedContinuous)
      return no(
        'no discrete or plated latent nodes to integrate',
        ['finite discrete latents', 'plated latents'],
        'use an optimiser on the density',
      )
    return yes(
      a.finiteDiscrete.length > 0
        ? 'params with discrete latent nodes (closed-form M-steps where recognised)'
        : 'params with plated continuous latents',
    )
  })
  const ensure = guard(options.name ?? 'em', model, check)
  return defineInferenceMethod<EmRunState, Config, EmPath, ModelAnalysis>({
    name: options.name ?? 'em',
    family: 'variational',
    description: 'expectation maximisation with exact E-steps, one iteration per step',
    config: emKnobs,
    check,
    posterior,
    defaults: { 'em/objective': Log() },
    contract: (config): Contract<EmRunState, EmPath> =>
      defineContract({
        'em/objective': scalar({
          description: 'log p(x, estimates) with the latent sites integrated out',
          direction: 'higher',
        }),
        'em/iteration': scalar({ description: 'iterations done' }),
        'em/estimate/*': tensor({
          role: 'param',
          description: 'the current estimate of a param or MAP site (one path per site)',
        }),
        'em/snapshot': derived<EmRunState>({
          cost: 'dear',
          description: 'the estimates and E-step marginals as JSON',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config, key): Algorithm<EmRunState, EmPath> => {
      const base = restorable(partsFor(config).algorithm, foldIn(key, 0))
      return {
        name: 'probprog/em-method',
        config: base.config,
        init: (k) => {
          ensure()
          return { iteration: 0, em: base.init(k) }
        },
        step: (s, ctx) => {
          const em = base.step(s.em, ctx as unknown as StepContext)
          const iteration = s.iteration + 1
          if (Number.isFinite(em.objective)) ctx.log('em/objective', em.objective)
          ctx.log('em/iteration', iteration)
          for (const [name, value] of Object.entries(em.values)) {
            const path = `em/estimate/${name}` as const
            if (ctx.wants(path)) ctx.log(path, primalOf(value as Value))
          }
          return { iteration, em }
        },
        done: (s, t) => base.done?.(s.em, t),
        serialise: (s) => ({ iteration: s.iteration, em: base.serialise?.(s.em) ?? null }),
        deserialise: (j) => {
          const o = j as unknown as { iteration: number; em: never }
          return { iteration: o.iteration, em: base.deserialise?.(o.em) as EmState }
        },
      }
    },
  })
}
