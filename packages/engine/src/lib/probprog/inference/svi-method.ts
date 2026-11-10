/**
 * Stochastic variational inference as an inference method: one optimiser step per step, with the ELBO and the guide's
 * parameters on the tape, a state that serialises (so a run resumes, seeks and forks), and `posterior` reading the
 * variational posterior of any state, which is what an anytime snapshot shows while the fit goes on. The guide family, the
 * ELBO estimator and the optimiser are knobs; the one-call `svi` and its planner entry are unchanged.
 */

import { optimisers } from '../../optim/index.ts'
import { foldIn, key as makeKey } from '../../random/index.ts'
import { choice, completeOption, type DictSpace, dict, int, real } from '../../space/index.ts'
import type { StepContext } from '../../trace/index.ts'
import { type Config, type Contract, defineContract, derived, Log, scalar, tensor } from '../../trace/index.ts'
import { type ElboEstimator, importanceWeightedElbo, traceElbo, traceMeanFieldElbo } from '../elbo.ts'
import { constrainParams, initialParams, type SviState, sviAlgorithm, sviEngine, sviPosterior } from '../engines/svi.ts'
import { type AutoGuide, autoDelta, autoMultivariateNormal, autoNormal } from '../guides.ts'
import type { Posterior } from '../posterior.ts'
import { definePosterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import {
  checkOf,
  guard,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  pick,
  posteriorJson,
  priorOnly,
  programOf,
  restorable,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod } from './method.ts'

/** The paths the SVI method logs. */
export type SviPath = 'svi/elbo' | 'svi/loss' | 'svi/smoothed' | `svi/params/${string}` | 'svi/snapshot'

const GUIDES = ['autoNormal', 'autoDelta', 'autoMultivariateNormal'] as const

/** The knobs of {@link sviMethod} (the budget and snapshot knobs are added to them). */
export const sviKnobs: DictSpace = dict({
  guide: choice([...GUIDES], { default: 'autoNormal', label: 'guide', description: 'the variational family' }),
  elbo: choice(['trace', 'traceMeanField', 'importanceWeighted'], {
    default: 'trace',
    label: 'ELBO estimator',
  }),
  numParticles: int(1, 256, { default: 1, label: 'particles', description: 'draws of the guide per ELBO estimate' }),
  optimiser: { ...optimisers.space, default: 'adam' },
  learningRate: real(1e-5, 1, { log: true, default: 0.01, label: 'learning rate' }),
  tol: real(0, 1, {
    default: 0,
    label: 'tolerance',
    description: 'stop when the smoothed ELBO settles by this much (0: off)',
  }),
})

const guideOf = (name: string, program: Model): AutoGuide<void> =>
  name === 'autoDelta'
    ? autoDelta(program)
    : name === 'autoMultivariateNormal'
      ? autoMultivariateNormal(program)
      : autoNormal(program)

const elboOf = (name: string, particles: number): ElboEstimator =>
  name === 'traceMeanField'
    ? traceMeanFieldElbo({ numParticles: particles })
    : name === 'importanceWeighted'
      ? importanceWeightedElbo({ numParticles: particles })
      : traceElbo({ numParticles: particles })

/**
 * SVI on a model as a method with a fine step: one gradient step of the guide's parameters per step, the ELBO and the
 * parameters on the tape, the state serialisable. `posterior(state)` is the variational posterior of the parameters at
 * that step, so an anytime snapshot shows the approximation sharpen while the fit runs. The run ends `converged` when the
 * smoothed ELBO settles (with `tol`) and `diverged` on a non-finite ELBO; otherwise the session's `steps` or a budget stops it.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @returns The method, with `check` (a likelihood and continuous latent sites) and `posterior`.
 * @concept variational-inference
 * @index estimator Stochastic variational inference as a steppable, resumable method with anytime posteriors.
 * @example
 * ```ts
 * import { sviMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * const method = sviMethod(model)
 * const s = session(method, { steps: 1500, mode: 'eager', config: { learningRate: 0.05 } })
 * Math.abs(item(method.posterior?.(s.state, s.config).mean('m') ?? 0) - 1) < 0.15 // => true
 * ```
 */
export function sviMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<SviState, Config, SviPath, ModelAnalysis> {
  const program = programOf(model)
  const primeKey = foldIn(makeKey(0), 0)
  const guides = new Map<string, AutoGuide<void>>()
  const guide = (name: string): AutoGuide<void> => {
    let g = guides.get(name)
    if (g === undefined) {
      g = guideOf(name, program)
      guides.set(name, g)
    }
    return g
  }
  const supportsOf = new Map<string, ReturnType<typeof initialParams>['supports']>()
  const posterior = (state: SviState, config: Config): Posterior => {
    const g = guide(pick(config, 'guide'))
    const gname = pick(config, 'guide')
    let supports = supportsOf.get(gname)
    if (supports === undefined) {
      supports = initialParams(primeKey, program, g, undefined).supports
      supportsOf.set(gname, supports)
    }
    const params = constrainParams(state.params, supports)
    try {
      return sviPosterior(g, undefined, params, {
        key: primeKey,
        status: 'completed',
        diagnostics: { elbo: Number.isFinite(state.elbo) ? state.elbo : null, guide: g.kind },
      })
    } catch (e) {
      return definePosterior({
        form: 'variational',
        engine: 'svi',
        sites: [],
        status: 'failed',
        draws: {},
        diagnostics: { reason: e instanceof Error ? e.message : String(e) },
      })
    }
  }
  const check = checkOf(sviEngine(), (a) => {
    if (a.continuous.length === 0)
      return no('no continuous latent nodes', ['continuous latents'], 'use enumeration or Gibbs')
    if (a.unboundedDiscrete.length > 0)
      return no(
        `unbounded discrete latents (${a.unboundedDiscrete.join(', ')})`,
        ['finite discrete latents'],
        'bound them, or use Gibbs',
      )
    return yes(`continuous latents with a differentiable density${priorOnly(a)}`)
  })
  const ensure = guard(options.name ?? 'svi', model, check)
  return defineInferenceMethod<SviState, Config, SviPath, ModelAnalysis>({
    name: options.name ?? 'svi',
    family: 'variational',
    description: 'stochastic variational inference, one optimiser step per step',
    config: sviKnobs,
    check,
    posterior,
    defaults: { 'svi/elbo': Log(), 'svi/loss': Log() },
    contract: (config): Contract<SviState, SviPath> =>
      defineContract({
        'svi/elbo': scalar({ description: 'the estimated evidence lower bound', direction: 'higher' }),
        'svi/loss': scalar({ description: 'the surrogate loss, minus the ELBO', direction: 'lower' }),
        'svi/smoothed': scalar({
          description: 'the ELBO as a moving average (what the tolerance reads)',
          direction: 'higher',
        }),
        'svi/params/*': tensor({
          role: 'param',
          description: 'a guide parameter on the unconstrained scale (one path per parameter)',
        }),
        'svi/snapshot': derived<SviState>({
          cost: 'dear',
          description: 'the variational posterior of the state as JSON: means, sds, quantiles',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config, key) => {
      const inner = sviAlgorithm(program, undefined, {
        guide: guide(pick(config, 'guide')),
        elbo: elboOf(pick(config, 'elbo'), num(config, 'numParticles')),
        optimizer: optimisers.make(completeOption(optimisers, config['optimiser']))(num(config, 'learningRate')),
        tol: num(config, 'tol'),
      })
      // The supports of the parameters are found in `init`, so a state is restored into an algorithm that has run it, at the
      // key of the run (so the starting parameters differ by seed).
      const base = restorable(inner, foldIn(key, 0))
      return {
        ...base,
        init: (key) => {
          ensure()
          return base.init(key)
        },
        step: (s, ctx) => {
          const next = base.step(s, ctx as unknown as StepContext)
          if (Number.isFinite(next.elbo)) {
            ctx.log('svi/elbo', next.elbo)
            ctx.log('svi/loss', -next.elbo)
          }
          if (Number.isFinite(next.smoothed)) ctx.log('svi/smoothed', next.smoothed)
          for (const [name, value] of Object.entries(next.params)) {
            const path = `svi/params/${name}` as const
            if (ctx.wants(path)) ctx.log(path, value)
          }
          return next
        },
      }
    },
  })
}
