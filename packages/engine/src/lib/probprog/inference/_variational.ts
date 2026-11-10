/**
 * What the variational methods beyond `sviMethod` share, and nothing that chooses between them: the table of automatic
 * guides by name, the common knobs (guide, particles, optimiser, learning rate, tolerance, gradient-variance estimates),
 * and `variationalMethod`, which builds one method from a way of making its ELBO estimator and its own knobs. Each method
 * built from it has its own `Space`, its own tape paths and its own name; the step is one optimiser step. Internal to the
 * folder.
 */

import { type GradientTransformation, globalNorm, optimisers, schedules } from '../../optim/index.ts'
import { foldIn, key as makeKey } from '../../random/index.ts'
import { choice, completeOption, type DictSpace, int, real, type Space } from '../../space/index.ts'
import { item, primalOf, type Tensor } from '../../tensor/index.ts'
import {
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  type StepContext,
  scalar,
  tensor,
} from '../../trace/index.ts'
import type { ElboEstimator } from '../elbo.ts'
import { gradientMoments } from '../elbo-diagnostics.ts'
import { constrainParams, initialParams, type SviState, sviAlgorithm, sviEngine, sviPosterior } from '../engines/svi.ts'
import {
  type AutoGuide,
  autoDelta,
  autoIafNormal,
  autoLowRankMultivariateNormal,
  autoMultivariateNormal,
  autoNormal,
} from '../guides.ts'
import { autoFlowNormal } from '../guides-flow.ts'
import { type LaplaceInitGuide, laplaceInitialParams } from '../guides-init.ts'
import { autoMixtureNormal } from '../guides-mixture.ts'
import { autoStructured } from '../guides-structured.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
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

/** The paths the variational methods log (`svi/diag/*` carries what the estimator reports about itself). */
export type VariationalPath =
  | 'svi/elbo'
  | 'svi/loss'
  | 'svi/smoothed'
  | 'svi/gradNorm'
  | 'svi/gradVariance'
  | `svi/diag/${string}`
  | `svi/params/${string}`
  | 'svi/snapshot'

/** A way to make an automatic guide from a program and a configuration. */
export type GuideMaker = (program: Model, config: Config) => AutoGuide<void>

/** The automatic guides a method can offer by name. */
export const guideMakers: Record<string, GuideMaker> = {
  autoNormal: (m) => autoNormal(m),
  autoDelta: (m) => autoDelta(m),
  autoMultivariateNormal: (m) => autoMultivariateNormal(m),
  autoLowRankMultivariateNormal: (m, c) =>
    autoLowRankMultivariateNormal(m, num(c, 'rank') > 0 ? { rank: num(c, 'rank') } : {}),
  autoIafNormal: (m, c) => autoIafNormal(m, { numFlows: num(c, 'numFlows') }),
  autoFlowNormal: (m, c) =>
    autoFlowNormal(m, { flow: pick(c, 'flow') as 'planar' | 'radial' | 'coupling', numFlows: num(c, 'numFlows') }),
  autoStructured: (m, c) =>
    autoStructured(m, {
      dependencies: pick(c, 'dependencies') as 'none' | 'parents' | 'markov' | 'full',
      ...(pick(c, 'within') === 'auto' ? {} : { within: pick(c, 'within') as 'diag' | 'full' }),
    }),
  autoMixtureNormal: (m, c) => autoMixtureNormal(m, { components: num(c, 'components') }),
}

/** The knobs that decide which guide is built (the cache key of a guide, and what `init` reads). */
const GUIDE_KNOBS = ['guide', 'rank', 'numFlows', 'dependencies', 'within', 'flow', 'components'] as const
const guideKey = (config: Config): string => JSON.stringify(GUIDE_KNOBS.map((k) => config[k] ?? null))

/** The knobs of the guide choice: the family and the ones of the families that have any. */
export const guideKnobs = (names: readonly string[]): Record<string, Space> => ({
  guide: choice([...names], {
    default: names[0] ?? 'autoNormal',
    label: 'guide',
    description: 'the variational family',
  }),
  rank: int(0, 64, {
    default: 0,
    label: 'rank',
    description: 'columns of the low-rank covariance factor (0: ceil sqrt D)',
  }),
  numFlows: int(1, 8, {
    default: 2,
    label: 'flow layers',
    description: 'layers of autoIafNormal and autoFlowNormal',
  }),
  flow: choice(['planar', 'radial', 'coupling'], {
    default: 'planar',
    label: 'flow',
    description: 'the layer of autoFlowNormal',
  }),
  dependencies: choice(['none', 'parents', 'markov', 'full'], {
    default: 'markov',
    label: 'dependencies',
    description:
      'what a block of autoStructured depends on: nothing, its parents, its Markov blanket, every earlier block',
  }),
  within: choice(['auto', 'diag', 'full'], {
    default: 'auto',
    label: 'covariance within a block',
    description: 'autoStructured: full for a site with no plate and diagonal under one (auto), or the same for all',
  }),
  components: int(1, 16, { default: 2, label: 'components', description: 'Gaussians of autoMixtureNormal' }),
})

/** The `init` knob, for the methods whose guide families can start at the Laplace approximation of the model. */
export const initKnob: Record<string, Space> = {
  init: choice(['random', 'laplace'], {
    default: 'random',
    label: 'initialisation',
    description: 'random starting parameters, or the Laplace approximation of the model where the guide family allows',
  }),
}

/** The knobs the variational methods share: the optimiser, the step size, the tolerance and the variance estimates. */
export const optimiserKnobs: Record<string, Space> = {
  optimiser: { ...optimisers.space, default: 'adam' },
  schedule: { ...schedules.space, default: 'constant' },
  learningRate: real(1e-5, 1, { log: true, default: 0.01, label: 'learning rate' }),
  tol: real(0, 1, {
    default: 0,
    label: 'tolerance',
    description: 'stop when the smoothed ELBO settles by this much (0: off)',
  }),
  varianceEvery: int(0, 100_000, {
    default: 0,
    label: 'gradient variance every',
    description:
      'estimate the gradient variance every this many steps (0: never); it costs `varianceRepeats` extra estimates',
  }),
  varianceRepeats: int(2, 200, {
    default: 12,
    label: 'variance repeats',
    description: 'estimates per gradient-variance reading',
  }),
}

/** What a variational method is made of. */
export interface VariationalSpec {
  /** The method's name (the tape's prefix is `svi/` for all of them). */
  readonly name: string
  readonly description: string
  /** The method's own knobs: guide, estimator knobs, optimiser knobs (compose with {@link guideKnobs} and {@link optimiserKnobs}). */
  readonly knobs: DictSpace
  /** The estimator for a configuration. */
  readonly estimator: (config: Config) => ElboEstimator
  /** The automatic guides on offer (the first is the default of `guideKnobs`). */
  readonly guides: readonly string[]
  /** For a guide made by a function of the configuration: the knobs it reads (they key the guide's cache). */
  readonly guideKnobs?: readonly string[]
  /** Starting parameters (constrained, by name) for a configuration, or none: the Laplace start of the `init` knob. */
  readonly init?: boolean
}

/** Options of a variational method's factory. */
export interface VariationalOptions {
  readonly name?: string
  /** A guide written by hand, in place of the automatic guide the `guide` knob names. */
  readonly guide?: Model
  /** A function of the configuration that builds a guide (an amortised guide whose encoder's size is a knob). */
  readonly guideFrom?: (config: Config) => Model
}

/**
 * Build a method from a spec: SVI with the spec's estimator, one optimiser step per step, the ELBO, loss, gradient norm,
 * the estimator's own diagnostics and the parameters on the tape, the state serialisable, `posterior(state)` the
 * variational posterior of the parameters at that step.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param spec - The estimator, knobs and guides of the method.
 * @param options - A name and a hand-written guide.
 * @returns The method.
 */
export function variationalMethod(
  model: MethodModel,
  spec: VariationalSpec,
  options: VariationalOptions = {},
): InferenceMethod<SviState, Config, VariationalPath, ModelAnalysis> {
  const program = programOf(model)
  const name = options.name ?? spec.name
  const primeKey = foldIn(makeKey(0), 0)
  const guides = new Map<string, AutoGuide<void> | Model>()
  const custom = options.guide !== undefined || options.guideFrom !== undefined
  const keyOf = (config: Config): string =>
    custom ? JSON.stringify((spec.guideKnobs ?? []).map((k) => config[k] ?? null)) : guideKey(config)
  const guideFor = (config: Config): AutoGuide<void> | Model => {
    const cacheKey = keyOf(config)
    let g = guides.get(cacheKey)
    if (g !== undefined) return g
    if (options.guideFrom !== undefined) g = options.guideFrom(config)
    else if (options.guide !== undefined) g = options.guide
    else {
      const kind = pick(config, 'guide')
      const make = guideMakers[kind]
      if (make === undefined) throw new Error(`${name}: no guide called '${kind}'`)
      g = make(program, config)
    }
    guides.set(cacheKey, g)
    return g
  }
  const laplaceCache = new Map<string, Record<string, Tensor>>()
  /** The Laplace start of the configured guide family (computed once per family), or none where the family cannot start there. */
  const laplaceStart = (config: Config): Record<string, Tensor> | undefined => {
    const kind = pick(config, 'guide')
    const supported: readonly string[] = [
      'autoNormal',
      'autoDelta',
      'autoMultivariateNormal',
      'autoLowRankMultivariateNormal',
      'autoStructured',
      'autoIafNormal',
    ]
    if (custom || !supported.includes(kind)) return undefined
    const cacheKey = keyOf(config)
    let p = laplaceCache.get(cacheKey)
    if (p === undefined) {
      p = laplaceInitialParams(program, undefined, {
        guide: kind as LaplaceInitGuide,
        ...(num(config, 'rank') > 0 ? { rank: num(config, 'rank') } : {}),
        structured: {
          dependencies: pick(config, 'dependencies') as 'none' | 'parents' | 'markov' | 'full',
          ...(pick(config, 'within') === 'auto' ? {} : { within: pick(config, 'within') as 'diag' | 'full' }),
        },
      }).params as Record<string, Tensor>
      laplaceCache.set(cacheKey, p)
    }
    return p
  }
  const supportsOf = new Map<string, ReturnType<typeof initialParams>['supports']>()
  const posterior = (state: SviState, config: Config): Posterior => {
    const g = guideFor(config)
    const cacheKey = keyOf(config)
    let supports = supportsOf.get(cacheKey)
    if (supports === undefined) {
      supports = initialParams(primeKey, program, g, undefined).supports
      supportsOf.set(cacheKey, supports)
    }
    const params = constrainParams(state.params, supports)
    try {
      return sviPosterior(g, undefined, params, {
        key: primeKey,
        status: 'completed',
        diagnostics: {
          elbo: Number.isFinite(state.elbo) ? state.elbo : null,
          guide: (g as Partial<AutoGuide<void>>).kind ?? 'custom',
        },
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
  const base = checkOf(sviEngine(), (a) => {
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
  // A guide written by hand is the user's own: the method needs only a likelihood.
  const check = (a: ModelAnalysis) => (custom ? yes('a guide written by hand') : base(a))
  const ensure = guard(name, model, check)
  return defineInferenceMethod<SviState, Config, VariationalPath, ModelAnalysis>({
    name,
    family: 'variational',
    description: spec.description,
    config: spec.knobs,
    check,
    posterior,
    defaults: { 'svi/elbo': Log(), 'svi/loss': Log() },
    contract: (config): Contract<SviState, VariationalPath> =>
      defineContract({
        'svi/elbo': scalar({
          description: 'the estimated evidence lower bound (or the bound the estimator targets)',
          direction: 'higher',
        }),
        'svi/loss': scalar({ description: 'the surrogate loss, minus the ELBO', direction: 'lower' }),
        'svi/smoothed': scalar({
          description: 'the ELBO as a moving average (what the tolerance reads)',
          direction: 'higher',
        }),
        'svi/gradNorm': scalar({ description: 'the norm of the gradient the optimiser was given', direction: 'lower' }),
        'svi/gradVariance': scalar({
          description: 'the total variance of the gradient estimator at the current state, over repeated estimates',
          direction: 'lower',
        }),
        'svi/diag/*': scalar({
          description: 'what the estimator reports about its last estimate (baselines, costs, cells, weights)',
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
      const estimator = spec.estimator(config)
      let last: Readonly<Record<string, number>> = {}
      // The estimator and the optimiser are seen through wrappers that keep what the tape wants of one step.
      const watched: ElboEstimator = {
        ...estimator,
        estimate: (k, m, g, d, p, s) => {
          const r = estimator.estimate(k, m, g, d, p, s)
          last = r.diagnostics ?? {}
          return r
        },
      }
      const inner0: GradientTransformation = optimisers.make(completeOption(optimisers, config['optimiser']))(
        schedules.make(completeOption(schedules, config['schedule']))(num(config, 'learningRate')),
      )
      let gradNorm = Number.NaN
      const optimizer: GradientTransformation = {
        ...inner0,
        update: (grads, state, params) => {
          gradNorm = item(primalOf(globalNorm(grads)))
          return inner0.update(grads, state, params)
        },
      }
      const guide = guideFor(config)
      const initParams = spec.init === true && config['init'] === 'laplace' ? laplaceStart(config) : undefined
      const inner = sviAlgorithm(program, undefined, {
        ...(initParams === undefined ? {} : { initParams }),
        guide: guide as Model,
        elbo: watched,
        optimizer,
        tol: num(config, 'tol'),
      })
      const alg = restorable(inner, foldIn(key, 0))
      const varianceEvery = num(config, 'varianceEvery')
      const repeats = num(config, 'varianceRepeats')
      return {
        ...alg,
        init: (k) => {
          ensure()
          return alg.init(k)
        },
        step: (s, ctx) => {
          const next = alg.step(s, ctx as unknown as StepContext)
          if (Number.isFinite(next.elbo)) {
            ctx.log('svi/elbo', next.elbo)
            ctx.log('svi/loss', -next.elbo)
          }
          if (Number.isFinite(next.smoothed)) ctx.log('svi/smoothed', next.smoothed)
          if (Number.isFinite(gradNorm)) ctx.log('svi/gradNorm', gradNorm)
          for (const [k, v] of Object.entries(last)) if (Number.isFinite(v)) ctx.log(`svi/diag/${k}`, v)
          for (const [k, value] of Object.entries(next.params)) {
            const path = `svi/params/${k}` as const
            if (ctx.wants(path)) ctx.log(path, value)
          }
          if (varianceEvery > 0 && ctx.wants('svi/gradVariance') && ctx.t % varianceEvery === 0) {
            const g = guide
            const params = Object.fromEntries(
              Object.entries(constrainParams(next.params, initialParams(primeKey, program, g, undefined).supports)),
            )
            const m = gradientMoments({
              estimator: estimator,
              model: program,
              guide: g as Model,
              data: undefined,
              params,
              key: foldIn(ctx.key, 0x7a11),
              repeats,
              state: next.elboState,
            })
            ctx.log('svi/gradVariance', m.totalVariance)
          }
          return next
        },
      }
    },
  })
}
