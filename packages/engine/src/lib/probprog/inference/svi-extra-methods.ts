/**
 * Stochastic variational inference with another way of forming the gradient, each as its own method with its own knobs
 * and tape beside `sviMethod`: the Rao-Blackwellised score-function ELBO with baselines for non-reparameterised sites
 * (`traceGraphSviMethod`), the ELBO with discrete latents summed out (`traceEnumSviMethod`), sticking the landing and the
 * doubly reparameterised importance-weighted gradient (`stickingTheLandingMethod`) and the Renyi alpha-divergence bound
 * (`renyiSviMethod`). Every one is a steppable, resumable method that logs the ELBO, the gradient norm and what its estimator
 * reports about itself (baselines, enumerated cells, weights), can estimate the variance of its own gradient as it runs, and
 * takes a guide written by hand as well as the automatic ones.
 */

import { choice, dict, int, real } from '../../space/index.ts'
import type { Config } from '../../trace/index.ts'
import { renyiElbo } from '../elbo.ts'
import { traceEnumElbo } from '../elbo-enum.ts'
import { dregElbo, stlElbo } from '../elbo-extra.ts'
import { traceGraphElbo } from '../elbo-graph.ts'
import type { SviState } from '../engines/svi.ts'
import type { MethodModel, ModelAnalysis } from './_support.ts'
import { num, pick } from './_support.ts'
import {
  guideKnobs,
  optimiserKnobs,
  type VariationalOptions,
  type VariationalPath,
  variationalMethod,
} from './_variational.ts'
import type { InferenceMethod } from './method.ts'

const AUTO = [
  'autoNormal',
  'autoMultivariateNormal',
  'autoLowRankMultivariateNormal',
  'autoDelta',
  'autoIafNormal',
] as const
const REPARAMETERISED = [
  'autoNormal',
  'autoMultivariateNormal',
  'autoLowRankMultivariateNormal',
  'autoIafNormal',
] as const

type Made = InferenceMethod<SviState, Config, VariationalPath, ModelAnalysis>

/**
 * SVI with the Rao-Blackwellised score-function ELBO and a control variate: for each guide site without a reparameterised
 * sampler (a discrete site, a gamma or beta without implicit gradients) only the costs that depend on the site, found by
 * tracking which draws every density was computed from, score its draw, minus a baseline (`none`, a `decay`ing moving average,
 * `leaveOneOut` over particles, or one `learned` by recursive least squares from the site's parents). The tape adds
 * `svi/diag/baseline.<site>` and `svi/diag/cost.<site>`, and `svi/gradVariance` every `varianceEvery` steps. The guide knob
 * names an automatic guide; a guide written by hand (`options.guide`) is where non-reparameterised sites come from.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - A name and a hand-written guide.
 * @returns The method.
 * @concept variational-inference
 * @index estimator Variational inference with a Rao-Blackwellised score-function ELBO and baselines (TraceGraph).
 * @example
 * ```ts
 * import { traceGraphSviMethod } from 'omai-engine/probprog/inference'
 * import { bernoulli, normal } from 'omai-engine/dists'
 * import { observe, param, sample } from 'omai-engine/probprog'
 * import { sigmoid, tensor } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const c = sample('c', bernoulli({ probs: 0.4 })); observe('y', normal(c, 1), 1) }
 * const guide = () => { sample('c', bernoulli({ logits: param('l', tensor(0)) })) }
 * const s = session(traceGraphSviMethod(model, { guide }), { steps: 200, mode: 'eager', config: { learningRate: 0.1 } })
 * s.status // => 'completed'
 * ```
 */
export function traceGraphSviMethod(model: MethodModel, options: VariationalOptions = {}): Made {
  return variationalMethod(
    model,
    {
      name: 'svi-trace-graph',
      description: 'SVI with a Rao-Blackwellised score-function ELBO and baselines for non-reparameterised sites',
      guides: AUTO,
      knobs: dict({
        ...guideKnobs(AUTO),
        baseline: choice(['none', 'decay', 'leaveOneOut', 'learned'], {
          default: 'decay',
          label: 'baseline',
          description: 'the control variate of each score term',
        }),
        baselineDecay: real(0, 0.999, { default: 0.9, label: 'baseline decay' }),
        numParticles: int(1, 256, {
          default: 2,
          label: 'particles',
          description: 'draws of the guide per estimate (leave-one-out needs 2 or more)',
        }),
        scoreAll: { ...choice(['no', 'yes'], { default: 'no', label: 'score every site' }) },
        ...optimiserKnobs,
      }),
      estimator: (c) =>
        traceGraphElbo({
          baseline: pick(c, 'baseline') as 'none' | 'decay' | 'leaveOneOut' | 'learned',
          baselineDecay: num(c, 'baselineDecay'),
          numParticles: num(c, 'numParticles'),
          scoreAll: pick(c, 'scoreAll') === 'yes',
        }),
    },
    options,
  )
}

/**
 * SVI with discrete latents summed out: the model's finite discrete sites are marginalised by plated contraction inside
 * every particle, and the guide's finite discrete sites are enumerated (every joint state run and weighted by the guide's
 * probability of it) so their gradient is exact. Knobs: which guide sites to enumerate, the most joint states per particle,
 * the particles of the continuous draws. The tape adds `svi/diag/cells` and `svi/diag/maxWeight`.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - A name and a hand-written guide.
 * @returns The method.
 * @concept variational-inference
 * @index estimator Variational inference with discrete latents summed out exactly (TraceEnum).
 * @example
 * ```ts
 * import { traceEnumSviMethod } from 'omai-engine/probprog/inference'
 * import { bernoulli, normal } from 'omai-engine/dists'
 * import { observe, param, sample } from 'omai-engine/probprog'
 * import { tensor } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const c = sample('c', bernoulli({ probs: 0.4 })); observe('y', normal(c, 1), 1) }
 * const guide = () => { sample('c', bernoulli({ logits: param('l', tensor(0)) })) }
 * const s = session(traceEnumSviMethod(model, { guide }), { steps: 100, mode: 'eager', config: { learningRate: 0.1 } })
 * s.status // => 'completed'
 * ```
 */
export function traceEnumSviMethod(model: MethodModel, options: VariationalOptions = {}): Made {
  return variationalMethod(
    model,
    {
      name: 'svi-trace-enum',
      description: 'SVI with the discrete latents of the model summed out and those of the guide enumerated',
      guides: AUTO,
      knobs: dict({
        ...guideKnobs(AUTO),
        enumerateGuide: choice(['all', 'marked', 'none'], {
          default: 'all',
          label: 'enumerate guide sites',
          description: 'which finite discrete guide sites are enumerated',
        }),
        maxCells: int(1, 100_000, {
          default: 256,
          label: 'most joint states',
          description: 'per particle; more is refused',
        }),
        numParticles: int(1, 256, { default: 1, label: 'particles' }),
        ...optimiserKnobs,
      }),
      estimator: (c) =>
        traceEnumElbo({
          guide: pick(c, 'enumerateGuide') as 'all' | 'marked' | 'none',
          maxCells: num(c, 'maxCells'),
          numParticles: num(c, 'numParticles'),
        }),
    },
    options,
  )
}

/**
 * SVI whose gradient sticks the landing: the guide's density is evaluated at stopped parameters, so the gradient's noise
 * vanishes as the guide approaches the posterior (`stl`, one particle), or, for several particles, the doubly
 * reparameterised gradient of the importance-weighted bound (`dreg`). Needs reparameterised guide sites. The tape adds
 * `svi/diag/ess` and `svi/diag/maxWeight` for `dreg`.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - A name and a hand-written guide.
 * @returns The method.
 * @concept variational-inference
 * @index estimator Variational inference with sticking-the-landing and doubly reparameterised gradients.
 * @example
 * ```ts
 * import { stickingTheLandingMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * session(stickingTheLandingMethod(model), { steps: 20, mode: 'eager' }).status // => 'completed'
 * ```
 */
export function stickingTheLandingMethod(model: MethodModel, options: VariationalOptions = {}): Made {
  return variationalMethod(
    model,
    {
      name: 'svi-stl',
      description: 'SVI with sticking-the-landing and doubly reparameterised gradients',
      guides: REPARAMETERISED,
      knobs: dict({
        ...guideKnobs(REPARAMETERISED),
        estimator: choice(['stl', 'dreg'], {
          default: 'stl',
          label: 'estimator',
          description: 'stl: one particle; dreg: the importance-weighted bound',
        }),
        numParticles: int(1, 256, {
          default: 5,
          label: 'particles',
          description: 'of dreg (stl uses one per estimate)',
        }),
        ...optimiserKnobs,
      }),
      estimator: (c) =>
        pick(c, 'estimator') === 'dreg' ? dregElbo({ numParticles: num(c, 'numParticles') }) : stlElbo(),
    },
    options,
  )
}

/**
 * SVI on the Renyi alpha-divergence bound (Li and Turner 2016): alpha = 0 is the importance-weighted bound, alpha tending
 * to 1 the ELBO, a negative alpha a mass-covering fit. Knobs: alpha and the particles.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - A name and a hand-written guide.
 * @returns The method.
 * @concept variational-inference
 * @index estimator Variational inference on the Renyi alpha-divergence bound.
 * @example
 * ```ts
 * import { renyiSviMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * session(renyiSviMethod(model), { steps: 20, mode: 'eager', config: { alpha: 0.5 } }).status // => 'completed'
 * ```
 */
export function renyiSviMethod(model: MethodModel, options: VariationalOptions = {}): Made {
  return variationalMethod(
    model,
    {
      name: 'svi-renyi',
      description: 'SVI on the Renyi alpha-divergence bound',
      guides: REPARAMETERISED,
      knobs: dict({
        ...guideKnobs(REPARAMETERISED),
        alpha: real(-4, 0.99, {
          default: 0.5,
          label: 'alpha',
          description: '0: importance-weighted bound; towards 1: the ELBO; negative: mass-covering',
        }),
        numParticles: int(2, 256, { default: 4, label: 'particles' }),
        ...optimiserKnobs,
      }),
      estimator: (c) => renyiElbo({ alpha: num(c, 'alpha'), numParticles: num(c, 'numParticles') }),
    },
    options,
  )
}
