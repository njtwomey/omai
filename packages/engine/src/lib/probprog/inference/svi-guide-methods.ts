/**
 * Stochastic variational inference with a guide that follows the model, each as its own method with its own knobs and tape
 * beside `sviMethod`: structured and flow guides, optionally started at the Laplace approximation (`structuredSviMethod`), a
 * mixture-of-Gaussians guide whose component is enumerated (`mixtureSviMethod`), and an amortised guide whose local latents
 * come from a neural encoder of each data point (`amortisedSviMethod`).
 */

import { choice, dict, int } from '../../space/index.ts'
import { full, type TensorLike, toFlat } from '../../tensor/index.ts'
import type { Config } from '../../trace/index.ts'
import { traceElbo, traceMeanFieldElbo } from '../elbo.ts'
import { traceEnumElbo } from '../elbo-enum.ts'
import { stlElbo } from '../elbo-extra.ts'
import type { SviState } from '../engines/svi.ts'
import { amortisedGuide } from '../guides-amortised.ts'
import { type MethodModel, type ModelAnalysis, num, pick, programOf } from './_support.ts'
import {
  guideKnobs,
  initKnob,
  optimiserKnobs,
  type VariationalOptions,
  type VariationalPath,
  variationalMethod,
} from './_variational.ts'
import type { InferenceMethod } from './method.ts'

type Made = InferenceMethod<SviState, Config, VariationalPath, ModelAnalysis>

const STRUCTURED = [
  'autoStructured',
  'autoMultivariateNormal',
  'autoLowRankMultivariateNormal',
  'autoFlowNormal',
  'autoIafNormal',
  'autoNormal',
] as const

/**
 * SVI with a structured, low-rank, full-rank or normalising-flow guide, from the model's graph or the user's choice,
 * optionally started at the Laplace approximation of the model (`init: laplace`). Knobs: the family, what a structured
 * guide's blocks depend on and their covariance, the flow's layer and depth, the low-rank factor's rank, the estimator
 * (`stl`, `trace`, `meanField`) and its particles, the optimiser. The tape is that of the other SVI methods.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - A name and a guide written by hand.
 * @returns The method.
 * @concept variational-inference
 * @index estimator Variational inference with structured, full-rank, low-rank or flow guides, optionally Laplace-started.
 * @example
 * ```ts
 * import { structuredSviMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { add } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const a = sample('a', normal(0, 1)); const b = sample('b', normal(0, 1)); observe('y', normal(add(a, b), 0.5), 1) }
 * session(structuredSviMethod(model), { steps: 10, mode: 'eager' }).status // => 'completed'
 * ```
 */
export function structuredSviMethod(model: MethodModel, options: VariationalOptions = {}): Made {
  return variationalMethod(
    model,
    {
      name: 'svi-structured',
      description: 'SVI with a structured or flow guide, optionally started at the Laplace approximation',
      guides: STRUCTURED,
      init: true,
      knobs: dict({
        ...guideKnobs(STRUCTURED),
        ...initKnob,
        estimator: choice(['stl', 'trace', 'meanField'], { default: 'stl', label: 'estimator' }),
        numParticles: int(1, 256, { default: 2, label: 'particles' }),
        ...optimiserKnobs,
      }),
      estimator: (c) =>
        pick(c, 'estimator') === 'trace'
          ? traceElbo({ numParticles: num(c, 'numParticles') })
          : pick(c, 'estimator') === 'meanField'
            ? traceMeanFieldElbo({ numParticles: num(c, 'numParticles') })
            : stlElbo({ numParticles: num(c, 'numParticles') }),
    },
    options,
  )
}

/**
 * SVI with a mixture-of-Gaussians guide whose component is a guide site summed out exactly (`traceEnumElbo`), for
 * posteriors with several modes. Knobs: the number of components, the most joint states, the particles of the continuous
 * draws, the optimiser. The bound is that of the guide's joint of component and latent, below the mixture's own by the
 * information the latent carries about the component. The tape adds `svi/diag/cells` and `svi/diag/maxWeight`.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - A name.
 * @returns The method.
 * @concept variational-inference
 * @index estimator Variational inference with a mixture guide whose component is enumerated, for multimodal posteriors.
 * @example
 * ```ts
 * import { mixtureSviMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * session(mixtureSviMethod(model), { steps: 10, mode: 'eager' }).status // => 'completed'
 * ```
 */
export function mixtureSviMethod(model: MethodModel, options: VariationalOptions = {}): Made {
  return variationalMethod(
    model,
    {
      name: 'svi-mixture',
      description: 'SVI with a mixture-of-Gaussians guide whose component is enumerated',
      guides: ['autoMixtureNormal'],
      knobs: dict({
        ...guideKnobs(['autoMixtureNormal']),
        maxCells: int(1, 4096, { default: 64, label: 'most joint states' }),
        numParticles: int(1, 256, { default: 2, label: 'particles' }),
        ...optimiserKnobs,
      }),
      estimator: (c) =>
        traceEnumElbo({ guide: 'all', maxCells: num(c, 'maxCells'), numParticles: num(c, 'numParticles') }),
    },
    options,
  )
}

/** What an amortised method needs that a knob cannot carry. */
export interface AmortisedMethodOptions {
  readonly name?: string
  /** The local sites the encoder covers (latents of shape `[N]` or `[N, k]` under one plate). */
  readonly sites: readonly string[]
  /** The features of every element of the plate, `[N, d]`. */
  readonly input: () => TensorLike
}

/**
 * SVI with an amortised guide: the local latents of the listed sites come from a neural encoder of each data point's
 * features (hidden layers and width, activation, are knobs), the other sites are mean-field. The encoder's weights are
 * the guide's parameters, so they ride the tape as `svi/params/<site>_enc_<i>` and the posterior of any state reads through the
 * encoder.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The amortised sites, the features and a name.
 * @returns The method.
 * @concept variational-inference
 * @index estimator Variational inference with a neural encoder for the local latents (amortised inference).
 * @example
 * ```ts
 * import { amortisedSviMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, plate, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const x = [0.5, -0.3, 1.1]
 * const model = () => plate('n', 3, () => { const z = sample('z', normal(0, 1)); observe('x', normal(z, 0.5), x) })
 * session(amortisedSviMethod(model, { sites: ['z'], input: () => x }), { steps: 5, mode: 'eager' }).status // => 'completed'
 * ```
 */
export function amortisedSviMethod(model: MethodModel, options: AmortisedMethodOptions): Made {
  const program = programOf(model)
  return variationalMethod(
    model,
    {
      name: 'svi-amortised',
      description: 'SVI with a neural encoder for the local latents',
      guides: ['amortised'],
      guideKnobs: ['hiddenLayers', 'hiddenWidth', 'activation'],
      knobs: dict({
        hiddenLayers: int(0, 4, { default: 1, label: 'hidden layers', description: 'of the encoder (0: linear)' }),
        hiddenWidth: int(1, 256, { default: 16, label: 'hidden width' }),
        activation: choice(['tanh', 'relu'], { default: 'tanh', label: 'activation' }),
        estimator: choice(['stl', 'trace'], { default: 'stl', label: 'estimator' }),
        numParticles: int(1, 64, { default: 2, label: 'particles' }),
        ...optimiserKnobs,
      }),
      estimator: (c) =>
        pick(c, 'estimator') === 'trace'
          ? traceElbo({ numParticles: num(c, 'numParticles') })
          : stlElbo({ numParticles: num(c, 'numParticles') }),
    },
    {
      ...(options.name === undefined ? {} : { name: options.name }),
      guideFrom: (c) =>
        amortisedGuide(program, {
          sites: options.sites,
          input: options.input,
          hidden: toFlat(full([num(c, 'hiddenLayers')], num(c, 'hiddenWidth'))),
          activation: pick(c, 'activation') as 'tanh' | 'relu',
        }),
    },
  )
}
