/**
 * Natural-gradient stochastic variational inference as methods, each with its own knobs and tape: `naturalGradientSviMethod`
 * fits a mean-field or full-rank Gaussian guide to any differentiable model by steps in the mean parameters, and
 * `conjugateNaturalSviMethod` runs minibatch conjugate updates in natural parameters on a conjugate-exponential model (batch
 * CAVI at rate one). One step per step, a state that serialises, `posterior(state)` the variational posterior of any step, the
 * natural gradient's length and the rate on the tape.
 */

import { foldIn, key as makeKey } from '../../random/index.ts'
import { choice, dict, int, real } from '../../space/index.ts'
import { primalOf, type Tensor, toFlat, type Value } from '../../tensor/index.ts'
import { type Config, type Contract, defineContract, derived, Log, scalar, tensor } from '../../trace/index.ts'
import { traceElbo } from '../elbo.ts'
import { sviPosterior } from '../engines/svi.ts'
import {
  type ConjugateNaturalState,
  conjugateNaturalGradientAlgorithm,
  conjugateNaturalMarginals,
  type GaussianNaturalState,
  gaussianNaturalGradientAlgorithm,
} from '../engines/svi-natural.ts'
import { autoMultivariateNormal, autoNormal } from '../guides.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import {
  guard,
  type MethodModel,
  type ModelAnalysis,
  num,
  pick,
  posteriorJson,
  programOf,
  restorable,
} from './_support.ts'
import { conjugateMethod } from './exact-methods.ts'
import { defineInferenceMethod, type InferenceMethod } from './method.ts'
import { sviMethod } from './svi-method.ts'

/** The paths the Gaussian natural-gradient method logs. */
export type NaturalPath =
  | 'svi/elbo'
  | 'svi/loss'
  | 'svi/smoothed'
  | 'svi/naturalGradNorm'
  | 'svi/rate'
  | `svi/params/${string}`
  | 'svi/snapshot'

const rateKnobs = (delay: number) => ({
  rate0: real(0.01, 1, {
    log: true,
    default: 1,
    label: 'first rate',
    description: 'the first step size (1: the CAVI step on a conjugate model)',
  }),
  delay: real(0, 1000, {
    default: delay,
    label: 'delay',
    description: 'added to the step before the power of the rate',
  }),
  forgetting: real(0, 1, {
    default: 0.7,
    label: 'forgetting',
    description: 'the exponent of the Robbins-Monro rate (0: constant)',
  }),
})

/**
 * Natural-gradient SVI with a mean-field or full-rank Gaussian guide on the unconstrained scale: the step is the ordinary
 * gradient in the guide's mean parameters, with a Robbins-Monro rate (`rate0 (t + delay)^-forgetting`). On a conjugate Gaussian
 * model it reaches the posterior in a few steps where the plain gradient takes hundreds. Knobs: the family, the estimator's
 * particles, the rate, the tolerance. The tape adds `svi/naturalGradNorm` and `svi/rate`.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @returns The method.
 * @concept variational-inference
 * @index estimator Natural-gradient SVI with a mean-field or full-rank Gaussian guide.
 * @example
 * ```ts
 * import { naturalGradientSviMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * const method = naturalGradientSviMethod(model)
 * const s = session(method, { steps: 60, mode: 'eager', config: { numParticles: 16 } })
 * Math.abs(item(method.posterior?.(s.state, s.config).mean('m') ?? 0) - 1) < 0.1 // => true
 * ```
 */
export function naturalGradientSviMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<GaussianNaturalState, Config, NaturalPath, ModelAnalysis> {
  const program = programOf(model)
  const name = options.name ?? 'svi-natural'
  const primeKey = foldIn(makeKey(0), 0)
  const guides = {
    meanField: autoNormal(program, { initScale: 0.5 }),
    fullRank: autoMultivariateNormal(program, { initScale: 0.5 }),
  }
  const check = sviMethod(model).check
  const ensure = guard(name, model, check)
  const posterior = (state: GaussianNaturalState, config: Config): Posterior => {
    const g = guides[pick(config, 'family') as 'meanField' | 'fullRank']
    return sviPosterior(g, undefined, state.params as Record<string, Tensor>, {
      key: primeKey,
      status: 'completed',
      diagnostics: { elbo: Number.isFinite(state.elbo) ? state.elbo : null, guide: g.kind },
    })
  }
  return defineInferenceMethod<GaussianNaturalState, Config, NaturalPath, ModelAnalysis>({
    name,
    family: 'variational',
    description: 'natural-gradient SVI with a mean-field or full-rank Gaussian guide',
    config: dict({
      family: choice(['meanField', 'fullRank'], {
        default: 'meanField',
        label: 'guide',
        description: 'a normal per site, or one over the whole latent vector',
      }),
      numParticles: int(1, 256, { default: 8, label: 'particles', description: 'draws of the guide per gradient' }),
      ...rateKnobs(5),
      tol: real(0, 1, {
        default: 0,
        label: 'tolerance',
        description: 'stop when the smoothed ELBO settles by this much (0: off)',
      }),
    }),
    ...(check === undefined ? {} : { check }),
    posterior,
    defaults: { 'svi/elbo': Log(), 'svi/naturalGradNorm': Log() },
    contract: (config): Contract<GaussianNaturalState, NaturalPath> =>
      defineContract({
        'svi/elbo': scalar({ description: 'the estimated evidence lower bound', direction: 'higher' }),
        'svi/loss': scalar({ description: 'minus the ELBO', direction: 'lower' }),
        'svi/smoothed': scalar({ description: 'the ELBO as a moving average', direction: 'higher' }),
        'svi/naturalGradNorm': scalar({
          description: 'the length of the natural gradient (the gradient in the mean parameters)',
          direction: 'lower',
        }),
        'svi/rate': scalar({ description: 'the step size of the natural-gradient step' }),
        'svi/params/*': tensor({
          role: 'param',
          description: 'a guide parameter (location and scale or Cholesky factor), constrained scale',
        }),
        'svi/snapshot': derived<GaussianNaturalState>({
          cost: 'dear',
          description: 'the variational posterior of the state as JSON',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config, key) => {
      const alg = restorable(
        gaussianNaturalGradientAlgorithm(program, undefined, {
          family: pick(config, 'family') as 'meanField' | 'fullRank',
          elbo: traceElbo({ numParticles: num(config, 'numParticles') }),
          rate0: num(config, 'rate0'),
          delay: num(config, 'delay'),
          forgetting: num(config, 'forgetting'),
          tol: num(config, 'tol'),
        }),
        foldIn(key, 0),
      )
      return {
        ...alg,
        init: (k) => {
          ensure()
          return alg.init(k)
        },
        step: (s, ctx) => {
          const next = alg.step(s, ctx)
          if (Number.isFinite(next.elbo)) {
            ctx.log('svi/elbo', next.elbo)
            ctx.log('svi/loss', -next.elbo)
          }
          if (Number.isFinite(next.smoothed)) ctx.log('svi/smoothed', next.smoothed)
          if (Number.isFinite(next.naturalGradNorm)) ctx.log('svi/naturalGradNorm', next.naturalGradNorm)
          ctx.log('svi/rate', next.rate)
          for (const [k, value] of Object.entries(next.params)) {
            const path = `svi/params/${k}` as const
            if (ctx.wants(path)) ctx.log(path, value)
          }
          return next
        },
      }
    },
  })
}

/** The paths the conjugate natural-gradient method logs. */
export type ConjugateNaturalPath = 'svi/elbo' | 'svi/naturalGradNorm' | 'svi/rate' | 'svi/snapshot'

/**
 * Natural-gradient SVI on a conjugate-exponential model: every latent node is a conjugate prior of its observations, the
 * guide is that prior's family in natural parameters, and a step moves them towards the prior plus the (rescaled) sufficient
 * statistics of a minibatch by a Robbins-Monro rate. With `batchSize` 0 (all the data) and the first rate of one, the first step
 * is batch CAVI, the exact posterior. Knobs: the minibatch, the rate schedule, the ELBO draws, the tolerance. The tape is the ELBO,
 * the natural gradient's length and the rate.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @returns The method; its `check` is the conjugate method's (every latent node a conjugate prior).
 * @concept variational-inference
 * @index estimator Natural-gradient SVI on conjugate models, minibatch CAVI with a Robbins-Monro rate.
 * @example
 * ```ts
 * import { conjugateNaturalSviMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * const method = conjugateNaturalSviMethod(model)
 * const s = session(method, { steps: 1, mode: 'eager' })
 * item(method.posterior?.(s.state, s.config).mean('m') ?? 0) // => 1
 * ```
 */
export function conjugateNaturalSviMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<ConjugateNaturalState, Config, ConjugateNaturalPath, ModelAnalysis> {
  const program = programOf(model)
  const name = options.name ?? 'svi-conjugate-natural'
  const check = conjugateMethod(model).check
  const ensure = guard(name, model, check)
  const posterior = (state: ConjugateNaturalState): Posterior => {
    const marginals = conjugateNaturalMarginals(program, undefined, state)
    return definePosterior({
      form: 'variational',
      engine: 'svi',
      sites: Object.keys(marginals),
      marginals,
      status: 'completed',
      diagnostics: { elbo: Number.isFinite(state.elbo) ? state.elbo : null, guide: 'conjugate natural parameters' },
      sampler: (key, n) =>
        Object.fromEntries(
          Object.entries(marginals).map(([site, d], i) => [site, primalOf(d.sample(foldIn(key, i), [n])) as Tensor]),
        ),
      logProb: (values) =>
        Object.entries(marginals).reduce(
          (acc, [site, d]) => acc + toFlat(primalOf(d.logProb(values[site] as Value))).reduce((a, b) => a + b, 0),
          0,
        ),
    })
  }
  return defineInferenceMethod<ConjugateNaturalState, Config, ConjugateNaturalPath, ModelAnalysis>({
    name,
    family: 'variational',
    description: 'natural-gradient SVI on a conjugate-exponential model: minibatch conjugate updates',
    config: dict({
      batchSize: int(0, 1_000_000, {
        default: 0,
        label: 'minibatch',
        description: 'observations per step (0: all, which is batch CAVI)',
      }),
      ...rateKnobs(1),
      elboDraws: int(1, 256, { default: 8, label: 'ELBO draws' }),
      tol: real(0, 100, {
        default: 0,
        label: 'tolerance',
        description: 'stop when the natural gradient is this short (0: off)',
      }),
    }),
    ...(check === undefined ? {} : { check }),
    posterior,
    defaults: { 'svi/elbo': Log(), 'svi/naturalGradNorm': Log() },
    contract: (): Contract<ConjugateNaturalState, ConjugateNaturalPath> =>
      defineContract({
        'svi/elbo': scalar({ description: 'a Monte Carlo estimate of the evidence lower bound', direction: 'higher' }),
        'svi/naturalGradNorm': scalar({
          description: 'the length of the natural gradient: the gap to the minibatch target',
          direction: 'lower',
        }),
        'svi/rate': scalar({ description: 'the step size' }),
        'svi/snapshot': derived<ConjugateNaturalState>({
          cost: 'dear',
          description: 'the variational posterior of the state as JSON',
          compute: (s) => posteriorJson(posterior(s)),
        }),
      }),
    make: (config) => {
      const batch = num(config, 'batchSize')
      const alg = restorable(
        conjugateNaturalGradientAlgorithm(program, undefined, {
          ...(batch > 0 ? { batchSize: batch } : {}),
          rate0: num(config, 'rate0'),
          delay: num(config, 'delay'),
          forgetting: num(config, 'forgetting'),
          elboDraws: num(config, 'elboDraws'),
          tol: num(config, 'tol'),
        }),
        foldIn(makeKey(0), 1),
      )
      return {
        ...alg,
        init: (k) => {
          ensure()
          return alg.init(k)
        },
        step: (s, ctx) => {
          const next = alg.step(s, ctx)
          if (Number.isFinite(next.elbo)) ctx.log('svi/elbo', next.elbo)
          if (Number.isFinite(next.naturalGradNorm)) ctx.log('svi/naturalGradNorm', next.naturalGradNorm)
          ctx.log('svi/rate', next.rate)
          return next
        },
      }
    },
  })
}
