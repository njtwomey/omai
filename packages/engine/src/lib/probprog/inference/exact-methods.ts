/**
 * The exact inference engines as methods, each honestly one step (they are one computation, not an iteration):
 * conjugate updates, linear-Gaussian conditioning (dense, Kalman or Gaussian belief propagation), junction-tree
 * enumeration of discrete models and forward-backward on chains. Each has its own knobs, a typed tape with the evidence it
 * finds, a capability check that reads a spec's static analysis and says why the method does not apply, and a posterior;
 * the one-call engines of `omai-engine/probprog` are unchanged.
 */

import { choice, dict } from '../../space/index.ts'
import type { Config } from '../../trace/index.ts'
import { chainEngine, chainInference } from '../engines/chain.ts'
import { conjugateEngine, conjugateInference } from '../engines/conjugate.ts'
import { enumerationEngine, enumerationPosterior } from '../engines/enumeration.ts'
import { gaussianEngine, gaussianInference } from '../engines/gaussian.ts'
import { checkOf, type MethodModel, type ModelAnalysis, no, pick, programOf, yes } from './_support.ts'
import type { InferenceMethod } from './method.ts'
import { type OneShotPath, type OneShotState, oneShotMethod } from './one-step.ts'

/**
 * Closed-form conjugate posteriors as a method: every latent node a conjugate prior of its observed children.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @returns The method (one step; `check` names the latent nodes with no conjugate structure).
 * @concept exact-inference
 * @index estimator Conjugate posteriors in closed form, as a one-step method.
 * @example
 * ```ts
 * import { conjugateMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const m = conjugateMethod(() => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 2) })
 * const s = session(m, { mode: 'eager', steps: 1 })
 * item(m.posterior?.(s.state, s.config).mean('mu') ?? 0) // => 1
 * ```
 */
export function conjugateMethod(
  model: MethodModel,
): InferenceMethod<OneShotState, Config, OneShotPath<'conjugate'>, ModelAnalysis> {
  const program = programOf(model)
  return oneShotMethod({
    name: 'conjugate',
    family: 'exact',
    description: 'closed-form posteriors of conjugate models',
    config: dict({}),
    model,
    check: checkOf(conjugateEngine(), (a) => {
      if (!a.hasLikelihood) return no('observed nodes have no likelihood', ['likelihood'], 'add an observed node')
      if (a.latent.length === 0) return no('no latent nodes', ['latent nodes'])
      const covered = new Set(
        a.conjugate.filter((c) => c.known && c.hyperparametersKnown && c.exclusive).map((c) => c.prior),
      )
      const missing = a.latent.filter((l) => !covered.has(l))
      if (missing.length > 0)
        return no(`no usable conjugate structure for ${missing.join(', ')}`, ['conjugate sites'], 'use NUTS or Gibbs')
      return yes(
        `every latent node is a conjugate prior (${a.conjugate.map((c) => `${c.priorFamily}-${c.likelihoodFamily}`).join(', ')})`,
      )
    }),
    run: () => conjugateInference(program, undefined),
  })
}

/**
 * Exact linear-Gaussian conditioning as a method: dense conditioning, Kalman filtering and smoothing for chains, or
 * Gaussian belief propagation (the `method` knob).
 *
 * @param model - A bound spec, or a program closed over its data.
 * @returns The method (one step; `check` reads the spec's linear-Gaussian analysis and its reason).
 * @concept gaussian-conditioning
 * @index estimator Exact linear-Gaussian conditioning, Kalman and Gaussian BP, as a one-step method.
 * @example
 * ```ts
 * import { gaussianMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const m = gaussianMethod(() => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 2) })
 * const s = session(m, { mode: 'eager', steps: 1 })
 * item(m.posterior?.(s.state, s.config).mean('mu') ?? 0) // => 1
 * ```
 */
export function gaussianMethod(
  model: MethodModel,
): InferenceMethod<OneShotState, Config, OneShotPath<'gaussian'>, ModelAnalysis> {
  const program = programOf(model)
  return oneShotMethod({
    name: 'gaussian',
    family: 'exact',
    description: 'exact conditioning of linear-Gaussian models: dense, Kalman smoothing, Gaussian BP',
    config: dict({
      method: choice(['auto', 'joint', 'kalman', 'bp'], {
        default: 'auto',
        label: 'method',
        description: 'dense conditioning, Kalman filtering and smoothing of a chain, or Gaussian belief propagation',
      }),
    }),
    model,
    check: checkOf(gaussianEngine(), (a) =>
      a.linearGaussian.holds
        ? yes(a.linearGaussian.reason)
        : no(a.linearGaussian.reason, ['linear-Gaussian'], 'use NUTS, Laplace or SVI'),
    ),
    run: (config) =>
      gaussianInference(program, undefined, { method: pick(config, 'method') as 'auto' | 'joint' | 'kalman' | 'bp' }),
  })
}

/**
 * Exact inference for discrete models as a method: the program grounded into a factor graph and calibrated on a junction
 * tree (Shafer-Shenoy or Hugin).
 *
 * @param model - A bound spec, or a program closed over its data.
 * @returns The method (one step; `check` says which latent nodes are continuous or unbounded).
 * @concept exact-inference
 * @index estimator Exact enumeration of discrete models by a junction tree, as a one-step method.
 * @example
 * ```ts
 * import { enumerationMethod } from 'omai-engine/probprog/inference'
 * import { bernoulli } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { item, take, tensor } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const m = enumerationMethod(() => {
 *   const rain = sample('rain', bernoulli({ probs: 0.2 }))
 *   observe('wet', bernoulli({ probs: take(tensor([0.1, 0.9]), rain) }), 1)
 * })
 * const s = session(m, { mode: 'eager', steps: 1 })
 * Math.abs(item(m.posterior?.(s.state, s.config).mean('rain') ?? 0) - 0.18 / 0.26) < 1e-12 // => true
 * ```
 */
export function enumerationMethod(
  model: MethodModel,
): InferenceMethod<OneShotState, Config, OneShotPath<'enumeration'>, ModelAnalysis> {
  const program = programOf(model)
  return oneShotMethod({
    name: 'enumeration',
    family: 'exact',
    description: 'exact marginals and joint draws of discrete models by junction-tree propagation',
    config: dict({
      method: choice(['shaferShenoy', 'hugin'], { default: 'shaferShenoy', label: 'propagation' }),
    }),
    model,
    check: checkOf(enumerationEngine(), (a) => {
      if (a.latent.length === 0) return no('no latent nodes', ['latent nodes'])
      if (!a.hasLikelihood) return no('observed nodes have no likelihood', ['likelihood'], 'add an observed node')
      if (a.continuous.length > 0)
        return no(
          `continuous latent nodes (${a.continuous.join(', ')})`,
          ['finite discrete latents'],
          'use NUTS, Gibbs or Laplace',
        )
      if (a.unboundedDiscrete.length > 0)
        return no(
          `unbounded discrete latent nodes (${a.unboundedDiscrete.join(', ')})`,
          ['finite discrete latents'],
          'bound them, or use Gibbs',
        )
      return yes(`every latent node is finite and discrete (${a.finiteDiscrete.join(', ')})`)
    }),
    run: (config) =>
      enumerationPosterior(program, undefined, {
        method: pick(config, 'method') === 'hugin' ? 'hugin' : 'shaferShenoy',
      }),
  })
}

/**
 * Forward-backward on a discrete-state chain as a method: posterior marginals and the evidence, exact joint draws by
 * forward filtering and backward sampling, and the Viterbi path. A program with one `markov` loop, or a spec whose only
 * latent is one `scan` with finite discrete states (a hidden Markov model).
 *
 * @param model - A program closed over its data, or a bound spec with a discrete-state scan.
 * @returns The method (one step).
 * @concept hidden-markov-models
 * @index estimator Forward-backward, FFBS and Viterbi on a discrete chain, as a one-step method.
 * @example
 * ```ts
 * import { chainMethod } from 'omai-engine/probprog/inference'
 * import { bernoulli, categorical } from 'omai-engine/dists'
 * import { markov, observe, sample } from 'omai-engine/probprog'
 * import { take, tensor, type Value } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const hmm = () => markov('t', 3, (t, z: Value | undefined) => {
 *   const s = sample(`z${t}`, categorical({ probs: z === undefined ? [0.5, 0.5] : take(tensor([[0.9, 0.1], [0.1, 0.9]]), z) }))
 *   observe(`y${t}`, bernoulli({ probs: take(tensor([0.2, 0.8]), s) }), 1)
 *   return s
 * }, undefined as Value | undefined)
 * const m = chainMethod(hmm)
 * const s = session(m, { mode: 'eager', steps: 1 })
 * m.posterior?.(s.state, s.config).diagnostics['mapPath'] // => [1, 1, 1]
 * ```
 */
export function chainMethod(
  model: MethodModel,
): InferenceMethod<OneShotState, Config, OneShotPath<'chain'>, ModelAnalysis> {
  const program = programOf(model)
  return oneShotMethod({
    name: 'chain',
    family: 'exact',
    description: 'forward-backward, FFBS and Viterbi on a discrete-state chain',
    config: dict({}),
    model,
    check: checkOf(chainEngine(), (a) => {
      if (a.scans.length !== 1)
        return no(
          'the model has no single scan',
          ['scan'],
          'give a spec with one scan, or a program with one markov loop',
        )
      if (!a.hasLikelihood) return no('observed nodes have no likelihood', ['likelihood'], 'add an emission')
      const scan = a.scans[0]
      if (a.latent.length !== 1 || scan === undefined)
        return no(
          `latent nodes outside the scan (${a.latent.filter((l) => l !== scan?.name).join(', ')})`,
          ['scan'],
          'fix the parameters, or use NUTS or SMC',
        )
      if (scan.stateKind !== 'finite-discrete' || scan.eventRank !== 0)
        return no(
          `the states of '${scan.name}' are not finite and discrete`,
          ['finite discrete states'],
          'use the Kalman smoother or the particle filter',
        )
      return yes(
        `a hidden Markov chain: '${scan.name}' over ${scan.over}, ${scan.emissions.length} emission(s) per step`,
      )
    }),
    run: () => chainInference(program, undefined),
  })
}
