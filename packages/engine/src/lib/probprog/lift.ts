/**
 * Lifting parameters to random variables, Pyro and NumPyro's `lift` handler: a program written with `param` sites (a
 * maximum likelihood or point-estimate program) becomes a fully Bayesian one by giving each parameter a prior, with no
 * change to its source. The sites turn into sample sites of that prior, so every density, sampler and engine treats
 * them as latent variables.
 */

import type { Distribution } from '../dists/index.ts'
import { handle } from './handlers.ts'
import type { Message, Model } from './primitives.ts'

/** The priors of {@link lift}: one distribution for every param site, or a distribution by site name. */
export type LiftPriors = Distribution | Readonly<Record<string, Distribution>>

/**
 * Turn `param` sites into `sample` sites with priors (Pyro's `lift`): wherever the program calls `param(name, ...)` it
 * now draws `name` from the prior (a distribution for every param, or one by name; param sites without a prior are left
 * alone). Repeated calls of `param` with one name share a single draw within a run, as a parameter store would. The
 * prior must have the batch shape the site needs (under a plate, `.expand`ed to the plate), as in NumPyro.
 *
 * Apply `lift` inside `seed`, `condition` and `substitute` (they must see the site as a sample site).
 *
 * @param model - The program.
 * @param priors - A distribution, or distributions by site name.
 * @returns The lifted program.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { logJoint, lift, observe, param } from 'omai-engine/probprog'
 * import { exponential, normal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * const model = () => { const s = param('s', 1, { support: { kind: 'positive' } }); observe('y', normal(0, s), 0.5) }
 * // With the prior it is the program that draws s from Exponential(2): the density gains log Exp(s = 1; 2).
 * const lifted = item(logJoint(lift(model, { s: exponential(2) }), undefined, { s: 1 }).logJoint)
 * const gain = lifted - item(logJoint(model, undefined).logJoint)
 * Math.abs(gain - (Math.log(2) - 2)) < 1e-14 // => true
 * ```
 */
export function lift<D = void>(model: Model<D>, priors: LiftPriors): Model<D> {
  const prior = (name: string): Distribution | undefined => {
    if (isDistribution(priors)) return priors
    return priors[name]
  }
  return handle(model, () => {
    const drawn = new Map<string, Message>()
    return {
      process: (msg) => {
        if (msg.type !== 'param') return
        const fn = prior(msg.name)
        if (fn === undefined) return
        msg.type = 'sample'
        msg.fn = fn
        msg.init = undefined
        msg.sampleShape = []
        msg.infer = {}
        const first = drawn.get(msg.name)
        if (first === undefined) drawn.set(msg.name, msg)
        else {
          // A second call of the same param: block the site and reuse the first draw.
          msg.value = first.value
          msg.isObserved = true
          msg.stop = true
        }
      },
    }
  })
}

function isDistribution(x: LiftPriors): x is Distribution {
  return typeof (x as Distribution).logProb === 'function'
}
