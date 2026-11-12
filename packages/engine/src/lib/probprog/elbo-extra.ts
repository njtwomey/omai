/**
 * ELBO estimators that change how the gradient of a reparameterised bound is formed, not what it is the bound of:
 * sticking the landing (Roeder, Wu and Duvenaud 2017), which drops the score part of the pathwise gradient so that it
 * vanishes when the guide equals the posterior, and the doubly reparameterised gradient of the importance-weighted bound
 * (Tucker et al. 2018), which does the same for K particles. Both need every guide site reparameterised.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import {
  add,
  exp,
  item,
  logsumexp,
  mul,
  neg,
  primalOf,
  stack,
  stopGradient,
  sub,
  sum,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { logJoint, traceLogJoint } from './density.ts'
import type { ElboEstimate, ElboEstimator } from './elbo.ts'
import { enumerate, type ProgramTrace, replay, seed, substitute, traceOf } from './handlers.ts'
import type { Model } from './primitives.ts'

const num = (v: Value): number => item(primalOf(v))

/** One particle with its guide density evaluated at stopped parameters (the sticking-the-landing form). */
interface StoppedParticle {
  readonly logP: Value
  /** log q(z) with the guide's parameters held constant, so the gradient reaches them only through z. */
  readonly logQ: Value
  readonly guideTrace: ProgramTrace
}

function stoppedParticle<D>(
  key: Key,
  model: Model<D>,
  guide: Model<D>,
  data: D,
  params: Readonly<Record<string, TensorLike>>,
  who: string,
): StoppedParticle {
  const guideTrace = traceOf(seed(substitute(guide, params), key), data)
  const values: Record<string, Value> = {}
  for (const s of guideTrace.sites.values()) {
    if (s.type !== 'sample' || s.isObserved) continue
    if (!s.fn.hasRsample)
      throw new OmaiError('value', `${who}: guide site ${s.name} has no reparameterised sampler; use traceGraphElbo`, {
        site: s.name,
      })
    values[s.name] = s.value
  }
  const stopped = Object.fromEntries(Object.entries(params).map(([k, v]) => [k, stopGradient(v)]))
  const fixed = traceOf(seed(substitute(guide, { ...stopped, ...values }), key), data)
  const r = logJoint(seed(replay(substitute(enumerate(model), params), guideTrace), foldIn(key, 1 << 30)), data)
  return { logP: r.logJoint, logQ: traceLogJoint(fixed), guideTrace }
}

/**
 * Sticking the landing (Roeder, Wu and Duvenaud 2017): the ELBO's pathwise gradient has a term, the score of the guide at
 * the draw, that has mean zero but does not vanish for a guide that equals the posterior; this estimator evaluates the
 * guide's log density with its parameters held constant, keeping only the path through the draw. It is unbiased, and its
 * variance goes to zero as the guide approaches the posterior (for a guide family that contains it), which a plain
 * estimator's does not.
 *
 * @param options - Particles averaged per estimate (default 1).
 * @returns The estimator (stateless).
 * @throws OmaiError with code `value` when a guide site has no reparameterised sampler.
 * @concept variational-inference
 * @index estimator Sticking-the-landing ELBO: pathwise gradient without the zero-mean score of the guide.
 * @example
 * ```ts
 * import { stlElbo } from 'omai-engine/probprog'
 * stlElbo({ numParticles: 2 }).name // => 'stlElbo'
 * ```
 */
export function stlElbo(options: { readonly numParticles?: number } = {}): ElboEstimator<null> {
  const n = options.numParticles ?? 1
  if (!(Number.isInteger(n) && n >= 1)) throw new OmaiError('value', 'stlElbo: numParticles must be positive')
  return {
    name: 'stlElbo',
    init: () => null,
    estimate: (key, model, guide, data, params): ElboEstimate<null> => {
      let total: Value = tensor(0)
      for (let i = 0; i < n; i++) {
        const p = stoppedParticle(foldIn(key, i), model, guide, data, params, 'stlElbo')
        total = add(total, mul(sub(p.logP, p.logQ), 1 / n))
      }
      return { loss: neg(total), elbo: num(total), state: null }
    },
  }
}

/** Options of {@link dregElbo}. */
export interface DregElboOptions {
  /** Particles K (default 5; one particle is {@link stlElbo}). */
  readonly numParticles?: number
}

/**
 * The importance-weighted bound with the doubly reparameterised gradient (Tucker et al. 2018): the standard IWAE
 * gradient weights each particle's pathwise gradient by its normalised weight and adds a score term whose signal-to-noise
 * ratio falls as the number of particles grows; evaluating the guide's log density at stopped parameters and weighting by
 * the squared normalised weights removes that term. The value reported is the bound, log (1/K sum w_k). Weights and bound
 * are the same as {@link importanceWeightedElbo}'s; for guides whose parameters are all in the guide (a model with its own
 * learnable `param` sites needs the plain estimator's weights).
 *
 * @param options - Particles.
 * @returns The estimator (stateless); its diagnostics carry the effective sample size of the weights.
 * @throws OmaiError with code `value` when a guide site has no reparameterised sampler.
 * @concept variational-inference
 * @index estimator Importance-weighted ELBO with the doubly reparameterised (DReG) gradient.
 * @example
 * ```ts
 * import { dregElbo } from 'omai-engine/probprog'
 * dregElbo({ numParticles: 8 }).name // => 'dregElbo'
 * ```
 */
export function dregElbo(options: DregElboOptions = {}): ElboEstimator<null> {
  const k = options.numParticles ?? 5
  if (!(Number.isInteger(k) && k >= 1)) throw new OmaiError('value', 'dregElbo: numParticles must be positive')
  return {
    name: 'dregElbo',
    init: () => null,
    estimate: (key, model, guide, data, params): ElboEstimate<null> => {
      const logw = Array.from({ length: k }, (_, i) => {
        const p = stoppedParticle(foldIn(key, i), model, guide, data, params, 'dregElbo')
        return sub(p.logP, p.logQ)
      })
      const w = stack(logw)
      const bound = sub(logsumexp(stopGradient(w)), Math.log(k))
      const normalised = exp(sub(stopGradient(w), logsumexp(stopGradient(w))))
      const weights = toFlat(primalOf(normalised))
      // The surrogate: the squared normalised weights times the log weights; its value is not the bound's, its gradient is.
      const surrogate = sum(mul(mul(normalised, normalised), w))
      const ess = 1 / weights.reduce((a, x) => a + x * x, 0)
      return {
        loss: neg(surrogate),
        elbo: num(bound),
        state: null,
        diagnostics: { ess, maxWeight: Math.max(...weights) },
      }
    },
  }
}
