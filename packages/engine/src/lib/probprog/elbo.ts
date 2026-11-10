/**
 * Variational objectives, which score how well a guide (an approximate posterior) fits a model: the standard ELBO
 * estimator with reparameterised and score-function gradients (NumPyro's `Trace_ELBO`), the mean-field ELBO with
 * closed-form KL terms, the Renyi divergence bound and the importance-weighted (IWAE) bound.
 */

import { OmaiError } from '../base/index.ts'
import { klDivergence } from '../dists/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import {
  add,
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
  type Value,
} from '../tensor/index.ts'
import { maskedLogProb } from './_contract.ts'
import { logJoint, traceLogJoint } from './density.ts'
import { enumerate, type ProgramTrace, replay, seed, substitute, traceOf } from './handlers.ts'
import type { Model } from './primitives.ts'

/** One estimate of an ELBO: a differentiable surrogate loss, the ELBO's value, and the estimator's next state. */
export interface ElboEstimate<S> {
  /** Minimise this: its gradient is an unbiased estimate of minus the ELBO's gradient. */
  readonly loss: Value
  /** The ELBO estimate (or the bound the estimator targets). */
  readonly elbo: number
  readonly state: S
  /** Named numbers about the estimate (baselines, enumerated cells, weights' effective size) for a method's tape. */
  readonly diagnostics?: Readonly<Record<string, number>>
}

/**
 * An ELBO estimator (NumPyro's `Trace_ELBO` family): from a key, the model, the guide, the data and the current
 * parameter values, a surrogate loss whose gradient estimates minus the gradient of the bound; `state` carries
 * running quantities such as score-function baselines.
 *
 * @concept variational-inference
 */
export interface ElboEstimator<S = unknown> {
  readonly name: string
  init(): S
  estimate<D>(
    key: Key,
    model: Model<D>,
    guide: Model<D>,
    data: D,
    params: Readonly<Record<string, TensorLike>>,
    state: S,
  ): ElboEstimate<S>
}

const num = (v: Value): number => item(primalOf(v))

/** The traces and log densities of one particle: the guide's draw and the model replayed on it. */
export interface Particle {
  readonly guideTrace: ProgramTrace
  readonly modelTrace: ProgramTrace
  /** log p(z, x) with enumerated sites summed out. */
  readonly logP: Value
  /** log q(z). */
  readonly logQ: Value
}

/**
 * Run the guide (seeded, with parameters substituted) and the model replayed on its draws, with finite discrete model
 * sites left to enumeration (NumPyro's `get_importance_trace`).
 *
 * @param key - The key of the guide's draws.
 * @param model - The model.
 * @param guide - The guide.
 * @param data - The data.
 * @param params - Parameter values (of both programs) by name.
 * @returns The particle.
 * @concept variational-inference
 * @example
 * ```ts
 * import { autoNormal, observe, particle, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 0) }
 * particle(key(0), model, autoNormal(model), undefined, { m_auto_loc: 0, m_auto_scale: 1 }).guideTrace.sites.has('m') // => true
 * ```
 */
export function particle<D = void>(
  key: Key,
  model: Model<D>,
  guide: Model<D>,
  data: D,
  params: Readonly<Record<string, TensorLike>>,
): Particle {
  const guideTrace = traceOf(seed(substitute(guide, params), key), data)
  // The model is seeded too: plates the guide does not have draw their own subsamples.
  const r = logJoint(seed(replay(substitute(enumerate(model), params), guideTrace), foldIn(key, 1 << 30)), data)
  return { guideTrace, modelTrace: r.trace, logP: r.logJoint, logQ: traceLogJoint(guideTrace) }
}

/**
 * The ELBO terms at given guide draws (fixed reparameterised noise): substitute the guide's latent values, then
 * log p - log q. This is what each estimator averages, and what golden tests compare with NumPyro.
 *
 * @param model - The model.
 * @param guide - The guide.
 * @param data - The data.
 * @param params - Parameter values by name.
 * @param values - The guide's latent values by site.
 * @returns log p, log q and their difference.
 * @concept variational-inference
 * @example
 * ```ts
 * import { autoNormal, elboAt, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 0) }
 * // q = N(0, 1) = the prior: at m = 0 the ELBO term is log N(0; 0, 1).
 * Math.abs(elboAt(model, autoNormal(model), undefined, { m_auto_loc: 0, m_auto_scale: 1 }, { m: 0 }).elbo + 0.5 * Math.log(2 * Math.PI)) < 1e-14 // => true
 * ```
 */
export function elboAt<D = void>(
  model: Model<D>,
  guide: Model<D>,
  data: D,
  params: Readonly<Record<string, TensorLike>>,
  values: Readonly<Record<string, TensorLike>>,
): { readonly logP: number; readonly logQ: number; readonly elbo: number } {
  const guideTrace = traceOf(substitute(guide, { ...params, ...values }), data)
  const logP = item(logJoint(replay(substitute(enumerate(model), params), guideTrace), data).logJoint)
  const logQ = item(traceLogJoint(guideTrace))
  return { logP, logQ, elbo: logP - logQ }
}

/** Options of {@link traceElbo}. */
export interface TraceElboOptions {
  /** Particles averaged per estimate (default 1). */
  readonly numParticles?: number
  /** Decay of the moving-average baseline for score-function terms (default 0.9; 0 turns baselines off). */
  readonly baselineDecay?: number
}

/**
 * The score-function terms of a particle: log q(z) times the learning signal for every guide site without a
 * reparameterised sampler (what {@link traceElbo} adds to its pathwise surrogate), minus the sites in `skip`.
 *
 * @param p - The particle.
 * @param signal - The (baselined) signal that multiplies each log density.
 * @param skip - Sites left out (enumerated exactly, so they need no score term).
 * @returns The surrogate term.
 */
export function scoreTerms(p: Particle, signal: number, skip: ReadonlySet<string> = new Set()): Value {
  let total: Value = tensor(0)
  for (const site of p.guideTrace.sites.values()) {
    if (site.type !== 'sample' || site.isObserved || site.fn.hasRsample || skip.has(site.name)) continue
    const lq = mul(sum(maskedLogProb(site)), site.scale)
    total = add(total, mul(lq, signal))
  }
  return total
}

/**
 * The standard ELBO estimator (NumPyro's `Trace_ELBO`): E_q[log p(z, x) - log q(z)] by pathwise (reparameterised)
 * gradients, plus score-function terms log q(z) (signal - baseline) for guide sites without a reparameterised sampler,
 * with a moving-average baseline to reduce their variance.
 *
 * @param options - Particles and baseline decay.
 * @returns The estimator; its state is the baseline.
 * @concept variational-inference
 * @example
 * ```ts
 * import { traceElbo } from 'omai-engine/probprog'
 * traceElbo({ numParticles: 2 }).name // => 'traceElbo'
 * ```
 */
export function traceElbo(options: TraceElboOptions = {}): ElboEstimator<{ readonly baseline: number | undefined }> {
  const n = options.numParticles ?? 1
  const decay = options.baselineDecay ?? 0.9
  return {
    name: 'traceElbo',
    init: () => ({ baseline: undefined }),
    estimate: (key, model, guide, data, params, state) => {
      let loss: Value = tensor(0)
      let elbo = 0
      let baseline = state.baseline
      for (let i = 0; i < n; i++) {
        const p = particle(foldIn(key, i), model, guide, data, params)
        const value = sub(p.logP, p.logQ)
        const v = num(value)
        elbo += v / n
        const signal = decay > 0 && baseline !== undefined ? v - baseline : v
        loss = add(loss, mul(neg(add(value, scoreTerms(p, signal))), 1 / n))
      }
      if (decay > 0) baseline = baseline === undefined ? elbo : decay * baseline + (1 - decay) * elbo
      return { loss, elbo, state: { baseline } }
    },
  }
}

/**
 * The mean-field ELBO (NumPyro's `TraceMeanField_ELBO`): like {@link traceElbo}, but each latent site contributes
 * -KL(q || p) in closed form where the pair has one (normals, matching families), and log p - log q at the draw
 * otherwise; lower variance for mean-field guides.
 *
 * @param options - Particles.
 * @returns The estimator.
 * @concept variational-inference
 * @example
 * ```ts
 * import { traceMeanFieldElbo } from 'omai-engine/probprog'
 * traceMeanFieldElbo().name // => 'traceMeanFieldElbo'
 * ```
 */
export function traceMeanFieldElbo(options: { readonly numParticles?: number } = {}): ElboEstimator<null> {
  const n = options.numParticles ?? 1
  return {
    name: 'traceMeanFieldElbo',
    init: () => null,
    estimate: (key, model, guide, data, params) => {
      let total: Value = tensor(0)
      for (let i = 0; i < n; i++) {
        const guideTrace = traceOf(seed(substitute(guide, params), foldIn(key, i)), data)
        const modelTrace = traceOf(
          seed(replay(substitute(model, params), guideTrace), foldIn(foldIn(key, i), 1 << 30)),
          data,
        )
        let term: Value = tensor(0)
        for (const site of modelTrace.sites.values()) {
          if (site.type !== 'sample') continue
          const q = guideTrace.sites.get(site.name)
          if (site.isObserved || q?.type !== 'sample') {
            term = add(term, mul(sum(maskedLogProb(site)), site.scale))
            continue
          }
          let kl: Value
          try {
            kl = sum(klDivergence(q.fn, site.fn))
          } catch {
            kl = sub(sum(maskedLogProb(q)), sum(maskedLogProb(site)))
          }
          term = sub(term, mul(kl, site.scale))
        }
        for (const q of guideTrace.sites.values()) {
          if (q.type !== 'sample' || modelTrace.sites.has(q.name)) continue
          term = sub(term, mul(sum(maskedLogProb(q)), q.scale))
        }
        total = add(total, mul(term, 1 / n))
      }
      return { loss: neg(total), elbo: num(total), state: null }
    },
  }
}

/** Options of {@link renyiElbo}. */
export interface RenyiElboOptions {
  /** The order alpha (default 0, the importance-weighted bound; alpha = 1 is the ELBO and is excluded). */
  readonly alpha?: number
  /** Particles K (default 2). */
  readonly numParticles?: number
}

/**
 * The Renyi divergence bound (Li and Turner 2016; NumPyro's `RenyiELBO`): with log weights w_k = log p - log q of K
 * particles, 1 / (1 - alpha) (logsumexp((1 - alpha) w) - log K), which is the importance-weighted bound at alpha = 0
 * and tends to the ELBO as alpha tends to 1.
 *
 * @param options - Alpha and the number of particles.
 * @returns The estimator.
 * @throws OmaiError with code `value` for alpha = 1.
 * @concept variational-inference
 * @example
 * ```ts
 * import { renyiElbo } from 'omai-engine/probprog'
 * renyiElbo({ alpha: 0.5 }).name // => 'renyiElbo'
 * ```
 */
export function renyiElbo(options: RenyiElboOptions = {}): ElboEstimator<null> {
  const alpha = options.alpha ?? 0
  const k = options.numParticles ?? 2
  if (alpha === 1) throw new OmaiError('value', 'renyiElbo: alpha = 1 is the ELBO; use traceElbo')
  if (!(Number.isInteger(k) && k >= 1)) throw new OmaiError('value', 'renyiElbo: numParticles must be positive')
  return {
    name: 'renyiElbo',
    init: () => null,
    estimate: (key, model, guide, data, params) => {
      const w = stack(
        Array.from({ length: k }, (_, i) => {
          const p = particle(foldIn(key, i), model, guide, data, params)
          return sub(p.logP, p.logQ)
        }),
      )
      const bound = mul(sub(logsumexp(mul(w, 1 - alpha)), Math.log(k)), 1 / (1 - alpha))
      return { loss: neg(bound), elbo: num(bound), state: null }
    },
  }
}

/**
 * The importance-weighted bound (Burda, Grosse and Salakhutdinov 2016): logsumexp(w) - log K over K particles, a
 * tighter bound than the ELBO that tends to log p(x) as K grows.
 *
 * @param options - Particles K (default 5).
 * @returns The estimator.
 * @concept variational-inference
 * @example
 * ```ts
 * import { importanceWeightedElbo } from 'omai-engine/probprog'
 * importanceWeightedElbo({ numParticles: 8 }).name // => 'importanceWeightedElbo'
 * ```
 */
export function importanceWeightedElbo(options: { readonly numParticles?: number } = {}): ElboEstimator<null> {
  const inner = renyiElbo({ alpha: 0, numParticles: options.numParticles ?? 5 })
  return { ...inner, name: 'importanceWeightedElbo' }
}

/**
 * The bound of an estimator at fixed log weights, as Renyi and importance-weighted estimators compute it (for checks
 * against other libraries).
 *
 * @param logWeights - log p - log q per particle.
 * @param alpha - The Renyi order (0 for the importance-weighted bound).
 * @returns The bound.
 * @example
 * ```ts
 * import { renyiBound } from 'omai-engine/probprog'
 * renyiBound([0, 0], 0) // => 0
 * ```
 */
export function renyiBound(logWeights: readonly number[], alpha: number): number {
  const v = stopGradient(tensor(logWeights.map((x) => x * (1 - alpha))))
  return (item(logsumexp(v)) - Math.log(logWeights.length)) / (1 - alpha)
}
