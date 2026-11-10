/**
 * Reparameterisers, NumPyro's `reparam` handler and the set of `numpyro.infer.reparam` that matter: a latent site is
 * replaced by an auxiliary site with an easier geometry plus a deterministic transform back, so the program has the
 * same distribution but a sampler sees a different density. `locScale` (non-centring, the cure for Neal's funnel and
 * hierarchical models), `transform` (sample the base of a transformed distribution) and `circular` (an angle sampled on
 * the real line with its von Mises density as a factor).
 *
 * @index function reparam: replace a site by an auxiliary site plus a transform, to change the geometry samplers see.
 */

import { OmaiError } from '../base/index.ts'
import type { transformed } from '../dists/compose/index.ts'
import {
  cauchy,
  type Distribution,
  defineDistribution,
  gumbel,
  laplace,
  logistic,
  normal,
  studentT,
} from '../dists/index.ts'
import {
  add,
  asTensor,
  floor,
  mul,
  pow,
  primalOf,
  sub,
  type TensorLike,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { expandDistribution, unexpanded } from './_expand.ts'
import { handle } from './handlers.ts'
import { factor, type Message, type Model, sample } from './primitives.ts'

/** The site a reparameteriser is asked about: its name, its distribution and its observed value, if any. */
export interface ReparamSite {
  readonly name: string
  /** The distribution, already broadcast to the plates the site sits in. */
  readonly fn: Distribution
  /** The observed value, or undefined for a latent site. */
  readonly value: Value | undefined
}

/**
 * What a reparameteriser returns, as in NumPyro: a new distribution and value for the site (a `fn` of undefined turns
 * the site into a deterministic site holding `value`); `{ fn: site.fn, value: site.value }` leaves it unchanged.
 */
export interface ReparamResult {
  readonly fn: Distribution | undefined
  readonly value: Value | undefined
}

/**
 * A reparameteriser: called at a sample site, it may draw auxiliary sites (with `sample`, `factor`, which run through
 * the whole handler stack) and says what the original site becomes.
 *
 * @concept effect-handlers
 */
export interface Reparameteriser {
  readonly name: string
  apply(site: ReparamSite): ReparamResult
}

/** Reparameterisers by site name, or a function of the message returning one (or undefined). */
export type ReparamConfig = Readonly<Record<string, Reparameteriser>> | ((msg: Message) => Reparameteriser | undefined)

/**
 * Reparameterise sample sites (NumPyro's `reparam`): each affected site is replaced by one or more auxiliary sample
 * sites followed by a deterministic transformation back, so the program's distribution is unchanged but the density a
 * sampler works on has another geometry. The original name becomes a deterministic site (so posteriors still report
 * it), the auxiliary sites are named by the reparameteriser (`theta_decentered`, `theta_base`, ...).
 *
 * Condition or substitute the auxiliary names, not the original, once a site is reparameterised.
 *
 * @param model - The program.
 * @param config - Reparameterisers by site name, or a function of the message.
 * @returns The reparameterised program.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { locScale, logJoint, reparam, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * const model = () => sample('theta', normal(2, 3))
 * const decentred = reparam(model, { theta: locScale(0) })
 * // Same law: theta = 2 + 3 z with z ~ N(0, 1), so the density of z = 1 is that of theta = 5 times the scale 3.
 * const z = item(logJoint(decentred, undefined, { theta_decentered: 1 }).logJoint)
 * const t = item(logJoint(model, undefined, { theta: 5 }).logJoint)
 * Math.abs(z - (t + Math.log(3))) < 1e-14 // => true
 * ```
 */
export function reparam<D = void>(model: Model<D>, config: ReparamConfig): Model<D> {
  return handle(model, () => ({
    process: (msg) => {
      if (msg.type !== 'sample' || msg.isFactor || msg.fn === undefined) return
      const r = typeof config === 'function' ? config(msg) : config[msg.name]
      if (r === undefined) return
      const { fn, value } = r.apply({ name: msg.name, fn: msg.fn, value: msg.isObserved ? msg.value : undefined })
      if (value !== undefined && fn === undefined) {
        msg.type = 'deterministic'
        msg.value = value
        msg.fn = undefined
        msg.isObserved = false
        return
      }
      if (value !== undefined) {
        if (msg.value === undefined) msg.isObserved = true
        msg.value = value
      }
      msg.fn = fn
    },
  }))
}

interface LocScaleFamily {
  /** Parameters copied unchanged from the centred to the decentred distribution. */
  readonly shape: readonly string[]
  make(loc: Value, scale: Value, shape: Readonly<Record<string, Value>>): Distribution
}

const LOC_SCALE_FAMILIES: Readonly<Record<string, LocScaleFamily>> = {
  normal: { shape: [], make: (loc, scale) => normal(loc, scale) },
  laplace: { shape: [], make: (loc, scale) => laplace(loc, scale) },
  cauchy: { shape: [], make: (loc, scale) => cauchy(loc, scale) },
  logistic: { shape: [], make: (loc, scale) => logistic(loc, scale) },
  gumbel: { shape: [], make: (loc, scale) => gumbel(loc, scale) },
  studentT: {
    shape: ['df'],
    make: (loc, scale, shape) => studentT(shape['df'] as Value, loc, scale),
  },
}

/**
 * The decentring reparameteriser of latent sites with a location and a scale (NumPyro's `LocScaleReparam`, Gorinova,
 * Moore and Hoffman 2019): with centring `c`, a draw `x ~ F(loc, scale)` is replaced by `z ~ F(c loc, scale^c)` and
 * `x = loc + scale^(1 - c) (z - c loc)`. `c = 0` is the fully non-centred form `x = loc + scale z` with `z ~ F(0, 1)`
 * (the cure for the funnel of hierarchical models), `c = 1` leaves the site alone. Works for the normal, Laplace,
 * Cauchy, logistic, Gumbel and Student t families with scalar events; not for observed sites.
 *
 * @param centered - The centring in [0, 1] (default 0), a number or a value broadcasting against the site.
 * @returns The reparameteriser, for {@link reparam}.
 * @throws OmaiError with code `value` (when applied) for an observed site, a family without a location and a scale,
 *   or a centring outside [0, 1].
 * @concept effect-handlers
 * @example
 * ```ts
 * import { locScale, reparam, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => sample('theta', normal(2, 3))
 * const t = traceOf(seed(reparam(model, { theta: locScale(0) }), key(0)), undefined)
 * [...t.sites.keys()] // => ['theta_decentered', 'theta']
 * ```
 */
export function locScale(centered: TensorLike = 0): Reparameteriser {
  const c = asTensor(centered)
  const flat = c.shape.length === 0 ? toFlat(primalOf(c))[0] : undefined
  if (flat !== undefined && !(flat >= 0 && flat <= 1))
    throw new OmaiError('value', `locScale: the centring must lie in [0, 1], got ${flat}`)
  return {
    name: 'locScale',
    apply: ({ name, fn, value }) => {
      if (value !== undefined)
        throw new OmaiError('value', `locScale: site ${name} is observed; it needs a latent site`)
      if (flat === 1) return { fn, value }
      const base = unexpanded(fn)
      const family = LOC_SCALE_FAMILIES[base.name]
      const loc = base.params['loc']
      const scale = base.params['scale']
      if (family === undefined || loc === undefined || scale === undefined || fn.eventShape.length > 0)
        throw new OmaiError('value', `locScale: site ${name} is a ${base.name}, which has no scalar location and scale`)
      const shape: Record<string, Value> = {}
      for (const p of family.shape) shape[p] = base.params[p] as Value
      const decentredFn = expandDistribution(family.make(mul(loc, c), pow(scale, c), shape), fn.batchShape)
      const z = sample(`${name}_decentered`, decentredFn)
      const delta = sub(z, mul(c, loc))
      return { fn: undefined, value: add(loc, mul(pow(scale, sub(1, c)), delta)) }
    },
  }
}

/**
 * The reparameteriser of transformed distributions (NumPyro's `TransformReparam`): a site distributed as
 * `transformed(base, bijector)` is replaced by a draw of the base and the bijector applied to it, which is useful
 * when the transform changes the geometry (a log-normal, a squashed or an affine map) and the posterior is simple in
 * the base's space. Not for observed sites.
 *
 * @returns The reparameteriser, for {@link reparam}.
 * @throws OmaiError with code `value` (when applied) for an observed site or a distribution that is not a
 *   `transformed` one.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { reparam, sample, seed, traceOf, transformReparam } from 'omai-engine/probprog'
 * import { bijectors, normal } from 'omai-engine/dists'
 * import { transformed } from 'omai-engine/dists/compose'
 * import { key } from 'omai-engine/random'
 * const model = () => sample('s', transformed(normal(0, 1), bijectors.exp()))
 * const t = traceOf(seed(reparam(model, { s: transformReparam() }), key(0)), undefined)
 * [...t.sites.keys()] // => ['s_base', 's']
 * ```
 */
export function transformReparam(): Reparameteriser {
  return {
    name: 'transform',
    apply: ({ name, fn, value }) => {
      if (value !== undefined)
        throw new OmaiError('value', `transformReparam: site ${name} is observed; it needs a latent site`)
      const t = unexpanded(fn) as Partial<ReturnType<typeof transformed>>
      if (t.base === undefined || t.bijector === undefined)
        throw new OmaiError('value', `transformReparam: site ${name} is not a transformed distribution`)
      const x = sample(`${name}_base`, expandDistribution(t.base, fn.batchShape))
      return { fn: undefined, value: t.bijector.forward(x) }
    },
  }
}

function improperReal(batchShape: readonly number[]): Distribution {
  return defineDistribution({
    name: 'improperUniform',
    params: {},
    support: { kind: 'real' },
    batchShape: [...batchShape],
    logProb: (x) => mul(x, 0),
    // Only used to start a sampler: draws are standard normal, the density itself is flat.
    sample: (key, shape) => normal(0, 1).sample(key, [...shape, ...batchShape]),
    mean: () => zeros([...batchShape]),
    variance: () => zeros([...batchShape]),
    mode: () => zeros([...batchShape]),
    entropy: () => zeros([...batchShape]),
  })
}

/**
 * The reparameteriser of von Mises angles (NumPyro's `CircularReparam`): the angle is sampled on the whole real line
 * under a flat prior (`name_unwrapped`), wrapped into [-pi, pi), and its von Mises density added as a factor
 * (`name_factor`), so a sampler never meets the seam of the circle. Works for observed sites too.
 *
 * @returns The reparameteriser, for {@link reparam}.
 * @throws OmaiError with code `value` (when applied) for a distribution that is not a von Mises.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { circular, logJoint, reparam, sample } from 'omai-engine/probprog'
 * import { vonMises } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * const model = () => sample('phi', vonMises(0, 2))
 * // An angle one full turn from 0.5 has the density of 0.5.
 * const a = item(logJoint(reparam(model, { phi: circular() }), undefined, { phi_unwrapped: 0.5 + 2 * Math.PI }).logJoint)
 * Math.abs(a - item(vonMises(0, 2).logProb(0.5))) < 1e-12 // => true
 * ```
 */
export function circular(): Reparameteriser {
  return {
    name: 'circular',
    apply: ({ name, fn, value }) => {
      const base = unexpanded(fn)
      if (base.name !== 'vonMises')
        throw new OmaiError('value', `circular: site ${name} is a ${base.name}, not a von Mises distribution`)
      const flat = improperReal(fn.batchShape)
      const raw =
        value === undefined ? sample(`${name}_unwrapped`, flat) : sample(`${name}_unwrapped`, flat, { obs: value })
      const turns = floor(mul(add(raw, Math.PI), 1 / (2 * Math.PI)))
      const wrapped = sub(raw, mul(turns, 2 * Math.PI))
      factor(`${name}_factor`, fn.logProb(wrapped))
      return { fn: undefined, value: wrapped }
    },
  }
}
