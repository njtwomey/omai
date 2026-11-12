/**
 * Automatic guides, the approximate posteriors fitted by variational inference, NumPyro style: mean-field normal, point
 * mass (MAP), full-rank and low-rank multivariate normal, Laplace and normalising-flow (inverse autoregressive flow)
 * guides, each built from the model's continuous latent sites on their unconstrained scale.
 */

import { hessian } from '../autodiff/index.ts'
import { OmaiError } from '../base/index.ts'
import {
  bijectors,
  compose,
  type Distribution,
  defineDistribution,
  lowRankMultivariateNormal,
  multivariateNormal,
  normal,
  type Support,
  supportRank,
} from '../dists/index.ts'
import { inv } from '../numerics/linalg/index.ts'
import { foldIn, type Key, key as makeKey, normal as normalDraws, uniform } from '../random/index.ts'
import {
  add,
  arange,
  asTensor,
  broadcastTo,
  concat,
  div,
  eye,
  full,
  log,
  matmul,
  mul,
  neg,
  primalOf,
  reshape,
  sliceAxis,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tanh,
  toFlat,
  transpose,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { isolated } from './_stack.ts'
import { type InitStrategy, potentialEnergy } from './density.ts'
import { enumerate, isDiscrete, type SampleSite, seed, traceOf } from './handlers.ts'
import { definePosterior, type Posterior } from './posterior.ts'
import { type Model, type PlateFrame, param, plate, sample } from './primitives.ts'

/** One continuous latent site a guide covers, at full plate size. */
export interface GuideSite {
  readonly name: string
  /** The constrained shape (plates at full size). */
  readonly shape: readonly number[]
  readonly unconstrainedShape: readonly number[]
  readonly support: Support
  readonly bijector: bijectors.Bijector
  /** Event dimensions of the constrained and unconstrained values (beyond the plate batch axes). */
  readonly eventDim: number
  readonly unconstrainedEventDim: number
  /** The site's plates, outermost first. */
  readonly plates: readonly PlateFrame[]
  /** Offset and size in the flat unconstrained vector (for joint guides). */
  readonly offset: number
  readonly size: number
  /** Whether any of its plates is subsampled. */
  readonly subsampled: boolean
  /** The prototype site. */
  readonly prototype: SampleSite
}

/**
 * An automatic guide (NumPyro's `AutoGuide`): a program over the model's continuous latent sites with learnable
 * parameters (`param` sites), built from a prototype run of the model. Discrete sites with finite support are left to
 * enumeration in the model. `posterior` turns fitted parameters into a {@link Posterior}.
 *
 * @concept variational-inference
 */
export interface AutoGuide<D> {
  (data: D): unknown
  readonly kind: string
  readonly model: Model<D>
  /** The sites the guide covers, from the prototype run on `data`. */
  sites(data: D): readonly GuideSite[]
  /** The variational posterior at fitted parameters. */
  posterior(params: Readonly<Record<string, TensorLike>>, data: D): Posterior
  /** The guide's central point (constrained) at fitted parameters. */
  median(params: Readonly<Record<string, TensorLike>>, data: D): Record<string, Value>
}

/** Options shared by the automatic guides. */
export interface AutoGuideOptions {
  /** How location parameters start (default uniform in (-2, 2) on the unconstrained scale). */
  readonly init?: InitStrategy
  /** The initial scale of the normal families (default 0.1, as NumPyro). */
  readonly initScale?: number
}

const product = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1)

/**
 * The continuous latent sites of a model as a guide sees them (plates at full size, in execution order).
 *
 * @param model - The model.
 * @param data - Its data.
 * @returns The guide sites.
 * @throws OmaiError with code `value` for discrete latent sites with unbounded support.
 * @example
 * ```ts
 * import { guideSites, plate, sample } from 'omai-engine/probprog'
 * import { gamma } from 'omai-engine/dists'
 * guideSites(() => plate('n', 5, () => sample('s', gamma(2, 2)), { subsampleSize: 2 }), undefined)[0]?.shape // => [5]
 * ```
 */
export function guideSites<D = void>(model: Model<D>, data: D): GuideSite[] {
  const proto = isolated(() => traceOf(seed(enumerate(model), makeKey(0)), data))
  const out: GuideSite[] = []
  let offset = 0
  for (const site of proto.sites.values()) {
    if (site.type !== 'sample' || site.isObserved) continue
    if (isDiscrete(site.fn.support)) {
      if (site.infer.enumerate === 'parallel') continue
      throw new OmaiError('value', `guide: discrete site ${site.name} cannot be summed out`, { site: site.name })
    }
    // A support over the last axis (ordered, simplex) of a scalar-event distribution under a plate makes that plate axis part
    // of the event (the elements are not independent): the guide draws the whole vector, outside that plate.
    const coupled = Math.max(supportRank(site.fn.support) - site.fn.eventShape.length, 0)
    const batchRank = site.fn.batchShape.length - coupled
    const shape = [...site.value.shape]
    for (const f of site.plates) {
      const axis = shape.length - site.fn.eventShape.length + f.dim
      if (axis >= 0 && f.subsampleSize !== f.size) shape[axis] = f.size
    }
    const bijector = bijectors.bijectorForSupport(site.fn.support)
    const unconstrainedShape = bijector.inverseShape(shape)
    const size = product(unconstrainedShape)
    out.push({
      name: site.name,
      shape,
      unconstrainedShape,
      support: site.fn.support,
      bijector,
      eventDim: shape.length - batchRank,
      unconstrainedEventDim: unconstrainedShape.length - batchRank,
      plates: site.plates.filter((f) => f.dim < -coupled).reverse(),
      offset,
      size,
      subsampled: site.plates.some((f) => f.subsampleSize !== f.size),
      prototype: site,
    })
    offset += size
  }
  return out
}

/**
 * The guide sites of a model, found once per data value: what a guide calls on every run instead of tracing the model
 * again.
 *
 * @param model - The model.
 * @returns A function from the data to the guide sites (the last result is kept while the data is the same object).
 * @example
 * ```ts
 * import { cachedGuideSites, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * cachedGuideSites(() => { sample('x', normal(0, 1)) })(undefined).length // => 1
 * ```
 */
export function cachedGuideSites<D = void>(model: Model<D>): (data: D) => GuideSite[] {
  let last: { readonly data: D; readonly sites: GuideSite[] } | undefined
  return (data) => {
    if (last !== undefined && last.data === data) return last.sites
    const sites = guideSites(model, data)
    last = { data, sites }
    return sites
  }
}

/**
 * Run a function inside nested plates, outermost first, as a guide site's plates are (so subsampled plates subsample the
 * guide's local parameters).
 *
 * @param frames - The plates, outermost first (a {@link GuideSite}'s `plates`).
 * @param fn - What runs inside them.
 * @returns What `fn` returns.
 * @example
 * ```ts
 * import { guideSites, plate, sample, withPlates } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * const site = guideSites(() => plate('n', 3, () => sample('x', normal(0, 1))), undefined)[0]
 * withPlates(site?.plates ?? [], () => 1) // => 1
 * ```
 */
export function withPlates<T>(frames: readonly PlateFrame[], fn: () => T): T {
  const [first, ...rest] = frames
  if (first === undefined) return fn()
  return plate(first.name, first.size, () => withPlates(rest, fn), {
    subsampleSize: first.subsampleSize,
    dim: first.dim,
  })
}

/** A unconstrained starting value of a site at full size. */
function initialUnconstrained(site: GuideSite, key: Key, options: AutoGuideOptions): Value {
  if (options.init !== undefined && !site.subsampled) {
    const v = options.init(site.prototype, key)
    if (v !== undefined) return primalOf(site.bijector.inverse(broadcastTo(v, site.shape)))
  }
  return uniform(key, site.unconstrainedShape, { low: -2, high: 2 })
}

/**
 * A guide site's distribution pushed through its bijector, remembering each draw's unconstrained value so its log
 * density needs no inverse (NumPyro's intermediates).
 */
export function pushed(base: Distribution, bijector: bijectors.Bijector): Distribution {
  const t = compose.transformed(base, bijector)
  let lastY: Value | undefined
  let lastU: Value | undefined
  const draw = (u: Value): Value => {
    const y = bijector.forward(u)
    lastY = y
    lastU = u
    return y
  }
  const ldj = (u: Value): Value => {
    const j = bijector.forwardLogDetJacobian(u)
    const extra = j.shape.length - (u.shape.length - base.eventShape.length)
    return extra > 0 ? sum(j, toFlat(arange(j.shape.length - 1, j.shape.length - 1 - extra, -1))) : j
  }
  return defineDistribution({
    name: 'guide',
    params: base.params,
    support: bijector.codomain,
    batchShape: t.batchShape,
    eventShape: t.eventShape,
    logProb: (x) => {
      const u = x === lastY && lastU !== undefined ? lastU : bijector.inverse(x)
      const lp = base.logProb(u)
      return add(lp, neg(reduceTo(ldj(u), lp.shape)))
    },
    rsample: (key, shape) => draw(base.rsample(key, shape)),
    sample: (key, shape) => draw(base.sample(key, shape)),
    mean: () => t.mean(),
    variance: () => t.variance(),
    mode: () => t.mode(),
    entropy: () => t.entropy(),
  })
}

/** Sum trailing axes of `v` until it has `shape`'s rank. */
function reduceTo(v: Value, shape: readonly number[]): Value {
  const extra = v.shape.length - shape.length
  return extra > 0 ? sum(v, toFlat(arange(v.shape.length - 1, v.shape.length - 1 - extra, -1))) : v
}

/** A point mass at a value with a given log density (NumPyro's `Delta(value, log_density)`). */
function delta(value: Value, logDensity: Value, eventRank: number): Distribution {
  const batchShape = value.shape.slice(0, value.shape.length - eventRank)
  return defineDistribution({
    name: 'delta',
    params: { value },
    support: { kind: 'real' },
    batchShape,
    eventShape: value.shape.slice(value.shape.length - eventRank),
    logProb: () => broadcastTo(logDensity, batchShape),
    rsample: (_key, shape) => broadcastTo(value, [...shape, ...value.shape]),
    mean: () => value,
    variance: () => mul(value, 0),
    mode: () => value,
    entropy: () => zeros(batchShape),
  })
}

function paramOf(params: Readonly<Record<string, TensorLike>>, name: string): Value {
  const v = params[name]
  if (v === undefined) throw new OmaiError('value', `guide: no parameter ${name}`)
  return primalOf(asTensor(v))
}

/** Draw `n` constrained values per site from unconstrained draws `[n, D]`. */
function constrainFlat(sites: readonly GuideSite[], u: Value, n: number): Record<string, Tensor> {
  const out: Record<string, Tensor> = {}
  for (const s of sites) {
    const piece = reshape(sliceAxis(u, 1, s.offset, s.offset + s.size), [n, ...s.unconstrainedShape])
    out[s.name] = primalOf(s.bijector.forward(piece))
  }
  return out
}

/**
 * One site of a mean-field normal guide, as a guide program draws it: an independent normal on the site's unconstrained
 * scale (`<site>_auto_loc`, `<site>_auto_scale`), pushed onto its support, inside the site's plates. What
 * {@link autoNormal} does for every site, for a guide that does it for some and something else for the rest.
 *
 * @param site - The guide site.
 * @param options - Initialisation and initial scale.
 * @example
 * ```ts
 * import { guideSites, meanFieldSite, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { sample('x', normal(0, 1)) }
 * const guide = () => { for (const s of guideSites(model, undefined)) meanFieldSite(s) }
 * [...traceOf(seed(guide, key(0)), undefined).sites.keys()] // => ['x_auto_loc', 'x_auto_scale', 'x']
 * ```
 */
export function meanFieldSite(site: GuideSite, options: AutoGuideOptions = {}): void {
  const initScale = options.initScale ?? 0.1
  withPlates(site.plates, () => {
    const loc = param(`${site.name}_auto_loc`, (k) => initialUnconstrained(site, k, options), {
      eventDim: site.unconstrainedEventDim,
    })
    const scale = param(`${site.name}_auto_scale`, full(site.unconstrainedShape, initScale), {
      support: { kind: 'positive' },
      eventDim: site.unconstrainedEventDim,
    })
    const n = normal(loc, scale)
    const base = site.unconstrainedEventDim === 0 ? n : compose.independent(n, site.unconstrainedEventDim)
    // On the real line the guide site is the normal itself (as NumPyro), so analytic KL terms apply.
    sample(site.name, site.support.kind === 'real' ? base : pushed(base, site.bijector))
  })
}

/**
 * The distribution of one site of a mean-field normal guide at fitted parameters: the normal of `<site>_auto_loc` and
 * `<site>_auto_scale`, pushed onto the site's support.
 *
 * @param site - The guide site.
 * @param params - The fitted parameters by name (constrained).
 * @returns The marginal.
 * @example
 * ```ts
 * import { guideSites, meanFieldMarginal, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * const site = guideSites(() => { sample('x', normal(0, 1)) }, undefined)[0]
 * site === undefined ? 0 : item(meanFieldMarginal(site, { x_auto_loc: 1, x_auto_scale: 2 }).mean()) // => 1
 * ```
 */
export function meanFieldMarginal(site: GuideSite, params: Readonly<Record<string, TensorLike>>): Distribution {
  const base = normal(paramOf(params, `${site.name}_auto_loc`), paramOf(params, `${site.name}_auto_scale`))
  const event = site.unconstrainedEventDim === 0 ? base : compose.independent(base, site.unconstrainedEventDim)
  return site.support.kind === 'real' ? event : compose.transformed(event, site.bijector)
}

/**
 * The mean-field normal guide (NumPyro's `AutoNormal`): each continuous latent site gets an independent normal on its
 * unconstrained scale (`<site>_auto_loc`, `<site>_auto_scale`), pushed onto its support, inside the site's plates so
 * subsampled plates subsample the local parameters.
 *
 * @param model - The model.
 * @param options - Initialisation and initial scale.
 * @returns The guide.
 * @concept variational-inference
 * @example
 * ```ts
 * import { autoNormal, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { gamma } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const guide = autoNormal(() => { sample('s', gamma(2, 2)) })
 * [...traceOf(seed(guide, key(0)), undefined).sites.keys()] // => ['s_auto_loc', 's_auto_scale', 's']
 * ```
 */
export function autoNormal<D = void>(model: Model<D>, options: AutoGuideOptions = {}): AutoGuide<D> {
  const sitesOf = cachedGuideSites(model)
  const initScale = options.initScale ?? 0.1
  const program = (data: D): unknown => {
    for (const s of sitesOf(data)) meanFieldSite(s, { ...options, initScale })
    return undefined
  }
  const marginals = (params: Readonly<Record<string, TensorLike>>, data: D): Record<string, Distribution> => {
    const out: Record<string, Distribution> = {}
    for (const s of sitesOf(data)) out[s.name] = meanFieldMarginal(s, params)
    return out
  }
  const median = (params: Readonly<Record<string, TensorLike>>, data: D): Record<string, Value> =>
    Object.fromEntries(
      sitesOf(data).map((s) => [s.name, primalOf(s.bijector.forward(paramOf(params, `${s.name}_auto_loc`)))]),
    )
  return Object.assign(program, {
    kind: 'autoNormal',
    model,
    sites: sitesOf,
    median,
    posterior: (params: Readonly<Record<string, TensorLike>>, data: D): Posterior => {
      const ms = marginals(params, data)
      const sites = sitesOf(data)
      return definePosterior({
        form: 'variational',
        engine: 'svi',
        sites: sites.map((s) => s.name),
        marginals: Object.fromEntries(
          sites.map((s) => {
            const d = ms[s.name] as Distribution
            // Scalar maps keep quantiles and moments; vector supports fall back to draws.
            return [s.name, d]
          }),
        ),
        sampler: (key: Key, n: number) => {
          const out: Record<string, Tensor> = {}
          sites.forEach((s, i) => {
            const loc = paramOf(params, `${s.name}_auto_loc`)
            const scale = paramOf(params, `${s.name}_auto_scale`)
            const u = add(loc, mul(scale, normalDraws(foldIn(key, i), [n, ...s.unconstrainedShape])))
            out[s.name] = primalOf(s.bijector.forward(u))
          })
          return out
        },
        logProb: (values) =>
          sites.reduce((acc, s) => {
            const v = values[s.name]
            if (v === undefined) throw new OmaiError('value', `guide posterior: no value for ${s.name}`)
            return acc + toFlat(sum((ms[s.name] as Distribution).logProb(v))).reduce((x, y) => x + y, 0)
          }, 0),
      })
    },
  })
}

/**
 * The point-mass guide (NumPyro's `AutoDelta`): each continuous latent site is a learnable constrained value
 * (`<site>_auto_loc`), so maximising the ELBO is MAP estimation.
 *
 * @param model - The model.
 * @param options - Initialisation.
 * @returns The guide.
 * @concept variational-inference
 * @example
 * ```ts
 * import { autoDelta, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * [...traceOf(seed(autoDelta(() => { sample('x', normal(0, 1)) }), key(0)), undefined).sites.keys()] // => ['x_auto_loc', 'x']
 * ```
 */
export function autoDelta<D = void>(model: Model<D>, options: AutoGuideOptions = {}): AutoGuide<D> {
  const sitesOf = cachedGuideSites(model)
  const program = (data: D): unknown => {
    for (const s of sitesOf(data)) {
      withPlates(s.plates, () => {
        const v = param(
          `${s.name}_auto_loc`,
          (k) => primalOf(s.bijector.forward(initialUnconstrained(s, k, options))),
          {
            support: s.support,
            eventDim: s.eventDim,
          },
        )
        sample(s.name, delta(v, zeros(v.shape.slice(0, v.shape.length - s.eventDim)), s.eventDim))
      })
    }
    return undefined
  }
  const median = (params: Readonly<Record<string, TensorLike>>, data: D): Record<string, Value> =>
    Object.fromEntries(sitesOf(data).map((s) => [s.name, paramOf(params, `${s.name}_auto_loc`)]))
  return Object.assign(program, {
    kind: 'autoDelta',
    model,
    sites: sitesOf,
    median,
    posterior: (params: Readonly<Record<string, TensorLike>>, data: D): Posterior => {
      const values = median(params, data)
      return definePosterior({
        form: 'point',
        engine: 'svi',
        sites: Object.keys(values),
        marginals: Object.fromEntries(Object.entries(values).map(([k, v]) => [k, compose.deterministic(v)])),
        sampler: (_key, n) =>
          Object.fromEntries(
            Object.entries(values).map(([k, v]) => [
              k,
              primalOf(broadcastTo(reshape(v, [1, ...v.shape]), [n, ...v.shape])),
            ]),
          ),
      })
    },
  })
}

/**
 * How a joint (`AutoContinuous`) guide draws the flat unconstrained latent vector, and its density: a family is the
 * distribution of the flat vector inside a run, at fitted parameters, and its centre.
 */
export interface JointFamily {
  /** The distribution of the flat latent `[D]` at the current params (inside a guide run). */
  readonly latent: (D: number, sites: readonly GuideSite[], init: (k: Key) => Value) => Distribution
  /** The same at fitted params, outside a run (given the data, for families that need the model there). */
  readonly fitted: (params: Readonly<Record<string, TensorLike>>, D: number, data: unknown) => Distribution
  /** The flat latent's centre at fitted params. */
  readonly centre: (params: Readonly<Record<string, TensorLike>>) => Value
}

/**
 * A guide over the flat unconstrained latent vector of the model's continuous sites, from a {@link JointFamily}: the
 * vector is one auxiliary site (`_auto_latent`) and each model site is its slice pushed onto its support. The full-rank,
 * low-rank, Laplace and flow guides are families of this.
 *
 * @param model - The model.
 * @param kind - The guide's name.
 * @param family - The distribution of the flat latent.
 * @param options - Initialisation.
 * @returns The guide.
 * @throws OmaiError with code `value` when a site is in a subsampled plate.
 * @concept variational-inference
 * @example
 * ```ts
 * import { jointGuide, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { compose, normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { zeros } from 'omai-engine/tensor'
 * const family = { latent: (D: number) => compose.independent(normal(zeros([D]), 1), 1), fitted: (_p: never, D: number) => compose.independent(normal(zeros([D]), 1), 1), centre: () => zeros([1]) }
 * const guide = jointGuide(() => { sample('x', normal(0, 1)) }, 'standard', family, {})
 * traceOf(seed(guide, key(0)), undefined).sites.has('x') // => true
 * ```
 */
export function jointGuide<D = void>(
  model: Model<D>,
  kind: string,
  family: JointFamily,
  options: AutoGuideOptions,
): AutoGuide<D> {
  const sitesOf = cachedGuideSites(model)
  const dimOf = (sites: readonly GuideSite[]): number => sites.reduce((a, s) => a + s.size, 0)
  const initFlat =
    (sites: readonly GuideSite[]) =>
    (k: Key): Value =>
      concat(sites.map((s, i) => reshape(initialUnconstrained(s, foldIn(k, i), options), [s.size])))
  const program = (data: D): unknown => {
    const sites = sitesOf(data)
    const sub = sites.find((s) => s.subsampled)
    if (sub !== undefined)
      throw new OmaiError('value', `${kind}: site ${sub.name} is in a subsampled plate; use autoNormal`)
    const D = dimOf(sites)
    const z = sample('_auto_latent', family.latent(D, sites, initFlat(sites)), { infer: { isAuxiliary: true } })
    for (const s of sites) {
      const u = reshape(sliceAxis(z, -1, s.offset, s.offset + s.size), s.unconstrainedShape)
      const x = s.bijector.forward(u)
      const j = s.bijector.forwardLogDetJacobian(u)
      const batch = x.shape.slice(0, x.shape.length - s.eventDim)
      const logDensity = neg(reduceTo(j, batch))
      withPlates(s.plates, () => sample(s.name, delta(x, logDensity, s.eventDim)))
    }
    return undefined
  }
  const median = (params: Readonly<Record<string, TensorLike>>, data: D): Record<string, Value> => {
    const sites = sitesOf(data)
    const c = reshape(family.centre(params), [1, dimOf(sites)])
    const values = constrainFlat(sites, c, 1)
    return Object.fromEntries(Object.entries(values).map(([k, v]) => [k, primalOf(reshape(v, v.shape.slice(1)))]))
  }
  return Object.assign(program, {
    kind,
    model,
    sites: sitesOf,
    median,
    posterior: (params: Readonly<Record<string, TensorLike>>, data: D): Posterior => {
      const sites = sitesOf(data)
      const D = dimOf(sites)
      const q = family.fitted(params, D, data)
      return definePosterior({
        form: 'variational',
        engine: 'svi',
        sites: sites.map((s) => s.name),
        sampler: (key: Key, n: number) => constrainFlat(sites, primalOf(q.sample(key, [n])), n),
        diagnostics: { latentDimension: D },
      })
    },
  })
}

/**
 * The full-rank normal guide (NumPyro's `AutoMultivariateNormal`): the flat unconstrained latent vector is
 * N(`auto_loc`, L L^T) with `auto_scale_tril` lower triangular, and each site is its slice pushed onto its support
 * (a point mass carrying the log-Jacobian).
 *
 * @param model - The model.
 * @param options - Initialisation and initial scale.
 * @returns The guide.
 * @concept variational-inference
 * @example
 * ```ts
 * import { autoMultivariateNormal, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const t = traceOf(seed(autoMultivariateNormal(() => { sample('a', normal(0, 1)); sample('b', normal(0, 1)) }), key(0)), undefined)
 * t.sites.get('auto_scale_tril')?.value.shape // => [2, 2]
 * ```
 */
export function autoMultivariateNormal<D = void>(model: Model<D>, options: AutoGuideOptions = {}): AutoGuide<D> {
  const s0 = options.initScale ?? 0.1
  return jointGuide(
    model,
    'autoMultivariateNormal',
    {
      latent: (D, _sites, init) => {
        const loc = param('auto_loc', init)
        const L = param('auto_scale_tril', mul(eye(D), s0), { support: { kind: 'lowerCholesky' } })
        return multivariateNormal(loc, { scaleTril: L })
      },
      fitted: (params) =>
        multivariateNormal(paramOf(params, 'auto_loc'), { scaleTril: paramOf(params, 'auto_scale_tril') }),
      centre: (params) => paramOf(params, 'auto_loc'),
    },
    options,
  )
}

/**
 * The low-rank-plus-diagonal normal guide (NumPyro's `AutoLowRankMultivariateNormal`): covariance W W^T + diag(s^2)
 * from `auto_cov_factor` (`[D, rank]`) and `auto_scale`.
 *
 * @param model - The model.
 * @param options - Initialisation, initial scale and the rank (default ceil(sqrt(D))).
 * @returns The guide.
 * @concept variational-inference
 * @example
 * ```ts
 * import { autoLowRankMultivariateNormal, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const guide = autoLowRankMultivariateNormal(() => { sample('x', normal(0, 1), { sampleShape: [4] }) }, { rank: 1 })
 * traceOf(seed(guide, key(0)), undefined).sites.get('auto_cov_factor')?.value.shape // => [4, 1]
 * ```
 */
export function autoLowRankMultivariateNormal<D = void>(
  model: Model<D>,
  options: AutoGuideOptions & { readonly rank?: number } = {},
): AutoGuide<D> {
  const s0 = options.initScale ?? 0.1
  return jointGuide(
    model,
    'autoLowRankMultivariateNormal',
    {
      latent: (D, _sites, init) => {
        const r = options.rank ?? Math.ceil(Math.sqrt(D))
        const loc = param('auto_loc', init)
        const w = param('auto_cov_factor', (k) => mul(normalDraws(foldIn(k, 7), [D, r]), s0 * 0.1))
        const scale = param('auto_scale', full([D], s0), { support: { kind: 'positive' } })
        return lowRankMultivariateNormal(loc, w, mul(scale, scale))
      },
      fitted: (params) => {
        const scale = paramOf(params, 'auto_scale')
        return lowRankMultivariateNormal(
          paramOf(params, 'auto_loc'),
          paramOf(params, 'auto_cov_factor'),
          mul(scale, scale),
        )
      },
      centre: (params) => paramOf(params, 'auto_loc'),
    },
    options,
  )
}

/**
 * The Laplace guide (NumPyro's `AutoLaplaceApproximation`): during fitting a point mass at `auto_loc` on the
 * unconstrained scale (so the ELBO is the unconstrained log density, maximised at the MAP); its posterior is the normal
 * with the inverse Hessian of the negative log density at the fitted point as covariance.
 *
 * @param model - The model.
 * @param options - Initialisation.
 * @returns The guide.
 * @concept laplace-approximation
 * @example
 * ```ts
 * import { autoLaplace, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * [...traceOf(seed(autoLaplace(() => { sample('x', normal(0, 1)) }), key(0)), undefined).sites.keys()] // => ['auto_loc', '_auto_latent', 'x']
 * ```
 */
export function autoLaplace<D = void>(model: Model<D>, options: AutoGuideOptions = {}): AutoGuide<D> {
  return jointGuide(
    model,
    'autoLaplace',
    {
      latent: (_D, _sites, init) => compose.deterministic(param('auto_loc', init), 1),
      fitted: (params, _D, data) => {
        const loc = paramOf(params, 'auto_loc')
        const um = potentialEnergy(model, data as D)
        const cov = primalOf(inv(neg(hessian((u: Value) => um.fn(u))(loc))))
        return multivariateNormal(loc, { covariance: mul(add(cov, transpose(cov)), 0.5) })
      },
      centre: (params) => paramOf(params, 'auto_loc'),
    },
    options,
  )
}

/** Options of {@link autoIafNormal}. */
export interface AutoIafOptions extends AutoGuideOptions {
  /** The number of inverse autoregressive flow layers (default 2). */
  readonly numFlows?: number
  /** The width of each MADE conditioner's hidden layer (default 2 D + 2). */
  readonly hiddenDim?: number
}

/**
 * A normalising-flow guide (NumPyro's `AutoIAFNormal`): a standard normal over the flat unconstrained latent vector
 * pushed through inverse autoregressive flows (each an affine map whose shift and log-scale come from a MADE network on
 * the layer input, with the coordinate order reversed between layers) and an affine layer, then onto each site's
 * support. The flows start at the identity.
 *
 * @param model - The model.
 * @param options - Layers, hidden width, initialisation and initial scale.
 * @returns The guide.
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { autoIafNormal, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const guide = autoIafNormal(() => { sample('x', normal(0, 1), { sampleShape: [3] }) }, { numFlows: 1 })
 * traceOf(seed(guide, key(0)), undefined).sites.has('auto_iaf_0_w1') // => true
 * ```
 */
export function autoIafNormal<D = void>(model: Model<D>, options: AutoIafOptions = {}): AutoGuide<D> {
  const numFlows = options.numFlows ?? 2
  const s0 = options.initScale ?? 0.1
  const flowOf = (D: number, get: (name: string, init: (k: Key) => Value) => Value): bijectors.Bijector[] => {
    const H = options.hiddenDim ?? 2 * D + 2
    const parts: bijectors.Bijector[] = []
    for (let l = 0; l < numFlows; l++) {
      if (D < 2) break
      const [m1, m2] = bijectors.madeMasks(D, [H]) as [Tensor, Tensor]
      const w1 = get(`auto_iaf_${l}_w1`, (k) => mul(normalDraws(k, [D, H]), 1 / Math.sqrt(D)))
      const b1 = get(`auto_iaf_${l}_b1`, () => zeros([H]))
      const ws = get(`auto_iaf_${l}_shift`, () => zeros([H, D]))
      const wl = get(`auto_iaf_${l}_log_scale`, () => zeros([H, D]))
      const conditioner = (y: Value): bijectors.ShiftLogScale => {
        const h = tanh(add(matmul(y, mul(w1, m1)), b1))
        return { shift: matmul(h, mul(ws, m2)), logScale: matmul(h, mul(wl, m2)) }
      }
      parts.push(bijectors.invert(bijectors.autoregressive(conditioner)))
      parts.push(bijectors.permute(toFlat(arange(D - 1, -1, -1))))
    }
    return parts
  }
  const affineOf = (loc: Value, scale: Value): bijectors.Bijector =>
    bijectors.defineBijector({
      name: 'affineVector',
      inputRank: 1,
      forward: (x) => add(loc, mul(scale, x)),
      inverse: (y) => div(sub(y, loc), scale),
      forwardLogDetJacobian: (x) => add(sum(mul(x, 0), -1), sum(log(scale))),
    })
  return jointGuide(
    model,
    'autoIafNormal',
    {
      latent: (D, _sites, init) => {
        const loc = param('auto_loc', init)
        const scale = param('auto_scale', full([D], s0), { support: { kind: 'positive' } })
        const flows = flowOf(D, (name, i) => param(name, i))
        const base = compose.independent(normal(zeros([D]), 1), 1)
        return pushed(base, bijectors.chain([...flows, affineOf(loc, scale)]))
      },
      fitted: (params, D) => {
        const flows = flowOf(D, (name) => paramOf(params, name))
        const base = compose.independent(normal(zeros([D]), 1), 1)
        return compose.transformed(base, [
          ...flows,
          affineOf(paramOf(params, 'auto_loc'), paramOf(params, 'auto_scale')),
        ])
      },
      centre: (params) => paramOf(params, 'auto_loc'),
    },
    options,
  )
}
