/**
 * Amortised guides, the encoder of a variational autoencoder for the local latents of a model: a neural network (the
 * library's `mlp`) maps each data point's features to the location and scale of its latent's normal, so a plate of N local
 * sites is covered by one network's weights instead of 2 N free numbers, and a data point never seen in training gets a
 * guide at once. The sites that are not amortised (the model's global parameters) get a mean-field normal each, as in
 * `autoNormal`. The gap to the guide a per-datum optimisation would find is the amortisation gap.
 */

import { OmaiError } from '../base/index.ts'
import { compose, type Distribution, normal } from '../dists/index.ts'
import { type Module, mlp, ReLU, Tanh } from '../nn/index.ts'
import { adam } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import {
  add,
  asTensor,
  primalOf,
  reshape,
  sliceAxis,
  softplus,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  treeFlatten,
  treeUnflatten,
  type Value,
} from '../tensor/index.ts'
import { stlElbo } from './elbo-extra.ts'
import { initialParams, svi } from './engines/svi.ts'
import {
  type AutoGuide,
  type AutoGuideOptions,
  autoNormal,
  cachedGuideSites,
  type GuideSite,
  meanFieldMarginal,
  meanFieldSite,
  pushed,
} from './guides.ts'
import { definePosterior, type Posterior } from './posterior.ts'
import { type Model, param, plate, sample } from './primitives.ts'

/** Options of {@link amortisedGuide}. */
export interface AmortisedGuideOptions<D> extends AutoGuideOptions {
  /** The local sites the encoder covers: each is a latent of shape `[N]` or `[N, k]` under one plate of size N. */
  readonly sites: readonly string[]
  /** The features of every element of the plate, `[N, d]` (or `[N]`), from the data. */
  readonly input: (data: D) => TensorLike
  /** Hidden widths of the encoder (default `[16]`; `[]` is a linear encoder). */
  readonly hidden?: readonly number[]
  /** The hidden activation (default `tanh`). */
  readonly activation?: 'tanh' | 'relu'
  /** The key of the encoder's initial weights (default `key(0)`). */
  readonly encoderKey?: Key
}

/** An amortised guide: an automatic guide that also tells where its encoder is and what it predicts. */
export interface AmortisedGuide<D> extends AutoGuide<D> {
  /** The encoder's location and scale for every element of the plate at fitted parameters, per amortised site. */
  encode(
    params: Readonly<Record<string, TensorLike>>,
    data: D,
  ): Record<string, { readonly loc: Tensor; readonly scale: Tensor }>
  /** The number of weights of the encoder of each amortised site. */
  readonly encoderParameters: (data: D) => number
}

interface Encoder {
  readonly site: GuideSite
  readonly module: Module
  readonly k: number
  readonly def: ReturnType<typeof treeFlatten>['def']
  readonly init: readonly Value[]
}

// banned-ok: non-numeric: the names of the encoder sites
const names = (site: string, n: number): string[] => Array.from({ length: n }, (_, i) => `${site}_enc_${i}`)

/**
 * A guide whose local latents are drawn from the output of a neural encoder of each data point's features (Kingma and
 * Welling 2014's amortised inference, with the library's `mlp`): site `z` of shape `[N, k]` is `N(loc(x_n), scale(x_n))` on its
 * unconstrained scale, `[loc, log-scale]` being the encoder's `2 k` outputs, pushed onto its support. The encoder's weights
 * are the guide's parameters (`<site>_enc_<i>`), so SVI trains them with every other parameter; the sites not listed are
 * mean-field. Under a subsampled plate the encoder reads only the sampled rows. The scale starts at `initScale`.
 *
 * @param model - The model.
 * @param options - The amortised sites, the features, the encoder's shape and the initialisation.
 * @returns The guide, with `encode` for the encoder's predictions at fitted parameters.
 * @throws OmaiError with code `value` for an amortised site that is not under exactly one plate, or an unknown site.
 * @concept variational-inference
 * @index function Guide whose local latents come from a neural encoder of each data point (amortised inference).
 * @example
 * ```ts
 * import { amortisedGuide, observe, plate, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const x = [0.5, -0.3, 1.1]
 * const model = () => plate('n', 3, () => { const z = sample('z', normal(0, 1)); observe('x', normal(z, 0.5), x) })
 * const guide = amortisedGuide(model, { sites: ['z'], input: () => x, hidden: [4] })
 * traceOf(seed(guide, key(0)), undefined).sites.get('z')?.value.shape // => [3]
 * ```
 */
export function amortisedGuide<D = void>(model: Model<D>, options: AmortisedGuideOptions<D>): AmortisedGuide<D> {
  const sitesOf = cachedGuideSites(model)
  const initScale = options.initScale ?? 0.1
  const offset = Math.log(Math.expm1(initScale))
  const cache = new Map<GuideSite, Encoder>()
  const encoderOf = (site: GuideSite, data: D): Encoder => {
    const hit = cache.get(site)
    if (hit !== undefined) return hit
    if (site.plates.length !== 1)
      throw new OmaiError('value', `amortisedGuide: site ${site.name} must be under exactly one plate`, {
        site: site.name,
      })
    const x = asTensor(options.input(data))
    const d = x.shape.length === 1 ? 1 : (x.shape[x.shape.length - 1] ?? 1)
    const total = site.unconstrainedShape.reduce((a, b) => a * b, 1)
    const n = site.unconstrainedShape[0] ?? 1
    if (x.shape[0] !== n)
      throw new OmaiError(
        'shape',
        `amortisedGuide: the features have ${x.shape[0]} rows, the plate of ${site.name} has ${n}`,
      )
    const k = total / n
    const module = mlp([d, ...(options.hidden ?? [16]), 2 * k], {
      activation: options.activation === 'relu' ? () => new ReLU() : () => new Tanh(),
      key: options.encoderKey ?? makeKey(0),
    })
    const flat = treeFlatten(module.initParams(options.encoderKey ?? makeKey(0)))
    const e: Encoder = { site, module, k, def: flat.def, init: flat.leaves }
    cache.set(site, e)
    return e
  }
  /** The encoder's `[B, k]` location and scale for rows `idx` of the features (all rows when undefined). */
  const run = (e: Encoder, leaves: readonly Value[], data: D, idx: Value | undefined): { loc: Value; scale: Value } => {
    let x: Value = asTensor(options.input(data))
    if (x.shape.length === 1) x = reshape(x, [x.shape[0] ?? 1, 1])
    if (idx !== undefined) x = take(x, idx)
    const out = e.module.apply(treeUnflatten(e.def, leaves) as never, x as never) as Value
    const loc = sliceAxis(out, 1, 0, e.k)
    const scale = softplus(add(sliceAxis(out, 1, e.k, 2 * e.k), offset))
    const shape = [x.shape[0] ?? 1, ...e.site.unconstrainedShape.slice(1)]
    return { loc: reshape(loc, shape), scale: reshape(scale, shape) }
  }
  const covered = (data: D): { amortised: GuideSite[]; rest: GuideSite[] } => {
    const all = sitesOf(data)
    for (const n of options.sites)
      if (!all.some((s) => s.name === n))
        throw new OmaiError('value', `amortisedGuide: the guide does not cover a site called ${n}`)
    return {
      amortised: all.filter((s) => options.sites.includes(s.name)),
      rest: all.filter((s) => !options.sites.includes(s.name)),
    }
  }
  const program = (data: D): unknown => {
    for (const s of sitesOf(data)) {
      if (!options.sites.includes(s.name)) {
        meanFieldSite(s, options)
        continue
      }
      const e = encoderOf(s, data)
      const leaves = e.init.map((v, i) => param(names(s.name, e.init.length)[i] as string, v))
      const frame = s.plates[0]
      if (frame === undefined) continue
      plate(
        frame.name,
        frame.size,
        (idx) => {
          const { loc, scale } = run(e, leaves, data, frame.subsampleSize === frame.size ? undefined : idx)
          const n = normal(loc, scale)
          const base = s.unconstrainedEventDim === 0 ? n : compose.independent(n, s.unconstrainedEventDim)
          sample(s.name, s.support.kind === 'real' ? base : pushed(base, s.bijector))
        },
        { subsampleSize: frame.subsampleSize, dim: frame.dim },
      )
    }
    return undefined
  }
  const leavesOf = (e: Encoder, params: Readonly<Record<string, TensorLike>>): Value[] =>
    names(e.site.name, e.init.length).map((n) => {
      const v = params[n]
      if (v === undefined) throw new OmaiError('value', `amortisedGuide: no parameter ${n}`)
      return primalOf(asTensor(v))
    })
  const encode: AmortisedGuide<D>['encode'] = (params, data) => {
    const out: Record<string, { loc: Tensor; scale: Tensor }> = {}
    for (const s of covered(data).amortised) {
      const e = encoderOf(s, data)
      const r = run(e, leavesOf(e, params), data, undefined)
      out[s.name] = { loc: primalOf(r.loc), scale: primalOf(r.scale) }
    }
    return out
  }
  const marginals = (params: Readonly<Record<string, TensorLike>>, data: D): Record<string, Distribution> => {
    const out: Record<string, Distribution> = {}
    const enc = encode(params, data)
    for (const s of sitesOf(data)) {
      const q = enc[s.name]
      if (q === undefined) {
        out[s.name] = meanFieldMarginal(s, params)
        continue
      }
      const n = normal(q.loc, q.scale)
      const base = s.unconstrainedEventDim === 0 ? n : compose.independent(n, s.unconstrainedEventDim)
      out[s.name] = s.support.kind === 'real' ? base : compose.transformed(base, s.bijector)
    }
    return out
  }
  const median = (params: Readonly<Record<string, TensorLike>>, data: D): Record<string, Value> => {
    const enc = encode(params, data)
    return Object.fromEntries(
      sitesOf(data).map((s) => {
        const q = enc[s.name]
        const loc = q === undefined ? primalOf(asTensor(params[`${s.name}_auto_loc`] as TensorLike)) : q.loc
        return [s.name, primalOf(s.bijector.forward(loc))]
      }),
    )
  }
  return Object.assign(program, {
    kind: 'amortisedGuide',
    model,
    sites: sitesOf,
    median,
    encode,
    encoderParameters: (data: D): number =>
      covered(data).amortised.reduce(
        (a, s) => a + encoderOf(s, data).init.reduce((b, v) => b + v.shape.reduce((c, d) => c * d, 1), 0),
        0,
      ),
    posterior: (params: Readonly<Record<string, TensorLike>>, data: D): Posterior => {
      const ms = marginals(params, data)
      const sites = sitesOf(data)
      return definePosterior({
        form: 'variational',
        engine: 'svi',
        sites: sites.map((s) => s.name),
        marginals: ms,
        sampler: (key: Key, n: number) =>
          Object.fromEntries(
            sites.map((s, i) => [s.name, primalOf((ms[s.name] as Distribution).sample(foldIn(key, i), [n]))]),
          ),
        logProb: (values) =>
          sites.reduce((acc, s) => {
            const v = values[s.name]
            if (v === undefined) throw new OmaiError('value', `guide posterior: no value for ${s.name}`)
            return acc + toFlat(primalOf((ms[s.name] as Distribution).logProb(v))).reduce((x, y) => x + y, 0)
          }, 0),
      })
    },
  })
}

/** What {@link amortisationGap} measured. */
export interface AmortisationGap {
  /** The ELBO of the amortised guide at the given parameters, averaged over draws. */
  readonly amortised: number
  /** The ELBO of a mean-field guide with its own parameters per data point, fitted on the same data. */
  readonly perDatum: number
  /** `perDatum - amortised`: what amortising costs (it can be slightly negative from Monte Carlo error). */
  readonly gap: number
}

/**
 * The amortisation gap of an amortised guide (Cremer, Li and Duvenaud 2018): the ELBO a free mean-field guide reaches on
 * the same data, with one pair of parameters per local latent, minus the ELBO of the amortised guide at its fitted
 * parameters. Both are averages of the sticking-the-landing estimator's bound over independent draws.
 *
 * @param options - The model, data, the fitted guide and parameters, and the budget of the per-datum fit.
 * @returns The two bounds and their difference.
 * @concept variational-inference
 * @example
 * ```ts
 * import { amortisationGap, amortisedGuide, observe, plate, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const x = [0.5, -0.3, 1.1]
 * const model = () => plate('n', 3, () => { const z = sample('z', normal(0, 1)); observe('x', normal(z, 0.5), x) })
 * const guide = amortisedGuide(model, { sites: ['z'], input: () => x, hidden: [] })
 * typeof amortisationGap({ model, data: undefined, guide, params: {}, key: key(0), steps: 5, draws: 4 }).gap // => 'number'
 * ```
 */
export function amortisationGap<D = void>(options: {
  readonly model: Model<D>
  readonly data: D
  readonly guide: AutoGuide<D>
  /** The amortised guide's fitted parameters (constrained), by name; empty means its initial ones. */
  readonly params: Readonly<Record<string, TensorLike>>
  readonly key: Key
  /** Steps of the per-datum fit (default 1500). */
  readonly steps?: number
  /** Draws averaged for each bound (default 200). */
  readonly draws?: number
}): AmortisationGap {
  const draws = options.draws ?? 200
  const estimator = stlElbo({ numParticles: 1 })
  const bound = (guide: Model<D>, params: Readonly<Record<string, TensorLike>>): number => {
    let total = 0
    for (let i = 0; i < draws; i++)
      total += estimator.estimate(foldIn(options.key, i), options.model, guide, options.data, params, null).elbo
    return total / draws
  }
  let params = options.params
  if (Object.keys(params).length === 0)
    params = initialParams(options.key, options.model, options.guide, options.data).values
  const amortised = bound(options.guide, params)
  const free = autoNormal(options.model)
  const fit = svi(options.model, options.data, {
    key: foldIn(options.key, 1 << 20),
    guide: free,
    numSteps: options.steps ?? 1500,
    optimizer: adam({ learningRate: (t: number) => 0.05 / (1 + t / 300) }),
    elbo: stlElbo({ numParticles: 2 }),
  })
  const perDatum = bound(free, fit.params)
  return { amortised, perDatum, gap: perDatum - amortised }
}
