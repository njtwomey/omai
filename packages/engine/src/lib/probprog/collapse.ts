/**
 * Collapsing conjugate latent sites, the delayed sampling of Murray et al. as a handler over the programs of this
 * library: a latent site whose observed children are conjugate to it (a normal mean with normal observations, a beta
 * probability with Bernoulli trials, a gamma rate with Poisson counts, and the rest of `CONJUGATES`) is integrated out
 * analytically. The collapsed program has the marginal likelihood of the children in place of the site, so a sampler
 * on what remains moves in fewer dimensions with no loss of exactness, and `recover` draws the removed sites from
 * their closed-form posteriors given the rest (Rao-Blackwellisation by hand).
 */

import { OmaiError } from '../base/index.ts'
import { CONJUGATES, type ConjugatePair, type Distribution } from '../dists/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import {
  broadcastShapes,
  broadcastTo,
  neg,
  primalOf,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { unexpanded } from './_expand.ts'
import { analyse, type ConjugateStructure, fixedParameters, type ProgramAnalysis, pairParameter } from './analysis.ts'
import { conjugateData } from './engines/conjugate.ts'
import { handle, type SampleSite, substitute, traceOf } from './handlers.ts'
import { factor, type Message, type Model } from './primitives.ts'

/** Options of {@link collapse}. */
export interface CollapseOptions {
  /** The latent sites to collapse (default every latent site that can be). */
  readonly sites?: readonly string[]
  /** The key of the prototype run used to find the structure (default key(0)). */
  readonly key?: Key
}

/** A program with conjugate latent sites integrated out, and the means to draw them back. */
export interface Collapsed<D> {
  /** The collapsed program: the removed sites hold a placeholder, and a `<site>__marginal` factor replaces them. */
  readonly model: Model<D>
  /** The sites that were collapsed. */
  readonly sites: readonly string[]
  /** The conjugate structure found for each, with its children. */
  readonly structures: readonly ConjugateStructure[]
  /**
   * The closed-form conditional posterior of each removed site given values of every other latent site.
   *
   * @param values - Values of the latent sites that remain, by name.
   * @param data - The program's data.
   */
  posteriors(values: Readonly<Record<string, TensorLike>>, data: D): Record<string, Distribution>
  /**
   * Draw the removed sites given values of the others: `n` independent draws of each (shape `[n, ...]`).
   *
   * @param values - Values of the latent sites that remain, by name.
   * @param data - The program's data.
   * @param key - The key for the draws.
   * @param n - How many draws (default 1).
   */
  recover(values: Readonly<Record<string, TensorLike>>, data: D, key: Key, n?: number): Record<string, Tensor>
}

const isFiniteValue = (v: Value): boolean => toFlat(primalOf(v)).every(Number.isFinite)

/** A value in the support of a site's distribution to stand in for the site: the mean, else the median, else the mode. */
function placeholder(fn: Distribution): Value {
  const candidates: (() => Value)[] = [() => fn.mean(), () => fn.ppf(0.5), () => fn.mode()]
  for (const make of candidates) {
    try {
      const v = primalOf(make())
      if (isFiniteValue(v)) return v
    } catch {
      // Not available for this family: try the next.
    }
  }
  throw new OmaiError('value', `collapse: no finite point in the support of ${fn.name} to stand in for the site`)
}

function close(a: Value, b: Value): boolean {
  let shape: readonly number[]
  try {
    shape = broadcastShapes(a.shape, b.shape)
  } catch {
    return false
  }
  const x = toFlat(broadcastTo(primalOf(a), [...shape]))
  const y = toFlat(broadcastTo(primalOf(b), [...shape]))
  return x.every((v, i) => Math.abs(v - (y[i] ?? Number.NaN)) <= 1e-10 * (1 + Math.abs(v)))
}

/** Why a site cannot be collapsed, or the structure when it can. */
function structureOf<D>(
  model: Model<D>,
  data: D,
  a: ProgramAnalysis,
  name: string,
  key: Key,
): ConjugateStructure | string {
  const prior = a.prototype.sites.get(name)
  if (prior?.type !== 'sample' || prior.isObserved || prior.isFactor) return `${name} is not a latent sample site`
  if (a.sites.find((s) => s.name === name)?.discrete === true) return `${name} is discrete`
  const dependents = a.sites.filter((s) => s.parents.includes(name))
  if (dependents.length === 0) return `${name} has no observed children to collapse into`
  const priorFamily = unexpanded(prior.fn).name
  const likelihoods = new Set<string>()
  let found: ConjugatePair | undefined
  for (const d of dependents) {
    const site = a.prototype.sites.get(d.name)
    if (site?.type !== 'sample' || !site.isObserved || site.isFactor)
      return `${d.name}, which depends on ${name}, is not an observed sample site`
    const family = unexpanded(site.fn).name
    likelihoods.add(family)
    const pair = CONJUGATES.find((p) => {
      if (p.likelihood !== family || p.prior !== priorFamily) return false
      const v = pairParameter(site.fn, p)
      return v !== undefined && close(v, prior.value)
    })
    if (pair === undefined)
      return `${d.name} (${family}) is not a conjugate likelihood of the ${priorFamily} site ${name}`
    if (found !== undefined && pair !== found) return `the children of ${name} are not all of one conjugate pair`
    found = pair
  }
  if (found === undefined || likelihoods.size !== 1) return `${name} has children of several families`
  const structure: ConjugateStructure = {
    prior: name,
    priorFamily,
    likelihood: found.likelihood,
    parameter: found.parameter,
    children: dependents.map((d) => d.name),
    pair: found,
  }
  // The children must use the site only through the pair's parameter: move the site and see that nothing else moves.
  const other = prior.fn.sample(foldIn(key, 101), [])
  const latent = Object.fromEntries(
    [...a.prototype.sites.values()]
      .filter((s): s is SampleSite => s.type === 'sample' && !s.isObserved && !s.isFactor)
      .map((s) => [s.name, s.value]),
  )
  const moved = traceOf(substitute(model, { ...latent, [name]: other }), data)
  for (const c of structure.children) {
    const before = a.prototype.sites.get(c)
    const after = moved.sites.get(c)
    if (before?.type !== 'sample' || after?.type !== 'sample') return `${c} did not run again`
    const p = pairParameter(after.fn, found)
    if (p === undefined || !close(p, other)) return `${c} does not take ${name} as its ${found.parameter}`
    const f0 = fixedParameters(unexpanded(before.fn), found)
    const f1 = fixedParameters(unexpanded(after.fn), found)
    if (f0 === undefined || f1 === undefined) return `${c} lacks a known parameter`
    for (const [k, v] of Object.entries(f0)) {
      const w = f1[k]
      if (w === undefined || !close(v, w)) return `the known ${k} of ${c} depends on ${name}`
    }
  }
  const stray = moved.sites.get(name)
  if (stray?.type !== 'sample') return `${name} did not run again`
  return structure
}

/**
 * Integrate conjugate latent sites out of a program (delayed sampling, the idea of Murray et al. 2018 as a handler,
 * and the use `collapse` has in NumPyro without needing funsor): a site `s` whose observed children are all conjugate
 * to it, with the other parameters of the children known given the other sites (they may depend on other latent
 * sites), is replaced by its marginal. The collapsed program runs `s` at a placeholder in its support and adds the
 * factor `-log p(placeholder | children)` of the closed-form posterior, which turns the program's own terms
 * `log p(placeholder) + log p(children | placeholder)` into `log p(children)` exactly: for any placeholder
 * `log m(x) = log p(x | t) + log p(t) - log p(t | x)`. The result is a program in fewer sites with the same
 * distribution over them, so any engine runs on it.
 *
 * Sites are found by one traced run, as `analyse` does (the structure is fixed by the data), and checked by moving the
 * site and seeing that the children follow it only through the pair's parameter.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - The sites to collapse (default all that can be) and the prototype key.
 * @returns The collapsed program with `posteriors` and `recover`.
 * @throws OmaiError with code `value` when a requested site cannot be collapsed (the message says why), or when no
 *   site can; `not-implemented` for subsampled or masked sites.
 * @concept collapsed-inference
 * @example
 * ```ts
 * import { collapse, logJoint, observe, plate, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { item, tensor } from 'omai-engine/tensor'
 * const y = [1.2, 0.4, 2.1]
 * const model = () => { const mu = sample('mu', normal(0, 2)); plate('n', 3, () => observe('y', normal(mu, 1), y)) }
 * const c = collapse(model, undefined)
 * // The collapsed program is the marginal likelihood: y ~ N(0, I + 4 11') for a prior N(0, 2) on the mean.
 * const marginal = item(logJoint(c.model, undefined).logJoint)
 * const n = 3
 * const mean = y.reduce((a, b) => a + b, 0) / n
 * const ss = y.reduce((a, b) => a + (b - mean) ** 2, 0)
 * // log N(y; 0, I + s^2 11') = -n/2 log 2pi - 1/2 log(1 + n s^2) - 1/2 (ss + n mean^2 / (1 + n s^2))
 * const exact = -1.5 * Math.log(2 * Math.PI) - 0.5 * Math.log(1 + n * 4) - 0.5 * (ss + (n * mean ** 2) / (1 + n * 4))
 * Math.abs(marginal - exact) < 1e-12 // => true
 * ```
 */
export function collapse<D = void>(model: Model<D>, data: D, options: CollapseOptions = {}): Collapsed<D> {
  const k = options.key ?? makeKey(0)
  const a = analyse(model, data, { key: k })
  const requested = options.sites ?? a.latent
  const structures: ConjugateStructure[] = []
  const reasons: string[] = []
  for (const name of requested) {
    const s = structureOf(model, data, a, name, k)
    if (typeof s === 'string') reasons.push(s)
    else structures.push(s)
  }
  if (options.sites !== undefined && reasons.length > 0)
    throw new OmaiError('value', `collapse: cannot collapse ${reasons.join('; ')}`, { reasons: reasons.length })
  if (structures.length === 0)
    throw new OmaiError('value', `collapse: no latent site can be collapsed (${reasons.join('; ')})`)
  const byName = new Map(structures.map((s) => [s.prior, s]))

  const run = (d: D, sink: Map<string, Distribution> | undefined): unknown => {
    const handled = handle(model, () => ({
      process: (msg: Message) => {
        if (msg.type !== 'sample' || msg.isObserved || msg.fn === undefined || !byName.has(msg.name)) return
        msg.value = placeholder(msg.fn)
        msg.isObserved = true
      },
    }))
    const t = traceOf(handled, d)
    for (const s of structures) {
      const prior = t.sites.get(s.prior)
      if (prior?.type !== 'sample') throw new OmaiError('value', `collapse: the site ${s.prior} did not run`)
      for (const name of [s.prior, ...s.children]) {
        const site = t.sites.get(name)
        if (site?.type === 'sample' && (site.scale !== 1 || site.mask !== undefined))
          throw new OmaiError('not-implemented', `collapse: ${name} is subsampled or masked`, { site: name })
      }
      const { data: stacked, fixed } = conjugateData(t, s)
      const posterior = s.pair.posterior(prior.fn, stacked, fixed)
      sink?.set(s.prior, posterior)
      factor(`${s.prior}__marginal`, neg(sum(posterior.logProb(prior.value))))
    }
    return t.returnValue
  }

  const posteriors = (values: Readonly<Record<string, TensorLike>>, d: D): Record<string, Distribution> => {
    const sink = new Map<string, Distribution>()
    traceOf(
      substitute((dd: D) => run(dd, sink), values),
      d,
    )
    return Object.fromEntries(sink)
  }
  return {
    model: (d: D) => run(d, undefined),
    sites: structures.map((s) => s.prior),
    structures,
    posteriors,
    recover: (values, d, drawKey, n = 1) => {
      const out: Record<string, Tensor> = {}
      Object.entries(posteriors(values, d)).forEach(([name, dist], i) => {
        out[name] = primalOf(dist.sample(foldIn(drawKey, i), [n])) as Tensor
      })
      return out
    },
  }
}
