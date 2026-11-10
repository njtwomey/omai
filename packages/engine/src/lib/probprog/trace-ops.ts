/**
 * Trace operations, Gen's generative-function interface as small pure functions over a program's `ProgramTrace`:
 * `assess` (the log density of a full assignment), `generate` (run with some sites constrained, with the importance
 * weight), `regenerate` (resample selected sites from the program given the rest, with the weight Metropolis-Hastings
 * needs) and `update` (change some values and rerun, with the weight of the change). They are compositions of the
 * present handlers, and what a user needs to write their own sampler in a few lines (`defineKernel`).
 */

import { OmaiError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import { asTensor, item, type TensorLike, type Value } from '../tensor/index.ts'
import { siteLogProbs } from './density.ts'
import { handle, type ProgramTrace, type SampleSite, seed, substitute, traceOf } from './handlers.ts'
import type { Message, Model } from './primitives.ts'

/** Which latent sites an operation applies to: their names, or a predicate on the recorded site. */
export type SiteSelection = readonly string[] | ((site: SampleSite) => boolean)

/**
 * Whether a recorded site is in a selection.
 *
 * @param selection - Names, or a predicate.
 * @param site - The site.
 * @returns True when selected.
 * @example
 * ```ts
 * import { inSelection, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const site = traceOf(seed(() => sample('x', normal(0, 1)), key(0)), undefined).sites.get('x')
 * site?.type === 'sample' && inSelection(['x'], site) // => true
 * ```
 */
export function inSelection(selection: SiteSelection, site: SampleSite): boolean {
  return typeof selection === 'function' ? selection(site) : selection.includes(site.name)
}

/**
 * The latent sample sites of a trace: unobserved sample sites that are not factors, in execution order (the choices
 * of Gen's trace).
 *
 * @param t - The trace.
 * @returns The sites.
 * @example
 * ```ts
 * import { latentSites, observe, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 0) }
 * latentSites(traceOf(seed(model, key(0)), undefined)).map((s) => s.name) // => ['m']
 * ```
 */
export function latentSites(t: ProgramTrace): SampleSite[] {
  const out: SampleSite[] = []
  for (const s of t.sites.values()) if (s.type === 'sample' && !s.isObserved && !s.isFactor) out.push(s)
  return out
}

/**
 * The values of a trace's latent sites by name.
 *
 * @param t - The trace.
 * @returns The values.
 * @example
 * ```ts
 * import { latentValues, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * Object.keys(latentValues(traceOf(seed(() => sample('m', normal(0, 1)), key(0)), undefined))) // => ['m']
 * ```
 */
export function latentValues(t: ProgramTrace): Record<string, Value> {
  const out: Record<string, Value> = {}
  for (const s of latentSites(t)) out[s.name] = s.value
  return out
}

/** The summed, scaled and masked log density of each sample site of a trace, as numbers. */
function logProbs(t: ProgramTrace): Map<string, number> {
  const out = new Map<string, number>()
  for (const [name, v] of Object.entries(siteLogProbs(t))) out.set(name, item(v))
  return out
}

/**
 * The log density of a program at a full assignment of its latent sites (Gen's `assess`): the sum over sample sites of
 * the scaled, masked log density, observed sites included.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param values - A value for every latent sample site.
 * @returns log p(values, observations).
 * @throws OmaiError with code `value` when a latent site has no value.
 * @concept trace-operations
 * @example
 * ```ts
 * import { assess, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * Math.abs(assess(model, undefined, { m: 0 }) + Math.log(2 * Math.PI) + 0.5) < 1e-14 // => true
 * ```
 */
export function assess<D = void>(model: Model<D>, data: D, values: Readonly<Record<string, TensorLike>>): number {
  const t = traceOf(substitute(model, values), data)
  let total = 0
  for (const v of logProbs(t).values()) total += v
  return total
}

/** A trace with the log weight an operation produced. */
export interface WeightedTrace {
  readonly trace: ProgramTrace
  readonly logWeight: number
}

/**
 * Run a program, drawing every latent site from the program itself, and score the draw (Gen's `propose`).
 *
 * @param model - The program.
 * @param data - Its data.
 * @param key - The key for the draws.
 * @returns The trace and `logWeight`, the log density of the latent draws given the program (the sum of the latent
 *   sites' log probabilities, each given its parents).
 * @concept trace-operations
 * @example
 * ```ts
 * import { propose, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const { logWeight, trace } = propose(() => sample('x', normal(0, 1)), undefined, key(3))
 * [...trace.sites.keys(), Number.isFinite(logWeight)] // => ['x', true]
 * ```
 */
export function propose<D = void>(model: Model<D>, data: D, key: Key): WeightedTrace {
  const t = traceOf(seed(model, key), data)
  const lp = logProbs(t)
  let w = 0
  for (const s of latentSites(t)) w += lp.get(s.name) ?? 0
  return { trace: t, logWeight: w }
}

/**
 * Run a program with some latent sites constrained to values and the rest drawn (Gen's `generate`): the trace and the
 * importance weight, the log density of the constrained sites given the rest of the draw. With every latent site
 * constrained the weight is `assess`; with none it is the log likelihood of the observed sites, the weight of
 * likelihood weighting.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param key - The key for the sites left free.
 * @param constraints - Values of latent sites by name.
 * @returns The trace and the log weight.
 * @concept trace-operations
 * @example
 * ```ts
 * import { generate, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * // m free: the weight is the likelihood of y at the drawn m. m constrained to 0: add its prior density.
 * const w = generate(model, undefined, key(0), { m: 0 }).logWeight
 * Math.abs(w + Math.log(2 * Math.PI) + 0.5) < 1e-14 // => true
 * ```
 */
export function generate<D = void>(
  model: Model<D>,
  data: D,
  key: Key,
  constraints: Readonly<Record<string, TensorLike>> = {},
): WeightedTrace {
  const t = traceOf(seed(substitute(model, constraints), key), data)
  const lp = logProbs(t)
  let w = 0
  for (const s of t.sites.values()) {
    if (s.type !== 'sample' || s.isFactor) continue
    if (s.isObserved || constraints[s.name] !== undefined) w += lp.get(s.name) ?? 0
  }
  return { trace: t, logWeight: w }
}

/** The values a run reuses: latent sites and plate subsamples from a trace, except the ones overridden or redrawn. */
function reuse(
  old: ProgramTrace,
  skip: (site: SampleSite) => boolean,
  override: Readonly<Record<string, TensorLike>>,
): (msg: Message) => TensorLike | undefined {
  return (msg) => {
    const given = override[msg.name]
    if (given !== undefined) return given
    const site = old.sites.get(msg.name)
    if (site === undefined) return undefined
    if (msg.type === 'plate' && site.type === 'plate') return site.value
    if (msg.type === 'sample' && site.type === 'sample' && !site.isObserved && !skip(site)) return site.value
    return undefined
  }
}

/**
 * The Metropolis-Hastings weight between two runs of a program: the sum over sites present in both of the new minus
 * the old log density, with `skip` leaving out the sites that were redrawn from their conditional prior (their terms
 * cancel against the proposal). Observed sites that exist in only one run count with their own sign; latent sites
 * that exist in only one run were drawn from (or discarded with) the proposal and count for nothing.
 */
function weightBetween(old: ProgramTrace, next: ProgramTrace, skip: (site: SampleSite) => boolean): number {
  const before = logProbs(old)
  const after = logProbs(next)
  let w = 0
  for (const [name, now] of after) {
    const site = next.sites.get(name)
    if (site?.type !== 'sample' || site.isFactor) continue
    const was = before.get(name)
    if (was !== undefined) {
      const oldSite = old.sites.get(name)
      if (oldSite?.type === 'sample' && !skip(oldSite) && !skip(site)) w += now - was
    } else if (site.isObserved) w += now
  }
  for (const [name, was] of before) {
    const site = old.sites.get(name)
    if (!after.has(name) && site?.type === 'sample' && site.isObserved && !site.isFactor) w -= was
  }
  return w
}

/** What {@link regenerate} and {@link update} return: the new trace, the weight and the values it replaced. */
export interface Changed extends WeightedTrace {
  /** The old values of the sites that were redrawn or changed. */
  readonly discarded: Readonly<Record<string, Value>>
}

/**
 * Redraw selected latent sites from the program, given the rest (Gen's `regenerate`, the proposal of lightweight
 * Metropolis-Hastings): the unselected latent sites keep their values and the selected ones are drawn from their own
 * distributions given their (new) parents. `logWeight` is the sum, over the sites that were kept, of the new minus the
 * old log density (observed sites included), which is exactly the log of the Metropolis-Hastings ratio of this
 * proposal: accept with probability `min(1, exp(logWeight))`.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param trace - The current trace.
 * @param selection - The latent sites to redraw.
 * @param key - The key for the redraw.
 * @returns The new trace, the weight and the discarded old values.
 * @concept trace-operations
 * @index function Redraw selected sites from the program given the rest, with the MH weight (Gen's regenerate).
 * @example
 * ```ts
 * import { observe, regenerate, sample, seed, siteLogProbs, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const t = traceOf(seed(model, key(0)), undefined)
 * // Redrawing m from its prior: the weight is the change in the likelihood of y alone.
 * const r = regenerate(model, undefined, t, ['m'], key(1))
 * const lik = (tr: typeof t): number => item(siteLogProbs(tr)['y'] ?? 0)
 * Math.abs(r.logWeight - (lik(r.trace) - lik(t))) < 1e-14 // => true
 * ```
 */
export function regenerate<D = void>(
  model: Model<D>,
  data: D,
  trace: ProgramTrace,
  selection: SiteSelection,
  key: Key,
): Changed {
  const selected = (s: SampleSite): boolean => !s.isObserved && inSelection(selection, s)
  const next = traceOf(seed(substitute(model, reuse(trace, selected, {})), key), data)
  const discarded: Record<string, Value> = {}
  for (const s of latentSites(trace)) if (selected(s)) discarded[s.name] = s.value
  return { trace: next, logWeight: weightBetween(trace, next, selected), discarded }
}

/**
 * Change some values of a trace and rerun the program (Gen's `update`): the named sites take the new values (a latent
 * site keeps its status, an observed site gets a new observation), every other latent site keeps its value, and
 * latent sites that appear only now are drawn from the program (give a `key` for them). `logWeight` is the log of the
 * ratio of the new to the old density over the sites present in both runs, so a deterministic Metropolis-Hastings
 * move with a symmetric proposal is accepted with probability `min(1, exp(logWeight))`.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param trace - The current trace.
 * @param changes - New values by site name.
 * @param key - A key for latent sites that did not exist before (optional).
 * @returns The new trace, the weight (new minus old log density) and the replaced values.
 * @throws OmaiError with code `value` when a change names a site the program does not have.
 * @concept trace-operations
 * @example
 * ```ts
 * import { assess, observe, sample, seed, traceOf, update } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { item, type Value } from 'omai-engine/tensor'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const t = traceOf(seed(model, key(0)), undefined)
 * const old = item(t.sites.get('m')?.type === 'sample' ? (t.sites.get('m') as { value: Value }).value : 0)
 * const u = update(model, undefined, t, { m: 0.5 })
 * // The weight is the new minus the old log density.
 * Math.abs(u.logWeight - (assess(model, undefined, { m: 0.5 }) - assess(model, undefined, { m: old }))) < 1e-12 // => true
 * ```
 */
export function update<D = void>(
  model: Model<D>,
  data: D,
  trace: ProgramTrace,
  changes: Readonly<Record<string, TensorLike>>,
  key?: Key,
): Changed {
  for (const name of Object.keys(changes))
    if (!trace.sites.has(name)) throw new OmaiError('value', `update: the trace has no site ${name}`, { site: name })
  const override = Object.fromEntries(Object.entries(changes).map(([k, v]) => [k, asTensor(v)]))
  const fromTrace = reuse(trace, () => false, {})
  const placed = handle(model, () => ({
    process: (msg: Message) => {
      const given = override[msg.name]
      if (given !== undefined && (msg.type === 'sample' || msg.type === 'param' || msg.type === 'deterministic')) {
        msg.value = given
        return
      }
      if (msg.type === 'sample' && msg.isObserved) return
      const old = fromTrace(msg)
      if (old !== undefined) msg.value = asTensor(old)
    },
  }))
  const next = traceOf(key === undefined ? placed : seed(placed, key), data)
  const discarded: Record<string, Value> = {}
  for (const name of Object.keys(changes)) {
    const s = trace.sites.get(name)
    if (s?.type === 'sample') discarded[name] = s.value
  }
  return { trace: next, logWeight: weightBetween(trace, next, () => false), discarded }
}
