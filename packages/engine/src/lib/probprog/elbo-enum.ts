/**
 * The ELBO with discrete latents summed out exactly (Pyro's `TraceEnum_ELBO`). Discrete sites of the model that the guide
 * does not draw are marginalised by plated sum-product contraction inside every particle (the model's log joint is
 * `logP` with them summed out, so the bound is the tighter, collapsed one). Finite discrete sites that the guide does draw
 * are enumerated too: each joint state of them is run, weighted by the guide's probability of it and scored with the
 * continuous draws made under it, so their parameters get an exact gradient with no score-function noise.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import {
  add,
  cast,
  exp,
  item,
  mul,
  neg,
  primalOf,
  reshape,
  stopGradient,
  sub,
  sum,
  tensor,
  type Value,
} from '../tensor/index.ts'
import { maskedLogProb } from './_contract.ts'
import { type ElboEstimate, type ElboEstimator, particle, scoreTerms } from './elbo.ts'
import { isFiniteDiscrete, type ProgramTrace, type SampleSite, seed, substitute, traceOf } from './handlers.ts'

const num = (v: Value): number => item(primalOf(v))

/** Which guide sites are enumerated. */
export type GuideEnumeration = 'all' | 'marked' | 'none'

/** Options of {@link traceEnumElbo}. */
export interface TraceEnumOptions {
  /** Particles averaged per estimate: draws of the continuous sites, each with every state of the enumerated ones (default 1). */
  readonly numParticles?: number
  /**
   * The guide's finite discrete sites to enumerate: `all` of them (default), only those marked
   * `infer: { enumerate: 'parallel' }`, or `none` (they are then sampled, with score-function gradients).
   */
  readonly guide?: GuideEnumeration
  /** The most joint states of the guide's enumerated sites to run per particle (default 256). */
  readonly maxCells?: number
  /** Decay of the moving-average baseline of the score terms of any remaining non-reparameterised site (default 0.9). */
  readonly baselineDecay?: number
}

/** One enumerated guide site: its name, value shape and states. */
interface Enumerated {
  readonly name: string
  readonly shape: readonly number[]
  readonly dtype: Value['dtype']
  readonly low: number
  readonly states: number
  readonly size: number
}

function enumeratedSites(trace: ProgramTrace, which: GuideEnumeration): Enumerated[] {
  if (which === 'none') return []
  const out: Enumerated[] = []
  for (const s of trace.sites.values()) {
    if (s.type !== 'sample' || s.isObserved || s.isFactor || !isFiniteDiscrete(s.fn.support)) continue
    if (which === 'marked' && s.infer.enumerate !== 'parallel') continue
    const support = s.fn.support
    const [low, high] =
      support.kind === 'boolean' ? [0, 1] : support.kind === 'integerInterval' ? [support.low, support.high] : [0, 0]
    out.push({
      name: s.name,
      shape: s.value.shape,
      dtype: s.value.dtype,
      low,
      states: high - low + 1,
      size: s.value.shape.reduce((a, b) => a * b, 1),
    })
  }
  return out
}

/** The values of the enumerated sites at joint state `index` (the last site's elements vary fastest). */
function cellValues(sites: readonly Enumerated[], index: number): Record<string, Value> {
  const out: Record<string, Value> = {}
  let rest = index
  for (let s = sites.length - 1; s >= 0; s--) {
    const e = sites[s] as Enumerated
    const digits: number[] = []
    for (let j = 0; j < e.size; j++) {
      digits.unshift(e.low + (rest % e.states))
      rest = Math.floor(rest / e.states)
    }
    out[e.name] = cast(reshape(tensor(digits), e.shape), e.dtype)
  }
  return out
}

/**
 * The ELBO with the discrete latents summed out (Pyro's `TraceEnum_ELBO`). Model sites that are finite and discrete and
 * not drawn by the guide are summed out of the model's joint inside each particle by plated contraction, so many
 * assignments under plates cost nothing like their joint count. Guide sites that are finite and discrete are enumerated
 * (every joint state of them is run, weighted by the guide's probability): the ELBO is then
 * sum_c q(c) (log p(c, z, x) - log q(c) - log q(z | c)) estimated over the continuous draws z only, and the gradient with
 * respect to the discrete sites' parameters is exact. The number of joint states is the product of the sites' sizes, so a
 * discrete guide site under a plate of N elements has K^N of them and is refused above `maxCells`: leave such a site to the
 * model (the guide then does not draw it, and the model sums it out).
 *
 * @param options - Particles, which guide sites to enumerate, the cell limit and the baseline decay.
 * @returns The estimator; its state is the baseline of any remaining score-function site, and its diagnostics the number
 *   of cells and the largest cell probability.
 * @throws OmaiError with code `value` when the guide's enumerated sites have more joint states than `maxCells`.
 * @concept variational-inference
 * @index estimator ELBO with finite discrete latents summed out exactly: the model's by contraction, the guide's by cells.
 * @example
 * ```ts
 * import { traceEnumElbo } from 'omai-engine/probprog'
 * traceEnumElbo({ maxCells: 64 }).name // => 'traceEnumElbo'
 * ```
 */
export function traceEnumElbo(
  options: TraceEnumOptions = {},
): ElboEstimator<{ readonly baseline: number | undefined }> {
  const n = options.numParticles ?? 1
  const which = options.guide ?? 'all'
  const maxCells = options.maxCells ?? 256
  const decay = options.baselineDecay ?? 0.9
  if (!(Number.isInteger(n) && n >= 1)) throw new OmaiError('value', 'traceEnumElbo: numParticles must be positive')
  return {
    name: 'traceEnumElbo',
    init: () => ({ baseline: undefined }),
    estimate: (key, model, guide, data, params, state): ElboEstimate<{ readonly baseline: number | undefined }> => {
      let loss: Value = tensor(0)
      let elbo = 0
      let baseline = state.baseline
      let cells = 1
      let maxWeight = 0
      for (let i = 0; i < n; i++) {
        const k: Key = foldIn(key, i)
        // The guide's enumerated sites and their states, from one run of the guide at this key.
        const proto = traceOf(seed(substitute(guide, params), k), data)
        const sites = enumeratedSites(proto, which)
        cells = sites.reduce((a, e) => a * e.states ** e.size, 1)
        if (cells > maxCells)
          throw new OmaiError(
            'value',
            `traceEnumElbo: the guide's enumerated sites (${sites.map((e) => e.name).join(', ')}) have ${cells} joint states, more than maxCells = ${maxCells}; leave discrete sites under plates to the model, which sums them out`,
          )
        const skip = new Set(sites.map((e) => e.name))
        let surrogate: Value = tensor(0)
        let value = 0
        for (let c = 0; c < cells; c++) {
          const fixed = cellValues(sites, c)
          const p = particle(k, model, guide, data, { ...params, ...fixed })
          const v = sub(p.logP, p.logQ)
          // The cell's probability under the guide: the product over its enumerated sites of their density at the cell.
          let logw: Value = tensor(0)
          for (const e of sites) {
            const site = p.guideTrace.sites.get(e.name) as SampleSite
            logw = add(logw, sum(maskedLogProb(site)))
          }
          const w = exp(logw)
          const vNum = num(v)
          const signal = baseline === undefined ? vNum : vNum - baseline
          surrogate = add(surrogate, mul(w, v))
          surrogate = add(surrogate, mul(stopGradient(w), scoreTerms(p, signal, skip)))
          value += num(w) * vNum
          maxWeight = Math.max(maxWeight, num(w))
        }
        elbo += value / n
        loss = add(loss, mul(neg(surrogate), 1 / n))
      }
      if (decay > 0) baseline = baseline === undefined ? elbo : decay * baseline + (1 - decay) * elbo
      return { loss, elbo, state: { baseline }, diagnostics: { cells, maxWeight } }
    },
  }
}
