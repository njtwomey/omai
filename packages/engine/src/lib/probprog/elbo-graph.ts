/**
 * The Rao-Blackwellised score-function ELBO (Pyro's `TraceGraph_ELBO`) with baselines: for each guide site that has no
 * reparameterised sampler (a discrete site, a gamma or beta without implicit gradients), the learning signal of its score
 * term is only the cost of the sites that depend on it, found by tracking which values every log density was computed
 * from, instead of the whole ELBO, and a control variate (a moving average, a leave-one-out mean over particles, or a
 * baseline learned by recursive least squares from the site's parents) is subtracted. Costs are kept per plate element, so
 * a local latent is credited only with its own element's cost and the global ones.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import {
  add,
  item,
  mul,
  neg,
  primalOf,
  stopGradient,
  sub,
  sum,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { hasEnumerated, maskedLogProb, plateScale } from './_contract.ts'
import { provenanceTrace } from './_provenance.ts'
import { logJoint, traceLogJoint } from './density.ts'
import type { ElboEstimate, ElboEstimator } from './elbo.ts'
import { enumerate, handle, type ProgramTrace, replay, type SampleSite, seed, substitute, traceOf } from './handlers.ts'
import type { Message, Model, PlateFrame } from './primitives.ts'

/** How the score-function terms of {@link traceGraphElbo} are controlled. */
export type BaselineKind = 'none' | 'decay' | 'leaveOneOut' | 'learned'

/** Options of {@link traceGraphElbo}. */
export interface TraceGraphOptions {
  /** Particles averaged per estimate (default 1; `leaveOneOut` needs at least 2). */
  readonly numParticles?: number
  /** The control variate of each score term (default `decay`). */
  readonly baseline?: BaselineKind
  /** Decay of the moving-average baseline (default 0.9). */
  readonly baselineDecay?: number
  /** Forgetting factor of the learned baseline's recursive least squares (default 0.99). */
  readonly forgetting?: number
  /**
   * Treat every guide site as a score-function site, even those with a reparameterised sampler (black-box VI, Ranganath
   * et al. 2014): the pathwise term is then dropped for them. Mostly for comparing variances (default false).
   */
  readonly scoreAll?: boolean
}

/** The dependency structure the estimator reads from the first run (JSON, kept in the estimator's state). */
export interface DependencyGraph {
  /** The guide's latent sites. */
  readonly guide: readonly string[]
  /** For each guide site, the guide sites its distribution depends on (directly). */
  readonly parents: Readonly<Record<string, readonly string[]>>
  /** For each guide site, the cost terms downstream of it: `['model' | 'guide', site]`. */
  readonly downstream: Readonly<Record<string, readonly (readonly ['model' | 'guide', string])[]>>
}

/** The state of {@link traceGraphElbo}: the dependency graph and the control variates' running quantities. */
export interface TraceGraphState {
  readonly deps: DependencyGraph | null
  /** The moving-average baseline of each score site. */
  readonly baselines: Readonly<Record<string, number>>
  /** The recursive least squares of the learned baseline: weights and inverse-covariance matrix. */
  readonly learned: Readonly<
    Record<string, { readonly w: readonly number[]; readonly p: readonly (readonly number[])[] }>
  >
}

const num = (v: Value): number => item(primalOf(v))

/**
 * The dependency graph of a model and a guide: which guide draws every log density of the guide and of the model was
 * computed from (tracked through every operation, as `analyse` tracks a program's), closed under the guide's own
 * dependencies, and from that the cost terms downstream of each guide site.
 *
 * @param model - The model.
 * @param guide - The guide.
 * @param data - The data.
 * @param params - Parameter values (primal).
 * @param key - A key for one run of both programs.
 * @returns The graph.
 * @concept variational-inference
 * @example
 * ```ts
 * import { autoNormal, dependencyGraph, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const g = dependencyGraph(model, autoNormal(model), undefined, { m_auto_loc: 0, m_auto_scale: 1 }, key(0))
 * g.downstream['m']?.map((c) => c.join(':')) // => ['model:m', 'model:y', 'guide:m']
 * ```
 */
export function dependencyGraph<D = void>(
  model: Model<D>,
  guide: Model<D>,
  data: D,
  params: Readonly<Record<string, TensorLike>>,
  key: Key,
): DependencyGraph {
  const guideTrace = traceOf(seed(substitute(guide, params), key), data)
  const P = provenanceTrace()
  const latent = [...guideTrace.sites.values()].filter(
    (s): s is SampleSite => s.type === 'sample' && !s.isObserved && !s.isFactor,
  )
  const names = new Set(latent.map((s) => s.name))
  const tagging =
    (trace: ProgramTrace): ((m: Message) => void) =>
    (msg) => {
      if (msg.type !== 'sample' || msg.isObserved || msg.isFactor || !names.has(msg.name)) return
      const s = trace.sites.get(msg.name)
      if (s?.type === 'sample') msg.value = P.tag(primalOf(s.value), msg.name)
    }
  const sourcesOf = (t: ProgramTrace): Map<string, Set<string>> => {
    const out = new Map<string, Set<string>>()
    for (const site of t.sites.values()) {
      if (site.type !== 'sample') continue
      out.set(site.name, new Set([...P.sources(site.fn.logProb(site.value))].filter((n) => names.has(n))))
    }
    return out
  }
  const guideRun = traceOf(
    handle(seed(substitute(guide, params), key), () => ({ process: tagging(guideTrace) })),
    data,
  )
  const guideSources = sourcesOf(guideRun)
  const modelRun = traceOf(
    handle(seed(substitute(model, params), foldIn(key, 1 << 30)), () => ({ process: tagging(guideTrace) })),
    data,
  )
  const modelSources = sourcesOf(modelRun)
  const parents: Record<string, string[]> = {}
  for (const g of names) parents[g] = [...(guideSources.get(g) ?? [])].filter((p) => p !== g)
  const ancestors = new Map<string, Set<string>>()
  const ancestorsOf = (g: string): Set<string> => {
    const hit = ancestors.get(g)
    if (hit !== undefined) return hit
    const out = new Set([g])
    ancestors.set(g, out)
    for (const p of parents[g] ?? []) for (const a of ancestorsOf(p)) out.add(a)
    return out
  }
  const reach = (sources: Iterable<string>): Set<string> => {
    const out = new Set<string>()
    for (const s of sources) for (const a of ancestorsOf(s)) out.add(a)
    return out
  }
  const downstream: Record<string, (readonly ['model' | 'guide', string])[]> = {}
  for (const z of names) downstream[z] = []
  for (const [site, sources] of modelSources) {
    for (const z of reach(sources)) downstream[z]?.push(['model', site])
  }
  for (const g of names) for (const z of reach([g, ...(guideSources.get(g) ?? [])])) downstream[z]?.push(['guide', g])
  return { guide: [...names], parents, downstream }
}

/** The product of the subsampling scales of plates in `frames` that are not in `shared`. */
function unsharedScale(frames: readonly PlateFrame[], shared: ReadonlySet<string>): number {
  return frames.reduce(
    (acc, f) => (shared.has(f.name) ? acc : acc * (f.subsampleSize === 0 ? 1 : f.size / f.subsampleSize)),
    1,
  )
}

/** A cost term summed over every batch axis that is not one of `dims`, keeping the others as singleton-or-full axes. */
function alignedTo(lp: Value, dims: ReadonlySet<number>): Value {
  const rank = lp.shape.length
  const axes: number[] = []
  for (let a = 0; a < rank; a++) if ((lp.shape[a] ?? 1) > 1 && !dims.has(a - rank)) axes.push(a)
  return axes.length === 0 ? lp : sum(lp, axes, true)
}

const meanOf = (v: Value): number => {
  const x = toFlat(primalOf(v))
  return x.length === 0 ? 0 : x.reduce((a, b) => a + b, 0) / x.length
}

/** One draw: the traces, the per-site cost terms and the pathwise surrogate. */
interface Draw {
  readonly guideTrace: ProgramTrace
  readonly modelTrace: ProgramTrace
  readonly enumerated: boolean
  readonly logP: Value
  readonly logQ: Value
}

function draw<D>(
  key: Key,
  model: Model<D>,
  guide: Model<D>,
  data: D,
  params: Readonly<Record<string, TensorLike>>,
  scoreAll: boolean,
): Draw {
  let guideTrace = traceOf(seed(substitute(guide, params), key), data)
  if (scoreAll) {
    // The draw is a constant: its log density is evaluated at the stopped values, so the parameters act only through it.
    const values: Record<string, Value> = {}
    for (const s of guideTrace.sites.values())
      if (s.type === 'sample' && !s.isObserved) values[s.name] = stopGradient(s.value)
    guideTrace = traceOf(seed(substitute(guide, { ...params, ...values }), key), data)
  }
  const r = logJoint(seed(replay(substitute(enumerate(model), params), guideTrace), foldIn(key, 1 << 30)), data)
  return {
    guideTrace,
    modelTrace: r.trace,
    enumerated: hasEnumerated(r.trace),
    logP: r.logJoint,
    logQ: traceLogJoint(guideTrace),
  }
}

/**
 * The Rao-Blackwellised score-function ELBO (Pyro's `TraceGraph_ELBO`). The gradient is the pathwise gradient of
 * log p - log q through every reparameterised draw, plus, for each guide site z without a reparameterised sampler, the
 * score term log q(z) (Q_z - b_z) where Q_z is only the cost (log p minus log q) of the sites that depend on z, directly
 * or through other draws, and b_z is a baseline that does not depend on z. Dropping the costs z cannot influence is
 * exact in expectation and removes their noise; the baseline removes the rest of the level of the cost. Costs are
 * counted per plate element, so a local latent sees its own element and the global sites. Model sites summed out by
 * enumeration count as one cost that depends on every guide site.
 *
 * Baselines: `none`; `decay`, a moving average of the site's cost; `leaveOneOut`, the mean of the other particles' cost
 * (needs `numParticles` of 2 or more); `learned`, a linear regression of the cost on the values of the site's parents in
 * the guide, fitted online by recursive least squares from earlier steps.
 *
 * @param options - Particles, the baseline kind, its decay or forgetting factor, and `scoreAll`.
 * @returns The estimator; its state is the dependency graph and the baselines.
 * @throws OmaiError with code `value` for `leaveOneOut` with fewer than two particles.
 * @concept variational-inference
 * @index estimator Rao-Blackwellised score-function ELBO with baselines for non-reparameterised sites.
 * @example
 * ```ts
 * import { traceGraphElbo } from 'omai-engine/probprog'
 * traceGraphElbo({ baseline: 'leaveOneOut', numParticles: 4 }).name // => 'traceGraphElbo'
 * ```
 */
export function traceGraphElbo(options: TraceGraphOptions = {}): ElboEstimator<TraceGraphState> {
  const n = options.numParticles ?? 1
  const kind = options.baseline ?? 'decay'
  const decay = options.baselineDecay ?? 0.9
  const forgetting = options.forgetting ?? 0.99
  const scoreAll = options.scoreAll ?? false
  if (!(Number.isInteger(n) && n >= 1)) throw new OmaiError('value', 'traceGraphElbo: numParticles must be positive')
  if (kind === 'leaveOneOut' && n < 2)
    throw new OmaiError('value', 'traceGraphElbo: the leave-one-out baseline needs numParticles of at least 2')
  return {
    name: 'traceGraphElbo',
    init: () => ({ deps: null, baselines: {}, learned: {} }),
    estimate: (key, model, guide, data, params, state): ElboEstimate<TraceGraphState> => {
      const primal = Object.fromEntries(Object.entries(params).map(([k, v]) => [k, primalOf(v as Value)]))
      const deps = state.deps ?? dependencyGraph(model, guide, data, primal, foldIn(key, 1 << 29))
      const draws = Array.from({ length: n }, (_, i) => draw(foldIn(key, i), model, guide, data, params, scoreAll))
      // The pathwise value of every draw, and the score sites with their cost per draw.
      const scoreSites = (d: Draw): SampleSite[] =>
        [...d.guideTrace.sites.values()].filter(
          (s): s is SampleSite => s.type === 'sample' && !s.isObserved && !s.isFactor && (scoreAll || !s.fn.hasRsample),
        )
      const costs = draws.map((d) => {
        const out = new Map<string, Value>()
        for (const z of scoreSites(d)) {
          const zDims = new Set(z.plates.map((f) => f.dim))
          const zPlates = new Set(z.plates.map((f) => f.name))
          let q: Value = tensor(0)
          if (d.enumerated) {
            // One cost for the contracted model, and the guide's own sites that depend on z.
            q = add(q, stopGradient(d.logP))
          }
          for (const [type, name] of deps.downstream[z.name] ?? []) {
            if (type === 'model') {
              if (d.enumerated) continue
              const m = d.modelTrace.sites.get(name)
              if (m?.type !== 'sample') continue
              const f = (m.scale / plateScale(m.plates)) * unsharedScale(m.plates, zPlates)
              q = add(q, mul(alignedTo(stopGradient(maskedLogProb(m)), zDims), f))
            } else {
              const g = d.guideTrace.sites.get(name)
              if (g?.type !== 'sample') continue
              const f = (g.scale / plateScale(g.plates)) * unsharedScale(g.plates, zPlates)
              q = sub(q, mul(alignedTo(stopGradient(maskedLogProb(g)), zDims), f))
            }
          }
          out.set(z.name, q)
        }
        return out
      })
      const baselines: Record<string, number> = { ...state.baselines }
      const learned: Record<string, { w: number[]; p: number[][] }> = Object.fromEntries(
        Object.entries(state.learned).map(([k, v]) => [k, { w: [...v.w], p: v.p.map((r) => [...r]) }]),
      )
      const diagnostics: Record<string, number> = {}
      let loss: Value = tensor(0)
      let elbo = 0
      const seen = new Map<string, number[]>()
      draws.forEach((d, i) => {
        elbo += num(sub(d.logP, d.logQ)) / n
        // The pathwise surrogate: log p - log q, without the density of a score site's own draw (which enters through its
        // score term; its direct derivative has mean zero).
        let surrogate: Value = sub(d.logP, d.logQ)
        for (const z of scoreSites(d)) {
          surrogate = add(surrogate, mul(sum(maskedLogProb(z)), z.scale))
          const q = costs[i]?.get(z.name) as Value
          const outer = plateScale(z.plates)
          let b: Value | number = 0
          const mean = meanOf(q)
          if (kind === 'decay') b = baselines[z.name] ?? 0
          else if (kind === 'leaveOneOut') {
            let others: Value = tensor(0)
            for (let j = 0; j < n; j++) if (j !== i) others = add(others, costs[j]?.get(z.name) as Value)
            b = mul(others, 1 / (n - 1))
          } else if (kind === 'learned') {
            const parents = deps.parents[z.name] ?? []
            const phi = [
              1,
              ...parents.map((p) => {
                const s = d.guideTrace.sites.get(p)
                return s?.type === 'sample' ? meanOf(s.value) : 0
              }),
            ]
            let fit = learned[z.name]
            if (fit === undefined) {
              fit = { w: phi.map(() => 0), p: phi.map((_, r) => phi.map((__, c) => (r === c ? 100 : 0))) }
              learned[z.name] = fit
            }
            // Predict first (from earlier steps only), then update with this draw's cost.
            b = fit.w.reduce((a, wk, k) => a + wk * (phi[k] ?? 0), 0)
            const pphi = fit.p.map((row) => row.reduce((a, pk, k) => a + pk * (phi[k] ?? 0), 0))
            const denom = forgetting + phi.reduce((a, f, k) => a + f * (pphi[k] ?? 0), 0)
            const err = mean - (b as number)
            fit.w = fit.w.map((wk, k) => wk + ((pphi[k] ?? 0) / denom) * err)
            const phiP = phi.map((_, c) => fit.p.reduce((a, row, r) => a + (phi[r] ?? 0) * (row[c] ?? 0), 0))
            fit.p = fit.p.map((row, r) =>
              row.map((pk, c) => (pk - ((pphi[r] ?? 0) * (phiP[c] ?? 0)) / denom) / forgetting),
            )
          }
          const signal = stopGradient(sub(q, b))
          surrogate = add(surrogate, mul(sum(mul(maskedLogProb(z), signal)), outer))
          const arr = seen.get(z.name) ?? []
          arr.push(mean)
          seen.set(z.name, arr)
        }
        loss = add(loss, mul(neg(surrogate), 1 / n))
      })
      for (const [site, means] of seen) {
        const m = means.reduce((a, b) => a + b, 0) / means.length
        if (kind === 'decay')
          baselines[site] = baselines[site] === undefined ? m : decay * (baselines[site] as number) + (1 - decay) * m
        diagnostics[`baseline.${site}`] = kind === 'decay' ? (baselines[site] as number) : m
        diagnostics[`cost.${site}`] = m
      }
      return { loss, elbo, state: { deps, baselines, learned }, diagnostics }
    },
  }
}
