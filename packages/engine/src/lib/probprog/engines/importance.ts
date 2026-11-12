/**
 * Importance sampling on probabilistic programs: likelihood weighting by default (draws from the prior weighted by the
 * likelihood of the observations, Pyro's `Importance`), or draws from a guide, returned as a weighted posterior with
 * the evidence estimate and the Pareto k-hat diagnostic. Also the engine for the planner.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { foldIn } from '../../random/index.ts'
import { importanceEss, logEvidence, paretoSmoothedWeights } from '../../sampling/index.ts'
import { fromStorage, item, primalOf, type Tensor, toFlat, type Value } from '../../tensor/index.ts'
import { observedLogProb } from '../_contract.ts'
import type { ProgramAnalysis } from '../analysis.ts'
import { particle } from '../elbo.ts'
import { type ProgramTrace, seed, traceOf } from '../handlers.ts'
import { type Posterior, weightedPosterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine } from './engine.ts'

/** Options of {@link importanceInference}. */
export interface ImportanceOptions<D> {
  /** Draws (default 2000). */
  readonly numSamples?: number
  /** A proposal program over the latent sites (a guide); by default the prior (likelihood weighting). */
  readonly proposal?: Model<D>
  /** Parameters of the proposal. */
  readonly proposalParams?: Readonly<Record<string, Value>>
  /** Use Pareto-smoothed weights for the posterior (default false; k-hat is always reported). */
  readonly smooth?: boolean
}

/**
 * Stack per-draw values of sites into `[n, ...shape]` tensors (one trace per draw).
 *
 * @param traces - One trace per draw.
 * @param names - The sites to stack.
 * @returns The stacked values by site.
 * @throws OmaiError with code `value` when a draw lacks a site.
 * @example
 * ```ts
 * import { deterministic, stackSiteValues, traceOf } from 'omai-engine/probprog'
 * const t = traceOf(() => deterministic('d', [1, 2]), undefined)
 * stackSiteValues([t, t], ['d'])['d']?.shape // => [2, 2]
 * ```
 */
export function stackSiteValues(traces: readonly ProgramTrace[], names: readonly string[]): Record<string, Tensor> {
  const out: Record<string, Tensor> = {}
  const n = traces.length
  for (const name of names) {
    const rows: number[] = []
    let shape: readonly number[] = []
    for (const t of traces) {
      const s = t.sites.get(name)
      if (s === undefined || s.type === 'plate' || s.type === 'markov')
        throw new OmaiError('value', `site ${name} is missing from a draw`, { site: name })
      shape = s.value.shape
      rows.push(...toFlat(primalOf(s.value)))
    }
    out[name] = fromStorage(Float64Array.from(rows), [n, ...shape], 'float64')
  }
  return out
}

/**
 * Importance sampling on a program: likelihood weighting by default (draws from the prior, weights the likelihood of
 * the observations, Pyro's `Importance` without a guide), or a proposal program with weights log p(z, x) - log q(z).
 * Returns the weighted posterior with the evidence estimate, Kish's ESS and the Pareto k-hat of the weights (PSIS
 * diagnostics: above 0.7 the estimates are unreliable).
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Draws, proposal and smoothing, key.
 * @returns The weighted posterior.
 * @concept importance-sampling
 * @example
 * ```ts
 * import { importanceInference, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const post = importanceInference(model, undefined, { key: key(0), numSamples: 4000 })
 * Math.abs(item(post.mean('m')) - 0.5) < 0.05 // => true
 * ```
 */
export function importanceInference<D = void>(
  model: Model<D>,
  data: D,
  options: ImportanceOptions<D> & EngineRunOptions,
): Posterior {
  const n = options.numSamples ?? 2000
  if (!(Number.isInteger(n) && n >= 2)) throw new OmaiError('value', 'importance: numSamples must be at least 2')
  const traces: ProgramTrace[] = []
  // banned-ok: kernel: the log-weights, filled by the loop below
  const lw = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const d = importanceDraw(model, data, options, i)
    traces.push(d.trace)
    lw[i] = d.logWeight
  }
  const first = traces[0]
  const names =
    first === undefined
      ? []
      : [...first.sites.values()]
          .filter((s) => (s.type === 'sample' && !s.isObserved) || s.type === 'deterministic')
          .map((s) => s.name)
  return importancePosterior(stackSiteValues(traces, names), lw, options)
}

/**
 * One draw of {@link importanceInference}: draw `i` of a run at a key, from the prior (weighted by the likelihood of the
 * observed sites) or from the proposal program (weighted by log p(z, x) - log q(z)). The draws of a run are independent
 * and keyed by their index, so a run can be taken in batches and resumed.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - The key and the proposal.
 * @param i - The index of the draw.
 * @returns The trace of the model at the draw and its unnormalised log weight.
 * @concept importance-sampling
 * @example
 * ```ts
 * import { importanceDraw, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * Number.isFinite(importanceDraw(model, undefined, { key: key(0) }, 0).logWeight) // => true
 * ```
 */
export function importanceDraw<D = void>(
  model: Model<D>,
  data: D,
  options: Pick<ImportanceOptions<D>, 'proposal' | 'proposalParams'> & Pick<EngineRunOptions, 'key'>,
  i: number,
): { readonly trace: ProgramTrace; readonly logWeight: number } {
  const k = foldIn(options.key, i)
  if (options.proposal === undefined) {
    const t = traceOf(seed(model, k), data)
    return { trace: t, logWeight: observedLogProb(t) }
  }
  const p = particle(k, model, options.proposal, data, options.proposalParams ?? {})
  return { trace: p.modelTrace, logWeight: item(primalOf(p.logP)) - item(primalOf(p.logQ)) }
}

/**
 * The weighted posterior of importance draws: the Pareto k-hat of the weights (PSIS), Kish's ESS and the evidence
 * estimate, with optionally Pareto-smoothed weights. {@link importanceInference} returns it; a run taken in batches
 * reads the posterior of the draws so far the same way.
 *
 * @param draws - The latent and deterministic sites' draws stacked `[n, ...shape]`.
 * @param lw - The unnormalised log weights `[n]`.
 * @param options - Whether to smooth the weights, and whether a proposal was used (for the diagnostics).
 * @returns The weighted posterior.
 * @concept importance-sampling
 * @example
 * ```ts
 * import { importancePosterior } from 'omai-engine/probprog'
 * import { tensor } from 'omai-engine/tensor'
 * importancePosterior({ m: tensor([0, 1]) }, Float64Array.of(0, 0), {}).engine // => 'importance'
 * ```
 */
export function importancePosterior<D = void>(
  draws: Readonly<Record<string, Tensor>>,
  lw: Float64Array,
  options: Pick<ImportanceOptions<D>, 'proposal' | 'smooth'>,
): Posterior {
  const n = lw.length
  const finite = lw.some((v) => Number.isFinite(v))
  let k = Number.NaN
  let smoothed: Float64Array | undefined
  if (finite && n >= 25) {
    try {
      const ps = paretoSmoothedWeights(lw)
      k = ps.paretoK
      smoothed = Float64Array.from(toFlat(ps.logWeights))
    } catch {
      k = Number.NaN
    }
  }
  const evidence = finite
    ? logEvidence(lw)
    : { value: Number.NEGATIVE_INFINITY, standardError: Number.NaN, method: 'importance sampling' }
  const diagnostics: Record<string, Json> = {
    ess: finite ? importanceEss(lw) : 0,
    paretoK: Number.isNaN(k) ? null : k,
    proposal: options.proposal === undefined ? 'prior' : 'guide',
  }
  return weightedPosterior(draws, options.smooth === true && smoothed !== undefined ? smoothed : lw, {
    engine: 'importance',
    status: finite ? 'completed' : 'failed',
    logEvidence: evidence,
    diagnostics: finite ? diagnostics : { ...diagnostics, reason: 'every draw has zero likelihood' },
  })
}

/**
 * The importance engine: likelihood weighting from the prior (cost: draws x observations; its accuracy falls as the
 * posterior concentrates, which the Pareto k-hat reports).
 *
 * @param options - Draws and smoothing.
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { importanceEngine } from 'omai-engine/probprog'
 * importanceEngine().form // => 'weighted'
 * ```
 */
export function importanceEngine(
  options: Omit<ImportanceOptions<unknown>, 'proposal' | 'proposalParams'> = {},
): InferenceEngine {
  const n = options.numSamples ?? 2000
  return {
    name: 'importance',
    form: 'weighted',
    description: 'likelihood weighting with evidence estimates and Pareto k diagnostics',
    applies: (a: ProgramAnalysis) => {
      if (!a.hasLikelihood) return feasibility(false, 'weighted', 0, 'observed sites have no likelihood')
      if (a.latent.length === 0) return feasibility(false, 'weighted', 0, 'no latent sites')
      // Likelihood weighting degrades with the information in the data: charge for it.
      const cost = n * Math.max(1, a.observations) * 2 ** Math.min(30, a.latent.length + a.observations / 10)
      return feasibility(true, 'weighted', cost, `draws from the prior weighted by ${a.observed.length} observed sites`)
    },
    run: <D>(model: Model<D>, data: D, _a: ProgramAnalysis, run: EngineRunOptions) =>
      importanceInference(model, data, { ...options, ...run }),
  }
}
