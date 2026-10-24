import { type Json, WhiteboxError } from '../../base/index.ts'
import { foldIn } from '../../random/index.ts'
import { importanceEss, logEvidence, paretoSmoothedWeights } from '../../sampling/index.ts'
import { fromStorage, item, mul, primalOf, sum, type Tensor, toFlat, type Value } from '../../tensor/index.ts'
import { maskedLogProb } from '../_contract.ts'
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
 * @throws WhiteboxError with code `value` when a draw lacks a site.
 * @example
 * ```ts
 * import { deterministic, stackSiteValues, traceOf } from 'whitebox-engine/probprog'
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
        throw new WhiteboxError('value', `site ${name} is missing from a draw`, { site: name })
      shape = s.value.shape
      rows.push(...toFlat(primalOf(s.value)))
    }
    out[name] = fromStorage(Float64Array.from(rows), [n, ...shape], 'float64')
  }
  return out
}

function observedLogProb(t: ProgramTrace): number {
  let total = 0
  for (const s of t.sites.values()) {
    if (s.type !== 'sample' || !s.isObserved) continue
    total += item(primalOf(mul(sum(maskedLogProb(s)), s.scale)))
  }
  return total
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
 * import { importanceInference, observe, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { item } from 'whitebox-engine/tensor'
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
  if (!(Number.isInteger(n) && n >= 2)) throw new WhiteboxError('value', 'importance: numSamples must be at least 2')
  const traces: ProgramTrace[] = []
  const lw = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const k = foldIn(options.key, i)
    if (options.proposal === undefined) {
      const t = traceOf(seed(model, k), data)
      traces.push(t)
      lw[i] = observedLogProb(t)
    } else {
      const p = particle(k, model, options.proposal, data, options.proposalParams ?? {})
      traces.push(p.modelTrace)
      lw[i] = item(primalOf(p.logP)) - item(primalOf(p.logQ))
    }
  }
  const first = traces[0]
  const names =
    first === undefined
      ? []
      : [...first.sites.values()]
          .filter((s) => (s.type === 'sample' && !s.isObserved) || s.type === 'deterministic')
          .map((s) => s.name)
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
  return weightedPosterior(
    stackSiteValues(traces, names),
    options.smooth === true && smoothed !== undefined ? smoothed : lw,
    {
      engine: 'importance',
      status: finite ? 'completed' : 'failed',
      logEvidence: evidence,
      diagnostics: finite ? diagnostics : { ...diagnostics, reason: 'every draw has zero likelihood' },
    },
  )
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
 * import { importanceEngine } from 'whitebox-engine/probprog'
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
