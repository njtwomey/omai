/**
 * Approximate inference by sampling, pgmpy style: rejection sampling and likelihood weighting for Bayes networks, Gibbs
 * sampling on the factor graph of any discrete model (as a traced algorithm), effective sample size of importance
 * weights, and `approximateQuery` for posteriors estimated from samples.
 */

import { type Json, logSumExpOf, OmaiError } from '../../base/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import { dict, type OptionTable, optionTable } from '../../space/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  runAlgorithm,
  scalar,
  type TracePolicy,
  type Tracer,
} from '../../trace/index.ts'
import { type Factor, logFactor } from '../factor.ts'
import { type BayesNet, bayesNode, cptRow, forwardSample } from '../models/bayes-net.ts'
import { type DiscreteModel, evidenceIndices, modelVariable, modelVariables } from '../models/factor-graph.ts'
import { type Compiled, compile } from './_compiled.ts'
import { maxProductElimination } from './elimination.ts'

/** A sample: a state index for each variable. */
export type Sample = Record<string, number>

/** The paths the importance and rejection samplers log, as a type. */
export type ImportancePath = 'is/weight' | 'is/ess' | 'is/accepted' | 'is/estimate'

/**
 * What the rejection and importance samplers log, by path: the log weight of the sample drawn (0 for a rejection
 * sampler's accepted sample, minus infinity for a rejected one), the effective sample size so far, whether the sample
 * was accepted (rejection: agrees with the evidence; likelihood weighting: positive weight) and the running estimate
 * of the probability of the evidence (the acceptance rate, or the mean weight).
 */
export const importanceContract: Contract<unknown, ImportancePath> = defineContract({
  'is/weight': scalar({ description: 'the log importance weight of the sample drawn at this step' }),
  'is/ess': scalar({ description: 'the effective sample size of the weights so far' }),
  'is/accepted': scalar({ description: '1 when the sample counts (agrees with the evidence, or has positive weight)' }),
  'is/estimate': scalar({ description: 'the running estimate of the probability of the evidence' }),
})

/** The state of the rejection and likelihood-weighting samplers. The sample lists grow in place; copy to keep one. */
export interface ImportanceState {
  /** The samples kept: accepted ones for rejection sampling, all for likelihood weighting. */
  readonly samples: readonly Sample[]
  /** The log weights of the kept samples (all 0 for rejection sampling). */
  readonly logWeights: readonly number[]
  /** Samples drawn so far. */
  readonly drawn: number
  /** The sample drawn at the last step. */
  readonly last: Sample
}

/** A log weight as JSON (infinite values as strings). */
const weightsToJson = (w: readonly number[]): Json => w.map((x) => (Number.isFinite(x) ? x : String(x)))

function importanceCodec(): Pick<Algorithm<ImportanceState>, 'serialise' | 'deserialise'> {
  return {
    serialise: (s) => ({
      samples: s.samples.map((x) => ({ ...x })),
      logWeights: weightsToJson(s.logWeights),
      drawn: s.drawn,
      last: { ...s.last },
    }),
    deserialise: (j) => {
      const o = j as unknown as {
        readonly samples: readonly Sample[]
        readonly logWeights: readonly (number | string)[]
        readonly drawn: number
        readonly last: Sample
      }
      return {
        samples: o.samples.map((x) => ({ ...x })),
        logWeights: o.logWeights.map(Number),
        drawn: o.drawn,
        last: { ...o.last },
      }
    },
  }
}

function logImportance(
  ctx: Parameters<Algorithm<ImportanceState, ImportancePath>['step']>[1],
  s: ImportanceState,
  weight: number,
  accepted: boolean,
  estimate: number,
): void {
  if (ctx.wants('is/weight')) ctx.log('is/weight', weight)
  if (ctx.wants('is/ess')) ctx.log('is/ess', effectiveSampleSize(s.logWeights))
  if (ctx.wants('is/accepted')) ctx.log('is/accepted', accepted ? 1 : 0)
  if (ctx.wants('is/estimate')) ctx.log('is/estimate', estimate)
}

/**
 * Rejection sampling as a steppable algorithm: each step draws one forward (ancestral) sample and keeps it when it
 * agrees with the evidence; the kept samples are exact draws from the posterior, and the acceptance rate estimates
 * the probability of the evidence.
 *
 * @param bn - The network.
 * @param key - The random key.
 * @param n - Number of forward samples to draw.
 * @param evidence - Observed states (names or indices).
 * @returns The algorithm; it is done after `n` draws.
 * @concept sampling
 * @example
 * ```ts
 * import { bayesNet, rejectionSampleAlgorithm } from 'omai-engine/pgm'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [1, 0, 0, 1] }])
 * runAlgorithm(rejectionSampleAlgorithm(bn, key(0), 20, { b: '1' }), { maxSteps: 20 }).state.drawn // => 20
 * ```
 */
export function rejectionSampleAlgorithm(
  bn: BayesNet,
  key: Key,
  n: number,
  evidence: Readonly<Record<string, string | number>> = {},
): Algorithm<ImportanceState, ImportancePath> {
  const ev = evidenceIndices(bn, evidence)
  return {
    name: 'pgm/rejection-sampling',
    config: { nodes: bn.nodes.length, samples: n },
    init: () => ({ samples: [], logWeights: [], drawn: 0, last: {} }),
    step: (s, ctx) => {
      if (s.drawn >= n) return s
      const sample = forwardSample(bn, foldIn(key, s.drawn))
      const accepted = Object.entries(ev).every(([k, v]) => sample[k] === v)
      if (accepted) {
        ;(s.samples as Sample[]).push(sample)
        ;(s.logWeights as number[]).push(0)
      }
      const next: ImportanceState = { samples: s.samples, logWeights: s.logWeights, drawn: s.drawn + 1, last: sample }
      logImportance(ctx, next, accepted ? 0 : Number.NEGATIVE_INFINITY, accepted, next.samples.length / next.drawn)
      return next
    },
    done: (s) => (s.drawn >= n ? 'converged' : undefined),
    ...importanceCodec(),
  }
}

/**
 * Rejection sampling: draw forward (ancestral) samples and keep those that agree with the evidence; the kept samples
 * are exact draws from the posterior.
 *
 * @param bn - The network.
 * @param key - The random key.
 * @param n - Number of forward samples to draw.
 * @param evidence - Observed states (names or indices).
 * @returns The accepted samples and the acceptance rate (an estimate of P(evidence)).
 * @concept sampling
 * @example
 * ```ts
 * import { bayesNet, rejectionSample } from 'omai-engine/pgm'
 * import { key } from 'omai-engine/random'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [1, 0, 0, 1] }])
 * rejectionSample(bn, key(0), 100, { b: '1' }).samples.every((s) => s['a'] === 1) // => true
 * ```
 */
export function rejectionSample(
  bn: BayesNet,
  key: Key,
  n: number,
  evidence: Readonly<Record<string, string | number>> = {},
): { readonly samples: Sample[]; readonly acceptance: number } {
  const { samples } = runAlgorithm(rejectionSampleAlgorithm(bn, key, n, evidence), { maxSteps: n }).state
  return { samples: [...samples], acceptance: n === 0 ? 0 : samples.length / n }
}

/**
 * Likelihood weighting as a steppable algorithm: each step samples the unobserved variables forward with the observed
 * ones clamped and weights the sample by the probability of the evidence given its parents (importance sampling with
 * the prior as proposal). The mean weight estimates the probability of the evidence.
 *
 * @param bn - The network.
 * @param key - The random key.
 * @param n - Number of samples.
 * @param evidence - Observed states (names or indices).
 * @returns The algorithm; it is done after `n` samples.
 * @concept sampling
 * @example
 * ```ts
 * import { bayesNet, likelihoodWeightingAlgorithm } from 'omai-engine/pgm'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [0.9, 0.1, 0.2, 0.8] }])
 * runAlgorithm(likelihoodWeightingAlgorithm(bn, key(0), 5, { b: '1' }), { maxSteps: 5 }).state.samples.length // => 5
 * ```
 */
export function likelihoodWeightingAlgorithm(
  bn: BayesNet,
  key: Key,
  n: number,
  evidence: Readonly<Record<string, string | number>> = {},
): Algorithm<ImportanceState, ImportancePath> {
  const ev = evidenceIndices(bn, evidence)
  return {
    name: 'pgm/likelihood-weighting',
    config: { nodes: bn.nodes.length, samples: n },
    init: () => ({ samples: [], logWeights: [], drawn: 0, last: {} }),
    step: (st, ctx) => {
      if (st.drawn >= n) return st
      const u = uniform(foldIn(key, st.drawn), [bn.nodes.length]).data
      const s: Sample = {}
      let lw = 0
      for (const [j, node] of bn.nodes.entries()) {
        const row = cptRow(bn, node.name, s)
        const observed = ev[node.name]
        if (observed !== undefined) {
          s[node.name] = observed
          lw += Math.log(row[observed] ?? 0)
          continue
        }
        let acc = 0
        let state = row.length - 1
        for (const [k, p] of row.entries()) {
          acc += p
          if ((u[j] ?? 0) < acc) {
            state = k
            break
          }
        }
        s[node.name] = state
      }
      ;(st.samples as Sample[]).push(s)
      ;(st.logWeights as number[]).push(lw)
      const next: ImportanceState = { samples: st.samples, logWeights: st.logWeights, drawn: st.drawn + 1, last: s }
      logImportance(
        ctx,
        next,
        lw,
        lw > Number.NEGATIVE_INFINITY,
        Math.exp(logSumExpOf(next.logWeights) - Math.log(next.drawn)),
      )
      return next
    },
    done: (s) => (s.drawn >= n ? 'converged' : undefined),
    ...importanceCodec(),
  }
}

/** What builds an importance sampler: the network, the key, the sample count and the evidence. */
export type ImportanceBuilder = (
  bn: BayesNet,
  key: Key,
  n: number,
  evidence?: Readonly<Record<string, string | number>>,
) => Algorithm<ImportanceState, ImportancePath>

/** What an importance sampler can do, for validity rules and selectors. */
export interface ImportanceProperties {
  /** Whether it wastes the samples that disagree with the evidence (rejection) rather than weighting them. */
  readonly discards: boolean
  /** Whether the kept samples are exact draws from the posterior (unweighted). */
  readonly exact: boolean
}

/**
 * The samplers of a Bayes-net posterior as one slot: `rejection` (forward samples that disagree with the evidence are
 * discarded; the kept ones are exact) and `likelihood-weighting` (the evidence is clamped and each sample weighted).
 *
 * @concept sampling
 * @example
 * ```ts
 * import { importanceSamplers } from 'omai-engine/pgm'
 * importanceSamplers.where((p) => p.discards) // => ['rejection']
 * ```
 */
export const importanceSamplers: OptionTable<ImportanceBuilder, ImportanceProperties> = optionTable(
  {
    rejection: {
      description: 'forward samples, discarding those that disagree with the evidence',
      properties: { discards: true, exact: true },
      space: dict({}),
      make: () => (bn, key, n, evidence) => rejectionSampleAlgorithm(bn, key, n, evidence),
    },
    'likelihood-weighting': {
      description: 'sample with the evidence clamped and weight by its likelihood',
      properties: { discards: false, exact: false },
      space: dict({}),
      make: () => (bn, key, n, evidence) => likelihoodWeightingAlgorithm(bn, key, n, evidence),
    },
  },
  {
    default: 'likelihood-weighting',
    label: 'importance sampler',
    description: 'How samples of a Bayes net are made to respect the evidence.',
  },
)

/**
 * Likelihood weighting: sample the unobserved variables forward with the observed ones clamped, and weight each sample
 * by the probability of the evidence given its parents (importance sampling with the prior as proposal).
 *
 * @param bn - The network.
 * @param key - The random key.
 * @param n - Number of samples.
 * @param evidence - Observed states (names or indices).
 * @returns The samples, their log weights, and the effective sample size.
 * @concept sampling
 * @example
 * ```ts
 * import { bayesNet, likelihoodWeighting } from 'omai-engine/pgm'
 * import { key } from 'omai-engine/random'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [0.9, 0.1, 0.2, 0.8] }])
 * const r = likelihoodWeighting(bn, key(0), 3, { b: '1' })
 * r.logWeights.map((w) => Math.round(Math.exp(w) * 10) / 10).every((w) => w === 0.1 || w === 0.8) // => true
 * ```
 */
export function likelihoodWeighting(
  bn: BayesNet,
  key: Key,
  n: number,
  evidence: Readonly<Record<string, string | number>> = {},
): { readonly samples: Sample[]; readonly logWeights: number[]; readonly effectiveSampleSize: number } {
  const { samples, logWeights } = runAlgorithm(likelihoodWeightingAlgorithm(bn, key, n, evidence), {
    maxSteps: n,
  }).state
  return { samples: [...samples], logWeights: [...logWeights], effectiveSampleSize: effectiveSampleSize(logWeights) }
}

/**
 * The effective sample size of importance weights, (sum w)^2 / sum w^2, from log weights.
 *
 * @param logWeights - Log importance weights.
 * @returns The effective number of samples (0 when every weight is zero).
 * @concept sampling
 * @example
 * ```ts
 * import { effectiveSampleSize } from 'omai-engine/pgm'
 * effectiveSampleSize([0, 0, 0, 0]) // => 4
 * ```
 */
export function effectiveSampleSize(logWeights: readonly number[]): number {
  const a = logSumExpOf(logWeights)
  if (a === Number.NEGATIVE_INFINITY) return 0
  return Math.exp(2 * a - logSumExpOf(logWeights.map((w) => 2 * w)))
}

/**
 * Estimate the marginal (or joint) distribution of some variables from samples, optionally importance weighted.
 *
 * @param model - The model (for the variables' states).
 * @param samples - The samples.
 * @param variables - The variables to estimate (a joint over all of them, in this order).
 * @param logWeights - Log importance weights (default: equal).
 * @returns The estimated distribution as a normalised factor.
 * @throws OmaiError with code `value` when there are no samples or every weight is zero.
 * @concept sampling
 * @example
 * ```ts
 * import { bayesNet, sampleMarginal, factorValues } from 'omai-engine/pgm'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }])
 * factorValues(sampleMarginal(bn, [{ a: 0 }, { a: 1 }, { a: 1 }, { a: 1 }], ['a'])) // => [0.25, 0.75]
 * ```
 */
export function sampleMarginal(
  model: DiscreteModel,
  samples: readonly Readonly<Sample>[],
  variables: readonly string[],
  logWeights?: readonly number[],
): Factor {
  const card = variables.map((v) => modelVariable(model, v).states.length)
  const size = card.reduce((a, b) => a * b, 1)
  const cells: number[][] = Array.from({ length: size }, () => [])
  for (const [i, s] of samples.entries()) {
    let idx = 0
    for (const [k, v] of variables.entries()) idx = idx * (card[k] ?? 1) + (s[v] ?? 0)
    cells[idx]?.push(logWeights?.[i] ?? 0)
  }
  const logs = cells.map(logSumExpOf)
  const total = logSumExpOf(logs)
  if (total === Number.NEGATIVE_INFINITY)
    throw new OmaiError('value', 'sampleMarginal: no samples with positive weight')
  return logFactor(
    variables,
    card,
    logs.map((l) => l - total),
  )
}

/** Options of {@link gibbsSamplingAlgorithm} and {@link gibbsSample}. */
export interface GibbsOptions {
  /** Observed states (names or indices), held fixed. */
  readonly evidence?: Readonly<Record<string, string | number>>
  /** The starting state of the unobserved variables (default: the MPE, which always has positive probability). */
  readonly init?: Readonly<Record<string, number>>
  /** Sweeps discarded before collecting (default 100). */
  readonly burnIn?: number
  /** Keep every `thin`-th sweep (default 1). */
  readonly thin?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of a Gibbs run. */
export interface GibbsState {
  /** The current state of each unobserved variable, in compiled order. */
  readonly x: Int32Array
  /** The unnormalised log probability of the current state. */
  readonly logProbability: number
}

/** Per variable: the factors it is in, its position and stride in each. */
function neighbourhoods(c: Compiled): { factor: number; stride: number }[][] {
  return c.names.map((_, i) =>
    (c.variableEdges[i] ?? []).map((e) => {
      const { factor, position } = c.edges[e] ?? { factor: 0, position: 0 }
      return { factor, stride: c.factors[factor]?.strides[position] ?? 0 }
    }),
  )
}

function flatIndex(c: Compiled, a: number, x: Int32Array): number {
  const f = c.factors[a]
  let idx = 0
  for (const [k, v] of (f?.vars ?? []).entries()) idx += (x[v] ?? 0) * (f?.strides[k] ?? 0)
  return idx
}

function logProbabilityOf(c: Compiled, x: Int32Array): number {
  let s = c.constant
  for (const [a, f] of c.factors.entries()) s += f.logValues[flatIndex(c, a, x)] ?? Number.NEGATIVE_INFINITY
  return s
}

/**
 * Gibbs sampling on the factor graph of any discrete model, as an {@link Algorithm}: each step is one systematic sweep
 * that resamples every unobserved variable from its conditional given its Markov blanket (the product of the factors it
 * appears in). Records the unnormalised log probability of the state after each sweep.
 *
 * @param model - The model.
 * @param options - Evidence and the starting state.
 * @returns The algorithm.
 * @throws OmaiError with code `value` when the starting state has zero probability.
 * @concept gibbs-sampling
 * @example
 * ```ts
 * import { isingModel, gibbsSamplingAlgorithm } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(gibbsSamplingAlgorithm(isingModel({ rows: 2, cols: 2 })), { maxSteps: 10, seed: 1 }).steps // => 10
 * ```
 */
export function gibbsSamplingAlgorithm(model: DiscreteModel, options: GibbsOptions = {}): Algorithm<GibbsState> {
  const c = compile(model, options.evidence ?? {})
  const hood = neighbourhoods(c)
  const start = (): Int32Array => {
    if (options.init !== undefined) {
      return Int32Array.from(c.names, (v) => options.init?.[v] ?? 0)
    }
    const best = maxProductElimination(
      model,
      options.evidence === undefined ? {} : { evidence: options.evidence },
    ).indices
    return Int32Array.from(c.names, (v) => best[v] ?? 0)
  }
  return {
    name: 'gibbsSampling',
    config: { variables: [...c.names] } as Json,
    schema: channels({ metrics: { logProbability: { description: 'unnormalised log probability of the state' } } }),
    init: () => {
      const x = start()
      const lp = logProbabilityOf(c, x)
      if (lp === Number.NEGATIVE_INFINITY)
        throw new OmaiError('value', 'gibbs: the starting state has zero probability')
      return { x, logProbability: lp }
    },
    step: (state, ctx) => {
      const x = Int32Array.from(state.x)
      const u = uniform(ctx.key, [Math.max(c.names.length, 1)]).data
      for (let i = 0; i < c.names.length; i++) {
        const k = c.card[i] ?? 1
        const logs = new Float64Array(k)
        for (const { factor, stride } of hood[i] ?? []) {
          const f = c.factors[factor]
          const base = flatIndex(c, factor, x) - (x[i] ?? 0) * stride
          for (let s = 0; s < k; s++)
            logs[s] = (logs[s] ?? 0) + (f?.logValues[base + s * stride] ?? Number.NEGATIVE_INFINITY)
        }
        const z = logSumExpOf(logs)
        if (z === Number.NEGATIVE_INFINITY)
          throw new OmaiError('value', `gibbs: ${c.names[i]} has no state of positive probability`)
        let acc = 0
        let pick = k - 1
        for (let s = 0; s < k; s++) {
          acc += Math.exp((logs[s] ?? 0) - z)
          if ((u[i] ?? 0) < acc) {
            pick = s
            break
          }
        }
        x[i] = pick
      }
      return { x, logProbability: logProbabilityOf(c, x) }
    },
    observe: (state) => ({ metrics: { logProbability: state.logProbability } }),
  }
}

/**
 * Draw samples by Gibbs sampling ({@link gibbsSamplingAlgorithm}) after a burn-in, keeping every `thin`-th sweep.
 * Observed variables appear in every sample with their observed state.
 *
 * @param model - The model.
 * @param key - The random key.
 * @param n - Number of samples to keep.
 * @param options - Evidence, starting state, burn-in, thinning and tracer.
 * @returns The samples.
 * @concept gibbs-sampling
 * @example
 * ```ts
 * import { bayesNet, gibbsSample } from 'omai-engine/pgm'
 * import { key } from 'omai-engine/random'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [1, 0, 0, 1] }])
 * gibbsSample(bn, key(0), 5, { evidence: { b: '1' } }).every((s) => s['a'] === 1) // => true
 * ```
 */
export function gibbsSample(model: DiscreteModel, key: Key, n: number, options: GibbsOptions = {}): Sample[] {
  const c = compile(model, options.evidence ?? {})
  const burnIn = options.burnIn ?? 100
  const thin = Math.max(1, options.thin ?? 1)
  const algorithm = gibbsSamplingAlgorithm(model, options)
  const samples: Sample[] = []
  const collecting: Algorithm<GibbsState> = {
    ...algorithm,
    step: (state, ctx) => {
      const next = algorithm.step(state, ctx)
      const sweep = ctx.t + 1
      if (sweep > burnIn && (sweep - burnIn) % thin === 0) {
        const s: Sample = { ...c.evidence }
        for (const [i, v] of c.names.entries()) s[v] = next.x[i] ?? 0
        samples.push(s)
      }
      return next
    },
  }
  runAlgorithm(collecting, {
    maxSteps: burnIn + n * thin,
    key,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const order = modelVariables(model).map((v) => v.name)
  return samples.map((s) => Object.fromEntries(order.map((v) => [v, s[v] ?? 0])))
}

/**
 * Approximate inference by sampling (pgmpy's ApproxInference): the posterior of the query variables estimated from
 * rejection samples, likelihood-weighted samples (Bayes networks) or Gibbs samples (any discrete model).
 *
 * @param model - The model.
 * @param query - The query variables.
 * @param options - The method (default `likelihoodWeighting` for Bayes networks, `gibbs` otherwise), number of
 * samples (default 10000), key, evidence and Gibbs options.
 * @returns The estimated posterior over the query, in query order.
 * @throws OmaiError with code `value` for rejection or likelihood weighting on a model that is not a Bayes network.
 * @concept sampling
 * @example
 * ```ts
 * import { bayesNet, approximateQuery, factorValues } from 'omai-engine/pgm'
 * import { key } from 'omai-engine/random'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [1, 0, 0, 1] }])
 * factorValues(approximateQuery(bn, ['a'], { evidence: { b: '1' }, key: key(0), n: 200 })) // => [0, 1]
 * ```
 */
export function approximateQuery(
  model: DiscreteModel,
  query: readonly string[],
  options: GibbsOptions & {
    readonly method?: 'rejection' | 'likelihoodWeighting' | 'gibbs'
    readonly n?: number
    readonly key: Key
  },
): Factor {
  const n = options.n ?? 10000
  const method = options.method ?? (model.kind === 'bayesNet' ? 'likelihoodWeighting' : 'gibbs')
  const evidence = options.evidence ?? {}
  for (const q of query) {
    if (q in evidenceIndices(model, evidence)) throw new OmaiError('value', `approximateQuery: ${q} is observed`)
  }
  if (method === 'gibbs') return sampleMarginal(model, gibbsSample(model, options.key, n, options), query)
  if (model.kind !== 'bayesNet') throw new OmaiError('value', `approximateQuery: ${method} needs a Bayes network`)
  for (const q of query) bayesNode(model, q)
  if (method === 'rejection')
    return sampleMarginal(model, rejectionSample(model, options.key, n, evidence).samples, query)
  const lw = likelihoodWeighting(model, options.key, n, evidence)
  return sampleMarginal(model, lw.samples, query, lw.logWeights)
}
