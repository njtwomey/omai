/**
 * Approximate Bayesian computation (ABC), likelihood-free inference for models that can be simulated but whose
 * likelihood cannot be evaluated: parameters are kept when the simulated data's summary statistics land close to the
 * observed ones. A `simulator` distribution, default summaries, rejection ABC, ABC-SMC with a shrinking tolerance, and
 * the engine that offers them to the planner.
 */

import { type Json, OmaiError, quantileOf } from '../../base/index.ts'
import { bijectors, type Distribution, defineDistribution, multivariateNormal, unavailable } from '../../dists/index.ts'
import { cholesky } from '../../numerics/linalg/index.ts'
import { choice, foldIn, type Key, normal as normalDraws } from '../../random/index.ts'
import {
  abs,
  add,
  any,
  asTensor,
  broadcastTo,
  cast,
  div,
  exp,
  eye,
  full,
  greater,
  isnan,
  item,
  logsumexp,
  matmul,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  sort,
  sqrt,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  toFlat,
  transpose,
  type Value,
  where,
  zeros,
} from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import { maskedLogProb } from '../_contract.ts'
import type { ProgramAnalysis } from '../analysis.ts'
import { type LatentLayout, latentLayout, packLatents } from '../density.ts'
import { handle, type ProgramTrace, seed, traceOf, uncondition } from '../handlers.ts'
import { type Posterior, samplePosterior, weightedPosterior } from '../posterior.ts'
import type { Message, Model } from '../primitives.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine } from './engine.ts'

/**
 * A simulator as a distribution: draws come from `simulate(key)` (repeated over sample and batch shapes), and there is
 * no density, so a program that observes it has no likelihood and only likelihood-free engines (ABC) apply.
 *
 * @param name - The simulator's name.
 * @param simulate - One draw from a key.
 * @param options - The event shape of a draw and the support (default the real line).
 * @returns The distribution.
 * @concept approximate-bayesian-computation
 * @example
 * ```ts
 * import { simulator } from 'omai-engine/probprog'
 * import { key, normal } from 'omai-engine/random'
 * const sim = simulator('noisy', (k) => normal(k, [2]), { eventShape: [2] })
 * sim.sample(key(0), [3]).shape // => [3, 2]
 * ```
 */
export function simulator(
  name: string,
  simulate: (key: Key) => Value,
  options: { readonly eventShape?: readonly number[]; readonly support?: Distribution['support'] } = {},
): Distribution {
  const eventShape = options.eventShape ?? []
  const draws = (key: Key, shape: readonly number[]): Value => {
    const n = shape.reduce((a, b) => a * b, 1)
    const rows = Array.from({ length: n }, (_, i) => broadcastTo(simulate(foldIn(key, i)), eventShape))
    return n === 0 ? reshape(asTensor([]), [...shape, ...eventShape]) : reshape(stack(rows), [...shape, ...eventShape])
  }
  return defineDistribution({
    name,
    params: {},
    support: options.support ?? { kind: 'real' },
    batchShape: [],
    eventShape,
    logProb: () => unavailable(name, 'logProb (a simulator has no density)'),
    sample: draws,
    mean: () => unavailable(name, 'mean'),
    variance: () => unavailable(name, 'variance'),
    mode: () => unavailable(name, 'mode'),
    entropy: () => unavailable(name, 'entropy'),
  })
}

/** Summary statistics: numbers describing the observed sites of a run. */
export type SummaryStatistics = (observed: Readonly<Record<string, Value>>) => TensorLike

/**
 * Default summary statistics: for each observed site, the mean, standard deviation and quartiles of its values.
 *
 * @param observed - Observed values by site.
 * @returns The summaries.
 * @concept approximate-bayesian-computation
 * @example
 * ```ts
 * import { defaultSummaries } from 'omai-engine/probprog'
 * import { tensor } from 'omai-engine/tensor'
 * defaultSummaries({ y: tensor([1, 2, 3, 4, 5]) }) // => [3, 1.5811388300841898, 2, 3, 4]
 * ```
 */
export function defaultSummaries(observed: Readonly<Record<string, Value>>): number[] {
  const out: number[] = []
  for (const name of Object.keys(observed).sort()) {
    const v = toFlat(primalOf(observed[name] as Value))
    const n = v.length
    const m = v.reduce((a, b) => a + b, 0) / Math.max(1, n)
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, n - 1))
    out.push(m, sd, quantileOf(v, 0.25), quantileOf(v, 0.5), quantileOf(v, 0.75))
  }
  return out
}

function observedValues(t: ProgramTrace, names: readonly string[]): Record<string, Value> {
  const out: Record<string, Value> = {}
  for (const name of names) {
    const s = t.sites.get(name)
    if (s?.type === 'sample') out[name] = s.value
  }
  return out
}

/** One summary vector as a rank-1 tensor (a summary function may return a list of numbers or a tensor). */
function summaryRow(x: TensorLike): Tensor {
  return reshape(asTensor(x), [-1]) as Tensor
}

/** The distance from each row of an `[n, d]` summary matrix to the target, each summary divided by its scale. */
function scaledDistances(rows: Tensor, target: Tensor, scale: Tensor): Tensor {
  return sqrt(sum(square(div(sub(rows, target), scale)), 1)) as Tensor
}

/** The median of each column of an `[n, d]` tensor with the interpolation of `quantileOf` (a NaN in a column gives NaN). */
function columnMedians(x: Tensor): Tensor {
  const n = x.shape[0] ?? 0
  const h = (n - 1) * 0.5
  const lo = Math.floor(h)
  const sorted = sort(x, 0)
  const at = (i: number): Value => reshape(sliceAxis(sorted, 0, i, i + 1), [-1])
  const below = at(lo)
  const middle = add(below, mul(sub(at(Math.min(n - 1, lo + 1)), below), h - lo))
  return where(any(isnan(x), 0), Number.NaN, middle) as Tensor
}

/** The median absolute deviation of each column of an `[n, d]` summary matrix (1 where it is 0), the scale of the distance. */
function madScale(rows: Tensor): Tensor {
  const mad = columnMedians(abs(sub(rows, columnMedians(rows))) as Tensor)
  return where(greater(mad, 0), mad, 1) as Tensor
}

interface Setting<D> {
  readonly observed: readonly string[]
  readonly target: Tensor
  readonly summary: SummaryStatistics
  readonly simulate: (key: Key, values?: Readonly<Record<string, Value>>) => ProgramTrace
  readonly latent: readonly string[]
  readonly model: Model<D>
}

function setting<D = void>(model: Model<D>, data: D, summary: SummaryStatistics | undefined, key: Key): Setting<D> {
  const proto = traceOf(seed(model, key), data)
  const observed = [...proto.sites.values()]
    .filter((s) => s.type === 'sample' && s.isObserved && !s.isFactor)
    .map((s) => s.name)
  if (observed.length === 0) throw new OmaiError('value', 'abc: the program has no observed sites')
  const latent = [...proto.sites.values()]
    .filter((s) => (s.type === 'sample' && !s.isObserved) || s.type === 'deterministic')
    .map((s) => s.name)
  const stats = summary ?? defaultSummaries
  const target = summaryRow(stats(observedValues(proto, observed)))
  const simulate = (k: Key, values?: Readonly<Record<string, Value>>): ProgramTrace => {
    let m = uncondition(model)
    if (values !== undefined) {
      const fixed = values
      m = handle(m, () => ({
        process: (msg: Message) => {
          if (msg.type === 'sample' && !msg.isObserved && fixed[msg.name] !== undefined) msg.value = fixed[msg.name]
        },
      }))
    }
    return traceOf(seed(m, k), data)
  }
  return { observed, target, summary: stats, simulate, latent, model }
}

/** Options of {@link abcRejection}. */
export interface AbcOptions {
  /** Simulations (default 2000). */
  readonly numSamples?: number
  /** Keep this fraction of the closest simulations (default 0.05) unless `epsilon` is given. */
  readonly quantile?: number
  /** Keep simulations within this distance. */
  readonly epsilon?: number
  readonly summary?: SummaryStatistics
}

/**
 * Rejection ABC (Pritchard et al. 1999): simulate parameters from the prior and data from the program, keep the
 * draws whose summary statistics fall closest to the observed ones (distance scaled per summary by its median absolute
 * deviation across simulations), and return them as equally weighted draws.
 *
 * @param model - The program (observed sites may be simulators).
 * @param data - Its data.
 * @param options - Simulations, acceptance fraction or tolerance, summaries, key.
 * @returns The posterior of accepted draws; `diagnostics.epsilon` is the tolerance used.
 * @concept approximate-bayesian-computation
 * @example
 * ```ts
 * import { abcRejection, observe, plate, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * const y = [0.5, 1.2, 1.6, 1.9, 2, 2.1, 2.4, 2.8, 3.5, 2]
 * const model = () => { const m = sample('m', normal(0, 3)); plate('n', y.length, () => observe('y', normal(m, 1), y)) }
 * Math.abs(item(abcRejection(model, undefined, { key: key(0), numSamples: 1000, quantile: 0.05 }).mean('m')) - 2) < 0.5 // => true
 * ```
 */
export function abcRejection<D = void>(model: Model<D>, data: D, options: AbcOptions & EngineRunOptions): Posterior {
  const n = options.numSamples ?? 2000
  const s = setting(model, data, options.summary, foldIn(options.key, 0))
  const traces: ProgramTrace[] = []
  const rows: Tensor[] = []
  for (let i = 0; i < n; i++) {
    const t = s.simulate(foldIn(foldIn(options.key, 1), i))
    traces.push(t)
    rows.push(summaryRow(s.summary(observedValues(t, s.observed))))
  }
  const summaries = n === 0 ? zeros([0]) : (stack(rows) as Tensor)
  const distances = n === 0 ? [] : toFlat(scaledDistances(summaries, s.target, madScale(summaries)))
  const eps = options.epsilon ?? quantileOf(distances, options.quantile ?? 0.05)
  const kept = traces.filter((_, i) => (distances[i] ?? Number.POSITIVE_INFINITY) <= eps)
  if (kept.length === 0)
    return samplePosterior(
      {},
      { engine: 'abc', status: 'failed', diagnostics: { reason: 'no simulation within epsilon', epsilon: eps } },
    )
  return samplePosterior(stack1(kept, s.latent), {
    engine: 'abc',
    status: 'completed',
    diagnostics: { method: 'rejection', epsilon: eps, accepted: kept.length, simulations: n },
  })
}

function stack1(traces: readonly ProgramTrace[], names: readonly string[]): Record<string, Tensor> {
  const out: Record<string, Tensor> = {}
  for (const name of names) {
    const values = traces.flatMap((t) => {
      const site = t.sites.get(name)
      return site === undefined || site.type === 'plate' || site.type === 'markov' ? [] : [primalOf(site.value)]
    })
    out[name] = values.length === 0 ? zeros([0]) : (cast(stack(values), 'float64') as Tensor)
  }
  return out
}

/** The state of {@link abcSmcAlgorithm}: one population of weighted particles. */
export interface AbcSmcState {
  readonly generation: number
  /** Particles on the unconstrained scale, row-major `[n, dim]`. */
  readonly particles: Tensor
  /** Normalised log weights, a vector of one per particle. */
  readonly logWeights: Tensor
  /** The distance of each particle's simulation to the observed summaries, a vector of one per particle. */
  readonly distances: Tensor
  readonly epsilon: number
  readonly simulations: number
  readonly acceptance: number
  readonly traces: readonly ProgramTrace[]
}

/** Options of {@link abcSmcAlgorithm}. */
export interface AbcSmcOptions {
  readonly numParticles?: number
  /** Each generation's tolerance is this quantile of the previous generation's distances (default 0.5). */
  readonly quantile?: number
  /** Stop when the tolerance falls below this (default 0). */
  readonly minEpsilon?: number
  /** Stop when a generation accepts less than this fraction of its simulations (default 0.01). */
  readonly minAcceptance?: number
  /** The most simulations per generation (default 100 per particle). */
  readonly maxSimulations?: number
  readonly summary?: SummaryStatistics
}

/**
 * ABC-SMC (Toni et al. 2009; Beaumont et al. 2009, population Monte Carlo ABC) as a traced algorithm, one step per
 * generation: perturb particles drawn from the previous population with a Gaussian kernel of twice their weighted
 * covariance (on the unconstrained scale), simulate, accept within the shrinking tolerance, and weight by
 * prior / mixture-of-kernels density. Generations stop at the minimum tolerance or acceptance rate, and every generation
 * has a simulation cap (a generation that hits it ends the run with status `stopped`).
 *
 * @param model - The program (continuous latent sites).
 * @param data - Its data.
 * @param options - Particles, quantile, stopping rules and summaries.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for discrete latent sites.
 * @concept approximate-bayesian-computation
 * @example
 * ```ts
 * import { abcSmcAlgorithm, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 3)); observe('y', normal(m, 1), 1) }
 * runAlgorithm(abcSmcAlgorithm(model, undefined, { numParticles: 50 }), { seed: 0, maxSteps: 3 }).state.generation // => 3
 * ```
 */
export function abcSmcAlgorithm<D = void>(
  model: Model<D>,
  data: D,
  options: AbcSmcOptions = {},
): Algorithm<AbcSmcState> {
  const P = options.numParticles ?? 500
  const q = options.quantile ?? 0.5
  const minEps = options.minEpsilon ?? 0
  const minAcc = options.minAcceptance ?? 0.01
  const cap = options.maxSimulations ?? 100 * P
  let s: Setting<D> | undefined
  let layout: LatentLayout[] = []
  let dim = 0
  let scale: Tensor = zeros([0])
  const unconstrain = (t: ProgramTrace): Tensor => {
    if (dim === 0) return zeros([0])
    const points = layout.flatMap((l) => {
      const site = t.sites.get(l.name)
      return site?.type === 'sample' ? [[l.name, l.bijector.inverse(site.value)] as const] : []
    })
    return primalOf(packLatents(layout, Object.fromEntries(points))) as Tensor
  }
  const constrain = (u: Tensor): Record<string, Value> => {
    const out: Record<string, Value> = {}
    for (const l of layout)
      out[l.name] = primalOf(
        l.bijector.forward(reshape(sliceAxis(u, 0, l.offset, l.offset + l.size), l.unconstrainedShape)),
      )
    return out
  }
  const logPrior = (t: ProgramTrace, u: Tensor): number => {
    let lp = 0
    for (const l of layout) {
      const site = t.sites.get(l.name)
      if (site?.type !== 'sample') continue
      lp += item(primalOf(mul(sum(maskedLogProb(site)), site.scale)))
      const piece = reshape(sliceAxis(u, 0, l.offset, l.offset + l.size), l.unconstrainedShape)
      lp += item(primalOf(sum(bijectors.bijectorForSupport(site.fn.support).forwardLogDetJacobian(piece))))
    }
    return lp
  }
  return {
    name: 'abcSmc',
    config: { numParticles: P, quantile: q, minEpsilon: minEps, minAcceptance: minAcc } as Json,
    schema: channels({
      metrics: {
        epsilon: { description: 'the tolerance of the generation', direction: 'lower' },
        acceptance: { description: 'fraction of simulations accepted' },
        ess: { description: 'effective sample size of the population' },
      },
    }),
    init: (key) => {
      const st = setting(model, data, options.summary, foldIn(key, 0))
      s = st
      const proto = traceOf(seed(model, foldIn(key, 0)), data)
      const lay = latentLayout(proto)
      if (lay.enumerated.length > 0)
        throw new OmaiError('value', 'abcSmc: discrete latent sites are not supported; use abcRejection')
      layout = lay.layout
      dim = layout.reduce((a, l) => a + l.size, 0)
      const traces: ProgramTrace[] = []
      const rows: Tensor[] = []
      for (let i = 0; i < P; i++) {
        const t = st.simulate(foldIn(foldIn(key, 1), i))
        traces.push(t)
        rows.push(summaryRow(st.summary(observedValues(t, st.observed))))
      }
      const summaries = stack(rows) as Tensor
      scale = madScale(summaries)
      return {
        generation: 0,
        particles: stack(traces.map(unconstrain)) as Tensor,
        logWeights: full([P], -Math.log(P)),
        distances: scaledDistances(summaries, st.target, scale),
        epsilon: Number.POSITIVE_INFINITY,
        simulations: P,
        acceptance: 1,
        traces,
      }
    },
    step: (state, ctx) => {
      const st = s
      if (st === undefined) throw new OmaiError('value', 'abcSmc: not initialised')
      const eps = quantileOf(toFlat(state.distances), q)
      const w = exp(state.logWeights) as Tensor
      // Twice the weighted covariance of the population.
      const mean = sum(mul(reshape(w, [P, 1]), state.particles), 0)
      const centred = sub(state.particles, mean)
      const weighted = mul(reshape(mul(w, 2), [P, 1]), centred)
      const cov = sum(mul(reshape(weighted, [P, dim, 1]), reshape(centred, [P, 1, dim])), 0)
      const kernelCov = primalOf(add(cov, mul(eye(dim), 1e-10)))
      const chol = primalOf(cholesky(kernelCov))
      const mixture = multivariateNormal(state.particles, { scaleTril: chol })
      const points: Tensor[] = []
      const distances: number[] = []
      const logWeights: number[] = []
      const traces: ProgramTrace[] = []
      let accepted = 0
      let sims = 0
      while (accepted < P && sims < cap) {
        const k = foldIn(ctx.key, sims)
        sims += 1
        const pick = toFlat(choice(foldIn(k, 0), P, 1, { weights: w }))[0] ?? 0
        const centre = reshape(sliceAxis(state.particles, 0, pick, pick + 1), [dim])
        const u = add(centre, matmul(normalDraws(foldIn(k, 1), [dim]), transpose(chol))) as Tensor
        const t = st.simulate(foldIn(k, 2), constrain(u))
        const summary = reshape(summaryRow(st.summary(observedValues(t, st.observed))), [1, -1]) as Tensor
        const d = item(scaledDistances(summary, st.target, scale))
        if (!(d <= eps)) continue
        const lp = logPrior(t, u)
        if (!Number.isFinite(lp)) continue
        const lk = primalOf(mixture.logProb(u))
        const denom = item(logsumexp(add(lk, state.logWeights)))
        points.push(u)
        distances.push(d)
        logWeights.push(lp - denom)
        traces.push(t)
        accepted += 1
      }
      if (accepted < P)
        return {
          ...state,
          generation: state.generation + 1,
          epsilon: eps,
          simulations: sims,
          acceptance: accepted / sims,
        }
      const raw = tensor(logWeights)
      return {
        generation: state.generation + 1,
        particles: stack(points) as Tensor,
        logWeights: sub(raw, item(logsumexp(raw))) as Tensor,
        distances: tensor(distances),
        epsilon: eps,
        simulations: sims,
        acceptance: accepted / sims,
        traces,
      }
    },
    done: (state) => {
      if (state.generation === 0) return undefined
      if (state.traces.length < P || state.acceptance === 0) return 'stopped'
      if (state.epsilon <= minEps || state.acceptance < minAcc) return 'converged'
      return undefined
    },
    observe: (state) => ({
      metrics: {
        epsilon: Number.isFinite(state.epsilon) ? state.epsilon : -1,
        acceptance: state.acceptance,
        ess: Math.exp(-item(logsumexp(mul(state.logWeights, 2)))),
      },
    }),
  }
}

/**
 * Run ABC-SMC for at most `numGenerations` generations and return the final population as a weighted posterior.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - ABC-SMC options, generations (default 8), key and tracing.
 * @returns The weighted posterior; `diagnostics.epsilon` is the final tolerance.
 * @concept approximate-bayesian-computation
 * @example
 * ```ts
 * import { abcSmc, observe, plate, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * const y = [0.5, 1.2, 1.6, 1.9, 2, 2.1, 2.4, 2.8, 3.5, 2]
 * const model = () => { const m = sample('m', normal(0, 3)); plate('n', y.length, () => observe('y', normal(m, 1), y)) }
 * Math.abs(item(abcSmc(model, undefined, { key: key(0), numParticles: 200, numGenerations: 4 }).mean('m')) - 2) < 0.5 // => true
 * ```
 */
export function abcSmc<D = void>(
  model: Model<D>,
  data: D,
  options: AbcSmcOptions & EngineRunOptions & { readonly numGenerations?: number },
): Posterior {
  const r = runAlgorithm(abcSmcAlgorithm(model, data, options), {
    key: options.key,
    maxSteps: options.numGenerations ?? 8,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  })
  return abcSmcPosterior(abcSmcDraws(r.state.traces), r.state, r.status)
}

/**
 * The latent and deterministic sites of an ABC-SMC population's simulations, stacked `[n, ...shape]`: what a population
 * is as draws, and what a state needs to keep of its traces to be read as a posterior.
 *
 * @param traces - The traces of the population (`AbcSmcState.traces`).
 * @returns The draws by site.
 * @concept approximate-bayesian-computation
 * @example
 * ```ts
 * import { abcSmcAlgorithm, abcSmcDraws, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 3)); observe('y', normal(m, 1), 1) }
 * abcSmcDraws(runAlgorithm(abcSmcAlgorithm(model, undefined, { numParticles: 20 }), { seed: 0, maxSteps: 1 }).state.traces)['m']?.shape // => [20]
 * ```
 */
export function abcSmcDraws(traces: readonly ProgramTrace[]): Record<string, Tensor> {
  const first = traces[0]
  const names =
    first === undefined
      ? []
      : [...first.sites.values()]
          .filter((x) => (x.type === 'sample' && !x.isObserved) || x.type === 'deterministic')
          .map((x) => x.name)
  return stack1(traces, names)
}

/**
 * The weighted posterior of an ABC-SMC population: its draws (see {@link abcSmcDraws}), weights, the generation reached,
 * the tolerance and the acceptance rate.
 *
 * @param draws - The population's draws.
 * @param state - The generation, weights, tolerance and acceptance (`AbcSmcState` has them).
 * @param status - How the run ended.
 * @returns The weighted posterior.
 * @concept approximate-bayesian-computation
 * @example
 * ```ts
 * import { abcSmcPosterior } from 'omai-engine/probprog'
 * import { tensor } from 'omai-engine/tensor'
 * abcSmcPosterior({ m: tensor([0, 1]) }, { logWeights: Float64Array.of(-0.7, -0.7), generation: 1, epsilon: 0.5, acceptance: 0.4 }, 'completed').engine // => 'abc'
 * ```
 */
export function abcSmcPosterior(
  draws: Readonly<Record<string, Tensor>>,
  state: Pick<AbcSmcState, 'generation' | 'epsilon' | 'acceptance'> & { readonly logWeights: TensorLike },
  status: Posterior['status'],
): Posterior {
  return weightedPosterior(draws, state.logWeights, {
    engine: 'abc',
    status,
    diagnostics: {
      method: 'abc-smc',
      generations: state.generation,
      epsilon: Number.isFinite(state.epsilon) ? state.epsilon : null,
      acceptance: state.acceptance,
    },
  })
}

/**
 * The ABC engine: likelihood-free inference for simulator programs (ABC-SMC for continuous latent sites, rejection
 * ABC otherwise). It applies to any program with observations but ranks last, so the planner uses it only when no
 * likelihood exists.
 *
 * @param options - Particles, generations and summaries.
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { abcEngine } from 'omai-engine/probprog'
 * abcEngine().name // => 'abc'
 * ```
 */
export function abcEngine(
  options: AbcSmcOptions & AbcOptions & { readonly numGenerations?: number } = {},
): InferenceEngine {
  return {
    name: 'abc',
    form: 'weighted',
    description: 'approximate Bayesian computation: rejection ABC and ABC-SMC with summary statistics',
    applies: (a: ProgramAnalysis) => {
      if (a.latent.length === 0) return feasibility(false, 'likelihood-free', 0, 'no latent sites')
      if (a.observed.length === 0) return feasibility(false, 'likelihood-free', 0, 'no observed sites')
      const P = options.numParticles ?? 500
      return feasibility(
        true,
        'likelihood-free',
        P * (options.numGenerations ?? 8) * 20 * Math.max(1, a.observations),
        a.hasLikelihood
          ? 'simulation (a likelihood exists, so other engines are better)'
          : `simulators: ${a.simulators.join(', ')}`,
      )
    },
    run: <D>(model: Model<D>, data: D, a: ProgramAnalysis, run: EngineRunOptions) =>
      a.discrete.sites.length === 0 && a.continuous.length > 0
        ? abcSmc(model, data, { ...options, ...run })
        : abcRejection(model, data, { ...options, ...run }),
  }
}
