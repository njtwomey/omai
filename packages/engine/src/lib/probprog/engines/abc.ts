/**
 * Approximate Bayesian computation (ABC), likelihood-free inference for models that can be simulated but whose
 * likelihood cannot be evaluated: parameters are kept when the simulated data's summary statistics land close to the
 * observed ones. A `simulator` distribution, default summaries, rejection ABC, ABC-SMC with a shrinking tolerance, and
 * the engine that offers them to the planner.
 */

import { type Json, logSumExpOf, quantileOf, WhiteboxError } from '../../base/index.ts'
import { bijectors, type Distribution, defineDistribution, multivariateNormal, unavailable } from '../../dists/index.ts'
import { cholesky } from '../../numerics/linalg/index.ts'
import { choice, foldIn, type Key, normal as normalDraws } from '../../random/index.ts'
import {
  add,
  asTensor,
  broadcastTo,
  eye,
  fromStorage,
  item,
  matmul,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  stack,
  sum,
  type Tensor,
  toFlat,
  transpose,
  type Value,
} from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import { maskedLogProb } from '../_contract.ts'
import type { ProgramAnalysis } from '../analysis.ts'
import { type LatentLayout, latentLayout } from '../density.ts'
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
 * import { simulator } from 'whitebox-engine/probprog'
 * import { key, normal } from 'whitebox-engine/random'
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
export type SummaryStatistics = (observed: Readonly<Record<string, Value>>) => readonly number[]

/**
 * Default summary statistics: for each observed site, the mean, standard deviation and quartiles of its values.
 *
 * @param observed - Observed values by site.
 * @returns The summaries.
 * @concept approximate-bayesian-computation
 * @example
 * ```ts
 * import { defaultSummaries } from 'whitebox-engine/probprog'
 * import { tensor } from 'whitebox-engine/tensor'
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

function scaledDistance(a: readonly number[], b: readonly number[], scale: readonly number[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (((a[i] ?? 0) - (b[i] ?? 0)) / (scale[i] ?? 1)) ** 2
  return Math.sqrt(s)
}

function madScale(rows: readonly (readonly number[])[]): number[] {
  const d = rows[0]?.length ?? 0
  return Array.from({ length: d }, (_, j) => {
    const col = rows.map((r) => r[j] ?? 0)
    const med = quantileOf(col, 0.5)
    const mad = quantileOf(
      col.map((v) => Math.abs(v - med)),
      0.5,
    )
    return mad > 0 ? mad : 1
  })
}

interface Setting<D> {
  readonly observed: readonly string[]
  readonly target: readonly number[]
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
  if (observed.length === 0) throw new WhiteboxError('value', 'abc: the program has no observed sites')
  const latent = [...proto.sites.values()]
    .filter((s) => (s.type === 'sample' && !s.isObserved) || s.type === 'deterministic')
    .map((s) => s.name)
  const stats = summary ?? defaultSummaries
  const target = stats(observedValues(proto, observed))
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
 * import { abcRejection, observe, plate, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { item } from 'whitebox-engine/tensor'
 * const y = [0.5, 1.2, 1.6, 1.9, 2, 2.1, 2.4, 2.8, 3.5, 2]
 * const model = () => { const m = sample('m', normal(0, 3)); plate('n', y.length, () => observe('y', normal(m, 1), y)) }
 * Math.abs(item(abcRejection(model, undefined, { key: key(0), numSamples: 1000, quantile: 0.05 }).mean('m')) - 2) < 0.5 // => true
 * ```
 */
export function abcRejection<D = void>(model: Model<D>, data: D, options: AbcOptions & EngineRunOptions): Posterior {
  const n = options.numSamples ?? 2000
  const s = setting(model, data, options.summary, foldIn(options.key, 0))
  const traces: ProgramTrace[] = []
  const summaries: number[][] = []
  for (let i = 0; i < n; i++) {
    const t = s.simulate(foldIn(foldIn(options.key, 1), i))
    traces.push(t)
    summaries.push([...s.summary(observedValues(t, s.observed))])
  }
  const scale = madScale(summaries)
  const distances = summaries.map((x) => scaledDistance(x, s.target, scale))
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
    const rows: number[] = []
    let shape: readonly number[] = []
    for (const t of traces) {
      const site = t.sites.get(name)
      if (site === undefined || site.type === 'plate' || site.type === 'markov') continue
      shape = site.value.shape
      rows.push(...toFlat(primalOf(site.value)))
    }
    out[name] = fromStorage(Float64Array.from(rows), [traces.length, ...shape], 'float64')
  }
  return out
}

/** The state of {@link abcSmcAlgorithm}: one population of weighted particles. */
export interface AbcSmcState {
  readonly generation: number
  /** Particles on the unconstrained scale, row-major `[n, dim]`. */
  readonly particles: Float64Array
  readonly logWeights: Float64Array
  readonly distances: Float64Array
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
 * @throws WhiteboxError with code `value` for discrete latent sites.
 * @concept approximate-bayesian-computation
 * @example
 * ```ts
 * import { abcSmcAlgorithm, observe, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { runAlgorithm } from 'whitebox-engine/trace'
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
  let scale: number[] = []
  const unconstrain = (t: ProgramTrace): Float64Array => {
    const out = new Float64Array(dim)
    for (const l of layout) {
      const site = t.sites.get(l.name)
      if (site?.type === 'sample') out.set(toFlat(primalOf(l.bijector.inverse(site.value))), l.offset)
    }
    return out
  }
  const constrain = (u: Float64Array): Record<string, Value> => {
    const out: Record<string, Value> = {}
    const v = fromStorage(u, [dim], 'float64')
    for (const l of layout)
      out[l.name] = primalOf(
        l.bijector.forward(reshape(sliceAxis(v, 0, l.offset, l.offset + l.size), l.unconstrainedShape)),
      )
    return out
  }
  const logPrior = (t: ProgramTrace, u: Float64Array): number => {
    let lp = 0
    const v = fromStorage(u, [dim], 'float64')
    for (const l of layout) {
      const site = t.sites.get(l.name)
      if (site?.type !== 'sample') continue
      lp += item(primalOf(mul(sum(maskedLogProb(site)), site.scale)))
      const piece = reshape(sliceAxis(v, 0, l.offset, l.offset + l.size), l.unconstrainedShape)
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
        throw new WhiteboxError('value', 'abcSmc: discrete latent sites are not supported; use abcRejection')
      layout = lay.layout
      dim = layout.reduce((a, l) => a + l.size, 0)
      const traces: ProgramTrace[] = []
      const summaries: number[][] = []
      for (let i = 0; i < P; i++) {
        const t = st.simulate(foldIn(foldIn(key, 1), i))
        traces.push(t)
        summaries.push([...st.summary(observedValues(t, st.observed))])
      }
      scale = madScale(summaries)
      const particles = new Float64Array(P * dim)
      traces.forEach((t, i) => {
        particles.set(unconstrain(t), i * dim)
      })
      const distances = Float64Array.from(summaries, (x) => scaledDistance(x, st.target, scale))
      return {
        generation: 0,
        particles,
        logWeights: new Float64Array(P).fill(-Math.log(P)),
        distances,
        epsilon: Number.POSITIVE_INFINITY,
        simulations: P,
        acceptance: 1,
        traces,
      }
    },
    step: (state, ctx) => {
      const st = s
      if (st === undefined) throw new WhiteboxError('value', 'abcSmc: not initialised')
      const eps = quantileOf(state.distances, q)
      const w = Array.from(state.logWeights, (v) => Math.exp(v))
      // Twice the weighted covariance of the population.
      const mean = new Float64Array(dim)
      for (let i = 0; i < P; i++)
        for (let j = 0; j < dim; j++) mean[j] = (mean[j] ?? 0) + (w[i] ?? 0) * (state.particles[i * dim + j] ?? 0)
      const cov = new Float64Array(dim * dim)
      for (let i = 0; i < P; i++)
        for (let a = 0; a < dim; a++)
          for (let b = 0; b < dim; b++)
            cov[a * dim + b] =
              (cov[a * dim + b] ?? 0) +
              2 *
                (w[i] ?? 0) *
                ((state.particles[i * dim + a] ?? 0) - (mean[a] ?? 0)) *
                ((state.particles[i * dim + b] ?? 0) - (mean[b] ?? 0))
      const kernelCov = primalOf(add(fromStorage(cov, [dim, dim], 'float64'), mul(eye(dim), 1e-10)))
      const chol = primalOf(cholesky(kernelCov))
      const mixture = multivariateNormal(fromStorage(state.particles, [P, dim], 'float64'), { scaleTril: chol })
      const particles = new Float64Array(P * dim)
      const distances = new Float64Array(P)
      const logWeights = new Float64Array(P)
      const traces: ProgramTrace[] = []
      let accepted = 0
      let sims = 0
      while (accepted < P && sims < cap) {
        const k = foldIn(ctx.key, sims)
        sims += 1
        const pick = toFlat(choice(foldIn(k, 0), P, 1, { weights: w }))[0] ?? 0
        const centre = state.particles.subarray(pick * dim, (pick + 1) * dim)
        const noise = toFlat(primalOf(matmul(normalDraws(foldIn(k, 1), [dim]), transpose(chol))))
        const u = Float64Array.from(centre, (c, j) => c + (noise[j] ?? 0))
        const t = st.simulate(foldIn(k, 2), constrain(u))
        const d = scaledDistance([...st.summary(observedValues(t, st.observed))], st.target, scale)
        if (!(d <= eps)) continue
        const lp = logPrior(t, u)
        if (!Number.isFinite(lp)) continue
        const lk = toFlat(primalOf(mixture.logProb(fromStorage(u, [dim], 'float64'))))
        const denom = logSumExpOf(lk.map((v, i) => v + (state.logWeights[i] ?? 0)))
        particles.set(u, accepted * dim)
        distances[accepted] = d
        logWeights[accepted] = lp - denom
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
      const z = logSumExpOf(logWeights)
      return {
        generation: state.generation + 1,
        particles,
        logWeights: Float64Array.from(logWeights, (v) => v - z),
        distances,
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
        ess: Math.exp(-logSumExpOf(Array.from(state.logWeights, (v) => 2 * v))),
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
 * import { abcSmc, observe, plate, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { item } from 'whitebox-engine/tensor'
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
  const st = r.state
  const first = st.traces[0]
  const names =
    first === undefined
      ? []
      : [...first.sites.values()]
          .filter((x) => (x.type === 'sample' && !x.isObserved) || x.type === 'deterministic')
          .map((x) => x.name)
  return weightedPosterior(stack1(st.traces, names), st.logWeights, {
    engine: 'abc',
    status: r.status,
    diagnostics: {
      method: 'abc-smc',
      generations: st.generation,
      epsilon: Number.isFinite(st.epsilon) ? st.epsilon : null,
      acceptance: st.acceptance,
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
 * import { abcEngine } from 'whitebox-engine/probprog'
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
