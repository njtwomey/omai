/**
 * Sequential Monte Carlo on probabilistic programs: a bootstrap particle filter over a program's markov loop (weighted
 * particle paths and an evidence estimate) and adaptive-tempering SMC, which moves particles from the prior to the
 * posterior, for programs without markov structure. Also the engine for the planner.
 */

import { type Json, logSumExpOf, OmaiError } from '../../base/index.ts'
import { bijectors } from '../../dists/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import { type ChainProposal, proposal, type ResamplingScheme, resample, smcSampler } from '../../sampling/index.ts'
import { fromStorage, item, mul, primalOf, reshape, sliceAxis, sum, toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import { maskedLogProb, observedLogProb } from '../_contract.ts'
import type { ProgramAnalysis } from '../analysis.ts'
import { constrainDraws, latentLayout, potentialEnergy, type UnconstrainedModel } from '../density.ts'
import { handle, type ProgramTrace, seed, traceOf } from '../handlers.ts'
import { type Posterior, samplePosterior, weightedPosterior } from '../posterior.ts'
import type { Message, Model } from '../primitives.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine } from './engine.ts'
import { stackSiteValues } from './importance.ts'

/** The state of {@link particleFilterAlgorithm}. */
export interface ParticleFilterState {
  readonly t: number
  readonly carries: readonly unknown[]
  /** Normalised log weights. */
  readonly logWeights: Float64Array
  readonly logEvidence: number
  /** Each particle's ancestral path: one trace per step. */
  readonly paths: readonly (readonly ProgramTrace[])[]
  readonly ess: number
  readonly resampled: boolean
  readonly resamplings: number
}

/** Options of {@link particleFilterAlgorithm}. */
export interface ParticleFilterOptions {
  /** Particles (default 500). */
  readonly numParticles?: number
  /** Resample when the ESS falls below this fraction of the particles (default 0.5; 1 resamples every step). */
  readonly resampleThreshold?: number
  readonly scheme?: ResamplingScheme
}

interface Loop {
  readonly name: string
  readonly length: number
  readonly body: (t: number, carry: unknown) => unknown
  readonly init: unknown
  /** log density of the observed sites before the loop. */
  readonly before: number
}

function discover<D = void>(model: Model<D>, data: D): Loop {
  let found: Loop | undefined
  const t = traceOf(
    handle(model, () => ({
      process: (msg: Message) => {
        if (msg.type !== 'markov' || found !== undefined || msg.body === undefined) return
        found = { name: msg.name, length: msg.length ?? 0, body: msg.body, init: msg.carry, before: 0 }
        msg.handled = true
        msg.result = msg.carry
      },
    })),
    data,
  )
  if (found === undefined) throw new OmaiError('value', 'particleFilter: the program has no markov loop')
  for (const s of t.sites.values())
    if (s.type === 'sample' && !s.isObserved)
      throw new OmaiError('value', `particleFilter: latent site ${s.name} is outside the markov loop`)
  return { ...found, before: observedLogProb(t) }
}

function normalise(lw: Float64Array): Float64Array {
  const z = logSumExpOf(lw)
  return Float64Array.from(lw, (v) => v - z)
}

/**
 * A bootstrap particle filter over a program's markov loop, as a traced algorithm (one step per time step): each
 * particle runs the loop body from its own carry with prior draws, the observed sites of the step weight it, the
 * evidence accumulates, and particles are resampled when the ESS falls below the threshold. Latent sites must all be
 * inside the loop (the filter carries nothing else). A step where every particle has zero weight ends the run with
 * status `failed`. For a state-space model written as functions over tensors rather than as a program, see
 * `bootstrapParticleAlgorithm` of `dynamics/filtering`; the two share the idea, not the model.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Particles, resampling threshold and scheme.
 * @returns The algorithm.
 * @throws OmaiError with code `value` when the program has no markov loop or latent sites outside it.
 * @concept sequential-monte-carlo
 * @example
 * ```ts
 * import { markov, observe, particleFilterAlgorithm, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import type { Value } from 'omai-engine/tensor'
 * const walk = () => markov('t', 3, (t, x: Value | number) => { const z = sample(`x${t}`, normal(x, 1)); observe(`y${t}`, normal(z, 1), t); return z }, 0)
 * runAlgorithm(particleFilterAlgorithm(walk, undefined, { numParticles: 50 }), { seed: 0, maxSteps: 10 }).steps // => 3
 * ```
 */
export function particleFilterAlgorithm<D = void>(
  model: Model<D>,
  data: D,
  options: ParticleFilterOptions = {},
): Algorithm<ParticleFilterState> {
  const P = options.numParticles ?? 500
  const threshold = options.resampleThreshold ?? 0.5
  if (!(Number.isInteger(P) && P >= 1)) throw new OmaiError('value', 'particleFilter: numParticles must be positive')
  const loop = discover(model, data)
  return {
    name: 'particleFilter',
    config: { numParticles: P, resampleThreshold: threshold, scheme: options.scheme ?? 'systematic' } as Json,
    schema: channels({
      metrics: {
        ess: { description: 'effective sample size after the step' },
        logEvidence: { description: 'running estimate of log p(y_0..t)' },
        resampled: { description: '1 when the particles were resampled' },
      },
    }),
    init: () => ({
      t: 0,
      carries: new Array<unknown>(P).fill(loop.init),
      logWeights: new Float64Array(P).fill(-Math.log(P)),
      logEvidence: loop.before,
      paths: Array.from({ length: P }, () => []),
      ess: P,
      resampled: false,
      resamplings: 0,
    }),
    step: (s, ctx) => {
      const t = s.t
      const traces: ProgramTrace[] = []
      const incr = new Float64Array(P)
      for (let p = 0; p < P; p++) {
        const carry = s.carries[p]
        const stepModel = (): unknown => loop.body(t, carry)
        const tr = traceOf(seed(stepModel, foldIn(ctx.key, p)), undefined)
        traces.push(tr)
        const v = observedLogProb(tr)
        incr[p] = Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
      }
      const joint = Float64Array.from(s.logWeights, (w, p) => w + (incr[p] ?? 0))
      const step = logSumExpOf(joint)
      const logEvidence = s.logEvidence + step
      if (!Number.isFinite(step)) return { ...s, t: t + 1, logEvidence, logWeights: joint, ess: 0, resampled: false }
      let logWeights = normalise(joint)
      let carries = traces.map((tr) => tr.returnValue)
      let paths = s.paths.map((path, p) => [...path, traces[p] as ProgramTrace])
      const ess = Math.exp(-logSumExpOf(logWeights.map((v) => 2 * v)))
      let resampled = false
      if (ess < threshold * P) {
        const idx = resample(foldIn(ctx.key, P), Array.from(logWeights, Math.exp), {
          scheme: options.scheme ?? 'systematic',
        })
        carries = Array.from(idx, (i) => carries[i])
        paths = Array.from(idx, (i) => paths[i] ?? [])
        logWeights = new Float64Array(P).fill(-Math.log(P))
        resampled = true
      }
      return {
        t: t + 1,
        carries,
        logWeights,
        logEvidence,
        paths,
        ess,
        resampled,
        resamplings: s.resamplings + (resampled ? 1 : 0),
      }
    },
    done: (s) => {
      if (s.ess === 0 && s.t > 0) return 'failed'
      return s.t >= loop.length ? 'completed' : undefined
    },
    observe: (s) => ({ metrics: { ess: s.ess, logEvidence: s.logEvidence, resampled: s.resampled ? 1 : 0 } }),
  }
}

/**
 * Filter a markov program with the bootstrap particle filter and return the weighted particle paths as the posterior
 * (smoothing by genealogy), with the evidence estimate.
 *
 * @param model - The program (every latent site inside one markov loop).
 * @param data - Its data.
 * @param options - Particle filter options, key and tracing.
 * @returns The weighted posterior over the paths.
 * @concept sequential-monte-carlo
 * @example
 * ```ts
 * import { markov, observe, particleFilter, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import type { Value } from 'omai-engine/tensor'
 * const walk = () => markov('t', 3, (t, x: Value | number) => { const z = sample(`x${t}`, normal(x, 1)); observe(`y${t}`, normal(z, 1), t); return z }, 0)
 * particleFilter(walk, undefined, { key: key(0), numParticles: 100 }).draws?.['x2']?.shape // => [100]
 * ```
 */
export function particleFilter<D = void>(
  model: Model<D>,
  data: D,
  options: ParticleFilterOptions & EngineRunOptions,
): Posterior {
  const alg = particleFilterAlgorithm(model, data, options)
  const r = runAlgorithm(alg, {
    key: options.key,
    maxSteps: 1_000_000,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  })
  return particleFilterPosterior(r.state, r.status)
}

/**
 * The weighted posterior of a particle filter's state: the particles' ancestral paths (the sites of every step's trace
 * stacked by site) with the normalised weights, the evidence estimate and the filter's diagnostics. A state in the middle
 * of the run gives the filtering posterior up to that step.
 *
 * @param s - A state of {@link particleFilterAlgorithm}.
 * @param status - How the run ended.
 * @returns The weighted posterior.
 * @concept sequential-monte-carlo
 * @example
 * ```ts
 * import { markov, observe, particleFilterAlgorithm, particleFilterPosterior, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import type { Value } from 'omai-engine/tensor'
 * const walk = () => markov('t', 3, (t, x: Value | number) => { const z = sample(`x${t}`, normal(x, 1)); observe(`y${t}`, normal(z, 1), t); return z }, 0)
 * const run = runAlgorithm(particleFilterAlgorithm(walk, undefined, { numParticles: 20 }), { seed: 0, maxSteps: 10 })
 * particleFilterPosterior(run.state, run.status).sites // => ['x0', 'x1', 'x2']
 * ```
 */
export function particleFilterPosterior(s: ParticleFilterState, status: Posterior['status']): Posterior {
  const first = s.paths[0] ?? []
  const names = first.flatMap((tr) =>
    [...tr.sites.values()]
      .filter((x) => (x.type === 'sample' && !x.isObserved) || x.type === 'deterministic')
      .map((x) => x.name),
  )
  const merged = s.paths.map((path) => ({
    sites: new Map(path.flatMap((tr) => [...tr.sites.entries()])),
    returnValue: undefined,
  }))
  return weightedPosterior(stackSiteValues(merged, names), s.logWeights, {
    engine: 'smc',
    status,
    logEvidence: { value: s.logEvidence, standardError: Number.NaN, method: 'particle filter' },
    diagnostics: { method: 'particle filter', resamplings: s.resamplings, finalEss: s.ess, steps: s.t },
  })
}

/** Options of {@link temperedSmc}. */
export interface TemperedSmcOptions {
  readonly numParticles?: number
  readonly mcmcSteps?: number
  readonly maxStages?: number
}

/**
 * Sequential Monte Carlo with adaptive tempering on a program without markov structure (`sampling.smcSampler`): the
 * prior on the unconstrained scale is tempered towards the posterior by powers of the likelihood, with MCMC
 * rejuvenation, giving equally weighted draws and the evidence.
 *
 * @param model - The program (continuous latent sites).
 * @param data - Its data.
 * @param options - Particles, MCMC steps, stage cap, key and tracing.
 * @returns The posterior of draws.
 * @concept sequential-monte-carlo
 * @example
 * ```ts
 * import { observe, sample, temperedSmc } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * Math.abs(item(temperedSmc(model, undefined, { key: key(0), numParticles: 400 }).mean('m')) - 0.5) < 0.15 // => true
 * ```
 */
export function temperedSmc<D = void>(
  model: Model<D>,
  data: D,
  options: TemperedSmcOptions & EngineRunOptions,
): Posterior {
  const problem = temperedProblem(model, data, options.key)
  const r = smcSampler(problem.prior, problem.logLikelihood, {
    key: foldIn(options.key, 1),
    numParticles: options.numParticles ?? 500,
    ...(options.mcmcSteps === undefined ? {} : { mcmcSteps: options.mcmcSteps }),
    ...(options.maxStages === undefined ? {} : { maxStages: options.maxStages }),
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  return temperedPosterior(problem, Float64Array.from(toFlat(r.particles)), {
    logEvidence: r.logEvidence,
    status: r.converged ? 'converged' : 'failed',
    diagnostics: { method: 'tempered', stages: r.stages, temperatures: [...r.temperatures] },
  })
}

/** A program as the inputs of tempered SMC: the prior as a proposal, the likelihood, and the unconstrained model. */
export interface TemperedProblem {
  /** The prior on the unconstrained scale (draws and log density, Jacobians included). */
  readonly prior: ChainProposal
  /** The log likelihood of the observed sites at an unconstrained point. */
  readonly logLikelihood: (x: Float64Array) => number
  readonly um: UnconstrainedModel
}

/**
 * The prior and the likelihood of a program for the adaptive-tempering sampler of `sampling`, on the unconstrained scale
 * of its continuous latent sites. {@link temperedSmc} runs the sampler to the end on it; a method that steps the sampler
 * (`smcAlgorithm`, one temperature per step) builds it the same way.
 *
 * @param model - The program (continuous latent sites).
 * @param data - Its data.
 * @param key - The key of the prototype run (shapes only).
 * @returns The problem.
 * @throws OmaiError with code `value` for discrete latent sites.
 * @concept sequential-monte-carlo
 * @example
 * ```ts
 * import { observe, sample, temperedProblem } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * temperedProblem(model, undefined, key(0)).prior.dim // => 1
 * ```
 */
export function temperedProblem<D = void>(model: Model<D>, data: D, key: Key): TemperedProblem {
  const proto = traceOf(seed(model, foldIn(key, 0)), data)
  const { layout, enumerated } = latentLayout(proto)
  if (enumerated.length > 0)
    throw new OmaiError('value', 'temperedSmc: discrete latent sites are not supported; use another engine')
  const dim = layout.reduce((a, l) => a + l.size, 0)
  const byName = new Map(layout.map((l) => [l.name, l]))
  const split = (x: ArrayLike<number>): { readonly prior: number; readonly lik: number } => {
    const u = fromStorage(Float64Array.from(x), [dim], 'float64')
    let ldj = 0
    const placed = handle(model, () => ({
      process: (msg: Message) => {
        if (msg.type !== 'sample' || msg.isObserved || msg.fn === undefined) return
        const l = byName.get(msg.name)
        if (l === undefined) return
        const piece = reshape(sliceAxis(u, 0, l.offset, l.offset + l.size), l.unconstrainedShape)
        const b = bijectors.bijectorForSupport(msg.fn.support)
        msg.value = b.forward(piece)
        ldj += item(primalOf(mul(sum(b.forwardLogDetJacobian(piece)), msg.scale)))
      },
    }))
    const t = traceOf(placed, data)
    let prior = ldj
    for (const s of t.sites.values())
      if (s.type === 'sample' && !s.isObserved) prior += item(primalOf(mul(sum(maskedLogProb(s)), s.scale)))
    const lik = observedLogProb(t)
    return { prior: Number.isNaN(prior) ? Number.NEGATIVE_INFINITY : prior, lik }
  }
  const prior = proposal({
    dim,
    sample: (k: Key) => {
      const t = traceOf(seed(model, k), data)
      const out = new Float64Array(dim)
      for (const l of layout) {
        const s = t.sites.get(l.name)
        if (s?.type !== 'sample') continue
        out.set(toFlat(primalOf(l.bijector.inverse(s.value))), l.offset)
      }
      return out
    },
    logProb: (x) => split(x).prior,
  })
  const um = potentialEnergy(model, data, { key: foldIn(key, 0), enumerate: false })
  return { prior, logLikelihood: (x) => split(x).lik, um }
}

/**
 * The posterior of tempered-SMC particles (unconstrained, row-major `[n, dim]`): the particles mapped back to every
 * latent and deterministic site, equally weighted, with the evidence estimate.
 *
 * @param problem - The problem the particles are for.
 * @param particles - The particles.
 * @param run - The log evidence, the status and the diagnostics.
 * @returns The posterior of draws.
 * @concept sequential-monte-carlo
 * @example
 * ```ts
 * import { observe, sample, temperedPosterior, temperedProblem } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * temperedPosterior(temperedProblem(model, undefined, key(0)), Float64Array.of(0, 1), { logEvidence: 0, status: 'converged', diagnostics: {} }).sites // => ['m']
 * ```
 */
export function temperedPosterior(
  problem: TemperedProblem,
  particles: Float64Array,
  run: {
    readonly logEvidence: number
    readonly status: Posterior['status']
    readonly diagnostics: Readonly<Record<string, Json>>
  },
): Posterior {
  const draws = constrainDraws(problem.um, particles)
  return samplePosterior(draws, {
    engine: 'smc',
    status: run.status,
    logEvidence: { value: run.logEvidence, standardError: Number.NaN, method: 'tempered SMC' },
    diagnostics: run.diagnostics,
  })
}

function filterable(a: ProgramAnalysis): string | undefined {
  if (a.markov.length !== 1) return 'no single markov loop'
  const loop = a.markov[0]
  if (loop === undefined) return 'no markov loop'
  if (loop.outsideLatents.length > 0) return `latent sites outside the loop (${loop.outsideLatents.join(', ')})`
  if (loop.after.length > 0) return `sites after the loop (${loop.after.join(', ')})`
  return undefined
}

/**
 * The SMC engine: a particle filter for programs whose latent sites live in one markov loop, and adaptive-tempering
 * SMC for programs with continuous latent sites otherwise (cost: particles x steps x observations).
 *
 * @param options - Particle counts and resampling.
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { smcEngine } from 'omai-engine/probprog'
 * smcEngine().name // => 'smc'
 * ```
 */
export function smcEngine(options: ParticleFilterOptions & TemperedSmcOptions = {}): InferenceEngine {
  const P = options.numParticles ?? 500
  return {
    name: 'smc',
    form: 'weighted',
    description: 'particle filtering of markov programs and adaptive-tempering SMC with evidence estimates',
    applies: (a: ProgramAnalysis) => {
      if (!a.hasLikelihood) return feasibility(false, 'weighted', 0, 'observed sites have no likelihood')
      if (a.latent.length === 0) return feasibility(false, 'weighted', 0, 'no latent sites')
      const problem = filterable(a)
      if (problem === undefined) {
        const T = a.markov[0]?.length ?? 1
        return feasibility(
          true,
          'weighted',
          P * T * Math.max(1, a.observations / T) * 10,
          `a markov loop of ${T} steps`,
        )
      }
      if (a.discrete.sites.length > 0 || a.continuous.length === 0)
        return feasibility(false, 'weighted', 0, `${problem}, and discrete latent sites prevent tempering`)
      return feasibility(
        true,
        'weighted',
        P * 50 * (options.mcmcSteps ?? 5) * Math.max(1, a.observations),
        `tempering (${problem})`,
      )
    },
    run: <D>(model: Model<D>, data: D, a: ProgramAnalysis, run: EngineRunOptions) =>
      filterable(a) === undefined
        ? particleFilter(model, data, { ...options, ...run })
        : temperedSmc(model, data, { ...options, ...run }),
  }
}
