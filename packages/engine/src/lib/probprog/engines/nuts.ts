/**
 * Hamiltonian Monte Carlo with the No-U-Turn Sampler on probabilistic programs (NumPyro's `MCMC(NUTS(model))`):
 * sampling the unconstrained density with Jacobian corrections and enumerated discrete sites summed out, then drawing
 * those discrete sites back with `inferDiscrete`. Also the engine for the planner.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { categorical as drawCategorical, foldIn, type Key } from '../../random/index.ts'
import { type Chains, diagnose, hmc, nuts } from '../../sampling/index.ts'
import {
  add,
  cast,
  concat,
  div,
  fromStorage,
  full,
  maximum,
  mul,
  primalOf,
  reshape,
  slice,
  sliceAxis,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { enumeratedMarginals, maxPlateNesting, traceFactors } from '../_contract.ts'
import type { ProgramAnalysis } from '../analysis.ts'
import {
  constrainDraws,
  findInitialParams,
  type InitStrategy,
  initToUniform,
  logJoint,
  potentialEnergy,
  type UnconstrainedModel,
} from '../density.ts'
import { enumerate } from '../handlers.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine } from './engine.ts'

/** The paths the MCMC driver logs, as a type. */
export type McmcPath =
  | 'mcmc/chain'
  | 'mcmc/stepSize'
  | 'mcmc/accept'
  | 'mcmc/divergent'
  | 'mcmc/energy'
  | 'mcmc/site'
  | 'init/tries'
  | 'init/potential'

/**
 * What the MCMC driver logs. The starting-point steps log the chain and the tries of its search; the sampling step logs, per
 * chain, the mean step size, the mean acceptance, the divergences and the mean energy of the draws; each enumerated site
 * recovered logs its index.
 */
export const mcmcContract: Contract<unknown, McmcPath> = defineContract({
  'mcmc/chain': scalar({ description: 'the chain a starting point was found for' }),
  'mcmc/stepSize': tensor({
    role: 'output',
    shape: ['chains'],
    description: 'the mean step size of the draws of each chain',
  }),
  'mcmc/accept': tensor({
    role: 'output',
    shape: ['chains'],
    description: 'the mean acceptance probability of each chain',
  }),
  'mcmc/divergent': tensor({
    role: 'output',
    shape: ['chains'],
    description: 'the divergent transitions of each chain',
  }),
  'mcmc/energy': tensor({
    role: 'output',
    shape: ['chains'],
    description: 'the mean Hamiltonian of the draws of each chain',
  }),
  'mcmc/site': scalar({ description: 'the draw (or enumerated site) being filled in by discrete inference' }),
  'init/tries': scalar({ description: 'tries the starting-point search made' }),
  'init/potential': scalar({ description: 'the potential energy at the starting point', direction: 'lower' }),
})

/** The state of {@link inferDiscreteAlgorithm}: the values fixed so far (continuous draws, then the enumerated sites). */
export interface InferDiscreteState {
  readonly fixed: Record<string, TensorLike>
  readonly step: number
  readonly finished: boolean
}

/**
 * Recovery of the enumerated discrete sites at a draw of the continuous sites, as a stepped algorithm, one site per step: run
 * the program with the values fixed so far, sum out the enumeration, and draw the first enumerated site from its marginal
 * given the rest (sequentially, so later sites see earlier draws). It is done (`converged`) when no enumerated site is left.
 * The state serialises.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param values - The continuous sites' values at the draw.
 * @param key - The key of the draws.
 * @returns The algorithm; the values are the final state's `fixed`.
 * @concept nuts
 * @example
 * ```ts
 * import { inferDiscreteAlgorithm, observe, sample } from 'omai-engine/probprog'
 * import { bernoulli, normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = () => { const c = sample('c', bernoulli({ probs: 0.5 })); observe('y', normal(c, 1), 1) }
 * Object.keys(runAlgorithm(inferDiscreteAlgorithm(model, undefined, {}, key(0)), { maxSteps: 10 }).state.fixed) // => ['c']
 * ```
 */
export function inferDiscreteAlgorithm<D = void>(
  model: Model<D>,
  data: D,
  values: Readonly<Record<string, TensorLike>>,
  key: Key,
): Algorithm<InferDiscreteState, McmcPath> {
  const m = enumerate(model)
  return {
    name: 'probprog/infer-discrete',
    config: { sites: Object.keys(values).length },
    init: () => ({ fixed: { ...values }, step: 0, finished: false }),
    step: (st, ctx) => {
      const fixed = st.fixed
      const step = st.step
      const r = logJoint(m, data, fixed)
      const t = r.trace
      const next = [...t.sites.values()].find((s) => s.type === 'sample' && s.enumDim !== undefined)
      if (next === undefined || next.type !== 'sample') return { fixed, step, finished: true }
      const depth = maxPlateNesting(t)
      const { factors, sumVars, scales } = traceFactors(t, depth)
      const { marginals } = enumeratedMarginals(factors, sumVars, scales)
      const mg = marginals.get(next.name)
      if (mg === undefined) throw new OmaiError('value', `inferDiscrete: no marginal for ${next.name}`)
      const probs = primalOf(mg.probs)
      const k = probs.shape[probs.shape.length - 1] ?? 1
      const batch = probs.shape.slice(0, -1)
      const count = batch.reduce((a, b) => a * b, 1)
      const weights = maximum(reshape(probs, [count, k]), 0)
      const support = next.fn.support
      const low = support.kind === 'integerInterval' ? support.low : 0
      const drawn =
        count === 0
          ? zeros([0])
          : concat(
              Array.from({ length: count }, (_, i) =>
                drawCategorical(foldIn(foldIn(key, step), i), reshape(sliceAxis(weights, 0, i, i + 1), [k]), [1]),
              ),
            )
      const out = add(cast(drawn, 'float64'), low)
      // The site's own batch shape: plates of the marginal, padded with the singleton plate axes of its batch.
      const own = next.fn.batchShape.slice(
        next.fn.batchShape.length - next.plates.reduce((a, f) => Math.max(a, -f.dim), 0),
      )
      fixed[next.name] = reshape(out, own.reduce((a, b) => a * b, 1) === count ? own : batch)
      ctx.log('mcmc/site', step)
      return { fixed, step: step + 1, finished: false }
    },
    done: (st) => (st.finished ? 'converged' : st.step >= 10000 ? 'failed' : undefined),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as InferDiscreteState,
  }
}

/**
 * Draw the enumerated discrete sites of a program given values of its other sites (NumPyro's `infer_discrete` with
 * temperature 1): each site in execution order is drawn from its exact conditional marginal given the values and the
 * sites drawn before it (elementwise for plated sites), by differentiating the enumerated log density.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param values - Values of the non-enumerated latent sites.
 * @param key - The key.
 * @returns The values plus a draw of every enumerated site.
 * @concept exact-inference
 * @example
 * ```ts
 * import { inferDiscrete, observe, sample } from 'omai-engine/probprog'
 * import { bernoulli, normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * const model = () => { const c = sample('c', bernoulli({ probs: 0.5 })); observe('y', normal(c, 0.01), 1) }
 * item(inferDiscrete(model, undefined, {}, key(0))['c'] ?? 0) // => 1
 * ```
 */
export function inferDiscrete<D = void>(
  model: Model<D>,
  data: D,
  values: Readonly<Record<string, TensorLike>>,
  key: Key,
): Record<string, TensorLike> {
  const run = runAlgorithm(inferDiscreteAlgorithm(model, data, values, key), { maxSteps: 10001 })
  if (run.status === 'failed') throw new OmaiError('value', 'inferDiscrete: too many enumerated sites')
  return run.state.fixed
}

/** Options of {@link mcmc}. */
export interface McmcOptions {
  /** `nuts` (default) or `hmc` with a fixed number of integration steps. */
  readonly algorithm?: 'nuts' | 'hmc'
  readonly numChains?: number
  readonly numDraws?: number
  readonly numWarmup?: number
  readonly thin?: number
  /** HMC integration steps per transition (default 10). */
  readonly numSteps?: number
  readonly maxTreeDepth?: number
  readonly targetAcceptance?: number
  /** Adapt a dense mass matrix (default diagonal). */
  readonly dense?: boolean
  /** How chains start (default {@link initToUniform}, radius 2). */
  readonly init?: InitStrategy
  /** Draw enumerated discrete sites for every draw (default true when there are any). */
  readonly inferDiscrete?: boolean
}

function rowOf(d: Tensor, i: number): Tensor {
  const shape = d.shape.slice(1)
  const size = shape.reduce((a, b) => a * b, 1)
  return fromStorage(d.data.slice(i * size, (i + 1) * size) as Float64Array, shape, 'float64')
}

function failed(engine: string, reason: string, extra: Readonly<Record<string, Json>> = {}): Posterior {
  return definePosterior({
    form: 'samples',
    engine,
    sites: [],
    status: 'failed',
    draws: {},
    diagnostics: { reason, ...extra },
  })
}

/**
 * The state of {@link mcmcAlgorithm}: the starting points found so far, the chains once sampled, the enumerated sites
 * recovered draw by draw, and why the run failed if it did.
 */
export interface McmcState {
  /** `init` while finding starting points, `sample` for the chains, `discrete` while recovering enumerated sites, then `done`. */
  readonly stage: 'init' | 'sample' | 'discrete' | 'done'
  readonly starts: Float64Array[]
  readonly chains: Chains | null
  /** The recovered values of the enumerated sites, draw after draw (flat). */
  readonly discrete: Record<string, number[]>
  readonly shapes: Record<string, number[]>
  /** The next draw whose enumerated sites are recovered. */
  readonly drawIndex: number
  readonly failure: { readonly reason: string; readonly extra: Record<string, Json> } | null
}

const perChainMean = (t: Tensor | undefined, numChains: number): Value => {
  if (t === undefined) return full([numChains], Number.NaN)
  const flat = reshape(t, [-1])
  const n = Math.floor((flat.shape[0] ?? 0) / numChains)
  const perChain = reshape(slice(flat, [0], [numChains * n]), [numChains, n])
  return div(sum(perChain, 1), Math.max(1, n))
}

/**
 * Hamiltonian Monte Carlo on a program as a stepped algorithm, the driver of {@link mcmc}: one step per chain to find a
 * valid starting point (the search of `findInitialParams`), one step that runs NUTS or fixed-length HMC with window
 * adaptation on the unconstrained density for all chains (their warm-up and draws are traced runs of their own), then, when
 * discrete sites were enumerated, one step per draw that recovers them. It logs, per starting point, the chain and its
 * search, and per chain after sampling the mean step size, acceptance, divergences and energy. It is done (`converged`) after
 * the last stage and `failed` when the program has no valid starting point. The state serialises; read the posterior with
 * {@link mcmcPosterior}.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Sampler options, key and tracing.
 * @returns The algorithm.
 * @concept nuts
 * @example
 * ```ts
 * import { mcmcAlgorithm, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const run = runAlgorithm(mcmcAlgorithm(model, undefined, { key: key(0), numChains: 2, numWarmup: 50, numDraws: 20 }), { maxSteps: 10 })
 * [run.status, run.steps, run.state.chains?.numDraws] // => ['converged', 3, 20]
 * ```
 */
export function mcmcAlgorithm<D = void>(
  model: Model<D>,
  data: D,
  options: McmcOptions & EngineRunOptions,
): Algorithm<McmcState, McmcPath> {
  const engine = options.algorithm ?? 'nuts'
  const numChains = options.numChains ?? 2
  const numDraws = options.numDraws ?? 500
  const numWarmup = options.numWarmup ?? 500
  let umCache: UnconstrainedModel | undefined
  const um = (): UnconstrainedModel => {
    umCache ??= potentialEnergy(model, data, { key: foldIn(options.key, 0) })
    return umCache
  }
  let constrainedFor: { chains: Chains; draws: Record<string, Tensor> } | undefined
  const constrained = (chains: Chains): Record<string, Tensor> => {
    if (constrainedFor?.chains !== chains)
      constrainedFor = { chains, draws: constrainDraws(um(), chains.draws.data) as Record<string, Tensor> }
    return constrainedFor.draws
  }
  const discreteWanted = (): boolean => um().enumerated.length > 0 && options.inferDiscrete !== false
  const base = (over: Partial<McmcState>, s: McmcState): McmcState => ({ ...s, ...over })
  return {
    name: 'probprog/mcmc',
    config: { engine, numChains, numDraws, numWarmup },
    init: () => {
      const empty = { starts: [] as Float64Array[], chains: null, discrete: {}, shapes: {}, drawIndex: 0 }
      try {
        um()
      } catch (e) {
        return { ...empty, stage: 'done', failure: { reason: e instanceof Error ? e.message : String(e), extra: {} } }
      }
      return { ...empty, stage: 'init', failure: null }
    },
    step: (s, ctx) => {
      if (s.stage === 'init') {
        const c = s.starts.length
        const init = findInitialParams(um(), model, data, {
          key: foldIn(foldIn(options.key, 1), c),
          strategy: options.init ?? initToUniform(),
        })
        ctx.log('mcmc/chain', c)
        ctx.log('init/tries', init.tries)
        ctx.log('init/potential', init.potential)
        if (!init.valid)
          return base(
            {
              stage: 'done',
              failure: {
                reason: 'no starting point with a finite log density and gradient',
                extra: { tries: init.tries },
              },
            },
            s,
          )
        const starts = [...s.starts, init.unconstrained]
        return base({ starts, stage: starts.length >= numChains ? 'sample' : 'init' }, s)
      }
      if (s.stage === 'sample') {
        const common = {
          key: foldIn(options.key, 2),
          numChains,
          numDraws,
          numWarmup,
          init: s.starts,
          ...(options.thin === undefined ? {} : { thin: options.thin }),
          ...(options.targetAcceptance === undefined ? {} : { targetAcceptance: options.targetAcceptance }),
          ...(options.dense === undefined ? {} : { dense: options.dense }),
          ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
          ...(options.policy === undefined ? {} : { policy: options.policy }),
        }
        const chains: Chains =
          engine === 'hmc'
            ? hmc(um().logDensity, { ...common, numIntegrationSteps: options.numSteps ?? 10 })
            : nuts(um().logDensity, {
                ...common,
                ...(options.maxTreeDepth === undefined ? {} : { maxTreeDepth: options.maxTreeDepth }),
              })
        ctx.log('mcmc/stepSize', perChainMean(chains.stats.stepSize, numChains))
        ctx.log('mcmc/accept', perChainMean(chains.stats.acceptanceRate, numChains))
        ctx.log('mcmc/divergent', mul(perChainMean(chains.stats.divergent, numChains), chains.numDraws))
        ctx.log('mcmc/energy', perChainMean(chains.stats.energy, numChains))
        return base({ chains, stage: discreteWanted() ? 'discrete' : 'done' }, s)
      }
      // Recover the enumerated discrete sites at one draw.
      const chains = s.chains as Chains
      const total = numChains * chains.numDraws
      const draws = constrained(chains)
      const i = s.drawIndex
      const values: Record<string, Value> = {}
      for (const l of um().layout) {
        const d = draws[l.name]
        if (d !== undefined) values[l.name] = rowOf(d, i)
      }
      const r = inferDiscrete(model, data, values, foldIn(foldIn(options.key, 3), i))
      const discrete: Record<string, number[]> = { ...s.discrete }
      const shapes: Record<string, number[]> = { ...s.shapes }
      for (const name of um().enumerated) {
        const v = primalOf(r[name] as Tensor)
        shapes[name] = [...v.shape]
        discrete[name] = [...(discrete[name] ?? []), ...toFlat(v)]
      }
      ctx.log('mcmc/site', i)
      return base({ discrete, shapes, drawIndex: i + 1, stage: i + 1 >= total ? 'done' : 'discrete' }, s)
    },
    done: (s) => (s.stage !== 'done' ? undefined : s.failure === null ? 'converged' : 'failed'),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as McmcState,
  }
}

/**
 * The posterior of a finished MCMC state: the draws mapped back to every latent and deterministic site, the enumerated
 * sites recovered by {@link inferDiscrete}, and the diagnostics; a run without a valid starting point gives a failed
 * posterior.
 *
 * @param model - The program the state was run with.
 * @param data - Its data.
 * @param options - The options the state was run with.
 * @param state - A (finished) state of {@link mcmcAlgorithm}.
 * @returns The posterior.
 * @concept nuts
 * @example
 * ```ts
 * import { mcmcAlgorithm, mcmcPosterior, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const options = { key: key(0), numChains: 1, numWarmup: 40, numDraws: 10 }
 * mcmcPosterior(model, undefined, options, runAlgorithm(mcmcAlgorithm(model, undefined, options), { maxSteps: 10 }).state).status // => 'completed'
 * ```
 */
export function mcmcPosterior<D = void>(
  model: Model<D>,
  data: D,
  options: McmcOptions & EngineRunOptions,
  state: McmcState,
): Posterior {
  const engine = options.algorithm ?? 'nuts'
  if (state.failure !== null) return failed(engine, state.failure.reason, state.failure.extra)
  const um = potentialEnergy(model, data, { key: foldIn(options.key, 0) })
  const chains = state.chains as Chains
  const numChains = options.numChains ?? 2
  const draws = constrainDraws(um, chains.draws.data)
  const total = numChains * chains.numDraws
  if (um.enumerated.length > 0 && options.inferDiscrete !== false)
    for (const name of um.enumerated)
      draws[name] = fromStorage(
        Float64Array.from(state.discrete[name] ?? []),
        [total, ...(state.shapes[name] ?? [])],
        'float64',
      )
  const report = diagnose(chains, options.maxTreeDepth === undefined ? {} : { maxTreeDepth: options.maxTreeDepth })
  const nonFinite = toFlat(chains.draws).some((v) => !Number.isFinite(v))
  return definePosterior({
    form: 'samples',
    engine,
    sites: Object.keys(draws),
    draws,
    chainShape: [numChains, chains.numDraws],
    chains,
    status: nonFinite ? 'failed' : 'completed',
    diagnostics: {
      divergences: report.divergences?.count ?? 0,
      highRhat: [...report.highRhat],
      lowEss: [...report.lowEss],
      messages: [...report.messages],
      stepSizes: chains.parameters.map((p) => (p as { stepSize?: number }).stepSize ?? null),
      enumerated: um.enumerated,
    },
  })
}

/**
 * Hamiltonian Monte Carlo on a program (NumPyro's `MCMC(NUTS(model))`): the unconstrained potential (Jacobians
 * added, enumerated discrete sites summed out), valid starting points per chain, window adaptation of step size and
 * mass matrix, NUTS or fixed-length HMC as a traced run per chain, then draws mapped back to every latent and
 * deterministic site, with enumerated sites recovered by {@link inferDiscrete}. {@link mcmcAlgorithm} run to the end. Programs without a valid starting
 * point end with status `failed` and the reason; divergences, R-hat and ESS are in the diagnostics.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Sampler options, key and tracing.
 * @returns The posterior of draws (chains laid end to end, with `chains` on unconstrained coordinates).
 * @concept nuts
 * @index function Run MCMC on a program and return a posterior.
 * @example
 * ```ts
 * import { mcmc, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const post = mcmc(model, undefined, { key: key(0), numChains: 1, numWarmup: 100, numDraws: 100 })
 * post.draws?.['mu']?.shape // => [100]
 * ```
 */
export function mcmc<D = void>(model: Model<D>, data: D, options: McmcOptions & EngineRunOptions): Posterior {
  const run = runAlgorithm(mcmcAlgorithm(model, data, options), { maxSteps: Number.MAX_SAFE_INTEGER })
  return mcmcPosterior(model, data, options, run.state)
}

/**
 * The NUTS engine: Hamiltonian Monte Carlo on the unconstrained density with enumerated discrete sites summed out
 * (cost: chains x (warm-up + draws) x about 2^5 gradient evaluations x observations).
 *
 * @param options - Sampler options.
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { nutsEngine } from 'omai-engine/probprog'
 * nutsEngine({ numDraws: 200 }).name // => 'nuts'
 * ```
 */
export function nutsEngine(options: McmcOptions = {}): InferenceEngine {
  const name = options.algorithm ?? 'nuts'
  return {
    name,
    form: 'samples',
    description: 'NUTS and HMC on the unconstrained log density, discrete sites summed out by enumeration',
    applies: (a: ProgramAnalysis) => {
      if (!a.hasLikelihood) return feasibility(false, 'mcmc', 0, 'observed sites have no likelihood')
      if (a.continuous.length === 0) return feasibility(false, 'mcmc', 0, 'no continuous latent sites')
      if (a.discrete.unbounded.length > 0)
        return feasibility(false, 'mcmc', 0, `unbounded discrete sites (${a.discrete.unbounded.join(', ')})`)
      if (!a.differentiable) return feasibility(false, 'mcmc', 0, 'the log density is not differentiable')
      if (a.plates.some((p) => p.subsampleSize < p.size))
        return feasibility(false, 'mcmc', 0, 'subsampled plates need stochastic-gradient methods')
      const iterations = (options.numChains ?? 2) * ((options.numWarmup ?? 500) + (options.numDraws ?? 500))
      const enumCost = Math.max(1, Math.min(a.discrete.jointSize, 1e6))
      const cost = iterations * 32 * Math.max(1, a.observations + a.continuousDim) * enumCost
      return feasibility(true, 'mcmc', cost, `${a.continuousDim} continuous coordinates with a differentiable density`)
    },
    run: <D>(model: Model<D>, data: D, _a: ProgramAnalysis, run: EngineRunOptions) =>
      mcmc(model, data, { ...options, ...run }),
  }
}
