/**
 * The driver the kernel methods share: it turns a kernel family of the sampling library (random-walk Metropolis, MALA,
 * Barker, HMC, slice, the stochastic-gradient and microcanonical samplers, the ensemble samplers, parallel tempering)
 * into an inference method on a program or a bound spec, with the fine step the plan asks for. It writes no kernel and no
 * adaptation: a family says how to build its kernel from tuned parameters, and, if it has one, which of the library's
 * warm-up algorithms tunes them; the driver runs the chains one transition per step (warm-up, then draws, then, for a model
 * with enumerated discrete sites, one step per draw to recover them), keeps the draws in a state that serialises, reads
 * them as a posterior at any time, and logs the tape. Every method built on it has its own knobs (the family's own beside
 * the chains, warm-up, draws and thinning) and its own contract (the shared `mcmc/` paths plus the family's). Internal to
 * the folder.
 */

import type { Json } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  type Chains,
  chainFromJson,
  chainsFromDraws,
  chainToJson,
  diagnose,
  type MCMCKernel,
  type MCMCState,
  type SamplerStat,
  type TransitionInfo,
} from '../../sampling/index.ts'
import { bool, type DictSpace, dict, int } from '../../space/index.ts'
import {
  arange,
  asTensor,
  broadcastTo,
  cast,
  concat,
  full,
  primalOf,
  reshape,
  sliceAxis,
  stack,
  type Tensor,
  take,
  toArray,
  toFlat,
  transpose,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  message,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import {
  constrainDraws,
  findInitialParams,
  initToUniform,
  potentialEnergy,
  type UnconstrainedModel,
} from '../density.ts'
import { inferDiscrete, nutsEngine } from '../engines/nuts.ts'
import { substitute } from '../handlers.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Message, Model } from '../primitives.ts'
import {
  checkOf,
  flag,
  guard,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  posteriorJson,
  priorOnly,
  programOf,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod, type Verdict } from './method.ts'

/** The paths every kernel method logs (a family adds its own). */
export type McmcPath =
  | 'init/tries'
  | 'init/potential'
  | 'mcmc/stage'
  | 'mcmc/logDensity'
  | 'mcmc/stepSize'
  | 'mcmc/accept'
  | 'mcmc/divergent'
  | 'mcmc/treeDepth'
  | 'mcmc/energy'
  | 'mcmc/position'
  | 'mcmc/divergences'
  | 'mcmc/kept'
  | 'mcmc/windowEnd'
  | 'mcmc/site'
  | 'mcmc/snapshot'

/** The columns of one kept draw before the position: the statistics `samplerAlgorithm` reports, in this order. */
const STATS: readonly SamplerStat[] = [
  'logDensity',
  'acceptanceRate',
  'accepted',
  'divergent',
  'treeDepth',
  'numSteps',
  'energy',
  'stepSize',
]

/** What a family's warm-up gives the driver: a stepped algorithm over some state, and how to read the tuned parameters. */
export interface Warmer<T> {
  /** The state at a starting position (the algorithm's `init`). */
  init(start: Float64Array, key: Key): unknown
  /** One iteration of the warm-up. */
  step(w: unknown, ctx: StepContext): unknown
  /** The chain's current state. */
  chain(w: unknown): MCMCState
  /** What the last transition did. */
  info(w: unknown): TransitionInfo | undefined
  /** The parameters the warm-up has found so far. */
  result(w: unknown): T
  toJson(w: unknown): Json
  fromJson(j: Json): unknown
  /** The state with its chain started again at a position (the target moved under it: a block of a hybrid sampler). */
  restart?(w: unknown, position: Float64Array): unknown
  /** True when the last iteration closed an adaptation window (for the `mcmc/windowEnd` message). */
  windowEnded?(w: unknown): { readonly stepSize: number; readonly inverseMassMatrix: readonly number[] } | undefined
}

/** What a family gives the driver for one run on one model: the kernel, its parameters and how they are tuned. */
export interface McmcPlan<T> {
  /** The dimension of the kernel's position (walkers times the model's dimension for an ensemble). */
  readonly kernelDim: number
  /** Starting points a unit needs (the walkers of an ensemble; 1 otherwise). */
  readonly walkers: number
  /** The kernel at some parameters. */
  readonly kernel: (tuned: T) => MCMCKernel
  /** The parameters before any tuning. */
  readonly tuned0: T
  readonly warm?: Warmer<T>
  readonly tunedToJson: (t: T) => Json
  readonly tunedFromJson: (j: Json) => T
  /** The parameters as a diagnostic (step size, mass matrix, scale). */
  readonly tunedSummary: (t: T) => Json
  /** Log what the family adds to the tape after a step, from every unit's last state. */
  readonly extraTape?: (ctx: StepContext<string>, states: readonly MCMCState[], stage: number) => void
}

/** Everything a family's `plan` can read about a run. */
export interface McmcBuild {
  readonly um: UnconstrainedModel
  readonly program: Model
  readonly model: MethodModel
  readonly config: Config
}

/** A kernel method: its name, knobs, extra tape paths, check and how a run is planned. */
export interface McmcSpec<T, K extends string> {
  readonly name: string
  readonly description: string
  /** The method's own knobs. `numChains`, `numWarmup`, `numDraws` and `thin` are added unless it declares them. */
  readonly knobs: DictSpace
  /** The tape paths beyond {@link McmcPath}. */
  readonly extraPaths?: Contract<McmcRunState, K>
  /** What the family needs of the model beyond continuous latents and a differentiable density; a verdict when it does not hold. */
  readonly check?: (a: ModelAnalysis) => Verdict | undefined
  readonly plan: (build: McmcBuild) => McmcPlan<T>
  /** Whether the method runs on a program with subsampled plates (stochastic-gradient families). */
  readonly minibatch?: boolean
  /** Options of the diagnostics report of a posterior. */
  readonly diagnostics?: { readonly maxTreeDepth?: number }
}

/** One unit of the run (a chain, or an ensemble of walkers, or a tempering system): its warm-up, its state and its draws. */
export interface McmcChainState {
  readonly warm: unknown | null
  readonly chain: MCMCState | null
  readonly tuned: unknown | null
  /** The statistics of the last transition, in the columns of {@link STATS}. */
  readonly last: Float64Array
  /** The draws kept so far: a row of the statistics and then the position each. */
  readonly kept: readonly Float64Array[]
  readonly divergences: number
}

/** The state of a kernel method. */
export interface McmcRunState {
  readonly stage: 'init' | 'warmup' | 'sample' | 'discrete' | 'done'
  readonly iteration: number
  readonly chains: readonly McmcChainState[]
  /** The recovered values of the enumerated sites, draw after draw (flat). */
  readonly discrete: Readonly<Record<string, readonly number[]>>
  readonly shapes: Readonly<Record<string, readonly number[]>>
  readonly drawIndex: number
  readonly failure: { readonly reason: string; readonly extra: Readonly<Record<string, Json>> } | null
}

/** The knobs every kernel method has: the chains, the warm-up, the draws kept and the thinning. */
export const commonMcmcKnobs = (
  defaults: { readonly numChains?: number; readonly numWarmup?: number; readonly numDraws?: number } = {},
): DictSpace =>
  dict({
    numChains: int(1, 64, { default: defaults.numChains ?? 2, label: 'chains' }),
    numWarmup: int(0, 1_000_000, { default: defaults.numWarmup ?? 500, label: 'warm-up iterations' }),
    numDraws: int(1, 1_000_000, { default: defaults.numDraws ?? 500, label: 'draws per chain' }),
    thin: int(1, 1000, { default: 1, label: 'thinning' }),
    inferDiscrete: bool({
      default: true,
      label: 'recover enumerated sites',
      description: 'draw the enumerated discrete sites for every draw',
    }),
  })

const rowOf = (d: Tensor, i: number): Tensor => take(d, i, 0)

/** A transition's statistics as the columns of a kept row. */
const statsOf = (logDensity: number, info: TransitionInfo): number[] => [
  logDensity,
  info.acceptanceRate,
  info.accepted ? 1 : 0,
  info.divergent === undefined ? Number.NaN : info.divergent ? 1 : 0,
  info.treeDepth ?? Number.NaN,
  info.numSteps ?? Number.NaN,
  info.energy ?? Number.NaN,
  info.stepSize ?? Number.NaN,
]

const emptyChain = (): McmcChainState => ({
  warm: null,
  chain: null,
  tuned: null,
  last: Float64Array.from(toFlat(full([STATS.length], Number.NaN))),
  kept: [],
  divergences: 0,
})

const flatten = (rows: readonly Float64Array[]): Float64Array =>
  Float64Array.from(toFlat(rows.length === 0 ? zeros([0]) : concat(rows.map((r) => asTensor(r)))))

const unflatten = (flat: Float64Array, width: number): Float64Array[] =>
  width === 0
    ? []
    : (toArray(reshape(asTensor(flat), [flat.length / width, width])) as number[][]).map((r) => Float64Array.from(r))

/** The program with every subsampled plate replaced by the whole plate (so its likelihood is the full one, unscaled). */
function fullBatch(program: Model): Model {
  return substitute(program, (msg: Message) =>
    msg.type === 'plate' && (msg.subsampleSize ?? 0) < (msg.size ?? 0)
      ? cast(arange(msg.size ?? 0), 'int32')
      : undefined,
  ) as Model
}

const failedPosterior = (engine: string, reason: string, extra: Readonly<Record<string, Json>> = {}): Posterior =>
  definePosterior({
    form: 'samples',
    engine,
    sites: [],
    status: 'failed',
    draws: {},
    diagnostics: { reason, ...extra },
  })

/**
 * A kernel method from a spec: the method (with `check`, `posterior`, the budget and snapshot knobs and the typed
 * contract) for a model.
 *
 * @param spec - The family's name, knobs, tape paths and plan.
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name.
 * @returns The method.
 */
export function mcmcMethodOf<T, K extends string = never>(
  spec: McmcSpec<T, K>,
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<McmcRunState, Config, McmcPath | K, ModelAnalysis> {
  type Path = McmcPath | K
  const name = options.name ?? spec.name
  const raw = programOf(model)
  // A stochastic-gradient family may run on subsampled plates; its full-density evaluations (the start, the tape, the posterior's
  // constrained draws) see every row: the program with each subsampled plate replaced by the whole plate.
  const program: Model = spec.minibatch === true ? fullBatch(raw) : raw
  let umCache: UnconstrainedModel | undefined
  const unconstrained = (): UnconstrainedModel => {
    umCache ??= potentialEnergy(program, undefined)
    return umCache
  }
  const knobs: DictSpace = {
    ...spec.knobs,
    spaces: { ...commonMcmcKnobs().spaces, ...spec.knobs.spaces },
  }
  const planCache = new Map<string, McmcPlan<T>>()
  const planOf = (config: Config): McmcPlan<T> => {
    // The plan depends on the knobs only through what the family reads; one plan per distinct configuration.
    const id = JSON.stringify(config)
    let p = planCache.get(id)
    if (p === undefined) {
      p = spec.plan({ um: unconstrained(), program: raw, model, config })
      if (planCache.size > 8) planCache.clear()
      planCache.set(id, p)
    }
    return p
  }

  /** The chains a state holds so far as the library's `Chains`: every walker of every unit is a chain. */
  const chainsOf = (um: UnconstrainedModel, plan: McmcPlan<T>, state: McmcRunState): Chains | undefined => {
    const units = state.chains.length
    const n = Math.min(...state.chains.map((c) => c.kept.length))
    if (!Number.isFinite(n) || n < 1) return undefined
    const dim = um.dim
    const walkers = plan.walkers
    const numChains = units * walkers
    // One matrix per unit: a row per kept draw, the sampler statistics then the position of every walker.
    const units_ = state.chains.map((ch) => stack(ch.kept.slice(0, n).map((r) => asTensor(r))))
    const draws = concat(
      units_.map((m) =>
        transpose(reshape(sliceAxis(m, 1, STATS.length, STATS.length + walkers * dim), [n, walkers, dim]), [1, 0, 2]),
      ),
      0,
    ) as Tensor
    // The statistics of a unit are shared by its walkers.
    const stats = STATS.map((_, k) =>
      concat(
        units_.map((m) => broadcastTo(reshape(take(m, k, 1), [1, n]), [walkers, n])),
        0,
      ),
    )
    const base = chainsFromDraws(draws, {
      names: um.logDensity.names,
      stats: Object.fromEntries(
        STATS.flatMap((s, k) =>
          Number.isNaN(toFlat(stats[k] as Value)[0] ?? Number.NaN) ? [] : [[s, stats[k] as Tensor]],
        ),
      ),
    })
    return {
      ...base,
      // banned-ok: non-numeric: one summary record per chain
      parameters: Array.from({ length: numChains }, (_, ci) => {
        const ch = state.chains[Math.floor(ci / walkers)]
        return ch?.tuned === null || ch === undefined ? {} : (plan.tunedSummary(ch.tuned as T) as Json)
      }),
    }
  }

  const posterior = (state: McmcRunState, config: Config): Posterior => {
    if (state.failure !== null) return failedPosterior(name, state.failure.reason, state.failure.extra)
    const um = unconstrained()
    const plan = planOf(config)
    const chains = chainsOf(um, plan, state)
    if (chains === undefined) return failedPosterior(name, 'no draws yet', { stage: state.stage })
    const done = state.stage === 'done'
    const draws = constrainDraws(um, chains.draws.data)
    const total = chains.numChains * chains.numDraws
    if (done && um.enumerated.length > 0 && flag(config, 'inferDiscrete'))
      for (const site of um.enumerated)
        draws[site] = reshape(asTensor(state.discrete[site] ?? []), [total, ...(state.shapes[site] ?? [])]) as Tensor
    const diagnostics: Record<string, Json> = {
      partial: !done,
      iterations: state.iteration,
      draws: chains.numDraws,
      divergences: state.chains.reduce((a, ch) => a + ch.divergences, 0),
      parameters: chains.parameters as Json,
      enumerated: um.enumerated,
    }
    if (chains.numDraws >= 4 && chains.numChains >= 1) {
      try {
        const report = diagnose(chains, spec.diagnostics ?? {})
        diagnostics['highRhat'] = [...report.highRhat]
        diagnostics['lowEss'] = [...report.lowEss]
      } catch {
        // Too few draws for the diagnostics: the summary leaves R-hat and ESS out.
      }
    }
    return definePosterior({
      form: 'samples',
      engine: name,
      sites: Object.keys(draws),
      draws,
      chainShape: [chains.numChains, chains.numDraws],
      chains,
      status: done ? 'completed' : 'stopped',
      diagnostics,
    })
  }

  const baseOf = checkOf(nutsEngine(), (a) => {
    if (a.continuous.length === 0)
      return no('no continuous latent nodes', ['continuous latents'], 'use enumeration for a discrete model, or Gibbs')
    if (a.unboundedDiscrete.length > 0)
      return no(
        `unbounded discrete latents (${a.unboundedDiscrete.join(', ')})`,
        ['finite discrete latents'],
        'bound them, or use Gibbs',
      )
    return yes(
      `${
        a.finiteDiscrete.length > 0
          ? 'continuous latents; the finite discrete ones are summed out'
          : 'continuous latents with a differentiable density'
      }${priorOnly(a)}`,
    )
  })
  // A stochastic-gradient family runs on subsampled plates, which the engine's verdict alone would refuse.
  const base = (a: ModelAnalysis): Verdict => {
    const v = baseOf(a)
    return !v.applies && spec.minibatch === true && v.reason.startsWith('subsampled plates')
      ? yes('continuous latents; the subsampled plates give minibatch gradients')
      : v
  }
  // The family's own condition comes after the model's: continuous latents and a differentiable density first.
  const check = (a: ModelAnalysis): Verdict => {
    const v = base(a)
    if (!v.applies) return v
    const own = spec.check?.(a)
    if (own === undefined) return v
    return own.applies ? { ...v, reason: `${v.reason}; ${own.reason}` } : own
  }
  const ensure = guard(name, model, check)

  const extraEntries = (spec.extraPaths ?? {}) as Readonly<Record<string, ReturnType<typeof scalar>>>

  return defineInferenceMethod<McmcRunState, Config, Path, ModelAnalysis>({
    name,
    family: 'mcmc',
    description: spec.description,
    config: knobs,
    check,
    posterior,
    defaults: {
      'mcmc/stage': Log(),
      'mcmc/logDensity': Log(),
      'mcmc/stepSize': Log(),
      'mcmc/accept': Log(),
      'mcmc/divergences': Log(),
      'mcmc/kept': Log(),
      ...Object.fromEntries(Object.keys(extraEntries).map((p) => [p, Log()])),
    } as never,
    contract: (config): Contract<McmcRunState, Path> =>
      defineContract({
        'init/tries': tensor({
          role: 'output',
          shape: ['chains'],
          description: "tries the search for each chain's start made",
        }),
        'init/potential': tensor({
          role: 'output',
          shape: ['chains'],
          description: "the potential energy at each chain's starting point",
        }),
        'mcmc/stage': scalar({ description: '1 while adapting, 2 while drawing, 3 while recovering enumerated sites' }),
        'mcmc/logDensity': tensor({
          role: 'output',
          shape: ['chains'],
          description: 'the log density of each chain now',
        }),
        'mcmc/stepSize': tensor({
          role: 'output',
          shape: ['chains'],
          description: 'the step size, proposal scale or friction-scaled step each chain used',
        }),
        'mcmc/accept': tensor({
          role: 'output',
          shape: ['chains'],
          description: "the acceptance probability of each chain's transition",
        }),
        'mcmc/divergent': tensor({
          role: 'output',
          shape: ['chains'],
          description: '1 for each chain whose transition diverged',
        }),
        'mcmc/treeDepth': tensor({
          role: 'output',
          shape: ['chains'],
          description: 'the NUTS tree depth of each chain',
        }),
        'mcmc/energy': tensor({ role: 'output', shape: ['chains'], description: 'the Hamiltonian of each chain now' }),
        'mcmc/position': tensor({
          role: 'state',
          shape: ['chains', 'dim'],
          description: "every chain's position on the unconstrained scale",
        }),
        'mcmc/divergences': scalar({
          description: 'divergent transitions after warm-up, all chains, so far',
          direction: 'lower',
        }),
        'mcmc/kept': scalar({ description: 'draws kept per chain so far' }),
        'mcmc/windowEnd': message({
          description: 'a chain finished a slow warm-up window: its new step size and inverse mass matrix',
          payload: { chain: 'number', stepSize: 'number', inverseMassMatrix: 'number[]' },
        }),
        'mcmc/site': scalar({ description: 'the draw whose enumerated discrete sites are being recovered' }),
        'mcmc/snapshot': derived<McmcRunState>({
          cost: 'dear',
          description: 'the posterior so far as JSON: means, sds, R-hat and ESS, divergences, parameters',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
        ...extraEntries,
      } as never) as unknown as Contract<McmcRunState, Path>,
    make: (config) => {
      ensure()
      const c = {
        numChains: num(config, 'numChains'),
        numWarmup: num(config, 'numWarmup'),
        numDraws: num(config, 'numDraws'),
        thin: num(config, 'thin'),
        inferDiscrete: flag(config, 'inferDiscrete'),
      }
      const um = unconstrained()
      const plan = planOf(config)
      const dim = um.dim
      const kernels = new Map<number, { readonly tuned: T; readonly kernel: MCMCKernel }>()
      const kernelOf = (i: number, tuned: T): MCMCKernel => {
        const known = kernels.get(i)
        if (known !== undefined && known.tuned === tuned) return known.kernel
        const kernel = plan.kernel(tuned)
        kernels.set(i, { tuned, kernel })
        return kernel
      }
      // The inner algorithms are stepped by hand with a step context that logs nothing (the method's own tape is below).
      const inner = (ctx: StepContext<Path>, t: number, key: Key): StepContext => ({
        t,
        key,
        run: ctx.run,
        log: () => undefined,
        wants: () => false,
        emit: () => undefined,
        taps: ctx.taps,
      })
      let constrainedFor: { readonly rows: readonly Float64Array[]; readonly draws: Record<string, Tensor> } | undefined
      const constrainedDraws = (s: McmcRunState): Record<string, Tensor> => {
        const first = s.chains[0]?.kept ?? []
        if (constrainedFor?.rows !== first) {
          // All the draws are kept when the discrete sites are recovered: a matrix per unit, then the walkers of every unit as chains.
          const flat = Float64Array.from(
            toFlat(
              concat(
                s.chains.map((ch) =>
                  transpose(
                    reshape(
                      sliceAxis(
                        stack(ch.kept.map((r) => asTensor(r))),
                        1,
                        STATS.length,
                        STATS.length + plan.walkers * dim,
                      ),
                      [c.numDraws, plan.walkers, dim],
                    ),
                    [1, 0, 2],
                  ),
                ),
                0,
              ),
            ),
          )
          constrainedFor = { rows: first, draws: constrainDraws(um, flat) as Record<string, Tensor> }
        }
        return constrainedFor.draws
      }
      const enumerated = um.enumerated.length > 0 && c.inferDiscrete
      const logTape = (ctx: StepContext<Path>, stage: number, chains: readonly McmcChainState[]): void => {
        ctx.log('mcmc/stage', stage)
        const column = (i: number): number[] => chains.map((ch) => ch.last[i] ?? Number.NaN)
        const finite = (i: number): boolean => chains.some((ch) => !Number.isNaN(ch.last[i] ?? Number.NaN))
        ctx.log('mcmc/logDensity', column(0))
        ctx.log('mcmc/accept', column(1))
        if (finite(3)) ctx.log('mcmc/divergent', column(3))
        if (finite(4)) ctx.log('mcmc/treeDepth', column(4))
        if (finite(6)) ctx.log('mcmc/energy', column(6))
        if (finite(7)) ctx.log('mcmc/stepSize', column(7))
        ctx.log(
          'mcmc/divergences',
          chains.reduce((a, ch) => a + ch.divergences, 0),
        )
        ctx.log('mcmc/kept', chains[0]?.kept.length ?? 0)
        const states = chains.map(
          (ch) =>
            (ch.warm !== null ? (plan.warm === undefined ? ch.warm : plan.warm.chain(ch.warm)) : ch.chain) as MCMCState,
        )
        if (ctx.wants('mcmc/position'))
          ctx.log(
            'mcmc/position',
            states.flatMap((s) => Array.from(s?.position ?? [])),
          )
        if (plan.extraTape !== undefined && states.every((s) => s !== undefined))
          plan.extraTape(ctx as StepContext<string>, states, stage)
      }
      /** A start for each walker of a unit, joined into the kernel's position. */
      const startOf = (starts: readonly Float64Array[], key: Key): McmcChainState => {
        const joined = Float64Array.from(
          toFlat(
            concat(
              starts.map((s) => asTensor(s)),
              0,
            ),
          ),
        )
        if (c.numWarmup > 0) {
          const warm = plan.warm === undefined ? plan.kernel(plan.tuned0).init(joined) : plan.warm.init(joined, key)
          return { ...emptyChain(), warm }
        }
        const kernel = plan.kernel(plan.tuned0)
        return { ...emptyChain(), chain: kernel.init(joined), tuned: plan.tuned0 }
      }
      /** Warm-up without a family adaptation: plain burn-in transitions of the untuned kernel. */
      const burnIn = (ch: McmcChainState, ci: number, ctx: StepContext<Path>): McmcChainState => {
        const kernel = kernelOf(ci, plan.tuned0)
        const next = kernel.step(foldIn(ctx.key, ci), ch.warm as MCMCState)
        return { ...ch, warm: next.state, last: Float64Array.from(statsOf(next.state.logDensity, next.info)) }
      }
      const algorithm: Algorithm<McmcRunState, Path> = {
        name: `probprog/${name}`,
        config: { numChains: c.numChains, numWarmup: c.numWarmup, numDraws: c.numDraws },
        init: () => {
          ensure()
          return { stage: 'init', iteration: 0, chains: [], discrete: {}, shapes: {}, drawIndex: 0, failure: null }
        },
        step: (s, ctx) => {
          if (s.stage === 'init') {
            const startsByUnit: Float64Array[][] = []
            const tries: number[] = []
            const potentials: number[] = []
            for (let u = 0; u < c.numChains; u++) {
              const starts: Float64Array[] = []
              for (let w = 0; w < plan.walkers; w++) {
                const init = findInitialParams(um, program, undefined, {
                  key: foldIn(foldIn(foldIn(ctx.key, 1), u), w),
                  strategy: initToUniform(),
                })
                tries.push(init.tries)
                potentials.push(init.potential)
                if (!init.valid) {
                  ctx.log('init/tries', tries)
                  ctx.log('init/potential', potentials)
                  return {
                    ...s,
                    stage: 'done',
                    failure: {
                      reason: 'no starting point with a finite log density and gradient',
                      extra: { tries: init.tries },
                    },
                  }
                }
                starts.push(init.unconstrained)
              }
              startsByUnit.push(starts)
            }
            ctx.log('init/tries', tries)
            ctx.log('init/potential', potentials)
            const chains = startsByUnit.map((starts, u) => startOf(starts, foldIn(foldIn(ctx.key, 2), u)))
            return { ...s, chains, stage: c.numWarmup > 0 ? 'warmup' : 'sample', iteration: 0 }
          }
          if (s.stage === 'warmup') {
            const i = s.iteration
            const warmer = plan.warm
            const chains = s.chains.map((ch, ci): McmcChainState => {
              if (warmer === undefined) return burnIn(ch, ci, ctx)
              const warm = warmer.step(ch.warm, inner(ctx, i, foldIn(ctx.key, ci)))
              const info = warmer.info(warm)
              const closed = warmer.windowEnded?.(warm)
              if (closed !== undefined)
                ctx.emit('mcmc/windowEnd', {
                  chain: ci,
                  stepSize: closed.stepSize,
                  inverseMassMatrix: [...closed.inverseMassMatrix],
                })
              return {
                ...ch,
                warm,
                last: info === undefined ? ch.last : Float64Array.from(statsOf(warmer.chain(warm).logDensity, info)),
              }
            })
            if (i + 1 < c.numWarmup) {
              logTape(ctx, 1, chains)
              return { ...s, chains, iteration: i + 1 }
            }
            // Warm-up is over: every chain draws with its tuned kernel from where it stands.
            const sampling = chains.map((ch): McmcChainState => {
              if (warmer === undefined) return { ...ch, chain: ch.warm as MCMCState, warm: null, tuned: plan.tuned0 }
              return { ...ch, chain: warmer.chain(ch.warm), warm: null, tuned: warmer.result(ch.warm) }
            })
            logTape(ctx, 1, sampling)
            return { ...s, chains: sampling, stage: 'sample', iteration: 0 }
          }
          if (s.stage === 'sample') {
            const j = s.iteration
            const keep = (j + 1) % c.thin === 0
            const chains = s.chains.map((ch, ci): McmcChainState => {
              const chain = ch.chain as MCMCState
              const kernel = kernelOf(ci, ch.tuned as T)
              const next = kernel.step(foldIn(foldIn(ctx.key, ci), j), chain)
              const info = next.info
              const stats = statsOf(next.state.logDensity, info)
              return {
                ...ch,
                chain: next.state,
                last: Float64Array.from(stats),
                kept: keep ? [...ch.kept, Float64Array.from([...stats, ...next.state.position])] : ch.kept,
                divergences: ch.divergences + (info.divergent === true ? 1 : 0),
              }
            })
            const full = (chains[0]?.kept.length ?? 0) >= c.numDraws
            logTape(ctx, 2, chains)
            return { ...s, chains, iteration: j + 1, stage: full ? (enumerated ? 'discrete' : 'done') : 'sample' }
          }
          // Recover the enumerated discrete sites at one draw.
          const draws = constrainedDraws(s)
          const i = s.drawIndex
          const values: Record<string, Value> = {}
          for (const l of um.layout) {
            const d = draws[l.name]
            if (d !== undefined) values[l.name] = rowOf(d, i)
          }
          const r = inferDiscrete(program, undefined, values, foldIn(ctx.key, 3))
          const discrete: Record<string, readonly number[]> = { ...s.discrete }
          const shapes: Record<string, readonly number[]> = { ...s.shapes }
          for (const site of um.enumerated) {
            const v = primalOf(r[site] as Tensor)
            shapes[site] = [...v.shape]
            discrete[site] = [...(discrete[site] ?? []), ...toFlat(v)]
          }
          ctx.log('mcmc/stage', 3)
          ctx.log('mcmc/site', i)
          return {
            ...s,
            discrete,
            shapes,
            drawIndex: i + 1,
            stage: i + 1 >= c.numChains * plan.walkers * c.numDraws ? 'done' : 'discrete',
          }
        },
        done: (s) => (s.stage !== 'done' ? undefined : s.failure === null ? 'converged' : 'failed'),
        serialise: (s) => ({
          stage: s.stage,
          iteration: s.iteration,
          discrete: stateToJson(s.discrete),
          shapes: stateToJson(s.shapes),
          drawIndex: s.drawIndex,
          failure: s.failure === null ? null : stateToJson(s.failure),
          chains: s.chains.map((ch) => ({
            warm:
              ch.warm === null
                ? null
                : plan.warm !== undefined
                  ? { kind: 'warm', value: plan.warm.toJson(ch.warm) }
                  : { kind: 'burn', value: chainToJson(ch.warm as MCMCState) },
            chain: ch.chain === null ? null : chainToJson(ch.chain),
            tuned: ch.tuned === null ? null : plan.tunedToJson(ch.tuned as T),
            last: stateToJson(ch.last),
            kept: stateToJson(flatten(ch.kept)),
            width: ch.kept[0]?.length ?? 0,
            divergences: ch.divergences,
          })),
        }),
        deserialise: (j) => {
          const o = j as unknown as Record<string, Json> & { chains: Record<string, Json>[] }
          return {
            stage: o['stage'] as McmcRunState['stage'],
            iteration: o['iteration'] as number,
            discrete: stateFromJson(o['discrete'] as Json) as McmcRunState['discrete'],
            shapes: stateFromJson(o['shapes'] as Json) as McmcRunState['shapes'],
            drawIndex: o['drawIndex'] as number,
            failure: o['failure'] === null ? null : (stateFromJson(o['failure'] as Json) as McmcRunState['failure']),
            chains: o.chains.map((ch): McmcChainState => {
              const warm = ch['warm'] as { kind: string; value: Json } | null
              return {
                warm:
                  warm === null
                    ? null
                    : warm.kind === 'warm' && plan.warm !== undefined
                      ? plan.warm.fromJson(warm.value)
                      : chainFromJson<MCMCState>(warm.value),
                chain: ch['chain'] === null ? null : chainFromJson<MCMCState>(ch['chain'] as Json),
                tuned: ch['tuned'] === null ? null : plan.tunedFromJson(ch['tuned'] as Json),
                last: stateFromJson(ch['last'] as Json) as Float64Array,
                kept: unflatten(stateFromJson(ch['kept'] as Json) as Float64Array, ch['width'] as number),
                divergences: ch['divergences'] as number,
              }
            }),
          }
        },
      }
      return algorithm
    },
  })
}
