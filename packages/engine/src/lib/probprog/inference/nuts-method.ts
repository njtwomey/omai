/**
 * Hamiltonian Monte Carlo and the No-U-Turn Sampler as an inference method with a fine step: one iteration of every chain
 * (a transition, in warm-up an adapted one), so a run cancels within one iteration, publishes its draws so far, step sizes,
 * divergences and R-hat and ESS while it goes on, and resumes from a serialised state. It composes the library's pieces and
 * writes no kernel: `windowAdaptationAlgorithm` (Stan's warm-up, one chain at a time) for the warm-up, `samplerAlgorithm` of
 * the tuned kernel for the draws, `findInitialParams` for the starting points and `inferDiscrete` for enumerated discrete
 * sites. The one-call `mcmc` of `omai-engine/probprog` and its planner entry are unchanged.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  type Chains,
  chainFromJson,
  chainsFromDraws,
  chainToJson,
  diagnose,
  type HamiltonianParameters,
  type HMCState,
  hmcKernel,
  type MCMCKernel,
  massMatrixInit,
  nutsKernel,
  type SamplerStat,
  samplerAlgorithm,
  type WarmupState,
  windowAdaptationAlgorithm,
  windowAdaptationResult,
} from '../../sampling/index.ts'
import { bool, choice, type DictSpace, dict, int, real } from '../../space/index.ts'
import {
  asTensor,
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
  numberFromJson,
  numberToJson,
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
import { definePosterior, type Posterior } from '../posterior.ts'
import {
  checkOf,
  flag,
  guard,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  pick,
  posteriorJson,
  priorOnly,
  programOf,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod } from './method.ts'

/** The paths the fine-grained MCMC method logs. */
export type NutsPath =
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

/** One chain: its warm-up state while adapting, then its sampler state and tuned kernel parameters, and the draws kept. */
export interface NutsChainState {
  /** The warm-up algorithm's state while the chain adapts. */
  readonly warm: WarmupState<HMCState> | null
  /** The chain's state once it samples. */
  readonly chain: HMCState | null
  /** The tuned step size and inverse mass matrix the chain samples with. */
  readonly tuned: HamiltonianParameters | null
  /** The statistics of the last transition, as in the columns of a kept draw. */
  readonly last: Float64Array
  /** The draws kept so far, one row of the eight statistics and then the position each. */
  readonly kept: readonly Float64Array[]
  /** Divergent transitions after warm-up. */
  readonly divergences: number
}

/** The state of {@link nutsMethod}. */
export interface NutsRunState {
  /** `init` (one step finds every chain's start), `warmup`, `sample`, `discrete` (enumerated sites, a draw per step), `done`. */
  readonly stage: 'init' | 'warmup' | 'sample' | 'discrete' | 'done'
  /** Iterations done in the warm-up or sampling stage. */
  readonly iteration: number
  readonly chains: readonly NutsChainState[]
  /** The recovered values of the enumerated sites, draw after draw (flat). */
  readonly discrete: Readonly<Record<string, readonly number[]>>
  readonly shapes: Readonly<Record<string, readonly number[]>>
  /** The next draw whose enumerated sites are recovered. */
  readonly drawIndex: number
  readonly failure: { readonly reason: string; readonly extra: Readonly<Record<string, Json>> } | null
}

/** The knobs of {@link nutsMethod} (the budget and snapshot knobs are added to them). */
export const nutsKnobs: DictSpace = dict({
  algorithm: choice(['nuts', 'hmc'], {
    default: 'nuts',
    label: 'sampler',
    description: 'NUTS, or HMC with a fixed number of integrator steps',
  }),
  numChains: int(1, 64, { default: 2, label: 'chains' }),
  numWarmup: int(0, 1_000_000, { default: 500, label: 'warm-up iterations' }),
  numDraws: int(1, 1_000_000, { default: 500, label: 'draws per chain' }),
  thin: int(1, 1000, { default: 1, label: 'thinning' }),
  numSteps: int(1, 1000, { default: 10, label: 'HMC steps', description: 'integrator steps per HMC transition' }),
  maxTreeDepth: int(1, 15, { default: 10, label: 'maximum tree depth' }),
  targetAcceptance: real(0.5, 0.99, { default: 0.8, label: 'target acceptance' }),
  dense: bool({ default: false, label: 'dense mass matrix' }),
  inferDiscrete: bool({
    default: true,
    label: 'recover enumerated sites',
    description: 'draw the enumerated discrete sites for every draw',
  }),
})

interface Settings {
  readonly algorithm: 'nuts' | 'hmc'
  readonly numChains: number
  readonly numWarmup: number
  readonly numDraws: number
  readonly thin: number
  readonly numSteps: number
  readonly maxTreeDepth: number
  readonly targetAcceptance: number
  readonly dense: boolean
  readonly inferDiscrete: boolean
}

const settings = (config: Config): Settings => ({
  algorithm: pick(config, 'algorithm') === 'hmc' ? 'hmc' : 'nuts',
  numChains: num(config, 'numChains'),
  numWarmup: num(config, 'numWarmup'),
  numDraws: num(config, 'numDraws'),
  thin: num(config, 'thin'),
  numSteps: num(config, 'numSteps'),
  maxTreeDepth: num(config, 'maxTreeDepth'),
  targetAcceptance: num(config, 'targetAcceptance'),
  dense: flag(config, 'dense'),
  inferDiscrete: flag(config, 'inferDiscrete'),
})

const rowOf = (d: Tensor, i: number): Tensor => take(d, i, 0)

interface KernelTransition {
  readonly acceptanceRate: number
  readonly accepted: boolean
  readonly divergent?: boolean | undefined
  readonly treeDepth?: number | undefined
  readonly numSteps?: number | undefined
  readonly energy?: number | undefined
  readonly stepSize?: number | undefined
}

/** A transition's statistics as the columns of a kept row. */
const statsOf = (logDensity: number, info: KernelTransition): number[] => [
  logDensity,
  info.acceptanceRate,
  info.accepted ? 1 : 0,
  info.divergent === undefined ? Number.NaN : info.divergent ? 1 : 0,
  info.treeDepth ?? Number.NaN,
  info.numSteps ?? Number.NaN,
  info.energy ?? Number.NaN,
  info.stepSize ?? Number.NaN,
]

const emptyChain = (): NutsChainState => ({
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

const failedPosterior = (reason: string, extra: Readonly<Record<string, Json>> = {}): Posterior =>
  definePosterior({
    form: 'samples',
    engine: 'nuts',
    sites: [],
    status: 'failed',
    draws: {},
    diagnostics: { reason, ...extra },
  })

/** The chains a state holds so far as the library's `Chains` (the draws every chain has, and the tuned kernels). */
function chainsOf(um: UnconstrainedModel, state: NutsRunState): Chains | undefined {
  const n = Math.min(...state.chains.map((c) => c.kept.length))
  if (!Number.isFinite(n) || n < 1) return undefined
  const dim = um.dim
  // One matrix per chain: a row per kept draw, the sampler statistics then the position.
  const kept = state.chains.map((ch) => stack(ch.kept.slice(0, n).map((r) => asTensor(r))))
  const draws = stack(kept.map((m) => sliceAxis(m, 1, STATS.length, STATS.length + dim))) as Tensor
  const stats = STATS.map((_, k) => stack(kept.map((m) => take(m, k, 1))))
  const base = chainsFromDraws(draws, {
    names: um.logDensity.names,
    stats: Object.fromEntries(
      STATS.flatMap((name, k) =>
        Number.isNaN(toFlat(stats[k] as Value)[0] ?? Number.NaN) ? [] : [[name, stats[k] as Tensor]],
      ),
    ),
  })
  return {
    ...base,
    parameters: state.chains.map((ch) =>
      ch.tuned === null
        ? {}
        : { stepSize: ch.tuned.stepSize, inverseMassMatrix: Array.from(ch.tuned.inverseMassMatrix) },
    ),
  }
}

/**
 * HMC or NUTS on a model as a method with a fine step: each step is one iteration of every chain, so the run cancels
 * within one iteration; the state serialises, so it resumes and forks exactly; `posterior` reads the draws so far as a
 * posterior (with R-hat, ESS and divergences when there are enough draws), which is what an anytime snapshot shows. After a
 * step that finds the starting points the chains warm up for `numWarmup` iterations (Stan's windowed adaptation of the step
 * size and the mass matrix, per chain), then draw `numDraws` kept draws each, then, for a model with enumerated discrete
 * sites, one step per draw recovers them. The session's `steps` is the cap on the whole run (set it above the number of
 * iterations, or give a budget to `runAnytime`): the run ends `converged` after the last stage.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name (default `nuts`).
 * @returns The method, with `check` (a likelihood and continuous latent sites), `posterior` and the budget and snapshot knobs.
 * @concept nuts
 * @index estimator NUTS and HMC as a steppable, cancellable, resumable method with anytime snapshots.
 * @example
 * ```ts
 * import { nutsMethod, runAnytime } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const s = session(nutsMethod(model), { config: { numChains: 1, numWarmup: 30, numDraws: 20 }, steps: 1000 })
 * const r = runAnytime(s, { budget: { steps: 15 } })
 * [r.reason, r.t] // => ['budget', 15]
 * ```
 */
export function nutsMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<NutsRunState, Config, NutsPath, ModelAnalysis> {
  const program = programOf(model)
  // The unconstrained model, built once per method: its prototype run only gives shapes, so no key is needed.
  let umCache: UnconstrainedModel | undefined
  const unconstrained = (): UnconstrainedModel => {
    umCache ??= potentialEnergy(program, undefined)
    return umCache
  }
  const posterior = (state: NutsRunState, config: Config): Posterior => {
    if (state.failure !== null) return failedPosterior(state.failure.reason, state.failure.extra)
    const c = settings(config)
    const um = unconstrained()
    const chains = chainsOf(um, state)
    if (chains === undefined) return failedPosterior('no draws yet', { stage: state.stage })
    const done = state.stage === 'done'
    const draws = constrainDraws(um, chains.draws.data)
    const total = chains.numChains * chains.numDraws
    if (done && um.enumerated.length > 0 && c.inferDiscrete)
      for (const name of um.enumerated)
        draws[name] = reshape(asTensor(state.discrete[name] ?? []), [total, ...(state.shapes[name] ?? [])]) as Tensor
    const diagnostics: Record<string, Json> = {
      partial: !done,
      iterations: state.iteration,
      draws: chains.numDraws,
      divergences: state.chains.reduce((a, ch) => a + ch.divergences, 0),
      stepSizes: chains.parameters.map((p) => (p as { stepSize?: number }).stepSize ?? null),
      enumerated: um.enumerated,
    }
    if (chains.numDraws >= 4) {
      try {
        const report = diagnose(chains, { maxTreeDepth: c.maxTreeDepth })
        diagnostics['highRhat'] = [...report.highRhat]
        diagnostics['lowEss'] = [...report.lowEss]
      } catch {
        // Too few draws for the diagnostics: the summary leaves R-hat and ESS out.
      }
    }
    return definePosterior({
      form: 'samples',
      engine: c.algorithm,
      sites: Object.keys(draws),
      draws,
      chainShape: [chains.numChains, chains.numDraws],
      chains,
      status: done ? 'completed' : 'stopped',
      diagnostics,
    })
  }
  const check = checkOf(nutsEngine(), (a) => {
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
          ? 'continuous latents with a differentiable density; the finite discrete ones are summed out'
          : 'continuous latents with a differentiable density'
      }${priorOnly(a)}`,
    )
  })
  const ensure = guard(options.name ?? 'nuts', model, check)
  return defineInferenceMethod<NutsRunState, Config, NutsPath, ModelAnalysis>({
    name: options.name ?? 'nuts',
    family: 'mcmc',
    description: 'NUTS and HMC, one iteration of every chain per step, with Stan window adaptation',
    config: nutsKnobs,
    check,
    posterior,
    defaults: {
      'mcmc/stage': Log(),
      'mcmc/logDensity': Log(),
      'mcmc/stepSize': Log(),
      'mcmc/accept': Log(),
      'mcmc/divergent': Log(),
      'mcmc/divergences': Log(),
      'mcmc/kept': Log(),
    },
    contract: (config): Contract<NutsRunState, NutsPath> =>
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
        'mcmc/stepSize': tensor({ role: 'output', shape: ['chains'], description: 'the step size each chain used' }),
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
        'mcmc/snapshot': derived<NutsRunState>({
          cost: 'dear',
          description: 'the posterior so far as JSON: means, sds, R-hat and ESS, divergences, step sizes',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config) => {
      ensure()
      const c = settings(config)
      const um = unconstrained()
      const dim = um.dim
      const target = um.logDensity
      const kernelFor = (p: HamiltonianParameters): MCMCKernel<HMCState> =>
        c.algorithm === 'hmc'
          ? hmcKernel(target, { ...p, numIntegrationSteps: c.numSteps })
          : nutsKernel(target, { ...p, maxTreeDepth: c.maxTreeDepth })
      const adaptOptions = { numWarmup: c.numWarmup, dense: c.dense, targetAcceptance: c.targetAcceptance }
      const adaptFor = (start: Float64Array) => windowAdaptationAlgorithm(dim, kernelFor, start, adaptOptions)
      // Stepping and serialising a warm-up state do not depend on the starting position, so one algorithm serves every chain.
      const shared = adaptFor(Float64Array.from(toFlat(zeros([dim]))))
      const kernels = new Map<
        number,
        { readonly tuned: HamiltonianParameters; readonly kernel: MCMCKernel<HMCState> }
      >()
      const kernelOf = (i: number, tuned: HamiltonianParameters): MCMCKernel<HMCState> => {
        const known = kernels.get(i)
        if (known?.tuned.stepSize === tuned.stepSize && known.tuned.inverseMassMatrix === tuned.inverseMassMatrix)
          return known.kernel
        const kernel = kernelFor(tuned)
        kernels.set(i, { tuned, kernel })
        return kernel
      }
      // The inner algorithms are stepped by hand with a step context that logs nothing (the method's own tape is below).
      const inner = (ctx: StepContext<NutsPath>, t: number, key: Key): StepContext => ({
        t,
        key,
        run: ctx.run,
        log: () => undefined,
        wants: () => false,
        emit: () => undefined,
        taps: ctx.taps,
      })
      let constrainedFor: { readonly rows: readonly Float64Array[]; readonly draws: Record<string, Tensor> } | undefined
      const constrainedDraws = (s: NutsRunState): Record<string, Tensor> => {
        const first = s.chains[0]?.kept ?? []
        if (constrainedFor?.rows !== first) {
          // All the draws are kept when the discrete sites are recovered: a matrix per chain, then the chains one after the other.
          const flat = Float64Array.from(
            toFlat(
              concat(
                s.chains.map((ch) =>
                  sliceAxis(stack(ch.kept.map((r) => asTensor(r))), 1, STATS.length, STATS.length + dim),
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
      const logTape = (ctx: StepContext<NutsPath>, stage: number, chains: readonly NutsChainState[]): void => {
        ctx.log('mcmc/stage', stage)
        const column = (i: number): number[] => chains.map((ch) => ch.last[i] ?? Number.NaN)
        ctx.log('mcmc/logDensity', column(0))
        ctx.log('mcmc/accept', column(1))
        ctx.log('mcmc/divergent', column(3))
        ctx.log('mcmc/treeDepth', column(4))
        ctx.log('mcmc/energy', column(6))
        ctx.log('mcmc/stepSize', column(7))
        ctx.log(
          'mcmc/divergences',
          chains.reduce((a, ch) => a + ch.divergences, 0),
        )
        ctx.log('mcmc/kept', chains[0]?.kept.length ?? 0)
        if (ctx.wants('mcmc/position'))
          ctx.log(
            'mcmc/position',
            chains.flatMap((ch) => Array.from((ch.warm?.chain ?? ch.chain)?.position ?? [])),
          )
      }
      const startOf = (start: Float64Array, key: Key): NutsChainState => {
        if (c.numWarmup > 0) {
          const warm = adaptFor(start).init(key)
          const last = Float64Array.from(toFlat(full([STATS.length], Number.NaN)))
          last[7] = warm.stepSize
          return { ...emptyChain(), warm, last }
        }
        const tuned = { stepSize: 1, inverseMassMatrix: massMatrixInit(dim, { dense: c.dense }).inverseMassMatrix }
        return { ...emptyChain(), chain: kernelFor(tuned).init(start), tuned }
      }
      const algorithm: Algorithm<NutsRunState, NutsPath> = {
        name: 'probprog/nuts-method',
        config: { algorithm: c.algorithm, numChains: c.numChains, numWarmup: c.numWarmup, numDraws: c.numDraws },
        init: () => {
          ensure()
          return { stage: 'init', iteration: 0, chains: [], discrete: {}, shapes: {}, drawIndex: 0, failure: null }
        },
        step: (s, ctx) => {
          if (s.stage === 'init') {
            const starts: Float64Array[] = []
            const tries: number[] = []
            const potentials: number[] = []
            for (let i = 0; i < c.numChains; i++) {
              const init = findInitialParams(um, program, undefined, {
                key: foldIn(foldIn(ctx.key, 1), i),
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
            ctx.log('init/tries', tries)
            ctx.log('init/potential', potentials)
            const chains = starts.map((start, i) => startOf(start, foldIn(foldIn(ctx.key, 2), i)))
            return { ...s, chains, stage: c.numWarmup > 0 ? 'warmup' : 'sample', iteration: 0 }
          }
          if (s.stage === 'warmup') {
            const i = s.iteration
            const chains = s.chains.map((ch, ci): NutsChainState => {
              const warm = shared.step(ch.warm as WarmupState<HMCState>, inner(ctx, i, foldIn(ctx.key, ci)))
              const info = warm.info
              if (warm.windowEnded)
                ctx.emit('mcmc/windowEnd', {
                  chain: ci,
                  stepSize: warm.stepSize,
                  inverseMassMatrix: Array.from(warm.mm.inverseMassMatrix),
                })
              return {
                ...ch,
                warm,
                last: info === undefined ? ch.last : Float64Array.from(statsOf(warm.chain.logDensity, info)),
              }
            })
            if (i + 1 < c.numWarmup) {
              logTape(ctx, 1, chains)
              return { ...s, chains, iteration: i + 1 }
            }
            // Warm-up is over: every chain draws with its tuned kernel from where it stands.
            const sampling = chains.map((ch): NutsChainState => {
              const warm = ch.warm as WarmupState<HMCState>
              return { ...ch, warm: null, chain: warm.chain, tuned: windowAdaptationResult(warm) }
            })
            logTape(ctx, 1, sampling)
            return { ...s, chains: sampling, stage: 'sample', iteration: 0 }
          }
          if (s.stage === 'sample') {
            const j = s.iteration
            const keep = (j + 1) % c.thin === 0
            const chains = s.chains.map((ch, ci): NutsChainState => {
              const chain = ch.chain as HMCState
              const sampler = samplerAlgorithm(kernelOf(ci, ch.tuned as HamiltonianParameters), chain.position)
              const next = sampler.step({ state: chain }, inner(ctx, j, foldIn(ctx.key, ci)))
              const info = next.info
              if (info === undefined) throw new OmaiError('value', 'nuts: a transition reported nothing')
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
          for (const name of um.enumerated) {
            const v = primalOf(r[name] as Tensor)
            shapes[name] = [...v.shape]
            discrete[name] = [...(discrete[name] ?? []), ...toFlat(v)]
          }
          ctx.log('mcmc/stage', 3)
          ctx.log('mcmc/site', i)
          return {
            ...s,
            discrete,
            shapes,
            drawIndex: i + 1,
            stage: i + 1 >= c.numChains * c.numDraws ? 'done' : 'discrete',
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
            warm: ch.warm === null ? null : (shared.serialise?.(ch.warm) ?? null),
            chain: ch.chain === null ? null : chainToJson(ch.chain),
            tuned:
              ch.tuned === null
                ? null
                : {
                    stepSize: numberToJson(ch.tuned.stepSize),
                    inverseMassMatrix: stateToJson(ch.tuned.inverseMassMatrix),
                  },
            last: stateToJson(ch.last),
            kept: stateToJson(flatten(ch.kept)),
            width: ch.kept[0]?.length ?? 0,
            divergences: ch.divergences,
          })),
        }),
        deserialise: (j) => {
          const o = j as unknown as Record<string, Json> & { chains: Record<string, Json>[] }
          return {
            stage: o['stage'] as NutsRunState['stage'],
            iteration: o['iteration'] as number,
            discrete: stateFromJson(o['discrete'] as Json) as NutsRunState['discrete'],
            shapes: stateFromJson(o['shapes'] as Json) as NutsRunState['shapes'],
            drawIndex: o['drawIndex'] as number,
            failure: o['failure'] === null ? null : (stateFromJson(o['failure'] as Json) as NutsRunState['failure']),
            chains: o.chains.map((ch): NutsChainState => {
              const tuned = ch['tuned'] as { stepSize: Json; inverseMassMatrix: Json } | null
              return {
                warm: ch['warm'] === null ? null : (shared.deserialise?.(ch['warm'] as Json) ?? null),
                chain: ch['chain'] === null ? null : chainFromJson<HMCState>(ch['chain'] as Json),
                tuned:
                  tuned === null
                    ? null
                    : {
                        stepSize: numberFromJson(tuned.stepSize),
                        inverseMassMatrix: stateFromJson(tuned.inverseMassMatrix) as Float64Array,
                      },
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
