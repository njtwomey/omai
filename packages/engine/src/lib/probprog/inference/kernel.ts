/**
 * Writing your own sampler on the handler layer: `defineKernel` takes a few lines, a `step` that moves a program's
 * trace (using the trace operations `regenerate` and `update`, or anything else) and returns a kernel that runs on a
 * program and data, as a chain of draws or as an `MCMCKernel` of the sampling library (so it works with `sampleChain`,
 * the diagnostics and the invariance tests). Two worked kernels: lightweight Metropolis-Hastings by resimulation
 * (`resimulationKernel`) and a Gaussian random walk on the unconstrained scale (`randomWalkKernel`).
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { bijectors } from '../../dists/index.ts'
import { foldIn, type Key, key as makeKey, normal, uniform } from '../../random/index.ts'
import type { ChainTransition, MCMCKernel, MCMCState } from '../../sampling/index.ts'
import { add, item, stack, sub, sum, type Tensor, type TensorLike, type Value } from '../../tensor/index.ts'
import { potentialEnergy, traceLogJoint } from '../density.ts'
import { isDiscrete, type ProgramTrace, seed, substitute, traceOf } from '../handlers.ts'
import type { Model } from '../primitives.ts'
import {
  assess,
  type Changed,
  latentSites,
  latentValues,
  regenerate,
  type SiteSelection,
  update,
} from '../trace-ops.ts'

/** What a kernel's `step` is given about the program it moves. */
export interface KernelTarget<D> {
  readonly model: Model<D>
  readonly data: D
  /** The names of the latent sample sites, in execution order. */
  readonly latents: readonly string[]
  /** A trace of the program with every latent site drawn from the program itself. */
  initial(key: Key): ProgramTrace
  /** The log joint density of a trace (the sum of its sites' scaled, masked log densities). */
  logDensity(trace: ProgramTrace): number
  /** The log density of a full assignment of the latent sites. */
  assess(values: Readonly<Record<string, TensorLike>>): number
  /** Redraw selected latent sites from the program given the rest; the weight is the Metropolis-Hastings log ratio. */
  regenerate(trace: ProgramTrace, selection: SiteSelection, key: Key): Changed
  /** Change some values and rerun; the weight is the log ratio of new to old density. */
  update(trace: ProgramTrace, changes: Readonly<Record<string, TensorLike>>, key?: Key): Changed
}

/** What a kernel's `step` returns: the next trace, and whether the move was taken (with its acceptance probability). */
export interface KernelMove {
  readonly trace: ProgramTrace
  readonly accepted: boolean
  readonly acceptanceRate: number
}

/** The definition of a kernel: a name, its settings as JSON, and the transition. */
export interface KernelSpec<D> {
  readonly name: string
  readonly config?: Json
  /**
   * One transition: from a trace (and a key) to the next trace (a plain trace counts as accepted).
   *
   * @param key - The key for this transition.
   * @param trace - The current trace.
   * @param target - The program and the trace operations bound to it.
   */
  step(key: Key, trace: ProgramTrace, target: KernelTarget<D>): KernelMove | ProgramTrace
}

/** The state of a bound kernel: the trace and its log density. */
export interface KernelState {
  readonly trace: ProgramTrace
  readonly logDensity: number
}

/** Draws of a chain of a bound kernel. */
export interface KernelDraws {
  /** The latent values by site, stacked on a new first axis `[n, ...shape]`. */
  readonly draws: Readonly<Record<string, Tensor>>
  /** The log density of each draw. */
  readonly logDensity: Float64Array
  /** The mean acceptance probability. */
  readonly acceptanceRate: number
  /** The last state, to continue the chain. */
  readonly state: KernelState
}

/** Options of {@link BoundKernel.run}. */
export interface KernelRunOptions {
  readonly numDraws: number
  readonly numWarmup?: number
  /** Keep every `thin`-th draw after warm-up (default 1). */
  readonly thin?: number
  /** Start from this state (default a draw of the program's latent sites). */
  readonly state?: KernelState
}

/** A kernel bound to a program and its data. */
export interface BoundKernel<D> {
  readonly name: string
  readonly target: KernelTarget<D>
  /** A starting state drawn from the program. */
  init(key: Key): KernelState
  /** One transition. */
  step(key: Key, state: KernelState): { readonly state: KernelState; readonly move: KernelMove }
  /** A chain: warm-up, then `numDraws` draws (every `thin`-th transition). */
  run(key: Key, options: KernelRunOptions): KernelDraws
  /**
   * The kernel as an `MCMCKernel` of the sampling library on the unconstrained coordinates of the continuous latent
   * sites (the layout of `potentialEnergy`), so `sampleChain`, `sampleChains` and the sampler tests accept it.
   *
   * @throws OmaiError with code `value` for programs with discrete latent sites or no latent site.
   */
  asMcmcKernel(): MCMCKernel<MCMCState & { readonly trace: ProgramTrace }>
}

/** A kernel defined by {@link defineKernel}: bind it to a program to run it. */
export interface TraceKernel {
  readonly name: string
  readonly config: Json
  /**
   * Bind the kernel to a program and its data.
   *
   * @param model - The program.
   * @param data - Its data.
   */
  on<D>(model: Model<D>, data: D): BoundKernel<D>
}

function isMove(x: KernelMove | ProgramTrace): x is KernelMove {
  return 'accepted' in x && 'trace' in x
}

/**
 * Accept or reject a proposal by the Metropolis-Hastings rule: the proposal's `logWeight` is the log of the ratio (as
 * the weights of `regenerate` and `update` are), taken with probability `min(1, exp(logWeight))`.
 *
 * @param key - The key for the uniform draw.
 * @param current - The current trace.
 * @param proposal - The proposed trace and its log weight.
 * @returns The move: the proposal when accepted, the current trace otherwise.
 * @concept trace-operations
 * @example
 * ```ts
 * import { observe, regenerate, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { metropolis } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const t = traceOf(seed(model, key(0)), undefined)
 * const move = metropolis(key(2), t, regenerate(model, undefined, t, ['m'], key(1)))
 * move.acceptanceRate <= 1 // => true
 * ```
 */
export function metropolis(
  key: Key,
  current: ProgramTrace,
  proposal: { readonly trace: ProgramTrace; readonly logWeight: number },
): KernelMove {
  const rate = Number.isNaN(proposal.logWeight) ? 0 : Math.min(1, Math.exp(proposal.logWeight))
  const accepted = item(uniform(key, [])) < rate
  return { trace: accepted ? proposal.trace : current, accepted, acceptanceRate: rate }
}

/**
 * Define a sampler on the handler layer, in a few lines: say how one transition moves a trace and get a kernel that
 * runs on any program (a chain of draws, or an `MCMCKernel` of the sampling library). The step sees the program's
 * trace operations (`regenerate`, `update`, `assess`) and the key of the transition.
 *
 * @param spec - The name, the settings and the transition.
 * @returns The kernel, ready to bind to a program with `.on(model, data)`.
 * @concept trace-operations
 * @index function Define your own sampler on the handler layer in a few lines.
 * @example
 * ```ts
 * import { observe, sample } from 'omai-engine/probprog'
 * import { defineKernel, metropolis } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { foldIn, key } from 'omai-engine/random'
 * import { item, mean, tensor } from 'omai-engine/tensor'
 * // The posterior of m given y = 1.2 under m ~ N(0, 1), y ~ N(m, 1) is N(0.6, 0.5).
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1.2) }
 * // Mine: resimulate m from its prior and accept by the likelihood ratio.
 * const mine = defineKernel({
 *   name: 'resimulate-m',
 *   step: (k, trace, target) => metropolis(foldIn(k, 1), trace, target.regenerate(trace, ['m'], foldIn(k, 0))),
 * })
 * const chain = mine.on(model, undefined).run(key(0), { numDraws: 4000, numWarmup: 200 })
 * Math.abs(item(mean(chain.draws['m'] ?? tensor(0))) - 0.6) < 0.1 // => true
 * ```
 */
export function defineKernel<D = void>(spec: KernelSpec<D>): TraceKernel {
  return {
    name: spec.name,
    config: spec.config ?? {},
    on: <E>(model: Model<E>, data: E): BoundKernel<E> => bindKernel(spec as unknown as KernelSpec<E>, model, data),
  }
}

function bindKernel<D>(spec: KernelSpec<D>, model: Model<D>, data: D): BoundKernel<D> {
  // A run under a fixed key finds the latent sites' names; nothing else is taken from it.
  const proto = traceOf(seed(model, makeKey(0)), data)
  const latents = latentSites(proto).map((s) => s.name)
  const target: KernelTarget<D> = {
    model,
    data,
    latents,
    initial: (key) => traceOf(seed(model, key), data),
    logDensity: (t) => item(traceLogJoint(t)),
    assess: (values) => assess(model, data, values),
    regenerate: (trace, selection, key) => regenerate(model, data, trace, selection, key),
    update: (trace, changes, key) => update(model, data, trace, changes, key),
  }
  const stateOf = (trace: ProgramTrace): KernelState => ({ trace, logDensity: target.logDensity(trace) })
  const step = (key: Key, state: KernelState): { readonly state: KernelState; readonly move: KernelMove } => {
    const r = spec.step(key, state.trace, target)
    const move: KernelMove = isMove(r) ? r : { trace: r, accepted: r !== state.trace, acceptanceRate: 1 }
    return { state: move.trace === state.trace ? state : stateOf(move.trace), move }
  }
  const init = (key: Key): KernelState => stateOf(target.initial(key))
  return {
    name: spec.name,
    target,
    init,
    step,
    run: (key, options) => {
      const thin = options.thin ?? 1
      const kRun = foldIn(key, 1)
      let s = options.state ?? init(foldIn(key, 0))
      let accepted = 0
      let moves = 0
      const kept: Record<string, Value[]> = Object.fromEntries(latents.map((n) => [n, [] as Value[]]))
      const logDensity: number[] = []
      const total = (options.numWarmup ?? 0) + options.numDraws * thin
      for (let t = 0; t < total; t++) {
        const r = step(foldIn(kRun, t), s)
        s = r.state
        const afterWarmup = t - (options.numWarmup ?? 0)
        if (afterWarmup < 0) continue
        accepted += r.move.acceptanceRate
        moves += 1
        if ((afterWarmup + 1) % thin !== 0) continue
        const values = latentValues(s.trace)
        for (const n of latents) kept[n]?.push(values[n] as Value)
        logDensity.push(s.logDensity)
      }
      const draws: Record<string, Tensor> = {}
      for (const n of latents) draws[n] = stack(kept[n] ?? []) as Tensor
      return {
        draws,
        logDensity: Float64Array.from(logDensity),
        acceptanceRate: moves === 0 ? 0 : accepted / moves,
        state: s,
      }
    },
    asMcmcKernel: () => {
      if (latents.length === 0 || latentSites(proto).some((s) => isDiscrete(s.fn.support)))
        throw new OmaiError('value', `${spec.name}: asMcmcKernel needs latent sites, all of them continuous`)
      const um = potentialEnergy(model, data, { enumerate: false })
      const names = um.layout.map((l) => l.name)
      const toState = (position: Float64Array, trace: ProgramTrace): MCMCState & { readonly trace: ProgramTrace } => ({
        position,
        logDensity: um.logDensity.logDensity(position),
        trace,
      })
      return {
        name: spec.name,
        dim: um.dim,
        config: spec.config ?? {},
        init: (position) => {
          const p = Float64Array.from(position)
          const values = um.constrain(p)
          const picked = Object.fromEntries(names.map((n) => [n, values[n] as Value]))
          return toState(p, traceOf(substitute(model, picked), data))
        },
        step: (key, state): ChainTransition<MCMCState & { readonly trace: ProgramTrace }> => {
          const r = step(key, { trace: state.trace, logDensity: 0 })
          const values = latentValues(r.state.trace)
          const position = um.unconstrain(Object.fromEntries(names.map((n) => [n, values[n] as Value])))
          return {
            state: toState(position, r.state.trace),
            info: { acceptanceRate: r.move.acceptanceRate, accepted: r.move.accepted },
          }
        },
      }
    },
  }
}

/** Options of {@link resimulationKernel}. */
export interface ResimulationOptions {
  /** The latent sites to move, one at a time (default every latent site). */
  readonly sites?: readonly string[]
}

/**
 * Lightweight Metropolis-Hastings by resimulation (the single-site sampler of Wingate, Stuhlmuller and Goodman, and
 * Gen's `mh(trace, selection)`): each transition visits the sites in turn, redraws one from the program given the rest
 * and accepts by the likelihood ratio. It needs nothing from the model but the program itself, so it runs on every
 * program, discrete latent sites included, and it is the simplest kernel `regenerate` can make.
 *
 * @param options - The sites to move (default all latent sites).
 * @returns The kernel.
 * @concept trace-operations
 * @example
 * ```ts
 * import { observe, sample } from 'omai-engine/probprog'
 * import { resimulationKernel } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1.2) }
 * const chain = resimulationKernel().on(model, undefined).run(key(0), { numDraws: 50 })
 * chain.draws['m']?.shape // => [50]
 * ```
 */
export function resimulationKernel(options: ResimulationOptions = {}): TraceKernel {
  return defineKernel({
    name: 'resimulation',
    config: options.sites === undefined ? {} : { sites: [...options.sites] },
    step: (k, trace, target) => {
      const sites = options.sites ?? target.latents
      let current = trace
      let rate = 0
      let any = false
      sites.forEach((name, i) => {
        const kp = foldIn(foldIn(k, i), 0)
        const move = metropolis(foldIn(foldIn(k, i), 1), current, target.regenerate(current, [name], kp))
        rate += move.acceptanceRate
        any = any || move.accepted
        current = move.trace
      })
      return { trace: current, accepted: any, acceptanceRate: sites.length === 0 ? 1 : rate / sites.length }
    },
  })
}

/** Options of {@link randomWalkKernel}. */
export interface RandomWalkKernelOptions {
  /** The standard deviation of the Gaussian step on the unconstrained scale (default 0.5). */
  readonly scale?: number
  /** The continuous latent sites to move, one at a time (default all of them). */
  readonly sites?: readonly string[]
}

/**
 * A Gaussian random walk on the unconstrained scale, built from `update` and the bijector of each site's support:
 * each transition visits the continuous latent sites in turn, moves one by a normal step in its unconstrained
 * coordinates (so positive, bounded and simplex-valued sites stay in their support) and accepts by the density ratio
 * with the change-of-variables term. Discrete sites are left alone.
 *
 * @param options - The step scale and the sites to move.
 * @returns The kernel.
 * @concept trace-operations
 * @example
 * ```ts
 * import { observe, sample } from 'omai-engine/probprog'
 * import { randomWalkKernel } from 'omai-engine/probprog/inference'
 * import { gamma, poisson } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const r = sample('r', gamma(2, 1)); observe('k', poisson(r), 3) }
 * const chain = randomWalkKernel({ scale: 1 }).on(model, undefined).run(key(0), { numDraws: 40 })
 * chain.draws['r']?.shape // => [40]
 * ```
 */
export function randomWalkKernel(options: RandomWalkKernelOptions = {}): TraceKernel {
  const scale = options.scale ?? 0.5
  return defineKernel({
    name: 'random-walk',
    config: { scale, ...(options.sites === undefined ? {} : { sites: [...options.sites] }) },
    step: (k, trace, target) => {
      const names = options.sites ?? target.latents
      let current = trace
      let rate = 0
      let count = 0
      let any = false
      names.forEach((name, i) => {
        const site = current.sites.get(name)
        if (site?.type !== 'sample' || site.isObserved || isDiscrete(site.fn.support)) return
        const kn = foldIn(foldIn(k, i), 0)
        const b = bijectors.bijectorForSupport(site.fn.support)
        const u = b.inverse(site.value)
        const v = add(u, normal(kn, [...u.shape], { scale }))
        const next = target.update(current, { [name]: b.forward(v) })
        const jacobian = item(sum(sub(b.forwardLogDetJacobian(v), b.forwardLogDetJacobian(u))))
        const move = metropolis(foldIn(foldIn(k, i), 1), current, {
          trace: next.trace,
          logWeight: next.logWeight + jacobian,
        })
        rate += move.acceptanceRate
        count += 1
        any = any || move.accepted
        current = move.trace
      })
      return { trace: current, accepted: any, acceptanceRate: count === 0 ? 1 : rate / count }
    },
  })
}
