/**
 * Parallel tempering (replica exchange; Swendsen and Wang 1986, Geyer 1991): replicas of a chain run at a ladder of inverse
 * temperatures, the replica at temperature index i targeting pi(x)^beta_i, so the hot ones cross the barriers between modes
 * that the cold one (beta = 1, the target) cannot, and neighbouring replicas swap their states by a Metropolis rule that keeps
 * the product of the tempered densities invariant. It wraps any kernel: the caller says how to build a kernel on a tempered
 * target. Includes the tempered target itself, geometric and linear ladders, and the swap statistics (rates per pair and round
 * trips of a replica between the hottest and the coldest) that say whether the ladder is good.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn } from '../random/index.ts'
import { mul } from '../tensor/index.ts'
import { uniformOne } from './_mcmc.ts'
import type { MCMCKernel, MCMCState, TransitionInfo } from './chains.ts'
import type { LogDensity } from './target.ts'

/**
 * The target at an inverse temperature: log density and gradient multiplied by beta (the density raised to the power beta).
 *
 * @param target - The target.
 * @param beta - The inverse temperature (positive).
 * @returns The tempered target, with the same coordinates and constraints.
 * @throws OmaiError with code `value` for a non-positive beta.
 * @concept parallel-tempering
 * @example
 * ```ts
 * import { numericTarget, temperedTarget } from 'omai-engine/sampling'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * temperedTarget(t, 0.25).logDensity([2]) // => -0.5
 * ```
 */
export function temperedTarget(target: LogDensity, beta: number): LogDensity {
  if (!(beta > 0) || !Number.isFinite(beta))
    throw new OmaiError('value', 'temperedTarget: beta must be positive and finite')
  const fn = target.fn
  return {
    dim: target.dim,
    names: target.names,
    hasGradient: target.hasGradient,
    ...(target.constraints === undefined ? {} : { constraints: target.constraints }),
    logDensity: (x) => beta * target.logDensity(x),
    valueAndGrad: (x) => {
      const r = target.valueAndGrad(x)
      return { value: beta * r.value, grad: r.grad.map((g) => beta * g) }
    },
    constrain: (x) => target.constrain(x),
    unconstrain: (y) => target.unconstrain(y),
    ...(fn === undefined ? {} : { fn: (x) => mul(beta, fn(x)) }),
  }
}

/**
 * A ladder of inverse temperatures from 1 (the target) down to `minBeta`: geometric (equal ratios, the usual choice, which
 * for a Gaussian target gives equal swap rates between neighbours) or linear.
 *
 * @param count - The number of temperatures (at least 2).
 * @param minBeta - The hottest inverse temperature, in (0, 1).
 * @param scheme - `geometric` (default) or `linear`.
 * @returns The inverse temperatures, decreasing, the first 1.
 * @throws OmaiError with code `value` for fewer than two temperatures or a minimum outside (0, 1).
 * @concept parallel-tempering
 * @example
 * ```ts
 * import { temperatureLadder } from 'omai-engine/sampling'
 * temperatureLadder(3, 0.25) // => [1, 0.5, 0.25]
 * ```
 */
export function temperatureLadder(
  count: number,
  minBeta: number,
  scheme: 'geometric' | 'linear' = 'geometric',
): number[] {
  if (!(Number.isInteger(count) && count >= 2))
    throw new OmaiError('value', 'temperatureLadder: need at least 2 temperatures')
  if (!(minBeta > 0 && minBeta < 1)) throw new OmaiError('value', 'temperatureLadder: minBeta must lie in (0, 1)')
  return Array.from({ length: count }, (_, i) =>
    scheme === 'geometric' ? minBeta ** (i / (count - 1)) : 1 - ((1 - minBeta) * i) / (count - 1),
  )
}

/** The state of {@link parallelTemperingKernel}: the cold replica's position (the draw) and every replica with the swap statistics. */
export interface PTState extends MCMCState {
  /** The replicas' states, index 0 the cold one (beta = 1), each a state of the inner kernel on its tempered target. */
  readonly replicas: readonly MCMCState[]
  /** The inverse temperatures. */
  readonly betas: Float64Array
  /** The transitions taken (the swap schedule). */
  readonly t: number
  /** For each adjacent pair, whether the last swap proposal was accepted (1), rejected (0) or not attempted (NaN). */
  readonly swapAccepted: Float64Array
  /** For each adjacent pair, the acceptance probability of the last proposal (NaN when not attempted). */
  readonly swapProbability: Float64Array
  /** Cumulative swap proposals and acceptances per adjacent pair. */
  readonly swapAttempts: Float64Array
  readonly swapSuccesses: Float64Array
  /** Which walker (a label that travels with the state) sits at each temperature. */
  readonly labels: Int32Array
  /** The last end each walker visited: 1 the cold end, -1 the hot end, 0 neither yet. */
  readonly lastEnd: Int8Array
  /** Walkers that went from the hot end to the cold end (round trips, half counted by arrivals at the cold end). */
  readonly roundTrips: number
}

/** Options of {@link parallelTemperingKernel}. */
export interface ParallelTemperingOptions {
  /** The inverse temperatures, strictly decreasing, the first 1 (see {@link temperatureLadder}). */
  readonly betas: readonly number[]
  /** The kernel for the replica at an inverse temperature, on that replica's tempered target. */
  readonly kernelFor: (target: LogDensity, beta: number, replica: number) => MCMCKernel
  /** `even-odd` (default): alternate the even and the odd adjacent pairs, which moves replicas in a directed way; `random`: one random pair. */
  readonly swapScheme?: 'even-odd' | 'random'
  /** Propose swaps every this many transitions (default 1). */
  readonly swapEvery?: number
}

/**
 * Parallel tempering over any kernel: each transition moves every replica with its own kernel on its tempered target, then
 * proposes swaps between adjacent replicas, accepted with probability min(1, exp((beta_i - beta_j)(log pi(x_j) - log pi(x_i)))).
 * The swap moves leave the product of the tempered densities invariant, so the cold replica (beta = 1) is a chain of the
 * target, and its draws are the kernel's draws; the state, the acceptance and the divergence flags reported are the cold
 * replica's. The state keeps the swap acceptances (last and cumulative, per adjacent pair) and the round trips of walkers
 * between the hottest and the coldest replica, the diagnostics of a ladder. All replicas start at the position given to `init`.
 *
 * @param target - The target (with a gradient if the inner kernel needs one).
 * @param options - The ladder, the kernel at each temperature, the swap scheme and the swap interval.
 * @returns The kernel; `position` and `logDensity` of its state are the cold replica's.
 * @throws OmaiError with code `value` for fewer than two temperatures, a ladder that is not decreasing from 1, or a
 *   non-positive swap interval.
 * @concept parallel-tempering
 * @reference R. H. Swendsen and J.-S. Wang, Replica Monte Carlo simulation of spin-glasses (PRL 1986); C. J. Geyer, Markov
 *   chain Monte Carlo maximum likelihood (1991); D. J. Earl and M. W. Deem, Parallel tempering (PCCP 2005).
 * @index function Parallel tempering (replica exchange) over any kernel, with swap statistics.
 * @example
 * ```ts
 * import { parallelTemperingKernel, randomWalkMetropolis, numericTarget, temperatureLadder } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const k = parallelTemperingKernel(t, {
 *   betas: temperatureLadder(3, 0.25),
 *   kernelFor: (target, beta) => randomWalkMetropolis(target, { scale: 2 / Math.sqrt(beta) }),
 * })
 * k.step(key(0), k.init([0.5])).state.replicas.length // => 3
 * ```
 */
export function parallelTemperingKernel(target: LogDensity, options: ParallelTemperingOptions): MCMCKernel<PTState> {
  const betas = Float64Array.from(options.betas)
  const r = betas.length
  if (r < 2) throw new OmaiError('value', 'parallelTemperingKernel: need at least two temperatures')
  if (betas[0] !== 1) throw new OmaiError('value', 'parallelTemperingKernel: the first inverse temperature must be 1')
  for (let i = 1; i < r; i++)
    if (!((betas[i] ?? 0) > 0 && (betas[i] ?? 0) < (betas[i - 1] ?? 0)))
      throw new OmaiError('value', 'parallelTemperingKernel: the inverse temperatures must decrease and stay positive')
  const every = options.swapEvery ?? 1
  if (!(Number.isInteger(every) && every >= 1))
    throw new OmaiError('value', 'parallelTemperingKernel: swapEvery must be a positive integer')
  const scheme = options.swapScheme ?? 'even-odd'
  const kernels = Array.from(betas, (b, i) => options.kernelFor(temperedTarget(target, b), b, i))
  const dim = target.dim
  const pairs = r - 1
  const stateOf = (replicas: readonly MCMCState[], from: Partial<PTState>, extra: Partial<PTState>): PTState => ({
    position: (replicas[0] as MCMCState).position,
    logDensity: (replicas[0] as MCMCState).logDensity,
    replicas,
    betas,
    t: from.t ?? 0,
    swapAccepted: from.swapAccepted ?? new Float64Array(pairs).fill(Number.NaN),
    swapProbability: from.swapProbability ?? new Float64Array(pairs).fill(Number.NaN),
    swapAttempts: from.swapAttempts ?? new Float64Array(pairs),
    swapSuccesses: from.swapSuccesses ?? new Float64Array(pairs),
    labels: from.labels ?? Int32Array.from({ length: r }, (_, i) => i),
    lastEnd: from.lastEnd ?? Int8Array.from({ length: r }, (_, i) => (i === 0 ? 1 : i === r - 1 ? -1 : 0)),
    roundTrips: from.roundTrips ?? 0,
    ...extra,
  })
  return {
    name: 'parallel-tempering',
    dim,
    config: { dim, betas: Array.from(betas), swapScheme: scheme, swapEvery: every },
    init: (position) => {
      if (position.length !== dim)
        throw new OmaiError('shape', `parallelTemperingKernel: the position needs ${dim} values`)
      return stateOf(
        kernels.map((k) => k.init(position)),
        {},
        {},
      )
    },
    step: (key, state) => {
      const moved = kernels.map((k, i) => k.step(foldIn(foldIn(key, 0), i), state.replicas[i] as MCMCState))
      const replicas: MCMCState[] = moved.map((m) => m.state)
      const swapAccepted = new Float64Array(pairs).fill(Number.NaN)
      const swapProbability = new Float64Array(pairs).fill(Number.NaN)
      const swapAttempts = Float64Array.from(state.swapAttempts)
      const swapSuccesses = Float64Array.from(state.swapSuccesses)
      const labels = Int32Array.from(state.labels)
      const lastEnd = Int8Array.from(state.lastEnd)
      let roundTrips = state.roundTrips
      const t = state.t + 1
      if (t % every === 0) {
        const attempt: number[] = []
        if (scheme === 'random') attempt.push(Math.min(pairs - 1, Math.floor(uniformOne(foldIn(key, 1)) * pairs)))
        else for (let i = Math.floor(t / every) % 2; i < pairs; i += 2) attempt.push(i)
        for (const i of attempt) {
          const a = replicas[i] as MCMCState
          const b = replicas[i + 1] as MCMCState
          const bi = betas[i] as number
          const bj = betas[i + 1] as number
          // The untempered log densities: the replica's density is beta times the target's.
          const logI = a.logDensity / bi
          const logJ = b.logDensity / bj
          const logAlpha = (bi - bj) * (logJ - logI)
          const alpha = Number.isNaN(logAlpha) ? 0 : logAlpha >= 0 ? 1 : Math.exp(logAlpha)
          swapProbability[i] = alpha
          swapAttempts[i] = (swapAttempts[i] ?? 0) + 1
          const accepted = alpha >= 1 || uniformOne(foldIn(foldIn(key, 2), i)) < alpha
          swapAccepted[i] = accepted ? 1 : 0
          if (!accepted) continue
          swapSuccesses[i] = (swapSuccesses[i] ?? 0) + 1
          // Exchange the positions: each replica re-initialises at the other's position under its own temperature.
          const pa = a.position
          const pb = b.position
          replicas[i] = (kernels[i] as MCMCKernel).init(pb)
          replicas[i + 1] = (kernels[i + 1] as MCMCKernel).init(pa)
          const la = labels[i] as number
          labels[i] = labels[i + 1] as number
          labels[i + 1] = la
        }
        // Round trips: a walker that reached the hot end and then the cold end.
        const cold = labels[0] as number
        const hot = labels[r - 1] as number
        if (lastEnd[cold] === -1) roundTrips += 1
        lastEnd[cold] = 1
        lastEnd[hot] = -1
      }
      const info = (moved[0] as { info: TransitionInfo }).info
      return {
        state: stateOf(
          replicas,
          { t, swapAccepted, swapProbability, swapAttempts, swapSuccesses, labels, lastEnd, roundTrips },
          {},
        ),
        info,
      }
    },
  }
}
