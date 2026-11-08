/**
 * The No-U-Turn Sampler (NUTS), the adaptive form of HMC that doubles its trajectory in random directions until it
 * starts to turn back, so the path length need not be tuned: the multinomial NUTS kernel (Hoffman and Gelman,
 * Betancourt; as BlackJAX and NumPyro), its window adaptation, and `nuts` for several chains after Stan-style warm-up.
 */

import { logAddExp, OmaiError } from '../../base/index.ts'
import { foldIn, uniformStream } from '../../random/index.ts'
import { checkStepSize } from '../_mcmc.ts'
import { type Adaptation, type Chains, type MCMCKernel, type SampleChainsOptions, sampleChains } from '../chains.ts'
import type { LogDensity } from '../target.ts'
import { type WindowAdaptationOptions, windowAdaptation } from './adaptation.ts'
import { type HamiltonianOptions, type HMCState, hmcState } from './hmc.ts'
import {
  type EuclideanMetric,
  euclideanMetric,
  hamiltonian,
  type Integrator,
  type IntegratorState,
  integrator,
} from './integrators.ts'

/** A candidate next state with the log of its subtree's total weight and summed acceptance probabilities. */
interface Candidate {
  readonly state: IntegratorState
  readonly energy: number
  /** log sum of exp(H0 - H) over the subtree. */
  readonly weight: number
  /** log sum of min(1, exp(H0 - H)) over the subtree. */
  readonly sumLogAccept: number
}

/** The ends of a trajectory, the sum of its momenta and its number of integrated states. */
interface Trajectory {
  readonly left: IntegratorState
  readonly right: IntegratorState
  readonly momentumSum: Float64Array
  readonly numStates: number
}

interface Subtree {
  readonly candidate: Candidate
  readonly trajectory: Trajectory
  readonly diverging: boolean
  readonly turning: boolean
}

function addVectors(a: Float64Array, b: Float64Array): Float64Array {
  return a.map((v, i) => v + (b[i] ?? 0))
}

function merge(left: Trajectory, right: Trajectory): Trajectory {
  return {
    left: left.left,
    right: right.right,
    momentumSum: addVectors(left.momentumSum, right.momentumSum),
    numStates: left.numStates + right.numStates,
  }
}

/** The U-turn criterion on a merged trajectory, with Stan's extra checks across the two halves. */
function turnsAcross(metric: EuclideanMetric, left: Trajectory, right: Trajectory, merged: Trajectory): boolean {
  if (metric.isTurning(merged.left.momentum, merged.right.momentum, merged.momentumSum)) return true
  const sumLeft = addVectors(left.momentumSum, right.left.momentum)
  if (metric.isTurning(left.left.momentum, right.left.momentum, sumLeft)) return true
  const sumRight = addVectors(right.momentumSum, left.right.momentum)
  return metric.isTurning(left.right.momentum, right.right.momentum, sumRight)
}

/** Options of {@link nutsKernel}. */
export interface NUTSOptions extends HamiltonianOptions {
  /** The most trajectory doublings per transition (default 10: at most 1023 integrator steps). */
  readonly maxTreeDepth?: number
}

/**
 * The No-U-Turn Sampler with multinomial sampling (Hoffman and Gelman 2014, Betancourt 2017; BlackJAX and NumPyro
 * `nuts`): the trajectory doubles in a random direction until it makes a U-turn (the generalised criterion, checked
 * on every subtree and across the halves of each merge, as Stan does), diverges, or reaches the depth limit. The next
 * state is drawn from all states of the trajectory in proportion to exp(-H): uniform progressive sampling within
 * subtrees and biased progressive sampling (favouring the new half) when a subtree joins the trajectory. The reported
 * acceptance rate is the mean of min(1, exp(H0 - H)) over the integrated states, the statistic dual averaging tunes.
 *
 * @param target - The target (with a gradient).
 * @param options - Step size, inverse mass matrix, integrator, divergence threshold and maximum tree depth.
 * @returns The kernel.
 * @throws OmaiError with code `value` for a target without a gradient, a non-positive step size or a maximum
 *   tree depth outside [1, 30].
 * @concept nuts
 * @reference M. D. Hoffman and A. Gelman, The No-U-Turn Sampler (JMLR 2014); M. Betancourt, A conceptual introduction
 *   to Hamiltonian Monte Carlo (2017), appendix A.
 * @example
 * ```ts
 * import { nutsKernel, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const k = nutsKernel(t, { stepSize: 0.5 })
 * const { info } = k.step(key(0), k.init([0.5]))
 * // k doublings integrate at most 1 + 2 + ... + 2^(k-1) = 2^k - 1 states.
 * (info.numSteps ?? 0) <= 2 ** (info.treeDepth ?? 0) - 1 // => true
 * ```
 */
export function nutsKernel(target: LogDensity, options: NUTSOptions): MCMCKernel<HMCState> {
  const { stepSize } = options
  checkStepSize(stepSize, 'nutsKernel')
  const maxDepth = options.maxTreeDepth ?? 10
  if (!(Number.isInteger(maxDepth) && maxDepth >= 1 && maxDepth <= 30)) {
    throw new OmaiError('value', 'nutsKernel: maxTreeDepth must be an integer in [1, 30]')
  }
  const metric = euclideanMetric(options.inverseMassMatrix ?? 1, target.dim)
  const step: Integrator = integrator(target, metric, options.integrator ?? 'velocityVerlet')
  const threshold = options.divergenceThreshold ?? 1000

  const candidate = (state: IntegratorState, h0: number): Candidate => {
    const energy = hamiltonian(state, metric)
    const raw = h0 - energy
    const delta = Number.isNaN(raw) ? Number.NEGATIVE_INFINITY : raw
    return { state, energy, weight: delta, sumLogAccept: Math.min(delta, 0) }
  }

  const build = (start: IntegratorState, direction: 1 | -1, depth: number, h0: number, rand: () => number): Subtree => {
    if (depth === 0) {
      const next = step(start, direction * stepSize)
      const c = candidate(next, h0)
      return {
        candidate: c,
        trajectory: { left: next, right: next, momentumSum: Float64Array.from(next.momentum), numStates: 1 },
        diverging: -c.weight > threshold,
        turning: false,
      }
    }
    const first = build(start, direction, depth - 1, h0, rand)
    if (first.diverging || first.turning) return first
    const from = direction > 0 ? first.trajectory.right : first.trajectory.left
    const second = build(from, direction, depth - 1, h0, rand)
    const [left, right] = direction > 0 ? [first.trajectory, second.trajectory] : [second.trajectory, first.trajectory]
    const merged = merge(left, right)
    const turning = second.turning || (!second.diverging && turnsAcross(metric, left, right, merged))
    // Uniform progressive sampling inside a subtree: the new half's state with probability w_new / (w_old + w_new).
    const a = first.candidate
    const b = second.candidate
    const pNew = 1 / (1 + Math.exp(a.weight - b.weight))
    const chosen = rand() < pNew ? b : a
    return {
      candidate: {
        state: chosen.state,
        energy: chosen.energy,
        weight: logAddExp(a.weight, b.weight),
        sumLogAccept: logAddExp(a.sumLogAccept, b.sumLogAccept),
      },
      trajectory: merged,
      diverging: second.diverging,
      turning,
    }
  }

  return {
    name: 'nuts',
    dim: target.dim,
    config: { dim: target.dim, stepSize, maxTreeDepth: maxDepth, divergenceThreshold: threshold },
    init: (position) => hmcState(target, position),
    step: (key, state) => {
      const momentum = metric.sampleMomentum(foldIn(key, 0))
      const rand = uniformStream(foldIn(key, 1), 64)
      const initial: IntegratorState = {
        position: state.position,
        momentum,
        logDensity: state.logDensity,
        grad: state.grad,
      }
      const h0 = hamiltonian(initial, metric)
      let proposal: Candidate = { state: initial, energy: h0, weight: 0, sumLogAccept: Number.NEGATIVE_INFINITY }
      let trajectory: Trajectory = {
        left: initial,
        right: initial,
        momentumSum: Float64Array.from(momentum),
        numStates: 0,
      }
      let depth = 0
      let diverging = false
      while (depth < maxDepth) {
        const direction: 1 | -1 = rand() < 0.5 ? 1 : -1
        const sub = build(direction > 0 ? trajectory.right : trajectory.left, direction, depth, h0, rand)
        if (sub.diverging || sub.turning) {
          proposal = { ...proposal, sumLogAccept: logAddExp(proposal.sumLogAccept, sub.candidate.sumLogAccept) }
        } else {
          // Biased progressive sampling: move to the new subtree's state with probability min(1, w_new / w_old).
          const pNew = Math.min(1, Math.exp(sub.candidate.weight - proposal.weight))
          const take = rand() < pNew
          proposal = {
            state: take ? sub.candidate.state : proposal.state,
            energy: take ? sub.candidate.energy : proposal.energy,
            weight: logAddExp(proposal.weight, sub.candidate.weight),
            sumLogAccept: logAddExp(proposal.sumLogAccept, sub.candidate.sumLogAccept),
          }
        }
        const [left, right] = direction > 0 ? [trajectory, sub.trajectory] : [sub.trajectory, trajectory]
        const merged = merge(left, right)
        depth += 1
        diverging = sub.diverging
        const stop = sub.diverging || sub.turning || turnsAcross(metric, left, right, merged)
        trajectory = merged
        if (stop) break
      }
      const chosen = proposal.state
      const accepted = chosen !== initial
      return {
        state: accepted ? { position: chosen.position, logDensity: chosen.logDensity, grad: chosen.grad } : state,
        info: {
          acceptanceRate: Math.exp(proposal.sumLogAccept) / Math.max(1, trajectory.numStates),
          accepted,
          divergent: diverging,
          treeDepth: depth,
          numSteps: trajectory.numStates,
          energy: proposal.energy,
          stepSize,
        },
      }
    },
  }
}

/** Options of {@link nutsAdaptation}: NUTS's fixed settings and the window adaptation's. */
export type NUTSAdaptationOptions = Omit<NUTSOptions, 'stepSize' | 'inverseMassMatrix'> & WindowAdaptationOptions

/**
 * Window adaptation of NUTS's step size and mass matrix (Stan's warm-up, BlackJAX `window_adaptation(nuts, ...)`).
 *
 * @param target - The target (with a gradient).
 * @param options - Maximum tree depth, integrator, and the adaptation options.
 * @returns The adaptation.
 * @concept window-adaptation
 * @example
 * ```ts
 * import { nutsAdaptation, numericTarget } from 'omai-engine/sampling'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * nutsAdaptation(t).dim // => 1
 * ```
 */
export function nutsAdaptation(target: LogDensity, options: NUTSAdaptationOptions = {}): Adaptation<HMCState> {
  return windowAdaptation(target.dim, (p) => nutsKernel(target, { ...options, ...p }), options)
}

/** Options of {@link nuts}. */
export type NUTSRunOptions = NUTSAdaptationOptions & Omit<SampleChainsOptions, 'names'>

/**
 * Sample a target with NUTS after Stan-style window adaptation, over several chains (NumPyro's `MCMC(NUTS(...))`):
 * 1000 warm-up iterations and 1000 draws per chain, 4 chains, unless given. Per-draw statistics include divergences,
 * tree depth, step counts and energies for `diagnose`.
 *
 * @param target - The target (with a gradient).
 * @param options - Tree depth, adaptation options, and chain options (key, chains, draws, warm-up).
 * @returns The chains, on the target's unconstrained coordinates (see `constrainChains`).
 * @concept nuts
 * @index function No-U-Turn sampler chains.
 * @example
 * ```ts
 * import { nuts, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * Object.keys(nuts(t, { key: key(0), numChains: 1, numWarmup: 50, numDraws: 20 }).stats).includes('treeDepth') // => true
 * ```
 */
export function nuts(target: LogDensity, options: NUTSRunOptions): Chains {
  return sampleChains(nutsAdaptation(target, options), {
    numWarmup: 1000,
    ...options,
    names: target.names,
  })
}
