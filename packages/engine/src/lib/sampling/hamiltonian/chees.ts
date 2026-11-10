/**
 * ChEES-HMC (Hoffman, Radul and Sountsov 2021; BlackJAX `chees_adaptation`): Hamiltonian Monte Carlo for many parallel chains
 * with a jittered trajectory length, whose length is adapted by stochastic gradient ascent on the ChEES criterion (the change in the
 * estimator of the expected square of the position, a proxy for the effective sample size per gradient) and whose step size is
 * adapted by dual averaging on the harmonic mean of the chains' acceptance probabilities. The chains of one ensemble share the
 * step size, the trajectory length and the jitter, and the adaptation reads statistics across them, so the sampler is the one for
 * a population of chains. The kernel runs on the product space of the chains (the position holds every chain, row-major
 * `[chains, dim]`), like the ensemble samplers.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { foldIn, normal, uniform } from '../../random/index.ts'
import {
  type Algorithm,
  channels,
  numberFromJson,
  numberToJson,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'
import { type AdaptPath, chainFromJson, chainToJson } from '../adaptation-contract.ts'
import type { MCMCKernel, MCMCState } from '../chains.ts'
import type { LogDensity } from '../target.ts'
import { type DualAveragingState, dualAveragingInit, dualAveragingUpdate } from './adaptation.ts'

/** The state of {@link cheesKernel}: every chain's position, log density and gradient, and what the last transition did (for the adaptation). */
export interface CheesState extends MCMCState {
  readonly walkerLogDensity: Float64Array
  readonly grads: Float64Array
  /** The transitions taken: the index of the jitter sequence. */
  readonly t: number
  /** The positions and momenta at the end of the last trajectories (before the accept step), and the positions they started from. */
  readonly proposals: Float64Array
  readonly momenta: Float64Array
  readonly initials: Float64Array
  /** The Metropolis acceptance probability of each chain's last proposal, and whether it diverged (1). */
  readonly acceptProbs: Float64Array
  readonly divergent: Uint8Array
  /** The integrator steps of the last trajectory. */
  readonly numSteps: number
  /** The trajectory length the kernel was built with (for the tape). */
  readonly trajectoryLength: number
}

/** The base-2 Halton (van der Corput) sequence: the (i + 1)-th element, a quasi-random number in (0, 1). */
function halton(i: number, maxBits = 30): number {
  let n = i + 1
  let f = 0.5
  let r = 0
  for (let b = 0; b < maxBits && n > 0; b++) {
    r += f * (n % 2)
    n = Math.floor(n / 2)
    f /= 2
  }
  return r
}

/** Options of {@link cheesKernel}. */
export interface CheesKernelOptions {
  /** The number of chains in the ensemble. */
  readonly numChains: number
  /** The integrator step size (positive), shared by the chains. */
  readonly stepSize: number
  /** The trajectory length: steps times step size (positive). */
  readonly trajectoryLength: number
  /** Jitter the number of steps by a Halton fraction of the trajectory (default true). */
  readonly jitter?: boolean
  /** The energy error above which a transition counts as divergent (default 1000). */
  readonly divergenceThreshold?: number
}

/**
 * Jittered HMC for an ensemble of chains: each transition draws a number of leapfrog steps ceil(h T / eps) with h the next element of
 * a Halton sequence (the same for every chain, as a vectorised sampler has it), and moves every chain by HMC with that many steps
 * and its own momentum. The chains are independent Markov chains of the target, each leaving it invariant; what the ensemble
 * shares is the step size and the trajectory length, which {@link cheesAdaptationAlgorithm} tunes from all of them. The state keeps
 * the last trajectories' end points for that adaptation.
 *
 * @param target - The target (with a gradient).
 * @param options - Chains, step size, trajectory length, jitter and divergence threshold.
 * @returns The kernel on the product space of the chains.
 * @throws OmaiError with code `value` for a target without a gradient, fewer than one chain or non-positive sizes.
 * @concept chees-hmc
 * @index function Jittered HMC for an ensemble of chains with a shared step size and trajectory length.
 * @example
 * ```ts
 * import { cheesKernel, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const k = cheesKernel(t, { numChains: 3, stepSize: 0.3, trajectoryLength: 1.5 })
 * k.step(key(0), k.init([0.5, -0.5, 1])).state.acceptProbs.length // => 3
 * ```
 */
export function cheesKernel(target: LogDensity, options: CheesKernelOptions): MCMCKernel<CheesState> {
  const { numChains: c, stepSize: eps, trajectoryLength: length } = options
  if (!(Number.isInteger(c) && c >= 1)) throw new OmaiError('value', 'cheesKernel: need at least one chain')
  if (!(eps > 0 && length > 0))
    throw new OmaiError('value', 'cheesKernel: the step size and trajectory length must be positive')
  if (!target.hasGradient) throw new OmaiError('value', 'cheesKernel: the target has no gradient')
  const dim = target.dim
  const jitter = options.jitter ?? true
  const threshold = options.divergenceThreshold ?? 1000
  const kinetic = (p: Float64Array): number => 0.5 * p.reduce((s, v) => s + v * v, 0)
  return {
    name: 'chees-hmc',
    dim: c * dim,
    config: { dim, numChains: c, stepSize: eps, trajectoryLength: length, jitter },
    init: (position) => {
      if (position.length !== c * dim) {
        throw new OmaiError(
          'shape',
          `cheesKernel: the position has ${position.length} values, need ${c} chains of ${dim}`,
        )
      }
      const x = Float64Array.from(position)
      const lds = new Float64Array(c)
      const grads = new Float64Array(c * dim)
      for (let k = 0; k < c; k++) {
        const r = target.valueAndGrad(x.subarray(k * dim, (k + 1) * dim))
        lds[k] = r.value
        grads.set(r.grad, k * dim)
      }
      return {
        position: x,
        logDensity: lds.reduce((s, v) => s + v, 0),
        walkerLogDensity: lds,
        grads,
        t: 0,
        proposals: x.slice(),
        momenta: new Float64Array(c * dim),
        initials: x.slice(),
        acceptProbs: new Float64Array(c).fill(1),
        divergent: new Uint8Array(c),
        numSteps: 0,
        trajectoryLength: length,
      }
    },
    step: (key, state) => {
      const h = jitter ? halton(state.t) : 1
      const steps = Math.max(1, Math.ceil((h * length) / eps))
      const x = Float64Array.from(state.position)
      const lds = Float64Array.from(state.walkerLogDensity)
      const grads = Float64Array.from(state.grads)
      const proposals = new Float64Array(c * dim)
      const momenta = new Float64Array(c * dim)
      const accept = new Float64Array(c)
      const divergent = new Uint8Array(c)
      let accepted = 0
      for (let k = 0; k < c; k++) {
        const kk = foldIn(key, k)
        let q = x.slice(k * dim, (k + 1) * dim)
        let p = normal(foldIn(kk, 0), [dim]).data
        let g: Float64Array = grads.slice(k * dim, (k + 1) * dim)
        let value = lds[k] ?? Number.NEGATIVE_INFINITY
        const h0 = -value + kinetic(p)
        for (let s = 0; s < steps; s++) {
          const gg = g
          p = p.map((v, i) => v + 0.5 * eps * (gg[i] ?? 0))
          const pp = p
          q = q.map((v, i) => v + eps * (pp[i] ?? 0))
          const r = target.valueAndGrad(q)
          value = r.value
          g = r.grad
          const g2 = g
          p = p.map((v, i) => v + 0.5 * eps * (g2[i] ?? 0))
          if (!Number.isFinite(value)) break
        }
        const delta = h0 - (-value + kinetic(p))
        const dH = Number.isNaN(delta) ? Number.NEGATIVE_INFINITY : delta
        const alpha = dH >= 0 ? 1 : Math.exp(dH)
        accept[k] = alpha
        divergent[k] = -dH > threshold ? 1 : 0
        proposals.set(q, k * dim)
        momenta.set(p, k * dim)
        if ((uniform(foldIn(kk, 1), [1]).data[0] ?? 1) < alpha) {
          x.set(q, k * dim)
          lds[k] = value
          grads.set(g, k * dim)
          accepted += 1
        }
      }
      return {
        state: {
          position: x,
          logDensity: lds.reduce((s, v) => s + v, 0),
          walkerLogDensity: lds,
          grads,
          t: state.t + 1,
          proposals,
          momenta,
          initials: Float64Array.from(state.position),
          acceptProbs: accept,
          divergent,
          numSteps: steps,
          trajectoryLength: length,
        },
        info: {
          acceptanceRate: accept.reduce((s, v) => s + v, 0) / c,
          accepted: accepted > 0,
          divergent: divergent.some((d) => d === 1),
          numSteps: steps,
          stepSize: eps,
        },
      }
    },
  }
}

/** Options of {@link cheesAdaptationAlgorithm}. */
export interface CheesAdaptationOptions {
  readonly numChains: number
  /** The warm-up length. */
  readonly numWarmup: number
  /** The starting step size (default 0.1). */
  readonly initialStepSize?: number
  /** The harmonic-mean acceptance rate dual averaging aims for (default 0.651, the optimum for HMC). */
  readonly targetAcceptance?: number
  /** The learning rate of Adam on the log trajectory length (default 0.025). */
  readonly learningRate?: number
  /** How much to favour recent iterations in the averages of the log step size and log trajectory length (default 0.5). */
  readonly decayRate?: number
  /** The cap on integrator steps per trajectory (default 1000). */
  readonly maxLeapfrogSteps?: number
  readonly jitter?: boolean
}

/** The state of {@link cheesAdaptationAlgorithm}. */
export interface CheesWarmupState {
  readonly chain: CheesState
  readonly stepSize: number
  readonly logStepSizeAvg: number
  readonly trajectoryLength: number
  readonly logTrajectoryLengthAvg: number
  readonly da: DualAveragingState
  /** Adam's first and second moments of the log trajectory length, and its step count. */
  readonly m: number
  readonly v: number
  readonly adamStep: number
}

/** The clip of one update of the log trajectory length (TFP's and BlackJAX's). */
const LOG_UPDATE_CLIP = 0.35

/**
 * ChEES-HMC's warm-up as a stepped algorithm (BlackJAX `chees_adaptation`): each step is one jittered HMC transition of the whole
 * ensemble, then the step size follows dual averaging on the harmonic mean of the acceptance probabilities of the chains that did not
 * diverge, and the trajectory length follows Adam, ascending the gradient of the ChEES criterion with respect to its log:
 * sum_c a_c h T (|x'_c - m'|^2 - |x_c - m|^2) <x'_c - m', v'_c> / sum_c a_c over the chains, with a_c the acceptance probability,
 * h the jitter, x' the proposal, v' the velocity at the end of its trajectory and m' and m the (acceptance-weighted) mean of the
 * proposals and the mean of the starts. The updates are clipped to 0.35 in the log, and both parameters are read as exponentials of their
 * decaying-weight moving averages, with the trajectory length between one step and `maxLeapfrogSteps` steps. The mass matrix is the
 * identity (precondition the model, or use window adaptation for a diagonal one). The state serialises.
 *
 * @param target - The target (with a gradient).
 * @param positions - The starting positions of the chains, row-major `[chains, dim]`.
 * @param options - Chains, warm-up, starting step size, target acceptance, learning rate, decay, step cap and jitter.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for fewer than two chains (the criterion needs a population).
 * @concept chees-hmc
 * @reference M. D. Hoffman, A. Radul and P. Sountsov, An adaptive-MCMC scheme for setting trajectory lengths in Hamiltonian Monte
 *   Carlo (AISTATS 2021).
 * @example
 * ```ts
 * import { cheesAdaptationAlgorithm, cheesAdaptationResult, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const alg = cheesAdaptationAlgorithm(t, Float64Array.of(-1, -0.5, 0, 0.5, 1, 1.5), { numChains: 6, numWarmup: 40 })
 * cheesAdaptationResult(runAlgorithm(alg, { key: key(0), maxSteps: 40 }).state).stepSize > 0 // => true
 * ```
 */
export function cheesAdaptationAlgorithm(
  target: LogDensity,
  positions: Float64Array,
  options: CheesAdaptationOptions,
): Algorithm<CheesWarmupState, AdaptPath> {
  const c = options.numChains
  const dim = target.dim
  if (c < 2) throw new OmaiError('value', 'cheesAdaptationAlgorithm: the criterion needs at least two chains')
  const goal = options.targetAcceptance ?? 0.651
  const lr = options.learningRate ?? 0.025
  const decay = options.decayRate ?? 0.5
  const maxSteps = options.maxLeapfrogSteps ?? 1000
  const eps0 = options.initialStepSize ?? 0.1
  const jitter = options.jitter ?? true
  const kernelOf = (s: { stepSize: number; trajectoryLength: number }): MCMCKernel<CheesState> =>
    cheesKernel(target, { numChains: c, stepSize: s.stepSize, trajectoryLength: s.trajectoryLength, jitter })
  return {
    name: 'sampling/chees-adaptation',
    config: { numChains: c, numWarmup: options.numWarmup, targetAcceptance: goal, learningRate: lr },
    schema: channels({
      metrics: {
        stepSize: { description: 'step size in force' },
        trajectoryLength: { description: 'trajectory length in force' },
        acceptanceRate: { description: 'mean acceptance probability of the chains', direction: 'higher' },
        logDensity: { description: 'summed log density of the chains' },
      },
    }),
    init: () => ({
      chain: kernelOf({ stepSize: eps0, trajectoryLength: eps0 }).init(positions),
      stepSize: eps0,
      logStepSizeAvg: 0,
      trajectoryLength: eps0,
      logTrajectoryLengthAvg: 0,
      da: dualAveragingInit(eps0),
      m: 0,
      v: 0,
      adamStep: 0,
    }),
    step: (s, ctx) => {
      const { state } = kernelOf(s).step(ctx.key, s.chain)
      const t = ctx.t + 1
      ctx.log('adapt/stepSize', s.stepSize)
      ctx.log('adapt/accept', state.acceptProbs.reduce((a, v) => a + v, 0) / c)
      ctx.log('adapt/logDensity', state.logDensity)
      // Step size: dual averaging on the harmonic mean acceptance of the chains that did not diverge.
      let inv = 0
      let n = 0
      for (let k = 0; k < c; k++)
        if (state.divergent[k] === 0) {
          inv += 1 / Math.max(state.acceptProbs[k] ?? 0, 1e-300)
          n += 1
        }
      const harmonic = n === 0 || !Number.isFinite(inv) ? 0 : n / inv
      const da = dualAveragingUpdate(s.da, harmonic, { targetAcceptance: goal })
      const proposedEps = Math.exp(da.logStepSize)
      const newDa = Number.isFinite(proposedEps) ? da : s.da
      const newEps = Number.isFinite(proposedEps) ? proposedEps : s.stepSize
      const weight = t ** -decay
      const logEpsAvg = (1 - weight) * s.logStepSizeAvg + weight * Math.log(newEps)
      // Trajectory length: Adam ascent on the ChEES criterion's gradient with respect to the log length.
      const a = new Float64Array(c)
      let aSum = 0
      for (let k = 0; k < c; k++) {
        const ok = state.divergent[k] === 0 && Number.isFinite(state.acceptProbs[k] ?? Number.NaN)
        a[k] = ok ? (state.acceptProbs[k] ?? 0) : 0
        aSum += a[k] ?? 0
      }
      const proposalMean = new Float64Array(dim)
      const initialMean = new Float64Array(dim)
      for (let k = 0; k < c; k++)
        for (let d = 0; d < dim; d++) {
          proposalMean[d] =
            (proposalMean[d] ?? 0) + ((a[k] ?? 0) * (state.proposals[k * dim + d] ?? 0)) / (aSum + 1e-20)
          initialMean[d] = (initialMean[d] ?? 0) + (state.initials[k * dim + d] ?? 0) / c
        }
      const h = jitter ? halton(state.t - 1) : 1
      let gradient = 0
      for (let k = 0; k < c; k++) {
        if ((a[k] ?? 0) === 0) continue
        let pm = 0
        let im = 0
        let dot = 0
        for (let d = 0; d < dim; d++) {
          const dp = (state.proposals[k * dim + d] ?? 0) - (proposalMean[d] ?? 0)
          const di = (state.initials[k * dim + d] ?? 0) - (initialMean[d] ?? 0)
          pm += dp * dp
          im += di * di
          dot += dp * (state.momenta[k * dim + d] ?? 0)
        }
        gradient += (a[k] ?? 0) * h * s.trajectoryLength * (pm - im) * dot
      }
      gradient /= aSum + 1e-20
      const logT = Math.log(s.trajectoryLength)
      const adamStep = s.adamStep + 1
      const m = 0.9 * s.m + 0.1 * gradient
      const v = 0.999 * s.v + 0.001 * gradient * gradient
      const update = (lr * (m / (1 - 0.9 ** adamStep))) / (Math.sqrt(v / (1 - 0.999 ** adamStep)) + 1e-8)
      const clipped = Math.max(-LOG_UPDATE_CLIP, Math.min(LOG_UPDATE_CLIP, update))
      const logTNew = Number.isFinite(clipped) ? logT + clipped : logT
      const logTAvg = (1 - weight) * s.logTrajectoryLengthAvg + weight * logTNew
      const trajectoryLength = Math.min(maxSteps * newEps, Math.max(newEps, Math.exp(logTAvg)))
      ctx.log('adapt/window', 0)
      return {
        chain: state,
        stepSize: newEps,
        logStepSizeAvg: logEpsAvg,
        trajectoryLength,
        logTrajectoryLengthAvg: logTAvg,
        da: newDa,
        m: Number.isFinite(clipped) ? m : s.m,
        v: Number.isFinite(clipped) ? v : s.v,
        adamStep: Number.isFinite(clipped) ? adamStep : s.adamStep,
      }
    },
    observe: (s) => ({
      metrics: {
        stepSize: s.stepSize,
        trajectoryLength: s.trajectoryLength,
        acceptanceRate: s.chain.acceptProbs.reduce((a, v) => a + v, 0) / c,
        logDensity: s.chain.logDensity,
      },
    }),
    done: (_, t) => (t >= options.numWarmup ? 'completed' : undefined),
    serialise: (s) => ({
      chain: chainToJson(s.chain),
      stepSize: numberToJson(s.stepSize),
      logStepSizeAvg: numberToJson(s.logStepSizeAvg),
      trajectoryLength: numberToJson(s.trajectoryLength),
      logTrajectoryLengthAvg: numberToJson(s.logTrajectoryLengthAvg),
      da: stateToJson(s.da),
      m: numberToJson(s.m),
      v: numberToJson(s.v),
      adamStep: s.adamStep,
    }),
    deserialise: (j) => {
      const o = j as unknown as Record<string, Json>
      return {
        chain: chainFromJson<CheesState>(o['chain'] as Json),
        stepSize: numberFromJson(o['stepSize']),
        logStepSizeAvg: numberFromJson(o['logStepSizeAvg']),
        trajectoryLength: numberFromJson(o['trajectoryLength']),
        logTrajectoryLengthAvg: numberFromJson(o['logTrajectoryLengthAvg']),
        da: stateFromJson(o['da'] as Json) as DualAveragingState,
        m: numberFromJson(o['m']),
        v: numberFromJson(o['v']),
        adamStep: o['adamStep'] as number,
      }
    },
  }
}

/** The parameters {@link cheesKernel} takes, as {@link cheesAdaptationResult} finds them. */
export interface CheesParameters {
  readonly stepSize: number
  readonly trajectoryLength: number
}

/**
 * The step size and trajectory length a warm-up state ends on: the exponentials of the moving averages of their logs, the length
 * between one step and the cap.
 *
 * @param state - A state of {@link cheesAdaptationAlgorithm}.
 * @returns The step size and the trajectory length.
 * @concept chees-hmc
 * @example
 * ```ts
 * import { cheesAdaptationAlgorithm, cheesAdaptationResult, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const run = runAlgorithm(cheesAdaptationAlgorithm(t, Float64Array.of(-1, 0, 1, 2), { numChains: 4, numWarmup: 30 }), { key: key(1), maxSteps: 30 })
 * cheesAdaptationResult(run.state).trajectoryLength >= cheesAdaptationResult(run.state).stepSize // => true
 * ```
 */
export function cheesAdaptationResult(state: CheesWarmupState): CheesParameters {
  const stepSize = Math.exp(state.logStepSizeAvg)
  const ok = Number.isFinite(stepSize) && stepSize > 0
  const eps = ok ? stepSize : state.stepSize
  const length = Math.exp(state.logTrajectoryLengthAvg)
  return { stepSize: eps, trajectoryLength: Math.max(eps, Number.isFinite(length) ? length : state.trajectoryLength) }
}
