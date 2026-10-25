/**
 * Hamiltonian Monte Carlo (HMC), which proposes distant moves by simulating Hamiltonian dynamics on the log density
 * with momentum and corrects the integration error with a Metropolis step: the HMC kernel, its window adaptation, and
 * `hmc`, which samples several chains after Stan-style warm-up as NumPyro's `MCMC(HMC(...))`.
 */

import { WhiteboxError } from '../../base/index.ts'
import { foldIn } from '../../random/index.ts'
import type { TensorLike } from '../../tensor/index.ts'
import { acceptWith, checkStepSize } from '../_mcmc.ts'
import {
  type Adaptation,
  type Chains,
  type MCMCKernel,
  type MCMCState,
  type SampleChainsOptions,
  sampleChains,
} from '../chains.ts'
import type { LogDensity } from '../target.ts'
import { type WindowAdaptationOptions, windowAdaptation } from './adaptation.ts'
import { euclideanMetric, hamiltonian, type IntegratorScheme, integrator } from './integrators.ts'

/** The state of the Hamiltonian kernels: the position with its log density and gradient. */
export interface HMCState extends MCMCState {
  readonly grad: Float64Array
}

/**
 * The starting state of a Hamiltonian kernel at a position.
 *
 * @param target - The target (with a gradient).
 * @param position - The position.
 * @returns The state.
 * @throws WhiteboxError with code `shape` for a position of the wrong length, and `value` for a target without a
 *   gradient.
 * @example
 * ```ts
 * import { hmcState, numericTarget } from 'whitebox-engine/sampling'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * Array.from(hmcState(t, [2]).grad) // => [-2]
 * ```
 */
export function hmcState(target: LogDensity, position: ArrayLike<number>): HMCState {
  if (!target.hasGradient) throw new WhiteboxError('value', 'hmcState: the target has no gradient')
  if (position.length !== target.dim) {
    throw new WhiteboxError('shape', `hmcState: the position has ${position.length} values, need ${target.dim}`)
  }
  const x = Float64Array.from(position)
  const r = target.valueAndGrad(x)
  return { position: x, logDensity: r.value, grad: r.grad }
}

/** Options shared by the Hamiltonian kernels. */
export interface HamiltonianOptions {
  /** The integrator step size (positive). */
  readonly stepSize: number
  /** M^-1: a number, the diagonal, a matrix or a flattened matrix (default the identity). */
  readonly inverseMassMatrix?: TensorLike | number
  /** The splitting scheme (default `velocityVerlet`, the leapfrog). */
  readonly integrator?: IntegratorScheme
  /** The energy error H - H0 above which a transition counts as divergent (default 1000, as Stan and BlackJAX). */
  readonly divergenceThreshold?: number
}

/** Options of {@link hmcKernel}. */
export interface HMCOptions extends HamiltonianOptions {
  /** Integrator steps per transition (a positive integer, at most 100000). */
  readonly numIntegrationSteps: number
}

/**
 * Hamiltonian Monte Carlo (Duane et al. 1987, Neal 2011; BlackJAX `hmc`): draw a momentum p ~ N(0, M), integrate
 * Hamilton's equations for a fixed number of steps with a symplectic integrator, and accept the end point with
 * probability min(1, exp(H0 - H1)). The integration stops early when the log density stops being finite, and a
 * transition whose energy error exceeds the threshold is flagged divergent.
 *
 * @param target - The target (with a gradient).
 * @param options - Step size, number of integrator steps, inverse mass matrix, integrator and divergence threshold.
 * @returns The kernel.
 * @throws WhiteboxError with code `value` for a target without a gradient, a non-positive step size or an invalid
 *   number of steps.
 * @concept hamiltonian-monte-carlo
 * @reference R. M. Neal, MCMC using Hamiltonian dynamics (Handbook of Markov Chain Monte Carlo, 2011).
 * @example
 * ```ts
 * import { hmcKernel, numericTarget } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const k = hmcKernel(t, { stepSize: 0.1, numIntegrationSteps: 10 })
 * k.step(key(0), k.init([0.5])).info.numSteps // => 10
 * ```
 */
export function hmcKernel(target: LogDensity, options: HMCOptions): MCMCKernel<HMCState> {
  const { stepSize, numIntegrationSteps: steps } = options
  checkStepSize(stepSize, 'hmcKernel')
  if (!(Number.isInteger(steps) && steps >= 1 && steps <= 100000)) {
    throw new WhiteboxError('value', 'hmcKernel: numIntegrationSteps must be an integer in [1, 100000]')
  }
  const metric = euclideanMetric(options.inverseMassMatrix ?? 1, target.dim)
  const step = integrator(target, metric, options.integrator ?? 'velocityVerlet')
  const threshold = options.divergenceThreshold ?? 1000
  return {
    name: 'hmc',
    dim: target.dim,
    config: { dim: target.dim, stepSize, numIntegrationSteps: steps, divergenceThreshold: threshold },
    init: (position) => hmcState(target, position),
    step: (key, state) => {
      const momentum = metric.sampleMomentum(foldIn(key, 0))
      let s = { position: state.position, momentum, logDensity: state.logDensity, grad: state.grad }
      const h0 = hamiltonian(s, metric)
      let taken = 0
      while (taken < steps) {
        s = step(s, stepSize)
        taken += 1
        if (!Number.isFinite(s.logDensity)) break
      }
      const h1 = hamiltonian(s, metric)
      const raw = h0 - h1
      const delta = Number.isNaN(raw) ? Number.NEGATIVE_INFINITY : raw
      const alpha = delta >= 0 ? 1 : Math.exp(delta)
      const accepted = acceptWith(foldIn(key, 1), alpha)
      return {
        state: accepted ? { position: s.position, logDensity: s.logDensity, grad: s.grad } : state,
        info: {
          acceptanceRate: alpha,
          accepted,
          divergent: -delta > threshold,
          numSteps: taken,
          energy: accepted ? h1 : h0,
          stepSize,
        },
      }
    },
  }
}

/** Options of {@link hmcAdaptation}: HMC's fixed settings and the window adaptation's. */
export type HMCAdaptationOptions = Omit<HMCOptions, 'stepSize' | 'inverseMassMatrix'> & WindowAdaptationOptions

/**
 * Window adaptation of HMC's step size and mass matrix (Stan's warm-up, BlackJAX `window_adaptation(hmc, ...)`).
 *
 * @param target - The target (with a gradient).
 * @param options - Number of integrator steps, integrator, and the adaptation options.
 * @returns The adaptation.
 * @concept window-adaptation
 * @example
 * ```ts
 * import { hmcAdaptation, numericTarget } from 'whitebox-engine/sampling'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * hmcAdaptation(t, { numIntegrationSteps: 10 }).name // => 'windowAdaptation'
 * ```
 */
export function hmcAdaptation(target: LogDensity, options: HMCAdaptationOptions): Adaptation<HMCState> {
  return windowAdaptation(target.dim, (p) => hmcKernel(target, { ...options, ...p }), options)
}

/** Options of {@link hmc}. */
export type HMCRunOptions = HMCAdaptationOptions & Omit<SampleChainsOptions, 'names'>

/**
 * Sample a target with HMC after Stan-style window adaptation, over several chains (NumPyro's `MCMC(HMC(...))`):
 * 1000 warm-up iterations and 1000 draws per chain, 4 chains, unless given.
 *
 * @param target - The target (with a gradient).
 * @param options - Number of integrator steps, adaptation options, and chain options (key, chains, draws, warm-up).
 * @returns The chains, on the target's unconstrained coordinates (see `constrainChains`).
 * @concept hamiltonian-monte-carlo
 * @example
 * ```ts
 * import { hmc, numericTarget } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * hmc(t, { key: key(0), numIntegrationSteps: 5, numChains: 1, numWarmup: 50, numDraws: 20 }).draws.shape // => [1, 20, 1]
 * ```
 */
export function hmc(target: LogDensity, options: HMCRunOptions): Chains {
  return sampleChains(hmcAdaptation(target, options), {
    numWarmup: 1000,
    ...options,
    names: target.names,
  })
}
