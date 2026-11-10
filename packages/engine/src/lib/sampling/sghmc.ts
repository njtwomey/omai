/**
 * Stochastic gradient Hamiltonian Monte Carlo (Chen, Fox and Guestrin 2014, with the scaling of Ma, Chen and Fox 2015;
 * BlackJAX `sghmc`): Hamiltonian dynamics with friction, driven by a noisy (minibatch) gradient, so that the momentum
 * noise the gradient adds is compensated by the friction and no Metropolis correction is needed. It scales to large data
 * the way SGLD does and keeps the momentum's persistence, which helps on correlated targets.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, normal } from '../random/index.ts'
import { checkStepSize } from './_mcmc.ts'
import type { MCMCKernel, MCMCState } from './chains.ts'
import type { StochasticGradient } from './langevin.ts'
import type { LogDensity } from './target.ts'

/** The state of {@link sghmcKernel}: the position and the iteration (for the step-size schedule). */
export interface SGHMCState extends MCMCState {
  readonly t: number
}

/** Options of {@link sghmcKernel}. */
export interface SGHMCOptions {
  /** A constant step size or a schedule t -> eps_t (see `polynomialStepSize`). */
  readonly stepSize: number | ((t: number) => number)
  /** Leapfrog-like steps per transition, with a fresh momentum at its start (default 10). */
  readonly numSteps?: number
  /** The friction alpha (default 0.01): larger damps the momentum faster and adds more noise. */
  readonly friction?: number
  /** The estimate beta of the variance of the gradient noise, subtracted from the injected noise (default 0). */
  readonly noiseEstimate?: number
  /** The temperature (default 1). */
  readonly temperature?: number
  /** A target to evaluate log densities for traces (default none: the log density is reported as NaN). */
  readonly target?: LogDensity
}

/**
 * Stochastic gradient HMC: each transition draws a momentum p ~ N(0, I) and takes `numSteps` Euler steps
 * x <- x + eps p; p <- (1 - alpha eps) p + eps g_hat(x) + sqrt(eps T (2 alpha - eps T beta)) z, with g_hat a stochastic
 * estimate of the gradient of the log density (see `minibatchGradient`). The friction alpha makes the dynamics' stationary
 * law the target when the gradient is exact; with minibatches the injected noise is reduced by the estimate beta of the
 * gradient noise. Like SGLD it is biased by the step size, which a decaying schedule removes.
 *
 * @param gradient - The stochastic gradient estimator.
 * @param dim - The dimension.
 * @param options - Step size or schedule, steps per transition, friction, noise estimate, temperature and a target.
 * @returns The kernel (acceptance rate always 1).
 * @throws OmaiError with code `value` for non-positive step sizes, steps, temperature or friction, or a noise estimate
 *   that makes the injected variance negative.
 * @concept stochastic-gradient-mcmc
 * @reference T. Chen, E. B. Fox and C. Guestrin, Stochastic gradient Hamiltonian Monte Carlo (ICML 2014).
 * @index function Stochastic-gradient Hamiltonian Monte Carlo kernel with friction, for minibatch gradients.
 * @example
 * ```ts
 * import { sghmcKernel } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const k = sghmcKernel((_, x) => [-(x[0] ?? 0)], 1, { stepSize: 0.05, numSteps: 5 })
 * k.step(key(0), k.init([1])).state.t // => 1
 * ```
 */
export function sghmcKernel(gradient: StochasticGradient, dim: number, options: SGHMCOptions): MCMCKernel<SGHMCState> {
  const schedule =
    typeof options.stepSize === 'number' ? ((() => options.stepSize) as (t: number) => number) : options.stepSize
  if (typeof options.stepSize === 'number') checkStepSize(options.stepSize, 'sghmcKernel')
  const steps = options.numSteps ?? 10
  const alpha = options.friction ?? 0.01
  const beta = options.noiseEstimate ?? 0
  const temperature = options.temperature ?? 1
  if (!(Number.isInteger(steps) && steps >= 1))
    throw new OmaiError('value', 'sghmcKernel: numSteps must be a positive integer')
  if (!(alpha > 0) || !(temperature > 0))
    throw new OmaiError('value', 'sghmcKernel: friction and temperature must be positive')
  const t = options.target
  const density = (x: Float64Array): number => (t === undefined ? Number.NaN : t.logDensity(x))
  return {
    name: 'sghmc',
    dim,
    config: {
      dim,
      numSteps: steps,
      friction: alpha,
      noiseEstimate: beta,
      temperature,
      stepSize: typeof options.stepSize === 'number' ? options.stepSize : 'schedule',
    },
    init: (position) => {
      if (position.length !== dim) throw new OmaiError('shape', `sghmcKernel: the position needs ${dim} values`)
      const x = Float64Array.from(position)
      return { position: x, logDensity: density(x), t: 0 }
    },
    step: (key, state) => {
      const eps = schedule(state.t)
      const variance = eps * temperature * (2 * alpha - eps * temperature * beta)
      if (!(variance >= 0)) {
        throw new OmaiError(
          'value',
          'sghmcKernel: the noise estimate is too large for the step size (negative noise variance)',
        )
      }
      const noise = Math.sqrt(variance)
      let x = Float64Array.from(state.position)
      let p = normal(foldIn(key, 0), [dim]).data
      for (let i = 0; i < steps; i++) {
        const g = gradient(foldIn(foldIn(key, 1), i), x)
        const z = normal(foldIn(foldIn(key, 2), i), [dim]).data
        const xx = x
        const pp = p
        x = xx.map((v, j) => v + eps * (pp[j] ?? 0))
        p = pp.map((v, j) => (1 - alpha * eps) * v + eps * (g[j] ?? 0) + noise * (z[j] ?? 0))
      }
      return {
        state: { position: x, logDensity: density(x), t: state.t + 1 },
        info: { acceptanceRate: 1, accepted: true, stepSize: eps, numSteps: steps },
      }
    },
  }
}
