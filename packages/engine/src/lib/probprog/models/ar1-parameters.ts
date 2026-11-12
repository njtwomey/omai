/**
 * The AR(1) state-space benchmark with an unknown autoregression coefficient, as a family for particle MCMC: the data are those
 * of `ar1StateSpace` (a stationary AR(1) state observed with noise, the state noise and observation noise known), the
 * coefficient `phi` has a uniform prior on (-0.99, 0.99), and the truth is the posterior of `phi` computed from the Kalman filter's
 * exact marginal likelihood on a fine grid: the reference that particle marginal Metropolis-Hastings and particle Gibbs must
 * reproduce, since they approximate nothing in the limit.
 */

import { gaussianParticleModel, kalmanLogLikelihood, type StateSpaceFamily } from '../../dynamics/filtering/index.ts'
import { add, arange, div, item, mul, toFlat, type Value } from '../../tensor/index.ts'
import { ar1StateSpace } from './state-space.ts'

/** The state and observation noise of the benchmark (the same as `ar1StateSpace`'s). */
export const AR1_NOISE = { sigmaX: 0.5, sigmaY: 0.6 } as const

/** The benchmark's observations as the `[T, 1]` rows the filters take. */
export const ar1Observations: readonly (readonly number[])[] = (ar1StateSpace.data.data['y'] as number[]).map((v) => [
  v,
])

/**
 * The family of the AR(1) benchmark indexed by `phi`: the stationary start `N(0, sigmaX^2 / (1 - phi^2))`, the transition
 * `x_t = phi x_{t-1} + N(0, sigmaX^2)` and the observation `y_t = x_t + N(0, sigmaY^2)`, with the transition density and the
 * density of the first state that particle Gibbs needs, and the uniform prior on (-0.99, 0.99).
 *
 * @concept particle-mcmc
 * @index dataset The AR(1) state-space benchmark with an unknown coefficient, as a family for particle MCMC.
 * @example
 * ```ts
 * import { ar1CoefficientFamily } from 'omai-engine/probprog/models'
 * ar1CoefficientFamily.prior.logProb([0.5]) === -Math.log(1.98) // => true
 * ```
 */
export const ar1CoefficientFamily: StateSpaceFamily = {
  dim: 1,
  names: ['phi'],
  prior: {
    dim: 1,
    logProb: (t) => (Math.abs(t[0] ?? 2) < 0.99 ? -Math.log(1.98) : Number.NEGATIVE_INFINITY),
    // banned-ok: sentinel: the sample of the family is a typed array (its API type), a constant here
    sample: () => new Float64Array([0.3]),
  },
  model: (t) => {
    const phi = t[0] ?? 0
    return gaussianParticleModel({
      f: (x: Value) => mul(x, phi),
      h: (x: Value) => x,
      Q: [[AR1_NOISE.sigmaX ** 2]],
      R: [[AR1_NOISE.sigmaY ** 2]],
      x0: [0],
      P0: [[AR1_NOISE.sigmaX ** 2 / (1 - phi ** 2)]],
    })
  },
  logInitial: (t, x) => {
    const v = AR1_NOISE.sigmaX ** 2 / (1 - (t[0] ?? 0) ** 2)
    return -0.5 * ((x[0] ?? 0) ** 2 / v + Math.log(2 * Math.PI * v))
  },
}

/**
 * The exact posterior mean and standard deviation of `phi` given the benchmark's observations: the Kalman filter's marginal
 * likelihood on a grid of `points` values of `phi`, normalised under the uniform prior.
 *
 * @param points - Grid points (default 397).
 * @returns The mean and sd.
 * @concept particle-mcmc
 * @example
 * ```ts
 * import { ar1CoefficientPosterior } from 'omai-engine/probprog/models'
 * ar1CoefficientPosterior(99).mean > 0 // => true
 * ```
 */
export function ar1CoefficientPosterior(points = 397): { readonly mean: number; readonly sd: number } {
  const grid = toFlat(add(div(mul(add(arange(points), 0.5), 1.98), points), -0.99))
  const logL = grid.map((phi) =>
    item(
      kalmanLogLikelihood(
        {
          F: [[phi]],
          H: [[1]],
          Q: [[AR1_NOISE.sigmaX ** 2]],
          R: [[AR1_NOISE.sigmaY ** 2]],
          x0: [0],
          P0: [[AR1_NOISE.sigmaX ** 2 / (1 - phi ** 2)]],
        },
        ar1Observations,
      ),
    ),
  )
  const top = Math.max(...logL)
  const w = logL.map((l) => Math.exp(l - top))
  const z = w.reduce((a, b) => a + b, 0)
  const mean = grid.reduce((a, phi, i) => a + (phi * (w[i] ?? 0)) / z, 0)
  const sd = Math.sqrt(grid.reduce((a, phi, i) => a + ((phi - mean) ** 2 * (w[i] ?? 0)) / z, 0))
  return { mean, sd }
}
