/**
 * Pseudo-marginal Metropolis-Hastings (Andrieu and Roberts 2009), the kernel under particle marginal Metropolis-Hastings:
 * a random-walk Metropolis chain on parameters whose likelihood is only available through an unbiased, noisy estimate (a
 * particle filter, an importance sampler, a simulator). The estimate at the current point is kept with the state and never
 * redrawn, which is what makes the chain's marginal exactly the posterior whatever the estimator's noise.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { dgemv } from '../numerics/linalg/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { lowerCholesky, squareMatrix } from './_linear.ts'
import { acceptWith } from './_mcmc.ts'
import type { MCMCKernel, MCMCState } from './chains.ts'
import { metropolisAcceptance } from './metropolis.ts'
import type { ChainProposal } from './target.ts'

/** The state of {@link pseudoMarginalMetropolis}: the parameters, the estimate of their log likelihood and the log posterior it implies. */
export interface PseudoMarginalState extends MCMCState {
  /** The log of the (noisy) likelihood estimate carried with the current parameters. */
  readonly logLikelihood: number
}

/** Options of {@link pseudoMarginalMetropolis}. */
export interface PseudoMarginalOptions {
  /** The key of the estimate at the starting point (`init` has none of its own). */
  readonly key: Key
  /** The proposal standard deviation, one for all coordinates (default 2.38 / sqrt(dim)). */
  readonly scale?: number
  /** A full proposal covariance `[dim, dim]` (overrides `scale`). */
  readonly covariance?: TensorLike
}

/**
 * Pseudo-marginal random-walk Metropolis: propose theta' = theta + L z, draw a fresh log likelihood estimate at theta'
 * from `estimate`, and accept with min(1, p(theta') Lhat(theta') / (p(theta) Lhat(theta))), keeping the estimate of
 * whichever point is kept. The estimator must be unbiased for the likelihood on the natural scale (the exponential of a
 * particle filter's log evidence is); then the chain targets p(theta | data) exactly, and its mixing worsens as the
 * log-estimate's variance grows (aim for a standard deviation near 1 to 1.5 at a typical posterior draw). A parameter
 * outside the prior's support is rejected without an estimate.
 *
 * @param prior - The prior: its density, and `sample` is not used.
 * @param estimate - `estimate(theta, key)` is the log of an unbiased likelihood estimate at `theta` (NaN reads as -Infinity).
 * @param options - The key of the first estimate, and the proposal scale or covariance.
 * @returns The kernel; `init(theta)` estimates the likelihood at the start, the state carries the estimate.
 * @throws OmaiError with code `value` for a non-positive scale or a covariance that is not positive definite.
 * @concept pseudo-marginal-mcmc
 * @reference C. Andrieu and G. O. Roberts, The pseudo-marginal approach for efficient Monte Carlo computations (Annals of Statistics 2009).
 * @index function Pseudo-marginal random-walk Metropolis on a noisy unbiased likelihood estimate.
 * @example
 * ```ts
 * import { proposalFromDistribution, pseudoMarginalMetropolis } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * // An exact likelihood is a (noise-free) unbiased estimate: N(y = 1; theta, 1).
 * const kernel = pseudoMarginalMetropolis(proposalFromDistribution(normal(0, 2)), (x) => -0.5 * (1 - (x[0] ?? 0)) ** 2, { key: key(0), scale: 1 })
 * const s = kernel.init([0.5])
 * [s.position[0], Number.isFinite(s.logDensity)] // => [0.5, true]
 * ```
 */
export function pseudoMarginalMetropolis(
  prior: Pick<ChainProposal, 'dim' | 'logProb'>,
  estimate: (theta: Float64Array, key: Key) => number,
  options: PseudoMarginalOptions,
): MCMCKernel<PseudoMarginalState> {
  const dim = prior.dim
  const factor =
    options.covariance === undefined
      ? undefined
      : lowerCholesky(
          squareMatrix(options.covariance, dim, 'pseudoMarginalMetropolis'),
          dim,
          'pseudoMarginalMetropolis',
        )
  const scale = options.scale ?? 2.38 / Math.sqrt(dim)
  if (!(scale > 0 && Number.isFinite(scale)))
    throw new OmaiError('value', 'pseudoMarginalMetropolis: the scale must be positive and finite')
  const logEstimate = (theta: Float64Array, key: Key): number => {
    const v = estimate(theta, key)
    return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
  }
  const config: Json = {
    dim,
    ...(factor === undefined ? { scale } : { covarianceFactor: Array.from(factor) }),
  }
  return {
    name: 'pseudoMarginalMetropolis',
    dim,
    config,
    init: (position) => {
      if (position.length !== dim)
        throw new OmaiError('shape', `pseudoMarginalMetropolis: the position needs ${dim} values`)
      const theta = Float64Array.from(position)
      const logLikelihood = logEstimate(theta, foldIn(options.key, 0))
      return { position: theta, logLikelihood, logDensity: prior.logProb(theta) + logLikelihood }
    },
    step: (key, state) => {
      const z = normal(foldIn(key, 0), [dim]).data
      const dz = factor === undefined ? z.map((v) => v * scale) : dgemv(factor, dim, dim, z)
      const proposal = state.position.map((x, i) => x + (dz[i] ?? 0))
      const lp = prior.logProb(proposal)
      // Outside the prior's support there is nothing to estimate.
      const ll = lp === Number.NEGATIVE_INFINITY ? Number.NEGATIVE_INFINITY : logEstimate(proposal, foldIn(key, 1))
      const alpha = metropolisAcceptance(state.logDensity, lp + ll, 0, 0)
      const accepted = acceptWith(foldIn(key, 2), alpha)
      return {
        state: accepted ? { position: proposal, logLikelihood: ll, logDensity: lp + ll } : state,
        info: { acceptanceRate: alpha, accepted },
      }
    },
  }
}
