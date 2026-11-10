/**
 * Barker's proposal (Livingstone and Zanella 2022; BlackJAX `barker`): a gradient-informed Metropolis-Hastings kernel
 * that is robust to the step size. Each coordinate of a symmetric Gaussian increment keeps its sign with a probability
 * that grows when the move goes uphill, so the proposal skews towards higher density without ever overshooting the way
 * a Langevin drift does, and the acceptance step is exact.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, normal, uniformStream } from '../random/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { vectorOf } from './_linear.ts'
import { acceptWith, checkStepSize } from './_mcmc.ts'
import type { MCMCKernel } from './chains.ts'
import type { LangevinState } from './langevin.ts'
import type { LogDensity } from './target.ts'

/** Options of {@link barkerKernel}. */
export interface BarkerOptions {
  /** The global scale sigma of the Gaussian increment (positive). */
  readonly stepSize: number
  /**
   * The per-coordinate scale of the increment, as the square root of a diagonal inverse mass matrix (BlackJAX's
   * preconditioner): a number or a vector (default 1).
   */
  readonly scale?: TensorLike | number
}

/** log(1 + exp(a)) without overflow. */
function log1pexp(a: number): number {
  return a > 0 ? a + Math.log1p(Math.exp(-a)) : Math.log1p(Math.exp(a))
}

/**
 * Barker's proposal kernel with a Gaussian base: draw z_i ~ N(0, (sigma s_i)^2) and move to x_i + z_i with probability
 * 1 / (1 + exp(-z_i s_i g_i(x))) and to x_i - z_i otherwise (g the gradient of the log density), so the proposal leans
 * the way the gradient points. The proposal is not symmetric, and the Metropolis-Hastings ratio
 * pi(y) q(y, x) / (pi(x) q(x, y)) is computed in closed form from the gradients at both ends, so the target is exactly
 * invariant. Unlike a Langevin drift, the move per coordinate is bounded by the Gaussian increment, so the kernel is
 * stable for a much wider range of step sizes.
 *
 * @param target - The target (with a gradient).
 * @param options - The global step size and the per-coordinate scale.
 * @returns The kernel; its state carries the gradient.
 * @throws OmaiError with code `value` for a target without a gradient, a non-positive step size or non-positive scales.
 * @concept barker-proposal
 * @reference S. Livingstone and G. Zanella, The Barker proposal: combining robustness and efficiency in
 *   gradient-based MCMC (JRSS B 2022).
 * @index function Barker's gradient-skewed Metropolis-Hastings kernel, robust to the step size.
 * @example
 * ```ts
 * import { barkerKernel, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const k = barkerKernel(t, { stepSize: 1 })
 * k.step(key(0), k.init([0.2])).info.acceptanceRate > 0.3 // => true
 * ```
 */
export function barkerKernel(target: LogDensity, options: BarkerOptions): MCMCKernel<LangevinState> {
  const sigma = options.stepSize
  checkStepSize(sigma, 'barkerKernel')
  if (!target.hasGradient) throw new OmaiError('value', 'barkerKernel: the target has no gradient')
  const dim = target.dim
  const s = vectorOf(options.scale ?? 1, dim, 'barkerKernel')
  if (!s.every((v) => v > 0 && Number.isFinite(v))) {
    throw new OmaiError('value', 'barkerKernel: the scales must be positive and finite')
  }
  const stateAt = (x: Float64Array): LangevinState => {
    const r = target.valueAndGrad(x)
    return { position: x, logDensity: r.value, grad: r.grad }
  }
  return {
    name: 'barker',
    dim,
    config: { dim, stepSize: sigma, scale: Array.from(s) },
    init: (position) => {
      if (position.length !== dim) {
        throw new OmaiError('shape', `barkerKernel: the position has ${position.length} values, need ${dim}`)
      }
      return stateAt(Float64Array.from(position))
    },
    step: (key, state) => {
      const z = normal(foldIn(key, 0), [dim]).data
      const u = uniformStream(foldIn(key, 1), 64)
      const proposal = new Float64Array(dim)
      for (let i = 0; i < dim; i++) {
        const zi = sigma * (z[i] ?? 0)
        const c = (s[i] ?? 1) * (state.grad[i] ?? 0)
        // Keep the increment's sign with probability 1 / (1 + exp(-z c)).
        const keep = u() < 1 / (1 + Math.exp(-zi * c))
        proposal[i] = (state.position[i] ?? 0) + (s[i] ?? 1) * (keep ? zi : -zi)
      }
      const next = stateAt(proposal)
      let alpha = 0
      if (Number.isFinite(next.logDensity)) {
        let logRatio = next.logDensity - state.logDensity
        for (let i = 0; i < dim; i++) {
          // The whitened increment from x to y and the gradients at both ends give q(y, x) / q(x, y).
          const zxy = ((proposal[i] ?? 0) - (state.position[i] ?? 0)) / (s[i] ?? 1)
          const cx = (s[i] ?? 1) * (state.grad[i] ?? 0)
          const cy = (s[i] ?? 1) * (next.grad[i] ?? 0)
          logRatio += log1pexp(-zxy * cx) - log1pexp(zxy * cy)
        }
        alpha = Number.isNaN(logRatio) ? 0 : logRatio >= 0 ? 1 : Math.exp(logRatio)
      }
      const accepted = acceptWith(foldIn(key, 2), alpha)
      return { state: accepted ? next : state, info: { acceptanceRate: alpha, accepted, stepSize: sigma } }
    },
  }
}
