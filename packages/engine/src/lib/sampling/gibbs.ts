import { WhiteboxError } from '../base/index.ts'
import { ddot, dgemv } from '../numerics/linalg/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import { asTensor, type Tensor, type TensorLike } from '../tensor/index.ts'
import { inverse, lowerCholesky, squareMatrix, vectorOf } from './_linear.ts'
import { uniformOne, uniforms } from './_mcmc.ts'
import type { MCMCKernel, MCMCState } from './chains.ts'
import type { LogDensity } from './target.ts'

/**
 * One block of a Gibbs sampler: the coordinates it updates and a draw of them from their full conditional given the
 * current point (or any kernel that leaves that conditional invariant, for Metropolis-within-Gibbs).
 */
export interface GibbsBlock {
  readonly indices: readonly number[]
  /** New values for `indices`, drawn with the key given the full current point. */
  sample(key: Key, x: Float64Array): ArrayLike<number>
}

/** Options of {@link blockedGibbs}. */
export interface GibbsOptions {
  /** Update every block in order (`systematic`, default) or one block chosen uniformly at random (`random`). */
  readonly scan?: 'systematic' | 'random'
  /** A target, to report log densities in traces (default none: NaN). */
  readonly target?: LogDensity
}

/**
 * Blocked Gibbs sampling: each transition replaces the coordinates of each block (or of one random block) with a
 * draw from their full conditional distribution. Every block update leaves the joint invariant, so their composition
 * does; random scan is also reversible. Blocks may overlap or be Metropolis steps within Gibbs.
 *
 * @param dim - The dimension.
 * @param blocks - The blocks and their conditional samplers.
 * @param options - Systematic or random scan, and a target for traced log densities.
 * @returns The kernel (acceptance rate 1).
 * @throws WhiteboxError with code `value` for no blocks or indices outside [0, dim), and `shape` when a block returns
 *   the wrong number of values.
 * @concept gibbs-sampling
 * @reference S. Geman and D. Geman, Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images
 *   (IEEE PAMI 1984).
 * @example
 * ```ts
 * import { blockedGibbs, sampleChain } from 'whitebox-engine/sampling'
 * import { key, normal } from 'whitebox-engine/random'
 * // A bivariate normal with correlation 0.5 from its two conditionals.
 * const rho = 0.5
 * const s = Math.sqrt(1 - rho * rho)
 * const k = blockedGibbs(2, [
 *   { indices: [0], sample: (key, x) => [rho * (x[1] ?? 0) + s * (normal(key, [1]).data[0] ?? 0)] },
 *   { indices: [1], sample: (key, x) => [rho * (x[0] ?? 0) + s * (normal(key, [1]).data[0] ?? 0)] },
 * ])
 * sampleChain(k, [0, 0], { key: key(0), numDraws: 10 }).draws.length // => 20
 * ```
 */
export function blockedGibbs(dim: number, blocks: readonly GibbsBlock[], options: GibbsOptions = {}): MCMCKernel {
  if (blocks.length === 0) throw new WhiteboxError('value', 'blockedGibbs: need at least one block')
  for (const b of blocks) {
    if (b.indices.length === 0 || !b.indices.every((i) => Number.isInteger(i) && i >= 0 && i < dim)) {
      throw new WhiteboxError('value', `blockedGibbs: block indices must lie in [0, ${dim})`)
    }
  }
  const scan = options.scan ?? 'systematic'
  const t = options.target
  const density = (x: Float64Array): number => (t === undefined ? Number.NaN : t.logDensity(x))
  const update = (key: Key, x: Float64Array, b: GibbsBlock): Float64Array => {
    const v = b.sample(key, x)
    if (v.length !== b.indices.length) {
      throw new WhiteboxError('shape', `blockedGibbs: a block of ${b.indices.length} returned ${v.length} values`)
    }
    const next = Float64Array.from(x)
    b.indices.forEach((i, j) => {
      next[i] = v[j] ?? 0
    })
    return next
  }
  return {
    name: 'blockedGibbs',
    dim,
    config: { dim, blocks: blocks.map((b) => [...b.indices]), scan },
    init: (position) => {
      if (position.length !== dim) throw new WhiteboxError('shape', `blockedGibbs: the position needs ${dim} values`)
      const x = Float64Array.from(position)
      const s: MCMCState = { position: x, logDensity: density(x) }
      return s
    },
    step: (key, state) => {
      let x = state.position
      if (scan === 'random') {
        const j = Math.min(blocks.length - 1, Math.floor(uniformOne(foldIn(key, 0)) * blocks.length))
        x = update(foldIn(key, 1), x, blocks[j] as GibbsBlock)
      } else
        blocks.forEach((b, j) => {
          x = update(foldIn(key, j + 1), x, b)
        })
      return { state: { position: x, logDensity: density(x) }, info: { acceptanceRate: 1, accepted: true } }
    },
  }
}

/** Options of {@link sliceSampler}. */
export interface SliceOptions {
  /** The initial bracket width w, one for all coordinates or one each (default 1). */
  readonly width?: number | TensorLike
  /** The most stepping-out expansions m per coordinate (default 50). */
  readonly maxStepsOut?: number
  /** The most shrinkage proposals per coordinate before giving up and keeping the value (default 200). */
  readonly maxShrink?: number
}

/**
 * Neal's slice sampler, one coordinate at a time: draw a level log y = log pi(x) - Exp(1) under the density, place a
 * bracket of width w randomly around x_i, step it out until both ends are below the level (at most m steps, split at
 * random between the sides), then sample uniformly in the bracket, shrinking it towards x_i after each rejection.
 * No tuning beyond w is needed, and the chain leaves the target invariant. The work per coordinate is capped, so a
 * degenerate target never stalls a run.
 *
 * @param target - The target (no gradient needed).
 * @param options - Bracket width, stepping-out cap and shrinkage cap.
 * @returns The kernel; `numSteps` reports the log-density evaluations of the transition.
 * @throws WhiteboxError with code `value` for non-positive widths or caps.
 * @concept slice-sampling
 * @reference R. M. Neal, Slice sampling (Annals of Statistics 2003), sections 4 and 5.
 * @example
 * ```ts
 * import { numericTarget, sliceSampler } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const k = sliceSampler(t, { width: 2 })
 * (k.step(key(0), k.init([0.3])).info.numSteps ?? 0) >= 2 // => true
 * ```
 */
export function sliceSampler(target: LogDensity, options: SliceOptions = {}): MCMCKernel {
  const dim = target.dim
  const width = vectorOf(options.width ?? 1, dim, 'sliceSampler')
  const m = options.maxStepsOut ?? 50
  const maxShrink = options.maxShrink ?? 200
  if (!width.every((w) => w > 0 && Number.isFinite(w)) || !(m >= 0) || !(maxShrink >= 1)) {
    throw new WhiteboxError('value', 'sliceSampler: widths must be positive and the caps non-negative')
  }
  return {
    name: 'slice',
    dim,
    config: { dim, width: Array.from(width), maxStepsOut: m, maxShrink },
    init: (position) => {
      if (position.length !== dim) throw new WhiteboxError('shape', `sliceSampler: the position needs ${dim} values`)
      const x = Float64Array.from(position)
      const s: MCMCState = { position: x, logDensity: target.logDensity(x) }
      return s
    },
    step: (key, state) => {
      const rand = uniforms(key)
      const x = Float64Array.from(state.position)
      let lp = state.logDensity
      let evaluations = 0
      const at = (i: number, v: number): number => {
        const old = x[i] ?? 0
        x[i] = v
        const r = target.logDensity(x)
        x[i] = old
        evaluations += 1
        return r
      }
      for (let i = 0; i < dim; i++) {
        const x0 = x[i] ?? 0
        const w = width[i] ?? 1
        // 1 - u lies in (0, 1], so the exponential draw is finite.
        const level = lp + Math.log(1 - rand())
        let left = x0 - w * rand()
        let right = left + w
        let j = Math.floor(m * rand())
        let k = m - 1 - j
        while (j > 0 && at(i, left) > level) {
          left -= w
          j -= 1
        }
        while (k > 0 && at(i, right) > level) {
          right += w
          k -= 1
        }
        let accepted = false
        for (let s = 0; s < maxShrink; s++) {
          const x1 = left + rand() * (right - left)
          const lp1 = at(i, x1)
          if (lp1 > level) {
            x[i] = x1
            lp = lp1
            accepted = true
            break
          }
          if (x1 < x0) left = x1
          else right = x1
        }
        if (!accepted) x[i] = x0
      }
      return {
        state: { position: x, logDensity: lp },
        info: { acceptanceRate: 1, accepted: true, numSteps: evaluations },
      }
    },
  }
}

/** The Gaussian prior of {@link ellipticalSlice}. */
export interface GaussianPrior {
  /** The prior mean (default zero). */
  readonly mean?: TensorLike
  /** The prior covariance `[dim, dim]`, or a scalar or vector for a diagonal one. */
  readonly covariance: TensorLike | number
}

/** Options of {@link ellipticalSlice}. */
export interface EllipticalSliceOptions {
  /** The most angle proposals per transition before keeping the current point (default 100). */
  readonly maxIter?: number
}

/**
 * Elliptical slice sampling (Murray, Adams and MacKay 2010; BlackJAX `elliptical_slice`) for a posterior
 * proportional to N(x; mu, Sigma) L(x): draw nu ~ N(0, Sigma) and a level under L, then propose points on the ellipse
 * mu + (x - mu) cos(theta) + nu sin(theta), shrinking the angle bracket towards 0 after each rejection. There are no
 * step sizes to tune, every transition moves, and it suits Gaussian-process and other strongly correlated Gaussian
 * priors. The number of proposals is capped.
 *
 * @param logLikelihood - log L(x).
 * @param prior - The Gaussian prior's mean and covariance.
 * @param dim - The dimension.
 * @param options - The cap on angle proposals.
 * @returns The kernel; the log density reported is log L + log N(x; mu, Sigma) up to a constant, and `numSteps` the
 *   number of likelihood evaluations.
 * @throws WhiteboxError with code `value` for a covariance that is not positive definite, and `shape` for a mean of
 *   the wrong length.
 * @concept elliptical-slice-sampling
 * @reference I. Murray, R. P. Adams and D. J. C. MacKay, Elliptical slice sampling (AISTATS 2010).
 * @example
 * ```ts
 * import { ellipticalSlice, sampleChain } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * // A N(0, 1) prior and a N(1; x, 1) likelihood: the posterior is N(0.5, 0.5).
 * const k = ellipticalSlice((x) => -0.5 * ((x[0] ?? 0) - 1) ** 2, { covariance: 1 }, 1)
 * sampleChain(k, [0], { key: key(0), numDraws: 5 }).draws.length // => 5
 * ```
 */
export function ellipticalSlice(
  logLikelihood: (x: Float64Array) => number,
  prior: GaussianPrior,
  dim: number,
  options: EllipticalSliceOptions = {},
): MCMCKernel {
  const mu = prior.mean === undefined ? new Float64Array(dim) : vectorOf(prior.mean, dim, 'ellipticalSlice')
  const c = prior.covariance
  let cov: Float64Array
  if (typeof c !== 'number' && (asTensor(c) as Tensor).shape.length === 2) cov = squareMatrix(c, dim, 'ellipticalSlice')
  else {
    const d = vectorOf(c, dim, 'ellipticalSlice')
    cov = Float64Array.from({ length: dim * dim }, (_, k) => (k % (dim + 1) === 0 ? (d[Math.floor(k / dim)] ?? 0) : 0))
  }
  const factor = lowerCholesky(cov, dim, 'ellipticalSlice')
  const precision = inverse(cov, dim)
  const maxIter = options.maxIter ?? 100
  const ll = (x: Float64Array): number => {
    const v = logLikelihood(x)
    return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
  }
  const logPrior = (x: Float64Array): number => {
    const d = x.map((v, i) => v - (mu[i] ?? 0))
    return -0.5 * ddot(d, dgemv(precision, dim, dim, d))
  }
  return {
    name: 'ellipticalSlice',
    dim,
    config: { dim, maxIter },
    init: (position) => {
      if (position.length !== dim) throw new WhiteboxError('shape', `ellipticalSlice: the position needs ${dim} values`)
      const x = Float64Array.from(position)
      const s: MCMCState = { position: x, logDensity: ll(x) + logPrior(x) }
      return s
    },
    step: (key, state) => {
      const rand = uniforms(foldIn(key, 1))
      const nu = dgemv(factor, dim, dim, normal(foldIn(key, 0), [dim]).data)
      const x = state.position
      const llx = state.logDensity - logPrior(x)
      const level = llx + Math.log(1 - rand())
      let theta = 2 * Math.PI * rand()
      let lo = theta - 2 * Math.PI
      let hi = theta
      for (let i = 1; i <= maxIter; i++) {
        const c = Math.cos(theta)
        const s = Math.sin(theta)
        const proposal = x.map((v, k) => (mu[k] ?? 0) + (v - (mu[k] ?? 0)) * c + (nu[k] ?? 0) * s)
        const l = ll(proposal)
        if (l > level) {
          return {
            state: { position: proposal, logDensity: l + logPrior(proposal) },
            info: { acceptanceRate: 1, accepted: true, numSteps: i },
          }
        }
        if (theta < 0) lo = theta
        else hi = theta
        theta = lo + rand() * (hi - lo)
      }
      return { state, info: { acceptanceRate: 0, accepted: false, numSteps: maxIter } }
    },
  }
}
