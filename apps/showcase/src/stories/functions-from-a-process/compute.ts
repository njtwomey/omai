/**
 * The computation of 'Functions drawn from a Gaussian process': random functions are drawn from a squared-exponential process
 * (the prior), five noisy points of one more such function are observed, and the process is conditioned on them exactly (the
 * posterior), which gives its mean, a band, and new random functions that now pass near the points. It is a pure function of
 * the seed that returns plain data and runs in the story's worker. The regions are the code the page shows.
 */

// region imports
import { gpRegression } from 'omai-engine/gp'
import { rbfKernel } from 'omai-engine/kernels'
import { foldIn, key, normal, uniform } from 'omai-engine/random'
import { sampleGaussianProcess } from 'omai-engine/stochastic'
import { add, arange, linspace, mul, reshape, sqrt, sub, toArray, toFlat, type Value } from 'omai-engine/tensor'
// endregion

import type { Say } from '../../kit/protocol.ts'

/** How wiggly the functions are: a function changes appreciably over about this much of x. */
export const LENGTH_SCALE = 0.55
/** The standard deviation of the noise on each observed point. */
export const NOISE = 0.1
/** The range of x the figures cover, and how many observed points there are. */
export const X_RANGE = [-3, 3] as const
export const OBSERVED = 5
/** The multiplier of a standard deviation that holds 90% of a normal distribution. */
export const Z90 = 1.645

/** A band and some functions: the random functions drawn from a process, with a 90% band around its mean. */
export interface Crowd {
  /** The random functions, one row per function, at each point of the grid. */
  readonly draws: readonly (readonly number[])[]
  /** The process's mean at each grid point, and the edges of the band holding 90% of the functions there. */
  readonly mean: readonly number[]
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  /** The standard deviation of the process at each grid point. */
  readonly sd: readonly number[]
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The x values everything is evaluated at. */
  readonly grid: readonly number[]
  /** The five observed points. */
  readonly data: { readonly x: readonly number[]; readonly y: readonly number[] }
  /** The process before the points: a mean of zero, and the functions it produces. */
  readonly prior: Crowd
  /** The same process after conditioning on the points. */
  readonly posterior: Crowd
}

/** How much to compute: the run uses {@link DEFAULT_SIZE}, a test a small one. */
export interface Size {
  /** How many random functions to draw, from the prior and from the posterior. */
  readonly functions: number
  /** How many points the functions are evaluated at. */
  readonly grid: number
}

/** The size of the real run. */
export const DEFAULT_SIZE: Size = { functions: 8, grid: 121 }

/** The functions, the mean and a 90% band, read off a set of draws and a standard deviation, as plain numbers. */
const crowdOf = (draws: Value, mean: Value, sd: Value): Crowd => ({
  draws: toArray(draws) as number[][],
  mean: toFlat(mean),
  lower: toFlat(sub(mean, mul(sd, Z90))),
  upper: toFlat(add(mean, mul(sd, Z90))),
  sd: toFlat(sd),
})

/**
 * The computation, from a seed: draw functions from the prior, observe five points of another one, condition on them and draw
 * again.
 *
 * @param seed - Decides the functions, where the five points are, and what they read.
 * @param say - Says what is happening, for the play button.
 * @param size - How much to compute.
 * @returns The result.
 */
export function draw(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  const seeds = key(seed)
  const width = X_RANGE[1] - X_RANGE[0]
  say('Drawing functions from the prior')
  // region prior
  const kernel = rbfKernel({ lengthScale: LENGTH_SCALE })
  const grid = linspace(X_RANGE[0], X_RANGE[1], size.grid)
  const wiggles = sampleGaussianProcess(foldIn(seeds, 0), kernel.k, grid, { paths: size.functions })
  // endregion
  say('Making five points')
  // region data
  // Five x values, one in each fifth of the range, each nudged a little.
  const slots = add(mul(arange(OBSERVED), width / OBSERVED), X_RANGE[0] + (0.15 * width) / OBSERVED)
  const nudges = mul(uniform(foldIn(seeds, 1), [OBSERVED]), (0.7 * width) / OBSERVED)
  const xs = add(slots, nudges)
  // One more function from the same process, read at those x values with a little noise.
  const hidden = reshape(sampleGaussianProcess(foldIn(seeds, 2), kernel.k, xs), [OBSERVED])
  const ys = add(hidden, mul(normal(foldIn(seeds, 3), [OBSERVED]), NOISE))
  // endregion
  say('Conditioning on the points')
  // region posterior
  const fitted = gpRegression
    .with({ kernel, alpha: NOISE ** 2, optimise: false })
    .fit({ x: reshape(xs, [OBSERVED, 1]), y: ys })
  const rows = reshape(grid, [size.grid, 1])
  const { mean, std } = fitted.predictWithStd(rows)
  const conditioned = fitted.sample(rows, size.functions, foldIn(seeds, 4))
  // endregion
  const priorSd = sqrt(kernel.diag(rows))
  return {
    grid: toFlat(grid),
    data: { x: toFlat(xs), y: toFlat(ys) },
    prior: crowdOf(wiggles, mul(grid, 0), priorSd),
    posterior: crowdOf(conditioned, mean, std),
  }
}

/**
 * The run of the story: the computation at its full size.
 *
 * @param seed - The seed.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => draw(seed, say, DEFAULT_SIZE)
