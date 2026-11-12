/**
 * The computation of 'A network that knows what it does not know': a small neural network (one input, three tanh hidden units, one
 * output, ten weights in all) is conditioned on sixteen points by NUTS, and each draw of the weights, a whole network, is pushed
 * through a grid of inputs to give the curves the figure draws (a fan of networks, their mean and a 90% credible band). It is a pure
 * function of the seed that returns plain data, and it runs in the story's worker. The regions are the code the page shows.
 */

// region imports
import { mcmc } from 'omai-engine/probprog'
import { toProgram } from 'omai-engine/probprog/model'
import { bayesianNeuralNetwork } from 'omai-engine/probprog/models'
import { key } from 'omai-engine/random'
import { add, arange, cast, linspace, mean, mul, reshape, sliceAxis, sort, sum, take, tanh } from 'omai-engine/tensor'
// endregion

import { toArray, toFlat } from 'omai-engine/tensor'
import type { Say } from '../../kit/protocol.ts'
import { DRAW_RANGE, POINTS } from './data.ts'

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The inputs the networks are evaluated at. */
  readonly grid: readonly number[]
  /** The mean of the networks' outputs at each input. */
  readonly mean: readonly number[]
  /** The 90% credible band of the function (not of a new noisy point) at each input. */
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  /** Sampled networks, one row per draw, as their outputs on the grid. */
  readonly fan: readonly (readonly number[])[]
  /** How many networks (draws of the weights) were kept, and how many divergent transitions the sampler reported. */
  readonly draws: number
  readonly divergences: number
}

/** How much to sample: the run uses {@link DEFAULT_SIZE}, a test a small one. */
export interface Size {
  readonly chains: number
  readonly warmup: number
  readonly draws: number
  readonly lines: number
  readonly grid: number
}

/** The size of the real run: four chains (the weights have many equivalent modes), each warmed up for 100 and kept for 200. */
export const DEFAULT_SIZE: Size = { chains: 4, warmup: 100, draws: 200, lines: 30, grid: 141 }

/**
 * The inference, from a seed: bind the library's Bayesian neural network to the sixteen points, sample the posterior of its
 * weights, and push every draw through the network on a grid of inputs.
 *
 * @param seed - The sampler's seed (the data do not change with it).
 * @param say - Says what is happening, for the play button.
 * @param size - How much to sample.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  say('Binding a three-unit network to the sixteen points')
  // region infer
  // the model also records its output at test inputs (xs); we draw our own curves below, from the weights
  const network = bayesianNeuralNetwork.bound({ data: { ...POINTS, xs: [0] }, generating: {} })
  const posterior = mcmc(toProgram(network), undefined, {
    key: key(seed),
    numChains: size.chains,
    numWarmup: size.warmup,
    numDraws: size.draws,
    // short trajectories keep each draw cheap, and the function values mix well anyway
    maxTreeDepth: 5,
  })
  // endregion
  say('Turning the draws into networks')
  const w1 = posterior.draws?.['w1']
  const b1 = posterior.draws?.['b1']
  const w2 = posterior.draws?.['w2']
  const b2 = posterior.draws?.['b2']
  if (w1 === undefined || b1 === undefined || w2 === undefined || b2 === undefined)
    throw new Error(`the sampler returned no draws (${posterior.status})`)
  // region networks
  const count = w1.shape[0] ?? 0
  const grid = linspace(DRAW_RANGE[0], DRAW_RANGE[1], size.grid)
  // Axes: draw, input, hidden unit.
  const slope = reshape(w1, [count, 1, 3])
  const shift = reshape(b1, [count, 1, 3])
  const across = reshape(grid, [1, size.grid, 1])
  const hidden = tanh(add(mul(slope, across), shift))
  // The output of each network at each input: the hidden units weighted by w2, summed, plus the output bias.
  const weighted = mul(hidden, reshape(w2, [count, 1, 3]))
  const curves = add(sum(weighted, 2), reshape(b2, [count, 1]))
  // Sort each column: its 5% and 95% rows are the edges of a 90% band.
  const sorted = sort(curves, 0)
  const low = Math.round(0.05 * (count - 1))
  const high = Math.round(0.95 * (count - 1))
  const lower = sliceAxis(sorted, 0, low, low + 1)
  const upper = sliceAxis(sorted, 0, high, high + 1)
  const average = mean(curves, 0)
  // A few networks, spread evenly over the draws, make the fan.
  const kept = cast(mul(arange(size.lines), count / size.lines), 'int32')
  const fan = take(curves, kept, 0)
  // endregion
  return {
    grid: toFlat(grid),
    mean: toFlat(average),
    lower: toFlat(lower),
    upper: toFlat(upper),
    fan: toArray(fan) as number[][],
    draws: count,
    divergences: Number(posterior.diagnostics?.['divergences'] ?? 0),
  }
}

/**
 * The run of the story: the inference at its full size.
 *
 * @param seed - The sampler's seed.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => fit(seed, say, DEFAULT_SIZE)
