/**
 * The computation of 'Is this coin fair?': a coin with a secret bias is tossed a hundred times from a seeded key, and after
 * 0, 1, 3, 10, 30 and 100 of the flips the library's conjugate engine turns the Beta-Bernoulli coin model and the flips so
 * far into the exact posterior of the bias, which is summarised as a curve, a 90% interval and the probability that the bias
 * is within a hair of fair. A pure function of the seed that returns plain data; it runs in the story's worker.
 */

// region imports
import { infer } from 'omai-engine/probprog'
import { toProgram } from 'omai-engine/probprog/model'
import { betaBernoulliModel } from 'omai-engine/probprog/models'
import { bernoulli, foldIn, key, uniform } from 'omai-engine/random'
import { add, item, linspace, mul, sliceAxis, sum, toFlat } from 'omai-engine/tensor'
// endregion

import type { Say } from '../../kit/protocol.ts'
import { STAGES } from './data.ts'

/** What is believed about the bias after some of the flips. */
export interface Stage {
  /** How many flips have been seen. */
  readonly flips: number
  /** How many of them were heads. */
  readonly heads: number
  /** The posterior mean of the bias. */
  readonly mean: number
  /** The 90% credible interval of the bias. */
  readonly lo: number
  readonly hi: number
  /** The posterior probability that the bias is within the tolerance of one half. */
  readonly fair: number
  /** The posterior density on the result's grid. */
  readonly density: readonly number[]
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The bias the coin really had, which no part of the inference was told. */
  readonly truth: number
  /** The flips, 1 for heads and 0 for tails, in the order they came. */
  readonly flips: readonly number[]
  /** The grid of biases the densities are on. */
  readonly grid: readonly number[]
  /** The belief after each of {@link STAGES}, in order. */
  readonly stages: readonly Stage[]
}

/**
 * The story, from a seed: make a coin and flip it, then read the posterior of its bias after each stage.
 *
 * @param seed - The seed of the coin and of its flips.
 * @param say - Says what is happening, for the play button.
 * @param stages - The numbers of flips to read the belief at (the story's own list unless a test asks for fewer).
 * @returns The result.
 */
export function flipCoin(seed: number, say: Say = () => {}, stages: readonly number[] = STAGES): Result {
  say('Making a coin and flipping it 100 times')
  // region flips
  const base = key(seed)
  // the coin's secret bias: somewhere between 0.35 and 0.8
  const bias = item(add(mul(uniform(foldIn(base, 0), []), 0.45), 0.35))
  // one hundred flips of it, 1 for heads and 0 for tails
  const flips = bernoulli(foldIn(base, 1), bias, [100])
  // endregion
  say('Setting up the model')
  // region model
  // a Beta(1, 1) prior is flat: before any flip, every bias is as likely as any other
  const coin = betaBernoulliModel({ a: 1, b: 1 })
  // endregion
  say('Reading the belief at each stage')
  // region posterior
  const grid = linspace(0, 1, 101)
  // for each stage: 0, 1, 3, 10, 30 and 100 flips
  const beliefs = stages.map((n) => {
    // only the first n flips are known
    const seen = sliceAxis(flips, 0, 0, n)
    const bound = coin.bind({ y: toFlat(seen) })
    // a Beta prior with Bernoulli flips is conjugate: the engine returns the exact Beta posterior
    const posterior = infer(toProgram(bound), undefined, { engine: 'conjugate' })
    const belief = posterior.marginal('p')
    const [lo, hi] = toFlat(belief.ppf([0.05, 0.95]))
    // the chance that the bias is within a hair of fair, between 0.45 and 0.55
    const [below, above] = toFlat(belief.cdf([0.45, 0.55]))
    return {
      flips: n,
      heads: item(sum(seen)),
      mean: item(belief.mean()),
      lo: lo as number,
      hi: hi as number,
      fair: (above as number) - (below as number),
      density: toFlat(belief.prob(grid)),
    }
  })
  // endregion
  return { truth: bias, flips: toFlat(flips), grid: toFlat(grid), stages: beliefs }
}

/**
 * The run of the story: the coin, the flips and the beliefs at the story's stages.
 *
 * @param seed - The seed.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => flipCoin(seed, say)
