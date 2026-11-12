/**
 * The computation of 'Which button wins?': two versions of a page are shown to a thousand visitors each, and the library turns
 * the clicks into a posterior of each version's click rate (two Beta-Bernoulli rates, solved exactly by the conjugate engine),
 * draws from the two posteriors to ask how likely it is that B's rate is the higher, and asks that again after the first 20,
 * 40, 60 ... visitors of each version, to show how the answer grows with the evidence. A pure function of the seed that returns
 * plain data; it runs in the story's worker.
 */

// region imports
import type { Distribution } from 'omai-engine/dists'
import { infer, type Posterior } from 'omai-engine/probprog'
import { defineModel, toProgram } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { foldIn, key } from 'omai-engine/random'
import { histogram, quantile } from 'omai-engine/stats'
import { cast, greater, item, linspace, mean, sliceAxis, sub, sum, type Tensor, toFlat } from 'omai-engine/tensor'
// endregion

import type { Say } from '../../kit/protocol.ts'
import { CHECKPOINTS, simulate, type Tally, type Truth, tally, VISITORS } from './data.ts'

/** A mean and a 90% interval. */
export interface Interval {
  readonly mean: number
  readonly lo: number
  readonly hi: number
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The raw counts of the experiment. */
  readonly tally: Tally
  /** The click rates the visitors were really drawn with, which the inference was not told. */
  readonly truth: Truth
  /** The grid of click rates the two densities are on. */
  readonly grid: readonly number[]
  readonly densityA: readonly number[]
  readonly densityB: readonly number[]
  /** The posterior mean and 90% interval of each rate. */
  readonly rateA: Interval
  readonly rateB: Interval
  /** The posterior of the difference of the rates (B minus A): its summary, and a histogram of draws (bin centres, densities). */
  readonly difference: Interval & { readonly centres: readonly number[]; readonly density: readonly number[] }
  /** The probability that B's rate is higher than A's, with all the visitors. */
  readonly better: number
  /** How many pairs of draws that probability (and the histogram) was made from. */
  readonly draws: number
  /** The same probability after each of the first `visitors` visitors of each version. */
  readonly strip: { readonly visitors: readonly number[]; readonly better: readonly number[] }
}

/** How much to draw: the run uses {@link DEFAULT_SIZE}, a test a small one. */
export interface Size {
  /** Draws from the two posteriors for each probability that B is better. */
  readonly draws: number
  /** The numbers of visitors per version after which to look. */
  readonly checkpoints: readonly number[]
  readonly bins: number
}

/** The size of the real run. */
export const DEFAULT_SIZE: Size = { draws: 20_000, checkpoints: CHECKPOINTS, bins: 48 }

/**
 * The story, from a seed: simulate the visitors, write and solve the model, and read off what it believes.
 *
 * @param seed - The seed of the visitors, and of the draws.
 * @param say - Says what is happening, for the play button.
 * @param size - How much to draw.
 * @returns The result.
 */
export function compare(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  say('Showing the page to two thousand visitors')
  const experiment = simulate(seed)
  const { a, b } = experiment
  const base = key(seed)
  say('Writing the model: one click rate for each version')
  // region model
  const buttons = defineModel('buttons', (m) => {
    const visitorsA = m.data('visitorsA')
    const visitorsB = m.data('visitorsB')
    const clicksA = m.data('clicksA')
    const clicksB = m.data('clicksB')
    // each version has its own click rate; before the test, any rate is as likely as any other
    const rateA = m.latent('rateA', d.beta(1, 1))
    const rateB = m.latent('rateB', d.beta(1, 1))
    // the clicks out of the visitors each version was shown
    m.observe('clicksA', d.binomial(visitorsA, rateA), clicksA)
    m.observe('clicksB', d.binomial(visitorsB, rateB), clicksB)
  })
  // endregion
  say('Solving it after the first visitors, then after more and more')
  // region infer
  // what the model believes after the first n visitors of each version
  const after = (n: number): Posterior => {
    const seen = {
      visitorsA: n,
      visitorsB: n,
      clicksA: item(sum(sliceAxis(a, 0, 0, n))),
      clicksB: item(sum(sliceAxis(b, 0, 0, n))),
    }
    // Beta priors with binomial counts are conjugate: each posterior is an exact Beta curve
    return infer(toProgram(buttons.bind(seen)), undefined, { engine: 'conjugate' })
  }
  // the chance that B's rate is the higher: sampled, from many exact draws of the two Beta curves
  const chanceBBeatsA = (posterior: Posterior, draws: number): number => {
    const rates = posterior.sample(foldIn(base, 4), draws)
    const [drawsA, drawsB] = [rates['rateA'] as Tensor, rates['rateB'] as Tensor]
    return item(mean(cast(greater(drawsB, drawsA), 'float64')))
  }
  const strip = size.checkpoints.map((n) => chanceBBeatsA(after(n), size.draws))
  // endregion
  say('Reading off the rates and their difference')
  // region difference
  const final = after(VISITORS)
  const rates = final.sample(foldIn(base, 4), size.draws)
  // the difference of the two rates, one value for each pair of draws
  const gap = sub(rates['rateB'] as Tensor, rates['rateA'] as Tensor)
  const [lo, hi] = [quantile(gap, 0.05), quantile(gap, 0.95)]
  // the share of the pairs in which B's rate is the higher
  const better = item(mean(cast(greater(gap, 0), 'float64')))
  // a histogram of the differences, with an edge at zero so that 'B is better' is whole bins
  const reach = Math.max(Math.abs(lo), Math.abs(hi)) * 1.5
  const bins = histogram(gap, { bins: size.bins, range: [-reach, reach], density: true })
  // endregion
  const rateA = final.marginal('rateA')
  const rateB = final.marginal('rateB')
  const edges = toFlat(bins.edges)
  const width = (edges[1] as number) - (edges[0] as number)
  const grid = linspace(
    Math.min(item(rateA.ppf(0.0005)), item(rateB.ppf(0.0005))),
    Math.max(item(rateA.ppf(0.9995)), item(rateB.ppf(0.9995))),
    160,
  )
  const interval = (p: Distribution): Interval => ({
    mean: item(p.mean()),
    lo: item(p.ppf(0.05)),
    hi: item(p.ppf(0.95)),
  })
  return {
    tally: tally(experiment),
    truth: experiment.truth,
    grid: toFlat(grid),
    densityA: toFlat(rateA.prob(grid)),
    densityB: toFlat(rateB.prob(grid)),
    rateA: interval(rateA),
    rateB: interval(rateB),
    difference: {
      mean: item(mean(gap)),
      lo,
      hi,
      centres: edges.slice(0, -1).map((e) => e + width / 2),
      density: toFlat(bins.counts),
    },
    better,
    draws: size.draws,
    strip: { visitors: [...size.checkpoints], better: strip },
  }
}

/**
 * The run of the story: the experiment, the posteriors and the strip at full size.
 *
 * @param seed - The seed.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => compare(seed, say)
