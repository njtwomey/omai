/**
 * The computation of 'A line, and how sure we are': a line with an unknown intercept, slope and noise is conditioned on twelve
 * points by NUTS, and the draws are turned into what the figures show (the mean line, a fan of lines, a credible band, and the
 * cloud of (intercept, slope) pairs). It is a pure function of the seed that returns plain data, and it runs in the story's
 * worker, so the engine is bundled there and not in the page. The regions are the code the page shows.
 */

// region imports
import { mcmc } from 'omai-engine/probprog'
import { defineModel, toProgram } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { key } from 'omai-engine/random'
import { add, arange, cast, linspace, mean, mul, reshape, sliceAxis, sort, take } from 'omai-engine/tensor'
// endregion

import { toArray, toFlat, type Value } from 'omai-engine/tensor'
import type { Say } from '../../kit/protocol.ts'
import { DRAW_RANGE, simulate } from './data.ts'

/** A posterior mean and 90% interval. */
export interface Interval {
  readonly mean: number
  readonly lo: number
  readonly hi: number
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The x values the lines are evaluated at. */
  readonly grid: readonly number[]
  /** The posterior mean of the line at each grid point. */
  readonly mean: readonly number[]
  /** The 90% credible band of the line (not of a new point) at each grid point. */
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  /** Sampled lines, one row per draw. */
  readonly fan: readonly (readonly number[])[]
  /** Some draws of (intercept, slope), for the cloud. */
  readonly cloud: { readonly intercept: readonly number[]; readonly slope: readonly number[] }
  /** The three unknowns, each with its posterior mean and 90% interval. */
  readonly summary: Readonly<Record<'intercept' | 'slope' | 'noise', Interval>>
  /** How many draws were kept, and how many the sampler flagged as divergent (it should be none). */
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
  readonly cloud: number
}

/** The size of the real run: four chains, each warmed up for 250 iterations and then kept for 300. */
export const DEFAULT_SIZE: Size = { chains: 4, warmup: 250, draws: 300, lines: 40, grid: 90, cloud: 300 }

/** The interval of a vector of draws: its mean, and its 5% and 95% points. */
function intervalOf(draws: Value): Interval {
  const sorted = sort(draws, 0)
  const count = sorted.shape[0] ?? 0
  const at = (q: number): number =>
    toFlat(sliceAxis(sorted, 0, Math.round(q * (count - 1)), Math.round(q * (count - 1)) + 1))[0] ?? 0
  return { mean: Number(toFlat(mean(draws))[0]), lo: at(0.05), hi: at(0.95) }
}

/** `kept` of the draws, spread evenly over all of them (the chains are laid end to end), so that a few stand for the whole. */
const spread = (draws: Value, kept: number): Value => {
  const count = draws.shape[0] ?? 0
  return take(draws, cast(mul(arange(kept), count / kept), 'int32'), 0)
}

/**
 * The inference, from a seed: write the model, sample its posterior given the (fixed) data, and summarise the lines it implies.
 *
 * @param seed - The sampler's seed (the data do not change with it).
 * @param say - Says what is happening, for the play button.
 * @param size - How much to sample.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  const data = simulate()
  say('Setting up the model')
  // region model
  const regression = defineModel('line', (m) => {
    const row = m.range('row')
    const x = m.data('x', { dims: [row] })
    const y = m.data('y', { dims: [row] })
    // where the line crosses x = 0
    const intercept = m.latent('intercept', d.normal(0, 5))
    // how much y rises for one more unit of x
    const slope = m.latent('slope', d.normal(0, 5))
    // how far the points scatter about the line
    const noise = m.latent('noise', d.halfNormal(2))
    m.observe('y', d.normal(intercept.add(slope.mul(x)), noise), y)
  })
  // endregion
  say('Running the sampler')
  // region infer
  const posterior = mcmc(toProgram(regression.bind(data)), undefined, {
    key: key(seed),
    numChains: size.chains,
    numWarmup: size.warmup,
    numDraws: size.draws,
  })
  // endregion
  say('Turning the draws into lines')
  const intercepts = posterior.draws?.['intercept']
  const slopes = posterior.draws?.['slope']
  const noises = posterior.draws?.['noise']
  if (intercepts === undefined || slopes === undefined || noises === undefined)
    throw new Error(`the sampler returned no draws (${posterior.status})`)
  // region lines
  const count = intercepts.shape[0] ?? 0
  const grid = linspace(DRAW_RANGE[0], DRAW_RANGE[1], size.grid)
  // One line per draw: a row for each draw, a column for each point of the grid.
  const offsets = reshape(intercepts, [count, 1])
  const tilts = reshape(slopes, [count, 1])
  const lines = add(offsets, mul(tilts, grid))
  // Sort each column: its 5% and 95% rows are the edges of a 90% band.
  const sorted = sort(lines, 0)
  const low = Math.round(0.05 * (count - 1))
  const high = Math.round(0.95 * (count - 1))
  const lower = sliceAxis(sorted, 0, low, low + 1)
  const upper = sliceAxis(sorted, 0, high, high + 1)
  const average = mean(lines, 0)
  // A few lines, spread evenly over the draws, make the fan.
  const kept = cast(mul(arange(size.lines), count / size.lines), 'int32')
  const fan = take(lines, kept, 0)
  // endregion
  return {
    grid: toFlat(grid),
    mean: toFlat(average),
    lower: toFlat(lower),
    upper: toFlat(upper),
    fan: toArray(fan) as number[][],
    cloud: { intercept: toFlat(spread(intercepts, size.cloud)), slope: toFlat(spread(slopes, size.cloud)) },
    summary: { intercept: intervalOf(intercepts), slope: intervalOf(slopes), noise: intervalOf(noises) },
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
