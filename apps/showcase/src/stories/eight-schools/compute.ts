/**
 * The computation of 'Eight schools, and why the average is not enough': the hierarchical model of the eight schools (each school's
 * effect drawn from a common distribution with an unknown mean and spread) is conditioned on the classic estimates by NUTS, and
 * the draws are turned into what the figures show (each school's shrunk estimate and interval, how hard each was pulled, and the
 * posterior of the spread between schools). It is a pure function of the seed that returns plain data, and it runs in the story's
 * worker. The regions are the code the page shows.
 */

// region imports
import { mcmc } from 'omai-engine/probprog'
import { toProgram } from 'omai-engine/probprog/model'
import { eightSchoolsDefinition } from 'omai-engine/probprog/models'
import { key } from 'omai-engine/random'
import { add, div, mean, reshape, sliceAxis, sort, square, tensor, toFlat } from 'omai-engine/tensor'
// endregion

import { quantile } from 'omai-engine/stats'
import type { Say } from '../../kit/protocol.ts'
import { EFFECTS, ERRORS, NAMES } from './data.ts'

/** A posterior mean and the 5% and 95% points (a 90% interval). */
export interface Interval {
  readonly mean: number
  readonly lo: number
  readonly hi: number
}

/** One school: its raw estimate and what the model makes of it. */
export interface School {
  readonly name: string
  /** The raw estimate and its standard error (the data). */
  readonly estimate: number
  readonly error: number
  /** The school's effect after pooling: its posterior mean and 90% interval. */
  readonly pooled: Interval
  /** How much of the way to the common mean the model pulls this school: the average over the draws of the spread's share. */
  readonly pull: number
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  readonly schools: readonly School[]
  /** The common mean effect. */
  readonly mu: Interval
  /** The spread of true effects between schools: its interval and median, and all its draws (for the histogram). */
  readonly tau: Interval & { readonly median: number }
  readonly tauDraws: readonly number[]
  /** How many draws were kept, and how many the sampler flagged as divergent (it should be none). */
  readonly draws: number
  readonly divergences: number
}

/** How much to sample: the run uses {@link DEFAULT_SIZE}, a test a small one. */
export interface Size {
  readonly chains: number
  readonly warmup: number
  readonly draws: number
}

/** The size of the real run: four chains, each warmed up for 250 iterations and then kept for 300. */
export const DEFAULT_SIZE: Size = { chains: 4, warmup: 250, draws: 300 }

/**
 * The inference, from a seed: write the model, sample its posterior given the classic data, and summarise what it says about each
 * school and about the spread between schools.
 *
 * @param seed - The sampler's seed (the data do not change with it).
 * @param say - Says what is happening, for the play button.
 * @param size - How much to sample.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  say('Writing the model: eight schools drawn from one population')
  // region model
  const schools = eightSchoolsDefinition('non-centred')
  // endregion
  say('Warming the sampler up, then drawing from the posterior')
  // region infer
  const posterior = mcmc(toProgram(schools.bind({ y: EFFECTS, sigma: ERRORS })), undefined, {
    key: key(seed),
    numChains: size.chains,
    numWarmup: size.warmup,
    numDraws: size.draws,
    targetAcceptance: 0.9,
  })
  // endregion
  say('Reading off what each school looks like now')
  const mus = posterior.draws?.['mu']
  const taus = posterior.draws?.['tau']
  const thetas = posterior.draws?.['theta']
  if (mus === undefined || taus === undefined || thetas === undefined)
    throw new Error(`the sampler returned no draws (${posterior.status})`)
  // region summary
  const count = thetas.shape[0] ?? 0
  // Sort each school's draws: its 5% and 95% rows are the edges of a 90% interval.
  const sorted = sort(thetas, 0)
  const low = Math.round(0.05 * (count - 1))
  const high = Math.round(0.95 * (count - 1))
  const lower = sliceAxis(sorted, 0, low, low + 1)
  const upper = sliceAxis(sorted, 0, high, high + 1)
  const average = mean(thetas, 0)
  // Given the spread tau, school j is pulled by sigma_j² / (sigma_j² + tau²) of the way to the mean.
  const noise = square(tensor(ERRORS))
  const share = mean(div(noise, add(noise, reshape(square(taus), [count, 1]))), 0)
  // endregion
  const [tauLo = 0, tauMid = 0, tauHi = 0] = toFlat(quantile(taus, [0.05, 0.5, 0.95]))
  const [muLo = 0, , muHi = 0] = toFlat(quantile(mus, [0.05, 0.5, 0.95]))
  const averages = toFlat(average)
  const lowers = toFlat(lower)
  const uppers = toFlat(upper)
  const pulls = toFlat(share)
  return {
    schools: NAMES.map((name, j) => ({
      name,
      estimate: EFFECTS[j] ?? 0,
      error: ERRORS[j] ?? 0,
      pooled: { mean: averages[j] ?? 0, lo: lowers[j] ?? 0, hi: uppers[j] ?? 0 },
      pull: pulls[j] ?? 0,
    })),
    mu: { mean: Number(toFlat(mean(mus))[0]), lo: muLo, hi: muHi },
    tau: { mean: Number(toFlat(mean(taus))[0]), lo: tauLo, hi: tauHi, median: tauMid },
    tauDraws: toFlat(taus),
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
