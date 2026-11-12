/**
 * The computation of 'How people rate things': scores from 1 to 5, given by raters of different generosity to items of different
 * quality, are turned into beliefs about the quality of each item, the generosity of each rater and the cut points of the scale by
 * expectation propagation. Each rating is a factor that joins the quality of its item, the generosity of its rater and the two
 * cut points on either side of the score it received. The cut points are shared by every rating, so every rating sends them a
 * message and their beliefs narrow as ratings arrive. The model is fitted on the first few ratings, the first dozens and so on,
 * and on all of them, so the page can show the cut points narrowing. The result is plain data and a pure function of the seed
 * (which decides the order the ratings arrive in), so it runs in the story's worker. The regions are the code shown.
 */

// region imports
import { ndtr } from 'omai-engine/numerics/special'
import { epRoute } from 'omai-engine/probprog/inference'
import { defineModel } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import {
  add,
  concat,
  div,
  full,
  reshape,
  sliceAxis,
  sqrt,
  square,
  sub,
  sum,
  take,
  tensor,
  toFlat,
} from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
// endregion

import type { Say } from '../../kit/protocol.ts'
import { CATEGORIES, drawRatings, GENEROSITY_SD, ITEMS, QUALITY_SD, RATERS, type Ratings, shuffled } from './data.ts'

/** A belief about a list of quantities: a mean and a standard deviation for each. */
export interface Belief {
  readonly mean: readonly number[]
  readonly sd: readonly number[]
}

/** What the model believes after some ratings. */
export interface Fit {
  readonly cuts: Belief
  readonly quality: Belief
  readonly generosity: Belief
  /** How many ratings of each value, from 1 to 5, the fitted model expects among the ratings it was given. */
  readonly expected: readonly number[]
  /** How many sweeps of messages it took, and whether the messages settled. */
  readonly sweeps: number
  readonly settled: boolean
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The ratings, in the order they were given to the model, with the truth they were made from. */
  readonly ratings: Ratings
  /** The beliefs after all the ratings. */
  readonly final: Fit
  /** The beliefs about the cut points after the first `after` ratings, for each stop; the last stop is all the ratings. */
  readonly history: readonly { readonly after: number; readonly cuts: Belief }[]
}

/** How much to do: a test uses fewer stops. */
export interface Size {
  readonly stops: readonly number[]
}

/** The real run fits after the first 10, 25 and 50 ratings and after all of them. */
export const DEFAULT_SIZE: Size = { stops: [10, 25, 50] }

const beliefOf = (mean: readonly number[], sd: readonly number[]): Belief => ({ mean, sd })

// region model
export const ratingModel = defineModel('ratings', (m) => {
  const rating = m.range('rating')
  const item = m.range('item', { size: ITEMS })
  const rater = m.range('rater', { size: RATERS })
  const cut = m.range('cut', { size: CATEGORIES - 1 })
  // the item and the rater of each rating
  const itemOf = m.data('itemOf', { dims: [rating], support: 'index', indexes: item })
  const raterOf = m.data('raterOf', { dims: [rating], support: 'index', indexes: rater })
  // the score each rating was given, from 0 to 4
  const score = m.data('score', { dims: [rating], support: 'count' })
  // how good each item is
  const quality = m.latent('quality', d.normal(0, QUALITY_SD), { dims: [item] })
  // how generous each rater is
  const generosity = m.latent('generosity', d.normal(0, GENEROSITY_SD), {
    dims: [rater],
  })
  // the cut points of the scale: shared by every rating, kept in increasing order
  const cuts = m.latent('cuts', d.normal(0, 3), {
    dims: [cut],
    constraint: { kind: 'ordered' },
  })
  // the score before cutting: the item's quality plus the rater's generosity
  const predictor = quality.at(itemOf).add(generosity.at(raterOf))
  m.observe('score', d.orderedProbit(predictor, cuts, { over: cut }), score)
})
// endregion

/**
 * The ratings in the form the model takes them.
 *
 * @param ratings - The ratings.
 * @returns The data of the model, by name.
 */
export const dataOf = (ratings: Ratings): Record<string, readonly number[]> => ({
  itemOf: ratings.item,
  raterOf: ratings.rater,
  score: ratings.score,
})

/**
 * Fit the model to some ratings by expectation propagation.
 *
 * @param ratings - The ratings to fit.
 * @returns The beliefs, the expected counts, the sweeps taken and whether the messages settled.
 */
export function fitRatings(ratings: Ratings): Fit {
  // region infer
  // give the model the ratings
  const bound = ratingModel.bind(dataOf(ratings))
  // ask the library how expectation propagation runs this model
  const route = epRoute(bound)
  if (route.via !== 'ep') throw new Error(`no message-passing route: ${route.verdict.reason}`)
  // stop when no belief moves by more than 0.001 in a sweep
  // and visit the ratings along a tree
  const config = { tol: 1e-3, schedule: 'tree' }
  // one step is one sweep: every rating sends its messages, then the beliefs update
  const run = session(route.method, { steps: 300, mode: 'eager', logs: 'off', config })
  // the bell curve each quantity settled on
  const beliefs = route.method.posterior?.(run.state, run.config)
  // endregion
  if (beliefs === undefined) throw new Error('the method returned no beliefs')
  // region predict
  const itemOf = tensor(ratings.item)
  const raterOf = tensor(ratings.rater)
  const qualityMean = beliefs.mean('quality')
  const qualitySd = beliefs.stddev('quality')
  const generosityMean = beliefs.mean('generosity')
  const generositySd = beliefs.stddev('generosity')
  // the mean and the variance of the belief about each rating's predictor
  const centre = add(take(qualityMean, itemOf), take(generosityMean, raterOf))
  const itemVar = square(take(qualitySd, itemOf))
  const raterVar = square(take(generositySd, raterOf))
  // the score has noise of variance 1, and each cut point is uncertain too
  const rowVar = reshape(add(1, add(itemVar, raterVar)), [-1, 1])
  const cutVar = reshape(square(beliefs.stddev('cuts')), [1, -1])
  const spread = sqrt(add(rowVar, cutVar))
  // the chance that each rating is at or below each cut point
  const gap = sub(reshape(beliefs.mean('cuts'), [1, -1]), reshape(centre, [-1, 1]))
  const below = ndtr(div(gap, spread))
  // the expected number of ratings at or below each cut point
  const cumulative = concat([full([1], 0), sum(below, 0), full([1], ratings.score.length)])
  // and from those the expected number of each score
  const upper = sliceAxis(cumulative, 0, 1, CATEGORIES + 1)
  const expected = sub(upper, sliceAxis(cumulative, 0, 0, CATEGORIES))
  // endregion
  const belief = (site: string): Belief => beliefOf(toFlat(beliefs.mean(site)), toFlat(beliefs.stddev(site)))
  return {
    cuts: belief('cuts'),
    quality: belief('quality'),
    generosity: belief('generosity'),
    expected: toFlat(expected),
    sweeps: run.t,
    settled: run.status === 'converged',
  }
}

/** The first `count` ratings. */
const firstRatings = (ratings: Ratings, count: number): Ratings => ({
  ...ratings,
  item: ratings.item.slice(0, count),
  rater: ratings.rater.slice(0, count),
  score: ratings.score.slice(0, count),
})

/**
 * The inference of the story, from a seed: give the model the ratings in an order drawn from the seed, fit it after the first few
 * and after all of them.
 *
 * @param seed - Decides the order in which the ratings arrive.
 * @param say - Says what is happening, for the play button.
 * @param size - How much to do.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  say('Making the ratings')
  const ratings = shuffled(drawRatings(), seed)
  const total = ratings.score.length
  const stops = size.stops.filter((k) => k < total)
  const history = stops.map((after) => {
    say(`Passing messages for the first ${after} of ${total} ratings`)
    return { after, cuts: fitRatings(firstRatings(ratings, after)).cuts }
  })
  say(`Passing messages for all ${total} ratings`)
  const final = fitRatings(ratings)
  return { ratings, final, history: [...history, { after: total, cuts: final.cuts }] }
}

/**
 * The run of the story.
 *
 * @param seed - Decides the order of the ratings.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => fit(seed, say, DEFAULT_SIZE)
