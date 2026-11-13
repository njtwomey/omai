/**
 * The data of 'How people rate things': ratings from 1 to 5 given by raters of different generosity to items of different quality,
 * drawn from the library's `ordinalRatings` generator, which knows the true quality of every item, the generosity of every
 * rater and the cut points of the scale. The ratings are the same every time, so the page can describe them before anything runs;
 * the run's seed only decides the order in which the ratings are handed to the model.
 */

import { generate } from 'omai-engine/data'
import { ordinalRatings } from 'omai-engine/datasets'
import { foldIn, key, permutation } from 'omai-engine/random'
import { arange, equal, reshape, sum, type Tensor, take, tensor, toFlat } from 'omai-engine/tensor'

/** The number of raters and items, the share of pairs rated, and the seed the ratings are drawn from. */
export const RATERS = 15
export const ITEMS = 20
export const DENSITY = 0.33
export const DATA_SEED = 4

/** The number of rating values, and so of cut points plus one. */
export const CATEGORIES = 5

/** The spread of the item qualities and the rater generosities the ratings were made with, which the model is also given. */
export const QUALITY_SD = 1
export const GENEROSITY_SD = 0.5

/** Ratings, in the order they arrive, with the truth they were made from. */
export interface Ratings {
  /** The item and the rater of each rating, counted from 0. */
  readonly item: readonly number[]
  readonly rater: readonly number[]
  /** Each rating, from 0 to 4 (the page shows 1 to 5). */
  readonly score: readonly number[]
  /** The true quality of each item, the true generosity of each rater and the true cut points. */
  readonly quality: readonly number[]
  readonly generosity: readonly number[]
  readonly cuts: readonly number[]
}

/**
 * The ratings everyone sees.
 *
 * @returns The ratings in the order the generator made them.
 */
export function drawRatings(): Ratings {
  const { data, truth } = generate(ordinalRatings, {
    key: key(DATA_SEED),
    knobs: {
      raters: RATERS,
      items: ITEMS,
      density: DENSITY,
      categories: CATEGORIES,
      qualitySd: QUALITY_SD,
      generositySd: GENEROSITY_SD,
    },
  })
  return {
    item: [...toFlat(data.items)],
    rater: [...toFlat(data.users)],
    // the generator rates from 1 to 5, the model counts from 0
    score: [...toFlat(data.values)].map((v) => v - 1),
    quality: [...toFlat(truth.latents?.['quality'] as Tensor)],
    generosity: [...toFlat(truth.latents?.['generosity'] as Tensor)],
    cuts: [...toFlat(truth.extra?.['cuts'] as Tensor)],
  }
}

/**
 * The same ratings in another order, drawn from a seed: the order the model takes them in.
 *
 * @param ratings - The ratings.
 * @param seed - Decides the order.
 * @returns The ratings, in a new order.
 */
export function shuffled(ratings: Ratings, seed: number): Ratings {
  const order = permutation(foldIn(key(seed), 7), ratings.score.length)
  const pick = (v: readonly number[]): number[] => [...toFlat(take(tensor(v), order))]
  return { ...ratings, item: pick(ratings.item), rater: pick(ratings.rater), score: pick(ratings.score) }
}

/**
 * How many ratings have each value, from 1 to 5.
 *
 * @param score - The ratings, from 0 to 4.
 * @returns The five counts.
 */
export function counts(score: readonly number[]): number[] {
  return [...toFlat(sum(equal(reshape(tensor(score), [-1, 1]), arange(CATEGORIES)), 0))]
}
