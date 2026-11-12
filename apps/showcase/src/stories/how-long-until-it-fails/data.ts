/**
 * The data of 'How long until it fails': forty units put on test, made up here so that the true survival curve is known. The
 * lifetimes come from the library's `survivalData` generator with no covariates: Weibull lifetimes, with a shape of 1.5 and a
 * failure rate that sets the scale, and independent exponential follow-up times. A unit whose lifetime is longer than its follow-up
 * time is censored: the page knows its follow-up time and not its lifetime. The forty units are the same every time, so the
 * page can say how many failed and how many were censored before anything runs.
 */

import { generate } from 'omai-engine/data'
import { survivalData } from 'omai-engine/datasets'
import { foldIn, key } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'

/** How many units are on test, the seed they are drawn from, and the censoring rate of the generator. */
export const UNITS = 40
export const UNITS_SEED = 3
export const CENSORING_RATE = 0.12

/** The Weibull shape the lifetimes are drawn with, and the failure rate that sets their scale. */
export const TRUE_SHAPE = 1.5
export const FAILURE_RATE = 0.1

/** The scale of the true lifetimes: the time by which 63 in 100 units have failed. */
export const TRUE_SCALE = FAILURE_RATE ** (-1 / TRUE_SHAPE)

/** The times the survival curves are drawn over. */
export const HORIZON = 14

/** The units: when each one was last seen, and whether it failed then (1) or was still working (0). */
export interface Lifetimes {
  readonly time: readonly number[]
  readonly event: readonly number[]
  /** The true survival curve: the chance that a unit is still working at a time. */
  readonly survival: (time: number) => number
}

/**
 * The units everyone sees.
 *
 * @param n - How many units.
 * @returns Their times, whether each failed, and the true survival curve.
 */
export function lifetimes(n: number = UNITS): Lifetimes {
  const { data, truth } = generate(survivalData, {
    knobs: { n, features: 0, model: 'weibull', shape: TRUE_SHAPE, scale: FAILURE_RATE, censoring: CENSORING_RATE },
    key: foldIn(key(UNITS_SEED), 0),
  })
  const flat = toFlat(data.y)
  const survival = truth.extra?.['survival'] as (time: number, row: ArrayLike<number>) => number
  return {
    time: flat.filter((_, i) => i % 2 === 0),
    event: flat.filter((_, i) => i % 2 === 1),
    survival: (time) => survival(time, []),
  }
}
