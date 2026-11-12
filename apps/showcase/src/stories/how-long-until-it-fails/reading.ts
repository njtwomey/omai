/**
 * What the prose of 'How long until it fails' says: from the units alone (how many failed, how long the follow-up lasted), and from
 * the result (the scale and shape and the times they imply, how the settled belief compares with the exact one, and how the
 * fitted curves compare with the true one). Pure functions, light enough for the page's bundle.
 */

import type { Bell, Fit, Result } from './compute.ts'
import { TRUE_SCALE, TRUE_SHAPE } from './data.ts'

/** The numbers about the units alone. */
export interface DataReading {
  readonly count: number
  readonly failed: number
  readonly censored: number
  /** The longest time any unit was followed, and the median follow-up time of the censored units. */
  readonly longest: number
  readonly censoredMedian: number
}

/**
 * The numbers about the units alone.
 *
 * @param time - When each unit was last seen.
 * @param event - Whether it failed then (1) or was still working (0).
 * @returns The counts and times.
 */
export function readData(time: readonly number[], event: readonly number[]): DataReading {
  const censored = time.filter((_, i) => event[i] === 0).sort((a, b) => a - b)
  return {
    count: time.length,
    failed: event.filter((e) => e === 1).length,
    censored: censored.length,
    longest: Math.max(...time),
    censoredMedian: censored[Math.floor(censored.length / 2)] ?? Number.NaN,
  }
}

/** A quantity that is the exponential of a bell-curve variable: its middle and its 90% interval. */
export interface Quantity {
  readonly value: number
  readonly low: number
  readonly high: number
}

/** The exponential of a bell curve's mean, with the exponentials of its mean plus and minus 1.645 spreads. */
export const quantityOf = (bell: Bell): Quantity => ({
  value: Math.exp(bell.mean),
  low: Math.exp(bell.mean - 1.645 * bell.sd),
  high: Math.exp(bell.mean + 1.645 * bell.sd),
})

/** The time by which half the units have failed, for a scale and a shape: scale times (log 2) to the power of 1 over the shape. */
export const medianLife = (scale: number, shape: number): number => scale * Math.LN2 ** (1 / shape)

/** The median life at the middle of a fit. */
export const medianOf = (fit: Fit): number => medianLife(Math.exp(fit.scale.mean), Math.exp(fit.shape.mean))

/** The numbers the prose quotes after the run. */
export interface Reading {
  readonly scale: Quantity
  readonly shape: Quantity
  /** The median life: the true one, the fitted one and the one fitted without the censored units. */
  readonly median: { readonly truth: number; readonly fitted: number; readonly dropped: number }
  /** The spread of the belief about each unknown as a share of the exact spread. */
  readonly spreadShare: { readonly scale: number; readonly shape: number }
  /** The largest difference between a mean of EP and the exact one, in exact spreads. */
  readonly meanOff: number
  /** The first sweep's spread of the scale as a share of the settled one. */
  readonly firstSweepShare: number
  /** What the unit that is taken apart did to the cavity: how the mean of the log scale moved. */
  readonly moved: { readonly failed: 'down' | 'up'; readonly censored: 'down' | 'up' }
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The numbers.
 */
export function readResult(result: Result): Reading {
  const { all, exact, update, sharpening } = result
  const direction = (c: { readonly before: Bell; readonly after: Bell }): 'down' | 'up' =>
    c.after.mean < c.before.mean ? 'down' : 'up'
  const last = sharpening.scale.length - 1
  return {
    scale: quantityOf(all.scale),
    shape: quantityOf(all.shape),
    median: {
      truth: medianLife(TRUE_SCALE, TRUE_SHAPE),
      fitted: medianOf(all),
      dropped: medianOf(result.dropped),
    },
    spreadShare: { scale: all.scale.sd / exact.scale.sd, shape: all.shape.sd / exact.shape.sd },
    meanOff: Math.max(
      Math.abs(all.scale.mean - exact.scale.mean) / exact.scale.sd,
      Math.abs(all.shape.mean - exact.shape.mean) / exact.shape.sd,
    ),
    firstSweepShare: (sharpening.scale[last] as number) / all.scale.sd,
    moved: { failed: direction(update.failed), censored: direction(update.censored) },
  }
}
