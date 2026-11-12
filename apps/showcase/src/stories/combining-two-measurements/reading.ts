/**
 * What the prose of the story says about a result: the three messages in numbers, how much of the total precision each one
 * supplies, the belief worked out by hand from the messages beside the belief EP reports, and how the sweeps went. Pure functions
 * of the result, light enough for the page's bundle.
 */

import type { Message, Result } from './compute.ts'
import type { SOURCES } from './data.ts'

/** The numbers the prose quotes. */
export interface Reading {
  readonly prior: Message
  readonly digital: Message
  readonly dial: Message
  /** The sum of the three precisions: the precision of the product of the messages. */
  readonly total: number
  /** The part of the total precision each message supplies, between 0 and 1. */
  readonly share: Readonly<Record<(typeof SOURCES)[number], number>>
  /** The product of the three messages worked out by hand: the mean is the precision-weighted average of their means. */
  readonly product: { readonly mean: number; readonly sd: number }
  /** Whether the belief EP reports is the same as the product (to six decimals). */
  readonly agrees: boolean
  /** How far the belief's mean is from each reading. */
  readonly gapToDigital: number
  readonly gapToDial: number
  /** The largest change of a belief in the last sweep. */
  readonly lastChange: number | undefined
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readCombination(result: Result): Reading {
  const [prior, digital, dial] = result.messages as readonly [Message, Message, Message]
  const total = prior.precision + digital.precision + dial.precision
  const mean = (prior.precision * prior.mean + digital.precision * digital.mean + dial.precision * dial.mean) / total
  const sd = 1 / Math.sqrt(total)
  return {
    prior,
    digital,
    dial,
    total,
    share: { prior: prior.precision / total, digital: digital.precision / total, dial: dial.precision / total },
    product: { mean, sd },
    agrees: Math.abs(mean - result.belief.mean) < 1e-6 && Math.abs(sd - result.belief.sd) < 1e-6,
    gapToDigital: Math.abs(result.belief.mean - result.digital),
    gapToDial: Math.abs(result.belief.mean - result.dial),
    lastChange: result.changes.at(-1),
  }
}

/** The density of a bell curve with this mean and variance at x. */
export const bell = (mean: number, variance: number, x: number): number =>
  Math.exp(-((x - mean) ** 2) / (2 * variance)) / Math.sqrt(2 * Math.PI * variance)
