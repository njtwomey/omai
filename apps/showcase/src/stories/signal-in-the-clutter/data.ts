/**
 * The data of 'A signal in the clutter': made-up readings of one hidden value, so the page can say where the value really is.
 * Eight readings are the value plus a little noise (a spread of one), four are clutter (draws from a wide bell curve around zero
 * that say nothing about the value), and the order is shuffled. These twelve readings are the same every time, so the page can
 * draw them before anything runs. The second set adds four more readings that all lie near a different place, a group that could
 * pass for the signal; their noise is drawn from the run's seed.
 */

import { foldIn, key, normal, uniform } from 'omai-engine/random'
import { add, argsort, concat, full, mul, ones, take, toFlat, zeros } from 'omai-engine/tensor'

/** The value the signal readings are drawn around. */
export const SIGNAL = 3

/** Where the extra group of readings lies. */
export const DECOY = 7.5

/** The standard deviation of the clutter: the square root of ten, as in the library's clutter problem. */
export const CLUTTER_SD = Math.sqrt(10)

/** How many readings of each kind the first set has. */
export const SIGNAL_READINGS = 8
export const CLUTTER_READINGS = 4

/** How many readings the second set adds. */
export const DECOY_READINGS = 4

/** The range of values the figures draw. */
export const RANGE = [-6, 14] as const

/** The seed the first twelve readings are made from. */
export const READINGS_SEED = 3

/** A set of readings, with the kind of each (0 for a reading of the signal, 1 for clutter, 2 for the decoy group). */
export interface Readings {
  readonly values: readonly number[]
  readonly kind: readonly number[]
}

/** The twelve readings everyone sees: eight of the signal and four of clutter, in a shuffled order. */
export function clearReadings(): Readings {
  const k = key(READINGS_SEED)
  const signal = add(SIGNAL, normal(foldIn(k, 0), [SIGNAL_READINGS]))
  const clutter = mul(CLUTTER_SD, normal(foldIn(k, 1), [CLUTTER_READINGS]))
  const kinds = concat([zeros([SIGNAL_READINGS]), ones([CLUTTER_READINGS])])
  const order = argsort(uniform(foldIn(k, 3), [SIGNAL_READINGS + CLUTTER_READINGS]), 0)
  return { values: toFlat(take(concat([signal, clutter]), order)), kind: toFlat(take(kinds, order)) }
}

/**
 * The twelve readings with a decoy group added at the end.
 *
 * @param seed - Decides the noise of the decoy readings.
 * @returns The sixteen readings; the first twelve are {@link clearReadings}.
 */
export function withDecoys(seed: number): Readings {
  const clear = clearReadings()
  const decoy = add(DECOY, normal(foldIn(key(seed), 2), [DECOY_READINGS]))
  return {
    values: [...clear.values, ...toFlat(decoy)],
    kind: [...clear.kind, ...toFlat(full([DECOY_READINGS], 2))],
  }
}
