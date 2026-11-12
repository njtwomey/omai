/**
 * The data of 'Two humps': the 272 waiting times (minutes between one eruption of Old Faithful and the next) from the library's
 * dataset catalogue. They are a lookup in a table, so the page shows their histogram before anything runs; the inference, which
 * is the computation the reader asks for by pressing play, happens in the worker.
 */

import { oldFaithful } from 'omai-engine/datasets'
import { reshape, take, toFlat, type Value } from 'omai-engine/tensor'

/** The range of waiting times the figures draw, in minutes (the data run from 43 to 96). */
export const WAIT_RANGE = [38, 101] as const

/** The waiting times as a tensor (the second column of the dataset). */
export function waitingTimes(): Value {
  return take(reshape(oldFaithful().data.x, [-1, 2]), 1, 1)
}

/** The waiting times as numbers. */
export const waits = (): readonly number[] => toFlat(waitingTimes())
