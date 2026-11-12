/**
 * The data of 'Gaussian mixtures': the 272 eruptions of Old Faithful from the library's dataset catalogue, each with its length
 * and the wait before it, both in minutes. They are a table lookup, so the page shows them before anything runs.
 */

import { oldFaithful } from 'omai-engine/datasets'
import { reshape, take, toFlat, type Value } from 'omai-engine/tensor'

/** The range of the eruption lengths the figures draw (minutes; the data run from 1.6 to 5.1). */
export const LENGTH_RANGE = [1.2, 5.6] as const

/** The range of the waits the figures draw (minutes; the data run from 43 to 96). */
export const WAIT_RANGE = [38, 101] as const

/** The two columns as tensors: how long each eruption lasted, and how long the wait before it was. */
export function columns(): { readonly length: Value; readonly wait: Value } {
  const table = reshape(oldFaithful().data.x, [-1, 2])
  return { length: take(table, 0, 1), wait: take(table, 1, 1) }
}

/** The eruptions as numbers. */
export function eruptions(): { readonly length: readonly number[]; readonly wait: readonly number[] } {
  const { length, wait } = columns()
  return { length: toFlat(length), wait: toFlat(wait) }
}
