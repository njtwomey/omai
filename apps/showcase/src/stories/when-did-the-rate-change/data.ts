/**
 * The data of 'When did the rate change?': the yearly counts of British coal-mining disasters (explosions that killed ten or more
 * miners), 1851 to 1962, from the library's dataset catalogue, and the two made-up series the story compares them with (a change
 * of the rate that is abrupt and one that is gentle), whose settings are stated here and drawn from a seed in the worker. The real
 * counts are a table lookup, so the page shows them before anything runs.
 */

import { coalMiningCounts } from 'omai-engine/datasets'
import { toFlat } from 'omai-engine/tensor'

/** The year of the first count. */
export const FIRST_YEAR = 1851

/** The yearly counts, 1851 to 1962. */
export const coalCounts = (): readonly number[] => toFlat(coalMiningCounts().values)

/** The made-up series: as long as the real one, with the rate changing at the same place and by different amounts. */
export const SIMULATED = {
  /** The index of the first year at the new rate. */
  at: 60,
  /** Events per year before and after: a drop of seven eighths, and a drop of under a third. */
  abrupt: { before: 4, after: 0.5 },
  gentle: { before: 2.2, after: 1.6 },
} as const

/** The year a series' index stands for. */
export const yearOf = (index: number): number => FIRST_YEAR + index
