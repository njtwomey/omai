// Rating-system hub entries (private to this collection): the comparison data of the chosen dataset.
import { WhiteboxError } from 'whitebox-engine/base'
import type { ComparisonData, Dataset } from 'whitebox-engine/data'

/**
 * The comparison data of an entry's dataset.
 *
 * @param data - The dataset.
 * @returns Its comparisons.
 * @throws WhiteboxError `value` for other kinds of data.
 */
export function comparisonsOf(data: Dataset | undefined): ComparisonData {
  if (data?.kind !== 'comparisons') throw new WhiteboxError('value', 'rating: needs games or a tournament')
  return data as ComparisonData
}
