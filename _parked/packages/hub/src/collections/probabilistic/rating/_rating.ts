// Rating-system hub entries (private to this collection): the comparison data of the chosen dataset.
import { MdmlError } from 'mdml-engine/base'
import type { ComparisonData, Dataset } from 'mdml-engine/data'

/**
 * The comparison data of an entry's dataset.
 *
 * @param data - The dataset.
 * @returns Its comparisons.
 * @throws MdmlError `value` for other kinds of data.
 */
export function comparisonsOf(data: Dataset | undefined): ComparisonData {
  if (data?.kind !== 'comparisons') throw new MdmlError('value', 'rating: needs games or a tournament')
  return data as ComparisonData
}
