import { WhiteboxError } from '../base/index.ts'
import { type Key, permutation } from '../random/index.ts'
import { type Tensor, take as takeRows, tensor } from '../tensor/index.ts'
import { perRowFields, rowCount, type TabularData } from './schema/tabular.ts'

/**
 * The rows of a dataset at the given indices (every per-row field together).
 *
 * @param data - The dataset.
 * @param indices - Row indices.
 * @returns A dataset of the selected rows, with the same metadata.
 * @example
 * ```ts
 * import { take, xy } from 'whitebox-engine/data'
 * take(xy([[1], [2], [3]], [10, 20, 30]), [2, 0]).y.data // => Float64Array [30, 10]
 * ```
 */
export function take<D extends TabularData>(data: D, indices: readonly number[] | Tensor<'int32'>): D {
  const idx = Array.isArray(indices) ? tensor(indices as number[], { dtype: 'int32' }) : (indices as Tensor<'int32'>)
  const out: Record<string, unknown> = { ...data }
  for (const f of perRowFields(data)) out[f] = takeRows(data[f], idx, 0) as Tensor
  return out as D
}

/**
 * Shuffle the rows of a dataset.
 *
 * @param data - The dataset.
 * @param key - The random key.
 * @returns The shuffled dataset.
 * @example
 * ```ts
 * import { shuffleRows, xy } from 'whitebox-engine/data'
 * import { key } from 'whitebox-engine/random'
 * shuffleRows(xy([[1], [2], [3]], [1, 2, 3]), key(0)).y.shape // => [3]
 * ```
 */
export function shuffleRows<D extends TabularData>(data: D, key: Key): D {
  return take(data, permutation(key, rowCount(data)))
}

/** Options of {@link trainTestSplit}. */
export interface SplitOptions {
  /** The fraction of rows held out (default 0.25, as sklearn) or a number of rows. */
  readonly testSize?: number
  /** Shuffle before splitting with this key (default: split in order, without shuffling). */
  readonly key?: Key
}

/**
 * Split a dataset into training and test parts (sklearn.model_selection.train_test_split, test size rounded up).
 *
 * @param data - The dataset.
 * @param options - Test size and shuffling key.
 * @returns `{ train, test }`.
 * @example
 * ```ts
 * import { trainTestSplit, xy } from 'whitebox-engine/data'
 * trainTestSplit(xy([[1], [2], [3], [4]], [1, 2, 3, 4])).test.y.shape // => [1]
 * ```
 */
export function trainTestSplit<D extends TabularData>(
  data: D,
  options: SplitOptions = {},
): { readonly train: D; readonly test: D } {
  const n = rowCount(data)
  const size = options.testSize ?? 0.25
  const nTest = size < 1 ? Math.ceil(size * n) : Math.round(size)
  if (nTest <= 0 || nTest >= n)
    throw new WhiteboxError('value', `trainTestSplit: test size ${nTest} leaves no rows to train on`)
  const order =
    options.key === undefined ? Array.from({ length: n }, (_, i) => i) : Array.from(permutation(options.key, n).data)
  return { train: take(data, order.slice(0, n - nTest)), test: take(data, order.slice(n - nTest)) }
}
