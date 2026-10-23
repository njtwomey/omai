import { foldIn, type Key } from '../random/index.ts'
import { shuffleRows, take } from './rows.ts'
import { rowCount, type TabularData } from './schema/tabular.ts'

/** Options of {@link batches}. */
export interface BatchOptions {
  readonly batchSize: number
  /** Shuffle the rows first with this key. */
  readonly key?: Key
  /** Drop a final batch smaller than `batchSize`. */
  readonly dropLast?: boolean
}

/**
 * Minibatches of a dataset, in order or shuffled by a key (a torch DataLoader over an in-memory dataset).
 *
 * @param data - The dataset.
 * @param options - Batch size, shuffling key and whether to drop the last partial batch.
 * @returns The batches.
 * @concept datasets
 * @example
 * ```ts
 * import { batches, xy } from 'whitebox-engine/data'
 * batches(xy([[1], [2], [3]], [1, 2, 3]), { batchSize: 2 }).map((b) => b.y.shape[0]) // => [2, 1]
 * ```
 */
export function batches<D extends TabularData>(data: D, options: BatchOptions): D[] {
  const rows = options.key === undefined ? data : shuffleRows(data, options.key)
  const n = rowCount(rows)
  const out: D[] = []
  for (let start = 0; start < n; start += options.batchSize) {
    const stop = Math.min(n, start + options.batchSize)
    if (options.dropLast === true && stop - start < options.batchSize) break
    out.push(
      take(
        rows,
        Array.from({ length: stop - start }, (_, i) => start + i),
      ),
    )
  }
  return out
}

/**
 * The batches of epoch `epoch`, each epoch shuffled with its own key derived from `key`.
 *
 * @param data - The dataset.
 * @param epoch - The epoch number.
 * @param options - Batch size, base key and whether to drop the last partial batch.
 * @returns The epoch's batches.
 * @example
 * ```ts
 * import { epochBatches, xy } from 'whitebox-engine/data'
 * import { key } from 'whitebox-engine/random'
 * epochBatches(xy([[1], [2], [3], [4]], [1, 2, 3, 4]), 0, { batchSize: 2, key: key(0) }).length // => 2
 * ```
 */
export function epochBatches<D extends TabularData>(data: D, epoch: number, options: BatchOptions): D[] {
  return batches(data, options.key === undefined ? options : { ...options, key: foldIn(options.key, epoch) })
}
