import { WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import { type DType, fromStorage, isTensor, stack, type Tensor, tensor } from '../tensor/index.ts'
import { take } from './rows.ts'
import { batchSampler, randomSampler, type Sampler, sequentialSampler } from './samplers.ts'
import { type PadLayout, padRagged } from './schema/_coerce.ts'
import { type Dataset, rowCount } from './schema/common.ts'

/** Options of {@link batches}. */
export interface BatchOptions {
  readonly batchSize: number
  /** Shuffle the rows first with this key. */
  readonly key?: Key
  /** Drop a final batch smaller than `batchSize`. */
  readonly dropLast?: boolean
  /** Visit rows in this sampler's order instead (it overrides `key`). */
  readonly sampler?: Sampler
  /** The epoch number passed to the sampler (default 0). */
  readonly epoch?: number
}

/**
 * Minibatches of a dataset, in order, shuffled by a key, or in a sampler's order (a torch DataLoader over an in-memory
 * dataset). Works for every schema: each batch is a dataset of the same kind.
 *
 * @param data - The dataset.
 * @param options - Batch size, shuffling key or sampler, and whether to drop the last partial batch.
 * @returns The batches.
 * @concept datasets
 * @example
 * ```ts
 * import { batches, xy } from 'whitebox-engine/data'
 * batches(xy([[1], [2], [3]], [1, 2, 3]), { batchSize: 2 }).map((b) => b.y.shape[0]) // => [2, 1]
 * ```
 */
export function batches<D extends Dataset>(data: D, options: BatchOptions): D[] {
  const n = rowCount(data)
  const sampler =
    options.sampler ??
    (options.key === undefined ? sequentialSampler(n) : (): Int32Array => permutation(options.key as Key, n).data)
  return batchSampler(
    sampler,
    options.batchSize,
    options.dropLast === true,
  )(options.epoch ?? 0).map((idx) => take(data, idx))
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
export function epochBatches<D extends Dataset>(data: D, epoch: number, options: BatchOptions): D[] {
  if (options.sampler !== undefined) return batches(data, { ...options, epoch })
  return batches(data, options.key === undefined ? options : { ...options, key: foldIn(options.key, epoch) })
}

/**
 * The batches of several epochs, each reshuffled from its own derived key (the same keys as {@link epochBatches}).
 *
 * @param data - The dataset.
 * @param nEpochs - The number of epochs.
 * @param options - Batch size, base key or sampler, and whether to drop the last partial batch.
 * @returns One list of batches per epoch.
 * @example
 * ```ts
 * import { epochs, xy } from 'whitebox-engine/data'
 * import { key } from 'whitebox-engine/random'
 * epochs(xy([[1], [2], [3]], [1, 2, 3]), 2, { batchSize: 2, key: key(0) }).map((e) => e.length) // => [2, 2]
 * ```
 */
export function epochs<D extends Dataset>(data: D, nEpochs: number, options: BatchOptions): D[][] {
  return Array.from({ length: nEpochs }, (_, e) => epochBatches(data, e, options))
}

/**
 * A reusable loader over a dataset: batches per epoch from a sampler (shuffled every epoch when a key is given),
 * as torch's `DataLoader(dataset, batch_size, shuffle, sampler, drop_last)`.
 *
 * @param data - The dataset.
 * @param options - Batch size, key or sampler, and whether to drop the last partial batch.
 * @returns `{ epoch(e), length }`: the batches of epoch `e`, and the number of batches per epoch.
 * @example
 * ```ts
 * import { dataLoader, xy } from 'whitebox-engine/data'
 * import { key } from 'whitebox-engine/random'
 * const loader = dataLoader(xy([[1], [2], [3], [4], [5]], [1, 2, 3, 4, 5]), { batchSize: 2, key: key(0), dropLast: true })
 * loader.length // => 2
 * ```
 */
export function dataLoader<D extends Dataset>(
  data: D,
  options: BatchOptions,
): { readonly epoch: (e: number) => D[]; readonly length: number } {
  const n = rowCount(data)
  const sampler = options.sampler ?? (options.key === undefined ? sequentialSampler(n) : randomSampler(n, options.key))
  const perEpoch = sampler(0).length
  const length =
    options.dropLast === true ? Math.floor(perEpoch / options.batchSize) : Math.ceil(perEpoch / options.batchSize)
  return { epoch: (e) => batches(data, { ...options, sampler, epoch: e }), length }
}

/** One item given to {@link collate}: a record of numbers, arrays and tensors. */
export type CollateItem = Readonly<Record<string, number | boolean | readonly number[] | Tensor>>

/**
 * Stack a list of items into batched tensors field by field (torch's `default_collate`): numbers become `[n]`,
 * arrays `[n, len]` and tensors gain a leading batch axis.
 *
 * @param items - The items, all with the same fields and shapes.
 * @returns A tensor per field.
 * @throws WhiteboxError with code `value` for no items or mismatched fields.
 * @example
 * ```ts
 * import { collate } from 'whitebox-engine/data'
 * collate([{ x: [1, 2], y: 0 }, { x: [3, 4], y: 1 }]).x.shape // => [2, 2]
 * ```
 */
export function collate(items: readonly CollateItem[]): Readonly<Record<string, Tensor>> {
  const first = items[0]
  if (first === undefined) throw new WhiteboxError('value', 'collate: no items')
  const out: Record<string, Tensor> = {}
  for (const name of Object.keys(first)) {
    const parts = items.map((it, i) => {
      const v = it[name]
      if (v === undefined) throw new WhiteboxError('value', `collate: item ${i} has no field ${name}`)
      return v
    })
    const head = parts[0]
    if (typeof head === 'number' || typeof head === 'boolean') {
      const dtype: DType =
        typeof head === 'boolean' ? 'bool' : parts.every((p) => Number.isInteger(p)) ? 'int32' : 'float64'
      out[name] = tensor(parts as number[], { dtype })
    } else if (isTensor(head)) {
      out[name] = stack(parts as Tensor[], 0) as Tensor
    } else {
      out[name] = stack(
        (parts as (readonly number[])[]).map((p) => tensor(p as number[])),
        0,
      ) as Tensor
    }
  }
  return out
}

/** Options of {@link padSequences}. */
export interface PadOptions extends PadLayout {
  /** The fill value (default 0). */
  readonly padValue?: number
  /** The output dtype (default float64, or int32 when every value is an integer). */
  readonly dtype?: DType
}

/**
 * Pad variable-length sequences into one rectangle with a mask (Keras `pad_sequences`, torch `pad_sequence`):
 * padding after or before the tokens, truncating long sequences from the end or the start. Tensor items `[len, ...]`
 * keep their trailing dimensions.
 *
 * @param seqs - The sequences (arrays or tensors whose first axis is time).
 * @param options - Pad value, maximum length, padding and truncating sides, dtype.
 * @returns The padded values `[n, T, ...]`, the mask `[n, T]` and the kept lengths `[n]`.
 * @example
 * ```ts
 * import { padSequences } from 'whitebox-engine/data'
 * const p = padSequences([[1, 2, 3], [4]], { padding: 'pre' })
 * Array.from(p.values.data) // => [1, 2, 3, 0, 0, 4]
 * Array.from(p.lengths.data) // => [3, 1]
 * ```
 */
export function padSequences(
  seqs: readonly (readonly number[] | Tensor)[],
  options: PadOptions = {},
): { readonly values: Tensor; readonly mask: Tensor<'bool'>; readonly lengths: Tensor<'int32'> } {
  const padValue = options.padValue ?? 0
  const trailing = seqs.find(isTensor)?.shape.slice(1) ?? []
  const width = trailing.reduce((a, b) => a * b, 1)
  // Pad each trailing coordinate as its own scalar sequence, then interleave.
  const lanes = Array.from({ length: width }, (_, c) =>
    padRagged(
      seqs.map((s) => {
        const data = isTensor(s) ? s.data : s
        const len = isTensor(s) ? (s.shape[0] ?? 0) : s.length
        return Float64Array.from({ length: len }, (_, t) => Number(data[t * width + c] ?? padValue))
      }),
      padValue,
      options,
    ),
  )
  const base =
    lanes[0] ??
    padRagged(
      seqs.map(() => []),
      padValue,
      options,
    )
  const n = seqs.length
  const t = base.length
  const values = new Float64Array(n * t * width)
  for (const [c, lane] of lanes.entries())
    for (let i = 0; i < n * t; i++) values[i * width + c] = lane.values[i] ?? padValue
  const allInt = values.every((v) => Number.isInteger(v))
  const dtype =
    options.dtype ?? (allInt && seqs.every((s) => !isTensor(s) || s.dtype === 'int32') ? 'int32' : 'float64')
  const typed = dtype === 'int32' ? Int32Array.from(values) : dtype === 'float32' ? Float32Array.from(values) : values
  return {
    values: fromStorage(typed as Float64Array, [n, t, ...trailing], dtype as 'float64'),
    mask: fromStorage(base.mask, [n, t], 'bool'),
    lengths: fromStorage(base.lengths, [n], 'int32'),
  }
}

/**
 * The length of each row of a sequence-shaped dataset: real tokens of a sequence dataset, tokens of a corpus or a
 * sequence-labelling set, words of a parallel corpus's source, steps of each trajectory row's episode.
 *
 * @param data - The dataset.
 * @returns One length per row.
 * @throws WhiteboxError with code `value` for datasets without sequence rows.
 * @example
 * ```ts
 * import { sequenceLengths, sequences } from 'whitebox-engine/data'
 * sequenceLengths(sequences([[1, 2], [3]], [[1], [2]], ['<pad>', 'a', 'b', 'c'])) // => [2, 1]
 * ```
 */
export function sequenceLengths(data: Dataset): number[] {
  switch (data.kind) {
    case 'sequence': {
      const [n = 0, t = 0] = data.inputMask.shape
      return Array.from({ length: n }, (_, i) => {
        let s = 0
        for (let j = 0; j < t; j++) s += data.inputMask.data[i * t + j] ?? 0
        return s
      })
    }
    case 'corpus':
      if (data.tokens !== undefined) return data.tokens.map((d) => d.length)
      return (data.documents ?? []).map((d) => d.split(/\s+/).filter((w) => w.length > 0).length)
    case 'sequence-labelling':
      return data.tokens.map((s) => s.length)
    case 'parallel':
      return data.source.map((s) => s.split(/\s+/).filter((w) => w.length > 0).length)
    default:
      throw new WhiteboxError('value', `sequenceLengths: ${data.kind} data has no sequence rows`)
  }
}

/** Options of {@link bucketBy}. */
export interface BucketOptions {
  /** Bucket edges: bucket `b` holds lengths in `[boundaries[b - 1], boundaries[b])`. */
  readonly boundaries: readonly number[]
  /** Rows per batch, one for all buckets or one per bucket (`boundaries.length + 1`). */
  readonly batchSize: number | readonly number[]
  /** Row lengths (default {@link sequenceLengths}). */
  readonly lengths?: readonly number[]
  /** Shuffle within buckets and the order of batches. */
  readonly key?: Key
  readonly dropLast?: boolean
}

/**
 * Batch rows of similar length together (tf.data's `bucket_by_sequence_length`), so padded batches waste little:
 * rows are grouped into length buckets, each bucket is batched, and batches are shuffled when a key is given.
 *
 * @param data - The dataset.
 * @param options - Bucket boundaries, batch sizes, lengths, key and whether to drop partial batches.
 * @returns The batches.
 * @throws WhiteboxError with code `value` when the batch sizes do not match the buckets.
 * @example
 * ```ts
 * import { bucketBy, corpus } from 'whitebox-engine/data'
 * const c = corpus({ tokens: [[1], [1, 2, 3], [2], [3, 3, 3]], vocabulary: ['a', 'b', 'c', 'd'] })
 * bucketBy(c, { boundaries: [2], batchSize: 2 }).map((b) => b.tokens?.map((t) => t.length)) // => [[1, 1], [3, 3]]
 * ```
 */
export function bucketBy<D extends Dataset>(data: D, options: BucketOptions): D[] {
  const lengths = options.lengths ?? sequenceLengths(data)
  const nb = options.boundaries.length + 1
  const sizes =
    typeof options.batchSize === 'number' ? new Array<number>(nb).fill(options.batchSize) : options.batchSize
  if (sizes.length !== nb) throw new WhiteboxError('value', `bucketBy: ${sizes.length} batch sizes for ${nb} buckets`)
  const buckets: number[][] = Array.from({ length: nb }, () => [])
  for (const [i, len] of lengths.entries()) {
    let b = 0
    while (b < options.boundaries.length && len >= (options.boundaries[b] ?? Infinity)) b++
    buckets[b]?.push(i)
  }
  const out: number[][] = []
  for (const [b, rows] of buckets.entries()) {
    const order =
      options.key === undefined
        ? rows
        : Array.from(permutation(foldIn(options.key, b), rows.length).data, (i) => rows[i] ?? 0)
    const size = sizes[b] ?? 1
    for (let s = 0; s < order.length; s += size) {
      const part = order.slice(s, s + size)
      if (options.dropLast === true && part.length < size) break
      out.push(part)
    }
  }
  const final =
    options.key === undefined
      ? out
      : Array.from(permutation(foldIn(options.key, -1 >>> 0), out.length).data, (i) => out[i] ?? [])
  return final.map((idx) => take(data, idx))
}
