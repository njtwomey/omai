/**
 * The schema for sequence-to-sequence tasks (copying, sorting, arithmetic, translation): input and target token
 * sequences padded to a common length, with masks marking the real tokens. The `sequences` constructor accepts ragged
 * sequences or already padded arrays.
 */

import type { Tensor, TensorLike } from '../../tensor/index.ts'
import { fromStorage } from '../../tensor/index.ts'
import { checkRows, concrete, metaFor, padRagged } from './_coerce.ts'
import type { DatasetMeta } from './common.ts'

/**
 * Input and target token sequences for sequence tasks (copying, sorting, arithmetic, translation), padded to a common
 * length with masks marking the real tokens.
 */
export interface SequenceData {
  readonly kind: 'sequence'
  /** Input token ids `[n, inputLength]` (int32). */
  readonly inputs: Tensor<'int32'>
  /** Target token ids `[n, targetLength]` (int32). */
  readonly targets: Tensor<'int32'>
  /** True at real (unpadded) input positions `[n, inputLength]`. */
  readonly inputMask: Tensor<'bool'>
  /** True at real target positions `[n, targetLength]`. */
  readonly targetMask: Tensor<'bool'>
  readonly vocabulary: readonly string[]
  /** The id used for padding. */
  readonly padId: number
  readonly meta: DatasetMeta
}

type Side = { readonly ids: Tensor<'int32'>; readonly mask: Tensor<'bool'> }

function side(s: readonly (readonly number[])[] | TensorLike, mask: TensorLike | undefined, padId: number): Side {
  if (Array.isArray(s) && (s.length === 0 || Array.isArray(s[0])) && mask === undefined) {
    const p = padRagged(s as readonly (readonly number[])[], padId)
    const shape = [p.lengths.length, p.length]
    return { ids: fromStorage(Int32Array.from(p.values), shape, 'int32'), mask: fromStorage(p.mask, shape, 'bool') }
  }
  const ids = concrete(s as TensorLike, 'int32', 'sequences')
  const m =
    mask === undefined
      ? fromStorage(
          Uint8Array.from(ids.data, (v) => (v === padId ? 0 : 1)),
          ids.shape,
          'bool',
        )
      : concrete(mask, 'bool', 'sequences')
  return { ids, mask: m }
}

/**
 * A sequence dataset from (possibly ragged) input and target sequences, padded with `padId`, or from padded arrays
 * with their masks (without masks, positions equal to `padId` count as padding).
 *
 * @param inputs - Input sequences (arrays of ids) or a padded `[n, T]` tensor.
 * @param targets - Target sequences or a padded tensor.
 * @param vocabulary - The token strings; id `i` is `vocabulary[i]`.
 * @param options - The padding id (default 0), masks for padded input, and metadata.
 * @returns The dataset.
 * @example
 * ```ts
 * import { sequences } from 'whitebox-engine/data'
 * const d = sequences([[1, 2, 3], [2]], [[3, 2, 1], [2]], ['<pad>', 'a', 'b', 'c'])
 * d.inputs.shape // => [2, 3]
 * Array.from(d.inputMask.data) // => [1, 1, 1, 1, 0, 0]
 * ```
 */
export function sequences(
  inputs: readonly (readonly number[])[] | TensorLike,
  targets: readonly (readonly number[])[] | TensorLike,
  vocabulary: readonly string[],
  options: {
    readonly padId?: number
    readonly inputMask?: TensorLike
    readonly targetMask?: TensorLike
    readonly meta?: Partial<DatasetMeta>
  } = {},
): SequenceData {
  const padId = options.padId ?? 0
  const a = side(inputs, options.inputMask, padId)
  const b = side(targets, options.targetMask, padId)
  checkRows('sequences', a.ids.shape[0] ?? 0, { targets: b.ids, inputMask: a.mask, targetMask: b.mask })
  return {
    kind: 'sequence',
    inputs: a.ids,
    targets: b.ids,
    inputMask: a.mask,
    targetMask: b.mask,
    vocabulary,
    padId,
    meta: metaFor('sequence', options.meta),
  }
}
