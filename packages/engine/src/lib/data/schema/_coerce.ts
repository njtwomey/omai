import { WhiteboxError } from '../../base/index.ts'
import { asTensor, cast, type DType, isTensor, type Tensor, type TensorLike } from '../../tensor/index.ts'
import type { DatasetKind, DatasetMeta } from './common.ts'

/** A concrete tensor of the given dtype (datasets hold concrete tensors, never traced values). */
export function concrete<D extends DType>(x: TensorLike, dtype: D, name: string): Tensor<D> {
  const t = asTensor(x)
  if (!isTensor(t)) throw new WhiteboxError('value', `${name}: datasets hold concrete tensors`)
  return (t.dtype === dtype ? t : cast(t, dtype)) as Tensor<D>
}

/** A concrete floating tensor, keeping float32 input as float32. */
export function concreteFloat(x: TensorLike, name: string): Tensor {
  const t = asTensor(x)
  if (!isTensor(t)) throw new WhiteboxError('value', `${name}: datasets hold concrete tensors`)
  return (t.dtype === 'float64' || t.dtype === 'float32' ? t : cast(t, 'float64')) as Tensor
}

/** Optional concrete tensor. */
export function maybe<D extends DType>(x: TensorLike | undefined, dtype: D, name: string): Tensor<D> | undefined {
  return x === undefined ? undefined : concrete(x, dtype, name)
}

/** Fill the required metadata fields. */
export function metaFor(task: DatasetKind, meta: Partial<DatasetMeta> = {}): DatasetMeta {
  return { ...meta, name: meta.name ?? 'data', description: meta.description ?? '', task }
}

/** Check that every given per-row field has `n` entries. */
export function checkRows(
  name: string,
  n: number,
  fields: Readonly<Record<string, Tensor | readonly unknown[] | undefined>>,
): void {
  for (const [f, v] of Object.entries(fields)) {
    if (v === undefined) continue
    const m = isTensor(v) ? (v.shape[0] ?? 0) : v.length
    if (m !== n) throw new WhiteboxError('shape', `${name}: ${f} has ${m} rows, expected ${n}`)
  }
}

/** Names `prefix0, prefix1, ...`. */
export function defaultNames(prefix: string, n: number): string[] {
  return Array.from({ length: n }, (_, i) => `${prefix}${i}`)
}

/** How {@link padRagged} pads and truncates. */
export interface PadLayout {
  /** The padded length (default: the longest sequence). */
  readonly maxLength?: number
  /** Pad after the tokens (`post`, default) or before them (`pre`). */
  readonly padding?: 'pre' | 'post'
  /** Drop tokens from the end (`post`, default) or the start (`pre`) of sequences longer than `maxLength`. */
  readonly truncating?: 'pre' | 'post'
}

/** Pad ragged sequences to a rectangle: row-major values, a 0/1 mask and the kept lengths. */
export function padRagged(
  seqs: readonly ArrayLike<number>[],
  padValue: number,
  layout: PadLayout = {},
): { values: Float64Array; mask: Uint8Array; lengths: Int32Array; length: number } {
  let longest = 0
  for (const s of seqs) longest = Math.max(longest, s.length)
  const t = layout.maxLength ?? longest
  const values = new Float64Array(seqs.length * t).fill(padValue)
  const mask = new Uint8Array(seqs.length * t)
  const lengths = new Int32Array(seqs.length)
  for (const [i, s] of seqs.entries()) {
    const keep = Math.min(t, s.length)
    const from = layout.truncating === 'pre' ? s.length - keep : 0
    const to = layout.padding === 'pre' ? t - keep : 0
    for (let j = 0; j < keep; j++) {
      values[i * t + to + j] = s[from + j] ?? padValue
      mask[i * t + to + j] = 1
    }
    lengths[i] = keep
  }
  return { values, mask, lengths, length: t }
}

/** One more than the largest entry (0 for none), the number of categories an index array implies. */
export function cardinality(values: ArrayLike<number>): number {
  let m = -1
  for (let i = 0; i < values.length; i++) if ((values[i] ?? -1) > m) m = values[i] ?? -1
  return m + 1
}
