// Shared plumbing for the signal domain: float views of inputs, primal number arrays, axis handling and small
// constructors. Private to signal.
import { OmaiError } from '../base/index.ts'
import type { Lti } from '../systems/index.ts'
import {
  fromStorage,
  moveaxis,
  primalOf,
  promoteFloat,
  reshape,
  type Tensor,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'

/** A value as floating point (tensors and traced values pass through). */
export function float(x: TensorLike): Value {
  return (promoteFloat(x) as [Value])[0]
}

/** A non-empty 1-D floating value. */
export function vector(x: TensorLike, name: string): Value {
  const v = float(x)
  if (v.shape.length !== 1 || (v.shape[0] ?? 0) === 0) throw new OmaiError('shape', `${name}: expected a 1-D array`)
  return v
}

/** The primal numbers of an input, row-major, as a fresh Float64Array. */
export function numbers(x: TensorLike): Float64Array {
  return Float64Array.from(primalOf(float(x)).data)
}

/** The primal numbers of a 1-D input (any shape is flattened only when 1-D; otherwise a `shape` error). */
export function samples(x: TensorLike, name: string): Float64Array {
  return Float64Array.from(primalOf(vector(x, name)).data)
}

/** A float64 tensor over existing storage. */
export function constant(data: Float64Array | readonly number[], shape?: readonly number[]): Tensor {
  const d = data instanceof Float64Array ? data : Float64Array.from(data)
  return fromStorage(d, shape === undefined ? [d.length] : [...shape], 'float64')
}

/** An int32 index tensor. */
export function indexTensor(idx: readonly number[] | Int32Array): Tensor {
  const d = idx instanceof Int32Array ? idx : Int32Array.from(idx)
  return fromStorage(d, [d.length], 'int32')
}

/** Normalise an axis against a rank. */
export function axisOf(axis: number, rank: number, name: string): number {
  const a = axis < 0 ? axis + rank : axis
  if (a < 0 || a >= rank) throw new OmaiError('value', `${name}: axis ${axis} out of range for rank ${rank}`)
  return a
}

/**
 * Move an axis to the end and flatten the others, giving `[batch, n]` rows, with the inverse for results whose last
 * axis has a new length.
 */
export function rowsOf(
  x: Value,
  axis: number,
  name: string,
): { readonly rows: Value; readonly n: number; readonly restore: (y: Value) => Value } {
  const rank = x.shape.length
  if (rank === 0) throw new OmaiError('shape', `${name}: expected at least one axis`)
  const a = axisOf(axis, rank, name)
  const moved = a === rank - 1 ? x : moveaxis(x, a, rank - 1)
  const lead = moved.shape.slice(0, -1)
  const n = moved.shape[rank - 1] ?? 0
  const batch = lead.reduce((p, s) => p * s, 1)
  return {
    rows: reshape(moved, [batch, n]),
    n,
    restore: (y: Value) => {
      const m = y.shape[y.shape.length - 1] ?? 0
      const back = reshape(y, [...lead, m])
      return a === rank - 1 ? back : moveaxis(back, rank - 1, a)
    },
  }
}

/** numpy.sinc: sin(pi x) / (pi x), 1 at 0 and exactly 0 at the other integers. */
export function sinc(x: number): number {
  if (x === 0) return 1
  if (Number.isInteger(x)) return 0
  const y = Math.PI * x
  return Math.sin(y) / y
}

/** Check a positive integer option. */
export function positiveInt(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) throw new OmaiError('value', `${name} must be a positive integer`)
  return value
}

/** Whether a value is a systems-domain LTI system (rather than an array of coefficients or sections). */
export function isLti(x: unknown): x is Lti {
  const kind = typeof x === 'object' && x !== null ? (x as { kind?: unknown }).kind : undefined
  return kind === 'tf' || kind === 'zpk' || kind === 'ss' || kind === 'sos'
}
