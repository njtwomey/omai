// Helpers shared by the linear-algebra primitives: batching of the kernels over leading axes, and the small matrix
// operations (transpose of the last two axes, diagonal embedding, symmetrisation) the derivative rules are written in.
import { OmaiError } from '../../base/index.ts'
import {
  add,
  allocate,
  broadcastShapes,
  broadcastTo,
  cast,
  diagonal,
  expandDims,
  eye,
  type FloatDType,
  mul,
  sizeOf,
  swapaxes,
  type Tensor,
  type Value,
} from '../../tensor/index.ts'
import { assertFinite } from '../_vector.ts'

/** The batch shape and matrix size of a value with at least two axes. */
export function matrixShape(shape: readonly number[], name: string): { batch: number[]; m: number; n: number } {
  if (shape.length < 2) throw new OmaiError('shape', `${name}: expected a matrix (at least two axes)`)
  return { batch: shape.slice(0, -2), m: shape[shape.length - 2] ?? 0, n: shape[shape.length - 1] ?? 0 }
}

/**
 * Run an iteration built on `inv` and `solve` for the matrix function `name`: a singular matrix inside it is reported as
 * `name: <reason>` (a `value` error), not as the message of the inner solve, which names neither the function nor the cause.
 */
export function namingSingular<T>(name: string, reason: string, run: () => T): T {
  try {
    return run()
  } catch (error) {
    if (error instanceof OmaiError && error.code === 'value' && error.message.includes('singular'))
      throw new OmaiError('value', `${name}: ${reason}`)
    throw error
  }
}

/** Require a square matrix (or batch of them) and return its size. */
export function squareSize(shape: readonly number[], name: string): number {
  const { m, n } = matrixShape(shape, name)
  if (m !== n) throw new OmaiError('shape', `${name}: expected a square matrix, got ${m} x ${n}`)
  return n
}

/**
 * Run `run`, a routine built on the factorisations, for the function `name`: a non-finite entry rejected inside it is
 * reported as `name: input has a non-finite entry ...` (a `value` error), not as the message of the inner kernel, which
 * names a function the caller did not write.
 */
export function namingNonFinite<T>(name: string, run: () => T): T {
  try {
    return run()
  } catch (error) {
    if (error instanceof OmaiError && error.code === 'value' && error.message.includes('non-finite entry'))
      throw new OmaiError('value', error.message.replace(/^[^:]*:/, `${name}:`))
    throw error
  }
}

/**
 * The matrices of a tensor as float64 copies, one per batch element. With a `name`, the routine is one that decides or
 * summarises (a factorisation or a solve), and a NaN or infinite entry throws `value` naming it (the NaN rule of the
 * domain: one O(n) scan against the O(n^3) of the kernel).
 */
export function matrices(x: Tensor, name?: string): Float64Array[] {
  const { m, n } = matrixShape(x.shape, 'matrices')
  if (name !== undefined) assertFinite(name, x.data)
  const size = m * n
  const count = size === 0 ? 0 : x.data.length / size
  return Array.from({ length: count }, (_, b) => Float64Array.from(x.data.subarray(b * size, (b + 1) * size)))
}

/** Assemble per-batch-element results into a tensor of shape [...batch, ...inner] with the given dtype. */
export function assemble(
  like: Tensor,
  dtype: FloatDType,
  batch: readonly number[],
  inner: readonly number[],
  parts: readonly ArrayLike<number>[],
): Tensor {
  const out = allocate(dtype, sizeOf(batch) * sizeOf(inner))
  const size = sizeOf(inner)
  parts.forEach((p, b) => {
    for (let i = 0; i < size; i++) out[b * size + i] = p[i] ?? 0
  })
  return like.backend.wrap(dtype, [...batch, ...inner], out)
}

/** Transpose of the last two axes. */
export function mT(x: Value): Value {
  return swapaxes(x, -1, -2)
}

/** The diagonal of the last two axes, as the last axis. */
export function diagPart(x: Value): Value {
  return diagonal(x, 0, -2, -1)
}

/** A diagonal matrix (or batch) from a vector (or batch of vectors) along the last axis. */
export function diagEmbed(v: Value): Value {
  const k = v.shape[v.shape.length - 1] ?? 0
  return mul(expandDims(v, -1), eye(k))
}

/** (X + X^T) / 2 over the last two axes. */
export function sym(x: Value): Value {
  return mul(add(x, mT(x)), 0.5)
}

/** Broadcast the batch axes of two matrix operands to a common batch shape (the matrix axes are kept). */
export function broadcastBatch(a: Value, b: Value): [Value, Value] {
  const batch = broadcastShapes(a.shape.slice(0, -2), b.shape.slice(0, -2))
  const to = (x: Value): Value => {
    const target = [...batch, ...x.shape.slice(-2)]
    return x.shape.length === target.length && x.shape.every((d, i) => d === target[i]) ? x : broadcastTo(x, target)
  }
  return [to(a), to(b)]
}

/** A 1-D right-hand side is a vector (numpy 2's rule): add a trailing axis, and say so. */
export function asColumns(_a: Value, b: Value): { b: Value; vector: boolean } {
  return b.shape.length === 1 ? { b: expandDims(b, -1), vector: true } : { b, vector: false }
}

/** The n x n identity in the dtype of `like`. */
export function identity(n: number, like: Value): Value {
  return like.dtype === 'float64' ? eye(n) : cast(eye(n), like.dtype)
}
