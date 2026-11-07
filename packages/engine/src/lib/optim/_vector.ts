/**
 * The readers the public functions of `optim` share to take their numbers from a tensor-like once, at the boundary
 * (a matrix as rows of plain numbers, the elements of a value as a float64 array, a float64 vector built from a list), so the loops stay on plain arrays and the
 * tensor is built or read only at entry and exit; and `DenseMatrix`, the row-major record the dictionary and pursuit
 * routines read a matrix into. Private to the domain, as `numerics/_vector.ts` is.
 */

import { MdmlError } from '../base/index.ts'
import { asTensor, fromStorage, primalOf, type Tensor, type TensorLike, toFlat, type Value } from '../tensor/index.ts'

/** A dense matrix as row-major data with its dimensions, read once from a tensor-like. */
export interface DenseMatrix {
  readonly data: Float64Array
  readonly rows: number
  readonly cols: number
}

/**
 * The rows of a matrix given as a tensor-like, as plain numbers.
 *
 * @param x - A tensor, a typed array of rank 2 or a nested array of depth two; an empty list is no rows.
 * @param name - The caller's name, for error messages.
 * @returns The rows.
 * @throws MdmlError with code `shape` unless `x` has rank 2, and `not-differentiable` for a traced value.
 */
export function rowsOf(x: TensorLike, name: string): number[][] {
  const t = asTensor(x)
  if (t.shape.length === 1 && t.shape[0] === 0) return []
  if (t.shape.length !== 2)
    throw new MdmlError('shape', `${name}: expected a matrix, got shape [${t.shape.join(', ')}]`)
  const [n = 0, m = 0] = t.shape
  const flat = toFlat(t)
  return Array.from({ length: n }, (_, i) => flat.slice(i * m, (i + 1) * m))
}

/**
 * The elements of a value as a float64 array, copied once (the value's primal, so a traced value reads its number).
 *
 * @param x - A tensor or a traced value with a concrete primal.
 * @returns A copy of the elements, row-major.
 * @throws MdmlError with code `not-differentiable` for a trace that holds no concrete value.
 */
export function flatOf(x: Value): Float64Array {
  return Float64Array.from(primalOf(x).data)
}

/**
 * A float64 vector holding a copy of a list of numbers.
 *
 * @param a - The numbers.
 * @returns A tensor of shape `[a.length]`.
 */
export function vectorOf(a: ArrayLike<number>): Tensor<'float64', [number]> {
  return fromStorage(Float64Array.from(a), [a.length], 'float64')
}

/**
 * The row-major storage of a matrix given as a tensor-like, with its shape.
 *
 * @param x - A tensor or a nested array of depth two.
 * @param name - The caller's name, for error messages.
 * @returns A copy of the elements, row-major, and the numbers of rows and columns.
 * @throws MdmlError with code `shape` unless `x` has rank 2, and `not-differentiable` for a traced value.
 */
export function matrixOf(
  x: TensorLike,
  name: string,
): { readonly data: Float64Array; readonly rows: number; readonly cols: number } {
  const t = asTensor(x)
  if (t.shape.length !== 2)
    throw new MdmlError('shape', `${name}: expected a matrix, got shape [${t.shape.join(', ')}]`)
  const [rows = 0, cols = 0] = t.shape
  return { data: Float64Array.from(toFlat(t)), rows, cols }
}
