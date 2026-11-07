/**
 * The two ways the public functions of `numerics` hand plain numbers back as tensors: a float64 vector from a list of
 * numbers, and a float64 matrix from row-major storage. Private to the domain, so the kernels stay on plain arrays and
 * the tensor is built once, at the boundary.
 */

import { fromStorage, type Tensor } from '../tensor/index.ts'

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
 * A float64 matrix over row-major storage, without a copy: the caller hands the storage over.
 *
 * @param data - Row-major storage of `rows * cols` numbers.
 * @param rows - Number of rows.
 * @param cols - Number of columns.
 * @returns A tensor of shape `[rows, cols]`.
 */
export function matrixOf(data: Float64Array, rows: number, cols: number): Tensor<'float64', [number, number]> {
  return fromStorage(data, [rows, cols], 'float64')
}
