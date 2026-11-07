/**
 * The helpers the public functions of `numerics` share at their boundary. Two ways to hand plain numbers back as
 * tensors: a float64 vector from a list of numbers, and a float64 matrix from row-major storage, so the kernels stay on
 * plain arrays and the tensor is built once. And the one place the NaN rule of the domain is enforced for routines that
 * summarise or decide (factorisations, solvers, queries): a single scan that rejects a non-finite entry with a `value`
 * error naming the function. Private to the domain.
 */

import { MdmlError } from '../base/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'

/**
 * Throw a `value` error when any entry is NaN or infinite: the boundary check of a routine that decides or summarises
 * (a factorisation, a solver, a neighbour query), where a non-finite entry would otherwise leave garbage that looks like
 * an answer. One O(n) pass, early exit at the first offender. Elementwise operations do not call it: they propagate.
 *
 * @param name - The calling function, as the user wrote it (`eigh`).
 * @param data - The numbers to scan.
 * @param what - What they are, for the message (default `input`).
 */
export function assertFinite(name: string, data: ArrayLike<number>, what = 'input'): void {
  for (let i = 0; i < data.length; i++) {
    const v = data[i] as number
    if (v - v !== 0) {
      throw new MdmlError(
        'value',
        `${name}: ${what} has a non-finite entry (${Number.isNaN(v) ? 'NaN' : v > 0 ? 'Infinity' : '-Infinity'} at index ${i}); the routine needs finite numbers`,
      )
    }
  }
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
