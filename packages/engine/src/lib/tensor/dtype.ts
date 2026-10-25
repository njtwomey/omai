/**
 * Element types: the dtypes a tensor can hold (bool, int32, float32, float64) and their typed-array storage,
 * numpy-style type promotion between operands and with weak JavaScript-number scalars (as in numpy and JAX), and
 * `finfo` and `iinfo` for the properties of float and integer types.
 */

import { MdmlError } from '../base/index.ts'

/** Element types a tensor can hold. Arithmetic is defined on floats; `int32` and `bool` hold indices and masks. */
export type DType = 'float64' | 'float32' | 'int32' | 'bool'

/** Floating-point element types. */
export type FloatDType = 'float64' | 'float32'

/** The typed array that stores elements of a dtype. */
export type Storage<D extends DType> = D extends 'float64'
  ? Float64Array
  : D extends 'float32'
    ? Float32Array
    : D extends 'int32'
      ? Int32Array
      : Uint8Array

const RANK: Readonly<Record<DType, number>> = { bool: 0, int32: 1, float32: 2, float64: 3 }
const BY_RANK: readonly DType[] = ['bool', 'int32', 'float32', 'float64']

/**
 * Allocate zeroed storage for `size` elements of a dtype.
 *
 * @param dtype - The element type.
 * @param size - Number of elements.
 * @returns A zero-filled typed array of the matching class.
 * @example
 * ```ts
 * import { allocate } from 'mdml-engine/tensor'
 * allocate('int32', 3) instanceof Int32Array // => true
 * ```
 */
export function allocate<D extends DType>(dtype: D, size: number): Storage<D> {
  switch (dtype) {
    case 'float64':
      return new Float64Array(size) as Storage<D>
    case 'float32':
      return new Float32Array(size) as Storage<D>
    case 'int32':
      return new Int32Array(size) as Storage<D>
    case 'bool':
      return new Uint8Array(size) as Storage<D>
    default:
      throw new MdmlError('dtype', `unknown dtype ${String(dtype)}`)
  }
}

/**
 * Whether a dtype is floating point.
 *
 * @param dtype - The element type.
 * @returns `true` for `float64` and `float32`.
 * @example
 * ```ts
 * import { isFloat } from 'mdml-engine/tensor'
 * isFloat('float32') // => true
 * isFloat('bool') // => false
 * ```
 */
export function isFloat(dtype: DType): dtype is FloatDType {
  return dtype === 'float64' || dtype === 'float32'
}

/**
 * The dtype two operands promote to (numpy rules on the chain bool < int32 < float32 < float64, except that int32
 * with float32 gives float64 so integers are represented exactly).
 *
 * @param a - First dtype.
 * @param b - Second dtype.
 * @returns The common dtype.
 * @example
 * ```ts
 * import { resultType } from 'mdml-engine/tensor'
 * resultType('int32', 'float32') // => 'float64'
 * resultType('bool', 'int32') // => 'int32'
 * ```
 */
export function resultType(a: DType, b: DType): DType {
  if ((a === 'int32' && b === 'float32') || (a === 'float32' && b === 'int32')) return 'float64'
  return BY_RANK[Math.max(RANK[a], RANK[b])] ?? 'float64'
}

/**
 * The dtype a tensor promotes to when combined with a plain JavaScript number ("weak" scalars, as in numpy and JAX):
 * the tensor's dtype wins unless the number cannot be represented in it.
 *
 * @param dtype - The tensor's dtype.
 * @param scalar - The number.
 * @returns The result dtype.
 * @example
 * ```ts
 * import { weakResultType } from 'mdml-engine/tensor'
 * weakResultType('float32', 2.5) // => 'float32'
 * weakResultType('int32', 2.5) // => 'float64'
 * ```
 */
export function weakResultType(dtype: DType, scalar: number): DType {
  if (isFloat(dtype)) return dtype
  if (Number.isInteger(scalar) && Math.abs(scalar) <= 2147483647) return dtype === 'bool' ? 'int32' : dtype
  return 'float64'
}

/**
 * Floating-point properties of a float dtype.
 *
 * @param dtype - `float64` or `float32`.
 * @returns Machine epsilon, the largest finite value and the smallest normal value.
 * @example
 * ```ts
 * import { finfo } from 'mdml-engine/tensor'
 * finfo('float64').eps // => 2.220446049250313e-16
 * ```
 */
export function finfo(dtype: FloatDType): { readonly eps: number; readonly max: number; readonly tiny: number } {
  return dtype === 'float64'
    ? { eps: Number.EPSILON, max: Number.MAX_VALUE, tiny: 2.2250738585072014e-308 }
    : { eps: 1.1920928955078125e-7, max: 3.4028234663852886e38, tiny: 1.1754943508222875e-38 }
}

/**
 * Integer properties of an integer-like dtype, as numpy's `iinfo` (bool is treated as a one-bit unsigned integer).
 *
 * @param dtype - `int32` or `bool`.
 * @returns The smallest and largest values and the number of bits.
 * @example
 * ```ts
 * import { iinfo } from 'mdml-engine/tensor'
 * iinfo('int32').max // => 2147483647
 * ```
 */
export function iinfo(dtype: 'int32' | 'bool'): { readonly min: number; readonly max: number; readonly bits: number } {
  return dtype === 'int32' ? { min: -2147483648, max: 2147483647, bits: 32 } : { min: 0, max: 1, bits: 1 }
}
