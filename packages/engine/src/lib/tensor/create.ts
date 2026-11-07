/**
 * Tensor constructors, numpy style: filled arrays (`zeros`, `ones`, `full` and their `Like` forms), ranges and spacings
 * (`arange`, `linspace`, `logspace`, `geomspace`), identity, diagonal and triangular matrices, `meshgrid` and index
 * grids.
 *
 * @index function Tensor constructors: zeros, ones, full, arange, linspace, eye.
 */

import { MdmlError } from '../base/index.ts'
import { cpu } from './cpu.ts'
import { allocate, type DType } from './dtype.ts'
import { cast } from './ops/arithmetic.ts'
import { equal, where } from './ops/compare.ts'
import { type Shape, sizeOf } from './shape.ts'
import type { CreateOptions, Tensor, TensorLike } from './tensor.ts'
import { asTensor } from './tensor.ts'
import type { Value } from './traced.ts'

/**
 * A tensor filled with one value (like `numpy.full`).
 *
 * @param shape - The shape.
 * @param value - The fill value.
 * @param options - The dtype (default float64) and backend.
 * @returns The filled tensor.
 * @example
 * ```ts
 * import { full, toArray } from 'mdml-engine/tensor'
 * toArray(full([2], 7)) // => [7, 7]
 * ```
 */
export function full<D extends DType = 'float64'>(
  shape: Shape,
  value: number,
  options: CreateOptions<D> = {},
): Tensor<D> {
  const dtype = (options.dtype ?? 'float64') as D
  const data = allocate(dtype, sizeOf(shape))
  data.fill(dtype === 'int32' ? Math.trunc(value) : dtype === 'bool' ? (value ? 1 : 0) : value)
  return (options.backend ?? cpu).wrap(dtype, shape, data)
}

/**
 * A tensor of zeros (like `numpy.zeros`).
 *
 * @param shape - The shape.
 * @param options - The dtype and backend.
 * @returns The zero tensor.
 * @example
 * ```ts
 * import { zeros } from 'mdml-engine/tensor'
 * zeros([2, 3]).shape // => [2, 3]
 * ```
 */
export function zeros<D extends DType = 'float64'>(shape: Shape, options: CreateOptions<D> = {}): Tensor<D> {
  return full(shape, 0, options)
}

/**
 * A tensor of ones (like `numpy.ones`).
 *
 * @param shape - The shape.
 * @param options - The dtype and backend.
 * @returns The tensor of ones.
 * @example
 * ```ts
 * import { ones, toArray } from 'mdml-engine/tensor'
 * toArray(ones([3])) // => [1, 1, 1]
 * ```
 */
export function ones<D extends DType = 'float64'>(shape: Shape, options: CreateOptions<D> = {}): Tensor<D> {
  return full(shape, 1, options)
}

/**
 * Zeros with the shape and dtype of another value.
 *
 * @param x - The template.
 * @returns A zero tensor.
 * @example
 * ```ts
 * import { zerosLike, tensor } from 'mdml-engine/tensor'
 * zerosLike(tensor([[1, 2]])).shape // => [1, 2]
 * ```
 */
export function zerosLike(x: TensorLike): Tensor {
  const v = asTensor(x)
  return full(v.shape, 0, { dtype: v.dtype })
}

/**
 * Ones with the shape and dtype of another value.
 *
 * @param x - The template.
 * @returns A tensor of ones.
 * @example
 * ```ts
 * import { onesLike, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(onesLike(tensor([5, 6]))) // => [1, 1]
 * ```
 */
export function onesLike(x: TensorLike): Tensor {
  const v = asTensor(x)
  return full(v.shape, 1, { dtype: v.dtype })
}

/**
 * A constant with the shape and dtype of another value.
 *
 * @param x - The template.
 * @param value - The fill value.
 * @returns The filled tensor.
 * @example
 * ```ts
 * import { fullLike, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(fullLike(tensor([1, 2]), 9)) // => [9, 9]
 * ```
 */
export function fullLike(x: TensorLike, value: number): Tensor {
  const v: Value = asTensor(x)
  return full(v.shape, value, { dtype: v.dtype })
}

/**
 * Evenly spaced values in `[start, stop)` (like `numpy.arange`); with one argument, `[0, start)`.
 *
 * @param start - First value (or the stop when it is the only argument).
 * @param stop - Exclusive end.
 * @param step - Spacing (default 1, may be negative).
 * @returns A 1-D float64 tensor.
 * @example
 * ```ts
 * import { arange, toArray } from 'mdml-engine/tensor'
 * toArray(arange(0, 5, 2)) // => [0, 2, 4]
 * toArray(arange(3)) // => [0, 1, 2]
 * ```
 */
export function arange(start: number, stop?: number, step = 1): Tensor<'float64'> {
  const [lo, hi] = stop === undefined ? [0, start] : [start, stop]
  if (step === 0) throw new MdmlError('value', 'arange: step must be non-zero')
  const n = Math.max(0, Math.ceil((hi - lo) / step))
  const data = new Float64Array(n)
  for (let i = 0; i < n; i++) data[i] = lo + i * step
  return cpu.wrap('float64', [n], data)
}

/**
 * `num` evenly spaced values from `start` to `stop` inclusive (like `numpy.linspace`).
 *
 * @param start - First value.
 * @param stop - Last value.
 * @param num - Number of values (at least 1).
 * @returns A 1-D float64 tensor.
 * @example
 * ```ts
 * import { linspace, toArray } from 'mdml-engine/tensor'
 * toArray(linspace(0, 1, 5)) // => [0, 0.25, 0.5, 0.75, 1]
 * ```
 */
export function linspace(start: number, stop: number, num: number): Tensor<'float64'> {
  if (!Number.isInteger(num) || num < 1) throw new MdmlError('value', 'linspace: num must be a positive integer')
  const data = new Float64Array(num)
  const step = num === 1 ? 0 : (stop - start) / (num - 1)
  for (let i = 0; i < num; i++) data[i] = i === num - 1 && num > 1 ? stop : start + i * step
  return cpu.wrap('float64', [num], data)
}

/**
 * Values spaced evenly on a log scale: `base ** linspace(start, stop, num)` (like `numpy.logspace`).
 *
 * @param start - First exponent.
 * @param stop - Last exponent.
 * @param num - Number of values.
 * @param base - The base (default 10).
 * @returns A 1-D float64 tensor.
 * @example
 * ```ts
 * import { logspace, toArray } from 'mdml-engine/tensor'
 * toArray(logspace(0, 2, 3)) // => [1, 10, 100]
 * ```
 */
export function logspace(start: number, stop: number, num: number, base = 10): Tensor<'float64'> {
  const exps = linspace(start, stop, num)
  return cpu.wrap(
    'float64',
    [num],
    exps.data.map((e) => base ** e),
  )
}

/**
 * Values spaced evenly in ratio from `start` to `stop` (like `numpy.geomspace`).
 *
 * @param start - First value (non-zero).
 * @param stop - Last value (same sign).
 * @param num - Number of values.
 * @returns A 1-D float64 tensor.
 * @example
 * ```ts
 * import { geomspace, toArray } from 'mdml-engine/tensor'
 * toArray(geomspace(1, 1000, 4)) // => [1, 10, 100, 1000]
 * ```
 */
export function geomspace(start: number, stop: number, num: number): Tensor<'float64'> {
  if (start === 0 || stop === 0 || Math.sign(start) !== Math.sign(stop)) {
    throw new MdmlError('value', 'geomspace: endpoints must be non-zero with the same sign')
  }
  const out = logspace(Math.log10(Math.abs(start)), Math.log10(Math.abs(stop)), num)
  const data = out.data.map((v) => Math.sign(start) * v)
  data[0] = start
  if (num > 1) data[num - 1] = stop
  return cpu.wrap('float64', [num], data)
}

/**
 * The identity matrix, or an `n` by `m` matrix with ones on a diagonal (like `numpy.eye`).
 *
 * @param n - Rows.
 * @param m - Columns (default `n`).
 * @param k - Diagonal offset.
 * @returns A float64 matrix.
 * @example
 * ```ts
 * import { eye, toArray } from 'mdml-engine/tensor'
 * toArray(eye(2)) // => [[1, 0], [0, 1]]
 * ```
 */
export function eye(n: number, m: number = n, k = 0): Tensor<'float64'> {
  const data = new Float64Array(n * m)
  for (let i = 0; i < n; i++) if (i + k >= 0 && i + k < m) data[i * m + i + k] = 1
  return cpu.wrap('float64', [n, m], data)
}

/**
 * A square matrix with the given values on a diagonal (from a concrete 1-D input).
 *
 * @param values - The diagonal values.
 * @param k - Diagonal offset.
 * @returns The matrix.
 * @example
 * ```ts
 * import { diag, toArray } from 'mdml-engine/tensor'
 * toArray(diag([1, 2])) // => [[1, 0], [0, 2]]
 * ```
 */
export function diag(values: TensorLike, k = 0): Tensor {
  const v = asTensor(values)
  if (v.kind !== 'tensor' || v.shape.length !== 1) throw new MdmlError('shape', 'diag: expects a concrete 1-D input')
  const n = (v.shape[0] ?? 0) + Math.abs(k)
  const data = allocate(v.dtype, n * n)
  v.data.forEach((x, i) => {
    const r = k >= 0 ? i : i - k
    data[r * n + r + k] = x
  })
  return v.backend.wrap(v.dtype, [n, n], data)
}

/**
 * A matrix with ones at and below a diagonal (like `numpy.tri`).
 *
 * @param n - Rows.
 * @param m - Columns (default `n`).
 * @param k - Diagonal offset.
 * @returns A float64 matrix.
 * @example
 * ```ts
 * import { tri, toArray } from 'mdml-engine/tensor'
 * toArray(tri(2)) // => [[1, 0], [1, 1]]
 * ```
 */
export function tri(n: number, m: number = n, k = 0): Tensor<'float64'> {
  const data = new Float64Array(n * m)
  for (let i = 0; i < n; i++) for (let j = 0; j <= Math.min(m - 1, i + k); j++) data[i * m + j] = 1
  return cpu.wrap('float64', [n, m], data)
}

/**
 * A grid of coordinates from 1-D axes (numpy `meshgrid`; `xy` indexing by default).
 *
 * @param axes - The coordinate vectors.
 * @param indexing - `xy` (Cartesian: the first two output axes are swapped) or `ij` (matrix) indexing.
 * @returns One coordinate tensor per input.
 * @example
 * ```ts
 * import { meshgrid } from 'mdml-engine/tensor'
 * meshgrid([[1, 2], [3, 4, 5]]).map((g) => g.shape) // => [[3, 2], [3, 2]]
 * ```
 */
export function meshgrid(axes: readonly TensorLike[], indexing: 'xy' | 'ij' = 'xy'): Tensor<'float64'>[] {
  const values = axes.map((a) => {
    const v = asTensor(a)
    if (v.kind !== 'tensor') throw new MdmlError('value', 'meshgrid: expects concrete inputs')
    return Array.from(v.data)
  })
  const lengths = values.map((a) => a.length)
  const order = values.map((_, k) => k)
  if (indexing === 'xy' && order.length >= 2) [order[0], order[1]] = [1, 0]
  const shape = order.map((k) => lengths[k] ?? 1)
  const size = sizeOf(shape)
  return values.map((vals, k) => {
    const axis = order.indexOf(k)
    const inner = sizeOf(shape.slice(axis + 1))
    const data = new Float64Array(size)
    for (let i = 0; i < size; i++) data[i] = vals[Math.floor(i / inner) % (shape[axis] ?? 1)] ?? 0
    return cpu.wrap('float64', shape, data)
  })
}

/**
 * The lower triangle of a matrix (entries above the `k`th diagonal set to zero, whatever they hold: Infinity and NaN
 * above the diagonal do not leak; the dtype is kept, so a bool mask stays a bool mask).
 *
 * @param x - A matrix (or batch of matrices).
 * @param k - Diagonal offset.
 * @returns The lower triangle.
 * @example
 * ```ts
 * import { tril, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(tril(tensor([[1, 2], [3, 4]]))) // => [[1, 0], [3, 4]]
 * ```
 */
export function tril(x: TensorLike, k = 0): Value {
  const v = asTensor(x)
  const n = v.shape[v.shape.length - 2] ?? 1
  const m = v.shape[v.shape.length - 1] ?? 1
  return where(cast(tri(n, m, k), 'bool'), v, zeros([], { dtype: v.dtype }))
}

/**
 * The upper triangle of a matrix (entries below the `k`th diagonal set to zero, whatever they hold: Infinity and NaN
 * below the diagonal do not leak; the dtype is kept).
 *
 * @param x - A matrix (or batch of matrices).
 * @param k - Diagonal offset.
 * @returns The upper triangle.
 * @example
 * ```ts
 * import { triu, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(triu(tensor([[1, 2], [3, 4]]))) // => [[1, 2], [0, 4]]
 * ```
 */
export function triu(x: TensorLike, k = 0): Value {
  const v = asTensor(x)
  const n = v.shape[v.shape.length - 2] ?? 1
  const m = v.shape[v.shape.length - 1] ?? 1
  return where(equal(tri(n, m, k - 1), 0), v, zeros([], { dtype: v.dtype }))
}

/**
 * The index grid of a shape, as numpy's `indices`: entry `[a, i0, ..., in-1]` is `ia`.
 *
 * @param shape - The grid's shape.
 * @returns An int32 tensor `[shape.length, ...shape]`.
 * @example
 * ```ts
 * import { indices, toArray } from 'mdml-engine/tensor'
 * toArray(indices([2, 3])) // => [[[0, 0, 0], [1, 1, 1]], [[0, 1, 2], [0, 1, 2]]]
 * ```
 */
export function indices(shape: Shape): Tensor<'int32'> {
  const size = sizeOf(shape)
  const data = new Int32Array(shape.length * size)
  for (const [a, n] of shape.entries()) {
    const inner = sizeOf(shape.slice(a + 1))
    for (let i = 0; i < size; i++) data[a * size + i] = Math.floor(i / inner) % n
  }
  return cpu.wrap('int32', [shape.length, ...shape], data)
}
