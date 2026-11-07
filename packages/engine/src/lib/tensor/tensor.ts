/**
 * The `Tensor` type, an n-dimensional array with a dtype, a shape and contiguous row-major storage owned by a backend,
 * and the ways in and out of it: `tensor` (like `numpy.array`), wrapping existing storage, conversion of tensor-like
 * inputs, and `toArray`, `toFlat`, `toVector`, `item` and `clone`.
 *
 * @index structure The Tensor type, its constructors and conversions to plain arrays.
 */

import { MdmlError } from '../base/index.ts'
import type { Backend } from './backend.ts'
import { cpu } from './cpu.ts'
import { allocate, type DType, type Storage } from './dtype.ts'
import { formatShape, type Shape, sizeOf } from './shape.ts'
import { isTraced, type Value } from './traced.ts'

/**
 * An n-dimensional array: a dtype, a shape and contiguous row-major storage, plus the backend that owns it.
 *
 * Tensors are immutable by contract: no operation writes into an existing tensor's `data`. Outside a transformation a
 * tensor is just an array, with no gradient machinery attached.
 *
 * @concept tensor
 */
export interface Tensor<D extends DType = DType, S extends Shape = Shape> {
  readonly kind: 'tensor'
  readonly dtype: D
  readonly shape: S
  readonly data: Storage<D>
  readonly backend: Backend
}

/** Nested arrays of numbers (or booleans), as accepted by {@link tensor} and returned by {@link toArray}. */
export type NestedArray = number | boolean | readonly NestedArray[]

/** Anything that can be turned into a tensor: a value, a typed array, or nested arrays of numbers. */
export type TensorLike = Value | NestedArray | Float64Array | Float32Array | Int32Array | Uint8Array

/** Options shared by tensor constructors. */
export interface CreateOptions<D extends DType> {
  readonly dtype?: D
  readonly backend?: Backend
}

function inferShape(values: NestedArray): number[] {
  const shape: number[] = []
  let level: NestedArray = values
  while (Array.isArray(level)) {
    shape.push(level.length)
    level = (level as readonly NestedArray[])[0] ?? 0
  }
  return shape
}

function flatten(values: NestedArray, shape: Shape, depth: number, out: number[]): void {
  if (depth === shape.length) {
    if (typeof values === 'number') out.push(values)
    else if (typeof values === 'boolean') out.push(values ? 1 : 0)
    else throw new MdmlError('shape', 'ragged nested array: expected a number')
    return
  }
  if (!Array.isArray(values) || values.length !== shape[depth]) {
    throw new MdmlError('shape', `ragged nested array at depth ${depth}: expected length ${shape[depth]}`)
  }
  for (const v of values as readonly NestedArray[]) flatten(v, shape, depth + 1, out)
}

function inferDType(values: NestedArray): DType {
  let level: NestedArray = values
  while (Array.isArray(level)) level = (level as readonly NestedArray[])[0] ?? 0
  return typeof level === 'boolean' ? 'bool' : 'float64'
}

/**
 * Create a tensor from a number or nested arrays (like `numpy.array`). Numbers give float64, booleans give bool; a
 * `bool` dtype stores 0 or 1 whatever the numbers (nonzero and NaN are true).
 *
 * @param values - A scalar or rectangular nested array.
 * @param options - The dtype (inferred by default) and backend (default the CPU backend).
 * @returns A new tensor whose shape is inferred from the nesting.
 * @throws MdmlError with code `shape` for ragged input.
 * @concept tensor
 * @index function Create a tensor from numbers or nested arrays, like numpy.array.
 * @example
 * ```ts
 * import { tensor } from 'mdml-engine/tensor'
 * const t = tensor([[1, 2, 3], [4, 5, 6]])
 * t.shape // => [2, 3]
 * t.dtype // => 'float64'
 * ```
 */
export function tensor<D extends DType = 'float64'>(values: NestedArray, options: CreateOptions<D> = {}): Tensor<D> {
  const dtype = (options.dtype ?? inferDType(values)) as D
  const shape = inferShape(values)
  const flat: number[] = []
  flatten(values, shape, 0, flat)
  const data = allocate(dtype, flat.length)
  for (let i = 0; i < flat.length; i++) {
    const v = flat[i] ?? 0
    // Bool storage is 0 or 1 (nonzero and NaN are true, as numpy and `cast`), never a raw byte.
    data[i] = dtype === 'int32' ? Math.trunc(v) : dtype === 'bool' ? (v !== 0 ? 1 : 0) : v
  }
  return (options.backend ?? cpu).wrap(dtype, shape, data)
}

/**
 * Wrap existing storage as a tensor without copying. The caller hands over `data` and must not mutate it afterwards,
 * and is trusted: bool storage must already hold only 0 and 1.
 *
 * @param data - Row-major storage of the dtype's class.
 * @param shape - The shape; its size must equal `data.length`.
 * @param dtype - The element type.
 * @param backend - The owning backend (default CPU).
 * @returns The tensor.
 * @throws MdmlError with code `shape` when sizes disagree.
 * @example
 * ```ts
 * import { fromStorage } from 'mdml-engine/tensor'
 * fromStorage(new Float64Array([1, 2, 3, 4]), [2, 2], 'float64').shape // => [2, 2]
 * ```
 */
export function fromStorage<D extends DType, S extends Shape>(
  data: Storage<D>,
  shape: S,
  dtype: D,
  backend: Backend = cpu,
): Tensor<D, S> {
  return backend.wrap(dtype, shape, data) as Tensor<D, S>
}

/**
 * Whether a value is a concrete tensor.
 *
 * @param value - Any value.
 * @returns `true` for tensors.
 * @example
 * ```ts
 * import { isTensor, tensor } from 'mdml-engine/tensor'
 * isTensor(tensor([1])) // => true
 * isTensor([1]) // => false
 * ```
 */
export function isTensor(value: unknown): value is Tensor {
  return typeof value === 'object' && value !== null && (value as { kind?: unknown }).kind === 'tensor'
}

/**
 * Whether a value is a tensor or a traced value.
 *
 * @param value - Any value.
 * @returns `true` for values operations accept without conversion.
 * @example
 * ```ts
 * import { isValue, tensor } from 'mdml-engine/tensor'
 * isValue(tensor(1)) // => true
 * isValue(1) // => false
 * ```
 */
export function isValue(value: unknown): value is Value {
  return isTensor(value) || isTraced(value)
}

/**
 * Convert anything tensor-like to a value: values pass through; typed arrays and nested arrays become tensors.
 *
 * @param x - A value, typed array, number, boolean or nested array.
 * @param dtype - The dtype for new tensors (inferred by default; typed arrays keep their dtype).
 * @returns A value.
 * @example
 * ```ts
 * import { asTensor } from 'mdml-engine/tensor'
 * asTensor([1, 2]).shape // => [2]
 * asTensor(new Int32Array([1, 2])).dtype // => 'int32'
 * ```
 */
export function asTensor(x: TensorLike, dtype?: DType): Value {
  if (isValue(x)) return x
  if (x instanceof Float64Array) return cpu.wrap('float64', [x.length], x.slice())
  if (x instanceof Float32Array) return cpu.wrap('float32', [x.length], x.slice())
  if (x instanceof Int32Array) return cpu.wrap('int32', [x.length], x.slice())
  if (x instanceof Uint8Array)
    return cpu.wrap(
      'bool',
      [x.length],
      x.map((v) => (v === 0 ? 0 : 1)),
    )
  return dtype === undefined ? tensor(x as NestedArray) : tensor(x as NestedArray, { dtype })
}

/**
 * Convert a tensor to nested arrays of numbers (like `tolist()`); booleans become 0 and 1.
 *
 * @param t - A concrete tensor.
 * @returns A number for a scalar, otherwise nested arrays matching the shape.
 * @example
 * ```ts
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * toArray(tensor([[1, 2], [3, 4]])) // => [[1, 2], [3, 4]]
 * ```
 */
export function toArray(t: Value): NestedArray {
  const x = concrete(t, 'toArray')
  const build = (depth: number, offset: number): NestedArray => {
    if (depth === x.shape.length) return x.data[offset] ?? 0
    const n = x.shape[depth] ?? 0
    const inner = sizeOf(x.shape.slice(depth + 1))
    const out: NestedArray[] = []
    for (let i = 0; i < n; i++) out.push(build(depth + 1, offset + i * inner))
    return out
  }
  return build(0, 0)
}

/**
 * The elements of a tensor as a flat array of numbers, in row-major order.
 *
 * @param t - A concrete tensor.
 * @returns The elements.
 * @example
 * ```ts
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(tensor([[1, 2], [3, 4]])) // => [1, 2, 3, 4]
 * ```
 */
export function toFlat(t: Value): number[] {
  return Array.from(concrete(t, 'toFlat').data)
}

/**
 * The numbers of a vector given as a tensor-like, for a function whose own loop reads plain numbers (a sampler's weights,
 * a list of fractions): a rank-1 tensor, a typed array or a flat array of numbers. A flat array of numbers is returned
 * as it is, without a copy, so the caller must not write into the result.
 *
 * @param x - The vector: a tensor, a typed array or a nested array of depth one.
 * @param name - The caller's name, for error messages.
 * @returns The elements.
 * @throws MdmlError with code `shape` unless `x` has rank 1, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { tensor, toVector } from 'mdml-engine/tensor'
 * toVector(tensor([1, 2, 3]), 'example') // => [1, 2, 3]
 * toVector(new Float64Array([4, 5]), 'example') // => [4, 5]
 * ```
 */
export function toVector(x: TensorLike, name: string): readonly number[] {
  if (Array.isArray(x) && x.every((v) => typeof v === 'number')) return x as readonly number[]
  if (ArrayBuffer.isView(x) && !(x instanceof DataView)) return Array.from(x as ArrayLike<number>)
  const t = concrete(asTensor(x), name)
  if (t.shape.length !== 1)
    throw new MdmlError('shape', `${name}: expected a vector, got shape ${formatShape(t.shape)}`)
  return Array.from(t.data)
}

/**
 * The single value of a one-element tensor (like `item()`).
 *
 * @param t - A concrete tensor with exactly one element.
 * @returns The value.
 * @throws MdmlError with code `shape` when the tensor has more than one element.
 * @example
 * ```ts
 * import { item, tensor } from 'mdml-engine/tensor'
 * item(tensor([[3.5]])) // => 3.5
 * ```
 */
export function item(t: Value): number {
  const x = concrete(t, 'item')
  if (x.data.length !== 1) {
    throw new MdmlError('shape', `item: tensor of shape ${formatShape(x.shape)} has ${x.data.length} elements`)
  }
  return x.data[0] ?? 0
}

/**
 * A copy of a tensor with fresh storage.
 *
 * @param t - A concrete tensor.
 * @returns The copy.
 * @example
 * ```ts
 * import { clone, tensor } from 'mdml-engine/tensor'
 * const a = tensor([1, 2])
 * clone(a).data === a.data // => false
 * ```
 */
export function clone<D extends DType>(t: Tensor<D>): Tensor<D> {
  return t.backend.wrap(t.dtype, t.shape, t.data.slice() as Storage<D>)
}

function concrete(t: Value, where: string): Tensor {
  if (isTraced(t)) throw new MdmlError('not-differentiable', `${where} needs a concrete tensor, not a traced value`)
  return t
}
