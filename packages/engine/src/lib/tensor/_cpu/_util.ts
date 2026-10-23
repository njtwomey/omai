import { WhiteboxError } from '../../base/index.ts'
import { cpu } from '../cpu.ts'
import type { DType, Storage } from '../dtype.ts'
import { formatShape, type Shape, shapesEqual, sizeOf, stridesOf } from '../shape.ts'
import type { Tensor } from '../tensor.ts'

/** Wrap storage as a CPU tensor, checking the size. */
export function wrap<D extends DType>(dtype: D, shape: Shape, data: Storage<D>): Tensor<D> {
  if (data.length !== sizeOf(shape)) {
    throw new WhiteboxError('shape', `data of length ${data.length} does not fill shape ${formatShape(shape)}`, {
      shape,
      length: data.length,
    })
  }
  return { kind: 'tensor', dtype, shape: [...shape], data, backend: cpu }
}

/** For each flat index of `outShape`, the flat index of the broadcast source element of `inShape`. */
export function broadcastIndex(outShape: Shape, inShape: Shape): Int32Array {
  const size = sizeOf(outShape)
  const map = new Int32Array(size)
  if (shapesEqual(outShape, inShape)) {
    for (let i = 0; i < size; i++) map[i] = i
    return map
  }
  if (sizeOf(inShape) === 1) return map
  const rank = outShape.length
  const inStrides = stridesOf(inShape)
  const strides = new Array<number>(rank).fill(0)
  for (let axis = 0; axis < inShape.length; axis++) {
    const outAxis = rank - inShape.length + axis
    strides[outAxis] = (inShape[axis] ?? 1) === 1 ? 0 : (inStrides[axis] ?? 0)
  }
  walkStrided(outShape, strides, (i, offset) => {
    map[i] = offset
  })
  return map
}

/**
 * Visit every multi-index of `shape` in row-major order, tracking an offset that advances by `strides` per axis.
 * `visit(i, offset)` receives the flat output index and the strided offset.
 */
export function walkStrided(
  shape: Shape,
  strides: readonly number[],
  visit: (i: number, offset: number) => void,
  base = 0,
): void {
  const rank = shape.length
  const size = sizeOf(shape)
  const counter = new Array<number>(rank).fill(0)
  let offset = base
  for (let i = 0; i < size; i++) {
    visit(i, offset)
    for (let axis = rank - 1; axis >= 0; axis--) {
      counter[axis] = (counter[axis] ?? 0) + 1
      offset += strides[axis] ?? 0
      if ((counter[axis] ?? 0) < (shape[axis] ?? 1)) break
      offset -= (strides[axis] ?? 0) * (counter[axis] ?? 0)
      counter[axis] = 0
    }
  }
}

/** Split a shape around an axis into outer, axis and inner extents (flat index = (o * n + i) * inner + k). */
export function lanes(
  shape: Shape,
  axis: number,
): { readonly outer: number; readonly n: number; readonly inner: number } {
  return { outer: sizeOf(shape.slice(0, axis)), n: shape[axis] ?? 1, inner: sizeOf(shape.slice(axis + 1)) }
}

export function assertSameDtype(a: Tensor, b: Tensor, op: string): void {
  if (a.dtype !== b.dtype) {
    throw new WhiteboxError('dtype', `${op}: dtypes ${a.dtype} and ${b.dtype} differ`, { a: a.dtype, b: b.dtype })
  }
}
