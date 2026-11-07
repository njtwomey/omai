/**
 * Indexing, gathering and sorting: Python-style slicing, `take` and `takeAlongAxis` with their scatter adjoints,
 * segment sums, coordinate-tuple `gather` and JAX-style `scatter` (set, add, multiply, max, min), stable sorting and
 * argsort, top-k, one-hot encoding, and the data-dependent operations (`nonzero`, masked selection, `unique`,
 * `bincount`, `searchsorted`).
 *
 * @index function Slicing, take, gather, scatter and sort, all differentiable.
 */

import { MdmlError, maxOf } from '../../base/index.ts'
import { batchAtFront, batchedAxis, moveBatchAxis } from '../batching.ts'
import { cpu } from '../cpu.ts'
import { allocate } from '../dtype.ts'
import { arg, bind1, definePrimitive, type Primitive } from '../primitive.ts'
import { formatShape, normaliseAxis, type Shape, sizeOf } from '../shape.ts'
import { isTensor, type Tensor, type TensorLike } from '../tensor.ts'
import { concreteOrThrow, primalOf, type Value } from '../traced.ts'
import { asDtype, coerce2, val } from './_coerce.ts'
import { add, cast, mul } from './arithmetic.ts'
import { equal, where } from './compare.ts'
import { broadcastTo, expandDims, moveaxis, reshape } from './shape.ts'

interface SliceParams {
  readonly starts: readonly number[]
  readonly stops: readonly number[]
  readonly steps: readonly number[]
}

interface SliceScatterParams {
  readonly shape: Shape
  readonly starts: readonly number[]
  readonly steps: readonly number[]
}

function sliceShape(p: SliceParams): number[] {
  return p.starts.map((start, i) => Math.max(0, Math.ceil(((p.stops[i] ?? start) - start) / (p.steps[i] ?? 1))))
}

const slicePrimitive: Primitive<SliceParams> = definePrimitive<SliceParams>({
  name: 'slice',
  linear: true,
  abstractEval: (xs, p) => [{ shape: sliceShape(p), dtype: arg(xs, 0).dtype }],
  impl: (xs, p) => [arg(xs, 0).backend.slice(arg(xs, 0), p.starts, p.stops, p.steps)],
  jvp: (_xs, ts, p) => {
    const t = arg(ts, 0)
    return [t === null ? null : bind1(slicePrimitive, [t], p)]
  },
  vjp: (gs, xs, p) => {
    const g = arg(gs, 0)
    return [
      g === null
        ? null
        : bind1(sliceScatterPrimitive, [g], { shape: arg(xs, 0).shape, starts: p.starts, steps: p.steps }),
    ]
  },
  batch: (xs, ds, p, n) => {
    const d = arg(ds, 0) ?? 0
    const at = <T>(list: readonly T[], v: T): T[] => [...list.slice(0, d), v, ...list.slice(d)]
    return {
      outputs: [
        bind1(slicePrimitive, [arg(xs, 0)], { starts: at(p.starts, 0), stops: at(p.stops, n), steps: at(p.steps, 1) }),
      ],
      dims: [d],
    }
  },
})

const sliceScatterPrimitive: Primitive<SliceScatterParams> = definePrimitive<SliceScatterParams>({
  name: 'sliceScatter',
  abstractEval: (xs, p) => [{ shape: p.shape, dtype: arg(xs, 0).dtype }],
  impl: (xs, p) => [arg(xs, 0).backend.sliceScatter(arg(xs, 0), p.shape, p.starts, p.steps)],
  jvp: (_xs, ts, p) => {
    const t = arg(ts, 0)
    return [t === null ? null : bind1(sliceScatterPrimitive, [t], p)]
  },
  vjp: (gs, xs, p) => {
    const g = arg(gs, 0)
    const shape = arg(xs, 0).shape
    const stops = p.starts.map((s, i) => s + (shape[i] ?? 0) * (p.steps[i] ?? 1))
    return [g === null ? null : bind1(slicePrimitive, [g], { starts: p.starts, stops, steps: p.steps })]
  },
  batch: (xs, ds, p, n) => {
    const d = arg(ds, 0) ?? 0
    const at = <T>(list: readonly T[], v: T): T[] => [...list.slice(0, d), v, ...list.slice(d)]
    return {
      outputs: [
        bind1(sliceScatterPrimitive, [arg(xs, 0)], {
          shape: at(p.shape, n),
          starts: at(p.starts, 0),
          steps: at(p.steps, 1),
        }),
      ],
      dims: [d],
    }
  },
})

const takePrimitive: Primitive<{ readonly axis: number }> = definePrimitive<{ readonly axis: number }>({
  name: 'take',
  linear: true,
  abstractEval: (xs, p) => [
    { shape: arg(xs, 0).shape.map((n, a) => (a === p.axis ? (arg(xs, 1).shape[0] ?? 0) : n)), dtype: arg(xs, 0).dtype },
  ],
  impl: (xs, p) => [arg(xs, 0).backend.take(arg(xs, 0), arg(xs, 1).data as Int32Array, p.axis)],
  jvp: (xs, ts, p) => {
    const t = arg(ts, 0)
    return [t === null ? null : bind1(takePrimitive, [t, arg(xs, 1)], p)]
  },
  vjp: (gs, xs, p) => {
    const g = arg(gs, 0)
    if (g === null) return [null, null]
    const size = arg(xs, 0).shape[p.axis] ?? 0
    return [bind1(takeScatterPrimitive, [g, arg(xs, 1)], { axis: p.axis, size }), null]
  },
  batch: (xs, ds, p, n) => {
    const x = arg(xs, 0)
    const dx = arg(ds, 0)
    const di = arg(ds, 1)
    if (di === null) {
      const d = dx ?? 0
      return { outputs: [bind1(takePrimitive, [x, arg(xs, 1)], { axis: batchedAxis(p.axis, d) })], dims: [d] }
    }
    const idx = moveBatchAxis(arg(xs, 1), di, 0, n)
    const k = idx.shape[1] ?? 0
    if (dx === null) {
      // One gather with every example's indices, then split the gathered axis into (example, index).
      const out = bind1(takePrimitive, [x, reshape(idx, [-1])], p)
      const shape = [...x.shape]
      shape.splice(p.axis, 1, n, k)
      return { outputs: [reshape(out, shape)], dims: [p.axis] }
    }
    // Both batched: gather from the examples stacked along the indexed axis, offsetting each example's indices.
    const lanes = moveaxis(moveaxis(x, dx, 0), p.axis + 1, 1)
    const m = lanes.shape[1] ?? 0
    const rest = lanes.shape.slice(2)
    const flat = bind1(
      takePrimitive,
      [reshape(lanes, [n * m, ...rest]), reshape(add(idx, exampleOffsets(n, m)), [-1])],
      {
        axis: 0,
      },
    )
    return { outputs: [moveaxis(reshape(flat, [n, k, ...rest]), 1, p.axis + 1)], dims: [0] }
  },
  describe: (p) => `axis=${p.axis}`,
})

/** `b * stride` for each example `b`, as an int32 column to offset per-example indices into stacked examples. */
function exampleOffsets(n: number, stride: number): Value {
  return cpu.wrap(
    'int32',
    [n, 1],
    Int32Array.from({ length: n }, (_, b) => b * stride),
  )
}

const takeScatterPrimitive: Primitive<{ readonly axis: number; readonly size: number }> = definePrimitive<{
  readonly axis: number
  readonly size: number
}>({
  name: 'takeScatter',
  linear: true,
  abstractEval: (xs, p) => [
    { shape: arg(xs, 0).shape.map((n, a) => (a === p.axis ? p.size : n)), dtype: arg(xs, 0).dtype },
  ],
  impl: (xs, p) => [arg(xs, 0).backend.takeScatter(arg(xs, 0), arg(xs, 1).data as Int32Array, p.axis, p.size)],
  jvp: (xs, ts, p) => {
    const t = arg(ts, 0)
    return [t === null ? null : bind1(takeScatterPrimitive, [t, arg(xs, 1)], p)]
  },
  vjp: (gs, xs, p) => {
    const g = arg(gs, 0)
    return [g === null ? null : bind1(takePrimitive, [g, arg(xs, 1)], { axis: p.axis }), null]
  },
  batch: (xs, ds, p, n) => {
    const dg = arg(ds, 0)
    const di = arg(ds, 1)
    if (di === null) {
      const d = dg ?? 0
      const params = { axis: batchedAxis(p.axis, d), size: p.size }
      return { outputs: [bind1(takeScatterPrimitive, [arg(xs, 0), arg(xs, 1)], params)], dims: [d] }
    }
    // Scatter every example into its own block of a stacked output, offsetting its indices.
    const idx = moveBatchAxis(arg(xs, 1), di, 0, n)
    const lanes = moveaxis(moveBatchAxis(arg(xs, 0), dg, 0, n), p.axis + 1, 1)
    const k = lanes.shape[1] ?? 0
    const rest = lanes.shape.slice(2)
    const flat = bind1(
      takeScatterPrimitive,
      [reshape(lanes, [n * k, ...rest]), reshape(add(idx, exampleOffsets(n, p.size)), [-1])],
      { axis: 0, size: n * p.size },
    )
    return { outputs: [moveaxis(reshape(flat, [n, p.size, ...rest]), 1, p.axis + 1)], dims: [0] }
  },
})

const takeAlongPrimitive: Primitive<{ readonly axis: number }> = definePrimitive<{ readonly axis: number }>({
  name: 'takeAlongAxis',
  linear: true,
  abstractEval: (xs) => [{ shape: arg(xs, 1).shape, dtype: arg(xs, 0).dtype }],
  impl: (xs, p) => [arg(xs, 0).backend.takeAlongAxis(arg(xs, 0), arg(xs, 1) as Tensor<'int32'>, p.axis)],
  jvp: (xs, ts, p) => {
    const t = arg(ts, 0)
    return [t === null ? null : bind1(takeAlongPrimitive, [t, arg(xs, 1)], p)]
  },
  vjp: (gs, xs, p) => {
    const g = arg(gs, 0)
    if (g === null) return [null, null]
    return [bind1(putAlongPrimitive, [g, arg(xs, 1)], { axis: p.axis, shape: arg(xs, 0).shape }), null]
  },
  batch: (xs, ds, p, n) => ({
    outputs: [bind1(takeAlongPrimitive, batchAtFront(xs, ds, n), { axis: p.axis + 1 })],
    dims: [0],
  }),
})

const putAlongPrimitive: Primitive<{ readonly axis: number; readonly shape: Shape }> = definePrimitive<{
  readonly axis: number
  readonly shape: Shape
}>({
  name: 'putAlongAxisAdd',
  linear: true,
  abstractEval: (xs, p) => [{ shape: p.shape, dtype: arg(xs, 0).dtype }],
  impl: (xs, p) => [arg(xs, 0).backend.putAlongAxisAdd(arg(xs, 0), arg(xs, 1) as Tensor<'int32'>, p.axis, p.shape)],
  jvp: (xs, ts, p) => {
    const t = arg(ts, 0)
    return [t === null ? null : bind1(putAlongPrimitive, [t, arg(xs, 1)], p)]
  },
  vjp: (gs, xs, p) => {
    const g = arg(gs, 0)
    return [g === null ? null : bind1(takeAlongPrimitive, [g, arg(xs, 1)], { axis: p.axis }), null]
  },
  batch: (xs, ds, p, n) => ({
    outputs: [bind1(putAlongPrimitive, batchAtFront(xs, ds, n), { axis: p.axis + 1, shape: [n, ...p.shape] })],
    dims: [0],
  }),
})

const argsortPrimitive: Primitive<{ readonly axis: number; readonly descending: boolean }> = definePrimitive<{
  readonly axis: number
  readonly descending: boolean
}>({
  name: 'argsort',
  abstractEval: (xs) => [{ shape: arg(xs, 0).shape, dtype: 'int32' }],
  impl: (xs, p) => [arg(xs, 0).backend.argsort(arg(xs, 0), p.axis, p.descending)],
  batch: (xs, ds, p) => {
    const d = arg(ds, 0) ?? 0
    const params = { axis: batchedAxis(p.axis, d), descending: p.descending }
    return { outputs: [bind1(argsortPrimitive, [arg(xs, 0)], params)], dims: [d] }
  },
  zeroDerivative: true,
})

/** Python slice semantics for one axis: negative indices count from the end; out-of-range bounds are clamped. */
function normaliseSlice(
  n: number,
  start: number | undefined,
  stop: number | undefined,
  step: number,
): [number, number] {
  if (step === 0) throw new MdmlError('value', 'slice: step must be non-zero')
  const clampIndex = (i: number, lo: number, hi: number): number => Math.min(Math.max(i < 0 ? i + n : i, lo), hi)
  if (step > 0)
    return [start === undefined ? 0 : clampIndex(start, 0, n), stop === undefined ? n : clampIndex(stop, 0, n)]
  return [
    start === undefined ? n - 1 : clampIndex(start, -1, n - 1),
    stop === undefined ? -1 : stop < -n ? -1 : clampIndex(stop, -1, n - 1),
  ]
}

/**
 * Negative indices counted from the end of a length-`n` axis, as numpy does. With `op` the indices are also range
 * checked (the scatter kernels are trusted and never look). A traced value passes through untouched.
 */
function wrapIndices(idx: Value, n: number, op?: string): Value {
  if (!isTensor(idx)) return idx
  const wrapped = Int32Array.from(idx.data as Int32Array, (i) => {
    const j = i < 0 ? i + n : i
    if (op !== undefined && (j < 0 || j >= n))
      throw new MdmlError('value', `${op}: index ${i} out of range for length ${n}`)
    return j
  })
  return cpu.wrap('int32', idx.shape, wrapped)
}

function indexTensor(indices: TensorLike): Value {
  if (Array.isArray(indices) && indices.every((i) => typeof i === 'number'))
    return cpu.wrap('int32', [indices.length], Int32Array.from(indices as readonly number[]))
  return asDtype(val(indices), 'int32')
}

/**
 * Slice with Python semantics per axis (missing entries mean the whole axis; negative indices count from the end;
 * negative steps reverse).
 *
 * @param x - Input.
 * @param starts - Start index per axis (`undefined` for the default).
 * @param stops - Exclusive stop per axis.
 * @param steps - Step per axis (default 1).
 * @returns The slice (always a copy).
 * @example
 * ```ts
 * import { slice, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(slice(tensor([0, 1, 2, 3, 4]), [1], [4])) // => [1, 2, 3]
 * toArray(slice(tensor([0, 1, 2, 3, 4]), [undefined], [undefined], [-2])) // => [4, 2, 0]
 * ```
 */
export function slice(
  x: TensorLike,
  starts: readonly (number | undefined)[],
  stops: readonly (number | undefined)[] = [],
  steps: readonly number[] = [],
): Value {
  const v = val(x)
  const s: number[] = []
  const e: number[] = []
  const k: number[] = []
  v.shape.forEach((n, a) => {
    const step = steps[a] ?? 1
    const [start, stop] = normaliseSlice(n, starts[a], stops[a], step)
    s.push(start)
    e.push(stop)
    k.push(step)
  })
  return bind1(slicePrimitive, [v], { starts: s, stops: e, steps: k })
}

/**
 * Select entries along an axis by integer index (like `numpy.take`); negative indices count from the end.
 *
 * @param x - Input.
 * @param indices - Indices into the axis.
 * @param axis - The axis (default 0).
 * @returns The gathered value.
 * @throws MdmlError with code `value` for an out-of-range index.
 * @example
 * ```ts
 * import { take, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(take(tensor([[1, 2], [3, 4], [5, 6]]), [2, 0])) // => [[5, 6], [1, 2]]
 * ```
 */
export function take(x: TensorLike, indices: TensorLike, axis = 0): Value {
  const v = val(x)
  const a = normaliseAxis(axis, v.shape.length)
  const n = v.shape[a] ?? 0
  const idx = indexTensor(indices)
  const fixed = wrapIndices(idx, n)
  if (fixed.shape.length !== 1) {
    const out = bind1(takePrimitive, [v, reshape(fixed, [-1])], { axis: a })
    return reshape(out, [...v.shape.slice(0, a), ...fixed.shape, ...v.shape.slice(a + 1)])
  }
  return bind1(takePrimitive, [v, fixed], { axis: a })
}

/**
 * Pick values along an axis with an index array of the same rank (numpy `take_along_axis`, torch `gather`).
 *
 * @param x - Input.
 * @param indices - int32 indices with the shape of `x` except along `axis`.
 * @param axis - The axis.
 * @returns A value with the shape of `indices`.
 * @example
 * ```ts
 * import { takeAlongAxis, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(takeAlongAxis(tensor([[10, 20], [30, 40]]), [[1], [0]], 1)) // => [[20], [30]]
 * ```
 */
export function takeAlongAxis(x: TensorLike, indices: TensorLike, axis: number): Value {
  const v = val(x)
  const a = normaliseAxis(axis, v.shape.length)
  const idx = asDtype(val(indices), 'int32')
  if (idx.shape.length !== v.shape.length || idx.shape.some((n, i) => i !== a && n !== v.shape[i])) {
    throw new MdmlError(
      'shape',
      `takeAlongAxis: indices ${formatShape(idx.shape)} do not match ${formatShape(v.shape)}`,
    )
  }
  return bind1(takeAlongPrimitive, [v, idx], { axis: a })
}

/**
 * Add `updates` into a zero array of `shape` at `indices` along an axis (the adjoint of {@link takeAlongAxis}; torch
 * `scatter_add`). Repeated indices accumulate; negative indices count from the end.
 *
 * @param shape - Shape of the result.
 * @param indices - int32 indices with the shape of `updates`.
 * @param updates - Values to add.
 * @param axis - The axis.
 * @returns The accumulated value.
 * @throws MdmlError with code `value` for an index outside `[-n, n)` along the axis.
 * @example
 * ```ts
 * import { scatterAdd, toArray } from 'mdml-engine/tensor'
 * toArray(scatterAdd([3], [0, 2, 0], [1, 2, 3], 0)) // => [4, 0, 2]
 * ```
 */
export function scatterAdd(shape: Shape, indices: TensorLike, updates: TensorLike, axis: number): Value {
  const u = val(updates)
  const a = normaliseAxis(axis, shape.length)
  const idx = wrapIndices(asDtype(val(indices), 'int32'), shape[a] ?? 0, 'scatterAdd')
  return bind1(putAlongPrimitive, [u, idx], { axis: a, shape: [...shape] })
}

/**
 * Sum rows of `data` into segments given by an id per row (TensorFlow `unsorted_segment_sum`). Negative ids count
 * from the end, as in {@link take}.
 *
 * @param data - Values with segments along axis 0.
 * @param segmentIds - The segment of each row (int32, length of axis 0).
 * @param numSegments - The number of segments.
 * @returns A value with `numSegments` rows.
 * @throws MdmlError with code `value` for an id outside `[-numSegments, numSegments)`.
 * @example
 * ```ts
 * import { segmentSum, toArray } from 'mdml-engine/tensor'
 * toArray(segmentSum([1, 2, 3], [0, 1, 0], 2)) // => [4, 2]
 * ```
 */
export function segmentSum(data: TensorLike, segmentIds: TensorLike, numSegments: number): Value {
  const ids = wrapIndices(indexTensor(segmentIds), numSegments, 'segmentSum')
  return bind1(takeScatterPrimitive, [val(data), ids], { axis: 0, size: numSegments })
}

/**
 * Indices that sort an axis (stable; NaN last). Its derivative is zero.
 *
 * @param x - Input.
 * @param axis - The axis (default -1).
 * @param descending - Sort largest first.
 * @returns int32 indices.
 * @example
 * ```ts
 * import { argsort, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(argsort(tensor([3, 1, 2]))) // => [1, 2, 0]
 * ```
 */
export function argsort(x: TensorLike, axis = -1, descending = false): Value {
  const v = val(x)
  return bind1(argsortPrimitive, [v], { axis: normaliseAxis(axis, v.shape.length), descending })
}

/**
 * Sort along an axis (stable; NaN last). Differentiable: the gradient flows back to each element's original position.
 *
 * @param x - Input.
 * @param axis - The axis (default -1).
 * @param descending - Sort largest first.
 * @returns The sorted values.
 * @example
 * ```ts
 * import { sort, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(sort(tensor([3, 1, 2]))) // => [1, 2, 3]
 * ```
 */
export function sort(x: TensorLike, axis = -1, descending = false): Value {
  const v = val(x)
  return takeAlongAxis(v, argsort(v, axis, descending), axis)
}

/**
 * The `k` largest (or smallest) values along an axis and their indices.
 *
 * @param x - Input.
 * @param k - How many.
 * @param axis - The axis (default -1).
 * @param largest - Largest first (default) or smallest first.
 * @returns `{ values, indices }`.
 * @example
 * ```ts
 * import { topk, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(topk(tensor([3, 1, 4, 1, 5]), 2).values) // => [5, 4]
 * ```
 */
export function topk(
  x: TensorLike,
  k: number,
  axis = -1,
  largest = true,
): { readonly values: Value; readonly indices: Value } {
  const v = val(x)
  const a = normaliseAxis(axis, v.shape.length)
  const order = argsort(v, a, largest)
  const indices = slice(
    order,
    v.shape.map(() => undefined),
    v.shape.map((n, i) => (i === a ? k : n)),
  )
  return { values: takeAlongAxis(v, indices, a), indices }
}

/**
 * One-hot encoding of integer labels along a new last axis.
 *
 * @param labels - Integer labels in `[0, n)`.
 * @param n - Number of classes.
 * @returns A float64 value with a trailing axis of length `n`.
 * @example
 * ```ts
 * import { oneHot, toArray } from 'mdml-engine/tensor'
 * toArray(oneHot([0, 2], 3)) // => [[1, 0, 0], [0, 0, 1]]
 * ```
 */
export function oneHot(labels: TensorLike, n: number): Value {
  const l = indexTensor(labels)
  const classes = cpu.wrap(
    'int32',
    [n],
    Int32Array.from({ length: n }, (_, i) => i),
  )
  return cast(equal(expandDims(l, -1), broadcastTo(classes, [...l.shape, n])), 'float64')
}

/**
 * Indices where a concrete value is non-zero, one int32 array per axis (numpy `nonzero`).
 *
 * @param x - A concrete input.
 * @returns One index tensor per axis.
 * @example
 * ```ts
 * import { nonzero, tensor, toArray } from 'mdml-engine/tensor'
 * nonzero(tensor([0, 3, 0, 5])).map(toArray) // => [[1, 3]]
 * ```
 */
export function nonzero(x: TensorLike): Tensor<'int32'>[] {
  const t = concreteOrThrow(val(x), 'nonzero')
  const rank = t.shape.length
  const lists: number[][] = t.shape.map(() => [])
  const strides = t.shape.map((_, a) => t.shape.slice(a + 1).reduce((p, n) => p * n, 1))
  t.data.forEach((v, i) => {
    if (v === 0) return
    for (let a = 0; a < rank; a++) lists[a]?.push(Math.floor(i / (strides[a] ?? 1)) % (t.shape[a] ?? 1))
  })
  return lists.map((l) => cpu.wrap('int32', [l.length], Int32Array.from(l)))
}

/**
 * Elements of a concrete value where a mask is true, as a 1-D tensor.
 *
 * @param x - A concrete input.
 * @param mask - A mask of the same shape.
 * @returns The selected elements.
 * @example
 * ```ts
 * import { maskedSelect, tensor, greater, toArray } from 'mdml-engine/tensor'
 * const x = tensor([1, -2, 3])
 * toArray(maskedSelect(x, greater(x, 0))) // => [1, 3]
 * ```
 */
export function maskedSelect(x: TensorLike, mask: TensorLike): Value {
  const v = val(x)
  const m = concreteOrThrow(broadcastTo(asDtype(val(mask), 'bool'), v.shape), 'maskedSelect')
  const indices: number[] = []
  m.data.forEach((b, i) => {
    if (b) indices.push(i)
  })
  return take(reshape(v, [-1]), indices)
}

/**
 * Insertion points that keep a sorted 1-D array sorted (numpy `searchsorted`).
 *
 * @param sorted - A concrete sorted 1-D array.
 * @param values - Values to place.
 * @param side - `left` (first suitable position) or `right` (last).
 * @returns int32 positions with the shape of `values`.
 * @example
 * ```ts
 * import { searchsorted, toArray } from 'mdml-engine/tensor'
 * toArray(searchsorted([1, 2, 3], [2, 2.5])) // => [1, 2]
 * ```
 */
export function searchsorted(sorted: TensorLike, values: TensorLike, side: 'left' | 'right' = 'left'): Tensor<'int32'> {
  const s = concreteOrThrow(val(sorted), 'searchsorted')
  const v = concreteOrThrow(val(values), 'searchsorted')
  const out = new Int32Array(v.data.length)
  v.data.forEach((x, i) => {
    let lo = 0
    let hi = s.data.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      const m = s.data[mid] ?? 0
      if (side === 'left' ? m < x : m <= x) lo = mid + 1
      else hi = mid
    }
    out[i] = lo
  })
  return cpu.wrap('int32', v.shape, out)
}

/**
 * Sorted unique values of a concrete array, with counts (NaNs count as one value and sort last, as in numpy).
 *
 * @param x - A concrete input.
 * @returns `{ values, counts }` as 1-D tensors.
 * @example
 * ```ts
 * import { unique, toArray } from 'mdml-engine/tensor'
 * toArray(unique([3, 1, 3, 2]).values) // => [1, 2, 3]
 * ```
 */
export function unique(x: TensorLike): { readonly values: Tensor; readonly counts: Tensor<'int32'> } {
  const t = concreteOrThrow(val(x), 'unique')
  const counts = new Map<number, number>()
  for (const v of t.data) counts.set(v, (counts.get(v) ?? 0) + 1)
  // NaN sorts last (as numpy), which the plain difference cannot say: it gives NaN against everything
  const keys = [...counts.keys()].sort((a, b) =>
    Number.isNaN(a) ? (Number.isNaN(b) ? 0 : 1) : Number.isNaN(b) ? -1 : a - b,
  )
  const values = allocate(t.dtype, keys.length)
  keys.forEach((k, i) => {
    values[i] = k
  })
  return {
    values: t.backend.wrap(t.dtype, [keys.length], values),
    counts: cpu.wrap(
      'int32',
      [keys.length],
      Int32Array.from(keys, (k) => counts.get(k) ?? 0),
    ),
  }
}

/**
 * Count occurrences of each non-negative integer (numpy `bincount`), optionally weighted.
 *
 * @param x - Non-negative integers.
 * @param minLength - Minimum length of the result.
 * @param weights - Optional weights per element.
 * @returns Counts (int32) or weighted sums (float64).
 * @example
 * ```ts
 * import { bincount, toArray } from 'mdml-engine/tensor'
 * toArray(bincount([0, 1, 1, 3])) // => [1, 2, 0, 1]
 * ```
 */
export function bincount(x: TensorLike, minLength = 0, weights?: TensorLike): Value {
  const t = concreteOrThrow(indexTensor(x), 'bincount')
  let size = minLength
  for (const v of t.data) {
    if (v < 0) throw new MdmlError('value', 'bincount: values must be non-negative')
    size = Math.max(size, v + 1)
  }
  if (weights === undefined) {
    const out = new Int32Array(size)
    for (const v of t.data) out[v] = (out[v] ?? 0) + 1
    return cpu.wrap('int32', [size], out)
  }
  return segmentSum(cast(weights, 'float64'), t, size)
}

/** Row-major linear indices into the leading axes of `shape` from an index array `[..., k]`. */
function leadingIndices(
  shape: Shape,
  indices: TensorLike,
  where: string,
): { readonly flat: Int32Array; readonly batch: Shape; readonly lead: number; readonly rest: Shape } {
  const idx = concreteOrThrow(asDtype(val(indices), 'int32'), where) as Tensor<'int32'>
  if (idx.shape.length === 0) throw new MdmlError('shape', `${where}: indices need a last axis of coordinates`)
  const k = idx.shape[idx.shape.length - 1] ?? 0
  if (k > shape.length)
    throw new MdmlError('shape', `${where}: ${k} coordinates index an array of rank ${shape.length}`)
  const batch = idx.shape.slice(0, -1)
  const m = sizeOf(batch)
  const flat = new Int32Array(m)
  let lead = 1
  for (let a = 0; a < k; a++) lead *= shape[a] ?? 0
  for (let r = 0; r < m; r++) {
    let linear = 0
    for (let a = 0; a < k; a++) {
      const n = shape[a] ?? 0
      const raw = idx.data[r * k + a] ?? 0
      const i = raw < 0 ? raw + n : raw
      if (i < 0 || i >= n) throw new MdmlError('value', `${where}: index ${raw} out of range for axis ${a} of ${n}`)
      linear = linear * n + i
    }
    flat[r] = linear
  }
  return { flat, batch, lead, rest: shape.slice(k) }
}

/**
 * Gather with coordinate tuples (TensorFlow `gather_nd`, numpy `x[tuple(indices.T)]`): the last axis of `indices`
 * holds k coordinates into the first k axes of x, and each tuple selects the slice `x[i0, ..., ik-1]`. Negative
 * coordinates count from the end. Differentiable in x (the gradient scatters back, accumulating repeats).
 *
 * @param x - Input.
 * @param indices - int32 coordinates `[..., k]` with k at most the rank of x.
 * @returns A value of shape `indices.shape[:-1] + x.shape[k:]`.
 * @throws MdmlError with code `value` for an out-of-range coordinate.
 * @example
 * ```ts
 * import { gather, tensor, toArray } from 'mdml-engine/tensor'
 * const x = tensor([[1, 2], [3, 4]])
 * toArray(gather(x, [[1, 0], [0, 1]])) // => [3, 2]
 * toArray(gather(x, [[1]])) // => [[3, 4]]
 * ```
 */
export function gather(x: TensorLike, indices: TensorLike): Value {
  const v = val(x)
  const { flat, batch, lead, rest } = leadingIndices(v.shape, indices, 'gather')
  const rows = bind1(takePrimitive, [reshape(v, [lead, ...rest]), cpu.wrap('int32', [flat.length], flat)], { axis: 0 })
  return reshape(rows, [...batch, ...rest])
}

/** How {@link scatter} combines updates with the existing values. */
export type ScatterMode = 'set' | 'add' | 'mul' | 'max' | 'min'

/**
 * Write `updates` into a copy of x at coordinate tuples (JAX `x.at[...].set/add/multiply/max/min`, numpy `x[idx] = u`
 * and `np.add.at` and friends): the last axis of `indices` holds k coordinates into the first k axes of x, and
 * `updates` has shape `indices.shape[:-1] + x.shape[k:]` (or broadcasts to it). Repeated coordinates accumulate for
 * `add`, `mul`, `max` and `min`; for `set` the last write wins. `max` and `min` propagate NaN, as numpy's `maximum.at`.
 * Differentiable in x and updates: every mode is built from differentiable gathers, sums and selections, so the
 * gradient goes to whichever value survives (for `max`/`min`, the winner; ties go to x, then to the earlier update).
 *
 * @param x - The array written into (not mutated).
 * @param indices - int32 coordinates `[..., k]`.
 * @param updates - The values to write.
 * @param mode - How to combine (default `'set'`).
 * @returns The updated copy.
 * @throws MdmlError with code `shape` when updates do not broadcast to the indexed shape.
 * @example
 * ```ts
 * import { scatter, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(scatter(tensor([0, 0, 0]), [[0], [2], [0]], [1, 2, 3], 'add')) // => [4, 0, 2]
 * toArray(scatter(tensor([0, 0, 0]), [[0], [2], [0]], [1, 2, 3])) // => [3, 0, 2]
 * toArray(scatter(tensor([[5, 5], [5, 5]]), [[1]], [[1, 9]], 'max')) // => [[5, 5], [5, 9]]
 * ```
 */
export function scatter(x: TensorLike, indices: TensorLike, updates: TensorLike, mode: ScatterMode = 'set'): Value {
  const [v, u0] = coerce2(x, updates)
  const { flat, batch, lead, rest } = leadingIndices(v.shape, indices, 'scatter')
  const m = flat.length
  const xf = reshape(v, [lead, ...rest])
  if (m === 0) return v
  let u: Value
  try {
    u = reshape(broadcastTo(u0, [...batch, ...rest]), [m, ...rest])
  } catch {
    throw new MdmlError(
      'shape',
      `scatter: updates ${formatShape(u0.shape)} do not broadcast to ${formatShape([...batch, ...rest])}`,
    )
  }
  const ones = rest.map(() => 1)
  const rowMask = (positions: Int32Array): Value => {
    const mask = new Uint8Array(lead)
    for (const p of positions) mask[p] = 1
    return cpu.wrap('bool', [lead, ...ones], mask)
  }
  // Place the update rows `rows` at the distinct positions `positions` of a zero array.
  const place = (rows: Int32Array, positions: Int32Array): Value =>
    bind1(
      takeScatterPrimitive,
      [
        bind1(takePrimitive, [u, cpu.wrap('int32', [rows.length], rows)], { axis: 0 }),
        cpu.wrap('int32', [positions.length], positions),
      ],
      { axis: 0, size: lead },
    )
  let out: Value
  if (mode === 'add') {
    out = add(xf, bind1(takeScatterPrimitive, [u, cpu.wrap('int32', [m], flat)], { axis: 0, size: lead }))
  } else if (mode === 'set' || mode === 'mul') {
    // Group repeated positions into rounds: round r holds the r-th update at each position, so positions are distinct.
    const seen = new Map<number, number>()
    const round = new Int32Array(m)
    for (let j = 0; j < m; j++) {
      const p = flat[j] ?? 0
      const r = seen.get(p) ?? 0
      round[j] = r
      seen.set(p, r + 1)
    }
    if (mode === 'set') {
      const last = new Map<number, number>()
      for (let j = 0; j < m; j++) last.set(flat[j] ?? 0, j)
      const positions = Int32Array.from(last.keys())
      const rows = Int32Array.from(last.values())
      out = where(rowMask(positions), place(rows, positions), xf)
    } else {
      out = xf
      for (let r = 0; r < maxOf(Array.from(seen.values())); r++) {
        const rows = Int32Array.from(round.keys()).filter((j) => round[j] === r)
        const positions = rows.map((j) => flat[j] ?? 0)
        out = mul(out, where(rowMask(positions), place(rows, positions), 1))
      }
    }
  } else {
    // max / min: choose the winner of every element from the primal values, then select it.
    const width = sizeOf(rest)
    const xp = primalOf(xf).data
    const up = primalOf(u).data
    const better =
      mode === 'max'
        ? (c: number, b: number): boolean => c > b || (Number.isNaN(c) && !Number.isNaN(b))
        : (c: number, b: number): boolean => c < b || (Number.isNaN(c) && !Number.isNaN(b))
    const fromUpdate = new Uint8Array(lead * width)
    const winner = new Int32Array(lead * width)
    const best = Float64Array.from(xp)
    for (let j = 0; j < m; j++) {
      const p = flat[j] ?? 0
      for (let r = 0; r < width; r++) {
        const c = up[j * width + r] ?? 0
        if (better(c, best[p * width + r] ?? 0)) {
          best[p * width + r] = c
          fromUpdate[p * width + r] = 1
          winner[p * width + r] = j
        }
      }
    }
    const picked = bind1(takeAlongPrimitive, [u, cpu.wrap('int32', [lead, ...rest], winner)], { axis: 0 })
    out = where(cpu.wrap('bool', [lead, ...rest], fromUpdate), picked, xf)
  }
  return reshape(out, v.shape)
}
