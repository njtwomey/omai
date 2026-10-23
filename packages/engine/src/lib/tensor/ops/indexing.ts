import { WhiteboxError } from '../../base/index.ts'
import { cpu } from '../cpu.ts'
import { arg, bind1, definePrimitive, type Primitive } from '../primitive.ts'
import { formatShape, normaliseAxis, type Shape } from '../shape.ts'
import { isTensor, type Tensor, type TensorLike } from '../tensor.ts'
import { concreteOrThrow, type Value } from '../traced.ts'
import { asDtype, val } from './_coerce.ts'
import { cast } from './arithmetic.ts'
import { equal } from './compare.ts'
import { broadcastTo, expandDims, reshape } from './shape.ts'

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
})

const takePrimitive: Primitive<{ readonly axis: number }> = definePrimitive<{ readonly axis: number }>({
  name: 'take',
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
  describe: (p) => `axis=${p.axis}`,
})

const takeScatterPrimitive: Primitive<{ readonly axis: number; readonly size: number }> = definePrimitive<{
  readonly axis: number
  readonly size: number
}>({
  name: 'takeScatter',
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
})

const takeAlongPrimitive: Primitive<{ readonly axis: number }> = definePrimitive<{ readonly axis: number }>({
  name: 'takeAlongAxis',
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
})

const putAlongPrimitive: Primitive<{ readonly axis: number; readonly shape: Shape }> = definePrimitive<{
  readonly axis: number
  readonly shape: Shape
}>({
  name: 'putAlongAxisAdd',
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
})

const argsortPrimitive: Primitive<{ readonly axis: number; readonly descending: boolean }> = definePrimitive<{
  readonly axis: number
  readonly descending: boolean
}>({
  name: 'argsort',
  abstractEval: (xs) => [{ shape: arg(xs, 0).shape, dtype: 'int32' }],
  impl: (xs, p) => [arg(xs, 0).backend.argsort(arg(xs, 0), p.axis, p.descending)],
  zeroDerivative: true,
})

/** Python slice semantics for one axis: negative indices count from the end; out-of-range bounds are clamped. */
function normaliseSlice(
  n: number,
  start: number | undefined,
  stop: number | undefined,
  step: number,
): [number, number] {
  if (step === 0) throw new WhiteboxError('value', 'slice: step must be non-zero')
  const clampIndex = (i: number, lo: number, hi: number): number => Math.min(Math.max(i < 0 ? i + n : i, lo), hi)
  if (step > 0)
    return [start === undefined ? 0 : clampIndex(start, 0, n), stop === undefined ? n : clampIndex(stop, 0, n)]
  return [
    start === undefined ? n - 1 : clampIndex(start, -1, n - 1),
    stop === undefined ? -1 : stop < -n ? -1 : clampIndex(stop, -1, n - 1),
  ]
}

function indexTensor(indices: TensorLike): Value {
  if (Array.isArray(indices)) return cpu.wrap('int32', [indices.length], Int32Array.from(indices as readonly number[]))
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
 * import { slice, tensor, toArray } from 'whitebox-engine/tensor'
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
 * @throws WhiteboxError with code `value` for an out-of-range index.
 * @example
 * ```ts
 * import { take, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(take(tensor([[1, 2], [3, 4], [5, 6]]), [2, 0])) // => [[5, 6], [1, 2]]
 * ```
 */
export function take(x: TensorLike, indices: TensorLike, axis = 0): Value {
  const v = val(x)
  const a = normaliseAxis(axis, v.shape.length)
  const n = v.shape[a] ?? 0
  const idx = indexTensor(indices)
  const flat = isTensor(idx) ? Int32Array.from(idx.data, (i) => (i < 0 ? i + n : i)) : undefined
  const fixed = flat === undefined ? idx : cpu.wrap('int32', [flat.length], flat)
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
 * import { takeAlongAxis, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(takeAlongAxis(tensor([[10, 20], [30, 40]]), [[1], [0]], 1)) // => [[20], [30]]
 * ```
 */
export function takeAlongAxis(x: TensorLike, indices: TensorLike, axis: number): Value {
  const v = val(x)
  const a = normaliseAxis(axis, v.shape.length)
  const idx = asDtype(val(indices), 'int32')
  if (idx.shape.length !== v.shape.length || idx.shape.some((n, i) => i !== a && n !== v.shape[i])) {
    throw new WhiteboxError(
      'shape',
      `takeAlongAxis: indices ${formatShape(idx.shape)} do not match ${formatShape(v.shape)}`,
    )
  }
  return bind1(takeAlongPrimitive, [v, idx], { axis: a })
}

/**
 * Add `updates` into a zero array of `shape` at `indices` along an axis (the adjoint of {@link takeAlongAxis}; torch
 * `scatter_add`). Repeated indices accumulate.
 *
 * @param shape - Shape of the result.
 * @param indices - int32 indices with the shape of `updates`.
 * @param updates - Values to add.
 * @param axis - The axis.
 * @returns The accumulated value.
 * @example
 * ```ts
 * import { scatterAdd, toArray } from 'whitebox-engine/tensor'
 * toArray(scatterAdd([3], [0, 2, 0], [1, 2, 3], 0)) // => [4, 0, 2]
 * ```
 */
export function scatterAdd(shape: Shape, indices: TensorLike, updates: TensorLike, axis: number): Value {
  const u = val(updates)
  const idx = asDtype(val(indices), 'int32')
  return bind1(putAlongPrimitive, [u, idx], { axis: normaliseAxis(axis, shape.length), shape: [...shape] })
}

/**
 * Sum rows of `data` into segments given by an id per row (TensorFlow `unsorted_segment_sum`).
 *
 * @param data - Values with segments along axis 0.
 * @param segmentIds - The segment of each row (int32, length of axis 0).
 * @param numSegments - The number of segments.
 * @returns A value with `numSegments` rows.
 * @example
 * ```ts
 * import { segmentSum, toArray } from 'whitebox-engine/tensor'
 * toArray(segmentSum([1, 2, 3], [0, 1, 0], 2)) // => [4, 2]
 * ```
 */
export function segmentSum(data: TensorLike, segmentIds: TensorLike, numSegments: number): Value {
  return bind1(takeScatterPrimitive, [val(data), indexTensor(segmentIds)], { axis: 0, size: numSegments })
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
 * import { argsort, tensor, toArray } from 'whitebox-engine/tensor'
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
 * import { sort, tensor, toArray } from 'whitebox-engine/tensor'
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
 * import { topk, tensor, toArray } from 'whitebox-engine/tensor'
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
 * import { oneHot, toArray } from 'whitebox-engine/tensor'
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
 * import { nonzero, tensor, toArray } from 'whitebox-engine/tensor'
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
 * import { maskedSelect, tensor, greater, toArray } from 'whitebox-engine/tensor'
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
 * import { searchsorted, toArray } from 'whitebox-engine/tensor'
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
 * Sorted unique values of a concrete array, with counts.
 *
 * @param x - A concrete input.
 * @returns `{ values, counts }` as 1-D tensors.
 * @example
 * ```ts
 * import { unique, toArray } from 'whitebox-engine/tensor'
 * toArray(unique([3, 1, 3, 2]).values) // => [1, 2, 3]
 * ```
 */
export function unique(x: TensorLike): { readonly values: Tensor; readonly counts: Tensor<'int32'> } {
  const t = concreteOrThrow(val(x), 'unique')
  const counts = new Map<number, number>()
  for (const v of t.data) counts.set(v, (counts.get(v) ?? 0) + 1)
  const keys = [...counts.keys()].sort((a, b) => a - b)
  return {
    values: t.backend.wrap(t.dtype, [keys.length], Float64Array.from(keys) as never),
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
 * import { bincount, toArray } from 'whitebox-engine/tensor'
 * toArray(bincount([0, 1, 1, 3])) // => [1, 2, 0, 1]
 * ```
 */
export function bincount(x: TensorLike, minLength = 0, weights?: TensorLike): Value {
  const t = concreteOrThrow(indexTensor(x), 'bincount')
  let size = minLength
  for (const v of t.data) {
    if (v < 0) throw new WhiteboxError('value', 'bincount: values must be non-negative')
    size = Math.max(size, v + 1)
  }
  if (weights === undefined) {
    const out = new Int32Array(size)
    for (const v of t.data) out[v] = (out[v] ?? 0) + 1
    return cpu.wrap('int32', [size], out)
  }
  return segmentSum(cast(weights, 'float64'), t, size)
}
