import { WhiteboxError } from '../../base/index.ts'
import { resultType } from '../dtype.ts'
import { arg, bind1, definePrimitive, type Primitive } from '../primitive.ts'
import { broadcastShapes, formatShape, normaliseAxis, type Shape, shapesEqual, sizeOf } from '../shape.ts'
import type { TensorLike } from '../tensor.ts'
import type { Value } from '../traced.ts'
import { asDtype, coerce2, val } from './_coerce.ts'
import { slice, take } from './indexing.ts'
import { sum } from './reduce.ts'

const reshapePrimitive: Primitive<{ readonly shape: Shape }> = definePrimitive<{ readonly shape: Shape }>({
  name: 'reshape',
  abstractEval: (xs, p) => [{ shape: p.shape, dtype: arg(xs, 0).dtype }],
  impl: (xs, p) => [arg(xs, 0).backend.wrap(arg(xs, 0).dtype, p.shape, arg(xs, 0).data)],
  jvp: (_xs, ts, p) => {
    const t = arg(ts, 0)
    return [t === null ? null : reshape(t, p.shape)]
  },
  vjp: (gs, xs) => {
    const g = arg(gs, 0)
    return [g === null ? null : reshape(g, arg(xs, 0).shape)]
  },
  describe: (p) => formatShape(p.shape),
})

const transposePrimitive: Primitive<{ readonly perm: readonly number[] }> = definePrimitive<{
  readonly perm: readonly number[]
}>({
  name: 'transpose',
  abstractEval: (xs, p) => [{ shape: p.perm.map((a) => arg(xs, 0).shape[a] ?? 1), dtype: arg(xs, 0).dtype }],
  impl: (xs, p) => [arg(xs, 0).backend.transpose(arg(xs, 0), p.perm)],
  jvp: (_xs, ts, p) => {
    const t = arg(ts, 0)
    return [t === null ? null : transpose(t, p.perm)]
  },
  vjp: (gs, _xs, p) => {
    const g = arg(gs, 0)
    const inverse = p.perm.map((_, i) => p.perm.indexOf(i))
    return [g === null ? null : transpose(g, inverse)]
  },
  describe: (p) => `[${p.perm.join(', ')}]`,
})

const broadcastPrimitive: Primitive<{ readonly shape: Shape }> = definePrimitive<{ readonly shape: Shape }>({
  name: 'broadcastTo',
  abstractEval: (xs, p) => [{ shape: p.shape, dtype: arg(xs, 0).dtype }],
  impl: (xs, p) => [arg(xs, 0).backend.broadcastTo(arg(xs, 0), p.shape)],
  jvp: (_xs, ts, p) => {
    const t = arg(ts, 0)
    return [t === null ? null : broadcastTo(t, p.shape)]
  },
  vjp: (gs, xs) => {
    const g = arg(gs, 0)
    return [g === null ? null : sumToShape(g, arg(xs, 0).shape)]
  },
  describe: (p) => formatShape(p.shape),
})

interface ConcatParams {
  readonly axis: number
}

const concatPrimitive: Primitive<ConcatParams> = definePrimitive<ConcatParams>({
  name: 'concat',
  abstractEval: (xs, p) => {
    const first = arg(xs, 0)
    const total = xs.reduce((acc, x) => acc + (x.shape[p.axis] ?? 0), 0)
    return [{ shape: first.shape.map((n, a) => (a === p.axis ? total : n)), dtype: first.dtype }]
  },
  impl: (xs, p) => [arg(xs, 0).backend.concat(xs, p.axis)],
  jvp: (xs, ts, p) => {
    if (ts.every((t) => t === null)) return [null]
    const parts = ts.map((t, i) => (t === null ? zerosOf(arg(xs, i)) : t))
    return [bind1(concatPrimitive, parts, p)]
  },
  vjp: (gs, xs, p) => {
    const g = arg(gs, 0)
    if (g === null) return xs.map(() => null)
    let offset = 0
    return xs.map((x) => {
      const n = x.shape[p.axis] ?? 0
      const part = sliceAxis(g, p.axis, offset, offset + n)
      offset += n
      return part
    })
  },
  describe: (p) => `axis=${p.axis}`,
})

interface PadParams {
  readonly widths: readonly (readonly [number, number])[]
  readonly value: number
}

const padPrimitive: Primitive<PadParams> = definePrimitive<PadParams>({
  name: 'pad',
  abstractEval: (xs, p) => [
    {
      shape: arg(xs, 0).shape.map((n, a) => n + (p.widths[a]?.[0] ?? 0) + (p.widths[a]?.[1] ?? 0)),
      dtype: arg(xs, 0).dtype,
    },
  ],
  impl: (xs, p) => [arg(xs, 0).backend.pad(arg(xs, 0), p.widths, p.value)],
  jvp: (_xs, ts, p) => {
    const t = arg(ts, 0)
    return [t === null ? null : bind1(padPrimitive, [t], { widths: p.widths, value: 0 })]
  },
  vjp: (gs, xs, p) => {
    const g = arg(gs, 0)
    if (g === null) return [null]
    const x = arg(xs, 0)
    const starts = p.widths.map(([before]) => before)
    const stops = x.shape.map((n, a) => (p.widths[a]?.[0] ?? 0) + n)
    return [slice(g, starts, stops)]
  },
})

function zerosOf(x: Value): Value {
  return broadcastTo(coerce2(0, x, 'any')[0], x.shape)
}

/**
 * Give a value a new shape with the same number of elements; one axis may be `-1` and is inferred. Reshaping never
 * copies (tensors are immutable, so sharing storage is safe).
 *
 * @param x - Input.
 * @param shape - The new shape.
 * @returns The reshaped value.
 * @throws WhiteboxError with code `shape` when sizes differ.
 * @example
 * ```ts
 * import { reshape, arange } from 'whitebox-engine/tensor'
 * reshape(arange(0, 6), [2, -1]).shape // => [2, 3]
 * ```
 */
export function reshape(x: TensorLike, shape: Shape): Value {
  const v = val(x)
  const total = sizeOf(v.shape)
  const known = shape.filter((n) => n !== -1)
  if (shape.length - known.length > 1) throw new WhiteboxError('shape', 'reshape: at most one axis may be -1')
  const knownSize = sizeOf(known)
  const resolved = shape.map((n) => (n === -1 ? total / knownSize : n))
  if (sizeOf(resolved) !== total || resolved.some((n) => !Number.isInteger(n) || n < 0)) {
    throw new WhiteboxError('shape', `reshape: cannot view ${formatShape(v.shape)} as ${formatShape(shape)}`, {
      from: v.shape,
      to: shape,
    })
  }
  if (shapesEqual(resolved, v.shape)) return v
  return bind1(reshapePrimitive, [v], { shape: resolved })
}

/**
 * Flatten to one dimension.
 *
 * @param x - Input.
 * @returns A 1-D value.
 * @example
 * ```ts
 * import { flatten, tensor } from 'whitebox-engine/tensor'
 * flatten(tensor([[1, 2], [3, 4]])).shape // => [4]
 * ```
 */
export function flatten(x: TensorLike): Value {
  return reshape(x, [-1])
}

/**
 * Permute axes; by default reverse them (a matrix transpose for 2-D input).
 *
 * @param x - Input.
 * @param perm - The new order of axes (negative axes allowed).
 * @returns The transposed value.
 * @example
 * ```ts
 * import { transpose, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(transpose(tensor([[1, 2, 3]]))) // => [[1], [2], [3]]
 * ```
 */
export function transpose(x: TensorLike, perm?: readonly number[]): Value {
  const v = val(x)
  const rank = v.shape.length
  const p = (perm ?? v.shape.map((_, i) => rank - 1 - i)).map((a) => normaliseAxis(a, rank))
  if (p.every((a, i) => a === i)) return v
  return bind1(transposePrimitive, [v], { perm: p })
}

/**
 * Exchange two axes.
 *
 * @param x - Input.
 * @param a - First axis.
 * @param b - Second axis.
 * @returns The value with the axes swapped.
 * @example
 * ```ts
 * import { swapaxes, zeros } from 'whitebox-engine/tensor'
 * swapaxes(zeros([2, 3, 4]), 0, 2).shape // => [4, 3, 2]
 * ```
 */
export function swapaxes(x: TensorLike, a: number, b: number): Value {
  const v = val(x)
  const rank = v.shape.length
  const perm = v.shape.map((_, i) => i)
  const i = normaliseAxis(a, rank)
  const j = normaliseAxis(b, rank)
  perm[i] = j
  perm[j] = i
  return transpose(v, perm)
}

/**
 * Move an axis to a new position, keeping the order of the others.
 *
 * @param x - Input.
 * @param source - The axis to move.
 * @param destination - Where it goes.
 * @returns The value with the axis moved.
 * @example
 * ```ts
 * import { moveaxis, zeros } from 'whitebox-engine/tensor'
 * moveaxis(zeros([2, 3, 4]), 0, -1).shape // => [3, 4, 2]
 * ```
 */
export function moveaxis(x: TensorLike, source: number, destination: number): Value {
  const v = val(x)
  const rank = v.shape.length
  const s = normaliseAxis(source, rank)
  const d = normaliseAxis(destination, rank)
  const rest = v.shape.map((_, i) => i).filter((i) => i !== s)
  rest.splice(d, 0, s)
  return transpose(v, rest)
}

/**
 * Insert a length-1 axis.
 *
 * @param x - Input.
 * @param axis - Position of the new axis (negative counts from the end of the result).
 * @returns The expanded value.
 * @example
 * ```ts
 * import { expandDims, zeros } from 'whitebox-engine/tensor'
 * expandDims(zeros([3]), 0).shape // => [1, 3]
 * ```
 */
export function expandDims(x: TensorLike, axis: number): Value {
  const v = val(x)
  const a = normaliseAxis(axis, v.shape.length + 1)
  const shape = [...v.shape]
  shape.splice(a, 0, 1)
  return reshape(v, shape)
}

/**
 * Remove length-1 axes (all of them, or the given ones).
 *
 * @param x - Input.
 * @param axis - Axes to remove; each must have length 1.
 * @returns The squeezed value.
 * @throws WhiteboxError with code `shape` when a named axis does not have length 1.
 * @example
 * ```ts
 * import { squeeze, zeros } from 'whitebox-engine/tensor'
 * squeeze(zeros([1, 3, 1])).shape // => [3]
 * ```
 */
export function squeeze(x: TensorLike, axis?: number | readonly number[]): Value {
  const v = val(x)
  const rank = v.shape.length
  const axes =
    axis === undefined
      ? v.shape.map((n, i) => (n === 1 ? i : -1)).filter((i) => i >= 0)
      : (typeof axis === 'number' ? [axis] : axis).map((a) => normaliseAxis(a, rank))
  for (const a of axes) {
    if (v.shape[a] !== 1) throw new WhiteboxError('shape', `squeeze: axis ${a} has length ${v.shape[a]}, not 1`)
  }
  return reshape(
    v,
    v.shape.filter((_, i) => !axes.includes(i)),
  )
}

/**
 * Broadcast a value to a larger shape (numpy rules).
 *
 * @param x - Input.
 * @param shape - A shape `x` broadcasts to.
 * @returns The broadcast value.
 * @throws WhiteboxError with code `shape` when `x` does not broadcast to `shape`.
 * @example
 * ```ts
 * import { broadcastTo, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(broadcastTo(tensor([1, 2]), [2, 2])) // => [[1, 2], [1, 2]]
 * ```
 */
export function broadcastTo(x: TensorLike, shape: Shape): Value {
  const v = val(x)
  if (shapesEqual(v.shape, shape)) return v
  const target = broadcastShapes(v.shape, shape)
  if (!shapesEqual(target, shape)) {
    throw new WhiteboxError('shape', `cannot broadcast ${formatShape(v.shape)} to ${formatShape(shape)}`)
  }
  return bind1(broadcastPrimitive, [v], { shape: [...shape] })
}

/**
 * Sum a broadcast value back to a smaller shape: the adjoint of broadcasting, used by every broadcasting derivative.
 *
 * @param x - A value whose shape is a broadcast of `shape`.
 * @param shape - The target shape.
 * @returns A value with exactly `shape`.
 * @example
 * ```ts
 * import { sumToShape, ones, toArray } from 'whitebox-engine/tensor'
 * toArray(sumToShape(ones([3, 2]), [2])) // => [3, 3]
 * ```
 */
export function sumToShape(x: TensorLike, shape: Shape): Value {
  const v = val(x)
  if (shapesEqual(v.shape, shape)) return v
  const lead = v.shape.length - shape.length
  const axes: number[] = []
  for (let axis = 0; axis < v.shape.length; axis++) {
    if (axis < lead || ((shape[axis - lead] ?? 1) === 1 && (v.shape[axis] ?? 1) !== 1)) axes.push(axis)
  }
  return reshape(axes.length === 0 ? v : sum(v, axes, true), shape)
}

/**
 * Join values along an existing axis.
 *
 * @param xs - Values with equal shapes except along `axis`.
 * @param axis - The axis (default 0).
 * @returns The concatenation.
 * @throws WhiteboxError with code `shape` when shapes disagree off the axis.
 * @example
 * ```ts
 * import { concat, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(concat([tensor([1, 2]), tensor([3])])) // => [1, 2, 3]
 * ```
 */
export function concat(xs: readonly TensorLike[], axis = 0): Value {
  const vs = xs.map(val)
  const first = vs[0]
  if (first === undefined) throw new WhiteboxError('value', 'concat: need at least one value')
  const a = normaliseAxis(axis, first.shape.length)
  const dtype = vs.reduce((d, v) => resultType(d, v.dtype), first.dtype)
  for (const v of vs) {
    if (v.shape.length !== first.shape.length || v.shape.some((n, i) => i !== a && n !== first.shape[i])) {
      throw new WhiteboxError(
        'shape',
        `concat: ${formatShape(v.shape)} does not match ${formatShape(first.shape)} off axis ${a}`,
      )
    }
  }
  const same = vs.map((v) => asDtype(v, dtype))
  return same.length === 1 ? (same[0] as Value) : bind1(concatPrimitive, same, { axis: a })
}

/**
 * Join values along a new axis.
 *
 * @param xs - Values of one shape.
 * @param axis - Position of the new axis (default 0).
 * @returns The stack.
 * @example
 * ```ts
 * import { stack, tensor } from 'whitebox-engine/tensor'
 * stack([tensor([1, 2]), tensor([3, 4])]).shape // => [2, 2]
 * ```
 */
export function stack(xs: readonly TensorLike[], axis = 0): Value {
  return concat(
    xs.map((x) => expandDims(x, axis)),
    axis,
  )
}

/**
 * Take a contiguous range along one axis.
 *
 * @param x - Input.
 * @param axis - The axis.
 * @param start - First index.
 * @param stop - Exclusive end.
 * @returns The slice.
 * @example
 * ```ts
 * import { sliceAxis, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(sliceAxis(tensor([[1, 2, 3]]), 1, 1, 3)) // => [[2, 3]]
 * ```
 */
export function sliceAxis(x: TensorLike, axis: number, start: number, stop: number): Value {
  const v = val(x)
  const a = normaliseAxis(axis, v.shape.length)
  return slice(
    v,
    v.shape.map((_, i) => (i === a ? start : 0)),
    v.shape.map((n, i) => (i === a ? stop : n)),
  )
}

/**
 * Split into pieces along an axis, by a number of equal sections or at given indices (numpy `split`).
 *
 * @param x - Input.
 * @param sections - A number of equal sections, or the indices at which to split.
 * @param axis - The axis (default 0).
 * @returns The pieces.
 * @throws WhiteboxError with code `shape` when equal sections do not divide the axis.
 * @example
 * ```ts
 * import { split, tensor, toArray } from 'whitebox-engine/tensor'
 * split(tensor([1, 2, 3, 4]), 2).map(toArray) // => [[1, 2], [3, 4]]
 * split(tensor([1, 2, 3, 4]), [1]).map(toArray) // => [[1], [2, 3, 4]]
 * ```
 */
export function split(x: TensorLike, sections: number | readonly number[], axis = 0): Value[] {
  const v = val(x)
  const a = normaliseAxis(axis, v.shape.length)
  const n = v.shape[a] ?? 0
  let cuts: number[]
  if (typeof sections === 'number') {
    if (n % sections !== 0) throw new WhiteboxError('shape', `split: ${sections} sections do not divide length ${n}`)
    cuts = Array.from({ length: sections - 1 }, (_, i) => ((i + 1) * n) / sections)
  } else cuts = [...sections]
  const bounds = [0, ...cuts, n]
  return bounds.slice(0, -1).map((start, i) => sliceAxis(v, a, start, bounds[i + 1] ?? n))
}

/**
 * Split into `n` nearly equal chunks along an axis (the last may be shorter, like `torch.chunk`).
 *
 * @param x - Input.
 * @param n - Number of chunks.
 * @param axis - The axis (default 0).
 * @returns The chunks.
 * @example
 * ```ts
 * import { chunk, arange } from 'whitebox-engine/tensor'
 * chunk(arange(0, 5), 2).map((c) => c.shape[0]) // => [3, 2]
 * ```
 */
export function chunk(x: TensorLike, n: number, axis = 0): Value[] {
  const v = val(x)
  const a = normaliseAxis(axis, v.shape.length)
  const len = v.shape[a] ?? 0
  const size = Math.ceil(len / n)
  const cuts: number[] = []
  for (let c = size; c < len; c += size) cuts.push(c)
  return split(v, cuts, a)
}

/**
 * Reverse the order of elements along axes (all by default).
 *
 * @param x - Input.
 * @param axis - An axis or axes.
 * @returns The flipped value.
 * @example
 * ```ts
 * import { flip, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(flip(tensor([1, 2, 3]))) // => [3, 2, 1]
 * ```
 */
export function flip(x: TensorLike, axis?: number | readonly number[]): Value {
  const v = val(x)
  const rank = v.shape.length
  const axes =
    axis === undefined
      ? v.shape.map((_, i) => i)
      : (typeof axis === 'number' ? [axis] : axis).map((a) => normaliseAxis(a, rank))
  return slice(
    v,
    v.shape.map(() => undefined),
    v.shape.map(() => undefined),
    v.shape.map((_, i) => (axes.includes(i) ? -1 : 1)),
  )
}

/**
 * Roll elements along an axis (flattened when no axis is given): elements shifted past the end wrap to the start.
 *
 * @param x - Input.
 * @param shift - Number of places (may be negative).
 * @param axis - The axis.
 * @returns The rolled value.
 * @example
 * ```ts
 * import { roll, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(roll(tensor([1, 2, 3, 4]), 1)) // => [4, 1, 2, 3]
 * ```
 */
export function roll(x: TensorLike, shift: number, axis?: number): Value {
  const v = val(x)
  if (axis === undefined) return reshape(roll(reshape(v, [-1]), shift, 0), v.shape)
  const a = normaliseAxis(axis, v.shape.length)
  const n = v.shape[a] ?? 0
  const s = ((shift % n) + n) % n
  if (s === 0) return v
  return concat([sliceAxis(v, a, n - s, n), sliceAxis(v, a, 0, n - s)], a)
}

/**
 * Repeat a whole value along each axis (numpy `tile`).
 *
 * @param x - Input.
 * @param reps - Repetitions per axis (aligned to the trailing axes).
 * @returns The tiled value.
 * @example
 * ```ts
 * import { tile, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(tile(tensor([1, 2]), [2])) // => [1, 2, 1, 2]
 * ```
 */
export function tile(x: TensorLike, reps: readonly number[]): Value {
  let v = val(x)
  const rank = Math.max(v.shape.length, reps.length)
  v = reshape(v, [...new Array<number>(rank - v.shape.length).fill(1), ...v.shape])
  const r = [...new Array<number>(rank - reps.length).fill(1), ...reps]
  // Interleave a repetition axis before each axis, broadcast it, then merge pairs.
  const paired = v.shape.flatMap((n) => [1, n])
  const target = v.shape.flatMap((n, i) => [r[i] ?? 1, n])
  return reshape(
    broadcastTo(reshape(v, paired), target),
    v.shape.map((n, i) => n * (r[i] ?? 1)),
  )
}

/**
 * Repeat each element along an axis (numpy `repeat`); flattens when no axis is given.
 *
 * @param x - Input.
 * @param repeats - Copies of each element (one number, or one per element along the axis).
 * @param axis - The axis.
 * @returns The value with elements repeated.
 * @example
 * ```ts
 * import { repeat, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(repeat(tensor([1, 2]), 2)) // => [1, 1, 2, 2]
 * ```
 */
export function repeat(x: TensorLike, repeats: number | readonly number[], axis?: number): Value {
  const v = axis === undefined ? reshape(val(x), [-1]) : val(x)
  const a = axis === undefined ? 0 : normaliseAxis(axis, v.shape.length)
  const n = v.shape[a] ?? 0
  const counts = typeof repeats === 'number' ? new Array<number>(n).fill(repeats) : repeats
  const indices: number[] = []
  counts.forEach((c, i) => {
    for (let k = 0; k < c; k++) indices.push(i)
  })
  return take(v, indices, a)
}

/**
 * Pad with a constant value (other modes are in `numerics/convolution`).
 *
 * @param x - Input.
 * @param widths - `[before, after]` per axis, or one pair for every axis.
 * @param value - The fill value (default 0).
 * @returns The padded value.
 * @example
 * ```ts
 * import { pad, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(pad(tensor([1, 2]), [[1, 2]])) // => [0, 1, 2, 0, 0]
 * ```
 */
export function pad(x: TensorLike, widths: readonly (readonly [number, number])[], value = 0): Value {
  const v = val(x)
  const all = widths.length === 1 ? v.shape.map(() => widths[0] as readonly [number, number]) : widths
  if (all.length !== v.shape.length) throw new WhiteboxError('shape', `pad: need ${v.shape.length} width pairs`)
  return bind1(padPrimitive, [v], { widths: all, value })
}
