import { OmaiError } from '../../base/index.ts'
import { allocate, type DType } from '../dtype.ts'
import { broadcastShapes, formatShape, type Shape, sizeOf, stridesOf } from '../shape.ts'
import type { Tensor } from '../tensor.ts'
import { assertSameDtype, broadcastIndex, lanes, walkStrided, wrap } from './_util.ts'
import { planBroadcast, runCopy } from './broadcast.ts'

export function matmul<D extends DType>(a: Tensor<D>, b: Tensor<D>): Tensor<D> {
  assertSameDtype(a, b, 'matmul')
  const ra = a.shape.length
  const rb = b.shape.length
  const m = a.shape[ra - 2] ?? 1
  const k = a.shape[ra - 1] ?? 1
  const k2 = b.shape[rb - 2] ?? 1
  const n = b.shape[rb - 1] ?? 1
  if (ra < 2 || rb < 2 || k !== k2) {
    throw new OmaiError('shape', `matmul: shapes ${formatShape(a.shape)} and ${formatShape(b.shape)} do not align`)
  }
  const batchA = a.shape.slice(0, -2)
  const batchB = b.shape.slice(0, -2)
  const batch = broadcastShapes(batchA, batchB)
  const ia = broadcastIndex(batch, batchA)
  const ib = broadcastIndex(batch, batchB)
  const count = sizeOf(batch)
  const out = allocate(a.dtype, count * m * n)
  for (let t = 0; t < count; t++) {
    const offA = (ia[t] ?? 0) * m * k
    const offB = (ib[t] ?? 0) * k * n
    const offC = t * m * n
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        // No zero skipping: 0 * Infinity and 0 * NaN are NaN, as in IEEE and numpy.
        const aip = a.data[offA + i * k + p] ?? 0
        const rowB = offB + p * n
        const rowC = offC + i * n
        for (let j = 0; j < n; j++) out[rowC + j] = (out[rowC + j] ?? 0) + aip * (b.data[rowB + j] ?? 0)
      }
    }
  }
  return wrap(a.dtype, [...batch, m, n], out)
}

export function transpose<D extends DType>(x: Tensor<D>, perm: readonly number[]): Tensor<D> {
  const rank = x.shape.length
  if (perm.length !== rank || new Set(perm).size !== rank || perm.some((p) => p < 0 || p >= rank)) {
    throw new OmaiError('shape', `transpose: [${perm.join(', ')}] is not a permutation of ${rank} axes`)
  }
  const outShape = perm.map((p) => x.shape[p] ?? 1)
  const inStrides = stridesOf(x.shape)
  const out = allocate(x.dtype, x.data.length)
  walkStrided(
    outShape,
    perm.map((p) => inStrides[p] ?? 0),
    (i, offset) => {
      out[i] = x.data[offset] ?? 0
    },
  )
  return wrap(x.dtype, outShape, out)
}

export function broadcastTo<D extends DType>(x: Tensor<D>, shape: Shape): Tensor<D> {
  const target = broadcastShapes(x.shape, shape)
  if (target.length !== shape.length || target.some((n, i) => n !== shape[i])) {
    throw new OmaiError('shape', `cannot broadcast ${formatShape(x.shape)} to ${formatShape(shape)}`)
  }
  const plan = sizeOf(shape) === 0 ? null : planBroadcast(shape, [x.shape])
  if (plan === null) return broadcastToIndexed(x, shape)
  const out = allocate(x.dtype, sizeOf(shape))
  runCopy(out, x.data, plan)
  return wrap(x.dtype, shape, out)
}

/** The general `broadcastTo`: an output-sized index map (used when {@link broadcastTo} has no layout for the shapes). */
export function broadcastToIndexed<D extends DType>(x: Tensor<D>, shape: Shape): Tensor<D> {
  const index = broadcastIndex(shape, x.shape)
  const out = allocate(x.dtype, index.length)
  for (let i = 0; i < out.length; i++) out[i] = x.data[index[i] ?? 0] ?? 0
  return wrap(x.dtype, shape, out)
}

function sliceShape(starts: readonly number[], stops: readonly number[], steps: readonly number[]): number[] {
  return starts.map((start, i) => {
    const step = steps[i] ?? 1
    const stop = stops[i] ?? start
    return Math.max(0, Math.ceil((stop - start) / step))
  })
}

export function slice<D extends DType>(
  x: Tensor<D>,
  starts: readonly number[],
  stops: readonly number[],
  steps: readonly number[],
): Tensor<D> {
  const outShape = sliceShape(starts, stops, steps)
  const strides = stridesOf(x.shape)
  const base = starts.reduce((acc, s, i) => acc + s * (strides[i] ?? 0), 0)
  const out = allocate(x.dtype, sizeOf(outShape))
  walkStrided(
    outShape,
    strides.map((s, i) => s * (steps[i] ?? 1)),
    (i, offset) => {
      out[i] = x.data[offset] ?? 0
    },
    base,
  )
  return wrap(x.dtype, outShape, out)
}

export function sliceScatter<D extends DType>(
  values: Tensor<D>,
  shape: Shape,
  starts: readonly number[],
  steps: readonly number[],
): Tensor<D> {
  const strides = stridesOf(shape)
  const base = starts.reduce((acc, s, i) => acc + s * (strides[i] ?? 0), 0)
  const out = allocate(values.dtype, sizeOf(shape))
  walkStrided(
    values.shape,
    strides.map((s, i) => s * (steps[i] ?? 1)),
    (i, offset) => {
      out[offset] = (out[offset] ?? 0) + (values.data[i] ?? 0)
    },
    base,
  )
  return wrap(values.dtype, shape, out)
}

export function concat<D extends DType>(xs: readonly Tensor<D>[], axis: number): Tensor<D> {
  const first = xs[0]
  if (first === undefined) throw new OmaiError('value', 'concat: need at least one tensor')
  const total = xs.reduce((acc, x) => acc + (x.shape[axis] ?? 0), 0)
  const shape = first.shape.map((n, a) => (a === axis ? total : n))
  const { outer, inner } = lanes(shape, axis)
  const out = allocate(first.dtype, sizeOf(shape))
  let offsetAxis = 0
  for (const x of xs) {
    const n = x.shape[axis] ?? 0
    for (let o = 0; o < outer; o++) {
      const src = o * n * inner
      const dst = (o * total + offsetAxis) * inner
      out.set(x.data.subarray(src, src + n * inner), dst)
    }
    offsetAxis += n
  }
  return wrap(first.dtype, shape, out)
}

export function pad<D extends DType>(
  x: Tensor<D>,
  widths: readonly (readonly [number, number])[],
  value: number,
): Tensor<D> {
  const shape = x.shape.map((n, a) => n + (widths[a]?.[0] ?? 0) + (widths[a]?.[1] ?? 0))
  const out = allocate(x.dtype, sizeOf(shape))
  if (value !== 0) out.fill(value)
  const strides = stridesOf(shape)
  const base = widths.reduce((acc, [before], a) => acc + before * (strides[a] ?? 0), 0)
  walkStrided(
    x.shape,
    strides,
    (i, offset) => {
      out[offset] = x.data[i] ?? 0
    },
    base,
  )
  return wrap(x.dtype, shape, out)
}

export function take<D extends DType>(x: Tensor<D>, indices: Int32Array, axis: number): Tensor<D> {
  const { outer, n, inner } = lanes(x.shape, axis)
  const out = allocate(x.dtype, outer * indices.length * inner)
  if (inner === 1 && outer > 0) {
    // Gathering along the last axis (or a vector): the bounds are checked once over the indices (a negative index is a huge
    // unsigned one, so one comparison covers both ends; every row reads the same indices, so the first failing index is the
    // one reported, as in the general loop), then each row is a plain copy.
    const m = indices.length
    for (let t = 0; t < m; t++) {
      const index = indices[t] as number
      if (index >>> 0 >= n) throw new OmaiError('value', `take: index ${index} out of range for length ${n}`)
    }
    const data = x.data
    if (outer === 1) for (let t = 0; t < m; t++) out[t] = data[indices[t] as number] as number
    else
      for (let o = 0; o < outer; o++) {
        const from = o * n
        const to = o * m
        for (let t = 0; t < m; t++) out[to + t] = data[from + (indices[t] as number)] as number
      }
    return wrap(
      x.dtype,
      x.shape.map((s, a) => (a === axis ? m : s)),
      out,
    )
  }
  for (let o = 0; o < outer; o++) {
    for (let t = 0; t < indices.length; t++) {
      const index = indices[t] ?? 0
      if (index < 0 || index >= n) throw new OmaiError('value', `take: index ${index} out of range for length ${n}`)
      const src = (o * n + index) * inner
      const dst = (o * indices.length + t) * inner
      for (let k = 0; k < inner; k++) out[dst + k] = x.data[src + k] ?? 0
    }
  }
  return wrap(
    x.dtype,
    x.shape.map((s, a) => (a === axis ? indices.length : s)),
    out,
  )
}

export function takeScatter<D extends DType>(
  updates: Tensor<D>,
  indices: Int32Array,
  axis: number,
  size: number,
): Tensor<D> {
  const shape = updates.shape.map((s, a) => (a === axis ? size : s))
  const { outer, inner } = lanes(shape, axis)
  const out = allocate(updates.dtype, sizeOf(shape))
  if (inner === 1) {
    // Scattering along the last axis (segment sums of a vector): the accumulation order is that of the loop below.
    const m = indices.length
    const data = updates.data
    for (let o = 0; o < outer; o++) {
      const from = o * m
      const to = o * size
      for (let t = 0; t < m; t++) {
        const dst = to + (indices[t] as number)
        out[dst] = (out[dst] as number) + (data[from + t] as number)
      }
    }
    return wrap(updates.dtype, shape, out)
  }
  for (let o = 0; o < outer; o++) {
    for (let t = 0; t < indices.length; t++) {
      const src = (o * indices.length + t) * inner
      const dst = (o * size + (indices[t] ?? 0)) * inner
      for (let k = 0; k < inner; k++) out[dst + k] = (out[dst + k] ?? 0) + (updates.data[src + k] ?? 0)
    }
  }
  return wrap(updates.dtype, shape, out)
}

export function takeAlongAxis<D extends DType>(x: Tensor<D>, indices: Tensor<'int32'>, axis: number): Tensor<D> {
  const { outer, n, inner } = lanes(x.shape, axis)
  const m = indices.shape[axis] ?? 0
  const out = allocate(x.dtype, outer * m * inner)
  if (inner === 1) {
    // Picking along the last axis: a running offset per row, the bounds check per index as below.
    const data = x.data
    const picks = indices.data
    for (let o = 0; o < outer; o++) {
      const from = o * n
      const to = o * m
      for (let t = 0; t < m; t++) {
        const index = picks[to + t] as number
        if (index < 0 || index >= n) {
          throw new OmaiError('value', `takeAlongAxis: index ${index} out of range for length ${n}`)
        }
        out[to + t] = data[from + index] as number
      }
    }
    return wrap(x.dtype, indices.shape, out)
  }
  for (let o = 0; o < outer; o++) {
    for (let t = 0; t < m; t++) {
      for (let k = 0; k < inner; k++) {
        const pos = (o * m + t) * inner + k
        const index = indices.data[pos] ?? 0
        if (index < 0 || index >= n) {
          throw new OmaiError('value', `takeAlongAxis: index ${index} out of range for length ${n}`)
        }
        out[pos] = x.data[(o * n + index) * inner + k] ?? 0
      }
    }
  }
  return wrap(x.dtype, indices.shape, out)
}

export function putAlongAxisAdd<D extends DType>(
  updates: Tensor<D>,
  indices: Tensor<'int32'>,
  axis: number,
  shape: Shape,
): Tensor<D> {
  const { outer, n, inner } = lanes(shape, axis)
  const m = indices.shape[axis] ?? 0
  const out = allocate(updates.dtype, sizeOf(shape))
  if (inner === 1) {
    const data = updates.data
    const picks = indices.data
    for (let o = 0; o < outer; o++) {
      const from = o * m
      const to = o * n
      for (let t = 0; t < m; t++) {
        const dst = to + (picks[from + t] as number)
        out[dst] = (out[dst] as number) + (data[from + t] as number)
      }
    }
    return wrap(updates.dtype, shape, out)
  }
  for (let o = 0; o < outer; o++) {
    for (let t = 0; t < m; t++) {
      for (let k = 0; k < inner; k++) {
        const pos = (o * m + t) * inner + k
        const dst = (o * n + (indices.data[pos] ?? 0)) * inner + k
        out[dst] = (out[dst] ?? 0) + (updates.data[pos] ?? 0)
      }
    }
  }
  return wrap(updates.dtype, shape, out)
}
