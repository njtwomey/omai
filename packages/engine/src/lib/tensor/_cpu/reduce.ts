import type { ReduceKernel } from '../backend.ts'
import { allocate, type DType } from '../dtype.ts'
import { sizeOf, stridesOf } from '../shape.ts'
import type { Tensor } from '../tensor.ts'
import { lanes, walkStrided, wrap } from './_util.ts'

const identity: Record<ReduceKernel, number> = { sum: 0, prod: 1, max: -Infinity, min: Infinity }
const combine: Record<ReduceKernel, (acc: number, x: number) => number> = {
  sum: (acc, x) => acc + x,
  prod: (acc, x) => acc * x,
  max: (acc, x) => (Number.isNaN(acc) || Number.isNaN(x) ? Number.NaN : Math.max(x, acc)),
  min: (acc, x) => (Number.isNaN(acc) || Number.isNaN(x) ? Number.NaN : Math.min(x, acc)),
}

export function reduce<D extends DType>(
  op: ReduceKernel,
  x: Tensor<D>,
  axes: readonly number[],
  keepDims: boolean,
): Tensor<D> {
  const reduced = new Set(axes)
  const keptShape = x.shape.map((n, axis) => (reduced.has(axis) ? 1 : n))
  const outShape = keepDims ? keptShape : x.shape.filter((_, axis) => !reduced.has(axis))
  // Accumulate in float64: integer storage cannot hold the +-Infinity identities of max and min.
  const acc = new Float64Array(sizeOf(outShape)).fill(identity[op])
  // Walk the input with the output's strides on kept axes and zero on reduced axes.
  const strides = stridesOf(keptShape).map((s, axis) => (reduced.has(axis) ? 0 : s))
  const f = combine[op]
  walkStrided(x.shape, strides, (i, offset) => {
    acc[offset] = f(acc[offset] ?? 0, x.data[i] ?? 0)
  })
  const out = allocate(x.dtype, acc.length)
  for (let i = 0; i < acc.length; i++) out[i] = acc[i] ?? 0
  return wrap(x.dtype, outShape, out)
}

export function argReduce(op: 'max' | 'min', x: Tensor, axis: number, keepDims: boolean): Tensor<'int32'> {
  const { outer, n, inner } = lanes(x.shape, axis)
  const out = new Int32Array(outer * inner)
  for (let o = 0; o < outer; o++) {
    for (let k = 0; k < inner; k++) {
      let best = 0
      let bestValue = x.data[o * n * inner + k] ?? Number.NaN
      for (let i = 1; i < n && !Number.isNaN(bestValue); i++) {
        const v = x.data[(o * n + i) * inner + k] ?? Number.NaN
        // NaN wins (numpy returns the first NaN); otherwise the first extreme value.
        if (Number.isNaN(v) || (op === 'max' ? v > bestValue : v < bestValue)) {
          best = i
          bestValue = v
        }
      }
      out[o * inner + k] = best
    }
  }
  const shape = keepDims ? x.shape.map((s, a) => (a === axis ? 1 : s)) : x.shape.filter((_, a) => a !== axis)
  return wrap('int32', shape, out)
}

export function scan<D extends DType>(op: 'sum' | 'prod', x: Tensor<D>, axis: number, reverse: boolean): Tensor<D> {
  const { outer, n, inner } = lanes(x.shape, axis)
  const out = allocate(x.dtype, x.data.length)
  for (let o = 0; o < outer; o++) {
    for (let k = 0; k < inner; k++) {
      let acc = op === 'sum' ? 0 : 1
      for (let step = 0; step < n; step++) {
        const i = reverse ? n - 1 - step : step
        const idx = (o * n + i) * inner + k
        acc = op === 'sum' ? acc + (x.data[idx] ?? 0) : acc * (x.data[idx] ?? 0)
        out[idx] = acc
      }
    }
  }
  return wrap(x.dtype, x.shape, out)
}
