import { OmaiError } from '../../base/index.ts'
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
  // max and min have no identity in the dtype: an empty reduction with outputs to fill is numpy's ValueError.
  if ((op === 'max' || op === 'min') && x.data.length === 0 && sizeOf(x.shape.filter((_, a) => !reduced.has(a))) > 0) {
    throw new OmaiError('value', `${op}: cannot reduce a zero-length axis (no identity)`, { shape: x.shape })
  }
  const keptShape = x.shape.map((n, axis) => (reduced.has(axis) ? 1 : n))
  const outShape = keepDims ? keptShape : x.shape.filter((_, axis) => !reduced.has(axis))
  // Accumulate in float64: integer storage cannot hold the +-Infinity identities of max and min.
  const acc = new Float64Array(sizeOf(outShape)).fill(identity[op])
  const rank = x.shape.length
  const k = reduced.size
  const inRange = axes.every((a) => a >= 0 && a < rank)
  const trailing = inRange && axes.every((a) => a >= rank - k)
  const leading = inRange && axes.every((a) => a < k)
  if (rank > 0 && (trailing || leading)) {
    // The reduced axes are the last k or the first k: each accumulator takes its elements in the same ascending order
    // as the strided walk below, so the results are identical; only the offset arithmetic goes.
    const split = trailing ? rank - k : k
    const head = sizeOf(x.shape.slice(0, split))
    const tail = sizeOf(x.shape.slice(split))
    const data = x.data
    if (trailing) {
      for (let o = 0; o < head; o++) {
        let a = identity[op]
        const base = o * tail
        if (op === 'sum') for (let j = 0; j < tail; j++) a += data[base + j] ?? 0
        else for (let j = 0; j < tail; j++) a = combine[op](a, data[base + j] ?? 0)
        acc[o] = a
      }
    } else {
      for (let o = 0; o < head; o++) {
        const base = o * tail
        if (op === 'sum') for (let j = 0; j < tail; j++) acc[j] = (acc[j] ?? 0) + (data[base + j] ?? 0)
        else for (let j = 0; j < tail; j++) acc[j] = combine[op](acc[j] ?? 0, data[base + j] ?? 0)
      }
    }
  } else {
    // Walk the input with the output's strides on kept axes and zero on reduced axes.
    const strides = stridesOf(keptShape).map((s, axis) => (reduced.has(axis) ? 0 : s))
    const f = combine[op]
    walkStrided(x.shape, strides, (i, offset) => {
      acc[offset] = f(acc[offset] ?? 0, x.data[i] ?? 0)
    })
  }
  const out = allocate(x.dtype, acc.length)
  for (let i = 0; i < acc.length; i++) out[i] = acc[i] ?? 0
  return wrap(x.dtype, outShape, out)
}

export function argReduce(op: 'max' | 'min', x: Tensor, axis: number, keepDims: boolean): Tensor<'int32'> {
  const { outer, n, inner } = lanes(x.shape, axis)
  if (n === 0 && outer * inner > 0) throw new OmaiError('value', `arg${op}: cannot reduce a zero-length axis`)
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
