import { OmaiError } from '../../base/index.ts'
import type { ReduceKernel } from '../backend.ts'
import { allocate, type DType } from '../dtype.ts'
import { sizeOf, stridesOf } from '../shape.ts'
import type { Tensor } from '../tensor.ts'
import { lanes, walkStrided, wrap } from './_util.ts'

/**
 * The reduced axes as one block of outer, reduced and inner extents (flat index `(o * n + i) * inner + k`), when they are
 * adjacent; `null` for a set with a gap (`[0, 2]`), which keeps the strided walk.
 */
function reducedBlock(
  shape: readonly number[],
  axes: readonly number[],
): { readonly outer: number; readonly n: number; readonly inner: number } | null {
  const lo = Math.min(...axes)
  const hi = Math.max(...axes)
  if (axes.length === 0 || lo < 0 || hi >= shape.length || hi - lo + 1 !== new Set(axes).size) return null
  return { outer: sizeOf(shape.slice(0, lo)), n: sizeOf(shape.slice(lo, hi + 1)), inner: sizeOf(shape.slice(hi + 1)) }
}

/**
 * The reductions over a block of adjacent axes, as plain loops per operation (no closure per element). Every output takes its
 * elements in ascending storage order from the identity, as the strided walk does, so sums and products are bit-identical to
 * it; the maximum and minimum are order-free (NaN propagates, +0 beats -0), so a row is reduced with four running values.
 */
function reduceLanes(
  op: ReduceKernel,
  data: ArrayLike<number>,
  outer: number,
  n: number,
  inner: number,
  acc: Float64Array,
): void {
  if (inner === 1) {
    for (let o = 0; o < outer; o++) {
      const base = o * n
      if (op === 'sum') {
        let a = 0
        for (let j = 0; j < n; j++) a += data[base + j] as number
        acc[o] = a
      } else if (op === 'prod') {
        let a = 1
        for (let j = 0; j < n; j++) a *= data[base + j] as number
        acc[o] = a
      } else if (op === 'max') {
        let a = -Infinity
        let b = -Infinity
        let c = -Infinity
        let d = -Infinity
        let j = 0
        for (; j + 3 < n; j += 4) {
          a = Math.max(a, data[base + j] as number)
          b = Math.max(b, data[base + j + 1] as number)
          c = Math.max(c, data[base + j + 2] as number)
          d = Math.max(d, data[base + j + 3] as number)
        }
        for (; j < n; j++) a = Math.max(a, data[base + j] as number)
        acc[o] = Math.max(Math.max(a, b), Math.max(c, d))
      } else {
        let a = Infinity
        let b = Infinity
        let c = Infinity
        let d = Infinity
        let j = 0
        for (; j + 3 < n; j += 4) {
          a = Math.min(a, data[base + j] as number)
          b = Math.min(b, data[base + j + 1] as number)
          c = Math.min(c, data[base + j + 2] as number)
          d = Math.min(d, data[base + j + 3] as number)
        }
        for (; j < n; j++) a = Math.min(a, data[base + j] as number)
        acc[o] = Math.min(Math.min(a, b), Math.min(c, d))
      }
    }
    return
  }
  for (let o = 0; o < outer; o++) {
    const dst = o * inner
    for (let i = 0; i < n; i++) {
      const src = (o * n + i) * inner
      if (op === 'sum')
        for (let k = 0; k < inner; k++) acc[dst + k] = (acc[dst + k] as number) + (data[src + k] as number)
      else if (op === 'prod')
        for (let k = 0; k < inner; k++) acc[dst + k] = (acc[dst + k] as number) * (data[src + k] as number)
      else if (op === 'max')
        for (let k = 0; k < inner; k++) acc[dst + k] = Math.max(acc[dst + k] as number, data[src + k] as number)
      else for (let k = 0; k < inner; k++) acc[dst + k] = Math.min(acc[dst + k] as number, data[src + k] as number)
    }
  }
}

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
  const block = reducedBlock(x.shape, axes)
  if (block !== null && !(op === 'sum' && (trailing || leading))) {
    // Adjacent reduced axes: plain loops per operation (the sums over the first or last axes below are already such loops).
    reduceLanes(op, x.data, block.outer, block.n, block.inner, acc)
  } else if (rank > 0 && (trailing || leading)) {
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

/**
 * `log(sum(exp(x)))` over a block of adjacent axes of a float64 tensor, fused: a maximum pass, then one pass of `exp(x - shift)`
 * summed in storage order from zero, then `log(sum) + shift`, with the shift the maximum where it is finite and 0 where it is
 * not (all `-Infinity`, a `+Infinity` or a NaN). It is the arithmetic of the composition `ops/reduce.ts` otherwise builds from
 * `max`, `sub`, `exp`, `sum`, `log` and `add`, bit for bit, in two passes and one allocation. `undefined` where the fused loop does not apply
 * (another dtype, axes with a gap, an empty reduction), and the caller composes the operations instead.
 */
export function logsumexp(x: Tensor, axes: readonly number[], keepDims: boolean): Tensor | undefined {
  if (x.dtype !== 'float64') return undefined
  const block = reducedBlock(x.shape, axes)
  if (block === null || block.n === 0) return undefined
  const { outer, n, inner } = block
  const data = x.data as Float64Array
  const out = new Float64Array(outer * inner)
  if (inner === 1) {
    for (let o = 0; o < outer; o++) {
      const base = o * n
      let a = -Infinity
      let b = -Infinity
      let c = -Infinity
      let d = -Infinity
      let j = 0
      for (; j + 3 < n; j += 4) {
        a = Math.max(a, data[base + j] as number)
        b = Math.max(b, data[base + j + 1] as number)
        c = Math.max(c, data[base + j + 2] as number)
        d = Math.max(d, data[base + j + 3] as number)
      }
      for (; j < n; j++) a = Math.max(a, data[base + j] as number)
      const top = Math.max(Math.max(a, b), Math.max(c, d))
      const shift = Number.isFinite(top) ? top : 0
      let total = 0
      for (let i = 0; i < n; i++) total += Math.exp((data[base + i] as number) - shift)
      out[o] = Math.log(total) + shift
    }
  } else {
    const shift = new Float64Array(inner)
    const total = new Float64Array(inner)
    for (let o = 0; o < outer; o++) {
      shift.fill(-Infinity)
      total.fill(0)
      for (let i = 0; i < n; i++) {
        const src = (o * n + i) * inner
        for (let k = 0; k < inner; k++) shift[k] = Math.max(shift[k] as number, data[src + k] as number)
      }
      for (let k = 0; k < inner; k++) if (!Number.isFinite(shift[k])) shift[k] = 0
      for (let i = 0; i < n; i++) {
        const src = (o * n + i) * inner
        for (let k = 0; k < inner; k++)
          total[k] = (total[k] as number) + Math.exp((data[src + k] as number) - (shift[k] as number))
      }
      for (let k = 0; k < inner; k++) out[o * inner + k] = Math.log(total[k] as number) + (shift[k] as number)
    }
  }
  const reduced = new Set(axes)
  const shape = keepDims ? x.shape.map((m, a) => (reduced.has(a) ? 1 : m)) : x.shape.filter((_, a) => !reduced.has(a))
  return wrap('float64', shape, out)
}

export function argReduce(op: 'max' | 'min', x: Tensor, axis: number, keepDims: boolean): Tensor<'int32'> {
  const { outer, n, inner } = lanes(x.shape, axis)
  if (n === 0 && outer * inner > 0) throw new OmaiError('value', `arg${op}: cannot reduce a zero-length axis`)
  const out = new Int32Array(outer * inner)
  const data = x.data
  if (inner === 1) {
    // Along the last axis: one running extreme per row, no strided index (the same first-NaN and first-extreme rules as below).
    for (let o = 0; o < outer; o++) {
      const base = o * n
      let best = 0
      let bestValue = data[base] as number
      if (!Number.isNaN(bestValue)) {
        if (op === 'max') {
          for (let i = 1; i < n; i++) {
            const v = data[base + i] as number
            if (v > bestValue) {
              best = i
              bestValue = v
            } else if (Number.isNaN(v)) {
              best = i
              break
            }
          }
        } else {
          for (let i = 1; i < n; i++) {
            const v = data[base + i] as number
            if (v < bestValue) {
              best = i
              bestValue = v
            } else if (Number.isNaN(v)) {
              best = i
              break
            }
          }
        }
      }
      out[o] = best
    }
    const kept = keepDims ? x.shape.map((s, a) => (a === axis ? 1 : s)) : x.shape.filter((_, a) => a !== axis)
    return wrap('int32', kept, out)
  }
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
