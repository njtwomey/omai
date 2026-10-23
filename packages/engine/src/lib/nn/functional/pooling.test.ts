import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { toArray, type Value } from '../../tensor/index.ts'
import {
  adaptiveAvgPool,
  adaptiveMaxPool,
  avgPool,
  globalAvgPool,
  globalMaxPool,
  lpPool,
  maxPool,
  maxPoolWithIndices,
  maxUnpool,
} from './pooling.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const ok = (f: (x: Value) => Value, x: Value): void => {
  expect(checkPrimitive(f, [x], { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
}

describe('pooling gradients', () => {
  it('max pooling with padding, dilation and ceil mode', () => {
    ok((x) => maxPool(x, [3], { stride: 2, padding: 1, ceilMode: true }), n(0, [1, 2, 8]))
    ok((x) => maxPool(x, [2, 2], { dilation: 2 }), n(1, [1, 1, 5, 5]))
  })
  it('average pooling, counted padding or not', () => {
    ok((x) => avgPool(x, [3], { stride: 2, padding: 1, countIncludePad: false }), n(2, [1, 2, 7]))
    ok((x) => avgPool(x, [2, 2], { ceilMode: true }), n(3, [1, 1, 5, 3]))
  })
  it('adaptive and global pooling', () => {
    ok((x) => adaptiveAvgPool(x, [3, 2]), n(4, [1, 2, 5, 4]))
    ok((x) => adaptiveMaxPool(x, [2, 3]), n(5, [1, 1, 5, 4]))
    ok((x) => globalAvgPool(x), n(6, [2, 2, 3]))
    ok((x) => globalMaxPool(x, true), n(7, [2, 2, 3]))
  })
  it('Lp pooling and unpooling', () => {
    ok((x) => lpPool(x, 2, [2]), n(8, [1, 2, 6]))
    const { indices } = maxPoolWithIndices(n(9, [1, 1, 4, 4]), [2, 2])
    ok((v) => maxUnpool(v, indices, [2, 2]), n(10, [1, 1, 2, 2]))
  })
})

describe('pooling behaviour', () => {
  it('keeps an axis of an adaptive size given as null and accepts unbatched input', () => {
    expect(adaptiveAvgPool(n(0, [1, 2, 6, 5]), [3, null]).shape).toEqual([1, 2, 3, 5])
    expect(adaptiveMaxPool(n(0, [2, 6]), [3]).shape).toEqual([2, 3])
    expect(maxPool(n(0, [2, 6]), [2]).shape).toEqual([2, 3])
  })
  it('infinite Lp pooling is max pooling', () => {
    const x = n(1, [1, 1, 6])
    expect(toArray(lpPool(x, Number.POSITIVE_INFINITY, [2]))).toEqual(toArray(maxPool(x, [2])))
  })
  it('rejects padding over half the kernel', () => {
    expect(() => maxPool(n(0, [1, 1, 6]), [2], { padding: 2 })).toThrow()
  })
  it('unpooling restores the maxima at their positions', () => {
    const x = n(2, [1, 1, 4, 4])
    const { values, indices } = maxPoolWithIndices(x, [2, 2])
    const back = maxUnpool(values, indices, [2, 2])
    expect(toArray(maxPool(back, [2, 2]))).toEqual(toArray(values))
  })
})
