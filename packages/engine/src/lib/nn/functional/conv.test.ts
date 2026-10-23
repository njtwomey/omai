import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { toArray, type Value } from '../../tensor/index.ts'
import {
  conv,
  conv1d,
  conv2d,
  conv3d,
  convTranspose1d,
  convTranspose2d,
  convTransposed,
  depthwiseConv,
  fold,
  separableConv,
  unfold,
} from './conv.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const ok = (f: (...xs: Value[]) => Value, xs: Value[]): void => {
  expect(checkPrimitive(f, xs, { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
}

describe('convolution gradients', () => {
  it('conv1d with stride, padding, dilation and bias', () => {
    ok(
      (x, w, b) => conv1d(x, w, { bias: b, stride: 2, padding: 1, dilation: 2 }),
      [n(0, [2, 2, 7]), n(1, [3, 2, 2]), n(2, [3])],
    )
  })
  it('conv2d grouped with reflect padding', () => {
    ok(
      (x, w) => conv2d(x, w, { groups: 2, padding: 1, paddingMode: 'reflect' }),
      [n(3, [1, 4, 4, 3]), n(4, [2, 2, 3, 2])],
    )
  })
  it('conv2d same padding with circular mode', () => {
    ok((x, w) => conv2d(x, w, { padding: 'same', paddingMode: 'circular' }), [n(5, [1, 1, 4, 4]), n(6, [2, 1, 3, 3])])
  })
  it('conv3d', () => {
    ok((x, w, b) => conv3d(x, w, { bias: b }), [n(7, [1, 1, 3, 3, 2]), n(8, [2, 1, 2, 2, 2]), n(9, [2])])
  })
  it('transposed convolutions', () => {
    ok(
      (x, w, b) => convTranspose1d(x, w, { bias: b, stride: 2, padding: 1, outputPadding: 1 }),
      [n(10, [1, 2, 3]), n(11, [2, 3, 3]), n(12, [3])],
    )
    ok((x, w) => convTranspose2d(x, w, { stride: 2, groups: 2 }), [n(13, [1, 2, 2, 2]), n(14, [2, 1, 2, 2])])
  })
  it('unfold and fold', () => {
    ok((x) => unfold(x, [2, 2], { stride: 1, padding: 1 }), [n(15, [1, 2, 3, 3])])
    ok((c) => fold(c, [3, 3], 2), [n(16, [1, 8, 4])])
  })
  it('depthwise and separable convolutions', () => {
    ok((x, w) => depthwiseConv(x, w, { padding: 1 }), [n(17, [1, 2, 4]), n(18, [4, 1, 3])])
    ok((x, d, p) => separableConv(x, d, p), [n(19, [1, 2, 4]), n(20, [2, 1, 2]), n(21, [3, 2, 1])])
  })
})

describe('convolution shapes and options', () => {
  it('accepts unbatched inputs', () => {
    expect(conv1d([[1, 2, 3]], [[[1, 1]]]).shape).toEqual([1, 2])
    expect(toArray(conv1d([[1, 2, 3]], [[[1, 1]]]))).toEqual([[3, 5]])
  })
  it('reaches an asked-for transposed output size', () => {
    const y = convTransposed(n(22, [1, 1, 4, 4]), n(23, [1, 1, 3, 3]), { stride: 2, padding: 1, outputSize: [8, 8] })
    expect(y.shape).toEqual([1, 1, 8, 8])
    expect(() => convTransposed(n(22, [1, 1, 4]), n(23, [1, 1, 3]), { stride: 2, outputSize: [20] })).toThrow()
  })
  it('rejects weights of the wrong rank', () => {
    expect(() => conv2d(n(0, [1, 1, 4]), n(1, [1, 1, 2]))).toThrow()
    expect(() => conv(n(0, [1, 1]), n(1, [1]))).toThrow()
  })
  it('fold inverts unfold for non-overlapping blocks', () => {
    const x = n(24, [1, 2, 4, 4])
    expect(toArray(fold(unfold(x, 2, { stride: 2 }), [4, 4], 2, { stride: 2 }))).toEqual(toArray(x))
  })
})
